"""Shared plumbing for the cloud vision-LLM engines (OpenAI, Anthropic, DeepSeek).

Every provider gets the *same* input — each page rasterised to an image — and the
*same* transcription prompt, so the accuracy numbers compare the models, not the
pre-processing. PDFs are always rasterised (never sent as PDF files) because some
providers would otherwise read the embedded text layer instead of doing OCR.

A provider module only has to supply ``call_page(image_b64, mime) -> PageResult``;
``run_engine`` handles .env loading, rendering, concurrency, timing and the result
dict the Streamlit app expects.
"""
from __future__ import annotations

import base64
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from engines.logging_setup import get_logger
from engines.utils import load_as_pil, timer

log = get_logger("ocr.vision_llm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_PROMPT = """You are an OCR engine. Transcribe ALL text visible in this document page image.

Rules:
- Copy the text exactly as printed: same spelling, punctuation, capitalisation, numbers,
  currency symbols, signs and decimal separators. Do not correct, translate or summarise.
- Follow natural reading order (top to bottom; for multi-column layouts, finish one column
  before starting the next). Keep each printed line on its own line.
- Render every table as a GitHub-flavoured Markdown table (| col | col |), one row per line,
  keeping empty cells empty.
- Include headers, footers, page numbers, captions and text inside charts or figures.
- If a word is illegible write [illegible]. Never guess or invent content.
- Output only the transcription: no preamble, no explanations, no code fences."""

MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_OCR_MAX_TOKENS", "16000"))
# Anthropic rejects images over 5 MB of base64; ~3.7 MB raw keeps every provider happy.
_MAX_IMAGE_BYTES = 3_700_000


@dataclass
class PageResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def load_env() -> None:
    """Load ./.env into os.environ (existing variables win)."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass
    # Minimal fallback parser so python-dotenv stays optional.
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Page images
# ---------------------------------------------------------------------------
def encode_page(img: Image.Image) -> tuple[str, str]:
    """Return (base64, mime) for a page, downscaled so its long side fits the limit."""
    max_side = int(os.environ.get("LLM_OCR_MAX_IMAGE_SIDE", "2048"))
    img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    mime = "image/png"
    if buf.tell() > _MAX_IMAGE_BYTES:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode("ascii"), mime


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
# **bold** / __bold__ wrapped around non-space text (leaves "5 ** 2" alone)
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def strip_fences(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not _FENCE_RE.match(ln.strip())).strip()


def markdown_to_plain(text: str) -> str:
    """Turn Markdown tables into space-separated text and drop bold markers so
    the output scores fairly against a plain-text ground truth."""
    out: list[str] = []
    for line in text.splitlines():
        if _TABLE_SEP_RE.match(line) and "-" in line:
            continue
        line = _BOLD_RE.sub(r"\2", line)
        if _is_table_row(line):
            out.append(" ".join(c for c in _split_row(line) if c))
        else:
            out.append(line)
    return "\n".join(out)


def parse_tables(text: str, page: int, start_index: int = 1) -> list[dict]:
    """Extract Markdown tables in the shape app.py renders: {page, index, rows}."""
    tables: list[dict] = []
    block: list[list[str]] = []

    def flush() -> None:
        if len(block) >= 2:
            tables.append({"page": page, "index": start_index + len(tables), "rows": list(block)})
        block.clear()

    for line in text.splitlines():
        if _TABLE_SEP_RE.match(line) and "-" in line:
            continue
        if _is_table_row(line):
            block.append(_split_row(line))
        else:
            flush()
    flush()
    return tables


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _error(engine: str, msg: str) -> dict:
    return {
        "engine": engine, "device": "api", "elapsed_sec": 0.0, "pages": 0,
        "text": "", "lines": [], "tables": [], "error": msg,
    }


def run_engine(
    engine: str,
    model: str,
    api_key_env: str,
    file_bytes: bytes,
    filename: str,
    call_page: Callable[[str, str], PageResult],
) -> dict:
    """Rasterise ``file_bytes``, send each page to ``call_page`` and build the result."""
    load_env()
    if not os.environ.get(api_key_env):
        msg = f"{api_key_env} is not set. Add it to {PROJECT_ROOT / '.env'}."
        log.error(msg)
        return _error(engine, msg)

    log.info("%s [%s] → %s (%d bytes)", engine, model, filename, len(file_bytes))
    try:
        pages = load_as_pil(file_bytes, filename)
        encoded = [encode_page(p) for p in pages]
        workers = max(1, int(os.environ.get("LLM_OCR_CONCURRENCY", "4")))
        log.info("%s: %d page(s), concurrency=%d", engine, len(pages), workers)

        def _one(idx: int) -> PageResult:
            b64, mime = encoded[idx]
            res = call_page(b64, mime)
            log.debug("%s: page %d → %d chars (in=%d out=%d tok)", engine, idx + 1,
                      len(res.text), res.input_tokens, res.output_tokens)
            if res.truncated:
                log.warning("%s: page %d output hit the token limit (LLM_OCR_MAX_TOKENS=%d)",
                            engine, idx + 1, MAX_OUTPUT_TOKENS)
            return res

        with timer() as t:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_one, range(len(pages))))
    except Exception as e:
        log.exception("%s failed", engine)
        return _error(engine, f"{type(e).__name__}: {e}")

    # A page that ran out of tokens before producing any text (e.g. a reasoning
    # model that spent the whole budget thinking) must not be scored as "0% OK".
    empty = [i + 1 for i, r in enumerate(results) if r.truncated and not r.text.strip()]
    if empty:
        msg = (f"Page(s) {empty} returned no text: the model used the whole output token "
               f"limit ({MAX_OUTPUT_TOKENS}, LLM_OCR_MAX_TOKENS) without answering — usually "
               "hidden reasoning. Disable the model's thinking mode or raise the limit.")
        log.error("%s: %s", engine, msg)
        return _error(engine, msg)

    page_texts: list[str] = []
    lines: list[dict] = []
    tables: list[dict] = []
    for page_no, res in enumerate(results, start=1):
        md = strip_fences(res.text)
        tables.extend(parse_tables(md, page_no, start_index=len(tables) + 1))
        plain = markdown_to_plain(md)
        page_texts.append(plain)
        lines.extend(
            {"page": page_no, "text": ln.strip(), "conf": None, "bbox": None}
            for ln in plain.splitlines() if ln.strip()
        )

    usage = {
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "truncated_pages": [i + 1 for i, r in enumerate(results) if r.truncated],
    }
    log.info("%s ✓ %.2fs · %d pages · %d lines · %d tables · tokens in=%d out=%d",
             engine, t["elapsed"], len(pages), len(lines), len(tables),
             usage["input_tokens"], usage["output_tokens"])
    return {
        "engine": engine,
        "device": "api",
        "model": model,
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": len(pages),
        "text": "\n\n".join(page_texts),
        "markdown": "\n\n".join(strip_fences(r.text) for r in results),
        "lines": lines,
        "tables": tables,
        "usage": usage,
    }


def cli(extract: Callable[[bytes, str], dict]) -> None:
    """Shared ``python <engine>.py file`` entry point."""
    import sys
    from engines.logging_setup import setup_logging
    setup_logging()
    p = Path(sys.argv[1])
    out = extract(p.read_bytes(), p.name)
    if out.get("error"):
        print(f"[{out['engine']}] ERROR: {out['error']}")
        return
    print(out["text"][:500])
    u = out["usage"]
    print(f"[{out['engine']}] {out['model']} — {out['elapsed_sec']}s, {out['pages']} page(s), "
          f"{len(out['lines'])} lines, {len(out['tables'])} tables, "
          f"tokens in={u['input_tokens']} out={u['output_tokens']}")
