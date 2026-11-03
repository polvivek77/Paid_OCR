"""Surya OCR runner (Surya 0.22+ predictor API).

Backend selection:
    - Default is **llamacpp**: the `llama-server` binary runs the recognition
      VLM. On an NVIDIA machine the CUDA build is used (Surya offloads every
      layer, LLAMA_CPP_NGL=99), so no Docker is needed for GPU inference.
    - `SURYA_INFERENCE_BACKEND=vllm` opts into the vllm backend, which spawns a
      Docker container (Docker Desktop must be running).
    - Detection always runs in-process on torch (CUDA when available).
"""
from __future__ import annotations

import os
from pathlib import Path

from engines.llamacpp_helper import LlamaServerNotFound, ensure_llama_server, is_cuda_build
from engines.logging_setup import get_logger, preload_windows_dlls
from engines.utils import has_nvidia_gpu, has_torch_gpu, load_as_pil, timer

preload_windows_dlls()

log = get_logger("ocr.surya")

_state: dict = {}


def _choose_backend() -> str:
    # Explicit override always wins.
    env = os.environ.get("SURYA_INFERENCE_BACKEND")
    if env:
        return env
    return "llamacpp"


def _recognition_device(backend: str, server: str | None) -> str:
    """Where the recognition VLM (the slow part) really runs."""
    if backend == "vllm":
        return "cuda"
    return "cuda" if server and has_nvidia_gpu() and is_cuda_build(server) else "cpu"


def _load():
    if _state:
        return
    backend = _choose_backend()
    os.environ["SURYA_INFERENCE_BACKEND"] = backend
    log.info("Surya: backend=%s", backend)

    server = None
    if backend == "llamacpp":
        server = ensure_llama_server(auto_install=False, prefer_cuda=has_nvidia_gpu())
        log.info("Surya: llama-server=%s (cuda build=%s)", server, is_cuda_build(server))

    log.info("Surya: initialising predictors…")
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    _state["det"] = DetectionPredictor()
    _state["rec"] = RecognitionPredictor()
    _state["device"] = _recognition_device(backend, server)
    _state["backend"] = backend
    log.info("Surya: ready (device=%s, detection on %s, backend=%s)",
             _state["device"], "cuda" if has_torch_gpu() else "cpu", backend)


import html as _html
import re as _re


_TAG_RE = _re.compile(r"<[^>]+>")
_WS_RE = _re.compile(r"[ \t]+")


def _html_to_text(html_str: str) -> str:
    """Strip Surya-OCR-2's block HTML down to plain text while keeping newlines."""
    if not html_str:
        return ""
    s = html_str
    # Block-level tags → newline (including table cells so cells don't collide)
    s = _re.sub(
        r"</?(p|br|div|li|h[1-6]|tr|table|thead|tbody|td|th|pre|blockquote)[^>]*>",
        "\n", s, flags=_re.IGNORECASE,
    )
    # Strip remaining tags
    s = _TAG_RE.sub("", s)
    # HTML entities
    s = _html.unescape(s)
    # Collapse whitespace within lines but keep line breaks
    lines = [_WS_RE.sub(" ", ln).strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _line_to_dict(line, page_num: int) -> dict | None:
    """Convert a Surya line-like object to our uniform dict shape. Returns None
    when the line has no usable text."""
    txt = getattr(line, "text", None)
    if not txt:
        return None
    return {
        "page": page_num,
        "text": str(txt),
        "conf": float(getattr(line, "confidence", 0.0) or 0.0),
        "bbox": getattr(line, "bbox", None),
    }


def _collect_page_lines(page, page_num: int) -> list[dict]:
    """Extract text lines from a Surya ``PageOCRResult``.

    Layout of Surya 0.22's Surya-OCR-2 output (see
    ``.venv/…/surya/recognition/schema.py``):

        PageOCRResult
          blocks: list[BlockOCRResult]  ← each block has:
            label: str        (Picture / Text / Table / …)
            html: str         ← THE OCR TEXT lives here as HTML
            skipped: bool     (True for images/pictures we can't OCR)
            error: bool
            confidence, polygon, bbox   (inherited from PolygonBox)

    We turn each block's HTML into a plain-text line entry and skip blocks that
    were skipped, errored, or have empty html. We also keep the pre-0.22
    fallbacks (page.text_lines, block.text, block.text_lines) so older Surya
    versions still parse.
    """
    out: list[dict] = []

    # 0. Legacy top-level text_lines (Surya <= 0.14-ish)
    for line in (getattr(page, "text_lines", None) or []):
        d = _line_to_dict(line, page_num)
        if d:
            out.append(d)

    for block in (getattr(page, "blocks", None) or []):
        if getattr(block, "skipped", False) or getattr(block, "error", False):
            continue

        # 1. Surya 0.22 canonical location: block.html
        html_str = getattr(block, "html", "") or ""
        text = _html_to_text(html_str)

        # 2. Older shapes as fallbacks
        if not text:
            for line in (getattr(block, "text_lines", None) or []):
                d = _line_to_dict(line, page_num)
                if d:
                    out.append(d)
                    continue
            if not text:
                text = getattr(block, "text", None) or ""

        if not text:
            continue

        out.append({
            "page": page_num,
            "text": str(text),
            "conf": float(getattr(block, "confidence", 0.0) or 0.0),
            "bbox": getattr(block, "bbox", None),
        })
    return out


def _friendly_error(exc: Exception) -> str:
    """Turn Surya's raw backend errors into an actionable one-liner."""
    msg = str(exc)
    lower = msg.lower()
    if "docker" in lower and ("cannot find" in lower or "pipe" in lower or "connect" in lower):
        return (
            "Surya's vllm backend needs Docker Desktop to be running.\n"
            "Start Docker Desktop and try again. Or set "
            "SURYA_INFERENCE_BACKEND=llamacpp to use llama.cpp instead."
        )
    if "llama-server" in lower and "not found" in lower:
        return (
            "Surya's llama.cpp backend needs the `llama-server` binary. Run:\n"
            "    python scripts/install_llama_cpp.py --cuda\n"
            "or start Docker and switch to the vllm backend."
        )
    return f"{type(exc).__name__}: {msg}"


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("Surya → %s (%d bytes)", filename, len(file_bytes))
    device = _state.get("device", "cpu")
    try:
        _load()
        pages = load_as_pil(file_bytes, filename)
        log.info("Surya: rasterised %d page(s)", len(pages))

        with timer() as t:
            preds = _state["rec"](pages, full_page=True)
    except LlamaServerNotFound as exc:
        log.error("Surya: %s", exc)
        return {
            "engine": "Surya OCR", "device": device,
            "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [], "error": _friendly_error(exc),
        }
    except Exception as exc:
        log.exception("Surya failed")
        return {
            "engine": "Surya OCR", "device": device,
            "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [], "error": _friendly_error(exc),
        }

    text_all: list[str] = []
    lines: list[dict] = []
    log.info("Surya raw preds: type=%s len=%s", type(preds).__name__,
             len(preds) if hasattr(preds, "__len__") else "?")

    for page_idx, page in enumerate(preds):
        page_lines = _collect_page_lines(page, page_idx + 1)
        log.info("Surya page %d: extracted %d lines", page_idx + 1, len(page_lines))
        lines.extend(page_lines)
        text_all.extend(ln["text"] for ln in page_lines)
    log.info(
        "Surya ✓ %.2fs · %d pages · %d lines · device=%s · backend=%s",
        t["elapsed"], len(preds), len(lines), _state["device"], _state["backend"],
    )
    return {
        "engine": "Surya OCR",
        "device": _state["device"],
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": len(pages),
        "text": "\n".join(text_all),
        "lines": lines,
        "tables": [],
    }


if __name__ == "__main__":
    import sys
    from engines.logging_setup import setup_logging
    setup_logging()
    p = Path(sys.argv[1])
    out = extract(p.read_bytes(), p.name)
    print(out["text"][:500])
    print(f"[{out['engine']}] {out['device']} — {out['elapsed_sec']}s, {len(out['lines'])} lines")
