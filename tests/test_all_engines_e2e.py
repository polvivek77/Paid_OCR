"""End-to-end: run every engine registered in app.py on sample documents
through the same `_run` the Streamlit app uses, and check which device each
one really used.

    uv run pytest -m integration -s

Samples (one page each to keep time and API cost down):
    ami_organics.pdf               scanned page (the main bug-report file)
    sbi_one_page.pdf               scanned, table-heavy page
    attention_is_all_you_need.pdf  digital PDF — only page 1 is sent,
                                   sliced exactly like the app's page selector

Cloud engines make one paid API call per sample.
"""
from __future__ import annotations

import pytest

from conftest import DATA_DIR
from engines.utils import has_nvidia_gpu, slice_pdf

pytestmark = pytest.mark.integration

# (file name, pages to send or None for all, has a text layer)
SAMPLES = [
    ("ami_organics.pdf", None, False),
    ("sbi_one_page.pdf", None, False),
    ("attention_is_all_you_need.pdf", [1], True),
]

# These tools have no GPU code path at all.
CPU_ONLY = {"Tesseract", "pdfplumber", "Camelot"}
CLOUD = {"OpenAI", "Anthropic", "DeepSeek"}
# Read the PDF text layer only; scanned pages have none.
TEXT_LAYER_ONLY = {"pdfplumber", "Camelot"}


def _engine_names() -> list[str]:
    import app
    return [e.name for e in app.ENGINES]


def _payload(name: str, pages: list[int] | None) -> tuple[bytes, str]:
    data = (DATA_DIR / name).read_bytes()
    if pages:
        return slice_pdf(data, pages), f"pages-{'-'.join(map(str, pages))}-of-{name}"
    return data, name


@pytest.fixture(scope="module", params=SAMPLES, ids=[s[0] for s in SAMPLES])
def sample_results(request):
    import app
    name, pages, has_text_layer = request.param
    data, filename = _payload(name, pages)
    out = {}
    print(f"\n=== {filename} ===")
    for eng in app.ENGINES:
        res = app._run(eng, data, filename)
        out[eng.name] = res
        print(f"{eng.name:<10} device={res.get('device')!s:<5} "
              f"time={res.get('elapsed_sec')}s chars={len(res.get('text', ''))} "
              f"error={str(res.get('error'))[:150] if res.get('error') else '-'}")
    return has_text_layer, out


@pytest.mark.parametrize("engine", _engine_names())
def test_engine_runs_without_error(sample_results, engine):
    has_text_layer, results = sample_results
    res = results[engine]
    if engine in TEXT_LAYER_ONLY and not has_text_layer:
        # Expected: these tools can't OCR and say so instead of returning junk.
        assert "scanned" in str(res.get("error", "")).lower()
        return
    assert not res.get("error"), res.get("error")
    if engine == "Camelot":  # tables only, no running text
        return
    assert len(res["text"].strip()) > 50, "suspiciously little text extracted"


@pytest.mark.skipif(not has_nvidia_gpu(), reason="no NVIDIA GPU on this machine")
@pytest.mark.parametrize("engine", _engine_names())
def test_engine_uses_the_gpu_when_it_can(sample_results, engine):
    _, results = sample_results
    device = results[engine].get("device")
    if engine in CLOUD:
        assert device == "api"
    elif engine in CPU_ONLY:
        assert device == "cpu"
    else:
        assert device in ("cuda", "gpu"), f"{engine} ran on {device}"
