"""The project config must pin GPU builds, so `uv sync` / `uv run` can never
silently swap CUDA torch for PyPI's CPU-only Windows wheel again (root cause of
the 2026-11-03 "no engine uses the GPU" bug)."""
from __future__ import annotations

import re
import tomllib

from conftest import PROJECT_ROOT

PYPROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
LOCK = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")


def _index_url(name: str) -> str:
    for idx in PYPROJECT["tool"]["uv"].get("index", []):
        if idx["name"] == name:
            return idx["url"]
    raise AssertionError(f"no [[tool.uv.index]] named {name!r}")


def _locked_source(package: str) -> str:
    m = re.search(rf'\[\[package\]\]\nname = "{re.escape(package)}"\nversion = "([^"]+)"\nsource = \{{ registry = "([^"]+)" \}}', LOCK)
    assert m, f"{package} missing from uv.lock"
    return m.group(2)


def test_torch_and_torchvision_come_from_a_cuda_index():
    sources = PYPROJECT["tool"]["uv"]["sources"]
    for pkg in ("torch", "torchvision"):
        url = _index_url(sources[pkg]["index"])
        assert re.search(r"download\.pytorch\.org/whl/cu1[2-9]\d", url), url


def test_cuda_index_is_explicit_so_it_only_serves_pinned_packages():
    for idx in PYPROJECT["tool"]["uv"]["index"]:
        assert idx.get("explicit") is True, idx["name"]


def test_paddle_gpu_runtime_is_declared_from_paddle_cuda_index():
    deps = " ".join(PYPROJECT["project"]["dependencies"])
    assert "paddlepaddle-gpu" in deps
    assert not re.search(r'"?paddlepaddle(?!-gpu)[<>=~ ]', deps), "CPU paddlepaddle must not be declared too"
    url = _index_url(PYPROJECT["tool"]["uv"]["sources"]["paddlepaddle-gpu"]["index"])
    assert "paddlepaddle.org.cn" in url and "/cu1" in url


def test_lockfile_resolves_torch_from_the_cuda_index():
    assert "download.pytorch.org/whl/cu" in _locked_source("torch")
    assert "download.pytorch.org/whl/cu" in _locked_source("torchvision")


def test_cloud_sdks_and_pytest_are_declared():
    deps = " ".join(PYPROJECT["project"]["dependencies"])
    for pkg in ("openai", "anthropic", "python-dotenv"):
        assert pkg in deps
    assert any("pytest" in d for d in PYPROJECT["dependency-groups"]["dev"])
