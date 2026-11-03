"""Unit tests for GPU / device selection. No GPU needed — everything is mocked.

Regression cover for the 2026-11-03 bug where every local engine ran on CPU:
  * the llama.cpp installer chose the CPU build because it asked *torch* (a
    CPU-only wheel) whether a GPU exists, instead of asking the hardware;
  * Marker never asked for the CUDA llama-server build;
  * locate_llama_server returned the CPU build even when a CUDA build existed;
  * Surya reported device=cuda while its llama-server ran on CPU, and picked the
    Docker-only vllm backend whenever torch saw CUDA.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from engines import llamacpp_helper, utils


# ---------------------------------------------------------------------------
# Hardware detection must not depend on how torch was built
# ---------------------------------------------------------------------------
class _Proc:
    def __init__(self, returncode: int, stdout: str):
        self.returncode = returncode
        self.stdout = stdout


def test_has_nvidia_gpu_true_when_nvidia_smi_lists_a_gpu(monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda name: "nvidia-smi.exe")
    monkeypatch.setattr(
        utils.subprocess, "run",
        lambda *a, **k: _Proc(0, "GPU 0: NVIDIA GeForce RTX 5050 Laptop GPU (UUID: GPU-x)\n"),
    )
    assert utils.has_nvidia_gpu() is True


def test_has_nvidia_gpu_false_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda name: None)
    assert utils.has_nvidia_gpu() is False


def test_has_nvidia_gpu_false_when_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda name: "nvidia-smi.exe")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(utils.subprocess, "run", boom)
    assert utils.has_nvidia_gpu() is False


def test_torch_gpu_problem_flags_cpu_only_torch_on_gpu_machine(monkeypatch):
    monkeypatch.setattr(utils, "has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(utils, "_torch_build", lambda: ("2.14.0+cpu", None, False))
    msg = utils.torch_gpu_problem()
    assert msg and "CPU-only" in msg and "uv sync" in msg


def test_torch_gpu_problem_none_when_cuda_works(monkeypatch):
    monkeypatch.setattr(utils, "has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(utils, "_torch_build", lambda: ("2.11.0+cu128", "12.8", True))
    assert utils.torch_gpu_problem() is None


def test_torch_gpu_problem_none_on_machine_without_gpu(monkeypatch):
    monkeypatch.setattr(utils, "has_nvidia_gpu", lambda: False)
    assert utils.torch_gpu_problem() is None


# ---------------------------------------------------------------------------
# llama.cpp build selection
# ---------------------------------------------------------------------------
def _fake_llama(dir_: Path, cuda: bool) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    exe = dir_ / "llama-server.exe"
    exe.write_bytes(b"")
    (dir_ / "ggml-base.dll").write_bytes(b"")
    (dir_ / ("ggml-cuda.dll" if cuda else "ggml-cpu-x64.dll")).write_bytes(b"")
    return exe


@pytest.fixture
def tools_dir(tmp_path, monkeypatch):
    d = tmp_path / ".tools" / "llama-cpp"
    monkeypatch.setattr(llamacpp_helper, "TOOLS_DIR", d)
    monkeypatch.delenv("LLAMA_CPP_BINARY", raising=False)
    monkeypatch.setattr(llamacpp_helper.shutil, "which", lambda name: None)
    return d


def test_is_cuda_build_detects_ggml_cuda_dll(tools_dir):
    assert llamacpp_helper.is_cuda_build(_fake_llama(tools_dir / "cuda", cuda=True))
    assert not llamacpp_helper.is_cuda_build(_fake_llama(tools_dir, cuda=False))


def test_locate_prefers_cuda_build_over_cpu_build(tools_dir):
    _fake_llama(tools_dir, cuda=False)
    cuda_exe = _fake_llama(tools_dir / "cuda", cuda=True)
    found = llamacpp_helper.locate_llama_server(prefer_cuda=True)
    assert Path(found) == cuda_exe.resolve()


def test_locate_prefers_cuda_even_if_env_points_at_cpu_build(tools_dir, monkeypatch):
    """ensure_llama_server() exports LLAMA_CPP_BINARY; a CPU path cached there
    from an earlier run must not shadow a CUDA build installed later."""
    cpu_exe = _fake_llama(tools_dir, cuda=False)
    cuda_exe = _fake_llama(tools_dir / "cuda", cuda=True)
    monkeypatch.setenv("LLAMA_CPP_BINARY", str(cpu_exe))
    found = llamacpp_helper.locate_llama_server(prefer_cuda=True)
    assert Path(found) == cuda_exe.resolve()


def test_locate_falls_back_to_cpu_build(tools_dir):
    cpu_exe = _fake_llama(tools_dir, cuda=False)
    found = llamacpp_helper.locate_llama_server(prefer_cuda=True)
    assert Path(found) == cpu_exe.resolve()


def test_ensure_auto_installs_cuda_when_only_cpu_build_exists(tools_dir, monkeypatch):
    _fake_llama(tools_dir, cuda=False)
    calls = []

    def fake_install(progress=None, *, cuda=False):
        calls.append(cuda)
        return _fake_llama(tools_dir / "cuda", cuda=True)

    monkeypatch.setattr(llamacpp_helper, "install_llama_cpp_windows", fake_install)
    monkeypatch.setattr(llamacpp_helper.os, "name", "nt")
    path = llamacpp_helper.ensure_llama_server(auto_install=True, prefer_cuda=True)
    assert calls == [True]
    assert llamacpp_helper.is_cuda_build(path)


# ---------------------------------------------------------------------------
# Surya / Marker wiring
# ---------------------------------------------------------------------------
def test_surya_defaults_to_llamacpp_backend_even_with_cuda(monkeypatch):
    import suryaocr_engine
    monkeypatch.delenv("SURYA_INFERENCE_BACKEND", raising=False)
    monkeypatch.setattr(suryaocr_engine, "has_torch_gpu", lambda: True)
    assert suryaocr_engine._choose_backend() == "llamacpp"


def test_surya_backend_env_override_still_wins(monkeypatch):
    import suryaocr_engine
    monkeypatch.setenv("SURYA_INFERENCE_BACKEND", "vllm")
    assert suryaocr_engine._choose_backend() == "vllm"


@pytest.mark.parametrize(
    "backend,gpu,cuda_build,expected",
    [
        ("llamacpp", True, True, "cuda"),
        ("llamacpp", True, False, "cpu"),   # CPU llama-server → recognition on CPU
        ("llamacpp", False, False, "cpu"),
        ("vllm", True, False, "cuda"),
    ],
)
def test_surya_reports_device_its_recogniser_really_uses(monkeypatch, backend, gpu, cuda_build, expected):
    import suryaocr_engine
    monkeypatch.setattr(suryaocr_engine, "has_nvidia_gpu", lambda: gpu)
    monkeypatch.setattr(suryaocr_engine, "is_cuda_build", lambda p: cuda_build)
    assert suryaocr_engine._recognition_device(backend, "llama-server.exe") == expected


def test_marker_requests_cuda_llama_server_on_gpu_machine(monkeypatch):
    import marker_engine
    seen = {}

    def fake_ensure(auto_install=False, *, prefer_cuda=False):
        seen["prefer_cuda"] = prefer_cuda
        raise llamacpp_helper.LlamaServerNotFound("stop here")

    monkeypatch.setattr(marker_engine, "ensure_llama_server", fake_ensure)
    monkeypatch.setattr(marker_engine, "has_nvidia_gpu", lambda: True)
    marker_engine._state.clear()
    with pytest.raises(llamacpp_helper.LlamaServerNotFound):
        marker_engine._load("cuda")
    assert seen["prefer_cuda"] is True


def test_surya_requests_cuda_llama_server_on_gpu_machine(monkeypatch):
    import suryaocr_engine
    seen = {}

    def fake_ensure(auto_install=False, *, prefer_cuda=False):
        seen["prefer_cuda"] = prefer_cuda
        raise llamacpp_helper.LlamaServerNotFound("stop here")

    monkeypatch.delenv("SURYA_INFERENCE_BACKEND", raising=False)
    monkeypatch.setattr(suryaocr_engine, "ensure_llama_server", fake_ensure)
    monkeypatch.setattr(suryaocr_engine, "has_nvidia_gpu", lambda: True)
    suryaocr_engine._state.clear()
    with pytest.raises(llamacpp_helper.LlamaServerNotFound):
        suryaocr_engine._load()
    assert seen["prefer_cuda"] is True
