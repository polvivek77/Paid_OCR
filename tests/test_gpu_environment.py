"""Checks against the real laptop: is every GPU runtime actually usable?

Run with:  uv run pytest -m gpu
Skipped automatically on machines without an NVIDIA GPU.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from conftest import PROJECT_ROOT
from engines.utils import has_nvidia_gpu

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_nvidia_gpu(), reason="no NVIDIA GPU on this machine"),
]


def test_torch_is_a_cuda_build_and_sees_the_gpu():
    import torch
    assert torch.version.cuda, f"torch {torch.__version__} is a CPU-only build"
    assert torch.cuda.is_available()


def test_torch_has_kernels_for_this_gpu_architecture():
    """RTX 50-series is sm_120; older wheels load but fail at the first kernel."""
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()
    x = torch.randn(256, 256, device="cuda")
    assert (x @ x).isfinite().all().item()


def test_paddle_runs_a_kernel_on_the_gpu():
    """Run in a child process, exactly as paddleocr_engine does (Paddle and Torch
    CUDA DLLs must not share a process on Windows)."""
    code = (
        "import paddle; assert paddle.device.is_compiled_with_cuda(), 'CPU paddle';"
        "paddle.device.set_device('gpu:0');"
        "x = paddle.randn([256, 256]); y = (x @ x).numpy();"
        "print('OK', paddle.device.get_device())"
    )
    env = {**os.environ}
    env.pop("PADDLE_FORCE_CPU", None)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300, env=env, cwd=PROJECT_ROOT)
    assert proc.returncode == 0 and "OK gpu" in proc.stdout, proc.stderr[-2000:]


def test_paddle_gpu_is_not_forced_off():
    from engines.utils import has_paddle_gpu
    assert os.environ.get("PADDLE_FORCE_CPU", "").lower() not in ("1", "true", "yes"), \
        "PADDLE_FORCE_CPU is set — remove it now that Paddle's GPU wheel supports this card"
    assert has_paddle_gpu()


def test_a_cuda_llama_server_is_installed_for_surya_and_marker():
    from engines.llamacpp_helper import is_cuda_build, locate_llama_server
    path = locate_llama_server(prefer_cuda=True)
    assert path, "no llama-server — run: python scripts/install_llama_cpp.py --cuda"
    assert is_cuda_build(path), f"{path} is a CPU-only llama.cpp build"
