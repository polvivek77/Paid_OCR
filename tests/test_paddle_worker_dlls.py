"""The Paddle worker must load the CUDA/cuDNN DLLs that match its wheel.

paddlepaddle-gpu (cu129) installs nvidia-*-cu12 pip packages. A system-wide
cuDNN under C:\\Program Files\\NVIDIA\\CUDNN (on this laptop: v9.26 built for
CUDA 13.4) must only be a fallback, or Paddle can pick up mismatched DLLs.
"""
from __future__ import annotations

from pathlib import Path

import paddleocr_worker as worker


def _mk(*parts: Path) -> None:
    for p in parts:
        p.mkdir(parents=True, exist_ok=True)


def test_venv_nvidia_wheels_win_over_system_cudnn(tmp_path):
    site = tmp_path / "site-packages"
    system = tmp_path / "Program Files" / "NVIDIA" / "CUDNN"
    _mk(site / "nvidia" / "cudnn" / "bin", site / "nvidia" / "cublas" / "bin", system / "v9.26" / "bin")

    dirs = worker._dll_dir_candidates(site, [system])

    assert site / "nvidia" / "cudnn" / "bin" in dirs
    assert not any(str(system) in str(d) for d in dirs)


def test_system_cudnn_used_when_no_nvidia_wheels(tmp_path):
    site = tmp_path / "site-packages"
    system = tmp_path / "Program Files" / "NVIDIA" / "CUDNN"
    _mk(site, system / "v9.26" / "bin")

    dirs = worker._dll_dir_candidates(site, [system])

    assert dirs == [system / "v9.26" / "bin"]
