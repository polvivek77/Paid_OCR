"""Shared pytest setup.

Markers (see pyproject.toml):
    gpu          - needs the real NVIDIA GPU + CUDA builds installed
    integration  - runs real engines end-to-end (slow; cloud engines cost money)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(autouse=True)
def _clear_device_caches():
    """Device probes are lru_cached in production; reset between tests."""
    from engines import utils
    for fn in (utils.has_nvidia_gpu, utils.has_paddle_gpu):
        fn.cache_clear()
    yield
    for fn in (utils.has_nvidia_gpu, utils.has_paddle_gpu):
        fn.cache_clear()
