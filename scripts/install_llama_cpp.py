"""Download and install a llama.cpp Windows binary for Surya / Marker.

Usage:
    python scripts/install_llama_cpp.py [--cuda]

Downloads the latest pre-built release from
https://github.com/ggml-org/llama.cpp/releases into ``.tools/llama-cpp/``.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
import argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Make the project root importable when this script is invoked from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.llamacpp_helper import LlamaServerNotFound, install_llama_cpp_windows
from engines.logging_setup import setup_logging
from engines.utils import has_nvidia_gpu


def _print_progress(read: int, total: int) -> None:
    if total <= 0:
        return
    pct = read * 100 // total
    mb_r = read / (1 << 20)
    mb_t = total / (1 << 20)
    sys.stdout.write(f"\r  {mb_r:6.1f} / {mb_t:6.1f} MB ({pct:3d}%)")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Install llama.cpp for Surya / Marker")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cuda", action="store_true", help="force the NVIDIA CUDA build")
    group.add_argument("--cpu", action="store_true", help="force the CPU build")
    args = parser.parse_args()
    setup_logging()
    # Default: CUDA build whenever the machine has an NVIDIA GPU.
    cuda = args.cuda or (not args.cpu and has_nvidia_gpu())
    print(f"Installing the {'CUDA' if cuda else 'CPU'} build of llama.cpp…")
    try:
        exe = install_llama_cpp_windows(progress=_print_progress, cuda=cuda)
    except LlamaServerNotFound as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    print()  # newline after progress
    print(f"[OK] llama-server installed at {exe}")
    print(f"You can now run Surya OCR and Marker ({'GPU' if cuda else 'CPU'} mode).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
