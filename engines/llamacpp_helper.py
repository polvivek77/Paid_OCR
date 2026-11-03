"""Locate / auto-install a llama.cpp `llama-server` binary for Surya + Marker.

Surya 0.22+ (and therefore Marker 2.0+, which delegates OCR to Surya) call an
external `llama-server` binary from https://github.com/ggml-org/llama.cpp/.
This helper looks in the usual places, and can download+extract a pre-built
Windows release into ``.tools/llama-cpp/`` on demand.

Public API:
    - locate_llama_server() -> str | None
    - ensure_llama_server(auto_install=False) -> str
    - install_llama_cpp_windows(progress=None) -> Path
"""
from __future__ import annotations

import io
import json
import os
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from engines.logging_setup import get_logger

log = get_logger("ocr.llamacpp")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / ".tools" / "llama-cpp"
GITHUB_RELEASES_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
GITHUB_RELEASES_LIST = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"

# Asset naming on GitHub, from newest to older schemas. First match wins.
_WINDOWS_CPU_ASSET_PREFERENCES = (
    "llama-*-bin-win-cpu-x64.zip",
    "llama-*-bin-win-avx2-x64.zip",
    "llama-*-bin-win-avx-x64.zip",
    "llama-*-bin-win-noavx-x64.zip",
    "llama-*-bin-win-openblas-x64.zip",
)

# Surya 0.22's recogniser is a VLM served by llama.cpp.  A CPU llama-server
# works, but takes many minutes per page and frequently hits Surya's request
# timeout.  Prefer CUDA 12 on this project: it works with current NVIDIA
# drivers (including RTX 50 series) and has the broadest Windows support.
_WINDOWS_CUDA_ASSET_PREFERENCES = (
    "llama-*-bin-win-cuda-12.*-x64.zip",
    "llama-*-bin-win-cuda-13.*-x64.zip",
)


class LlamaServerNotFound(RuntimeError):
    pass


def _binary_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("llama-server.exe",)
    return ("llama-server",)


def _local_install_candidates(prefer_cuda: bool = False) -> list[Path]:
    """All paths a project-local install could live at."""
    out: list[Path] = []
    if prefer_cuda:
        for name in _binary_names():
            out.append(TOOLS_DIR / "cuda" / name)
    for name in _binary_names():
        out.append(TOOLS_DIR / name)
        # llama.cpp Windows zips sometimes nest one folder deep
        for sub in TOOLS_DIR.glob("*"):
            if sub.is_dir():
                out.append(sub / name)
    return out


def is_cuda_build(server_path: str | Path) -> bool:
    """True if this llama-server was built with the CUDA backend.

    llama.cpp ships its GPU backend as a sibling ``ggml-cuda`` library; the CPU
    zips only contain ``ggml-cpu-*``."""
    folder = Path(server_path).resolve().parent
    return any(folder.glob("ggml-cuda*.dll")) or any(folder.glob("libggml-cuda*.so"))


def _all_llama_servers(prefer_cuda: bool) -> list[str]:
    """Every llama-server we can find, in precedence order, de-duplicated."""
    found: list[str] = []

    def add(p: str | Path | None) -> None:
        if p and Path(p).is_file():
            s = str(Path(p).resolve())
            if s not in found:
                found.append(s)

    # 1. Explicit environment variable (full path or bare command name)
    env = os.environ.get("LLAMA_CPP_BINARY")
    if env:
        add(env if Path(env).is_file() else shutil.which(env))
    # 2. On system PATH
    for name in _binary_names():
        add(shutil.which(name) or shutil.which(name.replace(".exe", "")))
    # 3. Project-local .tools/llama-cpp/
    for candidate in _local_install_candidates(prefer_cuda=prefer_cuda):
        add(candidate)
    return found


def locate_llama_server(prefer_cuda: bool = False) -> str | None:
    """Return an existing llama-server binary path, or None.

    With ``prefer_cuda`` a CUDA build anywhere beats a CPU build earlier in the
    precedence order — a CPU path left in LLAMA_CPP_BINARY by a previous run
    must not shadow a CUDA build installed since."""
    servers = _all_llama_servers(prefer_cuda)
    if prefer_cuda:
        for s in servers:
            if is_cuda_build(s):
                log.debug("llama-server (CUDA build): %s", s)
                return s
    if servers:
        log.debug("llama-server: %s", servers[0])
        return servers[0]
    return None


def ensure_llama_server(auto_install: bool = False, *, prefer_cuda: bool = False) -> str:
    """Return a llama-server path. Raise LlamaServerNotFound with an actionable
    message if none is available. When ``auto_install=True`` this may download
    a Windows binary (see install_llama_cpp_windows); with ``prefer_cuda`` it
    also downloads the CUDA build when only a CPU build is present."""
    path = locate_llama_server(prefer_cuda=prefer_cuda)
    can_install = auto_install and os.name == "nt"
    upgrade_to_cuda = can_install and prefer_cuda and path is not None and not is_cuda_build(path)
    if path and not upgrade_to_cuda:
        os.environ["LLAMA_CPP_BINARY"] = path
        return path

    if can_install:
        log.info("CUDA llama-server not found — auto-installing…" if prefer_cuda
                 else "llama-server not found — auto-installing…")
        install_llama_cpp_windows(cuda=prefer_cuda)
        path = locate_llama_server(prefer_cuda=prefer_cuda)
        if path:
            os.environ["LLAMA_CPP_BINARY"] = path
            return path

    raise LlamaServerNotFound(
        "llama-server binary not found. Run:\n"
        "    python scripts/install_llama_cpp.py\n"
        "…or download a release manually from "
        "https://github.com/ggml-org/llama.cpp/releases and set "
        "LLAMA_CPP_BINARY to its full path."
    )


# ---------------------------------------------------------------------------
# Auto-installer (Windows only)
# ---------------------------------------------------------------------------
def _pick_windows_asset(release: dict, *, cuda: bool = False) -> tuple[str, str]:
    import fnmatch
    assets = release.get("assets", [])
    names = [a["name"] for a in assets]
    log.debug("Release assets: %s", names)
    preferences = _WINDOWS_CUDA_ASSET_PREFERENCES if cuda else _WINDOWS_CPU_ASSET_PREFERENCES
    for pattern in preferences:
        for a in assets:
            if fnmatch.fnmatch(a["name"], pattern):
                return a["name"], a["browser_download_url"]
    raise LlamaServerNotFound(
        "No suitable Windows llama.cpp asset in the latest release. "
        f"Available: {names}"
    )


def _download(url: str, dest: Path, progress: Callable[[int, int], None] | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "ocr-benchmark"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        read = 0
        chunk = 1 << 16
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                read += len(buf)
                if progress:
                    progress(read, total)
    log.info("Downloaded %d bytes → %s", dest.stat().st_size, dest)


def _github_json(url: str) -> object:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ocr-benchmark", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _find_release_with_windows_assets(*, cuda: bool = False) -> dict:
    """The upstream 'latest' release is often just a nightly metadata blob with no
    binaries. Walk /releases and return the newest one whose asset list contains a
    matching Windows zip."""
    import fnmatch
    releases = _github_json(GITHUB_RELEASES_LIST)
    if not isinstance(releases, list):
        raise LlamaServerNotFound("Unexpected GitHub API response.")
    for rel in releases:
        assets = rel.get("assets") or []
        for a in assets:
            preferences = _WINDOWS_CUDA_ASSET_PREFERENCES if cuda else _WINDOWS_CPU_ASSET_PREFERENCES
            for pattern in preferences:
                if fnmatch.fnmatch(a["name"], pattern):
                    return rel
    raise LlamaServerNotFound(
        "No recent llama.cpp release has a Windows CPU binary matching "
        f"{_WINDOWS_CUDA_ASSET_PREFERENCES if cuda else _WINDOWS_CPU_ASSET_PREFERENCES}. Check "
        "https://github.com/ggml-org/llama.cpp/releases manually."
    )


def install_llama_cpp_windows(
    progress: Callable[[int, int], None] | None = None,
    *,
    cuda: bool = False,
) -> Path:
    """Download llama.cpp into ``.tools/llama-cpp/`` and return llama-server.

    ``cuda=True`` installs the CUDA build in a separate ``cuda`` folder, so it
    never overwrites a working CPU fallback.  The CUDA runtime DLL archive is
    downloaded as well when the release provides one.
    """
    if os.name != "nt":
        raise LlamaServerNotFound(
            "install_llama_cpp_windows only runs on Windows. "
            "On macOS: `brew install llama.cpp`. "
            "On Linux: install llama.cpp via your package manager or from source."
        )

    log.info("Fetching llama.cpp release list…")
    release = _find_release_with_windows_assets(cuda=cuda)
    asset_name, download_url = _pick_windows_asset(release, cuda=cuda)
    tag = release.get("tag_name", "?")
    log.info("Selected asset %s from release %s", asset_name, tag)

    install_dir = TOOLS_DIR / "cuda" if cuda else TOOLS_DIR
    install_dir.mkdir(parents=True, exist_ok=True)
    zip_path = install_dir / asset_name
    _download(download_url, zip_path, progress)

    log.info("Extracting %s…", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(install_dir)
    try:
        zip_path.unlink()
    except OSError:
        pass

    # CUDA archives omit the CUDA runtime DLLs.  Download the matching runtime
    # archive when available so a system CUDA toolkit is not required.
    if cuda:
        runtime_name = asset_name.replace("llama-", "cudart-llama-", 1)
        runtime = next((a for a in release.get("assets", []) if a["name"] == runtime_name), None)
        if runtime:
            runtime_zip = install_dir / runtime_name
            _download(runtime["browser_download_url"], runtime_zip, progress)
            with zipfile.ZipFile(runtime_zip) as z:
                z.extractall(install_dir)
            try:
                runtime_zip.unlink()
            except OSError:
                pass

    server_path = install_dir / "llama-server.exe"
    server = str(server_path.resolve()) if server_path.is_file() else None
    if not server:
        raise LlamaServerNotFound(
            f"Extracted {asset_name} but no llama-server.exe found in {install_dir}."
        )
    log.info("llama-server ready at %s", server)
    return Path(server)


if __name__ == "__main__":
    from engines.logging_setup import setup_logging
    setup_logging()
    try:
        p = ensure_llama_server(auto_install=True)
        print(f"llama-server: {p}")
    except LlamaServerNotFound as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
