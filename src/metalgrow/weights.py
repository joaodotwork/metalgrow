"""Weight file registry + download-on-first-use cache for learned backbones."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class WeightSpec:
    name: str
    url: str
    sha256: str


REGISTRY: dict[str, WeightSpec] = {
    "realesrgan-x2": WeightSpec(
        name="RealESRGAN_x2plus.pth",
        url=(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ),
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
    ),
    "realesrgan-x4": WeightSpec(
        name="RealESRGAN_x4plus.pth",
        url=(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ),
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    ),
}


def cache_dir() -> Path:
    """Return the weight cache directory, honoring METALGROW_CACHE_DIR."""
    override = os.environ.get("METALGROW_CACHE_DIR")
    root = Path(override) if override else Path.home() / ".cache" / "metalgrow"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    """Stream ``url`` to ``dst`` atomically, printing coarse progress to stderr."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urlopen(url) as resp:  # noqa: S310 — URLs come from a checksum-verified registry
        total = int(resp.headers.get("Content-Length", 0) or 0)
        read = 0
        with tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                read += len(chunk)
                if total:
                    pct = read * 100 // total
                    print(f"\rdownloading {dst.name}: {pct}%", end="", file=sys.stderr)
        if total:
            print("", file=sys.stderr)
    shutil.move(tmp, dst)


def ensure_weight(backbone: str) -> Path:
    """Return a local path to the verified weight file for ``backbone``.

    Downloads on first use into :func:`cache_dir`; re-verifies the checksum on
    every call. Raises ``RuntimeError`` if the downloaded file fails the hash
    check (and removes the bad file so a retry can re-download).
    """
    try:
        spec = REGISTRY[backbone]
    except KeyError:
        raise KeyError(f"no weights registered for backbone {backbone!r}") from None

    dst = cache_dir() / spec.name
    if not dst.exists():
        _download(spec.url, dst)

    actual = _sha256_of(dst)
    if actual != spec.sha256:
        dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {spec.name}: expected {spec.sha256}, got {actual}. "
            "Removed the corrupt file; retry to re-download."
        )
    return dst
