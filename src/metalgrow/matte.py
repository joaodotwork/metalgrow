"""Matte: knock the paper out of full-colour art, solid glass and translucent shadow.

Sources are full-colour renders on near-white paper, upscaled first (the matte
is per-pixel, so it is as sharp as its input). For each image:

- **Outside the glass**, colour-to-alpha against the paper colour sampled from
  the border: the cast shadow becomes translucent sepia that re-composites
  exactly over the paper.
- **The glass** is everything the paper can't reach from the border through
  light pixels (unmixed alpha < ``T``). Enclosed specks smaller than
  ``MIN_AREA`` are shadow hatching and stay outside. Inside, the render is kept
  opaque as drawn, except a thin ``EDGE`` ring that uses the unmixed matte, so
  the antialiased outline carries no paper-white fringe onto dark backgrounds.
- **Haze:** paper is never perfectly flat, so unmixing leaves a faint alpha
  haze (1-8/255) across the background. Alpha below ``HAZE`` is cleared and the
  rest rescaled, which moves the result over the paper by at most ``HAZE``.
- **Seal** (opt-in) is for glass with white highlights but no closed dark
  outline, where the paper floods into the glass through the highlights and
  they come out as holes. It closes gaps up to ``SEAL_R`` (fraction of the
  width) between neutral strokes and keeps as glass every enclosed area of at
  least ``SEAL_HOLE``, within each row's span of the glass found above. Cast
  shadows are hatched in warm sepia, above ``SEAL_CHROMA``, so they don't seal.

Ported from The Vanguard's ``matte_drinks.py``; output is pixel-identical to
it, with and without seal. Contrast with ink mode (``metalgrow.ink``), which
handles single-colour art. See ADR 0004.

Pure numpy + ``scipy.ndimage``. scipy is the optional ``metalgrow[matte]``
extra and is imported lazily, so importing this module never needs it (or
torch).
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image

T = 0.5  # unmixed alpha at or above which a pixel blocks the paper flood
MIN_AREA = 0.004  # smallest enclosed region kept as glass, fraction of the image
EDGE = 3  # px at the output size: ring inside the glass that keeps the unmixed matte
HAZE = 4 / 255  # paper noise floor: alpha below this is paper, the rest is rescaled
SEAL_R = 0.006  # gap closed between neutral strokes, fraction of the image width
SEAL_CHROMA = 0.12  # (max - min) / 255 at or under which a stroke is neutral linework
SEAL_HOLE = 0.0005  # smallest enclosed area kept as glass by seal, fraction of the image
BORDER = 16  # px band along each edge sampled for the paper colour

INSTALL_HINT = "metalgrow matte needs scipy: pip install 'metalgrow[matte]'"


class MatteDependencyError(ImportError):
    """scipy (the ``metalgrow[matte]`` extra) is not installed."""


def _ndimage() -> ModuleType:
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise MatteDependencyError(INSTALL_HINT) from exc
    return ndimage


def paper_colour(rgb: np.ndarray, w: int = BORDER) -> np.ndarray:
    """Median colour of a ``w`` px band along all four edges."""
    b = np.concatenate(
        [
            rgb[:w].reshape(-1, 3),
            rgb[-w:].reshape(-1, 3),
            rgb[:, :w].reshape(-1, 3),
            rgb[:, -w:].reshape(-1, 3),
        ]
    )
    return np.median(b, axis=0)


def clear_haze(a: np.ndarray) -> np.ndarray:
    """Clear alpha below ``HAZE`` and rescale the rest to stay continuous."""
    return np.clip((a - HAZE) / (1 - HAZE), 0, 1)


def _big(mask: np.ndarray, frac: float) -> np.ndarray:
    """Connected components of ``mask`` covering at least ``frac`` of the image."""
    nd = _ndimage()
    lab, n = nd.label(mask)
    if not n:
        return mask
    sizes = nd.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes >= frac * mask.size))


def _close(mask: np.ndarray, r: float) -> np.ndarray:
    """Morphological closing by a disc of radius ``r`` (via two distance transforms)."""
    nd = _ndimage()
    return nd.distance_transform_edt(nd.distance_transform_edt(~mask) <= r) > r


def _spans(mask: np.ndarray) -> np.ndarray:
    """Each row filled between its first and last pixel of ``mask``."""
    idx = np.arange(mask.shape[1])
    lo = np.where(mask, idx, mask.shape[1]).min(1)[:, None]
    hi = np.where(mask, idx, -1).max(1)[:, None]
    return (idx >= lo) & (idx <= hi)


def sealed(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Glass enclosed by neutral linework once its gaps are closed."""
    nd = _ndimage()
    strong = a >= T
    neutral = strong & ((rgb.max(axis=2) - rgb.min(axis=2)) / 255 <= SEAL_CHROMA)
    closed = _close(neutral, SEAL_R * rgb.shape[1])
    wall = closed | strong
    filled = nd.binary_fill_holes(wall)
    return _big(filled & ~wall, SEAL_HOLE) | (closed & filled)


def matte(rgb: np.ndarray, seal: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Matte an ``[H,W,3]`` float64 image in 0..255.

    Returns ``(colour, alpha, paper, glass)``: un-premultiplied colour in
    0..255, alpha in 0..1, the sampled paper colour, and the fraction of the
    image that is glass.
    """
    nd = _ndimage()
    paper = paper_colour(rgb)
    a = np.clip(((paper - rgb) / np.maximum(paper, 1)).max(axis=2), 0, 1)
    col = np.clip(paper - (paper - rgb) / np.maximum(a, 1e-6)[..., None], 0, 255)
    col[a == 0] = 0

    lab, _ = nd.label(a < T)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    inside = ~np.isin(lab, border[border > 0])
    glass = _big(inside, MIN_AREA)
    if seal:
        glass = glass | (sealed(rgb, a) & _spans(glass))
    glass = nd.binary_fill_holes(glass)
    core = nd.binary_erosion(glass, iterations=EDGE)

    alpha = np.where(core, 1.0, clear_haze(a))
    out = np.where(core[..., None], rgb, col)
    return out, alpha, paper, float(glass.mean())


def to_rgba(colour: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """Quantise a :func:`matte` result to an 8-bit RGBA image."""
    rgba = np.dstack([colour, alpha * 255]).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def matte_image(image: Image.Image, seal: bool = False) -> tuple[Image.Image, np.ndarray, float]:
    """Matte a PIL image; returns ``(rgba, paper, glass)``."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    colour, alpha, paper, glass = matte(rgb, seal)
    return to_rgba(colour, alpha), paper, glass


def matte_file(
    src: Path, dst: Path, seal: bool = False, dpi: float = 300
) -> tuple[np.ndarray, float]:
    """Matte ``src`` into a PNG at ``dst`` tagged ``dpi``; returns ``(paper, glass)``."""
    with Image.open(src) as image:
        rgba, paper, glass = matte_image(image, seal)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(dst, format="PNG", dpi=(dpi, dpi), optimize=False, compress_level=6)
    return paper, glass


def read_names(path: Path) -> dict[str, str]:
    """Read a ``source filename,output filename`` CSV (blank and ``#`` lines skipped).

    Raises ``ValueError`` on a malformed row, a duplicate source, two sources
    mapped to the same output, or an output that isn't a plain file name.
    """
    import csv

    names: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for n, row in enumerate(csv.reader(fh), 1):
            if not row or not "".join(row).strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) != 2:
                raise ValueError(f"{path}:{n}: expected 2 columns, got {len(row)}")
            src, out = (c.strip() for c in row)
            if not src or not out:
                raise ValueError(f"{path}:{n}: empty source or output name")
            if Path(out).name != out or out in (".", ".."):
                raise ValueError(f"{path}:{n}: output {out!r} must be a plain file name")
            if Path(out).suffix.lower() != ".png":
                raise ValueError(f"{path}:{n}: output {out!r} must end in .png")
            if src in names:
                raise ValueError(f"{path}:{n}: source {src!r} listed twice")
            names[src] = out
    return names


def plan_names(
    inputs: list[Path], dst_dir: Path, names: dict[str, str] | None = None
) -> list[tuple[Path, Path]]:
    """Map inputs to ``dst_dir / <stem>.png``, or to their name in ``names``.

    Raises ``ValueError`` if ``names`` lists a source that isn't an input, if
    two inputs would be written to the same file, or if an output is an input
    (see :func:`check_no_overwrite`).
    """
    names = names or {}
    unknown = sorted(set(names) - {p.name for p in inputs})
    if unknown:
        raise ValueError(f"--names lists sources not in the input: {', '.join(unknown)}")
    plan = [(p, dst_dir / names.get(p.name, p.with_suffix(".png").name)) for p in inputs]
    seen: dict[Path, Path] = {}
    for src, dst in plan:
        if dst in seen:
            raise ValueError(f"{seen[dst].name} and {src.name} would both be written to {dst.name}")
        seen[dst] = src
    check_no_overwrite(plan)
    return plan


def check_no_overwrite(plan: list[tuple[Path, Path]]) -> None:
    """Raise ``ValueError`` if any output resolves to any input of ``plan``.

    Sources are read and closed before their output is written, so without
    this check ``metalgrow matte renders/ renders/`` would replace the
    sources with their mattes.
    """
    sources = {src.resolve(): src for src, _ in plan}
    for src, dst in plan:
        hit = sources.get(dst.resolve())
        if hit is not None:
            raise ValueError(f"{src.name} would be written over the source {hit}")
