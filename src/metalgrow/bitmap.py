"""1-bit line-art masters: the print-shop format for single-ink artwork.

Coverage (alpha, or inverted luminance for an opaque scan) is Lanczos-resampled
to the target ppi at the image's physical size, thresholded, and returned as a
mode ``1`` image with black = ink. Resampling *before* thresholding is what
keeps the edges smooth: thresholding at 300 dpi and scaling up would print the
300 dpi stair-steps. The resample runs in horizontal strips (bit-identical to
one pass), so a 9k px source becoming a 38k px master stays within memory.
"""

from __future__ import annotations

from math import ceil, gcd

from PIL import Image, ImageOps

# Target output rows per strip.
_STRIP = 1024


def coverage(image: Image.Image) -> Image.Image:
    """Ink coverage as ``L``: alpha when present, otherwise inverted luminance."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        return image.convert("RGBA").getchannel("A")
    return ImageOps.invert(image.convert("L"))


def to_bitmap(
    image: Image.Image,
    ppi: int = 1200,
    threshold: int = 128,
    src_dpi: float | None = None,
) -> Image.Image:
    """Return a mode ``1`` master at ``ppi`` (same physical size as ``image``).

    ``src_dpi`` defaults to the image's DPI tag. With no tag the master keeps
    the source's pixel size and is only tagged ``ppi``.
    """
    if not 0 < threshold <= 255:
        raise ValueError("threshold must be in 1..255")
    cov = coverage(image)
    if src_dpi is None:
        dpi = image.info.get("dpi")
        src_dpi = float(dpi[0]) if dpi else float(ppi)
    k = ppi / src_dpi
    w, h = max(1, round(cov.width * k)), max(1, round(cov.height * k))

    lut = [0 if v >= threshold else 255 for v in range(256)]  # ink -> black
    master = Image.new("1", (w, h), 1)
    for y0, strip in _strips(cov, (w, h)):
        master.paste(strip.point(lut).convert("1", dither=Image.Dither.NONE), (0, y0))
    return master


def _strips(src: Image.Image, size: tuple[int, int]):
    """Yield ``(y, rows)`` of ``src.resize(size, LANCZOS)`` a strip at a time.

    Bit-identical to the single-pass resize. Strips start on rows where the
    source and output grids coincide (every ``src_h / g`` source rows, ``g`` =
    gcd of the heights), and each one resamples a crop with enough margin to
    cover the Lanczos support, then trims it. (Pillow's ``resize(box=...)``
    looks like the obvious tool, but its result differs from a single pass by
    up to 10 levels mid-strip.)
    """
    w, h = size
    g = gcd(src.height, h)
    s_step, o_step = src.height // g, h // g
    per = max(1, _STRIP // o_step)
    support = ceil(3 * max(1.0, src.height / h)) + 1  # Lanczos-3, in source px
    margin = ceil(support / s_step)
    for u0 in range(0, g, per):
        u1 = min(g, u0 + per)
        c0, c1 = max(0, u0 - margin), min(g, u1 + margin)
        crop = src.crop((0, c0 * s_step, src.width, c1 * s_step))
        rows = crop.resize((w, (c1 - c0) * o_step), Image.LANCZOS)
        yield u0 * o_step, rows.crop((0, (u0 - c0) * o_step, w, (u1 - c0) * o_step))
