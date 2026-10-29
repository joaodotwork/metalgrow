"""Capture and reapply print-relevant image metadata across an upscale.

The SR pipeline runs in RGB(A) and writes whatever PIL infers from the output
tensor, which silently discards three things that matter for print / preflight
workflows: the embedded ICC profile, the DPI tag, and a grayscale color mode
(a 1-channel scan comes back as 3-channel RGB). This module captures those on
the source image and reapplies them to the result so a grayscale scan stays
grayscale, the color profile survives, and the DPI is rescaled to keep the
image's *physical* size constant as its pixel count grows.
"""

from __future__ import annotations

from PIL import Image

# Modes PIL reports for single-channel / grayscale imagery. An RGB(A) result
# whose source was one of these is converted back so it prints as one ink.
GRAYSCALE_MODES = frozenset({"L", "LA", "I", "I;16", "I;16B", "I;16L", "1"})


def capture(image: Image.Image) -> dict:
    """Snapshot the metadata we want to survive an upscale."""
    return {
        "mode": image.mode,
        "icc_profile": image.info.get("icc_profile"),
        "dpi": image.info.get("dpi"),
        "size": image.size,  # (width, height) of the source, pre-upscale
    }


def reapply(result: Image.Image, meta: dict, dpi: float | None = None) -> tuple[Image.Image, dict]:
    """Restore grayscale mode and return ``(image, save_kwargs)``.

    ``save_kwargs`` carries the ICC profile (verbatim) and a DPI rescaled by the
    upscale ratio, so passing it to ``Image.save`` preserves both the color
    pipeline and the physical dimensions of the placed image. ``dpi``
    overrides the rescaled tag, for jobs specified as "print at N dpi".
    """
    src_mode = meta.get("mode")
    if src_mode in GRAYSCALE_MODES and result.mode not in ("L", "LA"):
        result = result.convert("LA" if result.mode == "RGBA" else "L")

    save_kwargs: dict = {}
    icc = meta.get("icc_profile")
    # An RGB profile on a grey image is invalid (ink mode can turn RGBA into
    # LA), so only carry the profile across when its colour space still fits.
    if icc and not (result.mode in ("L", "LA") and icc[16:20] == b"RGB "):
        save_kwargs["icc_profile"] = icc

    src_dpi = meta.get("dpi")
    size = meta.get("size")
    if dpi is not None:
        save_kwargs["dpi"] = (dpi, dpi)
    elif src_dpi and size and size[0]:
        ratio = result.width / size[0]
        save_kwargs["dpi"] = (src_dpi[0] * ratio, src_dpi[1] * ratio)

    return result, save_kwargs
