"""Ink mode: upscale single-colour artwork through its coverage, not its RGB.

Ink-on-transparent artwork (a scanned drawing knocked out of its paper) keeps
every bit of detail in the alpha channel; the RGB plane is one flat colour.
The default RGBA path runs the backbone on that flat plane and bicubics the
alpha, so the result is a plain bicubic upscale. Ink mode instead composites
the artwork over white paper, runs the backbone on that ordinary
dark-on-light image (squarely in-distribution for an SR model), and turns the
upscaled luminance back into coverage = alpha. Opaque scans on paper go
through the same path, which knocks the paper out as a side effect.
See ADR 0003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

ALPHA_MODES = ("separate", "ink", "auto")

# Rec. 601 luma weights; the exact choice is immaterial for monochrome ink.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# auto: a visible pixel further than this from the median colour (per channel,
# 0..1) counts as off-ink; more than _MAX_OFF_INK of them means real colour art.
_INK_TOLERANCE = 24 / 255
_MAX_OFF_INK = 0.005


def _rgba_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0


def is_single_ink(image: Image.Image) -> bool:
    """True when ``image`` has transparency and every visible pixel is one colour."""
    if image.mode not in ("RGBA", "LA") and "transparency" not in image.info:
        return False
    px = _rgba_float(image)
    visible = px[..., :3][px[..., 3] >= 0.5]
    if visible.size == 0:
        return False
    median = np.median(visible, axis=0)
    off = np.abs(visible - median).max(axis=1) > _INK_TOLERANCE
    return float(off.mean()) <= _MAX_OFF_INK


@dataclass(frozen=True)
class InkMatte:
    """The ink colour and paper level that map luminance back to coverage."""

    ink: tuple[float, float, float]  # RGB in 0..1
    ink_luma: float
    paper_luma: float

    @classmethod
    def fit(cls, image: Image.Image) -> tuple[InkMatte, torch.Tensor]:
        """Fit the matte and return it with the composite luminance ``[1,1,H,W]``."""
        px = _rgba_float(image)
        rgb, alpha = px[..., :3], px[..., 3:4]
        over_white = rgb * alpha + (1.0 - alpha)
        luma = over_white @ _LUMA

        # Ink: the colour at full coverage, i.e. the darkest pixels over white.
        # Edge pixels are ink blended with paper and would bias a plain mean
        # light. Keying on "opaque" pixels instead breaks on an opaque scan
        # that happens to carry a few stray alpha values: its opaque pixels
        # are mostly paper.
        ink = np.median(over_white[luma <= luma.min() + 0.02], axis=0)
        # Paper: near-white even when a scan's paper isn't 255, but robust to
        # the odd blown-out speck.
        paper = float(np.percentile(luma, 99))
        if paper - float(luma.min()) < 0.1:
            paper = 1.0  # ink covers nearly everything; assume white paper

        matte = cls(
            ink=(float(ink[0]), float(ink[1]), float(ink[2])),
            ink_luma=float(ink @ _LUMA),
            paper_luma=paper,
        )
        return matte, torch.from_numpy(luma)[None, None]

    @property
    def neutral(self) -> bool:
        return max(self.ink) - min(self.ink) <= 2 / 255

    def compose(self, luma: np.ndarray) -> Image.Image:
        """Turn an upscaled luminance plane back into ink + alpha."""
        span = self.paper_luma - self.ink_luma
        if span < 1e-3:  # no ink at all: nothing is covered
            coverage = np.zeros_like(luma)
        else:
            coverage = np.clip((self.paper_luma - luma) / span, 0.0, 1.0)
        alpha = np.round(coverage * 255).astype(np.uint8)

        if self.neutral:
            # One grey ink prints as one ink: keep it single-channel.
            grey = np.full_like(alpha, round(self.ink_luma * 255))
            return Image.fromarray(np.stack([grey, alpha], axis=-1))
        ink = np.round(np.array(self.ink) * 255).astype(np.uint8)
        rgb = np.broadcast_to(ink, (*alpha.shape, 3))
        return Image.fromarray(np.dstack([rgb, alpha]))
