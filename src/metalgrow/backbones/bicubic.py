from __future__ import annotations

import torch
import torch.nn.functional as F

from metalgrow.backbones.base import Backbone
from metalgrow.backbones.registry import register


class BicubicBackbone(Backbone):
    """Analytical bicubic resampler. Always available; no weights required."""

    name = "bicubic"
    supported_scales = None  # any scale > 1.0

    def upscale(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        self.validate_scale(scale)
        _, _, h, w = tensor.shape
        return F.interpolate(
            tensor,
            size=(int(h * scale), int(w * scale)),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)


register("bicubic", lambda device, dtype: BicubicBackbone(device, dtype))
