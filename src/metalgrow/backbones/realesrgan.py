from __future__ import annotations

import torch

from metalgrow.backbones.base import Backbone
from metalgrow.backbones.registry import register


class RealESRGANBackbone(Backbone):
    """Real-ESRGAN RRDBNet backbone. Weights wiring lands with issue #6."""

    name = "realesrgan"
    supported_scales = (2.0, 4.0)

    def __init__(self, device: torch.device, dtype: torch.dtype, scale: int):
        super().__init__(device, dtype)
        self._scale = scale

    def upscale(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        self.validate_scale(scale)
        raise NotImplementedError("Real-ESRGAN backbone is not yet wired up; tracked in issue #6.")


def _factory(scale: int):
    def make(device: torch.device, dtype: torch.dtype) -> RealESRGANBackbone:
        return RealESRGANBackbone(device, dtype, scale=scale)

    return make


register("realesrgan-x2", _factory(2))
register("realesrgan-x4", _factory(4))
