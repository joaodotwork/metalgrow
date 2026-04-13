from __future__ import annotations

import torch

from metalgrow.backbones.base import Backbone
from metalgrow.backbones.registry import register
from metalgrow.weights import ensure_weight


class SwinIRBackbone(Backbone):
    """SwinIR (Swin Transformer) backbone loaded via :mod:`spandrel`.

    SwinIR trades more memory and latency for generally sharper outputs than
    Real-ESRGAN on clean, photographic content. Attention-heavy, so we tile
    at a smaller default than Real-ESRGAN to keep per-tile activations in
    VRAM on the target device.
    """

    name = "swinir"
    supported_scales = (2.0, 4.0)
    input_channels = 3
    default_tile = 128
    default_tile_pad = 16

    def __init__(self, device: torch.device, dtype: torch.dtype, registry_key: str):
        super().__init__(device, dtype)
        self._registry_key = registry_key
        self._model: torch.nn.Module | None = None

    def _load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        from spandrel import ModelLoader

        weights_path = ensure_weight(self._registry_key)
        descriptor = ModelLoader().load_from_file(str(weights_path))
        model = descriptor.model.to(self.device, self.dtype)
        model.eval()
        self._model = model
        return model

    def upscale(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        self.validate_scale(scale)
        model = self._load()
        tensor = tensor.to(self.device, self.dtype)
        with torch.no_grad():
            return model(tensor).clamp(0.0, 1.0)


def _factory(registry_key: str):
    def make(device: torch.device, dtype: torch.dtype) -> SwinIRBackbone:
        return SwinIRBackbone(device, dtype, registry_key=registry_key)

    return make


register("swinir-x2", _factory("swinir-x2"))
register("swinir-x4", _factory("swinir-x4"))
