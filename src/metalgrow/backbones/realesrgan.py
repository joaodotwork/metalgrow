from __future__ import annotations

import torch

from metalgrow.backbones.base import Backbone
from metalgrow.backbones.registry import register
from metalgrow.weights import ensure_weight


class RealESRGANBackbone(Backbone):
    """Real-ESRGAN RRDBNet backbone loaded via :mod:`spandrel`."""

    name = "realesrgan"
    supported_scales = (2.0, 4.0)
    input_channels = 3

    # Tile input at this spatial size to bound VRAM on large images.
    # Real-ESRGAN is convolutional so the output is deterministic in tiles
    # as long as we pad with enough context ("overlap") on every edge.
    TILE = 256
    OVERLAP = 16

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
            return self._tiled_forward(model, tensor, int(scale))

    def _tiled_forward(self, model: torch.nn.Module, x: torch.Tensor, scale: int) -> torch.Tensor:
        _, _, h, w = x.shape
        if h <= self.TILE and w <= self.TILE:
            return model(x).clamp(0.0, 1.0)

        out = torch.zeros(
            x.shape[0],
            x.shape[1],
            h * scale,
            w * scale,
            dtype=x.dtype,
            device=x.device,
        )
        step = self.TILE - self.OVERLAP
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1 = min(y0 + self.TILE, h)
                x1 = min(x0 + self.TILE, w)
                # Pull overlap context from the inside edge so tile outputs
                # already include valid convolutional receptive field; we
                # then crop the output back to the non-overlapping region.
                py0 = max(0, y0 - self.OVERLAP)
                px0 = max(0, x0 - self.OVERLAP)
                py1 = min(h, y1 + self.OVERLAP)
                px1 = min(w, x1 + self.OVERLAP)
                tile = x[:, :, py0:py1, px0:px1]
                tile_out = model(tile).clamp(0.0, 1.0)

                # Slice out the region corresponding to [y0:y1, x0:x1] in the
                # original input coordinates, in the upscaled output frame.
                top = (y0 - py0) * scale
                left = (x0 - px0) * scale
                bot = top + (y1 - y0) * scale
                right = left + (x1 - x0) * scale
                out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = tile_out[
                    :, :, top:bot, left:right
                ]
        return out


def _factory(registry_key: str):
    def make(device: torch.device, dtype: torch.dtype) -> RealESRGANBackbone:
        return RealESRGANBackbone(device, dtype, registry_key=registry_key)

    return make


register("realesrgan-x2", _factory("realesrgan-x2"))
register("realesrgan-x4", _factory("realesrgan-x4"))
