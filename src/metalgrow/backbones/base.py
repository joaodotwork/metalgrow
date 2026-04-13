from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch


class Backbone(ABC):
    """Pluggable super-resolution backbone.

    Implementations accept a float tensor in [0, 1] of shape [N, C, H, W] on
    ``self.device`` and return an upscaled tensor of the same dtype/device.
    Preprocessing, model inference, and postprocessing all live inside
    :meth:`upscale` so that the shared :class:`~metalgrow.upscaler.Upscaler`
    stays agnostic to backbone internals.
    """

    name: ClassVar[str]
    # None means any scale > 1.0 is accepted (e.g. analytical resamplers).
    supported_scales: ClassVar[tuple[float, ...] | None] = None

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

    def validate_scale(self, scale: float) -> None:
        if scale <= 1.0:
            raise ValueError("scale must be > 1.0")
        if self.supported_scales is not None and scale not in self.supported_scales:
            allowed = ", ".join(f"{s:g}" for s in self.supported_scales)
            raise ValueError(f"backbone {self.name!r} supports scales [{allowed}], got {scale:g}")

    @abstractmethod
    def upscale(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Return an upscaled copy of ``tensor``."""
