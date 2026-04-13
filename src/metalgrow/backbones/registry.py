from __future__ import annotations

from collections.abc import Callable

import torch

from metalgrow.backbones.base import Backbone

BackboneFactory = Callable[[torch.device, torch.dtype], Backbone]

_REGISTRY: dict[str, BackboneFactory] = {}


def register(name: str, factory: BackboneFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"backbone {name!r} already registered")
    _REGISTRY[name] = factory


def get_backbone(name: str, device: torch.device, dtype: torch.dtype) -> Backbone:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"unknown backbone {name!r}; available: {available}") from None
    return factory(device, dtype)


def list_backbones() -> list[str]:
    return sorted(_REGISTRY)
