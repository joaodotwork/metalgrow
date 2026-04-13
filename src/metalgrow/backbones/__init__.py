# Register built-in backbones by importing their modules for side effects.
from metalgrow.backbones import bicubic as _bicubic  # noqa: F401
from metalgrow.backbones import realesrgan as _realesrgan  # noqa: F401
from metalgrow.backbones.base import Backbone
from metalgrow.backbones.registry import get_backbone, list_backbones, register

__all__ = ["Backbone", "get_backbone", "list_backbones", "register"]
