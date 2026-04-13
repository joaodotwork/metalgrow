import pytest
import torch

from metalgrow import Upscaler
from metalgrow.backbones import get_backbone, list_backbones, register
from metalgrow.backbones.base import Backbone


def test_builtin_backbones_registered():
    names = list_backbones()
    assert "bicubic" in names
    assert "realesrgan-x2" in names
    assert "realesrgan-x4" in names


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="unknown backbone"):
        Upscaler(backbone="does-not-exist", device="cpu")


def test_duplicate_registration_raises():
    def factory(device, dtype):
        raise AssertionError("unused")

    with pytest.raises(ValueError, match="already registered"):
        register("bicubic", factory)


def test_bicubic_accepts_arbitrary_scale():
    bb = get_backbone("bicubic", torch.device("cpu"), torch.float32)
    t = torch.zeros(1, 3, 8, 8)
    out = bb.upscale(t, scale=1.5)
    assert out.shape == (1, 3, 12, 12)


def test_bicubic_rejects_scale_le_one():
    bb = get_backbone("bicubic", torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="scale must be > 1.0"):
        bb.upscale(torch.zeros(1, 3, 4, 4), scale=1.0)


def test_realesrgan_scale_enforcement():
    bb = get_backbone("realesrgan-x4", torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match=r"supports scales \[2, 4\]"):
        bb.upscale(torch.zeros(1, 3, 4, 4), scale=3.0)


def test_backbone_is_abstract():
    with pytest.raises(TypeError):
        Backbone(torch.device("cpu"))  # type: ignore[abstract]
