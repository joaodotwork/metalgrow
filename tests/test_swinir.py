"""SwinIR integration tests.

These tests require the SwinIR weight files to be available in the cache
directory. They are skipped in CI (no weights) and run locally once the
user has fetched weights via ``metalgrow models download swinir-xN``.
"""

from __future__ import annotations

import pytest
from PIL import Image

from metalgrow import Upscaler, weights


def _weights_available(backbone: str) -> bool:
    spec = weights.REGISTRY[backbone]
    path = weights.cache_dir() / spec.name
    return path.exists()


requires_x2 = pytest.mark.skipif(
    not _weights_available("swinir-x2"),
    reason="SwinIR x2 weights not cached; run `metalgrow models download swinir-x2`",
)
requires_x4 = pytest.mark.skipif(
    not _weights_available("swinir-x4"),
    reason="SwinIR x4 weights not cached; run `metalgrow models download swinir-x4`",
)


def test_swinir_registered_in_backbone_registry():
    from metalgrow.backbones import list_backbones

    names = list_backbones()
    assert "swinir-x2" in names
    assert "swinir-x4" in names


def test_swinir_registered_in_weights_registry():
    assert "swinir-x2" in weights.REGISTRY
    assert "swinir-x4" in weights.REGISTRY
    for key in ("swinir-x2", "swinir-x4"):
        spec = weights.REGISTRY[key]
        assert len(spec.sha256) == 64
        assert spec.url.startswith("https://")


@requires_x2
def test_swinir_x2_upscales_rgb_image():
    img = Image.new("RGB", (32, 32), color=(90, 120, 200))
    out = Upscaler(backbone="swinir-x2", device="cpu").upscale(img, scale=2.0)
    assert out.size == (64, 64)
    assert out.mode == "RGB"


@requires_x4
def test_swinir_x4_upscales_rgb_image():
    img = Image.new("RGB", (32, 32), color=(90, 120, 200))
    out = Upscaler(backbone="swinir-x4", device="cpu").upscale(img, scale=4.0)
    assert out.size == (128, 128)
    assert out.mode == "RGB"


@requires_x2
def test_swinir_rejects_unsupported_scale():
    img = Image.new("RGB", (32, 32))
    up = Upscaler(backbone="swinir-x2", device="cpu")
    with pytest.raises(ValueError, match="supports scales"):
        up.upscale(img, scale=3.0)


@requires_x4
def test_swinir_preserves_alpha_via_bicubic_alpha_path():
    img = Image.new("RGBA", (32, 32), color=(200, 50, 100, 128))
    out = Upscaler(backbone="swinir-x4", device="cpu").upscale(img, scale=4.0)
    assert out.size == (128, 128)
    assert out.mode == "RGBA"
    # Alpha should stay ~constant through the bicubic-alpha path.
    alpha = out.split()[-1]
    lo, hi = alpha.getextrema()
    assert 120 <= lo <= hi <= 136
