"""Tiled inference tests.

The Upscaler-level tiler wraps any backbone's forward call with overlap
padding and feathered blending. These tests exercise it via the built-in
bicubic backbone plus a purpose-built nearest-neighbour backbone that is
translation-equivariant at integer scales — that equivariance is what lets
us assert bit-exact equality between tiled and non-tiled output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from metalgrow import Upscaler
from metalgrow.backbones.base import Backbone
from metalgrow.backbones.registry import _REGISTRY, register


class _NearestBackbone(Backbone):
    """Integer-scale nearest-neighbour upsampler — translation-equivariant.

    That property means the same pixel produces the same output no matter
    which tile it falls in, so tiled and non-tiled runs must match exactly.
    """

    name = "nearest-test"
    supported_scales = (2.0, 4.0)

    def upscale(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        self.validate_scale(scale)
        return F.interpolate(tensor, scale_factor=int(scale), mode="nearest")


@pytest.fixture(autouse=True)
def _register_nearest_backbone():
    if "nearest-test" in _REGISTRY:
        yield
        return
    register("nearest-test", lambda device, dtype: _NearestBackbone(device, dtype))
    yield
    _REGISTRY.pop("nearest-test", None)


def _random_image(h: int, w: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_tiled_forward_is_exact_for_translation_equivariant_fn():
    # Bypass PIL round-tripping (which truncates float→uint8 with a 1-lsb
    # error that masks the property we want to verify) and hit the tiler
    # directly with a nearest-neighbour forward fn.
    from metalgrow.upscaler import _tiled_forward

    torch.manual_seed(0)
    x = torch.rand(1, 3, 96, 128)

    def fn(t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t, scale_factor=2, mode="nearest")

    untiled = fn(x)
    tiled = _tiled_forward(fn, x, scale=2.0, tile=32, pad=8)

    assert untiled.shape == tiled.shape == (1, 3, 192, 256)
    # Feathered averaging of equal values is exact in float32, modulo ULP
    # noise from the divide. A ~1e-6 tolerance catches correctness
    # regressions while ignoring the divide's rounding.
    assert torch.allclose(untiled, tiled, atol=1e-6)


def test_tile_zero_disables_tiling():
    # tile=0 must skip the tiler entirely — we verify by patching the tiler
    # and confirming it isn't called.
    img = _random_image(64, 64, seed=1)
    up = Upscaler(backbone="bicubic", device="cpu")

    from metalgrow import upscaler as up_mod

    called = {"n": 0}
    original = up_mod._tiled_forward

    def spy(*args, **kwargs):
        called["n"] += 1
        return original(*args, **kwargs)

    up_mod._tiled_forward = spy
    try:
        up.upscale(img, scale=2.0, tile=0)
    finally:
        up_mod._tiled_forward = original

    assert called["n"] == 0


def test_tile_larger_than_image_skips_tiler():
    img = _random_image(32, 32, seed=2)
    up = Upscaler(backbone="bicubic", device="cpu")

    from metalgrow import upscaler as up_mod

    called = {"n": 0}
    original = up_mod._tiled_forward

    def spy(*args, **kwargs):
        called["n"] += 1
        return original(*args, **kwargs)

    up_mod._tiled_forward = spy
    try:
        up.upscale(img, scale=2.0, tile=256)
    finally:
        up_mod._tiled_forward = original

    assert called["n"] == 0


def test_tiled_4k_synthetic_cpu_runs():
    # The point of tiling is to upscale large images without OOM. A 4K
    # input with tile=512 exercises the full tile grid (8×4 tiles + pad)
    # on CPU, which is where CI runs.
    img = _random_image(2160, 3840, seed=7)
    up = Upscaler(backbone="bicubic", device="cpu")

    out = up.upscale(img, scale=2.0, tile=512, tile_pad=16)
    assert out.size == (7680, 4320)
    assert out.mode == "RGB"


def test_tiled_bicubic_matches_untiled_within_tolerance():
    # Bicubic isn't translation-equivariant at the sub-pixel level but with
    # generous pad, interior pixels should be very close. Mean absolute
    # error over all channels must stay tight.
    img = _random_image(128, 128, seed=9)
    up = Upscaler(backbone="bicubic", device="cpu")

    untiled = np.array(up.upscale(img, scale=2.0, tile=0)).astype(np.int16)
    tiled = np.array(up.upscale(img, scale=2.0, tile=32, tile_pad=16)).astype(np.int16)

    mae = np.abs(untiled - tiled).mean()
    assert mae < 1.0, f"mean abs diff {mae:.3f} too large — seams leaking through"
