"""Real-ESRGAN integration tests.

These tests require the Real-ESRGAN weight files to be available in the
cache directory. They are skipped in CI (no weights) and run locally once
the user has fetched weights via the CLI at least once.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from PIL import Image

from metalgrow import Upscaler, weights


def _weights_available(backbone: str) -> bool:
    spec = weights.REGISTRY[backbone]
    path = weights.cache_dir() / spec.name
    return path.exists()


requires_x4 = pytest.mark.skipif(
    not _weights_available("realesrgan-x4"),
    reason="Real-ESRGAN x4 weights not cached; run `metalgrow upscale ... -b realesrgan-x4` once",
)
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this host",
)


@requires_x4
def test_realesrgan_x4_upscales_rgb_image():
    img = Image.new("RGB", (16, 16), color=(90, 120, 200))
    out = Upscaler(backbone="realesrgan-x4", device="cpu").upscale(img, scale=4.0)
    assert out.size == (64, 64)
    assert out.mode == "RGB"


@requires_x4
def test_realesrgan_x4_preserves_alpha_via_bicubic_alpha_path():
    img = Image.new("RGBA", (16, 16), color=(200, 50, 100, 128))
    out = Upscaler(backbone="realesrgan-x4", device="cpu").upscale(img, scale=4.0)
    assert out.size == (64, 64)
    assert out.mode == "RGBA"
    # Alpha should stay ~constant (bicubic on a flat channel).
    alpha = out.split()[-1]
    lo, hi = alpha.getextrema()
    assert 120 <= lo <= hi <= 136


@requires_x4
def test_realesrgan_rejects_unsupported_scale():
    img = Image.new("RGB", (8, 8))
    up = Upscaler(backbone="realesrgan-x4", device="cpu")
    with pytest.raises(ValueError, match="supports scales"):
        up.upscale(img, scale=3.0)


@requires_x4
@requires_mps
@pytest.mark.skipif(
    os.environ.get("METALGROW_SKIP_MPS_PARITY") == "1",
    reason="MPS parity test explicitly skipped",
)
def test_realesrgan_mps_matches_cpu_within_tolerance():
    img = Image.new("RGB", (32, 32), color=(50, 160, 90))

    cpu_out = Upscaler(backbone="realesrgan-x4", device="cpu").upscale(img, scale=4.0)
    mps_out = Upscaler(backbone="realesrgan-x4", device="mps").upscale(img, scale=4.0)

    cpu_t = torch.from_numpy(np.array(cpu_out)).float()
    mps_t = torch.from_numpy(np.array(mps_out)).float()

    # Accept small numerical drift between MPS and CPU — we care that the
    # backend is functional and roughly correct, not bit-identical.
    diff = (cpu_t - mps_t).abs()
    assert diff.mean().item() < 2.0, f"mean pixel drift {diff.mean().item():.2f} > 2"
    assert diff.max().item() < 32.0, f"max pixel drift {diff.max().item():.2f} > 32"
