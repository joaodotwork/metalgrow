import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from metalgrow import Upscaler
from metalgrow.backbones import Backbone, register
from metalgrow.ink import InkMatte, is_single_ink
from metalgrow.metadata import reapply

SEEN: list[torch.Tensor] = []


class _RecordingBackbone(Backbone):
    """Nearest-neighbour 2x that records what the Upscaler fed it."""

    name = "test-recording"
    input_channels = 3
    native_scale = 2.0

    def upscale(self, tensor, scale):
        SEEN.append(tensor.detach().cpu().clone())
        return F.interpolate(tensor, scale_factor=2, mode="nearest")


register("test-recording", lambda device, dtype: _RecordingBackbone(device, dtype))


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()


def _ink_art(size=(24, 16), ink=(0, 0, 0)) -> Image.Image:
    """Ink-on-transparent: flat RGB, all detail in alpha (a ramp + a solid block)."""
    w, h = size
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[:, : w // 2] = np.linspace(0, 255, w // 2, dtype=np.uint8)
    alpha[4:8, w // 2 :] = 255
    rgb = np.broadcast_to(np.array(ink, dtype=np.uint8), (h, w, 3))
    return Image.fromarray(np.dstack([rgb, alpha]))


def test_ink_mode_feeds_backbone_the_coverage_not_the_flat_rgb():
    art = _ink_art()
    Upscaler(backbone="test-recording", device="cpu", alpha_mode="ink").upscale(art, 2.0)

    (fed,) = SEEN
    assert fed.shape[1] == 3
    # The default path would hand the backbone the flat black plane; ink mode
    # must hand it 1 - alpha, which carries the drawing.
    alpha = np.asarray(art)[..., 3] / 255.0
    np.testing.assert_allclose(fed[0, 0].numpy(), 1.0 - alpha, atol=1e-5)
    assert fed.std() > 0.1


def test_separate_mode_is_unchanged_default():
    art = _ink_art()
    Upscaler(backbone="test-recording", device="cpu").upscale(art, 2.0)
    (fed,) = SEEN
    assert fed.std() == 0  # flat black RGB, as before


def test_ink_mode_round_trips_alpha_and_outputs_grey_ink():
    art = _ink_art()
    out = Upscaler(backbone="test-recording", device="cpu", alpha_mode="ink").upscale(art, 2.0)

    assert out.mode == "LA"  # neutral ink prints as one ink
    assert out.size == (48, 32)
    grey, alpha = np.asarray(out)[..., 0], np.asarray(out)[..., 1]
    assert (grey == 0).all()
    src_alpha = np.asarray(art)[..., 3]
    np.testing.assert_allclose(alpha[::2, ::2], src_alpha, atol=1)


def test_ink_mode_knocks_out_paper_of_opaque_scan():
    # Black ink on slightly-off-white paper, no alpha at all.
    scan = np.full((16, 16, 3), 250, dtype=np.uint8)
    scan[4:12, 4:12] = 0
    out = Upscaler(backbone="test-recording", device="cpu", alpha_mode="ink").upscale(
        Image.fromarray(scan), 2.0
    )
    alpha = np.asarray(out)[..., 1]
    assert alpha[0, 0] == 0  # paper is transparent
    assert alpha[16, 16] == 255  # ink is opaque


def test_opaque_scan_with_stray_alpha_still_finds_dark_ink():
    # An opaque scan whose alpha channel has a few non-255 values must not be
    # read as ink-on-transparent: its opaque pixels are mostly paper.
    scan = np.full((16, 16, 4), 252, dtype=np.uint8)
    scan[..., 3] = 255
    scan[4:12, 4:12, :3] = 0
    scan[0, :3, 3] = 200
    matte, _ = InkMatte.fit(Image.fromarray(scan))
    assert matte.ink_luma < 0.01
    out = Upscaler(backbone="test-recording", device="cpu", alpha_mode="ink").upscale(
        Image.fromarray(scan), 2.0
    )
    alpha = np.asarray(out)[..., 1]
    assert alpha[16, 16] == 255 and alpha[30, 30] == 0


def test_coloured_ink_keeps_its_colour():
    out = Upscaler(backbone="test-recording", device="cpu", alpha_mode="ink").upscale(
        _ink_art(ink=(120, 20, 30)), 2.0
    )
    assert out.mode == "RGBA"
    px = np.asarray(out)
    solid = px[px[..., 3] == 255]
    assert np.abs(solid[:, :3].astype(int) - [120, 20, 30]).max() <= 1


def test_auto_picks_ink_only_for_single_colour_transparent_art():
    assert is_single_ink(_ink_art())
    assert not is_single_ink(Image.new("RGB", (8, 8)))  # no transparency

    multi = np.asarray(_ink_art()).copy()
    multi[4:8, 12:, :3] = (255, 0, 0)  # a second colour on the solid block
    assert not is_single_ink(Image.fromarray(multi))

    Upscaler(backbone="test-recording", device="cpu", alpha_mode="auto").upscale(_ink_art(), 2.0)
    assert SEEN[0].std() > 0.1


def test_empty_art_stays_transparent():
    blank = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    matte, _ = InkMatte.fit(blank)
    out = matte.compose(np.ones((8, 8), dtype=np.float32))
    assert np.asarray(out)[..., -1].max() == 0


def test_rejects_unknown_alpha_mode():
    with pytest.raises(ValueError, match="alpha_mode"):
        Upscaler(device="cpu", alpha_mode="matte")


def test_rgb_profile_dropped_from_grey_result_and_dpi_override():
    rgb_icc = b"\0" * 16 + b"RGB " + b"\0" * 108
    meta = {"mode": "RGBA", "icc_profile": rgb_icc, "dpi": (72, 72), "size": (10, 10)}
    _, kwargs = reapply(Image.new("LA", (40, 40)), meta, dpi=300)
    assert "icc_profile" not in kwargs
    assert kwargs["dpi"] == (300, 300)
    _, kwargs = reapply(Image.new("RGBA", (40, 40)), meta)
    assert kwargs["icc_profile"] == rgb_icc
    assert kwargs["dpi"] == (288, 288)
