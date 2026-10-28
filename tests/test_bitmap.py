import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from metalgrow import bitmap
from metalgrow.bitmap import coverage, to_bitmap
from metalgrow.cli import app


def _disc(size=64, mode="LA") -> Image.Image:
    """A soft-edged black disc on transparency (or on white paper when opaque)."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    cov = np.clip(size / 4 - r + 0.5, 0, 1)
    a = np.round(cov * 255).astype(np.uint8)
    if mode == "LA":
        return Image.fromarray(np.stack([np.zeros_like(a), a], axis=-1))
    return Image.fromarray(255 - a)  # opaque "L" scan


def test_master_is_one_bit_at_target_ppi_and_same_physical_size():
    src = _disc()
    out = to_bitmap(src, ppi=1200, src_dpi=300)
    assert out.mode == "1"
    assert out.size == (256, 256)
    px = np.asarray(out)
    assert not px[128, 128]  # centre is ink (black)
    assert px[0, 0]  # corner is paper (white)


def test_alpha_and_opaque_scan_give_the_same_master():
    a = to_bitmap(_disc(mode="LA"), ppi=600, src_dpi=150)
    b = to_bitmap(_disc(mode="L"), ppi=600, src_dpi=150)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_coverage_inverts_luminance_only_for_opaque():
    assert coverage(Image.new("L", (2, 2), 0)).getextrema() == (255, 255)
    assert coverage(Image.new("LA", (2, 2), (0, 0))).getextrema() == (0, 0)


def test_threshold_moves_the_edge():
    src = _disc()
    lo = (~np.asarray(to_bitmap(src, ppi=300, threshold=32, src_dpi=300))).sum()
    hi = (~np.asarray(to_bitmap(src, ppi=300, threshold=224, src_dpi=300))).sum()
    assert lo > hi


@pytest.mark.parametrize(("ppi", "src_dpi"), [(1200, 300), (1000, 300), (1200, 72)])
def test_strips_match_a_single_pass_resize(monkeypatch, ppi, src_dpi):
    cov = coverage(_disc(96))
    k = ppi / src_dpi
    size = (round(96 * k), round(96 * k))
    whole = np.asarray(cov.resize(size, Image.LANCZOS))
    monkeypatch.setattr(bitmap, "_STRIP", 37)  # odd size: many strip seams
    stripped = np.zeros_like(whole)
    for y0, rows in bitmap._strips(cov, size):
        stripped[y0 : y0 + rows.height] = np.asarray(rows)
    assert np.array_equal(whole, stripped)


def test_no_dpi_tag_keeps_pixel_size():
    assert to_bitmap(_disc(), ppi=1200).size == (64, 64)


def test_rejects_bad_threshold():
    with pytest.raises(ValueError):
        to_bitmap(_disc(), threshold=0)


def test_cli_bitmap_writes_group4_tiff(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    _disc().save(src_dir / "disc.png", dpi=(300, 300))
    result = CliRunner().invoke(app, ["bitmap", str(src_dir), str(tmp_path / "out")])
    assert result.exit_code == 0, result.stdout
    out = Image.open(tmp_path / "out" / "disc.tif")
    assert out.mode == "1"
    assert out.size == (256, 256)
    assert out.info.get("compression") == "group4"
    assert round(out.info["dpi"][0]) == 1200
