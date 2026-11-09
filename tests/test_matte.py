"""metalgrow.matte on small synthetic renders: CPU-only, no weights.

Each rule has a test that fails when the rule is mutated (see the per-test
"Rule:" line); the mutation run is recorded in the PR for #37.
"""

import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from metalgrow import matte as m
from metalgrow.cli import app

pytest.importorskip("scipy")

PAPER = np.array([240.0, 238.0, 232.0])
STROKE = (60, 60, 60)  # neutral linework
SEPIA = (150, 100, 50)  # warm hatching: strong, but not neutral
DRINK = (180, 40, 40)
HIGHLIGHT = (250, 250, 250)  # lighter than the paper: unmixed alpha 0
SHADOW = (200, 185, 160)  # unmixed alpha ~0.31, below T
HAZE_PX = PAPER - 3  # unmixed alpha ~0.013, below HAZE


def _canvas(size):
    rgb = np.empty((size, size, 3))
    rgb[:] = PAPER
    yy, xx = np.mgrid[:size, :size]
    return rgb, yy, xx


def _ring(r, cy, cx, radius, width):
    return np.abs(r - radius) <= width / 2


def _scene():
    """Glass (outlined disc: highlight over drink), a cast shadow, a speck, paper haze."""
    rgb, yy, xx = _canvas(400)
    r = np.hypot(yy - 180, xx - 200)
    rgb[r < 77] = HIGHLIGHT
    rgb[(r < 77) & (yy > 180)] = DRINK
    rgb[_ring(r, 180, 200, 80, 6)] = STROKE
    rgb[300:340, 220:380] = SHADOW
    rs = np.hypot(yy - 340, xx - 60)
    rgb[_ring(rs, 340, 60, 8, 2)] = STROKE  # hatching speck enclosing paper
    rgb[10:30, 300:390] = HAZE_PX
    return rgb


def _open_glass(colour):
    """An outline with a 4 px gap at the top and a highlit (paper-light) inside."""
    rgb, yy, xx = _canvas(500)
    r = np.hypot(yy - 250, xx - 250)
    rgb[r < 118] = HIGHLIGHT
    ring = _ring(r, 250, 250, 120, 5)
    ring &= ~((np.abs(xx - 249.5) < 2) & (yy < 250))
    rgb[ring] = colour
    return rgb


def _seal_scene():
    """Two drink discs (glass) and, between them, a thin small outline (under
    MIN_AREA) that is neutral on top with a 4 px gap and sepia below, around a
    highlight. Below the glass rows, a small closed neutral ring."""
    rgb, yy, xx = _canvas(500)
    for cx in (70, 430):
        rgb[np.hypot(yy - 250, xx - cx) < 55] = DRINK
    r = np.hypot(yy - 250, xx - 250)
    rgb[r < 45] = HIGHLIGHT
    rgb[_ring(r, 250, 250, 45, 2) & (yy >= 250)] = SEPIA
    rgb[_ring(r, 250, 250, 45, 4) & (yy < 250) & ~(np.abs(xx - 249.5) < 2)] = STROKE
    rgb[_ring(np.hypot(yy - 450, xx - 250), 450, 250, 12, 2)] = STROKE
    return rgb


def _alpha(rgb, seal=False):
    col, alpha, _, _ = m.matte(rgb, seal)
    return np.asarray(m.to_rgba(col, alpha)), alpha


def test_glass_is_opaque_and_kept_as_drawn():
    # Rule: the glass core is opaque with the render's own colour.
    rgba, _ = _alpha(_scene())
    for y, x, colour in [(150, 200, HIGHLIGHT), (220, 200, DRINK), (180, 200, HIGHLIGHT)]:
        assert tuple(rgba[y, x]) == (*colour, 255)


def test_glass_edge_ring_uses_the_unmixed_matte():
    # Rule: EDGE px at the glass boundary keep the unmixed alpha, not 255.
    rgba, _ = _alpha(_scene())
    # The stroke right of the disc spans columns 277..283; the outer EDGE = 3
    # of them are the ring, the rest is core.
    a = ((PAPER - np.array(STROKE)) / PAPER).max()
    unmixed = round(m.clear_haze(a) * 255)
    assert unmixed < 255
    assert rgba[180, 281:284, 3].tolist() == [unmixed] * 3
    assert rgba[180, 280, 3] == 255


def test_shadow_is_translucent_and_recomposites_over_paper():
    # Rule: outside the glass, colour-to-alpha against the paper.
    rgba, _ = _alpha(_scene())
    px = rgba[320, 300].astype(float)
    assert 0 < px[3] < 255
    a = px[3] / 255
    over = px[:3] * a + PAPER * (1 - a)
    assert np.abs(over - SHADOW).max() <= 5  # haze rescale moves it by at most ~HAZE


def test_small_enclosed_speck_stays_outside_the_glass():
    # Rule: enclosed areas under MIN_AREA are shadow hatching, not glass.
    rgba, _ = _alpha(_scene())
    assert rgba[340, 60, 3] == 0


def test_paper_haze_is_cleared():
    # Rule: alpha below HAZE is cleared.
    rgba, alpha = _alpha(_scene())
    assert (rgba[10:30, 300:390, 3] == 0).all()
    assert (alpha[:5] == 0).all()


def test_clear_haze_is_continuous():
    a = np.array([0.0, m.HAZE, 0.5, 1.0])
    assert m.clear_haze(a).tolist() == [0.0, 0.0, pytest.approx((0.5 - m.HAZE) / (1 - m.HAZE)), 1.0]


def test_seal_fills_a_highlight_hole_a_plain_run_leaves_open():
    # Rule: --seal closes small gaps in neutral outlines.
    plain, _ = _alpha(_open_glass(STROKE))
    sealed, _ = _alpha(_open_glass(STROKE), seal=True)
    assert plain[250, 250, 3] == 0  # paper floods in through the gap
    assert tuple(sealed[250, 250]) == (*HIGHLIGHT, 255)
    assert (plain[:, :, 3] <= sealed[:, :, 3]).all()


def test_sepia_hatching_does_not_seal():
    # Rule: strokes above SEAL_CHROMA (warm shadow hatching) never seal.
    sealed, _ = _alpha(_open_glass(SEPIA), seal=True)
    assert sealed[250, 250, 3] == 0


def test_seal_keeps_holes_walled_by_any_strong_stroke():
    # Rule: once the neutral gaps close, enclosed areas of at least SEAL_HOLE
    # are glass even where the wall is a non-neutral stroke.
    plain, _ = _alpha(_seal_scene())
    sealed, _ = _alpha(_seal_scene(), seal=True)
    assert plain[250, 250, 3] == 0
    assert sealed[250, 250, 3] == 255


def test_seal_stays_within_the_glass_rows():
    # Rule: sealed areas only count within each row's span of the glass.
    sealed, _ = _alpha(_seal_scene(), seal=True)
    assert sealed[450, 250, 3] == 0


def test_seal_is_a_no_op_on_closed_glass():
    plain, _ = _alpha(_scene())
    sealed, _ = _alpha(_scene(), seal=True)
    assert np.array_equal(plain, sealed)


def test_paper_colour_is_the_median_of_all_four_borders():
    # Rule: paper is sampled from all four edges, so one odd edge is outvoted.
    rgb = _scene()
    assert m.paper_colour(rgb).tolist() == PAPER.tolist()
    rgb[: m.BORDER] = 200  # a dark top edge (e.g. a scanner shadow)
    assert m.paper_colour(rgb).tolist() == PAPER.tolist()


def test_module_is_torch_free():
    code = "import sys, metalgrow.matte; assert 'torch' not in sys.modules, 'torch'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_scipy_raises_with_install_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.ndimage", None)
    with pytest.raises(m.MatteDependencyError, match=r"pip install 'metalgrow\[matte\]'"):
        m.matte(_scene())

    src = tmp_path / "in.png"
    Image.fromarray(_scene().astype(np.uint8)).save(src)
    result = CliRunner().invoke(app, ["matte", str(src), str(tmp_path / "out.png")])
    assert result.exit_code == 1
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert lines == ["error: metalgrow matte needs scipy: pip install 'metalgrow[matte]'"]
    assert not (tmp_path / "out.png").exists()


# --- CLI ------------------------------------------------------------------


def _write(path, rgb):
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def test_cli_single_file_writes_rgba_png_with_dpi(tmp_path):
    _write(tmp_path / "a.png", _scene())
    out = tmp_path / "out" / "a.png"
    result = CliRunner().invoke(app, ["matte", str(tmp_path / "a.png"), str(out), "--dpi", "600"])
    assert result.exit_code == 0, result.output
    with Image.open(out) as im:
        assert im.mode == "RGBA"
        assert im.info["dpi"] == pytest.approx((600, 600), abs=0.01)
        expected = m.matte_image(Image.open(tmp_path / "a.png"))[0]
        assert np.array_equal(np.asarray(im), np.asarray(expected))
    assert "glass" in result.output


def test_cli_seal_flag_reaches_the_matte(tmp_path):
    _write(tmp_path / "g.png", _open_glass(STROKE))
    runner = CliRunner()
    runner.invoke(app, ["matte", str(tmp_path / "g.png"), str(tmp_path / "plain.png")])
    runner.invoke(app, ["matte", str(tmp_path / "g.png"), str(tmp_path / "seal.png"), "--seal"])
    assert np.asarray(Image.open(tmp_path / "plain.png"))[250, 250, 3] == 0
    assert np.asarray(Image.open(tmp_path / "seal.png"))[250, 250, 3] == 255


def test_cli_rejects_non_png_output(tmp_path):
    _write(tmp_path / "a.png", _scene())
    result = CliRunner().invoke(app, ["matte", str(tmp_path / "a.png"), str(tmp_path / "a.tif")])
    assert result.exit_code != 0
    assert not (tmp_path / "a.tif").exists()


def _batch_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    small = _scene()[::4, ::4]
    for name in ("x_0001.jpg", "y_0002.png"):
        _write(src / name, small)
    return src


def test_cli_batch_names_outputs_stem_png(tmp_path):
    src = _batch_dir(tmp_path)
    result = CliRunner().invoke(app, ["matte", str(src), str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["x_0001.png", "y_0002.png"]


def test_cli_batch_names_map_renames(tmp_path):
    src = _batch_dir(tmp_path)
    names = tmp_path / "names.csv"
    names.write_text("# source,output\nx_0001.jpg,01-aviation.png\n\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["matte", str(src), str(tmp_path / "out"), "--names", str(names)]
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "01-aviation.png",
        "y_0002.png",
    ]


@pytest.mark.parametrize(
    "csv, message",
    [
        ("x_0001.jpg,a.png\nx_0001.jpg,b.png\n", "listed twice"),
        ("x_0001.jpg,a.png\ny_0002.png,a.png\n", "both be written to a.png"),
        ("x_0001.jpg,y_0002.png\n", "both be written to y_0002.png"),
        ("nope.png,a.png\n", "not in the input: nope.png"),
        ("x_0001.jpg,../a.png\n", "plain file name"),
        ("x_0001.jpg,a.tif\n", "must end in .png"),
        ("x_0001.jpg\n", "expected 2 columns"),
    ],
)
def test_cli_bad_names_map_fails_before_writing(tmp_path, csv, message):
    src = _batch_dir(tmp_path)
    names = tmp_path / "names.csv"
    names.write_text(csv, encoding="utf-8")
    result = CliRunner().invoke(
        app, ["matte", str(src), str(tmp_path / "out"), "--names", str(names)]
    )
    assert result.exit_code != 0
    assert message in " ".join(result.output.replace("│", " ").split())  # rich box wraps
    assert not (tmp_path / "out").exists()


def test_cli_names_rejected_for_single_file(tmp_path):
    _write(tmp_path / "a.png", _scene())
    names = tmp_path / "names.csv"
    names.write_text("a.png,b.png\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["matte", str(tmp_path / "a.png"), str(tmp_path / "o.png"), "--names", str(names)]
    )
    assert result.exit_code != 0
