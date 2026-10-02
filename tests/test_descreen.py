import math

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from metalgrow import Upscaler
from metalgrow.cli import app
from metalgrow.descreen import descreen, descreen_tensor

SIZE = 512
PERIOD = 3.0  # px along the screen axes — a ~67 lpi screen scanned at 200 dpi


def _tone(size: int = SIZE) -> np.ndarray:
    """A smooth continuous-tone 'photo': gradient plus a soft blob, in [0.15, 0.85]."""
    y, x = np.mgrid[0:size, 0:size] / size
    blob = np.exp(-((x - 0.6) ** 2 + (y - 0.4) ** 2) / 0.03)
    return 0.15 + 0.5 * x + 0.2 * blob


def _halftone(tone: np.ndarray, period: float = PERIOD, angle: float = 45.0) -> np.ndarray:
    """Binary AM halftone of ``tone`` with a dot screen at ``angle``."""
    h, w = tone.shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    a = math.radians(angle)
    u = (x * math.cos(a) + y * math.sin(a)) / period
    v = (-x * math.sin(a) + y * math.cos(a)) / period
    screen = (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v)) / 4 + 0.5
    return (screen < tone).astype(float)  # 1 = paper, 0 = ink


def _to_image(a: np.ndarray, mode: str = "L") -> Image.Image:
    img = Image.fromarray((a * 255).round().astype(np.uint8), mode="L")
    return img.convert(mode)


def _screen_energy(img: Image.Image, period: float = PERIOD, angle: float = 45.0) -> float:
    """Spectral magnitude at the screen's fundamental frequency."""
    a = np.asarray(img.convert("L"), float)
    a = a - a.mean()
    spec = np.abs(np.fft.fft2(a))
    n = a.shape[0]
    r = math.radians(angle)
    fy, fx = math.sin(r) / period, math.cos(r) / period
    i, j = round(fy * n) % n, round(fx * n) % n
    spec = np.roll(spec, (2 - i, 2 - j), axis=(0, 1))  # bring the peak away from the wrap
    return spec[:5, :5].max()


def _text_like(size: int = SIZE, pitch: int = 31) -> np.ndarray:
    """Rows of glyph-ish vertical strokes at a regular line pitch (1-D rhythm)."""
    rng = np.random.default_rng(0)
    a = np.ones((size, size))
    for top in range(8, size - 20, pitch):
        x = 6
        while x < size - 8:
            width = int(rng.integers(2, 4))
            height = int(rng.integers(12, 18))
            a[top : top + height, x : x + width] = 0.1
            x += width + int(rng.integers(3, 7))
    return a


def test_removes_45_degree_screen():
    img = _to_image(_halftone(_tone()))
    out, report = descreen(img)
    assert report.detected
    assert report.peaks[0].period == pytest.approx(PERIOD, rel=0.05)
    assert _screen_energy(out) < _screen_energy(img) / 10


def test_descreened_result_is_closer_to_continuous_tone():
    tone = _tone()
    img = _to_image(_halftone(tone))
    out, _ = descreen(img)
    # Compare against the tone after a mild blur, which is what a perfect
    # descreen of a binary screen can recover.
    target = np.asarray(_to_image(tone), float)
    before = np.abs(np.asarray(img, float) - target).mean()
    after = np.abs(np.asarray(out, float) - target).mean()
    assert after < before / 2


def test_detects_zero_degree_screen():
    img = _to_image(_halftone(_tone(), angle=0.0))
    out, report = descreen(img)
    assert report.detected
    assert _screen_energy(out, angle=0.0) < _screen_energy(img, angle=0.0) / 10


def test_text_line_rhythm_is_not_a_screen():
    # Regular line pitch puts compact peaks on the vertical frequency axis;
    # the lattice check must reject them so type is left alone.
    img = _to_image(_text_like())
    out, report = descreen(img)
    assert not report.detected
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_smooth_image_is_untouched():
    img = _to_image(_tone())
    out, report = descreen(img)
    assert not report.detected
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_only_screened_region_changes():
    # Left half halftone photo, right half text: the text side must come back
    # unchanged beyond one analysis window from the boundary.
    size = 2 * SIZE
    a = _text_like(size)
    a[:, :SIZE] = _halftone(_tone(size))[:, :SIZE]
    img = _to_image(a)
    out, report = descreen(img)
    assert report.detected
    diff = np.abs(np.asarray(out, float) - np.asarray(img, float))
    assert diff[:, SIZE + 256 :].max() <= 1.0
    assert diff[:, : SIZE - 128].mean() > 10


def test_strength_zero_is_noop():
    img = _to_image(_halftone(_tone()))
    out, report = descreen(img, strength=0.0)
    assert report.detected
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_rgb_and_alpha_passthrough():
    rgb = _to_image(_halftone(_tone()), "RGB")
    alpha = Image.new("L", rgb.size, 77)
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    out, report = descreen(rgba)
    assert out.mode == "RGBA"
    assert report.detected
    assert np.array_equal(np.asarray(out)[..., 3], np.asarray(alpha))


def test_rejects_bad_input():
    import torch

    with pytest.raises(ValueError):
        descreen_tensor(torch.zeros(1, 1, 8, 8))
    with pytest.raises(ValueError):
        descreen_tensor(torch.zeros(1, 8, 8), strength=1.5)


def test_upscaler_descreen_is_opt_in():
    img = _to_image(_halftone(_tone(256)[:256, :256]), "RGB")
    off = Upscaler(device="cpu")
    off.upscale(img, scale=2.0)
    assert off.last_descreen is None

    on = Upscaler(device="cpu", descreen=True)
    out = on.upscale(img, scale=2.0)
    assert out.size == (512, 512)
    assert on.last_descreen is not None and on.last_descreen.detected


def test_cli_descreen_command(tmp_path):
    src = tmp_path / "scan.png"
    dst = tmp_path / "clean.png"
    img = _to_image(_halftone(_tone()))
    img.save(src, dpi=(200, 200))
    result = CliRunner().invoke(app, ["descreen", str(src), str(dst), "-p"])
    assert result.exit_code == 0, result.output
    assert "lpi" in result.output
    out = Image.open(dst)
    assert out.size == img.size
    assert out.mode == "L"  # grayscale preserved with -p
    assert out.info["dpi"] == pytest.approx((200, 200), rel=1e-4)  # PNG stores px/m


def test_cli_upscale_with_descreen(tmp_path):
    src = tmp_path / "scan.png"
    dst = tmp_path / "up.png"
    _to_image(_halftone(_tone(256)[:256, :256]), "RGB").save(src)
    result = CliRunner().invoke(app, ["upscale", str(src), str(dst), "--descreen", "-d", "cpu"])
    assert result.exit_code == 0, result.output
    assert "descreen: screen" in result.output
    assert Image.open(dst).size == (512, 512)


def test_confirmed_lattice_notches_second_order_harmonics():
    # Dark, high-coverage dots put much of their energy into the 2nd-order
    # harmonics (v1 ± v2: on the axes for a 45° screen), which can be too
    # weak to seed or grow into. Once a window's lattice is confirmed their
    # positions are known, so they're notched outright — but only there.
    import torch

    from metalgrow.descreen import _harmonic_seeds

    win = 256
    f = torch.fft.fftfreq(win)
    allowed = torch.hypot(f.view(-1, 1), f.view(1, -1)) >= 0.08
    v1, v2 = (1 / 6, 1 / 6), (1 / 6, -1 / 6)
    basis = torch.tensor([[v1, v2], [v1, v2]])
    ok = torch.tensor([True, False])
    weak = torch.zeros(2, win, win, dtype=torch.bool)  # nothing prominent at all

    seeds = _harmonic_seeds(basis, ok, weak, allowed)
    on_axis = round((v1[0] + v2[0]) * win) % win  # v1 + v2 = (1/3, 0)
    assert seeds[0, on_axis, 0]
    assert not seeds[1].any()  # unconfirmed window: untouched


def test_growth_accepts_only_the_known_screen():
    # Neighbour growth: a window is accepted at the lower threshold when it is
    # prominent at *both* predicted fundamentals of a confirmed screen, and
    # rejected when its peaks sit elsewhere (e.g. text's axis harmonics).
    import torch

    from metalgrow.descreen import _notch_region

    win = 256
    f = torch.fft.fftfreq(win)
    allowed = torch.hypot(f.view(-1, 1), f.view(1, -1)) >= 0.08
    v1, v2 = (1 / 6, 1 / 6), (1 / 6, -1 / 6)  # 45° screen, 3 px period
    known = torch.tensor([[v1, v2]])

    def bump(prom, fy, fx, value):
        for sy, sx in ((fy, fx), (-fy, -fx)):
            prom[round(sy * win) % win, round(sx * win) % win] = value

    screen = torch.zeros(win, win)
    bump(screen, *v1, 1.3)  # between the growth (1.2) and seed (1.6) thresholds
    bump(screen, *v2, 1.3)
    text = torch.zeros(win, win)
    bump(text, 0.1, 0.0, 1.3)  # line pitch on the vertical axis
    bump(text, 0.0, 0.25, 1.3)  # stroke rhythm on the horizontal axis

    prom = torch.stack([screen, text])
    _, _, seeded = _notch_region(prom, allowed, 1.6)
    assert not seeded.any()  # neither seeds on its own
    region, _, ok = _notch_region(prom, allowed, 1.2, known_bases=known)
    assert ok.tolist() == [True, False]
    assert region[0].any() and not region[1].any()
