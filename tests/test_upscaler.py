from PIL import Image

from metalgrow import Upscaler


def test_select_native_scale_returns_fixed_factor_for_learned_backbone():
    # A weights-bound backbone always runs at its native factor, whatever the
    # requested target — the resample to target happens afterwards.
    up = Upscaler(backbone="realesrgan-x2", device="cpu")
    assert up.backbone.native_scale == 2.0
    assert up._select_native_scale(1.37) == 2.0
    assert up._select_native_scale(2.0) == 2.0
    assert up._select_native_scale(5.0) == 2.0


def test_bicubic_accepts_any_scale_natively():
    up = Upscaler(device="cpu")  # bicubic renders arbitrary scales directly
    assert up.backbone.native_scale is None
    assert up._select_native_scale(1.37) == 1.37


def test_learned_backbone_fractional_scale_resamples_to_target():
    # realesrgan-x2 runs at native 2x, then Lanczos-resamples down to 1.5x.
    img = Image.new("RGB", (20, 20), color=(120, 130, 140))
    out = Upscaler(backbone="realesrgan-x2", device="cpu").upscale(img, scale=1.5)
    assert out.size == (30, 30)


def test_upscale_doubles_dimensions():
    img = Image.new("RGB", (32, 24), color=(128, 64, 200))
    out = Upscaler(device="cpu").upscale(img, scale=2.0)
    assert out.size == (64, 48)


def test_upscale_preserves_alpha_in_memory():
    img = Image.new("RGBA", (16, 16), color=(200, 100, 50, 0))
    out = Upscaler(device="cpu").upscale(img, scale=2.0)
    assert out.mode == "RGBA"
    assert out.size == (32, 32)
    alpha = out.split()[-1]
    assert alpha.getextrema() == (0, 0)


def test_upscale_file_preserves_transparent_png(tmp_path):
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    # Half opaque red, half fully transparent — exercises the alpha path.
    img = Image.new("RGBA", (8, 8), color=(255, 0, 0, 255))
    for y in range(8):
        for x in range(4, 8):
            img.putpixel((x, y), (0, 0, 0, 0))
    img.save(src)

    Upscaler(device="cpu").upscale_file(src, dst, scale=2.0)

    out = Image.open(dst)
    assert out.mode == "RGBA"
    assert out.size == (16, 16)
    assert out.getpixel((1, 1))[3] == 255
    assert out.getpixel((14, 14))[3] == 0


def test_upscale_file_opaque_stays_rgb(tmp_path):
    src = tmp_path / "in.jpg"
    dst = tmp_path / "out.png"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(src)

    Upscaler(device="cpu").upscale_file(src, dst, scale=2.0)

    out = Image.open(dst)
    assert out.mode == "RGB"
    assert out.size == (32, 32)


def test_upscale_file_default_drops_grayscale_and_dpi(tmp_path):
    # Without the flag, the SR pipeline returns RGB and no DPI — the behaviour
    # the preserve flag exists to override.
    src = tmp_path / "in.tiff"
    dst = tmp_path / "out.tiff"
    Image.new("L", (16, 16), color=128).save(src, dpi=(144, 144))

    Upscaler(device="cpu").upscale_file(src, dst, scale=2.0)

    out = Image.open(dst)
    assert out.mode == "RGB"  # grayscale is lost
    # the source's real 144 DPI is not carried over (PIL stamps a bogus default)
    assert out.info.get("dpi") not in ((144.0, 144.0), (288.0, 288.0))


def test_upscale_file_preserve_metadata_keeps_gray_icc_dpi(tmp_path):
    src = tmp_path / "in.tiff"
    dst = tmp_path / "out.tiff"
    icc = b"fake-icc-profile-bytes"
    Image.new("L", (16, 16), color=128).save(src, dpi=(144, 144), icc_profile=icc)

    Upscaler(device="cpu").upscale_file(src, dst, scale=2.0, preserve_metadata=True)

    out = Image.open(dst)
    assert out.mode == "L"  # grayscale survives instead of becoming RGB
    assert out.size == (32, 32)
    assert out.info.get("icc_profile") == icc
    # 2x more pixels at the same physical size -> DPI doubles, size constant.
    dpi = out.info.get("dpi")
    assert dpi is not None
    assert round(dpi[0]) == 288 and round(dpi[1]) == 288
