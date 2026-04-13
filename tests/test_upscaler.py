from PIL import Image

from metalgrow import Upscaler


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
