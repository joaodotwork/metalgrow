from PIL import Image

from metalgrow import Upscaler


def test_upscale_doubles_dimensions():
    img = Image.new("RGB", (32, 24), color=(128, 64, 200))
    out = Upscaler(device="cpu").upscale(img, scale=2.0)
    assert out.size == (64, 48)
