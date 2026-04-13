from pathlib import Path

import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from metalgrow.device import get_device


class Upscaler:
    """Upscales images on the selected torch device.

    The current implementation uses high-quality bicubic resampling as a
    baseline so the pipeline is runnable end-to-end on MPS. A learned
    super-resolution backbone (e.g. Real-ESRGAN / SwinIR) will plug in here.
    """

    def __init__(self, device: str = "auto"):
        self.device = get_device(device)

    def upscale(self, image: Image.Image, scale: float = 2.0) -> Image.Image:
        if scale <= 1.0:
            raise ValueError("scale must be > 1.0")

        tensor = pil_to_tensor(image).float().unsqueeze(0).to(self.device) / 255.0
        _, _, h, w = tensor.shape
        out = F.interpolate(
            tensor,
            size=(int(h * scale), int(w * scale)),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)

        return to_pil_image(out.squeeze(0).cpu())

    def upscale_file(self, src: Path, dst: Path, scale: float = 2.0) -> Path:
        image = Image.open(src).convert("RGB")
        result = self.upscale(image, scale=scale)
        dst.parent.mkdir(parents=True, exist_ok=True)
        result.save(dst)
        return dst
