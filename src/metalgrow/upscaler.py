from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from metalgrow.backbones import get_backbone
from metalgrow.device import get_device


class Upscaler:
    """Upscales images using a pluggable backbone on the selected torch device.

    The ``bicubic`` backbone is always available and requires no weights, so the
    pipeline is runnable end-to-end in CI. Learned backbones (Real-ESRGAN,
    SwinIR, …) are selected by name via the registry in
    :mod:`metalgrow.backbones`.
    """

    def __init__(
        self,
        backbone: str = "bicubic",
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = get_device(device)
        self.dtype = dtype
        self.backbone = get_backbone(backbone, self.device, dtype)

    def upscale(self, image: Image.Image, scale: float = 2.0) -> Image.Image:
        tensor = pil_to_tensor(image).to(self.device, self.dtype).unsqueeze(0) / 255.0
        _, c, _, _ = tensor.shape

        if c == 4 and self.backbone.input_channels == 3:
            # Alpha is out-of-distribution for an RGB-trained SR model. Run
            # the backbone on the RGB plane and bicubic-upscale the alpha
            # channel, then recombine. Cheap and preserves transparency.
            rgb = self.backbone.upscale(tensor[:, :3], scale).clamp(0.0, 1.0)
            alpha = F.interpolate(
                tensor[:, 3:4], size=rgb.shape[-2:], mode="bicubic", align_corners=False
            ).clamp(0.0, 1.0)
            out = torch.cat([rgb, alpha], dim=1)
        else:
            out = self.backbone.upscale(tensor, scale).clamp(0.0, 1.0)

        mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(out.shape[1])
        return to_pil_image(out.squeeze(0).float().cpu(), mode=mode)

    def upscale_file(self, src: Path, dst: Path, scale: float = 2.0) -> Path:
        image = Image.open(src)
        has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
        image = image.convert("RGBA" if has_alpha else "RGB")
        result = self.upscale(image, scale=scale)
        dst.parent.mkdir(parents=True, exist_ok=True)
        result.save(dst)
        return dst
