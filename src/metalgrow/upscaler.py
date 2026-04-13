from collections.abc import Callable
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

    def upscale(
        self,
        image: Image.Image,
        scale: float = 2.0,
        tile: int | None = None,
        tile_pad: int | None = None,
    ) -> Image.Image:
        tensor = pil_to_tensor(image).to(self.device, self.dtype).unsqueeze(0) / 255.0
        _, c, _, _ = tensor.shape

        if c == 4 and self.backbone.input_channels == 3:
            # Alpha is out-of-distribution for an RGB-trained SR model. Run
            # the backbone on the RGB plane and bicubic-upscale the alpha
            # channel, then recombine. Cheap and preserves transparency.
            rgb = self._run_backbone(tensor[:, :3], scale, tile, tile_pad).clamp(0.0, 1.0)
            alpha = F.interpolate(
                tensor[:, 3:4], size=rgb.shape[-2:], mode="bicubic", align_corners=False
            ).clamp(0.0, 1.0)
            out = torch.cat([rgb, alpha], dim=1)
        else:
            out = self._run_backbone(tensor, scale, tile, tile_pad).clamp(0.0, 1.0)

        mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(out.shape[1])
        return to_pil_image(out.squeeze(0).float().cpu(), mode=mode)

    def upscale_file(
        self,
        src: Path,
        dst: Path,
        scale: float = 2.0,
        tile: int | None = None,
        tile_pad: int | None = None,
    ) -> Path:
        image = Image.open(src)
        has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
        image = image.convert("RGBA" if has_alpha else "RGB")
        result = self.upscale(image, scale=scale, tile=tile, tile_pad=tile_pad)
        dst.parent.mkdir(parents=True, exist_ok=True)
        result.save(dst)
        return dst

    def _run_backbone(
        self,
        tensor: torch.Tensor,
        scale: float,
        tile: int | None,
        tile_pad: int | None,
    ) -> torch.Tensor:
        if tile is None:
            tile = self.backbone.default_tile
        if tile_pad is None:
            tile_pad = self.backbone.default_tile_pad

        _, _, h, w = tensor.shape
        if not tile or tile <= 0 or (h <= tile and w <= tile):
            return self.backbone.upscale(tensor, scale)

        return _tiled_forward(
            lambda x: self.backbone.upscale(x, scale),
            tensor,
            scale=scale,
            tile=int(tile),
            pad=int(tile_pad),
        )


def _tiled_forward(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    scale: float,
    tile: int,
    pad: int,
) -> torch.Tensor:
    """Run ``fn`` over tiled crops of ``x`` and blend with a feathered overlap.

    Each tile is padded by ``pad`` pixels of real image context on every edge
    that isn't at the image boundary, giving the backbone enough receptive
    field to produce seam-free interior pixels. Overlapping regions are
    combined with a linear feather weight so any residual edge mismatch fades
    instead of appearing as a hard seam.
    """
    if tile <= 0:
        raise ValueError("tile must be positive")

    n, c, h, w = x.shape
    out_h = int(h * scale)
    out_w = int(w * scale)
    out_dtype = x.dtype
    out_device = x.device

    accum: torch.Tensor | None = None
    norm: torch.Tensor | None = None

    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            py0, py1 = max(0, y0 - pad), min(h, y1 + pad)
            px0, px1 = max(0, x0 - pad), min(w, x1 + pad)

            patch_out = fn(x[:, :, py0:py1, px0:px1])

            if accum is None:
                accum = torch.zeros(
                    n, patch_out.shape[1], out_h, out_w, dtype=out_dtype, device=out_device
                )
                norm = torch.zeros(1, 1, out_h, out_w, dtype=out_dtype, device=out_device)

            sy = patch_out.shape[-2] / (py1 - py0)
            sx = patch_out.shape[-1] / (px1 - px0)
            oy0, oy1 = round(py0 * sy), round(py1 * sy)
            ox0, ox1 = round(px0 * sx), round(px1 * sx)
            ph = oy1 - oy0
            pw = ox1 - ox0
            if patch_out.shape[-2:] != (ph, pw):
                # Fall back to the patch's actual size if the backbone didn't
                # scale integrally (e.g. non-integer scale factors).
                ph, pw = patch_out.shape[-2:]
                oy1 = oy0 + ph
                ox1 = ox0 + pw

            left_pad = round((y0 - py0) * sy)
            right_pad = round((py1 - y1) * sy)
            top_pad = round((x0 - px0) * sx)
            bot_pad = round((px1 - x1) * sx)

            wy = _feather(ph, left_pad, right_pad, out_dtype, out_device)
            wx = _feather(pw, top_pad, bot_pad, out_dtype, out_device)
            weight = wy.view(-1, 1) * wx.view(1, -1)

            accum[:, :, oy0:oy1, ox0:ox1] += patch_out * weight
            norm[:, :, oy0:oy1, ox0:ox1] += weight

    assert accum is not None and norm is not None
    return accum / norm.clamp_min(1e-8)


def _feather(length: int, left: int, right: int, dtype: torch.dtype, device: torch.device):
    w = torch.ones(length, dtype=dtype, device=device)
    if left > 0:
        ramp = torch.linspace(0.0, 1.0, left + 2, dtype=dtype, device=device)[1:-1]
        w[:left] = ramp
    if right > 0:
        ramp = torch.linspace(1.0, 0.0, right + 2, dtype=dtype, device=device)[1:-1]
        w[length - right :] = ramp
    return w
