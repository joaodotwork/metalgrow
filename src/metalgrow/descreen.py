"""Remove periodic halftone screens (moiré) from scanned print.

Scanned newspaper and magazine images carry the printer's halftone screen: a
regular dot lattice that shows up in the 2-D spectrum as compact, prominent
peaks (the screen's fundamentals plus their harmonics). SR backbones treat
those dots as detail and sharpen them into moiré, so they must go *before*
upscaling.

Print sources mix halftoned photos with type and linework, often with several
differently-screened photos on one page, so a single global notch either
misses screens or bites into text. The filter is therefore **local**:

1. The image is cut into overlapping windows (short-time Fourier transform,
   sqrt-Hann analysis + synthesis at 50 % overlap → exact reconstruction when
   nothing is changed).
2. Each window's luminance spectrum is scored by *prominence*: log-magnitude
   over a horizontal and a vertical 1-D spectral background, whichever is
   smaller. A halftone peak is compact and stands out both ways; a ridge
   (the line rhythm of typeset text, column rules) runs along an axis and
   scores ~0, so text isn't mistaken for a screen.
3. Bins above ``threshold`` seed a notch that grows by hysteresis into
   connected bins above ``threshold * _HYSTERESIS`` (fundamentals smeared by
   paper warp, harmonics). Nothing below ``min_freq`` is ever touched.
4. Windows with no seed are passed through unchanged, so text-only and
   blank-paper regions come back bit-identical (up to float rounding).

Runs on CPU in float32 regardless of the upscaler's device: an FFT pass is
cheap next to SR inference, and this keeps the result identical across
MPS / CUDA / CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur, pil_to_tensor, to_pil_image

# Window size in px. 256 resolves a ~65 lpi screen at 200 dpi (period ≈3 px)
# to a sharp peak while staying small enough to follow photo boundaries.
_WINDOW = 256
# Weak (growth) threshold as a fraction of the seed threshold.
_HYSTERESIS = 0.3
# A screen needs two seed peaks at least this far apart in direction ...
_MIN_SPREAD = math.radians(30)
# ... whose periods agree to within this fraction (square dot lattice).
_SAME_PERIOD = 0.05
# Windows processed per FFT batch (bounds peak memory on full newspaper pages).
_CHUNK = 64


@dataclass(frozen=True)
class Peak:
    """A screen peak, in cycles/pixel (``fy`` down, ``fx`` right)."""

    fy: float
    fx: float
    prominence: float

    @property
    def period(self) -> float:
        return 1.0 / math.hypot(self.fy, self.fx)

    @property
    def angle(self) -> float:
        """Screen angle in degrees, folded into ``[0, 180)``."""
        return math.degrees(math.atan2(self.fy, self.fx)) % 180.0


@dataclass(frozen=True)
class DescreenReport:
    peaks: tuple[Peak, ...]  # strongest peak of each screened window, strongest first
    screened: float  # fraction of windows a screen was found in

    @property
    def detected(self) -> bool:
        return bool(self.peaks)

    def describe(self, dpi: float | None = None) -> str:
        if not self.peaks:
            return "no halftone screen detected"
        fund = self.peaks[0]
        msg = f"screen {fund.period:.2f}px @ {fund.angle:.0f}°"
        if dpi:
            msg += f" (~{dpi / fund.period:.0f} lpi)"
        return f"{msg}; descreened {self.screened:.0%} of the image"


def descreen(
    image: Image.Image,
    strength: float = 1.0,
    *,
    min_freq: float = 0.08,
    threshold: float = 1.6,
) -> tuple[Image.Image, DescreenReport]:
    """Descreen a PIL image (``L``, ``LA``, ``RGB``, ``RGBA``). Alpha passes through."""
    n_color = {"L": 1, "LA": 1, "RGB": 3, "RGBA": 3}.get(image.mode)
    if n_color is None:
        raise ValueError(f"unsupported mode {image.mode!r}")
    x = pil_to_tensor(image).float() / 255.0
    out, report = descreen_tensor(x[:n_color], strength, min_freq=min_freq, threshold=threshold)
    if not report.detected:
        return image.copy(), report
    out = torch.cat([out, x[n_color:]], dim=0)
    return to_pil_image(_quantize(out), mode=image.mode), report


def descreen_tensor(
    x: torch.Tensor,
    strength: float = 1.0,
    *,
    min_freq: float = 0.08,
    threshold: float = 1.6,
) -> tuple[torch.Tensor, DescreenReport]:
    """Descreen a ``[C, H, W]`` float tensor in ``[0, 1]``.

    ``strength`` is the notch depth (1 removes the screen, 0 is a no-op).
    ``min_freq`` (cycles/px) protects every frequency below it. ``threshold``
    is the natural-log prominence a spectral peak needs to count as a screen;
    raise it to be more conservative.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [C, H, W], got {tuple(x.shape)}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    src_device, src_dtype = x.device, x.dtype
    x = x.detach().to("cpu", torch.float32)
    c, h, w = x.shape
    win, hop = _WINDOW, _WINDOW // 2

    # Reflect-pad so every pixel is covered by exactly two windows per axis,
    # then round up to a whole number of hops.
    pad_b = hop + (-(h + hop) % hop)
    pad_r = hop + (-(w + hop) % hop)
    padded = F.pad(x[None], (hop, pad_r, hop, pad_b), mode="reflect")[0]
    ph, pw = padded.shape[-2:]
    ny, nx = (ph - win) // hop + 1, (pw - win) // hop + 1

    window = torch.hann_window(win, periodic=True).sqrt()
    window2d = window.view(-1, 1) * window.view(1, -1)
    freqs = torch.fft.fftfreq(win)
    radius = torch.hypot(freqs.view(-1, 1), freqs.view(1, -1))
    allowed = radius >= min_freq

    # [C, ny*nx, win, win] windowed tiles.
    tiles = padded.unfold(1, win, hop).unfold(2, win, hop).reshape(c, ny * nx, win, win)
    tiles = tiles * window2d
    out_tiles = tiles  # filtered in place, chunk by chunk, after its FFT is taken

    peaks: list[Peak] = []
    screened = 0
    for start in range(0, ny * nx, _CHUNK):
        sl = slice(start, start + _CHUNK)
        spec = torch.fft.fft2(tiles[:, sl])  # [C, n, win, win]
        prom = _prominence(spec.mean(dim=0))  # luminance-ish: mean of channel spectra
        region = _notch_region(prom, allowed, threshold)
        hit = region.flatten(1).any(dim=1)
        if not hit.any():
            continue
        screened += int(hit.sum())
        for k in hit.nonzero().flatten().tolist():
            peaks.append(_strongest(prom[k], region[k], freqs))
        atten = 1.0 - strength * _soften(region[hit], allowed)
        filtered = torch.fft.ifft2(spec[:, hit] * atten).real
        idx = hit.nonzero().flatten() + start
        out_tiles[:, idx] = filtered

    report = DescreenReport(
        peaks=tuple(sorted(peaks, key=lambda p: -p.prominence)),
        screened=screened / (ny * nx),
    )
    if not screened or strength == 0.0:
        return x.to(src_device, src_dtype), report

    # Overlap-add with the synthesis window; sqrt-Hann² at 50 % overlap sums
    # to 1, but divide by the actual weight anyway so the borders are exact.
    out_tiles = out_tiles * window2d
    cols = out_tiles.reshape(c, ny * nx, win * win).permute(0, 2, 1)
    acc = F.fold(cols, (ph, pw), kernel_size=win, stride=hop)
    norm = F.fold(
        (window2d * window2d).reshape(1, win * win, 1).expand(1, -1, ny * nx),
        (ph, pw),
        kernel_size=win,
        stride=hop,
    )
    out = (acc / norm.clamp_min(1e-8))[:, 0, hop : hop + h, hop : hop + w]
    return out.clamp(0.0, 1.0).to(src_device, src_dtype), report


def _prominence(spec: torch.Tensor) -> torch.Tensor:
    """Per-window peak prominence, ``[n, win, win]`` in FFT (unshifted) order.

    The smaller of the excess over a horizontal and a vertical 1-D background
    is kept: compact halftone peaks survive, axis-aligned ridges from text
    lines and rules don't.
    """
    log = torch.log1p(spec.abs())
    # Blur in the shifted domain so the kernel doesn't straddle the wrap.
    log = torch.fft.fftshift(log, dim=(-2, -1))
    along_x = gaussian_blur(log, kernel_size=[25, 1], sigma=[5.0, 1.0])
    along_y = gaussian_blur(log, kernel_size=[1, 25], sigma=[1.0, 5.0])
    prom = torch.minimum(log - along_x, log - along_y)
    return torch.fft.ifftshift(prom, dim=(-2, -1))


def _notch_region(prom: torch.Tensor, allowed: torch.Tensor, threshold: float) -> torch.Tensor:
    seed = (prom > threshold) & allowed
    seed &= _is_lattice(seed, prom).view(-1, 1, 1)
    weak = (prom > threshold * _HYSTERESIS) & allowed
    region = seed
    for _ in range(32):
        grown = _dilate(region, 3) & weak | region
        if torch.equal(grown, region):
            break
        region = grown
    # A real image has a conjugate-symmetric spectrum; keep the notch
    # symmetric so the filtered window stays real-valued.
    mirrored = torch.roll(region.flip(-2, -1), shifts=(1, 1), dims=(-2, -1))
    return region | mirrored


def _is_lattice(seed: torch.Tensor, prom: torch.Tensor) -> torch.Tensor:
    """Per window: do the seed peaks look like a halftone dot lattice?

    A halftone screen is a 2-D, (near-)square dot lattice: its fundamentals
    point in two directions at least ~30° apart (45°/135°, 0°/90°, …) *with
    the same period*. Typeset text can also put compact peaks on two axes
    (line pitch vertically, stroke rhythm horizontally), but at unrelated
    periods, and a block of lines alone is 1-D. Requiring a pair of seeds
    that differ in direction yet match in radius rejects both.
    """
    n, h, w = seed.shape
    fy = torch.fft.fftfreq(h).view(-1, 1).expand(h, w)
    fx = torch.fft.fftfreq(w).view(1, -1).expand(h, w)
    radius = torch.hypot(fy, fx)
    theta = torch.atan2(fy, fx) % math.pi  # a direction and its opposite coincide
    out = torch.zeros(n, dtype=torch.bool)
    for k in seed.flatten(1).any(dim=1).nonzero().flatten().tolist():
        m = seed[k]
        p, r, t = prom[k][m], radius[m], theta[m]
        # Anchor on the strongest seed: in a real screen that's a
        # fundamental, so its partner must match *it*. Matching any pair
        # would let one of text's many line-pitch harmonics line up by luck.
        a = int(p.argmax())
        dt = (t - t[a]).abs()
        dt = torch.minimum(dt, math.pi - dt)
        dr = (r - r[a]).abs() / r[a]
        out[k] = bool(((dt >= _MIN_SPREAD) & (dr <= _SAME_PERIOD)).any())
    return out


def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    """Binary dilation with wrap-around (the spectrum is periodic)."""
    p = k // 2
    x = F.pad(m.float()[:, None], (p, p, p, p), mode="circular")
    return F.max_pool2d(x, k, stride=1)[:, 0] > 0


def _soften(region: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Region → soft ``[0, 1]`` notch with a 1-bin margin and a feathered edge."""
    soft = _dilate(region, 3).float()
    soft = torch.fft.fftshift(soft, dim=(-2, -1))
    soft = gaussian_blur(soft, kernel_size=[5, 5], sigma=[1.0, 1.0])
    soft = torch.fft.ifftshift(soft, dim=(-2, -1))
    return torch.where(allowed, soft.clamp(0.0, 1.0), torch.zeros(()))


def _strongest(prom: torch.Tensor, region: torch.Tensor, freqs: torch.Tensor) -> Peak:
    masked = torch.where(region, prom, torch.full((), -torch.inf))
    i, j = divmod(int(masked.argmax()), masked.shape[-1])
    fy, fx = freqs[i].item(), freqs[j].item()
    if fx < 0 or (fx == 0 and fy < 0):  # fold to one half-plane
        fy, fx = -fy, -fx
    return Peak(fy=fy, fx=fx, prominence=masked[i, j].item())


def _quantize(x: torch.Tensor) -> torch.Tensor:
    return (x * 255.0).round().clamp(0, 255).to(torch.uint8)
