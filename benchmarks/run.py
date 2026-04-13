"""Reproducible benchmark runner for metalgrow backbones.

Measures per-iteration wall time and peak memory for a sweep over
``(device, backbone, scale)``. Inputs are synthetic (seeded uniform noise) so
the run is reproducible across machines without fixture downloads.

Not part of CI — invoke manually. CUDA numbers are produced by running this
same script on a CUDA-enabled runner (see ``benchmarks/README.md``).

Example:
    python -m benchmarks.run --device auto --size 256 --iters 3 \
        --output docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from PIL import Image

from metalgrow import weights
from metalgrow.device import get_device
from metalgrow.upscaler import Upscaler

SEED = 0

DEFAULT_BACKBONES = (
    "bicubic",
    "realesrgan-x2",
    "realesrgan-x4",
    "swinir-x2",
    "swinir-x4",
)
DEFAULT_SCALES = (2.0, 4.0)


@dataclass
class Result:
    device: str
    backbone: str
    scale: float
    in_size: int
    out_megapixels: float
    iters: int
    median_s: float
    mp_per_s: float
    peak_mib: float | None
    note: str = ""


@dataclass
class Run:
    host: dict
    torch: dict
    results: list[Result] = field(default_factory=list)


def _synth_image(size: int) -> Image.Image:
    """Return a seeded-random RGB image of ``size × size``."""
    g = torch.Generator().manual_seed(SEED)
    t = (torch.rand(3, size, size, generator=g) * 255).to(torch.uint8)
    return Image.fromarray(t.permute(1, 2, 0).numpy(), mode="RGB")


def _allowed_scales(backbone: str, scales: tuple[float, ...]) -> list[float]:
    # Learned backbones bake the scale into the weights; bicubic accepts any.
    if "-x" not in backbone:
        return list(scales)
    try:
        native = float(backbone.rsplit("-x", 1)[1])
    except ValueError:
        return []
    return [native] if native in scales else []


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak(device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        return 0
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return 0


def _read_peak(device: torch.device, baseline: int) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device.type == "mps":
        now = int(torch.mps.driver_allocated_memory())
        delta = max(0, now - baseline)
        return delta / (1024 * 1024)
    return None


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _host_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def _torch_info(device: torch.device) -> dict:
    info = {
        "version": torch.__version__,
        "device": device.type,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "commit": _git_commit(),
    }
    if device.type == "cuda":
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def bench_one(
    device_pref: str,
    backbone: str,
    scale: float,
    size: int,
    warmup: int,
    iters: int,
    skip_uncached: bool,
) -> Result:
    """Run one ``(device, backbone, scale)`` cell and return its ``Result``."""
    device = get_device(device_pref)

    if skip_uncached and backbone in weights.REGISTRY and not weights.is_cached(backbone):
        return Result(
            device=device.type,
            backbone=backbone,
            scale=scale,
            in_size=size,
            out_megapixels=0.0,
            iters=0,
            median_s=0.0,
            mp_per_s=0.0,
            peak_mib=None,
            note="skipped (weights not cached)",
        )

    upscaler = Upscaler(backbone=backbone, device=device_pref)
    img = _synth_image(size)

    for _ in range(warmup):
        upscaler.upscale(img, scale=scale)
    _sync(device)

    baseline = _reset_peak(device)

    times: list[float] = []
    last = None
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        last = upscaler.upscale(img, scale=scale)
        _sync(device)
        times.append(time.perf_counter() - t0)

    peak = _read_peak(device, baseline)
    assert last is not None
    out_w, out_h = last.size
    out_mp = out_w * out_h / 1e6
    median = statistics.median(times)

    return Result(
        device=device.type,
        backbone=backbone,
        scale=scale,
        in_size=size,
        out_megapixels=round(out_mp, 3),
        iters=iters,
        median_s=round(median, 4),
        mp_per_s=round(out_mp / median, 2) if median > 0 else 0.0,
        peak_mib=round(peak, 1) if peak is not None else None,
    )


def run_suite(
    device_pref: str,
    backbones: tuple[str, ...],
    scales: tuple[float, ...],
    size: int,
    warmup: int,
    iters: int,
    skip_uncached: bool,
) -> Run:
    device = get_device(device_pref)
    run = Run(host=_host_info(), torch=_torch_info(device))
    for backbone in backbones:
        for scale in _allowed_scales(backbone, scales):
            result = bench_one(
                device_pref=device_pref,
                backbone=backbone,
                scale=scale,
                size=size,
                warmup=warmup,
                iters=iters,
                skip_uncached=skip_uncached,
            )
            run.results.append(result)
            _print_progress(result)
    return run


def _print_progress(r: Result) -> None:
    tail = r.note or (
        f"{r.median_s:.3f}s · {r.mp_per_s:.1f} MP/s"
        + (f" · peak {r.peak_mib:.0f} MiB" if r.peak_mib is not None else "")
    )
    print(f"  {r.device:>4} · {r.backbone:<14} x{r.scale:g}  {tail}", file=sys.stderr)


def to_markdown(run: Run) -> str:
    host = run.host
    t = run.torch
    header = (
        f"Host: {host['system']} {host['release']} · {host['processor']} · "
        f"Python {host['python']}\n"
        f"Torch: {t['version']} · device={t['device']} · commit={t['commit']}"
    )
    if t["device"] == "cuda" and "cuda_device" in t:
        header += f" · {t['cuda_device']}"

    rows = [
        (
            "| Device | Backbone | Scale | In (px) | Out (MP) | Iters | "
            "Median (s) | MP/s | Peak (MiB) | Note |"
        ),
        (
            "|--------|----------|-------|---------|----------|-------|"
            "------------|------|------------|------|"
        ),
    ]
    for r in run.results:
        peak = f"{r.peak_mib:.1f}" if r.peak_mib is not None else "—"
        note = r.note or ""
        rows.append(
            f"| {r.device} | {r.backbone} | {r.scale:g} | {r.in_size} | "
            f"{r.out_megapixels:.2f} | {r.iters} | {r.median_s:.3f} | "
            f"{r.mp_per_s:.2f} | {peak} | {note} |"
        )
    return header + "\n\n" + "\n".join(rows) + "\n"


def to_json(run: Run) -> str:
    return json.dumps(
        {
            "host": run.host,
            "torch": run.torch,
            "results": [asdict(r) for r in run.results],
        },
        indent=2,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument(
        "--backbones",
        default=",".join(DEFAULT_BACKBONES),
        help="comma-separated backbone names (default: all)",
    )
    p.add_argument(
        "--scales",
        default=",".join(str(s) for s in DEFAULT_SCALES),
        help="comma-separated scales (default: 2,4)",
    )
    p.add_argument("--size", type=int, default=256, help="input edge length in px")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument(
        "--skip-uncached",
        action="store_true",
        help="skip learned backbones whose weights are not cached (no download)",
    )
    p.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "json"],
        help="output format",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="write output to file instead of stdout",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    backbones = tuple(b.strip() for b in args.backbones.split(",") if b.strip())
    scales = tuple(float(s) for s in args.scales.split(",") if s.strip())

    print(
        f"Benchmarking device={args.device} size={args.size} "
        f"warmup={args.warmup} iters={args.iters}",
        file=sys.stderr,
    )
    run = run_suite(
        device_pref=args.device,
        backbones=backbones,
        scales=scales,
        size=args.size,
        warmup=args.warmup,
        iters=args.iters,
        skip_uncached=args.skip_uncached,
    )

    rendered = to_markdown(run) if args.format == "markdown" else to_json(run)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
