from pathlib import Path

import torch
import typer
from PIL import Image

from metalgrow.backbones import list_backbones
from metalgrow.batch import discover_inputs, plan_outputs, run_batch
from metalgrow.bitmap import to_bitmap
from metalgrow.descreen import descreen as descreen_image
from metalgrow.ink import ALPHA_MODES
from metalgrow.metadata import capture, reapply
from metalgrow.upscaler import Upscaler
from metalgrow.weights import (
    REGISTRY,
    cached_path,
    ensure_weight,
    remove_cached,
)

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

app = typer.Typer(help="metalgrow — AI image upscaler on Apple Metal.")


@app.command()
def upscale(
    src: str = typer.Argument(..., help="Image file, directory, or glob (e.g. 'in/*.png')"),
    dst: Path = typer.Argument(..., help="Output file (single src) or directory (batch)"),
    scale: float = typer.Option(2.0, "--scale", "-s", min=1.01, max=8.0),
    device: str = typer.Option("auto", "--device", "-d", help="auto | mps | cuda | cpu"),
    backbone: str = typer.Option(
        "bicubic",
        "--backbone",
        "-b",
        help=f"SR backbone: {', '.join(list_backbones())}",
    ),
    dtype: str = typer.Option(
        "fp32", "--dtype", help="Inference dtype: fp32 | fp16 (fp16 MPS-only, noisier)"
    ),
    tile: int | None = typer.Option(
        None, "--tile", help="Tile size in input px (0 disables; omit = backbone default)"
    ),
    tile_pad: int | None = typer.Option(
        None, "--tile-pad", help="Context padding per tile edge (omit = backbone default)"
    ),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip outputs that already exist (batch mode)"
    ),
    preserve_metadata: bool = typer.Option(
        False,
        "--preserve-metadata",
        "-p",
        help="Keep grayscale mode, embedded ICC profile, and rescaled DPI tag",
    ),
    workers: int = typer.Option(
        4, "--workers", "-j", min=1, help="Parallel I/O workers (inference stays serial)"
    ),
    descreen: bool = typer.Option(
        False,
        "--descreen",
        help="Remove halftone screens (moiré) from scanned print before upscaling",
    ),
    descreen_strength: float = typer.Option(
        1.0, "--descreen-strength", min=0.0, max=1.0, help="Notch depth, 0..1"
    ),
    alpha_mode: str = typer.Option(
        "separate",
        "--alpha-mode",
        help="separate (backbone on RGB, bicubic alpha) | ink (single-colour art: "
        "backbone on coverage, paper knocked out) | auto (ink when the art is one colour)",
    ),
    dpi: float | None = typer.Option(
        None, "--dpi", min=1.0, help="Write this DPI tag instead of the source's (rescaled)"
    ),
):
    if dtype not in _DTYPES:
        raise typer.BadParameter(f"dtype must be one of {list(_DTYPES)}")
    if alpha_mode not in ALPHA_MODES:
        raise typer.BadParameter(f"alpha-mode must be one of {list(ALPHA_MODES)}")
    try:
        inputs = discover_inputs(src)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"source not found: {exc}") from None
    if not inputs:
        raise typer.BadParameter(f"no images found at {src!r}")

    upscaler = Upscaler(
        backbone=backbone,
        device=device,
        dtype=_DTYPES[dtype],
        descreen=descreen,
        descreen_strength=descreen_strength,
        alpha_mode=alpha_mode,
    )
    typer.echo(f"device: {upscaler.device}")
    typer.echo(f"backbone: {backbone}")
    typer.echo(f"dtype: {dtype}")
    if descreen:
        typer.echo(f"descreen: strength {descreen_strength}")
    if alpha_mode != "separate":
        typer.echo(f"alpha-mode: {alpha_mode}")

    batch_mode = len(inputs) > 1 or Path(src).is_dir() or any(ch in src for ch in "*?[")

    if not batch_mode:
        out = upscaler.upscale_file(
            inputs[0],
            dst,
            scale=scale,
            tile=tile,
            tile_pad=tile_pad,
            preserve_metadata=preserve_metadata,
            dpi=dpi,
        )
        if upscaler.last_descreen is not None:
            typer.echo(f"descreen: {upscaler.last_descreen.describe(_dpi(inputs[0]))}")
        typer.echo(f"wrote: {out}")
        return

    if dst.exists() and not dst.is_dir():
        raise typer.BadParameter(f"batch destination must be a directory: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    items = plan_outputs(inputs, dst)
    result = run_batch(
        upscaler,
        items,
        scale=scale,
        tile=tile,
        tile_pad=tile_pad,
        workers=workers,
        skip_existing=skip_existing,
        preserve_metadata=preserve_metadata,
        dpi=dpi,
    )
    typer.echo(
        f"done: {result.processed} processed, {result.skipped} skipped, "
        f"{result.failed} failed (of {result.total})"
    )


@app.command("descreen")
def descreen_cmd(
    src: str = typer.Argument(..., help="Image file, directory, or glob"),
    dst: Path = typer.Argument(..., help="Output file (single src) or directory (batch)"),
    strength: float = typer.Option(1.0, "--strength", min=0.0, max=1.0, help="Notch depth, 0..1"),
    preserve_metadata: bool = typer.Option(
        False, "--preserve-metadata", "-p", help="Keep grayscale mode, ICC profile, and DPI"
    ),
):
    """Remove halftone screens (moiré) without resizing."""
    try:
        inputs = discover_inputs(src)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"source not found: {exc}") from None
    if not inputs:
        raise typer.BadParameter(f"no images found at {src!r}")

    batch_mode = len(inputs) > 1 or Path(src).is_dir() or any(ch in src for ch in "*?[")
    if batch_mode:
        if dst.exists() and not dst.is_dir():
            raise typer.BadParameter(f"batch destination must be a directory: {dst}")
        items = plan_outputs(inputs, dst)
    else:
        items = [(inputs[0], dst)]

    for in_path, out_path in items:
        image = Image.open(in_path)
        meta = capture(image) if preserve_metadata else None
        has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
        working = image.convert("RGBA" if has_alpha else "RGB")
        result, report = descreen_image(working, strength)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if meta is not None:
            result, save_kwargs = reapply(result, meta)
            result.save(out_path, **save_kwargs)
        else:
            result.save(out_path)
        typer.echo(f"{in_path.name}: {report.describe(_dpi(in_path))} -> {out_path}")


@app.command("bitmap")
def bitmap_cmd(
    src: str = typer.Argument(..., help="Image file, directory, or glob"),
    dst: Path = typer.Argument(..., help="Output .tif (single src) or directory (batch)"),
    ppi: int = typer.Option(1200, "--ppi", min=1, help="Master resolution"),
    threshold: int = typer.Option(
        128, "--threshold", min=1, max=255, help="Coverage level counted as ink, 1..255"
    ),
    src_dpi: float | None = typer.Option(
        None, "--src-dpi", min=1.0, help="Source DPI if the file's tag is missing or wrong"
    ),
):
    """1-bit line-art masters (Group 4 TIFF) at the same physical size."""
    try:
        inputs = discover_inputs(src)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"source not found: {exc}") from None
    if not inputs:
        raise typer.BadParameter(f"no images found at {src!r}")

    batch_mode = len(inputs) > 1 or Path(src).is_dir() or any(ch in src for ch in "*?[")
    if batch_mode:
        if dst.exists() and not dst.is_dir():
            raise typer.BadParameter(f"batch destination must be a directory: {dst}")
        items = [(s, d.with_suffix(".tif")) for s, d in plan_outputs(inputs, dst)]
    else:
        items = [(inputs[0], dst)]

    for in_path, out_path in items:
        with Image.open(in_path) as image:
            if src_dpi is None and "dpi" not in image.info:
                typer.echo(f"{in_path.name}: no DPI tag; keeping pixel size (pass --src-dpi)")
            master = to_bitmap(image, ppi=ppi, threshold=threshold, src_dpi=src_dpi)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        master.save(out_path, format="TIFF", compression="group4", dpi=(ppi, ppi))
        cm = 2.54 / ppi
        typer.echo(
            f"{in_path.name}: {master.width}x{master.height} @ {ppi} ppi "
            f"({master.width * cm:.1f} x {master.height * cm:.1f} cm) -> {out_path}"
        )


def _dpi(path: Path) -> float | None:
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
    return float(dpi[0]) if dpi else None


@app.command()
def info():
    import torch

    typer.echo(f"torch: {torch.__version__}")
    typer.echo(f"mps available: {torch.backends.mps.is_available()}")
    typer.echo(f"cuda available: {torch.cuda.is_available()}")


models_app = typer.Typer(help="Manage cached model weights.")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list():
    """List registered models with cache status."""
    header = f"{'NAME':<20} {'SIZE':>10}  {'SHA256':<16}  STATUS"
    typer.echo(header)
    for name, spec in sorted(REGISTRY.items()):
        path = cached_path(name)
        if path.exists():
            size = f"{path.stat().st_size / 1e6:.1f} MB"
            status = "cached"
        else:
            size = "-"
            status = "missing"
        typer.echo(f"{name:<20} {size:>10}  {spec.sha256[:16]}  {status}")


@models_app.command("download")
def models_download(name: str = typer.Argument(...)):
    """Fetch weights for NAME and verify the sha256."""
    if name not in REGISTRY:
        raise typer.BadParameter(f"unknown model {name!r}; see `metalgrow models list`")
    path = ensure_weight(name)
    typer.echo(f"ok: {path}")


@models_app.command("rm")
def models_rm(name: str = typer.Argument(...)):
    """Remove the cached weight for NAME."""
    if name not in REGISTRY:
        raise typer.BadParameter(f"unknown model {name!r}; see `metalgrow models list`")
    typer.echo("removed" if remove_cached(name) else "not cached")


if __name__ == "__main__":
    app()
