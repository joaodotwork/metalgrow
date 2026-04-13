from pathlib import Path

import torch
import typer

from metalgrow.backbones import list_backbones
from metalgrow.batch import discover_inputs, plan_outputs, run_batch
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
    workers: int = typer.Option(
        4, "--workers", "-j", min=1, help="Parallel I/O workers (inference stays serial)"
    ),
):
    if dtype not in _DTYPES:
        raise typer.BadParameter(f"dtype must be one of {list(_DTYPES)}")
    try:
        inputs = discover_inputs(src)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"source not found: {exc}") from None
    if not inputs:
        raise typer.BadParameter(f"no images found at {src!r}")

    upscaler = Upscaler(backbone=backbone, device=device, dtype=_DTYPES[dtype])
    typer.echo(f"device: {upscaler.device}")
    typer.echo(f"backbone: {backbone}")
    typer.echo(f"dtype: {dtype}")

    batch_mode = len(inputs) > 1 or Path(src).is_dir() or any(ch in src for ch in "*?[")

    if not batch_mode:
        out = upscaler.upscale_file(inputs[0], dst, scale=scale, tile=tile, tile_pad=tile_pad)
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
    )
    typer.echo(
        f"done: {result.processed} processed, {result.skipped} skipped, "
        f"{result.failed} failed (of {result.total})"
    )


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
