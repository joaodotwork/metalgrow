from pathlib import Path

import torch
import typer

from metalgrow.backbones import list_backbones
from metalgrow.upscaler import Upscaler

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

app = typer.Typer(help="metalgrow — AI image upscaler on Apple Metal.")


@app.command()
def upscale(
    src: Path = typer.Argument(..., exists=True, readable=True),
    dst: Path = typer.Argument(...),
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
):
    if dtype not in _DTYPES:
        raise typer.BadParameter(f"dtype must be one of {list(_DTYPES)}")
    upscaler = Upscaler(backbone=backbone, device=device, dtype=_DTYPES[dtype])
    typer.echo(f"device: {upscaler.device}")
    typer.echo(f"backbone: {backbone}")
    typer.echo(f"dtype: {dtype}")
    out = upscaler.upscale_file(src, dst, scale=scale)
    typer.echo(f"wrote: {out}")


@app.command()
def info():
    import torch

    typer.echo(f"torch: {torch.__version__}")
    typer.echo(f"mps available: {torch.backends.mps.is_available()}")
    typer.echo(f"cuda available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    app()
