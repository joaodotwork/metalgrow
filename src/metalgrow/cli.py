from pathlib import Path

import typer

from metalgrow.upscaler import Upscaler

app = typer.Typer(help="metalgrow — AI image upscaler on Apple Metal.")


@app.command()
def upscale(
    src: Path = typer.Argument(..., exists=True, readable=True),
    dst: Path = typer.Argument(...),
    scale: float = typer.Option(2.0, "--scale", "-s", min=1.01, max=8.0),
    device: str = typer.Option("auto", "--device", "-d", help="auto | mps | cuda | cpu"),
):
    upscaler = Upscaler(device=device)
    typer.echo(f"device: {upscaler.device}")
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
