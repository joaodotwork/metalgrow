from PIL import Image
from typer.testing import CliRunner

from metalgrow.cli import app

runner = CliRunner()


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "torch:" in result.stdout
    assert "mps available:" in result.stdout
    assert "cuda available:" in result.stdout


def test_cli_upscale_smoke(tmp_path):
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(src)

    result = runner.invoke(
        app,
        ["upscale", str(src), str(dst), "--scale", "2", "--device", "cpu"],
    )
    assert result.exit_code == 0, result.stdout
    assert "backbone: bicubic" in result.stdout
    assert dst.exists()
    assert Image.open(dst).size == (32, 24)


def test_cli_upscale_accepts_fp16(tmp_path):
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    Image.new("RGB", (8, 8)).save(src)

    result = runner.invoke(
        app,
        ["upscale", str(src), str(dst), "--device", "cpu", "--dtype", "fp16"],
    )
    # fp16 on CPU is slow but functional; we only care the flag parses and
    # the pipeline round-trips without a type error.
    assert result.exit_code == 0, result.stdout
    assert "dtype: fp16" in result.stdout


def test_cli_upscale_rejects_unknown_dtype(tmp_path):
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    Image.new("RGB", (8, 8)).save(src)

    result = runner.invoke(
        app,
        ["upscale", str(src), str(dst), "--device", "cpu", "--dtype", "bogus"],
    )
    assert result.exit_code != 0


def test_cli_upscale_unknown_backbone(tmp_path):
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    Image.new("RGB", (8, 8)).save(src)

    result = runner.invoke(
        app,
        ["upscale", str(src), str(dst), "--device", "cpu", "--backbone", "nope"],
    )
    assert result.exit_code != 0
