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
    assert dst.exists()
    assert Image.open(dst).size == (32, 24)
