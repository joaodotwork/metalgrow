from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from metalgrow import Upscaler
from metalgrow.batch import IMAGE_EXTS, discover_inputs, plan_outputs, run_batch
from metalgrow.cli import app

runner = CliRunner()


def _seed_images(dirpath: Path, names: list[str], size=(16, 12)) -> list[Path]:
    paths = []
    for name in names:
        p = dirpath / name
        Image.new("RGB", size, color=(80, 120, 200)).save(p)
        paths.append(p)
    return paths


# ---------- discover_inputs ----------


def test_discover_single_file(tmp_path):
    [p] = _seed_images(tmp_path, ["a.png"])
    assert discover_inputs(str(p)) == [p]


def test_discover_directory_filters_by_extension(tmp_path):
    img_paths = _seed_images(tmp_path, ["a.png", "b.jpg"])
    (tmp_path / "notes.txt").write_text("not an image")
    found = discover_inputs(str(tmp_path))
    assert sorted(found) == sorted(img_paths)


def test_discover_glob(tmp_path):
    _seed_images(tmp_path, ["a.png", "b.png", "c.jpg"])
    found = discover_inputs(str(tmp_path / "*.png"))
    assert {p.name for p in found} == {"a.png", "b.png"}


def test_discover_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_inputs(str(tmp_path / "nothing-here"))


def test_image_ext_set_includes_common_formats():
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        assert ext in IMAGE_EXTS


# ---------- run_batch ----------


def test_run_batch_writes_all_outputs(tmp_path):
    src_dir = tmp_path / "in"
    dst_dir = tmp_path / "out"
    src_dir.mkdir()
    srcs = _seed_images(src_dir, ["a.png", "b.png", "c.png"])
    items = plan_outputs(srcs, dst_dir)

    up = Upscaler(device="cpu")
    result = run_batch(up, items, scale=2.0, tile=None, tile_pad=None, progress=False)

    assert result.processed == 3
    assert result.skipped == 0
    assert result.failed == 0
    for _src, dst in items:
        assert dst.exists()
        assert Image.open(dst).size == (32, 24)


def test_run_batch_skip_existing(tmp_path):
    src_dir = tmp_path / "in"
    dst_dir = tmp_path / "out"
    src_dir.mkdir()
    dst_dir.mkdir()
    srcs = _seed_images(src_dir, ["a.png", "b.png"])
    # Pre-create one output so the batch should skip it.
    Image.new("RGB", (1, 1)).save(dst_dir / "a.png")

    items = plan_outputs(srcs, dst_dir)
    up = Upscaler(device="cpu")
    result = run_batch(
        up, items, scale=2.0, tile=None, tile_pad=None, skip_existing=True, progress=False
    )

    assert result.processed == 1
    assert result.skipped == 1
    # The pre-seeded 1x1 file is left untouched.
    assert Image.open(dst_dir / "a.png").size == (1, 1)
    assert Image.open(dst_dir / "b.png").size == (32, 24)


# ---------- CLI ----------


def test_cli_batch_directory(tmp_path):
    src_dir = tmp_path / "in"
    dst_dir = tmp_path / "out"
    src_dir.mkdir()
    _seed_images(src_dir, ["a.png", "b.png"])

    result = runner.invoke(
        app,
        ["upscale", str(src_dir), str(dst_dir), "--scale", "2", "--device", "cpu"],
    )
    assert result.exit_code == 0, result.stdout
    assert "2 processed" in result.stdout
    assert (dst_dir / "a.png").exists()
    assert (dst_dir / "b.png").exists()


def test_cli_batch_skip_existing_flag(tmp_path):
    src_dir = tmp_path / "in"
    dst_dir = tmp_path / "out"
    src_dir.mkdir()
    dst_dir.mkdir()
    _seed_images(src_dir, ["a.png", "b.png"])
    Image.new("RGB", (1, 1)).save(dst_dir / "a.png")

    result = runner.invoke(
        app,
        [
            "upscale",
            str(src_dir),
            str(dst_dir),
            "--scale",
            "2",
            "--device",
            "cpu",
            "--skip-existing",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "1 processed" in result.stdout
    assert "1 skipped" in result.stdout


def test_cli_batch_glob(tmp_path):
    src_dir = tmp_path / "in"
    dst_dir = tmp_path / "out"
    src_dir.mkdir()
    _seed_images(src_dir, ["a.png", "b.png", "c.jpg"])

    result = runner.invoke(
        app,
        [
            "upscale",
            str(src_dir / "*.png"),
            str(dst_dir),
            "--scale",
            "2",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (dst_dir / "a.png").exists()
    assert (dst_dir / "b.png").exists()
    assert not (dst_dir / "c.jpg").exists()


def test_cli_empty_directory_errors(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    result = runner.invoke(app, ["upscale", str(src_dir), str(tmp_path / "out"), "--device", "cpu"])
    assert result.exit_code != 0
