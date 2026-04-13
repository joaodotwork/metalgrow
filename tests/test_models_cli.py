import hashlib

import pytest
from typer.testing import CliRunner

from metalgrow import weights
from metalgrow.cli import app

runner = CliRunner()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("METALGROW_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_model(cache, monkeypatch):
    content = b"pretend weights"
    spec = weights.WeightSpec(
        name="fake.pth",
        url="http://unused",
        sha256=hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setitem(weights.REGISTRY, "fake", spec)
    return spec, content


def test_models_list_shows_registered(cache):
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.stdout
    assert "realesrgan-x2" in result.stdout
    assert "realesrgan-x4" in result.stdout
    assert "missing" in result.stdout


def test_models_list_marks_cached(cache, fake_model):
    spec, content = fake_model
    (cache / spec.name).write_bytes(content)
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    # The `fake` row should report the cached status and a byte-count-derived size.
    line = next(line for line in result.stdout.splitlines() if line.startswith("fake"))
    assert "cached" in line


def test_models_rm_unknown_errors(cache):
    result = runner.invoke(app, ["models", "rm", "nope"])
    assert result.exit_code != 0


def test_models_rm_removes_cached_file(cache, fake_model):
    spec, content = fake_model
    (cache / spec.name).write_bytes(content)

    result = runner.invoke(app, ["models", "rm", "fake"])
    assert result.exit_code == 0
    assert "removed" in result.stdout
    assert not (cache / spec.name).exists()


def test_models_rm_when_missing_reports_noop(cache, fake_model):
    result = runner.invoke(app, ["models", "rm", "fake"])
    assert result.exit_code == 0
    assert "not cached" in result.stdout


def test_models_download_unknown_errors(cache):
    result = runner.invoke(app, ["models", "download", "nope"])
    assert result.exit_code != 0
