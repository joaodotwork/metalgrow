import hashlib

import pytest

from metalgrow import weights


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("METALGROW_CACHE_DIR", str(tmp_path))
    return tmp_path


def _seed(cache, spec, content: bytes) -> None:
    (cache / spec.name).write_bytes(content)


def test_cache_dir_honors_env(cache):
    assert weights.cache_dir() == cache


def test_ensure_weight_returns_cached_file_when_checksum_matches(cache, monkeypatch):
    content = b"pretend weights"
    spec = weights.WeightSpec(
        name="fake.pth", url="http://unused", sha256=hashlib.sha256(content).hexdigest()
    )
    monkeypatch.setitem(weights.REGISTRY, "fake", spec)
    _seed(cache, spec, content)

    path = weights.ensure_weight("fake")
    assert path == cache / "fake.pth"
    assert path.read_bytes() == content


def test_ensure_weight_rejects_corrupt_cache(cache, monkeypatch):
    spec = weights.WeightSpec(name="bad.pth", url="http://unused", sha256="0" * 64)
    monkeypatch.setitem(weights.REGISTRY, "bad", spec)
    _seed(cache, spec, b"not the right bytes")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        weights.ensure_weight("bad")
    # Corrupt file should be removed so a retry can re-download cleanly.
    assert not (cache / "bad.pth").exists()


def test_ensure_weight_unknown_backbone(cache):
    with pytest.raises(KeyError, match="no weights registered"):
        weights.ensure_weight("does-not-exist")


def test_registry_has_realesrgan_entries():
    assert "realesrgan-x2" in weights.REGISTRY
    assert "realesrgan-x4" in weights.REGISTRY
    for spec in weights.REGISTRY.values():
        assert len(spec.sha256) == 64
        assert spec.url.startswith("https://")
