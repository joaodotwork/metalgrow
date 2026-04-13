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


def test_download_progress_emits_no_cr_when_stderr_not_tty(cache, monkeypatch, capsys):
    """On non-TTY stderr, progress should be plain lines — no `\\r` carriage returns."""
    payload = b"x" * (3 << 20)  # 3 MB so decile updates actually fire
    spec = weights.WeightSpec(
        name="progress.pth",
        url="http://unused",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setitem(weights.REGISTRY, "progress", spec)

    class FakeResp:
        def __init__(self, data: bytes):
            self._buf = data
            self.headers = {"Content-Length": str(len(data))}

        def read(self, n: int) -> bytes:
            chunk, self._buf = self._buf[:n], self._buf[n:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(weights, "urlopen", lambda url: FakeResp(payload))
    # Force the non-TTY branch regardless of test runner env.
    monkeypatch.setattr(weights.sys.stderr, "isatty", lambda: False, raising=False)

    path = weights.ensure_weight("progress")
    assert path.exists()

    err = capsys.readouterr().err
    assert "\r" not in err, f"found carriage return in non-tty output: {err!r}"
    assert "downloading progress.pth: 100%" in err


def test_registry_has_realesrgan_entries():
    assert "realesrgan-x2" in weights.REGISTRY
    assert "realesrgan-x4" in weights.REGISTRY
    for spec in weights.REGISTRY.values():
        assert len(spec.sha256) == 64
        assert spec.url.startswith("https://")
