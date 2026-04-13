from unittest.mock import patch

import torch

from metalgrow.device import get_device


def test_device_cpu_explicit():
    assert get_device("cpu") == torch.device("cpu")


def test_device_fallback_when_mps_unavailable():
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert get_device("mps") == torch.device("cpu")
        assert get_device("auto") == torch.device("cpu")


def test_mps_selection_sets_fallback_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.delenv("METALGROW_DISABLE_MPS_FALLBACK", raising=False)
    with patch("torch.backends.mps.is_available", return_value=True):
        assert get_device("mps") == torch.device("mps")
    import os

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_mps_fallback_opt_out(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setenv("METALGROW_DISABLE_MPS_FALLBACK", "1")
    with patch("torch.backends.mps.is_available", return_value=True):
        get_device("mps")
    import os

    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
