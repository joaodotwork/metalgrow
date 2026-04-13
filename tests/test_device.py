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
