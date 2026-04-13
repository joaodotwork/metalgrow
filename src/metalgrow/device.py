import os

import torch


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        _enable_mps_fallback()
        return torch.device("mps")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _enable_mps_fallback() -> None:
    """Set PYTORCH_ENABLE_MPS_FALLBACK=1 so ops missing on MPS fall back to CPU.

    Several torch ops (notably some upsampling and reflection-pad variants)
    still have gaps on the MPS backend. Without the fallback, a single
    unsupported op aborts the forward pass; with it, torch silently executes
    the op on CPU and returns to MPS for the rest.

    Opt out by setting ``METALGROW_DISABLE_MPS_FALLBACK=1`` — useful when
    debugging which op is falling back.
    """
    if os.environ.get("METALGROW_DISABLE_MPS_FALLBACK") == "1":
        return
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
