from importlib.metadata import PackageNotFoundError, version

__all__ = ["Upscaler", "get_device"]

try:
    # Single source of truth: the version in pyproject.toml, via the installed
    # distribution's metadata (a hard-coded string here drifted to 0.0.1).
    __version__ = version("metalgrow")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"


def __getattr__(name: str):
    # Lazy (PEP 562), so torch-free submodules such as ``metalgrow.matte``
    # can be imported without loading torch.
    if name == "Upscaler":
        from metalgrow.upscaler import Upscaler

        return Upscaler
    if name == "get_device":
        from metalgrow.device import get_device

        return get_device
    raise AttributeError(f"module 'metalgrow' has no attribute {name!r}")
