from importlib.metadata import PackageNotFoundError, version

from metalgrow.device import get_device
from metalgrow.upscaler import Upscaler

__all__ = ["Upscaler", "get_device"]

try:
    # Single source of truth: the version in pyproject.toml, via the installed
    # distribution's metadata (a hard-coded string here drifted to 0.0.1).
    __version__ = version("metalgrow")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"
