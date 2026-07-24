"""Watermark Removal Lab package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("watermark-removal-lab")
except PackageNotFoundError:  # pragma: no cover - only used without an installed package
    __version__ = "0.0.0+uninstalled"

__all__ = ["__version__"]
