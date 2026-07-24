"""Tests for package metadata."""

from importlib.metadata import version

import watermark_removal_lab


def test_package_exposes_installed_version() -> None:
    """The public version must match the installed distribution metadata."""
    assert watermark_removal_lab.__version__ == version("watermark-removal-lab")
