"""Pillow-backed image and external-mask decoding."""

from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageOps

from watermark_removal_lab.common.types import ImageData, UInt8Array

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png"})


class UnsupportedImageFormatError(ValueError):
    """Raised when a path does not use an M1-supported image extension."""


class ImageReadError(RuntimeError):
    """Raised when a supported image file cannot be decoded."""


class MaskReadError(RuntimeError):
    """Raised when a supported external mask file cannot be decoded."""


def _validate_extension(path: Path) -> None:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        displayed_extension = path.suffix or "<none>"
        raise UnsupportedImageFormatError(
            f"unsupported image extension '{displayed_extension}'; expected one of: {supported}"
        )


def _has_alpha(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def _decode_oriented_image(source: Image.Image) -> ImageData:
    oriented = ImageOps.exif_transpose(source)
    try:
        if _has_alpha(oriented):
            converted = oriented.convert("RGBA")
            try:
                pixels = cast(UInt8Array, np.asarray(converted, dtype=np.uint8))
                return ImageData(rgb=pixels[:, :, :3], alpha=pixels[:, :, 3])
            finally:
                converted.close()

        converted = oriented.convert("RGB")
        try:
            pixels = cast(UInt8Array, np.asarray(converted, dtype=np.uint8))
            return ImageData(rgb=pixels)
        finally:
            converted.close()
    finally:
        oriented.close()


def read_image(path: Path) -> ImageData:
    """Decode an M1-supported image into canonical RGB and optional alpha.

    JPEG and PNG are supported case-insensitively. EXIF orientation is applied
    before dimensions are exposed. Grayscale and palette images are normalized
    to RGB, while transparency is retained as a separate alpha channel.
    """
    _validate_extension(path)
    try:
        with Image.open(path) as source:
            return _decode_oriented_image(source)
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ImageReadError(f"could not decode image '{path.name}'") from error


def read_mask_intensity(path: Path) -> UInt8Array:
    """Decode external-mask intensity without applying orientation metadata.

    When the file contains transparency, alpha is used as mask intensity.
    Otherwise the image is converted to grayscale. The returned array is a new
    ``uint8`` array shaped ``H x W``. Mask orientation is deliberately not
    guessed or transposed.
    """
    _validate_extension(path)
    try:
        with Image.open(path) as source:
            mode = "RGBA" if _has_alpha(source) else "L"
            converted = source.convert(mode)
            try:
                pixels = cast(UInt8Array, np.asarray(converted, dtype=np.uint8))
                if mode == "RGBA":
                    return pixels[:, :, 3].copy(order="C")
                return pixels.copy(order="C")
            finally:
                converted.close()
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise MaskReadError(f"could not decode mask '{path.name}'") from error
