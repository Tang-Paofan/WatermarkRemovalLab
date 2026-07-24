"""Lossless and lossy image encoding with atomic final-path publication."""

import os
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from watermark_removal_lab.common import BinaryMask, ImageData, UInt8Array, mask_to_uint8
from watermark_removal_lab.image.io import UnsupportedImageFormatError


class ImageWriteError(RuntimeError):
    """Raised when an encoded image cannot be committed to its destination."""


class OutputExistsError(ImageWriteError):
    """Raised when atomic publication would replace an existing output."""


def output_is_lossy(path: Path) -> bool:
    """Return whether an M1-supported output path uses lossy JPEG encoding."""
    extension = path.suffix.lower()
    if extension == ".png":
        return False
    if extension in {".jpg", ".jpeg"}:
        return True
    displayed_extension = path.suffix or "<none>"
    raise UnsupportedImageFormatError(
        f"unsupported output extension '{displayed_extension}'; expected one of: .jpeg, .jpg, .png"
    )


def _remove_temporary_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _write_pillow_image_atomic(
    encoded: Image.Image,
    destination: Path,
    *,
    image_format: str,
    replace: bool,
) -> None:
    if not destination.parent.is_dir():
        raise ImageWriteError(
            f"output directory for '{destination.name}' does not exist or is not a directory"
        )
    if destination.exists() and not replace:
        raise OutputExistsError(f"output '{destination.name}' already exists")

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
    except OSError as error:
        raise ImageWriteError(
            f"could not create temporary output for '{destination.name}'"
        ) from error

    temporary_path = Path(temporary_name)
    try:
        os.close(descriptor)

        if image_format == "JPEG":
            encoded.save(temporary_path, format=image_format, quality=95, subsampling=0)
        else:
            encoded.save(temporary_path, format=image_format)

        with Image.open(temporary_path) as verification:
            verification.verify()

        if destination.exists() and not replace:
            raise OutputExistsError(f"output '{destination.name}' already exists")
        os.replace(temporary_path, destination)
    except OutputExistsError:
        _remove_temporary_file(temporary_path)
        raise
    except (OSError, ValueError) as error:
        _remove_temporary_file(temporary_path)
        raise ImageWriteError(f"could not write output '{destination.name}'") from error


def write_image_atomic(
    image: ImageData,
    destination: Path,
    *,
    replace: bool = False,
) -> None:
    """Encode an image and atomically publish it at ``destination``.

    PNG preserves the optional alpha channel. JPEG is lossy and rejects
    alpha-bearing input. The destination is never changed until encoding and
    verification have completed successfully.
    """
    lossy = output_is_lossy(destination)
    if lossy and image.has_alpha:
        raise ImageWriteError("JPEG output cannot preserve the image alpha channel")

    if image.alpha is None:
        pixels = image.rgb
        mode = "RGB"
    else:
        pixels = cast(UInt8Array, np.dstack((image.rgb, image.alpha)))
        mode = "RGBA"

    image_format = "JPEG" if lossy else "PNG"
    with Image.fromarray(pixels, mode=mode) as encoded:
        _write_pillow_image_atomic(
            encoded,
            destination,
            image_format=image_format,
            replace=replace,
        )


def write_mask_atomic(
    mask: BinaryMask,
    destination: Path,
    *,
    replace: bool = False,
) -> None:
    """Serialize a binary mask as a 0/255 PNG and publish it atomically."""
    if destination.suffix.lower() != ".png":
        displayed_extension = destination.suffix or "<none>"
        raise UnsupportedImageFormatError(
            f"unsupported saved-mask extension '{displayed_extension}'; expected .png"
        )

    with Image.fromarray(mask_to_uint8(mask), mode="L") as encoded:
        _write_pillow_image_atomic(
            encoded,
            destination,
            image_format="PNG",
            replace=replace,
        )
