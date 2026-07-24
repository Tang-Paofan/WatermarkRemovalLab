"""Pure mask creation, refinement, and serialization utilities."""

from typing import cast

import numpy as np

from watermark_removal_lab.common.types import (
    BinaryMask,
    Box,
    DataContractError,
    UInt8Array,
)


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise DataContractError(f"{name} must be an integer, not a boolean")
    if not isinstance(value, int):
        raise DataContractError(f"{name} must be an integer")
    return value


def box_to_mask(*, box: Box, image_width: int, image_height: int) -> BinaryMask:
    """Create a binary mask from a half-open box inside an image.

    The returned mask is shaped ``(image_height, image_width)`` and selects
    exactly the pixels in ``box``. Edge-touching boxes are valid; boxes outside
    the image raise :class:`DataContractError`.
    """
    box.validate_within(image_width=image_width, image_height=image_height)
    data = np.zeros((image_height, image_width), dtype=np.bool_)
    data[box.y_min : box.y_max, box.x_min : box.x_max] = True
    return BinaryMask(data)


def threshold_mask(intensity: UInt8Array, *, threshold: int = 127) -> BinaryMask:
    """Select mask-intensity pixels strictly greater than ``threshold``.

    ``intensity`` must be a non-empty ``uint8`` array shaped ``H x W``.
    ``threshold`` must be an integer from 0 through 255. The input is never
    modified.
    """
    if not isinstance(intensity, np.ndarray):
        raise DataContractError("mask intensity must be a NumPy array")
    if intensity.dtype != np.uint8:
        raise DataContractError(f"mask intensity dtype must be uint8, got {intensity.dtype}")
    if intensity.ndim != 2:
        raise DataContractError(f"mask intensity must have 2 dimensions, got {intensity.ndim}")
    if intensity.shape[0] == 0:
        raise DataContractError("mask intensity height must be positive")
    if intensity.shape[1] == 0:
        raise DataContractError("mask intensity width must be positive")

    normalized_threshold = _require_int("threshold", threshold)
    if normalized_threshold < 0 or normalized_threshold > 255:
        raise DataContractError("threshold must be between 0 and 255")

    return BinaryMask(intensity > normalized_threshold)


def dilate_mask(mask: BinaryMask, *, radius: int) -> BinaryMask:
    """Dilate a mask with a clipped elliptical kernel.

    ``radius`` is a non-negative pixel radius. A positive radius uses a
    ``(2 * radius + 1)`` square footprint and selects offsets inside the
    corresponding circle. Expansion is clipped at image boundaries.
    """
    normalized_radius = _require_int("radius", radius)
    if normalized_radius < 0:
        raise DataContractError("radius must be non-negative")
    if normalized_radius == 0:
        return BinaryMask(mask.data)

    diameter = 2 * normalized_radius + 1
    padded = np.pad(mask.data, normalized_radius, mode="constant", constant_values=False)
    expanded = np.zeros_like(mask.data)
    squared_radius = normalized_radius * normalized_radius

    for row_offset in range(diameter):
        delta_y = row_offset - normalized_radius
        for column_offset in range(diameter):
            delta_x = column_offset - normalized_radius
            if delta_x * delta_x + delta_y * delta_y > squared_radius:
                continue
            expanded |= padded[
                row_offset : row_offset + mask.height,
                column_offset : column_offset + mask.width,
            ]

    return BinaryMask(expanded)


def mask_to_uint8(mask: BinaryMask) -> UInt8Array:
    """Serialize a binary mask to a new ``uint8`` array containing only 0 and 255."""
    serialized = np.where(mask.data, np.uint8(255), np.uint8(0))
    return cast(UInt8Array, serialized)
