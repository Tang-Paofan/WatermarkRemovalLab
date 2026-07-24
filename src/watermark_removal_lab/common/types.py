"""Canonical image, alpha, mask, and coordinate contracts."""

from dataclasses import dataclass
from typing import Self, TypeAlias

import numpy as np
from numpy.typing import NDArray

UInt8Array: TypeAlias = NDArray[np.uint8]
BoolArray: TypeAlias = NDArray[np.bool_]


class DataContractError(ValueError):
    """Raised when core image data violates a canonical contract."""


def _readonly_uint8_copy(array: UInt8Array) -> UInt8Array:
    copied = array.copy(order="C")
    copied.flags.writeable = False
    return copied


def _readonly_bool_copy(array: BoolArray) -> BoolArray:
    copied = array.copy(order="C")
    copied.flags.writeable = False
    return copied


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise DataContractError(f"{name} must be an integer, not a boolean")
    if not isinstance(value, int):
        raise DataContractError(f"{name} must be an integer")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class ImageData:
    """Validated RGB image with an optional separate alpha channel.

    The RGB data must be a non-empty ``uint8`` array shaped ``H x W x 3``.
    Alpha, when present, must be a ``uint8`` array shaped ``H x W``. The
    constructor takes read-only defensive copies and never mutates its inputs.
    """

    rgb: UInt8Array
    alpha: UInt8Array | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rgb, np.ndarray):
            raise DataContractError("rgb must be a NumPy array")
        if self.rgb.dtype != np.uint8:
            raise DataContractError(f"rgb dtype must be uint8, got {self.rgb.dtype}")
        if self.rgb.ndim != 3:
            raise DataContractError(f"rgb must have 3 dimensions, got {self.rgb.ndim}")
        if self.rgb.shape[2] != 3:
            raise DataContractError(f"rgb must have 3 channels, got {self.rgb.shape[2]}")
        if self.rgb.shape[0] == 0:
            raise DataContractError("rgb height must be positive")
        if self.rgb.shape[1] == 0:
            raise DataContractError("rgb width must be positive")

        rgb = _readonly_uint8_copy(self.rgb)
        object.__setattr__(self, "rgb", rgb)

        if self.alpha is None:
            return
        if not isinstance(self.alpha, np.ndarray):
            raise DataContractError("alpha must be a NumPy array")
        if self.alpha.dtype != np.uint8:
            raise DataContractError(f"alpha dtype must be uint8, got {self.alpha.dtype}")
        if self.alpha.ndim != 2:
            raise DataContractError(f"alpha must have 2 dimensions, got {self.alpha.ndim}")
        if self.alpha.shape != rgb.shape[:2]:
            raise DataContractError(
                f"alpha shape {self.alpha.shape} must match rgb shape {rgb.shape[:2]}"
            )

        object.__setattr__(self, "alpha", _readonly_uint8_copy(self.alpha))

    @property
    def height(self) -> int:
        """Return the image height in pixels."""
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        """Return the image width in pixels."""
        return int(self.rgb.shape[1])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return (self.height, self.width)

    @property
    def has_alpha(self) -> bool:
        """Return whether a separate alpha channel is present."""
        return self.alpha is not None


@dataclass(frozen=True, slots=True, eq=False)
class BinaryMask:
    """Validated binary mask stored as a read-only defensive copy.

    Mask data must be a non-empty ``bool`` array shaped ``H x W``. Empty and
    full-frame selections are both valid.
    """

    data: BoolArray

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise DataContractError("mask must be a NumPy array")
        if self.data.dtype != np.bool_:
            raise DataContractError(f"mask dtype must be bool, got {self.data.dtype}")
        if self.data.ndim != 2:
            raise DataContractError(f"mask must have 2 dimensions, got {self.data.ndim}")
        if self.data.shape[0] == 0:
            raise DataContractError("mask height must be positive")
        if self.data.shape[1] == 0:
            raise DataContractError("mask width must be positive")

        object.__setattr__(self, "data", _readonly_bool_copy(self.data))

    @property
    def height(self) -> int:
        """Return the mask height in pixels."""
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        """Return the mask width in pixels."""
        return int(self.data.shape[1])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return (self.height, self.width)

    @property
    def selected_pixels(self) -> int:
        """Return the number of selected pixels."""
        return int(np.count_nonzero(self.data))

    @property
    def is_empty(self) -> bool:
        """Return whether no pixel is selected."""
        return not bool(self.data.any())

    @property
    def is_full(self) -> bool:
        """Return whether every pixel is selected."""
        return bool(self.data.all())

    def validate_for(self, image: ImageData) -> None:
        """Raise when the mask and image spatial dimensions differ."""
        if self.spatial_shape != image.spatial_shape:
            raise DataContractError(
                f"mask shape {self.spatial_shape} must match image shape {image.spatial_shape}"
            )


@dataclass(frozen=True, slots=True)
class Box:
    """Half-open image box using ``(x_min, y_min, x_max, y_max)`` coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        _require_int("x_min", self.x_min)
        _require_int("y_min", self.y_min)
        _require_int("x_max", self.x_max)
        _require_int("y_max", self.y_max)

        if self.x_min < 0:
            raise DataContractError("x_min must be non-negative")
        if self.y_min < 0:
            raise DataContractError("y_min must be non-negative")
        if self.x_max <= self.x_min:
            raise DataContractError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise DataContractError("y_max must be greater than y_min")

    @classmethod
    def from_xywh(cls, x: int, y: int, width: int, height: int) -> Self:
        """Create a half-open box from ``(x, y, width, height)`` values."""
        normalized_x = _require_int("x", x)
        normalized_y = _require_int("y", y)
        normalized_width = _require_int("width", width)
        normalized_height = _require_int("height", height)

        if normalized_width <= 0:
            raise DataContractError("width must be positive")
        if normalized_height <= 0:
            raise DataContractError("height must be positive")

        return cls(
            x_min=normalized_x,
            y_min=normalized_y,
            x_max=normalized_x + normalized_width,
            y_max=normalized_y + normalized_height,
        )

    @property
    def width(self) -> int:
        """Return the box width in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Return the box height in pixels."""
        return self.y_max - self.y_min

    def validate_within(self, *, image_width: int, image_height: int) -> None:
        """Raise unless the complete box is inside the given image dimensions."""
        normalized_width = _require_int("image_width", image_width)
        normalized_height = _require_int("image_height", image_height)

        if normalized_width <= 0:
            raise DataContractError("image_width must be positive")
        if normalized_height <= 0:
            raise DataContractError("image_height must be positive")
        if self.x_max > normalized_width:
            raise DataContractError(
                f"box x_max {self.x_max} exceeds image width {normalized_width}"
            )
        if self.y_max > normalized_height:
            raise DataContractError(
                f"box y_max {self.y_max} exceeds image height {normalized_height}"
            )
