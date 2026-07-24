"""Tests for canonical image, alpha, mask, and box contracts."""

from typing import cast

import numpy as np
import pytest

from watermark_removal_lab.common import (
    BinaryMask,
    BoolArray,
    Box,
    DataContractError,
    ImageData,
    UInt8Array,
)


def make_rgb(*, height: int = 2, width: int = 3) -> UInt8Array:
    """Create deterministic synthetic RGB data."""
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def test_image_data_owns_read_only_copies_and_reports_shape() -> None:
    rgb = make_rgb()
    alpha = np.arange(6, dtype=np.uint8).reshape(2, 3)
    expected_rgb = rgb.copy()
    expected_alpha = alpha.copy()

    image = ImageData(rgb=rgb, alpha=alpha)
    rgb.fill(0)
    alpha.fill(0)

    assert image.height == 2
    assert image.width == 3
    assert image.spatial_shape == (2, 3)
    assert image.has_alpha
    assert np.array_equal(image.rgb, expected_rgb)
    assert image.alpha is not None
    assert np.array_equal(image.alpha, expected_alpha)
    assert image.rgb.flags.c_contiguous
    assert not image.rgb.flags.writeable
    assert not image.alpha.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        image.rgb[0, 0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        image.alpha[0, 0] = 0


def test_image_data_accepts_an_image_without_alpha() -> None:
    image = ImageData(rgb=make_rgb())

    assert image.alpha is None
    assert not image.has_alpha


@pytest.mark.parametrize(
    ("rgb", "message"),
    [
        (object(), "rgb must be a NumPy array"),
        (np.zeros((2, 3, 3), dtype=np.int16), "rgb dtype must be uint8"),
        (np.zeros((2, 3), dtype=np.uint8), "rgb must have 3 dimensions"),
        (np.zeros((2, 3, 4), dtype=np.uint8), "rgb must have 3 channels"),
        (np.zeros((0, 3, 3), dtype=np.uint8), "rgb height must be positive"),
        (np.zeros((2, 0, 3), dtype=np.uint8), "rgb width must be positive"),
    ],
)
def test_image_data_rejects_invalid_rgb(rgb: object, message: str) -> None:
    with pytest.raises(DataContractError, match=message):
        ImageData(rgb=cast(UInt8Array, rgb))


@pytest.mark.parametrize(
    ("alpha", "message"),
    [
        (object(), "alpha must be a NumPy array"),
        (np.zeros((2, 3), dtype=np.int16), "alpha dtype must be uint8"),
        (np.zeros((2, 3, 1), dtype=np.uint8), "alpha must have 2 dimensions"),
        (np.zeros((3, 2), dtype=np.uint8), "alpha shape .* must match rgb shape"),
    ],
)
def test_image_data_rejects_invalid_alpha(alpha: object, message: str) -> None:
    with pytest.raises(DataContractError, match=message):
        ImageData(rgb=make_rgb(), alpha=cast(UInt8Array, alpha))


def test_binary_mask_owns_a_read_only_copy_and_reports_selection() -> None:
    source = np.array([[False, True, False], [True, False, False]], dtype=np.bool_)
    expected = source.copy()

    mask = BinaryMask(source)
    source.fill(True)

    assert mask.height == 2
    assert mask.width == 3
    assert mask.spatial_shape == (2, 3)
    assert mask.selected_pixels == 2
    assert not mask.is_empty
    assert not mask.is_full
    assert np.array_equal(mask.data, expected)
    assert mask.data.flags.c_contiguous
    assert not mask.data.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        mask.data[0, 0] = True


def test_empty_and_full_binary_masks_are_valid() -> None:
    empty = BinaryMask(np.zeros((2, 3), dtype=np.bool_))
    full = BinaryMask(np.ones((2, 3), dtype=np.bool_))

    assert empty.is_empty
    assert not empty.is_full
    assert full.is_full
    assert not full.is_empty


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (object(), "mask must be a NumPy array"),
        (np.zeros((2, 3), dtype=np.uint8), "mask dtype must be bool"),
        (np.zeros((2, 3, 1), dtype=np.bool_), "mask must have 2 dimensions"),
        (np.zeros((0, 3), dtype=np.bool_), "mask height must be positive"),
        (np.zeros((2, 0), dtype=np.bool_), "mask width must be positive"),
    ],
)
def test_binary_mask_rejects_invalid_data(data: object, message: str) -> None:
    with pytest.raises(DataContractError, match=message):
        BinaryMask(cast(BoolArray, data))


def test_binary_mask_validates_image_dimensions() -> None:
    image = ImageData(make_rgb())
    matching_mask = BinaryMask(np.zeros((2, 3), dtype=np.bool_))
    mismatched_mask = BinaryMask(np.zeros((3, 2), dtype=np.bool_))

    matching_mask.validate_for(image)
    with pytest.raises(DataContractError, match=r"mask shape .* must match image shape"):
        mismatched_mask.validate_for(image)


def test_box_reports_half_open_dimensions_and_allows_edge_contact() -> None:
    box = Box(x_min=1, y_min=2, x_max=5, y_max=7)

    assert box.width == 4
    assert box.height == 5
    box.validate_within(image_width=5, image_height=7)


def test_box_converts_xywh_to_half_open_coordinates() -> None:
    box = Box.from_xywh(x=10, y=20, width=120, height=40)

    assert box == Box(x_min=10, y_min=20, x_max=130, y_max=60)


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        ((True, 0, 1, 1), "x_min must be an integer, not a boolean"),
        ((0, 0.5, 1, 1), "y_min must be an integer"),
        ((-1, 0, 1, 1), "x_min must be non-negative"),
        ((0, -1, 1, 1), "y_min must be non-negative"),
        ((1, 0, 1, 1), "x_max must be greater than x_min"),
        ((0, 1, 1, 1), "y_max must be greater than y_min"),
    ],
)
def test_box_rejects_invalid_coordinates(
    coordinates: tuple[object, object, object, object],
    message: str,
) -> None:
    x_min, y_min, x_max, y_max = coordinates

    with pytest.raises(DataContractError, match=message):
        Box(
            x_min=cast(int, x_min),
            y_min=cast(int, y_min),
            x_max=cast(int, x_max),
            y_max=cast(int, y_max),
        )


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (0, 1, "width must be positive"),
        (1, -1, "height must be positive"),
    ],
)
def test_box_from_xywh_rejects_non_positive_size(
    width: int,
    height: int,
    message: str,
) -> None:
    with pytest.raises(DataContractError, match=message):
        Box.from_xywh(x=0, y=0, width=width, height=height)


@pytest.mark.parametrize(
    ("image_width", "image_height", "message"),
    [
        (True, 10, "image_width must be an integer, not a boolean"),
        (10, 2.5, "image_height must be an integer"),
        (0, 10, "image_width must be positive"),
        (10, 0, "image_height must be positive"),
        (3, 10, "box x_max 4 exceeds image width 3"),
        (10, 3, "box y_max 4 exceeds image height 3"),
    ],
)
def test_box_rejects_invalid_or_out_of_bounds_image_dimensions(
    image_width: object,
    image_height: object,
    message: str,
) -> None:
    box = Box(x_min=1, y_min=1, x_max=4, y_max=4)

    with pytest.raises(DataContractError, match=message):
        box.validate_within(
            image_width=cast(int, image_width),
            image_height=cast(int, image_height),
        )
