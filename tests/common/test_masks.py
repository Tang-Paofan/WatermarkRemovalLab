"""Tests for pure mask utilities."""

from typing import cast

import numpy as np
import pytest

from watermark_removal_lab.common import (
    BinaryMask,
    Box,
    DataContractError,
    UInt8Array,
    box_to_mask,
    dilate_mask,
    mask_to_uint8,
    threshold_mask,
)


def test_box_to_mask_selects_exact_half_open_region() -> None:
    mask = box_to_mask(
        box=Box(x_min=1, y_min=1, x_max=4, y_max=3),
        image_width=4,
        image_height=3,
    )

    expected = np.array(
        [
            [False, False, False, False],
            [False, True, True, True],
            [False, True, True, True],
        ],
        dtype=np.bool_,
    )
    assert np.array_equal(mask.data, expected)


def test_box_to_mask_rejects_a_box_outside_the_image() -> None:
    with pytest.raises(DataContractError, match="exceeds image width"):
        box_to_mask(
            box=Box(x_min=0, y_min=0, x_max=3, y_max=2),
            image_width=2,
            image_height=2,
        )


def test_threshold_mask_uses_strict_greater_than_without_mutating_input() -> None:
    intensity = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    original = intensity.copy()

    mask = threshold_mask(intensity, threshold=127)

    assert np.array_equal(mask.data, [[False, False, True, True]])
    assert np.array_equal(intensity, original)


@pytest.mark.parametrize(
    ("intensity", "message"),
    [
        (object(), "mask intensity must be a NumPy array"),
        (np.zeros((2, 2), dtype=np.int16), "mask intensity dtype must be uint8"),
        (np.zeros((2, 2, 1), dtype=np.uint8), "mask intensity must have 2 dimensions"),
        (np.zeros((0, 2), dtype=np.uint8), "mask intensity height must be positive"),
        (np.zeros((2, 0), dtype=np.uint8), "mask intensity width must be positive"),
    ],
)
def test_threshold_mask_rejects_invalid_intensity(intensity: object, message: str) -> None:
    with pytest.raises(DataContractError, match=message):
        threshold_mask(cast(UInt8Array, intensity))


@pytest.mark.parametrize(
    ("threshold", "message"),
    [
        (True, "threshold must be an integer, not a boolean"),
        (1.5, "threshold must be an integer"),
        (-1, "threshold must be between 0 and 255"),
        (256, "threshold must be between 0 and 255"),
    ],
)
def test_threshold_mask_rejects_invalid_threshold(threshold: object, message: str) -> None:
    with pytest.raises(DataContractError, match=message):
        threshold_mask(
            np.zeros((2, 2), dtype=np.uint8),
            threshold=cast(int, threshold),
        )


def test_dilate_mask_with_zero_radius_returns_an_independent_mask() -> None:
    source = BinaryMask(np.array([[False, True]], dtype=np.bool_))

    result = dilate_mask(source, radius=0)

    assert result is not source
    assert result.data is not source.data
    assert np.array_equal(result.data, source.data)


def test_dilate_mask_uses_an_elliptical_kernel_and_clips_edges() -> None:
    source = BinaryMask(
        np.array(
            [
                [True, False, False],
                [False, False, False],
                [False, False, False],
            ],
            dtype=np.bool_,
        )
    )

    result = dilate_mask(source, radius=1)

    expected = np.array(
        [
            [True, True, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=np.bool_,
    )
    assert np.array_equal(result.data, expected)
    assert np.count_nonzero(source.data) == 1


def test_dilate_mask_radius_two_uses_a_five_by_five_disk() -> None:
    source_data = np.zeros((5, 5), dtype=np.bool_)
    source_data[2, 2] = True

    result = dilate_mask(BinaryMask(source_data), radius=2)

    expected = np.array(
        [
            [False, False, True, False, False],
            [False, True, True, True, False],
            [True, True, True, True, True],
            [False, True, True, True, False],
            [False, False, True, False, False],
        ],
        dtype=np.bool_,
    )
    assert np.array_equal(result.data, expected)


@pytest.mark.parametrize(
    ("radius", "message"),
    [
        (True, "radius must be an integer, not a boolean"),
        (1.5, "radius must be an integer"),
        (-1, "radius must be non-negative"),
    ],
)
def test_dilate_mask_rejects_invalid_radius(radius: object, message: str) -> None:
    source = BinaryMask(np.zeros((2, 2), dtype=np.bool_))

    with pytest.raises(DataContractError, match=message):
        dilate_mask(source, radius=cast(int, radius))


def test_mask_to_uint8_serializes_only_zero_and_255() -> None:
    source = BinaryMask(np.array([[False, True], [True, False]], dtype=np.bool_))

    serialized = mask_to_uint8(source)
    serialized[0, 0] = 255

    assert serialized.dtype == np.uint8
    assert np.array_equal(serialized, [[255, 255], [255, 0]])
    assert not source.data[0, 0]
