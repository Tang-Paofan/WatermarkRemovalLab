"""Tests for mask-constrained image compositing."""

import numpy as np
import pytest

from watermark_removal_lab.common import BinaryMask, DataContractError, ImageData
from watermark_removal_lab.image import composite_masked


def test_composite_masked_replaces_only_selected_rgb_and_preserves_alpha() -> None:
    original_rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    original_alpha = np.array([[0, 64, 128], [192, 224, 255]], dtype=np.uint8)
    candidate_rgb = np.full((2, 3, 3), 250, dtype=np.uint8)
    candidate_alpha = np.zeros((2, 3), dtype=np.uint8)
    mask_data = np.array(
        [[False, True, False], [True, False, True]],
        dtype=np.bool_,
    )
    original = ImageData(original_rgb, original_alpha)
    candidate = ImageData(candidate_rgb, candidate_alpha)
    mask = BinaryMask(mask_data)
    expected_rgb = original.rgb.copy()
    expected_rgb[mask.data] = candidate.rgb[mask.data]

    result = composite_masked(original, candidate, mask)

    assert np.array_equal(result.rgb, expected_rgb)
    assert result.alpha is not None
    assert np.array_equal(result.alpha, original_alpha)
    assert result.rgb is not original.rgb
    assert result.alpha is not original.alpha
    assert not result.rgb.flags.writeable
    assert not result.alpha.flags.writeable
    assert np.array_equal(original.rgb, original_rgb)
    assert np.array_equal(candidate.rgb, candidate_rgb)
    assert np.array_equal(mask.data, mask_data)


def test_composite_masked_empty_mask_returns_an_independent_copy() -> None:
    original = ImageData(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
    candidate = ImageData(np.full((2, 2, 3), 255, dtype=np.uint8))

    result = composite_masked(
        original,
        candidate,
        BinaryMask(np.zeros((2, 2), dtype=np.bool_)),
    )

    assert np.array_equal(result.rgb, original.rgb)
    assert result.rgb is not original.rgb
    assert result.alpha is None


def test_composite_masked_rejects_a_mismatched_mask() -> None:
    original = ImageData(np.zeros((2, 3, 3), dtype=np.uint8))
    candidate = ImageData(np.zeros((2, 3, 3), dtype=np.uint8))
    mask = BinaryMask(np.zeros((3, 2), dtype=np.bool_))

    with pytest.raises(DataContractError, match=r"mask shape .* must match image shape"):
        composite_masked(original, candidate, mask)


def test_composite_masked_rejects_a_mismatched_candidate() -> None:
    original = ImageData(np.zeros((2, 3, 3), dtype=np.uint8))
    candidate = ImageData(np.zeros((3, 2, 3), dtype=np.uint8))
    mask = BinaryMask(np.zeros((2, 3), dtype=np.bool_))

    with pytest.raises(DataContractError, match=r"candidate shape .* must match original shape"):
        composite_masked(original, candidate, mask)
