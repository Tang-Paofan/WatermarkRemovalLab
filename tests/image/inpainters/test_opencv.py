"""Tests for the OpenCV inpainting adapter."""

from typing import cast

import cv2
import numpy as np
import pytest

from watermark_removal_lab.common import (
    BinaryMask,
    DataContractError,
    ImageData,
    UInt8Array,
)
from watermark_removal_lab.image import (
    OpenCVInpaintError,
    OpenCVInpaintMethod,
    inpaint_opencv,
)


def test_inpaint_opencv_converts_rgb_and_mask_for_telea(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    alpha = np.array([[64, 192]], dtype=np.uint8)
    mask_data = np.array([[False, True]], dtype=np.bool_)
    image = ImageData(rgb, alpha)
    mask = BinaryMask(mask_data)

    def fake_inpaint(
        source: UInt8Array,
        inpaint_mask: UInt8Array,
        radius: float,
        flags: int,
    ) -> UInt8Array:
        assert np.array_equal(source, [[[30, 20, 10], [60, 50, 40]]])
        assert source.flags.writeable
        assert np.array_equal(inpaint_mask, [[0, 255]])
        assert inpaint_mask.dtype == np.uint8
        assert radius == 4.5
        assert flags == cv2.INPAINT_TELEA
        return np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)

    monkeypatch.setattr(cv2, "inpaint", fake_inpaint)

    result = inpaint_opencv(
        image,
        mask,
        method=OpenCVInpaintMethod.TELEA,
        radius=4.5,
    )

    assert np.array_equal(result.rgb, [[[1, 2, 3], [4, 5, 6]]])
    assert result.alpha is not None
    assert np.array_equal(result.alpha, alpha)
    assert np.array_equal(image.rgb, rgb)
    assert np.array_equal(mask.data, mask_data)


def test_inpaint_opencv_maps_navier_stokes_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = ImageData(np.zeros((1, 1, 3), dtype=np.uint8))
    mask = BinaryMask(np.ones((1, 1), dtype=np.bool_))

    def fake_inpaint(
        source: UInt8Array,
        inpaint_mask: UInt8Array,
        radius: float,
        flags: int,
    ) -> UInt8Array:
        del inpaint_mask, radius
        assert flags == cv2.INPAINT_NS
        return source.copy()

    monkeypatch.setattr(cv2, "inpaint", fake_inpaint)

    result = inpaint_opencv(
        image,
        mask,
        method=OpenCVInpaintMethod.NAVIER_STOKES,
        radius=1,
    )

    assert np.array_equal(result.rgb, image.rgb)


@pytest.mark.parametrize(
    "method",
    [OpenCVInpaintMethod.TELEA, OpenCVInpaintMethod.NAVIER_STOKES],
)
def test_inpaint_opencv_runs_both_algorithms_on_synthetic_data(
    method: OpenCVInpaintMethod,
) -> None:
    rgb = np.zeros((9, 9, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(9, dtype=np.uint8)
    rgb[:, :, 1] = np.arange(9, dtype=np.uint8)[:, None]
    rgb[:, :, 2] = 80
    alpha = np.arange(81, dtype=np.uint8).reshape(9, 9)
    mask_data = np.zeros((9, 9), dtype=np.bool_)
    mask_data[4, 4] = True
    image = ImageData(rgb, alpha)
    mask = BinaryMask(mask_data)

    result = inpaint_opencv(image, mask, method=method, radius=1.0)

    assert result.spatial_shape == image.spatial_shape
    assert result.rgb.dtype == np.uint8
    assert result.alpha is not None
    assert np.array_equal(result.alpha, alpha)
    assert np.array_equal(image.rgb, rgb)
    assert np.array_equal(mask.data, mask_data)


def test_inpaint_opencv_bypasses_backend_for_an_empty_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = ImageData(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
    mask = BinaryMask(np.zeros((2, 2), dtype=np.bool_))

    def fail_if_called(*args: object, **kwargs: object) -> UInt8Array:
        del args, kwargs
        pytest.fail("OpenCV must not run for an empty mask")

    monkeypatch.setattr(cv2, "inpaint", fail_if_called)

    result = inpaint_opencv(image, mask)

    assert np.array_equal(result.rgb, image.rgb)
    assert result.rgb is not image.rgb


def test_inpaint_opencv_rejects_an_invalid_method() -> None:
    image = ImageData(np.zeros((1, 1, 3), dtype=np.uint8))
    mask = BinaryMask(np.ones((1, 1), dtype=np.bool_))

    with pytest.raises(DataContractError, match="must be an OpenCVInpaintMethod"):
        inpaint_opencv(
            image,
            mask,
            method=cast(OpenCVInpaintMethod, "telea"),
        )


@pytest.mark.parametrize(
    ("radius", "message"),
    [
        (True, "must be a real number"),
        ("3", "must be a real number"),
        (0, "must be positive"),
        (-1.0, "must be positive"),
        (float("inf"), "must be finite"),
        (float("nan"), "must be finite"),
    ],
)
def test_inpaint_opencv_rejects_an_invalid_radius(
    radius: object,
    message: str,
) -> None:
    image = ImageData(np.zeros((1, 1, 3), dtype=np.uint8))
    mask = BinaryMask(np.ones((1, 1), dtype=np.bool_))

    with pytest.raises(DataContractError, match=message):
        inpaint_opencv(image, mask, radius=cast(float, radius))


def test_inpaint_opencv_rejects_a_mismatched_mask() -> None:
    image = ImageData(np.zeros((2, 3, 3), dtype=np.uint8))
    mask = BinaryMask(np.zeros((3, 2), dtype=np.bool_))

    with pytest.raises(DataContractError, match=r"mask shape .* must match image shape"):
        inpaint_opencv(image, mask)


def test_inpaint_opencv_translates_backend_failure_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = ImageData(np.zeros((1, 1, 3), dtype=np.uint8))
    mask = BinaryMask(np.ones((1, 1), dtype=np.bool_))

    def fail_inpaint(*args: object, **kwargs: object) -> UInt8Array:
        del args, kwargs
        raise cv2.error("backend failed")

    monkeypatch.setattr(cv2, "inpaint", fail_inpaint)

    with pytest.raises(OpenCVInpaintError, match="OpenCV inpainting failed") as captured:
        inpaint_opencv(image, mask)

    assert isinstance(captured.value.__cause__, cv2.error)
