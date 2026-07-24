"""Integration tests for the single-image removal application service."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

import watermark_removal_lab.application.image_removal as image_removal_module
from watermark_removal_lab.application import (
    BoxMaskSource,
    ImageRemovalError,
    ImageRemovalInputError,
    ImageRemovalOutputError,
    ImageRemovalProcessingError,
    ImageRemovalRequest,
    ImageRemovalStatus,
    MaskFileSource,
    OverwritePolicy,
    build_failed_image_removal_result,
    remove_image,
)
from watermark_removal_lab.common import Box, DataContractError, ImageData, UInt8Array
from watermark_removal_lab.image import OpenCVInpaintError, OpenCVInpaintMethod, read_image
from watermark_removal_lab.image.output import ImageWriteError


def _save_rgb(path: Path, pixels: UInt8Array) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def _save_rgba(path: Path, rgb: UInt8Array, alpha: UInt8Array) -> None:
    rgba = np.dstack((rgb, alpha)).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _base_request(tmp_path: Path) -> ImageRemovalRequest:
    input_path = tmp_path / "input.png"
    _save_rgb(input_path, np.full((5, 5, 3), 80, dtype=np.uint8))
    return ImageRemovalRequest(
        input_path=input_path,
        output_path=tmp_path / "output.png",
        mask_source=BoxMaskSource(Box.from_xywh(x=2, y=2, width=1, height=1)),
    )


def test_remove_image_box_pipeline_preserves_png_exterior_and_alpha(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    rgb = np.zeros((9, 9, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(9, dtype=np.uint8)
    rgb[:, :, 1] = np.arange(9, dtype=np.uint8)[:, None]
    rgb[:, :, 2] = 90
    alpha = np.arange(81, dtype=np.uint8).reshape(9, 9)
    _save_rgba(input_path, rgb, alpha)
    request = ImageRemovalRequest(
        input_path=input_path,
        output_path=output_path,
        mask_source=BoxMaskSource(Box.from_xywh(x=3, y=3, width=3, height=3)),
        radius=1.0,
    )

    result = remove_image(request)

    decoded = read_image(output_path)
    selected = np.zeros((9, 9), dtype=np.bool_)
    selected[3:6, 3:6] = True
    assert np.array_equal(decoded.rgb[~selected], rgb[~selected])
    assert decoded.alpha is not None
    assert np.array_equal(decoded.alpha, alpha)
    assert np.array_equal(read_image(input_path).rgb, rgb)
    assert result.status is ImageRemovalStatus.SUCCEEDED
    assert result.width == 9
    assert result.height == 9
    assert result.selected_pixels == 9
    assert result.mask_threshold is None
    assert result.duration_ms >= 0
    assert not result.lossy_output
    assert result.warnings == ()
    assert result.to_dict()["options"] == {
        "radius": 1.0,
        "dilate": 0,
        "mask_threshold": None,
    }


def test_remove_image_external_mask_dilates_and_saves_final_mask(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    mask_path = tmp_path / "source-mask.png"
    saved_mask_path = tmp_path / "final-mask.png"
    _save_rgb(input_path, np.full((7, 7, 3), 100, dtype=np.uint8))
    intensity = np.zeros((7, 7), dtype=np.uint8)
    intensity[3, 3] = 128
    Image.fromarray(intensity, mode="L").save(mask_path)
    request = ImageRemovalRequest(
        input_path=input_path,
        output_path=output_path,
        mask_source=MaskFileSource(mask_path, threshold=127),
        method=OpenCVInpaintMethod.NAVIER_STOKES,
        radius=1,
        dilation_radius=1,
        save_mask_path=saved_mask_path,
    )

    result = remove_image(request)

    with Image.open(saved_mask_path) as encoded:
        saved_mask = np.asarray(encoded.convert("L"), dtype=np.uint8)
    assert set(np.unique(saved_mask)) == {np.uint8(0), np.uint8(255)}
    assert np.count_nonzero(saved_mask) == 5
    assert result.method is OpenCVInpaintMethod.NAVIER_STOKES
    assert result.mask_threshold == 127
    assert result.selected_pixels == 5


def test_remove_image_empty_mask_is_a_successful_no_op(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "output.png"
    rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    _save_rgb(input_path, rgb)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(mask_path)

    result = remove_image(
        ImageRemovalRequest(
            input_path=input_path,
            output_path=output_path,
            mask_source=MaskFileSource(mask_path),
        )
    )

    assert np.array_equal(read_image(output_path).rgb, rgb)
    assert result.selected_pixels == 0
    assert result.warnings == ("empty_mask",)


def test_remove_image_reports_full_frame_and_lossy_warnings(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    _save_rgb(input_path, np.full((4, 4, 3), 120, dtype=np.uint8))
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8), mode="L").save(mask_path)

    result = remove_image(
        ImageRemovalRequest(
            input_path=input_path,
            output_path=tmp_path / "output.jpg",
            mask_source=MaskFileSource(mask_path),
        )
    )

    assert result.lossy_output
    assert result.warnings == ("full_frame_mask", "lossy_output")


def test_remove_image_skip_returns_without_reading_input(tmp_path: Path) -> None:
    output_path = tmp_path / "output.png"
    output_path.write_bytes(b"existing")
    request = ImageRemovalRequest(
        input_path=tmp_path / "missing.png",
        output_path=output_path,
        mask_source=BoxMaskSource(Box.from_xywh(x=0, y=0, width=1, height=1)),
        overwrite=OverwritePolicy.SKIP,
    )

    result = remove_image(request)

    assert result.status is ImageRemovalStatus.SKIPPED
    assert result.width is None
    assert result.height is None
    assert result.selected_pixels is None
    assert result.warnings == ("output_exists",)
    assert output_path.read_bytes() == b"existing"


def test_remove_image_skip_when_saved_mask_exists(tmp_path: Path) -> None:
    request = _base_request(tmp_path)
    saved_mask = tmp_path / "saved.png"
    saved_mask.write_bytes(b"existing")

    result = remove_image(
        replace(
            request,
            save_mask_path=saved_mask,
            overwrite=OverwritePolicy.SKIP,
        )
    )

    assert result.status is ImageRemovalStatus.SKIPPED
    assert not request.output_path.exists()


def test_remove_image_rejects_existing_output_by_default(tmp_path: Path) -> None:
    request = _base_request(tmp_path)
    request.output_path.write_bytes(b"existing")

    with pytest.raises(ImageRemovalInputError, match="already exists") as captured:
        remove_image(request)

    assert captured.value.code == "output_exists"
    assert request.output_path.read_bytes() == b"existing"


def test_remove_image_replaces_existing_output(tmp_path: Path) -> None:
    request = _base_request(tmp_path)
    request.output_path.write_bytes(b"existing")

    result = remove_image(replace(request, overwrite=OverwritePolicy.REPLACE))

    assert result.status is ImageRemovalStatus.SUCCEEDED
    assert read_image(request.output_path).spatial_shape == (5, 5)


@pytest.mark.parametrize(
    ("transform", "code"),
    [
        (
            lambda request: replace(request, output_path=request.input_path),
            "in_place_output",
        ),
        (
            lambda request: replace(
                request,
                mask_source=MaskFileSource(request.input_path),
                output_path=request.input_path,
            ),
            "in_place_output",
        ),
        (
            lambda request: replace(request, save_mask_path=request.input_path),
            "in_place_output",
        ),
        (
            lambda request: replace(request, save_mask_path=request.output_path),
            "duplicate_output",
        ),
        (
            lambda request: replace(
                request,
                output_path=request.output_path.parent / "missing" / "output.png",
            ),
            "invalid_output_directory",
        ),
        (
            lambda request: replace(request, output_path=request.output_path.with_suffix(".webp")),
            "unsupported_output_format",
        ),
        (
            lambda request: replace(
                request,
                save_mask_path=request.output_path.with_name("mask.jpg"),
            ),
            "unsupported_output_format",
        ),
    ],
)
def test_remove_image_rejects_unsafe_output_configuration(
    tmp_path: Path,
    transform: Callable[[ImageRemovalRequest], ImageRemovalRequest],
    code: str,
) -> None:
    request = _base_request(tmp_path)
    transformed = transform(request)

    with pytest.raises(ImageRemovalInputError) as captured:
        remove_image(transformed)

    assert captured.value.code == code


def test_remove_image_rejects_a_directory_as_the_output_path(tmp_path: Path) -> None:
    request = _base_request(tmp_path)
    request.output_path.mkdir()

    with pytest.raises(ImageRemovalInputError) as captured:
        remove_image(replace(request, overwrite=OverwritePolicy.SKIP))

    assert captured.value.code == "invalid_output_path"


@pytest.mark.parametrize(
    "request_transform",
    [
        lambda request: replace(request, input_path=cast(Path, "input.png")),
        lambda request: replace(request, output_path=cast(Path, "output.png")),
        lambda request: replace(request, mask_source=cast(BoxMaskSource, object())),
        lambda request: replace(
            request,
            mask_source=BoxMaskSource(cast(Box, object())),
        ),
        lambda request: replace(
            request,
            mask_source=MaskFileSource(cast(Path, "mask.png")),
        ),
        lambda request: replace(
            request,
            method=cast(OpenCVInpaintMethod, "telea"),
        ),
        lambda request: replace(
            request,
            overwrite=cast(OverwritePolicy, "error"),
        ),
        lambda request: replace(request, save_mask_path=cast(Path, "mask.png")),
    ],
)
def test_remove_image_rejects_invalid_request_contract_types(
    tmp_path: Path,
    request_transform: Callable[[ImageRemovalRequest], ImageRemovalRequest],
) -> None:
    request = _base_request(tmp_path)
    transformed = request_transform(request)

    with pytest.raises(ImageRemovalInputError):
        remove_image(transformed)


@pytest.mark.parametrize("radius", [True, "3", 0, -1.0, float("inf"), float("nan")])
def test_remove_image_rejects_invalid_radius(tmp_path: Path, radius: object) -> None:
    request = replace(_base_request(tmp_path), radius=cast(float, radius))

    with pytest.raises(ImageRemovalInputError) as captured:
        remove_image(request)

    assert captured.value.code == "invalid_radius"


@pytest.mark.parametrize(
    "request_transform",
    [
        lambda request, tmp_path: replace(
            request,
            mask_source=MaskFileSource(tmp_path / "missing.png"),
        ),
        lambda request, tmp_path: replace(
            request,
            mask_source=MaskFileSource(tmp_path / "mask.webp"),
        ),
        lambda request, tmp_path: replace(
            request,
            mask_source=MaskFileSource(tmp_path / "small.png"),
        ),
        lambda request, tmp_path: replace(
            request,
            mask_source=MaskFileSource(tmp_path / "threshold.png", threshold=256),
        ),
        lambda request, tmp_path: replace(request, dilation_radius=-1),
    ],
)
def test_remove_image_translates_invalid_mask_configuration(
    tmp_path: Path,
    request_transform: Callable[[ImageRemovalRequest, Path], ImageRemovalRequest],
) -> None:
    request = _base_request(tmp_path)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(tmp_path / "small.png")
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8), mode="L").save(tmp_path / "threshold.png")
    transformed = request_transform(request, tmp_path)

    with pytest.raises(ImageRemovalInputError) as captured:
        remove_image(transformed)

    assert captured.value.code == "invalid_mask"
    assert captured.value.__cause__ is not None


def test_remove_image_translates_image_read_failure(tmp_path: Path) -> None:
    request = ImageRemovalRequest(
        input_path=tmp_path / "missing.png",
        output_path=tmp_path / "output.png",
        mask_source=BoxMaskSource(Box.from_xywh(x=0, y=0, width=1, height=1)),
    )

    with pytest.raises(ImageRemovalOutputError) as captured:
        remove_image(request)

    assert captured.value.code == "image_read_failed"
    assert captured.value.__cause__ is not None


def test_remove_image_rejects_an_unsupported_input_format(tmp_path: Path) -> None:
    request = ImageRemovalRequest(
        input_path=tmp_path / "input.webp",
        output_path=tmp_path / "output.png",
        mask_source=BoxMaskSource(Box.from_xywh(x=0, y=0, width=1, height=1)),
    )

    with pytest.raises(ImageRemovalInputError) as captured:
        remove_image(request)

    assert captured.value.code == "unsupported_input_format"


def test_remove_image_rejects_alpha_bearing_jpeg_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _save_rgba(
        input_path,
        np.zeros((3, 3, 3), dtype=np.uint8),
        np.full((3, 3), 255, dtype=np.uint8),
    )
    request = ImageRemovalRequest(
        input_path=input_path,
        output_path=tmp_path / "output.jpg",
        mask_source=BoxMaskSource(Box.from_xywh(x=1, y=1, width=1, height=1)),
    )

    with pytest.raises(ImageRemovalInputError) as captured:
        remove_image(request)

    assert captured.value.code == "alpha_not_supported"


def test_remove_image_translates_backend_data_contract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _base_request(tmp_path)

    def fail_inpaint(*args: object, **kwargs: object) -> ImageData:
        del args, kwargs
        raise DataContractError("invalid backend options")

    monkeypatch.setattr(image_removal_module, "inpaint_opencv", fail_inpaint)

    with pytest.raises(ImageRemovalInputError) as captured:
        remove_image(request)

    assert captured.value.code == "invalid_inpaint_options"
    assert isinstance(captured.value.__cause__, DataContractError)


def test_remove_image_translates_backend_processing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _base_request(tmp_path)

    def fail_inpaint(*args: object, **kwargs: object) -> ImageData:
        del args, kwargs
        raise OpenCVInpaintError("backend failed")

    monkeypatch.setattr(image_removal_module, "inpaint_opencv", fail_inpaint)

    with pytest.raises(ImageRemovalProcessingError) as captured:
        remove_image(request)

    assert captured.value.code == "inpaint_failed"
    assert isinstance(captured.value.__cause__, OpenCVInpaintError)


def test_remove_image_translates_output_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _base_request(tmp_path)

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ImageWriteError("publish failed")

    monkeypatch.setattr(image_removal_module, "write_image_atomic", fail_write)

    with pytest.raises(ImageRemovalOutputError) as captured:
        remove_image(request)

    assert captured.value.code == "output_write_failed"
    assert isinstance(captured.value.__cause__, ImageWriteError)


def test_build_failed_image_removal_result_is_json_compatible(tmp_path: Path) -> None:
    request = _base_request(tmp_path)
    error = ImageRemovalError("simulated failure", code="simulated")

    result = build_failed_image_removal_result(request, error, duration_ms=12.5)
    payload = result.to_dict()

    assert result.status is ImageRemovalStatus.FAILED
    assert payload["schema_version"] == 1
    assert payload["item_id"] is None
    assert payload["status"] == "failed"
    assert payload["error_code"] == "simulated"
    assert payload["error_message"] == "simulated failure"
    assert payload["duration_ms"] == 12.5
