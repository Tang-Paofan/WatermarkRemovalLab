"""Single-image watermark-removal application service."""

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import TypeAlias

from watermark_removal_lab.common import (
    BinaryMask,
    Box,
    DataContractError,
    box_to_mask,
    dilate_mask,
    threshold_mask,
)
from watermark_removal_lab.image import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageReadError,
    MaskReadError,
    OpenCVInpaintError,
    OpenCVInpaintMethod,
    UnsupportedImageFormatError,
    composite_masked,
    inpaint_opencv,
    read_image,
    read_mask_intensity,
)
from watermark_removal_lab.image.output import (
    ImageWriteError,
    output_is_lossy,
    write_image_atomic,
    write_mask_atomic,
)

RESULT_SCHEMA_VERSION = 1


class OverwritePolicy(StrEnum):
    """Supported existing-output policies."""

    ERROR = "error"
    SKIP = "skip"
    REPLACE = "replace"


class ImageRemovalStatus(StrEnum):
    """Machine-readable single-item statuses shared with future batches."""

    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BoxMaskSource:
    """Create the initial mask from a validated half-open image box."""

    box: Box


@dataclass(frozen=True, slots=True)
class MaskFileSource:
    """Create the initial mask from an external image intensity channel."""

    path: Path
    threshold: int = 127


ImageMaskSource: TypeAlias = BoxMaskSource | MaskFileSource


@dataclass(frozen=True, slots=True)
class ImageRemovalRequest:
    """Framework-neutral request for one M1 image-removal operation."""

    input_path: Path
    output_path: Path
    mask_source: ImageMaskSource
    method: OpenCVInpaintMethod = OpenCVInpaintMethod.TELEA
    radius: float = 3.0
    dilation_radius: int = 0
    save_mask_path: Path | None = None
    overwrite: OverwritePolicy = OverwritePolicy.ERROR


@dataclass(frozen=True, slots=True)
class ImageRemovalResult:
    """Structured result for one successful, skipped, or failed operation."""

    input_path: Path
    output_path: Path
    status: ImageRemovalStatus
    method: OpenCVInpaintMethod
    radius: float
    dilation_radius: int
    mask_threshold: int | None
    width: int | None
    height: int | None
    selected_pixels: int | None
    duration_ms: float
    lossy_output: bool
    warnings: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible result representation."""
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "item_id": None,
            "input": str(self.input_path),
            "output": str(self.output_path),
            "status": self.status.value,
            "method": self.method.value,
            "options": {
                "radius": self.radius,
                "dilate": self.dilation_radius,
                "mask_threshold": self.mask_threshold,
            },
            "width": self.width,
            "height": self.height,
            "selected_pixels": self.selected_pixels,
            "duration_ms": self.duration_ms,
            "lossy_output": self.lossy_output,
            "warnings": list(self.warnings),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


class ImageRemovalError(RuntimeError):
    """Base class for structured single-image service failures."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ImageRemovalInputError(ImageRemovalError):
    """Raised for invalid requests, masks, boxes, or preflight collisions."""


class ImageRemovalProcessingError(ImageRemovalError):
    """Raised when the configured inpainting backend fails."""


class ImageRemovalOutputError(ImageRemovalError):
    """Raised when input decoding or atomic output publication fails."""


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve()


def _mask_threshold(request: ImageRemovalRequest) -> int | None:
    if isinstance(request.mask_source, MaskFileSource):
        return request.mask_source.threshold
    return None


def _validate_request(request: ImageRemovalRequest) -> bool:
    if not isinstance(request.input_path, Path):
        raise ImageRemovalInputError("input path must be a pathlib.Path", code="invalid_path")
    if not isinstance(request.output_path, Path):
        raise ImageRemovalInputError("output path must be a pathlib.Path", code="invalid_path")
    if not isinstance(request.mask_source, (BoxMaskSource, MaskFileSource)):
        raise ImageRemovalInputError("mask source is not supported", code="invalid_mask_source")
    if isinstance(request.mask_source, BoxMaskSource) and not isinstance(
        request.mask_source.box, Box
    ):
        raise ImageRemovalInputError("box mask source is invalid", code="invalid_mask_source")
    if isinstance(request.mask_source, MaskFileSource) and not isinstance(
        request.mask_source.path, Path
    ):
        raise ImageRemovalInputError("mask path must be a pathlib.Path", code="invalid_path")
    if not isinstance(request.method, OpenCVInpaintMethod):
        raise ImageRemovalInputError("inpaint method is not supported", code="invalid_method")
    if not isinstance(request.overwrite, OverwritePolicy):
        raise ImageRemovalInputError("overwrite policy is not supported", code="invalid_overwrite")
    if request.save_mask_path is not None and not isinstance(request.save_mask_path, Path):
        raise ImageRemovalInputError("saved-mask path must be a pathlib.Path", code="invalid_path")
    if request.input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageRemovalInputError(
            f"unsupported input extension '{request.input_path.suffix or '<none>'}'",
            code="unsupported_input_format",
        )

    try:
        lossy_output = output_is_lossy(request.output_path)
        if request.save_mask_path is not None:
            if request.save_mask_path.suffix.lower() != ".png":
                raise UnsupportedImageFormatError("saved-mask output must use the .png extension")
    except UnsupportedImageFormatError as error:
        raise ImageRemovalInputError(str(error), code="unsupported_output_format") from error

    source_paths = [request.input_path]
    if isinstance(request.mask_source, MaskFileSource):
        source_paths.append(request.mask_source.path)
    destination_paths = [request.output_path]
    if request.save_mask_path is not None:
        destination_paths.append(request.save_mask_path)

    for destination in destination_paths:
        if any(_same_path(destination, source) for source in source_paths):
            raise ImageRemovalInputError(
                f"destination '{destination.name}' must not overwrite an input",
                code="in_place_output",
            )
        if destination.exists() and not destination.is_file():
            raise ImageRemovalInputError(
                f"destination '{destination.name}' is not a regular file path",
                code="invalid_output_path",
            )
        if not destination.parent.is_dir():
            raise ImageRemovalInputError(
                f"output directory for '{destination.name}' does not exist or is not a directory",
                code="invalid_output_directory",
            )

    if request.save_mask_path is not None and _same_path(
        request.output_path, request.save_mask_path
    ):
        raise ImageRemovalInputError(
            "image output and saved-mask output must use different paths",
            code="duplicate_output",
        )

    existing = tuple(path for path in destination_paths if path.exists())
    if existing and request.overwrite is OverwritePolicy.ERROR:
        raise ImageRemovalInputError(
            f"output '{existing[0].name}' already exists",
            code="output_exists",
        )
    return lossy_output


def _prepare_mask(request: ImageRemovalRequest, *, width: int, height: int) -> BinaryMask:
    try:
        if isinstance(request.mask_source, BoxMaskSource):
            mask = box_to_mask(
                box=request.mask_source.box,
                image_width=width,
                image_height=height,
            )
        else:
            intensity = read_mask_intensity(request.mask_source.path)
            mask = threshold_mask(intensity, threshold=request.mask_source.threshold)
            if mask.spatial_shape != (height, width):
                raise DataContractError(
                    f"mask shape {mask.spatial_shape} must match image shape {(height, width)}"
                )
        return dilate_mask(mask, radius=request.dilation_radius)
    except (DataContractError, MaskReadError, UnsupportedImageFormatError) as error:
        raise ImageRemovalInputError(str(error), code="invalid_mask") from error


def _normalized_radius(request: ImageRemovalRequest) -> float:
    if isinstance(request.radius, bool) or not isinstance(request.radius, (int, float)):
        raise ImageRemovalInputError(
            "inpaint radius must be a real number",
            code="invalid_radius",
        )
    radius = float(request.radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ImageRemovalInputError(
            "inpaint radius must be finite and positive",
            code="invalid_radius",
        )
    return radius


def _warnings_for(mask: BinaryMask, *, lossy_output: bool) -> tuple[str, ...]:
    warnings: list[str] = []
    if mask.is_empty:
        warnings.append("empty_mask")
    if mask.is_full:
        warnings.append("full_frame_mask")
    if lossy_output:
        warnings.append("lossy_output")
    return tuple(warnings)


def remove_image(request: ImageRemovalRequest) -> ImageRemovalResult:
    """Execute one complete M1 image-removal operation."""
    started = perf_counter()
    lossy_output = _validate_request(request)
    normalized_radius = _normalized_radius(request)
    destination_paths = (request.output_path, request.save_mask_path)
    if request.overwrite is OverwritePolicy.SKIP and any(
        path is not None and path.exists() for path in destination_paths
    ):
        return ImageRemovalResult(
            input_path=request.input_path,
            output_path=request.output_path,
            status=ImageRemovalStatus.SKIPPED,
            method=request.method,
            radius=normalized_radius,
            dilation_radius=request.dilation_radius,
            mask_threshold=_mask_threshold(request),
            width=None,
            height=None,
            selected_pixels=None,
            duration_ms=(perf_counter() - started) * 1000,
            lossy_output=lossy_output,
            warnings=("output_exists",),
        )

    try:
        image = read_image(request.input_path)
    except ImageReadError as error:
        raise ImageRemovalOutputError(str(error), code="image_read_failed") from error

    if lossy_output and image.has_alpha:
        raise ImageRemovalInputError(
            "JPEG output cannot preserve the image alpha channel",
            code="alpha_not_supported",
        )

    final_mask = _prepare_mask(
        request,
        width=image.width,
        height=image.height,
    )

    try:
        candidate = inpaint_opencv(
            image,
            final_mask,
            method=request.method,
            radius=normalized_radius,
        )
    except DataContractError as error:
        raise ImageRemovalInputError(str(error), code="invalid_inpaint_options") from error
    except OpenCVInpaintError as error:
        raise ImageRemovalProcessingError(str(error), code="inpaint_failed") from error

    output = composite_masked(image, candidate, final_mask)
    replace = request.overwrite is OverwritePolicy.REPLACE
    try:
        if request.save_mask_path is not None:
            write_mask_atomic(final_mask, request.save_mask_path, replace=replace)
        write_image_atomic(output, request.output_path, replace=replace)
    except (ImageWriteError, UnsupportedImageFormatError) as error:
        raise ImageRemovalOutputError(str(error), code="output_write_failed") from error

    return ImageRemovalResult(
        input_path=request.input_path,
        output_path=request.output_path,
        status=ImageRemovalStatus.SUCCEEDED,
        method=request.method,
        radius=normalized_radius,
        dilation_radius=request.dilation_radius,
        mask_threshold=_mask_threshold(request),
        width=image.width,
        height=image.height,
        selected_pixels=final_mask.selected_pixels,
        duration_ms=(perf_counter() - started) * 1000,
        lossy_output=lossy_output,
        warnings=_warnings_for(final_mask, lossy_output=lossy_output),
    )


def build_failed_image_removal_result(
    request: ImageRemovalRequest,
    error: ImageRemovalError,
    *,
    duration_ms: float,
) -> ImageRemovalResult:
    """Build a stable failed result without exposing an exception traceback."""
    return ImageRemovalResult(
        input_path=request.input_path,
        output_path=request.output_path,
        status=ImageRemovalStatus.FAILED,
        method=request.method,
        radius=float(request.radius),
        dilation_radius=request.dilation_radius,
        mask_threshold=_mask_threshold(request),
        width=None,
        height=None,
        selected_pixels=None,
        duration_ms=duration_ms,
        lossy_output=request.output_path.suffix.lower() in {".jpg", ".jpeg"},
        error_code=error.code,
        error_message=str(error),
    )
