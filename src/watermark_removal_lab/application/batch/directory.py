"""Directory input adapter for deterministic B1 image batches."""

import math
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from watermark_removal_lab.application.batch.contracts import (
    BatchFailurePolicy,
    BatchInputError,
    BatchPlan,
    BatchSpec,
    ImageBatchItemSpec,
)
from watermark_removal_lab.application.batch.planning import plan_batch
from watermark_removal_lab.application.image_removal import (
    BoxMaskSource,
    ImageRemovalRequest,
    MaskFileSource,
    OverwritePolicy,
)
from watermark_removal_lab.common import Box
from watermark_removal_lab.image import (
    SUPPORTED_IMAGE_EXTENSIONS,
    OpenCVInpaintMethod,
)

_STATE_DIRECTORY_NAME = ".wrl-batch"


class DirectoryOutputFormat(StrEnum):
    """Output-extension policies supported by the B1 directory adapter."""

    PRESERVE = "preserve"
    PNG = "png"


def _normalize_radius(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchInputError(
            "inpaint radius must be finite and positive",
            code="invalid_radius",
        )
    try:
        normalized = float(value)
    except OverflowError as error:
        raise BatchInputError(
            "inpaint radius must be finite and positive",
            code="invalid_radius",
        ) from error
    if not math.isfinite(normalized) or normalized <= 0:
        raise BatchInputError(
            "inpaint radius must be finite and positive",
            code="invalid_radius",
        )
    return normalized


@dataclass(frozen=True, slots=True)
class DirectoryBatchRequest:
    """External directory-mode options before deterministic discovery."""

    input_directory: Path
    output_directory: Path
    box: Box | None = None
    mask_directory: Path | None = None
    recursive: bool = False
    method: OpenCVInpaintMethod = OpenCVInpaintMethod.TELEA
    radius: float = 3.0
    dilation_radius: int = 0
    output_format: DirectoryOutputFormat = DirectoryOutputFormat.PRESERVE
    overwrite_policy: OverwritePolicy = OverwritePolicy.ERROR
    failure_policy: BatchFailurePolicy = BatchFailurePolicy.CONTINUE
    results_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_directory, Path) or not isinstance(
            self.output_directory, Path
        ):
            raise BatchInputError(
                "input and output directories must be pathlib.Path values",
                code="invalid_directory_path",
            )
        if self.mask_directory is not None and not isinstance(self.mask_directory, Path):
            raise BatchInputError(
                "mask directory must be a pathlib.Path",
                code="invalid_directory_path",
            )
        if self.results_path is not None and not isinstance(self.results_path, Path):
            raise BatchInputError(
                "results path must be a pathlib.Path",
                code="invalid_results_path",
            )
        if (self.box is None) == (self.mask_directory is None):
            raise BatchInputError(
                "directory mode requires exactly one of box or mask directory",
                code="invalid_localization",
            )
        if self.box is not None and not isinstance(self.box, Box):
            raise BatchInputError(
                "directory box must be a Box",
                code="invalid_localization",
            )
        if not isinstance(self.recursive, bool):
            raise BatchInputError(
                "recursive must be a boolean",
                code="invalid_recursive",
            )
        if not isinstance(self.method, OpenCVInpaintMethod):
            raise BatchInputError(
                "directory inpaint method is not supported",
                code="invalid_method",
            )
        object.__setattr__(self, "radius", _normalize_radius(self.radius))
        if (
            isinstance(self.dilation_radius, bool)
            or not isinstance(self.dilation_radius, int)
            or self.dilation_radius < 0
        ):
            raise BatchInputError(
                "mask dilation radius must be a non-negative integer",
                code="invalid_dilation",
            )
        if not isinstance(self.output_format, DirectoryOutputFormat):
            raise BatchInputError(
                "directory output format is not supported",
                code="invalid_output_format",
            )
        if not isinstance(self.overwrite_policy, OverwritePolicy):
            raise BatchInputError(
                "directory overwrite policy is not supported",
                code="invalid_overwrite",
            )
        if not isinstance(self.failure_policy, BatchFailurePolicy):
            raise BatchInputError(
                "directory failure policy is not supported",
                code="invalid_failure_policy",
            )


def _resolve_directory(path: Path, *, role: str) -> Path:
    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise BatchInputError(
            f"could not resolve {role} directory",
            code="directory_resolution_failed",
        ) from error
    if not resolved.is_dir():
        raise BatchInputError(
            f"{role} directory does not exist or is not a directory",
            code=f"invalid_{role}_directory",
        )
    return resolved


def _is_state_directory(name: str) -> bool:
    return name.casefold() == _STATE_DIRECTORY_NAME.casefold()


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and path.is_file()


def _discovery_error(error: OSError) -> None:
    raise BatchInputError(
        "could not enumerate the input directory",
        code="directory_discovery_failed",
    ) from error


def _discover_images(input_directory: Path, *, recursive: bool) -> tuple[Path, ...]:
    discovered: list[Path] = []
    try:
        if recursive:
            for current, directories, filenames in os.walk(
                input_directory,
                topdown=True,
                onerror=_discovery_error,
                followlinks=False,
            ):
                directories[:] = sorted(
                    (name for name in directories if not _is_state_directory(name)),
                    key=lambda name: (name.casefold(), name),
                )
                current_path = Path(current)
                discovered.extend(
                    candidate
                    for name in filenames
                    if _is_supported_image(candidate := current_path / name)
                )
        else:
            discovered.extend(
                candidate
                for candidate in input_directory.iterdir()
                if _is_supported_image(candidate)
            )
    except OSError as error:
        _discovery_error(error)

    def sort_key(path: Path) -> tuple[str, str]:
        reference = path.relative_to(input_directory).as_posix()
        return (reference.casefold(), reference)

    return tuple(sorted(discovered, key=sort_key))


def _reject_nested_output(input_directory: Path, output_directory: Path) -> None:
    try:
        output_directory.relative_to(input_directory)
    except ValueError:
        return
    raise BatchInputError(
        "output directory must not be the input directory or one of its descendants",
        code="output_inside_input",
    )


def _output_path(relative_input: Path, output_format: DirectoryOutputFormat) -> Path:
    if output_format is DirectoryOutputFormat.PNG:
        return relative_input.with_suffix(".png")
    return relative_input


def build_directory_batch_spec(request: DirectoryBatchRequest) -> BatchSpec:
    """Discover directory inputs and compile ordinary ordered batch items."""
    if not isinstance(request, DirectoryBatchRequest):
        raise BatchInputError(
            "request must be a DirectoryBatchRequest",
            code="invalid_directory_request",
        )

    input_directory = _resolve_directory(request.input_directory, role="input")
    output_directory = _resolve_directory(request.output_directory, role="output")
    _reject_nested_output(input_directory, output_directory)
    mask_directory = (
        None
        if request.mask_directory is None
        else _resolve_directory(request.mask_directory, role="mask")
    )

    discovered = _discover_images(input_directory, recursive=request.recursive)
    if not discovered:
        raise BatchInputError(
            "input directory contains no supported images",
            code="no_input_files",
        )

    items: list[ImageBatchItemSpec] = []
    for input_path in discovered:
        relative_input = input_path.relative_to(input_directory)
        if request.box is not None:
            mask_source: BoxMaskSource | MaskFileSource = BoxMaskSource(request.box)
        else:
            mask_source = MaskFileSource(relative_input.with_suffix(".png"))
        items.append(
            ImageBatchItemSpec(
                item_id=relative_input.as_posix(),
                request=ImageRemovalRequest(
                    input_path=relative_input,
                    output_path=_output_path(relative_input, request.output_format),
                    mask_source=mask_source,
                    method=request.method,
                    radius=request.radius,
                    dilation_radius=request.dilation_radius,
                ),
            )
        )

    return BatchSpec(
        source_root=input_directory,
        output_root=output_directory,
        mask_root=mask_directory,
        items=tuple(items),
        results_path=request.results_path,
        overwrite_policy=request.overwrite_policy,
        failure_policy=request.failure_policy,
    )


def plan_directory_batch(
    request: DirectoryBatchRequest,
    *,
    run_id: str | None = None,
) -> BatchPlan:
    """Compile and preflight one directory-mode B1 batch."""
    return plan_batch(build_directory_batch_spec(request), run_id=run_id)
