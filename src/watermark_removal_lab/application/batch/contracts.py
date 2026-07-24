"""Framework-neutral contracts for B1 image batches."""

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from watermark_removal_lab.application.image_removal import (
    BoxMaskSource,
    ImageRemovalRequest,
    MaskFileSource,
    OverwritePolicy,
)
from watermark_removal_lab.common import Box
from watermark_removal_lab.image import OpenCVInpaintMethod

BATCH_SCHEMA_VERSION = 1


class BatchMedia(StrEnum):
    """Media types supported by versioned batch specifications."""

    IMAGE = "image"


class BatchOperation(StrEnum):
    """Operations supported by versioned batch specifications."""

    REMOVE = "remove"


class BatchFailurePolicy(StrEnum):
    """Behavior after an item failure."""

    CONTINUE = "continue"
    FAIL_FAST = "fail_fast"


class BatchItemStatus(StrEnum):
    """Terminal states recorded for every discovered batch item."""

    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchCancellationReason(StrEnum):
    """Stable reasons for cancelling an unscheduled B1 item."""

    FAIL_FAST = "fail_fast"
    USER_CANCELLED = "user_cancelled"


class BatchContractError(ValueError):
    """Raised when an in-memory batch contract is structurally invalid."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class BatchInputError(ValueError):
    """Raised when an external batch input cannot produce a valid specification."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        line_number: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.line_number = line_number


@dataclass(frozen=True, slots=True)
class BatchItemError:
    """Stable item error safe for machine-readable results."""

    code: str
    message: str
    category: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible error object."""
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category,
        }


def _validate_request_contract(request: ImageRemovalRequest) -> None:
    if not isinstance(request.input_path, Path):
        raise BatchContractError("item input must be a pathlib.Path", code="invalid_path")
    if not isinstance(request.output_path, Path):
        raise BatchContractError("item output must be a pathlib.Path", code="invalid_path")
    if not isinstance(request.mask_source, (BoxMaskSource, MaskFileSource)):
        raise BatchContractError(
            "item localization source is not supported",
            code="invalid_localization",
        )
    if isinstance(request.mask_source, BoxMaskSource) and not isinstance(
        request.mask_source.box, Box
    ):
        raise BatchContractError(
            "item box localization is invalid",
            code="invalid_localization",
        )
    if isinstance(request.mask_source, MaskFileSource) and not isinstance(
        request.mask_source.path, Path
    ):
        raise BatchContractError("item mask must be a pathlib.Path", code="invalid_path")
    if not isinstance(request.method, OpenCVInpaintMethod):
        raise BatchContractError("item inpaint method is not supported", code="invalid_method")
    if request.save_mask_path is not None and not isinstance(request.save_mask_path, Path):
        raise BatchContractError("saved-mask output must be a pathlib.Path", code="invalid_path")


@dataclass(frozen=True, slots=True)
class ImageBatchItemSpec:
    """One ordered image-removal request before batch preflight."""

    item_id: str
    request: ImageRemovalRequest

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise BatchContractError("item ID must be a non-empty string", code="invalid_item_id")
        if not isinstance(self.request, ImageRemovalRequest):
            raise BatchContractError(
                "item request must be an ImageRemovalRequest",
                code="invalid_item_request",
            )
        _validate_request_contract(self.request)


@dataclass(frozen=True, slots=True)
class BatchSpec:
    """Adapter-independent input for B1 image batch planning."""

    source_root: Path
    output_root: Path
    items: tuple[ImageBatchItemSpec, ...]
    mask_root: Path | None = None
    protected_paths: tuple[Path, ...] = ()
    results_path: Path | None = None
    schema_version: int = BATCH_SCHEMA_VERSION
    media: BatchMedia = BatchMedia.IMAGE
    operation: BatchOperation = BatchOperation.REMOVE
    overwrite_policy: OverwritePolicy = OverwritePolicy.ERROR
    failure_policy: BatchFailurePolicy = BatchFailurePolicy.CONTINUE
    worker_count: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != BATCH_SCHEMA_VERSION
        ):
            raise BatchContractError(
                f"unsupported batch schema version {self.schema_version!r}",
                code="unsupported_schema_version",
            )
        if self.media is not BatchMedia.IMAGE:
            raise BatchContractError("B1 supports image media only", code="unsupported_media")
        if self.operation is not BatchOperation.REMOVE:
            raise BatchContractError(
                "B1 supports the remove operation only",
                code="unsupported_operation",
            )
        if not isinstance(self.source_root, Path) or not isinstance(self.output_root, Path):
            raise BatchContractError("batch roots must be pathlib.Path values", code="invalid_path")
        if self.mask_root is not None and not isinstance(self.mask_root, Path):
            raise BatchContractError("mask root must be a pathlib.Path", code="invalid_path")
        if not isinstance(self.protected_paths, tuple) or not all(
            isinstance(path, Path) for path in self.protected_paths
        ):
            raise BatchContractError(
                "protected paths must be a tuple of pathlib.Path values",
                code="invalid_path",
            )
        if self.results_path is not None and not isinstance(self.results_path, Path):
            raise BatchContractError(
                "results path must be a pathlib.Path",
                code="invalid_path",
            )
        if not isinstance(self.items, tuple) or not self.items:
            raise BatchContractError(
                "batch items must be a non-empty tuple",
                code="empty_batch",
            )
        if not all(isinstance(item, ImageBatchItemSpec) for item in self.items):
            raise BatchContractError(
                "batch items contain an unsupported value",
                code="invalid_batch_item",
            )
        if not isinstance(self.overwrite_policy, OverwritePolicy):
            raise BatchContractError(
                "batch overwrite policy is not supported",
                code="invalid_overwrite",
            )
        if not isinstance(self.failure_policy, BatchFailurePolicy):
            raise BatchContractError(
                "batch failure policy is not supported",
                code="invalid_failure_policy",
            )
        if (
            isinstance(self.worker_count, bool)
            or not isinstance(self.worker_count, int)
            or self.worker_count != 1
        ):
            raise BatchContractError(
                "B1 requires exactly one worker",
                code="unsupported_worker_count",
            )


@dataclass(frozen=True, slots=True)
class PlannedImageBatchItem:
    """One normalized item in an immutable batch plan."""

    index: int
    item_id: str
    request: ImageRemovalRequest
    input_reference: str
    output_reference: str
    mask_reference: str | None
    save_mask_reference: str | None
    validation_error: BatchItemError | None = None

    @property
    def is_valid(self) -> bool:
        """Return whether the item passed cheap B1 validation."""
        return self.validation_error is None

    def normalized_request(self) -> dict[str, object]:
        """Return a portable JSON-compatible request representation."""
        source = self.request.mask_source
        if isinstance(source, BoxMaskSource):
            box = source.box
            localization: dict[str, object] = {
                "type": "box",
                "box": [box.x_min, box.y_min, box.width, box.height],
            }
            mask_threshold: int | None = None
        else:
            localization = {
                "type": "mask",
                "path": self.mask_reference,
                "threshold": source.threshold,
            }
            mask_threshold = source.threshold

        return {
            "media": BatchMedia.IMAGE.value,
            "operation": BatchOperation.REMOVE.value,
            "input": self.input_reference,
            "output": self.output_reference,
            "localization": localization,
            "method": self.request.method.value,
            "options": {
                "radius": _portable_scalar(self.request.radius),
                "dilate": _portable_scalar(self.request.dilation_radius),
                "mask_threshold": _portable_scalar(mask_threshold),
                "save_mask": self.save_mask_reference,
                "overwrite": self.request.overwrite.value,
            },
        }


def _portable_scalar(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Immutable normalized B1 plan produced before item execution."""

    run_id: str
    normalized_spec: BatchSpec
    planned_items: tuple[PlannedImageBatchItem, ...]
    required_resources: tuple[str, ...]
    warnings: tuple[str, ...]
    state_directory: Path
    run_file: Path
    result_file: Path
    summary_file: Path
    result_reference: str

    @property
    def validated_items(self) -> tuple[PlannedImageBatchItem, ...]:
        """Return items that passed cheap validation."""
        return tuple(item for item in self.planned_items if item.is_valid)

    @property
    def invalid_items(self) -> tuple[PlannedImageBatchItem, ...]:
        """Return items that must become validation-failure results."""
        return tuple(item for item in self.planned_items if not item.is_valid)

    @property
    def discovered_count(self) -> int:
        """Return the number of items supplied by the adapter."""
        return len(self.planned_items)

    @property
    def validated_count(self) -> int:
        """Return the number of items eligible for execution."""
        return len(self.validated_items)
