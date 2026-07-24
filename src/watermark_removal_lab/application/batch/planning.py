"""Deterministic B1 path normalization and preflight validation."""

import math
import os
import re
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from watermark_removal_lab.application.batch.contracts import (
    BatchContractError,
    BatchItemError,
    BatchPlan,
    BatchSpec,
    ImageBatchItemSpec,
    PlannedImageBatchItem,
)
from watermark_removal_lab.application.image_removal import (
    BoxMaskSource,
    ImageRemovalRequest,
    MaskFileSource,
    OverwritePolicy,
)
from watermark_removal_lab.image import (
    SUPPORTED_IMAGE_EXTENSIONS,
    UnsupportedImageFormatError,
)
from watermark_removal_lab.image.output import output_is_lossy

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_STATE_DIRECTORY_NAME = ".wrl-batch"


class BatchPreflightError(ValueError):
    """Raised when a batch cannot be planned safely."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError as error:
        raise BatchPreflightError(
            f"could not resolve path '{path}'",
            code="path_resolution_failed",
        ) from error


def _resolve_root(path: Path, *, role: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.is_dir():
        raise BatchPreflightError(
            f"{role} root does not exist or is not a directory",
            code=f"invalid_{role}_root",
        )
    return resolved


def _resolve_within(path: Path, *, root: Path, role: str) -> tuple[Path, str]:
    candidate = path if path.is_absolute() else root / path
    resolved = _resolve_path(candidate)
    try:
        reference = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise BatchPreflightError(
            f"{role} path escapes its declared root",
            code="path_outside_root",
        ) from error
    return resolved, reference


def _canonical_key(path: Path) -> str:
    return os.path.normcase(str(path))


def _first_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def _require_available_parent(path: Path, *, role: str) -> None:
    ancestor = _first_existing_ancestor(path.parent)
    if not ancestor.is_dir():
        raise BatchPreflightError(
            f"{role} parent is not a directory",
            code="invalid_output_directory",
        )


def _normalize_run_id(run_id: str | None) -> str:
    normalized = uuid4().hex if run_id is None else run_id
    if (
        not isinstance(normalized, str)
        or not normalized
        or _RUN_ID_PATTERN.fullmatch(normalized) is None
    ):
        raise BatchPreflightError(
            "run ID must contain only letters, numbers, underscores, or hyphens",
            code="invalid_run_id",
        )
    return normalized


def _item_validation_error(request: ImageRemovalRequest) -> BatchItemError | None:
    if not request.input_path.is_file():
        return BatchItemError(
            code="input_not_found",
            message="input does not exist or is not a regular file",
            category="validation",
        )
    if request.input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return BatchItemError(
            code="unsupported_input_format",
            message=f"unsupported input extension '{request.input_path.suffix or '<none>'}'",
            category="validation",
        )
    if isinstance(request.mask_source, MaskFileSource) and not request.mask_source.path.is_file():
        return BatchItemError(
            code="mask_not_found",
            message="mask does not exist or is not a regular file",
            category="validation",
        )
    if (
        isinstance(request.mask_source, MaskFileSource)
        and request.mask_source.path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
    ):
        return BatchItemError(
            code="unsupported_mask_format",
            message=f"unsupported mask extension '{request.mask_source.path.suffix or '<none>'}'",
            category="validation",
        )
    if isinstance(request.radius, bool) or not isinstance(request.radius, (int, float)):
        return BatchItemError(
            code="invalid_radius",
            message="inpaint radius must be a real number",
            category="validation",
        )
    if not math.isfinite(float(request.radius)) or float(request.radius) <= 0:
        return BatchItemError(
            code="invalid_radius",
            message="inpaint radius must be finite and positive",
            category="validation",
        )
    if (
        isinstance(request.dilation_radius, bool)
        or not isinstance(request.dilation_radius, int)
        or request.dilation_radius < 0
    ):
        return BatchItemError(
            code="invalid_dilation",
            message="mask dilation radius must be a non-negative integer",
            category="validation",
        )
    if isinstance(request.mask_source, MaskFileSource):
        threshold = request.mask_source.threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int)
            or not 0 <= threshold <= 255
        ):
            return BatchItemError(
                code="invalid_mask_threshold",
                message="mask threshold must be an integer from 0 through 255",
                category="validation",
            )
    try:
        output_is_lossy(request.output_path)
    except UnsupportedImageFormatError as error:
        return BatchItemError(
            code="unsupported_output_format",
            message=str(error),
            category="validation",
        )
    if request.save_mask_path is not None and request.save_mask_path.suffix.lower() != ".png":
        return BatchItemError(
            code="unsupported_output_format",
            message="saved-mask output must use the .png extension",
            category="validation",
        )
    return None


def _normalize_item(
    item: ImageBatchItemSpec,
    *,
    index: int,
    source_root: Path,
    mask_root: Path,
    output_root: Path,
    overwrite: OverwritePolicy,
) -> PlannedImageBatchItem:
    request = item.request
    input_path, input_reference = _resolve_within(
        request.input_path,
        root=source_root,
        role="input",
    )
    output_path, output_reference = _resolve_within(
        request.output_path,
        root=output_root,
        role="output",
    )

    if isinstance(request.mask_source, BoxMaskSource):
        normalized_mask_source: BoxMaskSource | MaskFileSource = request.mask_source
        mask_reference = None
    else:
        mask_path, mask_reference = _resolve_within(
            request.mask_source.path,
            root=mask_root,
            role="mask",
        )
        normalized_mask_source = MaskFileSource(
            path=mask_path,
            threshold=request.mask_source.threshold,
        )

    if request.save_mask_path is None:
        save_mask_path = None
        save_mask_reference = None
    else:
        save_mask_path, save_mask_reference = _resolve_within(
            request.save_mask_path,
            root=output_root,
            role="saved-mask output",
        )

    normalized_request = replace(
        request,
        input_path=input_path,
        output_path=output_path,
        mask_source=normalized_mask_source,
        save_mask_path=save_mask_path,
        overwrite=overwrite,
    )
    return PlannedImageBatchItem(
        index=index,
        item_id=item.item_id,
        request=normalized_request,
        input_reference=input_reference,
        output_reference=output_reference,
        mask_reference=mask_reference,
        save_mask_reference=save_mask_reference,
        validation_error=_item_validation_error(normalized_request),
    )


def _reject_duplicate_ids(items: tuple[ImageBatchItemSpec, ...]) -> None:
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise BatchPreflightError(
                f"duplicate item ID '{item.item_id}'",
                code="duplicate_item_id",
            )
        seen.add(item.item_id)


def _source_paths(items: tuple[PlannedImageBatchItem, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for item in items:
        paths.append(item.request.input_path)
        if isinstance(item.request.mask_source, MaskFileSource):
            paths.append(item.request.mask_source.path)
    return tuple(paths)


def _destination_paths(
    items: tuple[PlannedImageBatchItem, ...],
) -> tuple[tuple[str, Path], ...]:
    paths: list[tuple[str, Path]] = []
    for item in items:
        paths.append((item.item_id, item.request.output_path))
        if item.request.save_mask_path is not None:
            paths.append((item.item_id, item.request.save_mask_path))
    return tuple(paths)


def _validate_destinations(
    *,
    destinations: tuple[tuple[str, Path], ...],
    sources: tuple[Path, ...],
    output_root: Path,
    overwrite: OverwritePolicy,
) -> None:
    source_keys = {_canonical_key(path) for path in sources}
    destination_keys: dict[str, str] = {}
    reserved_root = _resolve_path(output_root / _STATE_DIRECTORY_NAME)

    for item_id, destination in destinations:
        key = _canonical_key(destination)
        if key in source_keys:
            raise BatchPreflightError(
                f"item '{item_id}' output aliases an input or mask",
                code="in_place_output",
            )
        previous_id = destination_keys.get(key)
        if previous_id is not None:
            raise BatchPreflightError(
                f"items '{previous_id}' and '{item_id}' use the same output path",
                code="duplicate_output",
            )
        destination_keys[key] = item_id

        try:
            destination.relative_to(reserved_root)
        except ValueError:
            pass
        else:
            raise BatchPreflightError(
                f"item '{item_id}' output uses the reserved batch-state directory",
                code="reserved_output_path",
            )

        _require_available_parent(destination, role="output")
        if destination.exists() and not destination.is_file():
            raise BatchPreflightError(
                f"item '{item_id}' output is not a regular file path",
                code="invalid_output_path",
            )
        if destination.exists() and overwrite is OverwritePolicy.ERROR:
            raise BatchPreflightError(
                f"item '{item_id}' output already exists",
                code="output_exists",
            )


def _result_reference(path: Path, *, output_root: Path) -> str:
    try:
        return path.relative_to(output_root).as_posix()
    except ValueError:
        return str(path)


def plan_batch(spec: BatchSpec, *, run_id: str | None = None) -> BatchPlan:
    """Normalize and preflight one deterministic sequential image batch."""
    if not isinstance(spec, BatchSpec):
        raise BatchContractError("spec must be a BatchSpec", code="invalid_batch_spec")

    normalized_run_id = _normalize_run_id(run_id)
    source_root = _resolve_root(spec.source_root, role="source")
    output_root = _resolve_root(spec.output_root, role="output")
    mask_root = (
        source_root if spec.mask_root is None else _resolve_root(spec.mask_root, role="mask")
    )
    _reject_duplicate_ids(spec.items)

    state_directory, _ = _resolve_within(
        output_root / _STATE_DIRECTORY_NAME / normalized_run_id,
        root=output_root,
        role="batch state",
    )
    _require_available_parent(state_directory, role="batch state")
    if state_directory.exists():
        raise BatchPreflightError(
            f"batch state for run '{normalized_run_id}' already exists",
            code="run_id_exists",
        )

    planned_items = tuple(
        _normalize_item(
            item,
            index=index,
            source_root=source_root,
            mask_root=mask_root,
            output_root=output_root,
            overwrite=spec.overwrite_policy,
        )
        for index, item in enumerate(spec.items)
    )
    normalized_items = tuple(
        ImageBatchItemSpec(item_id=item.item_id, request=item.request) for item in planned_items
    )

    destinations = _destination_paths(planned_items)
    sources = _source_paths(planned_items)
    _validate_destinations(
        destinations=destinations,
        sources=sources,
        output_root=output_root,
        overwrite=spec.overwrite_policy,
    )

    run_file = state_directory / "run.json"
    summary_file = state_directory / "summary.json"
    if spec.results_path is None:
        result_file = state_directory / "results.jsonl"
    else:
        candidate = spec.results_path
        result_file = _resolve_path(
            candidate if candidate.is_absolute() else output_root / candidate
        )
        if result_file.suffix.lower() != ".jsonl":
            raise BatchPreflightError(
                "custom results path must use the .jsonl extension",
                code="invalid_results_path",
            )
        _require_available_parent(result_file, role="results")

    metadata = (run_file, result_file, summary_file)
    protected_keys = {
        _canonical_key(path) for path in (*sources, *(path for _, path in destinations))
    }
    if any(_canonical_key(path) in protected_keys for path in metadata):
        raise BatchPreflightError(
            "batch metadata must not alias an input, mask, or media output",
            code="metadata_path_collision",
        )
    if any(path.exists() for path in metadata):
        raise BatchPreflightError(
            "batch metadata already exists for this run",
            code="metadata_exists",
        )

    normalized_spec = replace(
        spec,
        source_root=source_root,
        output_root=output_root,
        mask_root=mask_root,
        items=normalized_items,
        results_path=result_file,
    )
    return BatchPlan(
        run_id=normalized_run_id,
        normalized_spec=normalized_spec,
        planned_items=planned_items,
        required_resources=("cpu",),
        warnings=(),
        state_directory=state_directory,
        run_file=run_file,
        result_file=result_file,
        summary_file=summary_file,
        result_reference=_result_reference(result_file, output_root=output_root),
    )
