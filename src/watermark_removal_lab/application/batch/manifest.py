"""Versioned JSON Lines manifest adapter for B1 image batches."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

from watermark_removal_lab.application.batch.contracts import (
    BATCH_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchInputError,
    BatchMedia,
    BatchOperation,
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
from watermark_removal_lab.common import Box, DataContractError
from watermark_removal_lab.image import OpenCVInpaintMethod

_BATCH_FIELDS = frozenset({"record", "schema_version", "media", "operation", "defaults"})
_BATCH_REQUIRED_FIELDS = frozenset({"record", "schema_version", "media", "operation"})
_DEFAULT_FIELDS = frozenset({"method", "radius", "dilate", "mask_threshold"})
_ITEM_FIELDS = frozenset(
    {
        "record",
        "id",
        "input",
        "output",
        "box",
        "mask",
        "method",
        "radius",
        "dilate",
        "mask_threshold",
    }
)
_ITEM_REQUIRED_FIELDS = frozenset({"record", "id", "input", "output"})


class _ManifestJSONError(ValueError):
    """Internal signal for otherwise-valid JSON unsupported by the manifest."""


@dataclass(frozen=True, slots=True)
class ManifestBatchRequest:
    """External manifest-mode options before JSON Lines parsing."""

    manifest_path: Path
    output_directory: Path
    results_path: Path | None = None
    overwrite_policy: OverwritePolicy = OverwritePolicy.ERROR
    failure_policy: BatchFailurePolicy = BatchFailurePolicy.CONTINUE

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_path, Path) or not isinstance(self.output_directory, Path):
            raise BatchInputError(
                "manifest and output paths must be pathlib.Path values",
                code="invalid_manifest_path",
            )
        if self.results_path is not None and not isinstance(self.results_path, Path):
            raise BatchInputError(
                "results path must be a pathlib.Path",
                code="invalid_results_path",
            )
        if not isinstance(self.overwrite_policy, OverwritePolicy):
            raise BatchInputError(
                "manifest overwrite policy is not supported",
                code="invalid_overwrite",
            )
        if not isinstance(self.failure_policy, BatchFailurePolicy):
            raise BatchInputError(
                "manifest failure policy is not supported",
                code="invalid_failure_policy",
            )


@dataclass(frozen=True, slots=True)
class _ManifestDefaults:
    method: OpenCVInpaintMethod = OpenCVInpaintMethod.TELEA
    radius: float = 3.0
    dilation_radius: int = 0
    mask_threshold: int = 127


def _object_without_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _ManifestJSONError(f"duplicate object field '{key}'")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _ManifestJSONError(f"non-finite JSON number '{value}' is not supported")


def _read_manifest_records(
    path: Path,
) -> tuple[Path, tuple[tuple[int, dict[str, Any]], ...]]:
    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise BatchInputError(
            "could not resolve manifest path",
            code="manifest_read_failed",
        ) from error
    if not resolved.is_file():
        raise BatchInputError(
            "manifest does not exist or is not a regular file",
            code="manifest_not_found",
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BatchInputError(
            "could not read manifest as UTF-8",
            code="manifest_read_failed",
        ) from error

    lines = content.splitlines()
    if not lines:
        raise BatchInputError("manifest is empty", code="empty_manifest")

    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BatchInputError(
                "manifest contains a blank record",
                code="invalid_manifest_json",
                line_number=line_number,
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, _ManifestJSONError) as error:
            raise BatchInputError(
                f"invalid manifest JSON: {error}",
                code="invalid_manifest_json",
                line_number=line_number,
            ) from error
        if not isinstance(value, dict):
            raise BatchInputError(
                "manifest record must be a JSON object",
                code="invalid_manifest_record",
                line_number=line_number,
            )
        records.append((line_number, cast(dict[str, Any], value)))
    return resolved, tuple(records)


def _reject_unknown_fields(
    record: dict[str, Any],
    *,
    allowed: frozenset[str],
    line_number: int,
) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise BatchInputError(
            f"unknown manifest field '{unknown[0]}'",
            code="unknown_manifest_field",
            line_number=line_number,
        )


def _require_fields(
    record: dict[str, Any],
    *,
    required: frozenset[str],
    line_number: int,
) -> None:
    missing = sorted(required - set(record))
    if missing:
        raise BatchInputError(
            f"missing required manifest field '{missing[0]}'",
            code="missing_manifest_field",
            line_number=line_number,
        )


def _parse_method(value: Any, *, line_number: int) -> OpenCVInpaintMethod:
    if not isinstance(value, str):
        raise BatchInputError(
            "method must be a string",
            code="invalid_manifest_method",
            line_number=line_number,
        )
    try:
        return OpenCVInpaintMethod(value)
    except ValueError as error:
        raise BatchInputError(
            f"unsupported inpaint method '{value}'",
            code="invalid_manifest_method",
            line_number=line_number,
        ) from error


def _parse_radius(value: Any, *, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchInputError(
            "radius must be a finite positive number",
            code="invalid_manifest_radius",
            line_number=line_number,
        )
    try:
        normalized = float(value)
    except OverflowError as error:
        raise BatchInputError(
            "radius must be a finite positive number",
            code="invalid_manifest_radius",
            line_number=line_number,
        ) from error
    if not math.isfinite(normalized) or normalized <= 0:
        raise BatchInputError(
            "radius must be a finite positive number",
            code="invalid_manifest_radius",
            line_number=line_number,
        )
    return normalized


def _parse_dilation(value: Any, *, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BatchInputError(
            "dilate must be a non-negative integer",
            code="invalid_manifest_dilation",
            line_number=line_number,
        )
    return value


def _parse_threshold(value: Any, *, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise BatchInputError(
            "mask_threshold must be an integer from 0 through 255",
            code="invalid_manifest_threshold",
            line_number=line_number,
        )
    return value


def _parse_defaults(value: Any, *, line_number: int) -> _ManifestDefaults:
    if value is None:
        return _ManifestDefaults()
    if not isinstance(value, dict):
        raise BatchInputError(
            "defaults must be a JSON object",
            code="invalid_manifest_defaults",
            line_number=line_number,
        )
    defaults = cast(dict[str, Any], value)
    _reject_unknown_fields(defaults, allowed=_DEFAULT_FIELDS, line_number=line_number)
    return _ManifestDefaults(
        method=_parse_method(
            defaults.get("method", OpenCVInpaintMethod.TELEA.value),
            line_number=line_number,
        ),
        radius=_parse_radius(defaults.get("radius", 3.0), line_number=line_number),
        dilation_radius=_parse_dilation(
            defaults.get("dilate", 0),
            line_number=line_number,
        ),
        mask_threshold=_parse_threshold(
            defaults.get("mask_threshold", 127),
            line_number=line_number,
        ),
    )


def _parse_batch_record(
    record: dict[str, Any],
    *,
    line_number: int,
) -> _ManifestDefaults:
    _reject_unknown_fields(record, allowed=_BATCH_FIELDS, line_number=line_number)
    _require_fields(record, required=_BATCH_REQUIRED_FIELDS, line_number=line_number)
    if record["record"] != "batch":
        raise BatchInputError(
            "the first manifest record must be a batch record",
            code="invalid_manifest_record",
            line_number=line_number,
        )
    schema_version = record["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != BATCH_SCHEMA_VERSION
    ):
        raise BatchInputError(
            f"unsupported manifest schema version {schema_version!r}",
            code="unsupported_schema_version",
            line_number=line_number,
        )
    if record["media"] != BatchMedia.IMAGE.value:
        raise BatchInputError(
            "manifest media must be 'image'",
            code="unsupported_media",
            line_number=line_number,
        )
    if record["operation"] != BatchOperation.REMOVE.value:
        raise BatchInputError(
            "manifest operation must be 'remove'",
            code="unsupported_operation",
            line_number=line_number,
        )
    return _parse_defaults(record.get("defaults"), line_number=line_number)


def _parse_relative_path(value: Any, *, field: str, line_number: int) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise BatchInputError(
            f"{field} must be a non-empty relative path using forward slashes",
            code="invalid_manifest_path",
            line_number=line_number,
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or bool(windows_path.drive)
        or not posix_path.parts
        or posix_path == PurePosixPath(".")
    ):
        raise BatchInputError(
            f"{field} must be relative",
            code="invalid_manifest_path",
            line_number=line_number,
        )
    return Path(*posix_path.parts)


def _parse_box(value: Any, *, line_number: int) -> Box:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(number, bool) or not isinstance(number, int) for number in value)
    ):
        raise BatchInputError(
            "box must contain four integers: x, y, width, height",
            code="invalid_manifest_box",
            line_number=line_number,
        )
    try:
        return Box.from_xywh(*cast(list[int], value))
    except DataContractError as error:
        raise BatchInputError(
            str(error),
            code="invalid_manifest_box",
            line_number=line_number,
        ) from error


def _item_option(
    record: dict[str, Any],
    name: str,
    default: object,
) -> object:
    return record[name] if name in record else default


def _parse_item_record(
    record: dict[str, Any],
    *,
    line_number: int,
    defaults: _ManifestDefaults,
) -> ImageBatchItemSpec:
    _reject_unknown_fields(record, allowed=_ITEM_FIELDS, line_number=line_number)
    _require_fields(record, required=_ITEM_REQUIRED_FIELDS, line_number=line_number)
    if record["record"] != "item":
        raise BatchInputError(
            "every record after the batch record must be an item",
            code="invalid_manifest_record",
            line_number=line_number,
        )
    item_id = record["id"]
    if not isinstance(item_id, str) or not item_id.strip():
        raise BatchInputError(
            "item ID must be a non-empty string",
            code="invalid_manifest_item_id",
            line_number=line_number,
        )

    has_box = "box" in record
    has_mask = "mask" in record
    if has_box == has_mask:
        raise BatchInputError(
            "item requires exactly one of box or mask",
            code="invalid_manifest_localization",
            line_number=line_number,
        )

    method = _parse_method(
        _item_option(record, "method", defaults.method.value),
        line_number=line_number,
    )
    radius = _parse_radius(
        _item_option(record, "radius", defaults.radius),
        line_number=line_number,
    )
    dilation_radius = _parse_dilation(
        _item_option(record, "dilate", defaults.dilation_radius),
        line_number=line_number,
    )
    threshold = _parse_threshold(
        _item_option(record, "mask_threshold", defaults.mask_threshold),
        line_number=line_number,
    )

    if has_box:
        mask_source: BoxMaskSource | MaskFileSource = BoxMaskSource(
            _parse_box(record["box"], line_number=line_number)
        )
    else:
        mask_source = MaskFileSource(
            _parse_relative_path(record["mask"], field="mask", line_number=line_number),
            threshold=threshold,
        )

    return ImageBatchItemSpec(
        item_id=item_id,
        request=ImageRemovalRequest(
            input_path=_parse_relative_path(
                record["input"],
                field="input",
                line_number=line_number,
            ),
            output_path=_parse_relative_path(
                record["output"],
                field="output",
                line_number=line_number,
            ),
            mask_source=mask_source,
            method=method,
            radius=radius,
            dilation_radius=dilation_radius,
        ),
    )


def build_manifest_batch_spec(request: ManifestBatchRequest) -> BatchSpec:
    """Parse a strict manifest v1 file into ordinary ordered batch items."""
    if not isinstance(request, ManifestBatchRequest):
        raise BatchInputError(
            "request must be a ManifestBatchRequest",
            code="invalid_manifest_request",
        )

    manifest_path, records = _read_manifest_records(request.manifest_path)
    header_line, header = records[0]
    defaults = _parse_batch_record(header, line_number=header_line)
    if len(records) == 1:
        raise BatchInputError(
            "manifest contains no item records",
            code="empty_batch",
        )

    items: list[ImageBatchItemSpec] = []
    item_ids: set[str] = set()
    for line_number, record in records[1:]:
        item = _parse_item_record(
            record,
            line_number=line_number,
            defaults=defaults,
        )
        if item.item_id in item_ids:
            raise BatchInputError(
                f"duplicate item ID '{item.item_id}'",
                code="duplicate_item_id",
                line_number=line_number,
            )
        item_ids.add(item.item_id)
        items.append(item)

    return BatchSpec(
        source_root=manifest_path.parent,
        output_root=request.output_directory,
        mask_root=manifest_path.parent,
        protected_paths=(manifest_path,),
        items=tuple(items),
        results_path=request.results_path,
        overwrite_policy=request.overwrite_policy,
        failure_policy=request.failure_policy,
    )


def plan_manifest_batch(
    request: ManifestBatchRequest,
    *,
    run_id: str | None = None,
) -> BatchPlan:
    """Parse and preflight one manifest-mode B1 batch."""
    return plan_batch(build_manifest_batch_spec(request), run_id=run_id)
