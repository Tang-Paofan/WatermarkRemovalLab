"""Durable B1 run metadata, append-only results, and atomic summaries."""

import json
import os
import tempfile
from pathlib import Path

from watermark_removal_lab.application.batch.contracts import (
    BATCH_SCHEMA_VERSION,
    BatchPlan,
    BatchRunError,
)
from watermark_removal_lab.application.batch.results import BatchItemResult, BatchSummary


def _run_payload(plan: BatchPlan) -> dict[str, object]:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "media": plan.normalized_spec.media.value,
        "operation": plan.normalized_spec.operation.value,
        "execution": {
            "worker_count": plan.normalized_spec.worker_count,
            "failure_policy": plan.normalized_spec.failure_policy.value,
            "overwrite_policy": plan.normalized_spec.overwrite_policy.value,
        },
        "counts": {
            "discovered": plan.discovered_count,
            "validated": plan.validated_count,
        },
        "required_resources": list(plan.required_resources),
        "warnings": list(plan.warnings),
        "result_file": plan.result_reference,
        "items": [
            {
                "item_index": item.index,
                "item_id": item.item_id,
                "valid": item.is_valid,
                "normalized_request": item.normalized_request(),
                "validation_error": (
                    None if item.validation_error is None else item.validation_error.to_dict()
                ),
            }
            for item in plan.planned_items
        ],
    }


def _write_json_atomic(
    path: Path,
    payload: dict[str, object],
    *,
    error_code: str,
    description: str,
) -> None:
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, TypeError, ValueError) as error:
        raise BatchRunError(
            f"could not publish {description} '{path}': {error}",
            code=error_code,
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


class BatchStateStore:
    """Own the durable state lifecycle for one immutable batch plan."""

    def __init__(self, plan: BatchPlan) -> None:
        if not isinstance(plan, BatchPlan):
            raise TypeError("plan must be a BatchPlan")
        self._plan = plan
        self._initialized = False

    def initialize(self) -> None:
        """Reserve state paths and publish the normalized run plan."""
        if self._initialized:
            raise BatchRunError(
                "batch state store is already initialized",
                code="state_already_initialized",
            )
        try:
            self._plan.state_directory.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise BatchRunError(
                f"could not create batch state directory '{self._plan.state_directory}': {error}",
                code="state_init_failed",
            ) from error

        try:
            self._plan.result_file.parent.mkdir(parents=True, exist_ok=True)
            with self._plan.result_file.open("x", encoding="utf-8", newline="\n") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise BatchRunError(
                f"could not reserve batch result file '{self._plan.result_file}': {error}",
                code="result_file_init_failed",
            ) from error

        _write_json_atomic(
            self._plan.run_file,
            _run_payload(self._plan),
            error_code="run_commit_failed",
            description="batch run metadata",
        )
        self._initialized = True

    def append_result(self, result: BatchItemResult) -> None:
        """Append and flush one terminal item result."""
        if not self._initialized:
            raise BatchRunError(
                "batch state store is not initialized",
                code="state_not_initialized",
            )
        if not isinstance(result, BatchItemResult) or result.run_id != self._plan.run_id:
            raise BatchRunError(
                "batch result does not belong to this run",
                code="invalid_result_record",
            )
        try:
            with self._plan.result_file.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(result.to_json_line())
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, TypeError, ValueError) as error:
            raise BatchRunError(
                f"could not append batch result '{self._plan.result_file}': {error}",
                code="result_commit_failed",
            ) from error

    def write_summary(self, summary: BatchSummary) -> None:
        """Atomically publish the final summary after all item records exist."""
        if not self._initialized:
            raise BatchRunError(
                "batch state store is not initialized",
                code="state_not_initialized",
            )
        if not isinstance(summary, BatchSummary) or summary.run_id != self._plan.run_id:
            raise BatchRunError(
                "batch summary does not belong to this run",
                code="invalid_summary",
            )
        _write_json_atomic(
            self._plan.summary_file,
            summary.to_dict(),
            error_code="summary_commit_failed",
            description="batch summary",
        )
