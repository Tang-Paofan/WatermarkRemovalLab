"""Machine-readable B1 item results and final batch summaries."""

import json
import math
from dataclasses import dataclass

from watermark_removal_lab.application.batch.contracts import (
    BATCH_SCHEMA_VERSION,
    BatchCancellationReason,
    BatchContractError,
    BatchItemError,
    BatchItemStatus,
    BatchPlan,
    PlannedImageBatchItem,
)
from watermark_removal_lab.application.image_removal import (
    ImageRemovalResult,
    ImageRemovalStatus,
)


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    """One terminal result record in deterministic batch order."""

    run_id: str
    item: PlannedImageBatchItem
    status: BatchItemStatus
    width: int | None
    height: int | None
    selected_pixels: int | None
    duration_ms: float
    lossy_output: bool
    warnings: tuple[str, ...] = ()
    error: BatchItemError | None = None
    attempt: int = 1

    @classmethod
    def from_image_result(
        cls,
        *,
        run_id: str,
        item: PlannedImageBatchItem,
        result: ImageRemovalResult,
    ) -> "BatchItemResult":
        """Adapt a single-image service result without exposing absolute paths."""
        if not item.is_valid:
            raise BatchContractError(
                "cannot attach an execution result to an invalid planned item",
                code="invalid_result_item",
            )
        if (
            result.input_path.resolve() != item.request.input_path
            or result.output_path.resolve() != item.request.output_path
        ):
            raise BatchContractError(
                "single-image result paths do not match the planned item",
                code="result_path_mismatch",
            )

        status = BatchItemStatus(result.status.value)
        error = None
        if result.status is ImageRemovalStatus.FAILED:
            if result.error_code is None or result.error_message is None:
                raise BatchContractError(
                    "failed single-image result is missing an error",
                    code="missing_result_error",
                )
            error = BatchItemError(
                code=result.error_code,
                message=result.error_message,
                category="image_removal",
            )

        return cls(
            run_id=run_id,
            item=item,
            status=status,
            width=result.width,
            height=result.height,
            selected_pixels=result.selected_pixels,
            duration_ms=result.duration_ms,
            lossy_output=result.lossy_output,
            warnings=result.warnings,
            error=error,
        )

    @classmethod
    def from_validation_error(
        cls,
        *,
        run_id: str,
        item: PlannedImageBatchItem,
    ) -> "BatchItemResult":
        """Create the failed result for an item rejected during preflight."""
        if item.validation_error is None:
            raise BatchContractError(
                "planned item has no validation error",
                code="missing_validation_error",
            )
        return cls(
            run_id=run_id,
            item=item,
            status=BatchItemStatus.FAILED,
            width=None,
            height=None,
            selected_pixels=None,
            duration_ms=0.0,
            lossy_output=item.request.output_path.suffix.lower() in {".jpg", ".jpeg"},
            error=item.validation_error,
        )

    @classmethod
    def cancelled(
        cls,
        *,
        run_id: str,
        item: PlannedImageBatchItem,
        reason: BatchCancellationReason,
    ) -> "BatchItemResult":
        """Create a terminal result for one unscheduled item."""
        if not isinstance(reason, BatchCancellationReason):
            raise BatchContractError(
                "cancellation reason is not supported",
                code="invalid_cancellation_reason",
            )
        return cls(
            run_id=run_id,
            item=item,
            status=BatchItemStatus.CANCELLED,
            width=None,
            height=None,
            selected_pixels=None,
            duration_ms=0.0,
            lossy_output=item.request.output_path.suffix.lower() in {".jpg", ".jpeg"},
            error=BatchItemError(
                code=reason.value,
                message=f"item was cancelled because of {reason.value}",
                category="cancellation",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the append-only JSON Lines record representation."""
        return {
            "schema_version": BATCH_SCHEMA_VERSION,
            "run_id": self.run_id,
            "item_index": self.item.index,
            "item_id": self.item.item_id,
            "attempt": self.attempt,
            "status": self.status.value,
            "normalized_request": self.item.normalized_request(),
            "fingerprints": {},
            "output": {
                "path": self.item.output_reference,
                "published": self.status is BatchItemStatus.SUCCEEDED,
                "integrity_hash": None,
            },
            "metrics": {
                "width": self.width,
                "height": self.height,
                "selected_pixels": self.selected_pixels,
                "lossy_output": self.lossy_output,
            },
            "warnings": list(self.warnings),
            "error": None if self.error is None else self.error.to_dict(),
            "timing": {"duration_ms": self.duration_ms},
        }

    def to_json_line(self) -> str:
        """Serialize one deterministic newline-terminated JSON record."""
        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(frozen=True, slots=True)
class BatchSummary:
    """Final terminal counts and aggregate timing for one batch run."""

    run_id: str
    discovered: int
    validated: int
    succeeded: int
    skipped: int
    failed: int
    cancelled: int
    result_file: str
    duration_ms: float
    item_duration_ms: float
    metric_denominator: int

    @classmethod
    def from_results(
        cls,
        *,
        plan: BatchPlan,
        results: tuple[BatchItemResult, ...],
        duration_ms: float,
    ) -> "BatchSummary":
        """Build a summary only after every discovered item is terminal."""
        if len(results) != plan.discovered_count:
            raise BatchContractError(
                "summary requires one result for every discovered item",
                code="incomplete_batch_results",
            )
        expected = tuple(range(plan.discovered_count))
        actual = tuple(result.item.index for result in results)
        if actual != expected or any(result.run_id != plan.run_id for result in results):
            raise BatchContractError(
                "summary results do not match the batch plan",
                code="mismatched_batch_results",
            )
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, (int, float))
            or not math.isfinite(float(duration_ms))
            or duration_ms < 0
        ):
            raise BatchContractError(
                "batch duration must be non-negative",
                code="invalid_batch_duration",
            )

        counts = {
            status: sum(result.status is status for result in results) for status in BatchItemStatus
        }
        return cls(
            run_id=plan.run_id,
            discovered=plan.discovered_count,
            validated=plan.validated_count,
            succeeded=counts[BatchItemStatus.SUCCEEDED],
            skipped=counts[BatchItemStatus.SKIPPED],
            failed=counts[BatchItemStatus.FAILED],
            cancelled=counts[BatchItemStatus.CANCELLED],
            result_file=plan.result_reference,
            duration_ms=duration_ms,
            item_duration_ms=sum(result.duration_ms for result in results),
            metric_denominator=len(results),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the final JSON-compatible summary representation."""
        return {
            "schema_version": BATCH_SCHEMA_VERSION,
            "run_id": self.run_id,
            "counts": {
                "discovered": self.discovered,
                "validated": self.validated,
                "succeeded": self.succeeded,
                "skipped": self.skipped,
                "failed": self.failed,
                "cancelled": self.cancelled,
            },
            "aggregate_metrics": {
                "item_duration_ms": {
                    "denominator": self.metric_denominator,
                    "total": self.item_duration_ms,
                }
            },
            "result_file": self.result_file,
            "timing": {"duration_ms": self.duration_ms},
        }
