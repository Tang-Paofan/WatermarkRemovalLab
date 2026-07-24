"""Single-worker B1 batch orchestration."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Protocol, TypeAlias, runtime_checkable

from watermark_removal_lab.application.batch.contracts import (
    BatchCancellationReason,
    BatchContractError,
    BatchFailurePolicy,
    BatchItemStatus,
    BatchPlan,
    BatchRunError,
    PlannedImageBatchItem,
)
from watermark_removal_lab.application.batch.results import BatchItemResult, BatchSummary
from watermark_removal_lab.application.batch.state import BatchStateStore
from watermark_removal_lab.application.image_removal import (
    ImageRemovalError,
    ImageRemovalOutputError,
    ImageRemovalRequest,
    ImageRemovalResult,
    build_failed_image_removal_result,
    remove_image,
)

_LOGGER = logging.getLogger(__name__)


class BatchProgressKind(StrEnum):
    """Stable framework-neutral progress event kinds."""

    RUN_STARTED = "run_started"
    ITEM_STARTED = "item_started"
    ITEM_COMPLETED = "item_completed"
    RUN_COMPLETED = "run_completed"


@dataclass(frozen=True, slots=True)
class BatchProgressEvent:
    """One framework-neutral progress snapshot emitted by the orchestrator."""

    run_id: str
    kind: BatchProgressKind
    completed: int
    total: int
    item_index: int | None = None
    item_id: str | None = None
    status: BatchItemStatus | None = None


@runtime_checkable
class CancellationToken(Protocol):
    """Framework-neutral cancellation observation contract."""

    def is_cancelled(self) -> bool:
        """Return whether no additional work should be scheduled."""


class BatchCancellationToken:
    """Thread-safe cancellation token shared by CLI and future UI adapters."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation at the next safe batch boundary."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation was requested."""
        return self._event.is_set()


BatchProgressSink: TypeAlias = Callable[[BatchProgressEvent], None]
ImageRemovalService: TypeAlias = Callable[[ImageRemovalRequest], ImageRemovalResult]


def _is_cancelled(token: CancellationToken | None) -> bool:
    if token is None:
        return False
    try:
        cancelled = token.is_cancelled()
    except Exception as error:
        raise BatchRunError(
            f"could not read the batch cancellation token: {error}",
            code="cancellation_check_failed",
        ) from error
    if not isinstance(cancelled, bool):
        raise BatchContractError(
            "cancellation token must return a bool",
            code="invalid_cancellation_token",
        )
    return cancelled


def _emit(sink: BatchProgressSink | None, event: BatchProgressEvent) -> None:
    if sink is None:
        return
    try:
        sink(event)
    except Exception as error:
        _LOGGER.warning("batch progress sink failed: %s", error)
        _LOGGER.debug("batch progress sink traceback", exc_info=True)


def _ensure_destination_parent(destination: Path, *, output_root: Path) -> None:
    try:
        resolved_destination = destination.resolve()
        resolved_root = output_root.resolve()
        resolved_destination.relative_to(resolved_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        raise ImageRemovalOutputError(
            f"could not prepare output directory for '{destination.name}': {error}",
            code="output_directory_failed",
        ) from error


def _execute_item(
    *,
    plan: BatchPlan,
    item: PlannedImageBatchItem,
    image_service: ImageRemovalService,
) -> BatchItemResult:
    started = perf_counter()
    request = item.request
    try:
        _ensure_destination_parent(
            request.output_path,
            output_root=plan.normalized_spec.output_root,
        )
        if request.save_mask_path is not None:
            _ensure_destination_parent(
                request.save_mask_path,
                output_root=plan.normalized_spec.output_root,
            )
        image_result = image_service(request)
    except ImageRemovalError as error:
        image_result = build_failed_image_removal_result(
            request,
            error,
            duration_ms=(perf_counter() - started) * 1000,
        )
    except Exception as error:
        raise BatchRunError(
            f"item '{item.item_id}' failed outside the image-removal contract: {error}",
            code="unexpected_item_failure",
        ) from error

    if not isinstance(image_result, ImageRemovalResult):
        raise BatchRunError(
            f"item '{item.item_id}' returned an unsupported result",
            code="invalid_service_result",
        )
    try:
        return BatchItemResult.from_image_result(
            run_id=plan.run_id,
            item=item,
            result=image_result,
        )
    except BatchContractError as error:
        raise BatchRunError(
            f"item '{item.item_id}' returned an invalid result: {error}",
            code="invalid_service_result",
        ) from error


def _record_result(
    *,
    store: BatchStateStore,
    result: BatchItemResult,
    results: list[BatchItemResult],
    sink: BatchProgressSink | None,
    total: int,
) -> None:
    store.append_result(result)
    results.append(result)
    _emit(
        sink,
        BatchProgressEvent(
            run_id=result.run_id,
            kind=BatchProgressKind.ITEM_COMPLETED,
            completed=len(results),
            total=total,
            item_index=result.item.index,
            item_id=result.item.item_id,
            status=result.status,
        ),
    )


def _cancel_remaining(
    *,
    plan: BatchPlan,
    start_index: int,
    reason: BatchCancellationReason,
    store: BatchStateStore,
    results: list[BatchItemResult],
    sink: BatchProgressSink | None,
) -> None:
    for item in plan.planned_items[start_index:]:
        _record_result(
            store=store,
            result=BatchItemResult.cancelled(
                run_id=plan.run_id,
                item=item,
                reason=reason,
            ),
            results=results,
            sink=sink,
            total=plan.discovered_count,
        )


def run_batch(
    plan: BatchPlan,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_sink: BatchProgressSink | None = None,
    image_service: ImageRemovalService | None = None,
) -> BatchSummary:
    """Execute an immutable B1 plan sequentially and persist every terminal result."""
    if not isinstance(plan, BatchPlan):
        raise BatchContractError("plan must be a BatchPlan", code="invalid_batch_plan")
    if cancellation_token is not None and not isinstance(cancellation_token, CancellationToken):
        raise BatchContractError(
            "cancellation token does not implement is_cancelled()",
            code="invalid_cancellation_token",
        )
    if progress_sink is not None and not callable(progress_sink):
        raise BatchContractError(
            "progress sink must be callable",
            code="invalid_progress_sink",
        )
    if image_service is not None and not callable(image_service):
        raise BatchContractError(
            "image service must be callable",
            code="invalid_image_service",
        )
    _is_cancelled(cancellation_token)

    selected_service = remove_image if image_service is None else image_service
    started = perf_counter()
    store = BatchStateStore(plan)
    store.initialize()
    results: list[BatchItemResult] = []
    _emit(
        progress_sink,
        BatchProgressEvent(
            run_id=plan.run_id,
            kind=BatchProgressKind.RUN_STARTED,
            completed=0,
            total=plan.discovered_count,
        ),
    )

    next_index = 0
    while next_index < plan.discovered_count:
        if _is_cancelled(cancellation_token):
            _cancel_remaining(
                plan=plan,
                start_index=next_index,
                reason=BatchCancellationReason.USER_CANCELLED,
                store=store,
                results=results,
                sink=progress_sink,
            )
            break

        item = plan.planned_items[next_index]
        if item.is_valid:
            _emit(
                progress_sink,
                BatchProgressEvent(
                    run_id=plan.run_id,
                    kind=BatchProgressKind.ITEM_STARTED,
                    completed=len(results),
                    total=plan.discovered_count,
                    item_index=item.index,
                    item_id=item.item_id,
                ),
            )
            result = _execute_item(
                plan=plan,
                item=item,
                image_service=selected_service,
            )
        else:
            result = BatchItemResult.from_validation_error(run_id=plan.run_id, item=item)

        _record_result(
            store=store,
            result=result,
            results=results,
            sink=progress_sink,
            total=plan.discovered_count,
        )
        next_index += 1

        if _is_cancelled(cancellation_token):
            _cancel_remaining(
                plan=plan,
                start_index=next_index,
                reason=BatchCancellationReason.USER_CANCELLED,
                store=store,
                results=results,
                sink=progress_sink,
            )
            break
        if (
            result.status is BatchItemStatus.FAILED
            and plan.normalized_spec.failure_policy is BatchFailurePolicy.FAIL_FAST
        ):
            _cancel_remaining(
                plan=plan,
                start_index=next_index,
                reason=BatchCancellationReason.FAIL_FAST,
                store=store,
                results=results,
                sink=progress_sink,
            )
            break

    summary = BatchSummary.from_results(
        plan=plan,
        results=tuple(results),
        duration_ms=(perf_counter() - started) * 1000,
    )
    store.write_summary(summary)
    _emit(
        progress_sink,
        BatchProgressEvent(
            run_id=plan.run_id,
            kind=BatchProgressKind.RUN_COMPLETED,
            completed=len(results),
            total=plan.discovered_count,
        ),
    )
    return summary
