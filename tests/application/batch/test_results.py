"""Tests for append-only batch results and terminal summaries."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from watermark_removal_lab.application import (
    BatchCancellationReason,
    BatchContractError,
    BatchItemResult,
    BatchItemStatus,
    BatchPlan,
    BatchSpec,
    BatchSummary,
    BoxMaskSource,
    ImageBatchItemSpec,
    ImageRemovalRequest,
    ImageRemovalResult,
    ImageRemovalStatus,
    OverwritePolicy,
    PlannedImageBatchItem,
    plan_batch,
)
from watermark_removal_lab.common import Box
from watermark_removal_lab.image import OpenCVInpaintMethod


def _plan(tmp_path: Path, *, invalid_last: bool = False) -> BatchPlan:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    items: list[ImageBatchItemSpec] = []
    for index in range(5):
        name = f"input-{index}.png"
        if not invalid_last or index < 4:
            (source / name).write_bytes(b"synthetic")
        items.append(
            ImageBatchItemSpec(
                f"item-{index}",
                ImageRemovalRequest(
                    input_path=Path(name),
                    output_path=Path(f"output-{index}.png"),
                    mask_source=BoxMaskSource(Box.from_xywh(0, 0, 1, 1)),
                ),
            )
        )
    return plan_batch(
        BatchSpec(
            source_root=source,
            output_root=output,
            items=tuple(items),
            overwrite_policy=OverwritePolicy.SKIP,
        ),
        run_id="result-run",
    )


def _image_result(
    item: PlannedImageBatchItem,
    status: ImageRemovalStatus,
    *,
    error: bool = False,
) -> ImageRemovalResult:
    return ImageRemovalResult(
        input_path=item.request.input_path,
        output_path=item.request.output_path,
        status=status,
        method=OpenCVInpaintMethod.TELEA,
        radius=3.0,
        dilation_radius=0,
        mask_threshold=None,
        width=10 if status is ImageRemovalStatus.SUCCEEDED else None,
        height=8 if status is ImageRemovalStatus.SUCCEEDED else None,
        selected_pixels=4 if status is ImageRemovalStatus.SUCCEEDED else None,
        duration_ms=2.5,
        lossy_output=False,
        warnings=("warning",),
        error_code="simulated" if error else None,
        error_message="simulated failure" if error else None,
    )


def test_batch_item_result_adapts_success_skip_and_failure(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    success = BatchItemResult.from_image_result(
        run_id=plan.run_id,
        item=plan.planned_items[0],
        result=_image_result(plan.planned_items[0], ImageRemovalStatus.SUCCEEDED),
    )
    skipped = BatchItemResult.from_image_result(
        run_id=plan.run_id,
        item=plan.planned_items[1],
        result=_image_result(plan.planned_items[1], ImageRemovalStatus.SKIPPED),
    )
    failed = BatchItemResult.from_image_result(
        run_id=plan.run_id,
        item=plan.planned_items[2],
        result=_image_result(
            plan.planned_items[2],
            ImageRemovalStatus.FAILED,
            error=True,
        ),
    )

    assert success.status is BatchItemStatus.SUCCEEDED
    assert success.to_dict()["output"] == {
        "path": "output-0.png",
        "published": True,
        "integrity_hash": None,
    }
    assert skipped.status is BatchItemStatus.SKIPPED
    assert skipped.to_dict()["output"] == {
        "path": "output-1.png",
        "published": False,
        "integrity_hash": None,
    }
    assert failed.status is BatchItemStatus.FAILED
    assert failed.error is not None
    assert failed.error.category == "image_removal"


def test_batch_item_result_rejects_execution_for_invalid_item(tmp_path: Path) -> None:
    plan = _plan(tmp_path, invalid_last=True)
    item = plan.planned_items[-1]

    with pytest.raises(BatchContractError) as captured:
        BatchItemResult.from_image_result(
            run_id=plan.run_id,
            item=item,
            result=_image_result(item, ImageRemovalStatus.SUCCEEDED),
        )

    assert captured.value.code == "invalid_result_item"


def test_batch_item_result_rejects_path_mismatch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    item = plan.planned_items[0]
    result = replace(
        _image_result(item, ImageRemovalStatus.SUCCEEDED),
        output_path=tmp_path / "other.png",
    )

    with pytest.raises(BatchContractError) as captured:
        BatchItemResult.from_image_result(run_id=plan.run_id, item=item, result=result)

    assert captured.value.code == "result_path_mismatch"


def test_batch_item_result_requires_failed_error_details(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    item = plan.planned_items[0]

    with pytest.raises(BatchContractError) as captured:
        BatchItemResult.from_image_result(
            run_id=plan.run_id,
            item=item,
            result=_image_result(item, ImageRemovalStatus.FAILED),
        )

    assert captured.value.code == "missing_result_error"


def test_batch_item_result_builds_validation_failure(tmp_path: Path) -> None:
    plan = _plan(tmp_path, invalid_last=True)
    item = plan.planned_items[-1]

    result = BatchItemResult.from_validation_error(run_id=plan.run_id, item=item)
    payload = result.to_dict()

    assert result.status is BatchItemStatus.FAILED
    assert payload["error"] == {
        "code": "input_not_found",
        "message": "input does not exist or is not a regular file",
        "category": "validation",
    }


def test_batch_item_result_rejects_missing_validation_error(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    with pytest.raises(BatchContractError) as captured:
        BatchItemResult.from_validation_error(
            run_id=plan.run_id,
            item=plan.planned_items[0],
        )

    assert captured.value.code == "missing_validation_error"


def test_cancelled_result_records_reason_and_lossy_output(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    item = replace(
        plan.planned_items[0],
        request=replace(
            plan.planned_items[0].request,
            output_path=plan.planned_items[0].request.output_path.with_suffix(".jpg"),
        ),
    )

    result = BatchItemResult.cancelled(
        run_id=plan.run_id,
        item=item,
        reason=BatchCancellationReason.FAIL_FAST,
    )

    assert result.status is BatchItemStatus.CANCELLED
    assert result.lossy_output
    assert result.error is not None
    assert result.error.code == "fail_fast"


def test_cancelled_result_rejects_unknown_reason(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    with pytest.raises(BatchContractError) as captured:
        BatchItemResult.cancelled(
            run_id=plan.run_id,
            item=plan.planned_items[0],
            reason=cast(BatchCancellationReason, "unknown"),
        )

    assert captured.value.code == "invalid_cancellation_reason"


def test_batch_item_json_line_is_portable_and_deterministic(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    result = BatchItemResult.cancelled(
        run_id=plan.run_id,
        item=plan.planned_items[0],
        reason=BatchCancellationReason.USER_CANCELLED,
    )

    encoded = result.to_json_line()
    payload = json.loads(encoded)

    assert encoded.endswith("\n")
    assert payload["schema_version"] == 1
    assert payload["normalized_request"]["input"] == "input-0.png"
    assert str(tmp_path) not in encoded
    assert payload["error"]["code"] == "user_cancelled"


def test_batch_summary_counts_every_terminal_status(tmp_path: Path) -> None:
    plan = _plan(tmp_path, invalid_last=True)
    results = (
        BatchItemResult.from_image_result(
            run_id=plan.run_id,
            item=plan.planned_items[0],
            result=_image_result(plan.planned_items[0], ImageRemovalStatus.SUCCEEDED),
        ),
        BatchItemResult.from_image_result(
            run_id=plan.run_id,
            item=plan.planned_items[1],
            result=_image_result(plan.planned_items[1], ImageRemovalStatus.SKIPPED),
        ),
        BatchItemResult.from_image_result(
            run_id=plan.run_id,
            item=plan.planned_items[2],
            result=_image_result(
                plan.planned_items[2],
                ImageRemovalStatus.FAILED,
                error=True,
            ),
        ),
        BatchItemResult.cancelled(
            run_id=plan.run_id,
            item=plan.planned_items[3],
            reason=BatchCancellationReason.FAIL_FAST,
        ),
        BatchItemResult.from_validation_error(
            run_id=plan.run_id,
            item=plan.planned_items[4],
        ),
    )

    summary = BatchSummary.from_results(
        plan=plan,
        results=results,
        duration_ms=12.0,
    )
    payload = summary.to_dict()

    assert payload["counts"] == {
        "discovered": 5,
        "validated": 4,
        "succeeded": 1,
        "skipped": 1,
        "failed": 2,
        "cancelled": 1,
    }
    assert payload["aggregate_metrics"] == {"item_duration_ms": {"denominator": 5, "total": 7.5}}
    assert payload["result_file"] == ".wrl-batch/result-run/results.jsonl"
    assert payload["timing"] == {"duration_ms": 12.0}


def test_batch_summary_rejects_incomplete_results(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    with pytest.raises(BatchContractError) as captured:
        BatchSummary.from_results(plan=plan, results=(), duration_ms=1.0)

    assert captured.value.code == "incomplete_batch_results"


def test_batch_summary_rejects_wrong_result_order(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    results = tuple(
        BatchItemResult.cancelled(
            run_id=plan.run_id,
            item=item,
            reason=BatchCancellationReason.FAIL_FAST,
        )
        for item in plan.planned_items
    )

    with pytest.raises(BatchContractError) as captured:
        BatchSummary.from_results(
            plan=plan,
            results=(results[1], results[0], *results[2:]),
            duration_ms=1.0,
        )

    assert captured.value.code == "mismatched_batch_results"


def test_batch_summary_rejects_wrong_run_id(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    results = tuple(
        BatchItemResult.cancelled(
            run_id="other-run" if item.index == 4 else plan.run_id,
            item=item,
            reason=BatchCancellationReason.FAIL_FAST,
        )
        for item in plan.planned_items
    )

    with pytest.raises(BatchContractError) as captured:
        BatchSummary.from_results(plan=plan, results=results, duration_ms=1.0)

    assert captured.value.code == "mismatched_batch_results"


@pytest.mark.parametrize(
    "duration_ms",
    [True, cast(float, "one"), float("nan"), -1.0],
)
def test_batch_summary_rejects_invalid_duration(
    tmp_path: Path,
    duration_ms: float,
) -> None:
    plan = _plan(tmp_path)
    results = tuple(
        BatchItemResult.cancelled(
            run_id=plan.run_id,
            item=item,
            reason=BatchCancellationReason.FAIL_FAST,
        )
        for item in plan.planned_items
    )

    with pytest.raises(BatchContractError) as captured:
        BatchSummary.from_results(
            plan=plan,
            results=results,
            duration_ms=duration_ms,
        )

    assert captured.value.code == "invalid_batch_duration"
