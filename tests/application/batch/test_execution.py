"""Tests for sequential B1 execution, cancellation, and durable results."""

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from watermark_removal_lab.application import (
    BatchCancellationToken,
    BatchContractError,
    BatchFailurePolicy,
    BatchItemStatus,
    BatchPlan,
    BatchProgressEvent,
    BatchProgressKind,
    BatchRunError,
    BatchSpec,
    BoxMaskSource,
    CancellationToken,
    ImageBatchItemSpec,
    ImageRemovalInputError,
    ImageRemovalRequest,
    ImageRemovalResult,
    ImageRemovalStatus,
    OverwritePolicy,
    plan_batch,
    run_batch,
)
from watermark_removal_lab.common import Box


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), (240, 240, 240)).save(path)


def _plan(
    tmp_path: Path,
    *,
    count: int = 3,
    invalid_indices: frozenset[int] = frozenset(),
    failure_policy: BatchFailurePolicy = BatchFailurePolicy.CONTINUE,
    overwrite: OverwritePolicy = OverwritePolicy.ERROR,
    save_mask: bool = False,
    results_path: Path | None = None,
) -> BatchPlan:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir(parents=True)
    output_root.mkdir()
    items: list[ImageBatchItemSpec] = []
    for index in range(count):
        name = f"input-{index}.png"
        if index not in invalid_indices:
            _write_image(source_root / name)
        items.append(
            ImageBatchItemSpec(
                f"item-{index}",
                ImageRemovalRequest(
                    input_path=Path(name),
                    output_path=Path(f"nested/{index}/output.png"),
                    mask_source=BoxMaskSource(Box.from_xywh(1, 1, 2, 2)),
                    save_mask_path=(Path(f"masks/{index}/mask.png") if save_mask else None),
                ),
            )
        )
    return plan_batch(
        BatchSpec(
            source_root=source_root,
            output_root=output_root,
            items=tuple(items),
            failure_policy=failure_policy,
            overwrite_policy=overwrite,
            results_path=results_path,
        ),
        run_id="execution-run",
    )


def _service_result(
    request: ImageRemovalRequest,
    *,
    status: ImageRemovalStatus = ImageRemovalStatus.SUCCEEDED,
) -> ImageRemovalResult:
    failed = status is ImageRemovalStatus.FAILED
    return ImageRemovalResult(
        input_path=request.input_path,
        output_path=request.output_path,
        status=status,
        method=request.method,
        radius=float(cast(float, request.radius)),
        dilation_radius=request.dilation_radius,
        mask_threshold=None,
        width=8 if status is ImageRemovalStatus.SUCCEEDED else None,
        height=8 if status is ImageRemovalStatus.SUCCEEDED else None,
        selected_pixels=4 if status is ImageRemovalStatus.SUCCEEDED else None,
        duration_ms=1.25,
        lossy_output=False,
        error_code="simulated" if failed else None,
        error_message="simulated failure" if failed else None,
    )


def _result_payloads(plan: BatchPlan) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in plan.result_file.read_text(encoding="utf-8").splitlines()
    ]


def test_run_batch_executes_public_service_and_persists_state(tmp_path: Path) -> None:
    plan = _plan(tmp_path, count=2, save_mask=True)
    events: list[BatchProgressEvent] = []

    summary = run_batch(plan, progress_sink=events.append)

    assert summary.succeeded == 2
    assert summary.failed == 0
    assert (plan.normalized_spec.output_root / "nested/0/output.png").is_file()
    assert (plan.normalized_spec.output_root / "masks/0/mask.png").is_file()
    run_payload = json.loads(plan.run_file.read_text(encoding="utf-8"))
    result_payloads = _result_payloads(plan)
    summary_payload = json.loads(plan.summary_file.read_text(encoding="utf-8"))
    assert run_payload["counts"] == {"discovered": 2, "validated": 2}
    assert run_payload["execution"] == {
        "failure_policy": "continue",
        "overwrite_policy": "error",
        "worker_count": 1,
    }
    assert run_payload["items"][0]["normalized_request"]["input"] == "input-0.png"
    assert str(tmp_path) not in plan.run_file.read_text(encoding="utf-8")
    assert [payload["status"] for payload in result_payloads] == [
        "succeeded",
        "succeeded",
    ]
    assert summary_payload["counts"]["succeeded"] == 2
    assert [event.kind for event in events] == [
        BatchProgressKind.RUN_STARTED,
        BatchProgressKind.ITEM_STARTED,
        BatchProgressKind.ITEM_COMPLETED,
        BatchProgressKind.ITEM_STARTED,
        BatchProgressKind.ITEM_COMPLETED,
        BatchProgressKind.RUN_COMPLETED,
    ]
    assert events[-2].status is BatchItemStatus.SUCCEEDED
    assert events[-1].completed == events[-1].total == 2


def test_run_batch_uses_custom_result_path_and_skip_policy(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        count=1,
        overwrite=OverwritePolicy.SKIP,
        results_path=Path("reports/nested/items.jsonl"),
    )
    output_path = plan.planned_items[0].request.output_path
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"existing")

    summary = run_batch(plan)

    assert summary.skipped == 1
    assert plan.result_file == plan.normalized_spec.output_root / "reports/nested/items.jsonl"
    assert _result_payloads(plan)[0]["status"] == "skipped"
    assert output_path.read_bytes() == b"existing"


def test_run_batch_continues_after_validation_and_service_failures(tmp_path: Path) -> None:
    plan = _plan(tmp_path, invalid_indices=frozenset({0}))
    calls: list[Path] = []

    def service(request: ImageRemovalRequest) -> ImageRemovalResult:
        calls.append(request.input_path)
        if request.input_path.name == "input-1.png":
            raise ImageRemovalInputError("simulated failure", code="simulated")
        return _service_result(request)

    summary = run_batch(plan, image_service=service)
    payloads = _result_payloads(plan)

    assert summary.succeeded == 1
    assert summary.failed == 2
    assert len(calls) == 2
    assert [payload["status"] for payload in payloads] == [
        "failed",
        "failed",
        "succeeded",
    ]
    assert cast(dict[str, object], payloads[0]["error"])["code"] == "input_not_found"
    assert cast(dict[str, object], payloads[1]["error"])["code"] == "simulated"


@pytest.mark.parametrize("invalid_first", [False, True])
def test_run_batch_fail_fast_cancels_unscheduled_items(
    tmp_path: Path,
    invalid_first: bool,
) -> None:
    plan = _plan(
        tmp_path,
        invalid_indices=frozenset({0}) if invalid_first else frozenset(),
        failure_policy=BatchFailurePolicy.FAIL_FAST,
    )
    calls = 0

    def service(request: ImageRemovalRequest) -> ImageRemovalResult:
        nonlocal calls
        calls += 1
        return _service_result(request, status=ImageRemovalStatus.FAILED)

    summary = run_batch(plan, image_service=service)
    payloads = _result_payloads(plan)

    assert summary.failed == 1
    assert summary.cancelled == 2
    assert calls == (0 if invalid_first else 1)
    assert cast(dict[str, object], payloads[1]["error"])["code"] == "fail_fast"
    assert cast(dict[str, object], payloads[2]["error"])["code"] == "fail_fast"


def test_run_batch_pre_cancel_marks_every_item_user_cancelled(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    token = BatchCancellationToken()
    token.cancel()

    summary = run_batch(
        plan,
        cancellation_token=token,
        image_service=lambda request: pytest.fail(f"unexpected call for {request}"),
    )

    assert token.is_cancelled()
    assert summary.cancelled == 3
    assert all(
        cast(dict[str, object], payload["error"])["code"] == "user_cancelled"
        for payload in _result_payloads(plan)
    )


def test_run_batch_finishes_active_item_then_observes_cancellation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    token = BatchCancellationToken()
    calls = 0

    def service(request: ImageRemovalRequest) -> ImageRemovalResult:
        nonlocal calls
        calls += 1
        token.cancel()
        return _service_result(request)

    summary = run_batch(plan, cancellation_token=token, image_service=service)
    payloads = _result_payloads(plan)

    assert calls == 1
    assert summary.succeeded == 1
    assert summary.cancelled == 2
    assert payloads[0]["status"] == "succeeded"
    assert cast(dict[str, object], payloads[1]["error"])["code"] == "user_cancelled"


def test_run_batch_records_output_parent_failure_as_item_failure(tmp_path: Path) -> None:
    plan = _plan(tmp_path, count=1)
    blocked_parent = plan.normalized_spec.output_root / "nested"
    blocked_parent.write_bytes(b"not a directory")

    summary = run_batch(
        plan,
        image_service=lambda request: pytest.fail(f"unexpected call for {request}"),
    )
    payload = _result_payloads(plan)[0]

    assert summary.failed == 1
    assert cast(dict[str, object], payload["error"])["code"] == "output_directory_failed"


def test_run_batch_revalidates_destination_containment(tmp_path: Path) -> None:
    plan = _plan(tmp_path, count=1)
    item = plan.planned_items[0]
    outside_request = replace(item.request, output_path=tmp_path / "outside.png")
    outside_item = replace(item, request=outside_request)
    changed_plan = replace(plan, planned_items=(outside_item,))

    summary = run_batch(
        changed_plan,
        image_service=lambda request: pytest.fail(f"unexpected call for {request}"),
    )

    assert summary.failed == 1
    assert (
        cast(dict[str, object], _result_payloads(changed_plan)[0]["error"])["code"]
        == "output_directory_failed"
    )


@pytest.mark.parametrize(
    ("service", "code"),
    [
        (lambda request: cast(ImageRemovalResult, object()), "invalid_service_result"),
        (
            lambda request: (_ for _ in ()).throw(RuntimeError("unexpected")),
            "unexpected_item_failure",
        ),
    ],
)
def test_run_batch_rejects_broken_image_services(
    tmp_path: Path,
    service: Callable[[ImageRemovalRequest], ImageRemovalResult],
    code: str,
) -> None:
    plan = _plan(tmp_path, count=1)

    with pytest.raises(BatchRunError) as captured:
        run_batch(plan, image_service=service)

    assert captured.value.code == code
    assert plan.result_file.read_text(encoding="utf-8") == ""
    assert not plan.summary_file.exists()


def test_run_batch_rejects_mismatched_service_result(tmp_path: Path) -> None:
    plan = _plan(tmp_path, count=1)

    def service(request: ImageRemovalRequest) -> ImageRemovalResult:
        return replace(_service_result(request), output_path=tmp_path / "wrong.png")

    with pytest.raises(BatchRunError) as captured:
        run_batch(plan, image_service=service)

    assert captured.value.code == "invalid_service_result"


class _InvalidCancellationToken:
    def is_cancelled(self) -> str:
        return "no"


class _BrokenCancellationToken:
    def is_cancelled(self) -> bool:
        raise RuntimeError("broken token")


@pytest.mark.parametrize("kind", ["plan", "token", "sink", "service"])
def test_run_batch_rejects_invalid_adapters(tmp_path: Path, kind: str) -> None:
    plan = _plan(tmp_path, count=1)

    with pytest.raises(BatchContractError) as captured:
        if kind == "plan":
            run_batch(cast(BatchPlan, object()))
        elif kind == "token":
            run_batch(plan, cancellation_token=cast(CancellationToken, object()))
        elif kind == "sink":
            run_batch(
                plan,
                progress_sink=cast(Callable[[BatchProgressEvent], None], 3),
            )
        else:
            run_batch(
                plan,
                image_service=cast(
                    Callable[[ImageRemovalRequest], ImageRemovalResult],
                    3,
                ),
            )

    expected_codes = {
        "plan": "invalid_batch_plan",
        "token": "invalid_cancellation_token",
        "sink": "invalid_progress_sink",
        "service": "invalid_image_service",
    }
    assert captured.value.code == expected_codes[kind]


def test_run_batch_validates_and_wraps_cancellation_token_failures(tmp_path: Path) -> None:
    invalid_plan = _plan(tmp_path / "invalid", count=1)
    with pytest.raises(BatchContractError) as invalid:
        run_batch(
            invalid_plan,
            cancellation_token=cast(CancellationToken, _InvalidCancellationToken()),
        )
    assert invalid.value.code == "invalid_cancellation_token"
    assert not invalid_plan.state_directory.exists()

    broken_plan = _plan(tmp_path / "broken", count=1)
    with pytest.raises(BatchRunError) as broken:
        run_batch(
            broken_plan,
            cancellation_token=cast(CancellationToken, _BrokenCancellationToken()),
        )
    assert broken.value.code == "cancellation_check_failed"
    assert not broken_plan.state_directory.exists()


def test_progress_sink_failures_are_logged_without_stopping_run(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    plan = _plan(tmp_path, count=1)

    def broken_sink(event: BatchProgressEvent) -> None:
        raise RuntimeError(event.kind.value)

    with caplog.at_level(logging.WARNING):
        summary = run_batch(
            plan,
            image_service=_service_result,
            progress_sink=broken_sink,
        )

    assert summary.succeeded == 1
    assert "batch progress sink failed" in caplog.text


def test_run_batch_cannot_reuse_committed_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path, count=1)
    run_batch(plan, image_service=_service_result)

    with pytest.raises(BatchRunError) as captured:
        run_batch(plan, image_service=_service_result)

    assert captured.value.code == "state_init_failed"
