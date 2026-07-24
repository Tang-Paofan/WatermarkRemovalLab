"""Tests for B1 durable state ownership and atomic publication failures."""

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import watermark_removal_lab.application.batch.state as state_module
from watermark_removal_lab.application import (
    BatchCancellationReason,
    BatchItemResult,
    BatchPlan,
    BatchRunError,
    BatchSpec,
    BatchSummary,
    BoxMaskSource,
    ImageBatchItemSpec,
    ImageRemovalRequest,
    plan_batch,
)
from watermark_removal_lab.application.batch.state import BatchStateStore
from watermark_removal_lab.common import Box


def _plan(tmp_path: Path, *, results_path: Path | None = None) -> BatchPlan:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir(parents=True)
    output_root.mkdir()
    (source_root / "input.png").write_bytes(b"synthetic")
    return plan_batch(
        BatchSpec(
            source_root=source_root,
            output_root=output_root,
            results_path=results_path,
            items=(
                ImageBatchItemSpec(
                    "item",
                    ImageRemovalRequest(
                        input_path=Path("input.png"),
                        output_path=Path("output.png"),
                        mask_source=BoxMaskSource(Box.from_xywh(0, 0, 1, 1)),
                    ),
                ),
            ),
        ),
        run_id="state-run",
    )


def _cancelled_result(plan: BatchPlan, *, run_id: str | None = None) -> BatchItemResult:
    return BatchItemResult.cancelled(
        run_id=plan.run_id if run_id is None else run_id,
        item=plan.planned_items[0],
        reason=BatchCancellationReason.USER_CANCELLED,
    )


def _summary(plan: BatchPlan, *, run_id: str | None = None) -> BatchSummary:
    summary = BatchSummary.from_results(
        plan=plan,
        results=(_cancelled_result(plan),),
        duration_ms=1.0,
    )
    return summary if run_id is None else replace(summary, run_id=run_id)


def test_state_store_requires_plan_and_single_initialization(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="BatchPlan"):
        BatchStateStore(cast(BatchPlan, object()))

    plan = _plan(tmp_path)
    store = BatchStateStore(plan)
    store.initialize()

    with pytest.raises(BatchRunError) as captured:
        store.initialize()

    assert captured.value.code == "state_already_initialized"


def test_state_store_requires_initialization_for_writes(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = BatchStateStore(plan)

    with pytest.raises(BatchRunError) as result_error:
        store.append_result(_cancelled_result(plan))
    with pytest.raises(BatchRunError) as summary_error:
        store.write_summary(_summary(plan))

    assert result_error.value.code == "state_not_initialized"
    assert summary_error.value.code == "state_not_initialized"


def test_state_store_rejects_foreign_records(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = BatchStateStore(plan)
    store.initialize()

    with pytest.raises(BatchRunError) as invalid_result_type:
        store.append_result(cast(BatchItemResult, object()))
    with pytest.raises(BatchRunError) as foreign_result:
        store.append_result(_cancelled_result(plan, run_id="other"))
    with pytest.raises(BatchRunError) as invalid_summary_type:
        store.write_summary(cast(BatchSummary, object()))
    with pytest.raises(BatchRunError) as foreign_summary:
        store.write_summary(_summary(plan, run_id="other"))

    assert invalid_result_type.value.code == "invalid_result_record"
    assert foreign_result.value.code == "invalid_result_record"
    assert invalid_summary_type.value.code == "invalid_summary"
    assert foreign_summary.value.code == "invalid_summary"


def test_state_store_appends_result_and_atomically_writes_summary(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = BatchStateStore(plan)
    result = _cancelled_result(plan)
    summary = _summary(plan)

    store.initialize()
    store.append_result(result)
    store.write_summary(summary)

    assert json.loads(plan.result_file.read_text(encoding="utf-8"))["status"] == "cancelled"
    assert json.loads(plan.summary_file.read_text(encoding="utf-8"))["run_id"] == plan.run_id
    assert not tuple(plan.state_directory.glob("*.tmp"))


def test_state_store_reports_state_directory_race(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan.state_directory.mkdir(parents=True)

    with pytest.raises(BatchRunError) as captured:
        BatchStateStore(plan).initialize()

    assert captured.value.code == "state_init_failed"


def test_state_store_reports_result_file_race(tmp_path: Path) -> None:
    result_file = tmp_path / "external" / "items.jsonl"
    result_file.parent.mkdir()
    plan = _plan(tmp_path / "case", results_path=result_file)
    result_file.write_bytes(b"raced")

    with pytest.raises(BatchRunError) as captured:
        BatchStateStore(plan).initialize()

    assert captured.value.code == "result_file_init_failed"
    assert result_file.read_bytes() == b"raced"


def test_state_store_reports_blocked_result_parent(tmp_path: Path) -> None:
    result_file = tmp_path / "external" / "nested" / "items.jsonl"
    (tmp_path / "external").mkdir()
    plan = _plan(tmp_path / "case", results_path=result_file)
    (tmp_path / "external" / "nested").write_bytes(b"blocked")

    with pytest.raises(BatchRunError) as captured:
        BatchStateStore(plan).initialize()

    assert captured.value.code == "result_file_init_failed"


def test_state_store_wraps_run_metadata_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)

    def fail_mkstemp(**kwargs: object) -> tuple[int, str]:
        raise OSError("no temporary file")

    monkeypatch.setattr(tempfile, "mkstemp", fail_mkstemp)

    with pytest.raises(BatchRunError) as captured:
        BatchStateStore(plan).initialize()

    assert captured.value.code == "run_commit_failed"


def test_atomic_writer_removes_temporary_file_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "summary.json"

    def fail_replace(source: Path, target: Path) -> None:
        assert target == destination
        Path(source).unlink()
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(BatchRunError) as captured:
        state_module._write_json_atomic(
            destination,
            {"ok": True},
            error_code="summary_commit_failed",
            description="summary",
        )

    assert captured.value.code == "summary_commit_failed"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_atomic_writer_closes_descriptor_if_stream_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_close = os.close
    closed: list[int] = []

    def fail_fdopen(
        descriptor: int,
        mode: str,
        *,
        encoding: str,
        newline: str,
    ) -> object:
        raise OSError("fdopen failed")

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    monkeypatch.setattr(os, "close", record_close)

    with pytest.raises(BatchRunError) as captured:
        state_module._write_json_atomic(
            tmp_path / "run.json",
            {"ok": True},
            error_code="run_commit_failed",
            description="run",
        )

    assert captured.value.code == "run_commit_failed"
    assert len(closed) == 1


def test_state_store_wraps_append_and_summary_commit_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    append_plan = _plan(tmp_path / "append")
    append_store = BatchStateStore(append_plan)
    append_store.initialize()
    append_plan.result_file.unlink()
    append_plan.result_file.mkdir()

    with pytest.raises(BatchRunError) as append_error:
        append_store.append_result(_cancelled_result(append_plan))
    assert append_error.value.code == "result_commit_failed"

    summary_plan = _plan(tmp_path / "summary")
    summary_store = BatchStateStore(summary_plan)
    summary_store.initialize()
    monkeypatch.setattr(
        os,
        "replace",
        lambda source, target: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(BatchRunError) as summary_error:
        summary_store.write_summary(_summary(summary_plan))
    assert summary_error.value.code == "summary_commit_failed"


def test_state_store_wraps_result_serialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    store = BatchStateStore(plan)
    store.initialize()
    monkeypatch.setattr(
        BatchItemResult,
        "to_json_line",
        lambda self: (_ for _ in ()).throw(ValueError("bad JSON")),
    )

    with pytest.raises(BatchRunError) as captured:
        store.append_result(_cancelled_result(plan))

    assert captured.value.code == "result_commit_failed"
