"""Tests for deterministic B1 batch planning and preflight."""

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import watermark_removal_lab.application.batch.planning as planning_module
from watermark_removal_lab.application import (
    BatchContractError,
    BatchPreflightError,
    BatchSpec,
    BoxMaskSource,
    ImageBatchItemSpec,
    ImageRemovalRequest,
    MaskFileSource,
    OverwritePolicy,
    plan_batch,
)
from watermark_removal_lab.common import Box


def _make_roots(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    output_root.mkdir()
    return source_root, output_root


def _box_request(
    *,
    input_path: Path,
    output_path: Path,
    save_mask_path: Path | None = None,
    radius: float = 3.0,
    dilation_radius: int = 0,
) -> ImageRemovalRequest:
    return ImageRemovalRequest(
        input_path=input_path,
        output_path=output_path,
        mask_source=BoxMaskSource(Box.from_xywh(1, 2, 3, 4)),
        radius=radius,
        dilation_radius=dilation_radius,
        save_mask_path=save_mask_path,
    )


def _spec(
    tmp_path: Path,
    *,
    request: ImageRemovalRequest | None = None,
    overwrite: OverwritePolicy = OverwritePolicy.ERROR,
    results_path: Path | None = None,
) -> BatchSpec:
    source_root, output_root = _make_roots(tmp_path)
    input_path = source_root / "input.png"
    input_path.write_bytes(b"synthetic")
    selected_request = request or _box_request(
        input_path=Path("input.png"),
        output_path=Path("output.png"),
    )
    return BatchSpec(
        source_root=source_root,
        output_root=output_root,
        items=(ImageBatchItemSpec("item-a", selected_request),),
        overwrite_policy=overwrite,
        results_path=results_path,
    )


def _assert_preflight_error(spec: BatchSpec, code: str, *, run_id: str = "run-a") -> None:
    with pytest.raises(BatchPreflightError) as captured:
        plan_batch(spec, run_id=run_id)

    assert captured.value.code == code


def test_plan_batch_normalizes_box_and_mask_items(tmp_path: Path) -> None:
    source_root, output_root = _make_roots(tmp_path)
    (source_root / "a.png").write_bytes(b"a")
    (source_root / "b.jpg").write_bytes(b"b")
    (source_root / "mask.png").write_bytes(b"mask")
    items = (
        ImageBatchItemSpec(
            "box",
            _box_request(
                input_path=Path("a.png"),
                output_path=Path("nested/a.png"),
                save_mask_path=Path("masks/a.png"),
            ),
        ),
        ImageBatchItemSpec(
            "mask",
            ImageRemovalRequest(
                input_path=source_root / "b.jpg",
                output_path=output_root / "b.jpg",
                mask_source=MaskFileSource(Path("mask.png"), threshold=42),
            ),
        ),
    )
    spec = BatchSpec(
        source_root=source_root,
        output_root=output_root,
        items=items,
        overwrite_policy=OverwritePolicy.REPLACE,
    )

    plan = plan_batch(spec, run_id="run_1")

    assert plan.run_id == "run_1"
    assert plan.discovered_count == 2
    assert plan.validated_count == 2
    assert plan.invalid_items == ()
    assert plan.required_resources == ("cpu",)
    assert plan.warnings == ()
    assert plan.result_file == output_root / ".wrl-batch" / "run_1" / "results.jsonl"
    assert plan.run_file.name == "run.json"
    assert plan.summary_file.name == "summary.json"
    assert plan.result_reference == ".wrl-batch/run_1/results.jsonl"
    assert plan.normalized_spec.results_path == plan.result_file
    assert plan.normalized_spec.overwrite_policy is OverwritePolicy.REPLACE

    box_item, mask_item = plan.planned_items
    assert box_item.index == 0
    assert box_item.is_valid
    assert box_item.input_reference == "a.png"
    assert box_item.output_reference == "nested/a.png"
    assert box_item.save_mask_reference == "masks/a.png"
    assert box_item.request.overwrite is OverwritePolicy.REPLACE
    assert box_item.normalized_request()["localization"] == {
        "type": "box",
        "box": [1, 2, 3, 4],
    }
    assert mask_item.index == 1
    assert mask_item.mask_reference == "mask.png"
    assert mask_item.normalized_request()["localization"] == {
        "type": "mask",
        "path": "mask.png",
        "threshold": 42,
    }


def test_plan_batch_generates_a_safe_run_id(tmp_path: Path) -> None:
    plan = plan_batch(_spec(tmp_path))

    assert len(plan.run_id) == 32
    assert plan.run_id.isalnum()


def test_plan_batch_supports_a_separate_mask_root(tmp_path: Path) -> None:
    source_root, output_root = _make_roots(tmp_path)
    mask_root = tmp_path / "masks"
    mask_root.mkdir()
    (source_root / "input.png").write_bytes(b"input")
    (mask_root / "mark.png").write_bytes(b"mask")
    spec = BatchSpec(
        source_root=source_root,
        output_root=output_root,
        mask_root=mask_root,
        items=(
            ImageBatchItemSpec(
                "masked",
                ImageRemovalRequest(
                    input_path=Path("input.png"),
                    output_path=Path("output.png"),
                    mask_source=MaskFileSource(Path("mark.png")),
                ),
            ),
        ),
    )

    plan = plan_batch(spec, run_id="separate-mask-root")

    item = plan.planned_items[0]
    assert item.is_valid
    assert item.mask_reference == "mark.png"
    assert isinstance(item.request.mask_source, MaskFileSource)
    assert item.request.mask_source.path == mask_root / "mark.png"
    assert plan.normalized_spec.mask_root == mask_root


@pytest.mark.parametrize("run_id", ["", "bad/id", cast(str, 123)])
def test_plan_batch_rejects_invalid_run_id(tmp_path: Path, run_id: str) -> None:
    _assert_preflight_error(_spec(tmp_path), "invalid_run_id", run_id=run_id)


def test_plan_batch_rejects_non_spec() -> None:
    with pytest.raises(BatchContractError) as captured:
        plan_batch(cast(BatchSpec, object()), run_id="run-a")

    assert captured.value.code == "invalid_batch_spec"


@pytest.mark.parametrize("role", ["source", "output", "mask"])
def test_plan_batch_requires_existing_directory_roots(tmp_path: Path, role: str) -> None:
    spec = _spec(tmp_path)
    invalid = tmp_path / f"missing-{role}"
    transformed = replace(
        spec,
        source_root=invalid if role == "source" else spec.source_root,
        output_root=invalid if role == "output" else spec.output_root,
        mask_root=invalid if role == "mask" else spec.mask_root,
    )

    _assert_preflight_error(transformed, f"invalid_{role}_root")


def test_path_resolution_failure_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolve(self: Path, *, strict: bool = False) -> Path:
        del self, strict
        raise OSError("simulated")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    with pytest.raises(BatchPreflightError) as captured:
        planning_module._resolve_path(tmp_path)

    assert captured.value.code == "path_resolution_failed"
    assert isinstance(captured.value.__cause__, OSError)


def test_plan_batch_rejects_duplicate_item_ids(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    duplicate = replace(spec.items[0], request=replace(spec.items[0].request))

    _assert_preflight_error(replace(spec, items=(spec.items[0], duplicate)), "duplicate_item_id")


@pytest.mark.parametrize(
    "transform",
    [
        lambda request: replace(request, input_path=Path("../outside.png")),
        lambda request: replace(
            request,
            mask_source=MaskFileSource(Path("../outside-mask.png")),
        ),
        lambda request: replace(request, output_path=Path("../outside.png")),
        lambda request: replace(request, save_mask_path=Path("../outside-mask.png")),
    ],
)
def test_plan_batch_rejects_paths_outside_declared_roots(
    tmp_path: Path,
    transform: Callable[[ImageRemovalRequest], ImageRemovalRequest],
) -> None:
    spec = _spec(tmp_path)
    transformed = replace(
        spec,
        items=(
            replace(
                spec.items[0],
                request=transform(spec.items[0].request),
            ),
        ),
    )

    _assert_preflight_error(transformed, "path_outside_root")


def test_plan_batch_rejects_blocked_state_directory(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    (spec.output_root / ".wrl-batch").write_bytes(b"blocked")

    _assert_preflight_error(spec, "invalid_output_directory")


def test_plan_batch_rejects_existing_run_state(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    (spec.output_root / ".wrl-batch" / "run-a").mkdir(parents=True)

    _assert_preflight_error(spec, "run_id_exists")


def test_plan_batch_rejects_input_output_alias(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    (root / "input.png").write_bytes(b"input")
    spec = BatchSpec(
        source_root=root,
        output_root=root,
        items=(
            ImageBatchItemSpec(
                "item-a",
                _box_request(
                    input_path=Path("input.png"),
                    output_path=Path("input.png"),
                ),
            ),
        ),
    )

    _assert_preflight_error(spec, "in_place_output")


def test_plan_batch_rejects_duplicate_case_normalized_outputs(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    second_input = spec.source_root / "second.png"
    second_input.write_bytes(b"second")
    second = ImageBatchItemSpec(
        "item-b",
        _box_request(
            input_path=Path("second.png"),
            output_path=Path("OUTPUT.PNG"),
        ),
    )

    transformed = replace(spec, items=(spec.items[0], second))
    if os.path.normcase("output.png") == os.path.normcase("OUTPUT.PNG"):
        _assert_preflight_error(transformed, "duplicate_output")
    else:
        assert plan_batch(transformed, run_id="case-sensitive").validated_count == 2


def test_plan_batch_includes_saved_masks_in_output_collision_checks(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    second_input = spec.source_root / "second.png"
    second_input.write_bytes(b"second")
    first = replace(
        spec.items[0],
        request=replace(spec.items[0].request, save_mask_path=Path("shared.png")),
    )
    second = ImageBatchItemSpec(
        "item-b",
        _box_request(
            input_path=Path("second.png"),
            output_path=Path("shared.png"),
        ),
    )

    _assert_preflight_error(replace(spec, items=(first, second)), "duplicate_output")


def test_plan_batch_rejects_outputs_in_reserved_state_directory(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    request = replace(
        spec.items[0].request,
        output_path=Path(".wrl-batch/media.png"),
    )

    _assert_preflight_error(
        replace(spec, items=(replace(spec.items[0], request=request),)),
        "reserved_output_path",
    )


def test_plan_batch_rejects_output_below_a_file(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    (spec.output_root / "blocked").write_bytes(b"file")
    request = replace(
        spec.items[0].request,
        output_path=Path("blocked/nested/output.png"),
    )

    _assert_preflight_error(
        replace(spec, items=(replace(spec.items[0], request=request),)),
        "invalid_output_directory",
    )


def test_plan_batch_rejects_directory_as_output(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    (spec.output_root / "output.png").mkdir()

    _assert_preflight_error(spec, "invalid_output_path")


def test_plan_batch_applies_overwrite_preflight_policy(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    output_path = spec.output_root / "output.png"
    output_path.write_bytes(b"existing")

    _assert_preflight_error(spec, "output_exists")
    assert (
        plan_batch(
            replace(spec, overwrite_policy=OverwritePolicy.SKIP),
            run_id="skip",
        ).validated_count
        == 1
    )
    assert (
        plan_batch(
            replace(spec, overwrite_policy=OverwritePolicy.REPLACE),
            run_id="replace",
        ).validated_count
        == 1
    )


@pytest.mark.parametrize(
    ("transform", "code"),
    [
        (
            lambda request: replace(request, input_path=Path("missing.png")),
            "input_not_found",
        ),
        (
            lambda request: replace(request, input_path=Path("input.webp")),
            "unsupported_input_format",
        ),
        (
            lambda request: replace(
                request,
                mask_source=MaskFileSource(Path("missing-mask.png")),
            ),
            "mask_not_found",
        ),
        (
            lambda request: replace(
                request,
                mask_source=MaskFileSource(Path("mask.webp")),
            ),
            "unsupported_mask_format",
        ),
        (
            lambda request: replace(request, radius=cast(float, True)),
            "invalid_radius",
        ),
        (
            lambda request: replace(request, radius=cast(float, "large")),
            "invalid_radius",
        ),
        (
            lambda request: replace(request, radius=cast(float, object())),
            "invalid_radius",
        ),
        (
            lambda request: replace(request, radius=0),
            "invalid_radius",
        ),
        (
            lambda request: replace(request, radius=float("inf")),
            "invalid_radius",
        ),
        (
            lambda request: replace(request, dilation_radius=cast(int, True)),
            "invalid_dilation",
        ),
        (
            lambda request: replace(request, dilation_radius=cast(int, "one")),
            "invalid_dilation",
        ),
        (
            lambda request: replace(request, dilation_radius=-1),
            "invalid_dilation",
        ),
        (
            lambda request: replace(
                request,
                mask_source=MaskFileSource(
                    Path("mask.png"),
                    threshold=cast(int, True),
                ),
            ),
            "invalid_mask_threshold",
        ),
        (
            lambda request: replace(
                request,
                mask_source=MaskFileSource(
                    Path("mask.png"),
                    threshold=cast(int, "high"),
                ),
            ),
            "invalid_mask_threshold",
        ),
        (
            lambda request: replace(
                request,
                mask_source=MaskFileSource(Path("mask.png"), threshold=256),
            ),
            "invalid_mask_threshold",
        ),
        (
            lambda request: replace(request, output_path=Path("output.webp")),
            "unsupported_output_format",
        ),
        (
            lambda request: replace(request, save_mask_path=Path("mask.jpg")),
            "unsupported_output_format",
        ),
    ],
)
def test_plan_batch_records_item_validation_errors(
    tmp_path: Path,
    transform: Callable[[ImageRemovalRequest], ImageRemovalRequest],
    code: str,
) -> None:
    spec = _spec(tmp_path)
    (spec.source_root / "input.webp").write_bytes(b"unsupported")
    (spec.source_root / "mask.png").write_bytes(b"mask")
    (spec.source_root / "mask.webp").write_bytes(b"unsupported mask")
    request = transform(spec.items[0].request)
    transformed = replace(
        spec,
        items=(replace(spec.items[0], request=request),),
    )

    plan = plan_batch(transformed, run_id="invalid-item")

    assert plan.validated_items == ()
    assert len(plan.invalid_items) == 1
    assert plan.invalid_items[0].validation_error is not None
    assert plan.invalid_items[0].validation_error.code == code
    assert plan.invalid_items[0].normalized_request()["options"]


def test_plan_batch_supports_custom_relative_results_path(tmp_path: Path) -> None:
    spec = _spec(tmp_path, results_path=Path("reports/items.jsonl"))

    plan = plan_batch(spec, run_id="relative")

    assert plan.result_file == spec.output_root / "reports" / "items.jsonl"
    assert plan.result_reference == "reports/items.jsonl"


def test_plan_batch_supports_custom_absolute_results_path(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    result_file = external / "items.jsonl"
    spec = _spec(tmp_path, results_path=result_file)

    plan = plan_batch(spec, run_id="absolute")

    assert plan.result_file == result_file
    assert plan.result_reference == str(result_file)


def test_plan_batch_rejects_invalid_custom_results_extension(tmp_path: Path) -> None:
    spec = _spec(tmp_path, results_path=Path("results.json"))

    _assert_preflight_error(spec, "invalid_results_path")


def test_plan_batch_rejects_blocked_results_parent(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"file")
    transformed = replace(spec, results_path=blocked / "results.jsonl")

    _assert_preflight_error(transformed, "invalid_output_directory")


def test_plan_batch_rejects_result_media_collision(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    request = replace(spec.items[0].request, output_path=Path("results.jsonl"))
    transformed = replace(
        spec,
        items=(replace(spec.items[0], request=request),),
        results_path=Path("results.jsonl"),
    )

    _assert_preflight_error(transformed, "metadata_path_collision")


def test_plan_batch_rejects_existing_custom_results_file(tmp_path: Path) -> None:
    spec = _spec(tmp_path, results_path=Path("results.jsonl"))
    (spec.output_root / "results.jsonl").write_bytes(b"old")

    _assert_preflight_error(
        replace(spec, overwrite_policy=OverwritePolicy.REPLACE),
        "metadata_exists",
    )
