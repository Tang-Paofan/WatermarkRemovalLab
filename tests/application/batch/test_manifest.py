"""Tests for strict versioned JSON Lines batch manifests."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from watermark_removal_lab.application import (
    BatchFailurePolicy,
    BatchInputError,
    BatchPreflightError,
    BoxMaskSource,
    ManifestBatchRequest,
    MaskFileSource,
    OverwritePolicy,
    build_manifest_batch_spec,
    plan_manifest_batch,
)
from watermark_removal_lab.image import OpenCVInpaintMethod


def _header(**updates: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record": "batch",
        "schema_version": 1,
        "media": "image",
        "operation": "remove",
    }
    record.update(updates)
    return record


def _item(**updates: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record": "item",
        "id": "item-a",
        "input": "inputs/a.png",
        "output": "a.png",
        "box": [1, 2, 3, 4],
    }
    record.update(updates)
    return record


def _write_manifest(path: Path, records: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )


def _request(tmp_path: Path, records: Sequence[object]) -> ManifestBatchRequest:
    manifest_path = tmp_path / "config" / "batch.jsonl"
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    _write_manifest(manifest_path, records)
    return ManifestBatchRequest(
        manifest_path=manifest_path,
        output_directory=output_directory,
    )


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"manifest_path": cast(Path, "batch.jsonl")}, "invalid_manifest_path"),
        ({"output_directory": cast(Path, "output")}, "invalid_manifest_path"),
        ({"results_path": cast(Path, "results.jsonl")}, "invalid_results_path"),
        ({"overwrite_policy": cast(OverwritePolicy, "error")}, "invalid_overwrite"),
        (
            {"failure_policy": cast(BatchFailurePolicy, "continue")},
            "invalid_failure_policy",
        ),
    ],
)
def test_manifest_request_rejects_invalid_contract(
    tmp_path: Path,
    updates: dict[str, object],
    code: str,
) -> None:
    arguments: dict[str, object] = {
        "manifest_path": tmp_path / "batch.jsonl",
        "output_directory": tmp_path,
    }
    arguments.update(updates)

    with pytest.raises(BatchInputError) as captured:
        ManifestBatchRequest(**arguments)  # type: ignore[arg-type]

    assert captured.value.code == code


def test_manifest_request_accepts_all_policies(tmp_path: Path) -> None:
    request = ManifestBatchRequest(
        manifest_path=tmp_path / "batch.jsonl",
        output_directory=tmp_path,
        results_path=tmp_path / "results.jsonl",
        overwrite_policy=OverwritePolicy.REPLACE,
        failure_policy=BatchFailurePolicy.FAIL_FAST,
    )

    assert request.failure_policy is BatchFailurePolicy.FAIL_FAST


def test_manifest_adapter_applies_defaults_and_item_overrides(tmp_path: Path) -> None:
    records = [
        _header(
            defaults={
                "method": "telea",
                "radius": 4,
                "dilate": 1,
                "mask_threshold": 100,
            }
        ),
        _item(
            id="box",
            method="ns",
            radius=2.5,
            dilate=2,
        ),
        _item(
            id="mask",
            input="inputs/b.jpg",
            output="nested/b.png",
            box=None,
            mask="masks/b.png",
            mask_threshold=42,
        ),
    ]
    records[2].pop("box")
    request = _request(tmp_path, records)
    manifest_directory = request.manifest_path.parent
    (manifest_directory / "inputs").mkdir()
    (manifest_directory / "masks").mkdir()
    (manifest_directory / "inputs" / "a.png").write_bytes(b"a")
    (manifest_directory / "inputs" / "b.jpg").write_bytes(b"b")
    (manifest_directory / "masks" / "b.png").write_bytes(b"mask")

    plan = plan_manifest_batch(
        replace(
            request,
            overwrite_policy=OverwritePolicy.SKIP,
            failure_policy=BatchFailurePolicy.FAIL_FAST,
            results_path=Path("reports/results.jsonl"),
        ),
        run_id="manifest-run",
    )

    assert tuple(item.item_id for item in plan.planned_items) == ("box", "mask")
    assert plan.validated_count == 2
    assert plan.normalized_spec.protected_paths == (request.manifest_path,)
    assert plan.normalized_spec.overwrite_policy is OverwritePolicy.SKIP
    assert plan.normalized_spec.failure_policy is BatchFailurePolicy.FAIL_FAST
    box_item, mask_item = plan.planned_items
    assert isinstance(box_item.request.mask_source, BoxMaskSource)
    assert box_item.request.method is OpenCVInpaintMethod.NAVIER_STOKES
    assert box_item.request.radius == 2.5
    assert box_item.request.dilation_radius == 2
    assert isinstance(mask_item.request.mask_source, MaskFileSource)
    assert mask_item.request.method is OpenCVInpaintMethod.TELEA
    assert mask_item.request.radius == 4.0
    assert mask_item.request.dilation_radius == 1
    assert mask_item.request.mask_source.threshold == 42
    assert mask_item.output_reference == "nested/b.png"


def test_manifest_adapter_uses_documented_defaults(tmp_path: Path) -> None:
    request = _request(tmp_path, [_header(), _item()])

    spec = build_manifest_batch_spec(request)
    image_request = spec.items[0].request

    assert image_request.method is OpenCVInpaintMethod.TELEA
    assert image_request.radius == 3.0
    assert image_request.dilation_radius == 0


def test_manifest_adapter_rejects_non_request() -> None:
    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(cast(ManifestBatchRequest, object()))

    assert captured.value.code == "invalid_manifest_request"


def test_manifest_adapter_rejects_missing_manifest(tmp_path: Path) -> None:
    request = ManifestBatchRequest(
        manifest_path=tmp_path / "missing.jsonl",
        output_directory=tmp_path,
    )

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "manifest_not_found"


def test_manifest_adapter_translates_resolution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ManifestBatchRequest(
        manifest_path=tmp_path / "batch.jsonl",
        output_directory=tmp_path,
    )

    def fail_resolve(self: Path, *, strict: bool = False) -> Path:
        del self, strict
        raise OSError("simulated")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "manifest_read_failed"
    assert isinstance(captured.value.__cause__, OSError)


def test_manifest_adapter_translates_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, [_header(), _item()])

    def fail_read_text(self: Path, *, encoding: str) -> str:
        del self, encoding
        raise OSError("simulated")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "manifest_read_failed"
    assert isinstance(captured.value.__cause__, OSError)


def test_manifest_adapter_rejects_non_utf8_file(tmp_path: Path) -> None:
    request = _request(tmp_path, [_header(), _item()])
    request.manifest_path.write_bytes(b"\xff")

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "manifest_read_failed"
    assert isinstance(captured.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize(
    ("content", "code", "line_number"),
    [
        ("", "empty_manifest", None),
        (
            f"{json.dumps(_header())}\n\n{json.dumps(_item())}\n",
            "invalid_manifest_json",
            2,
        ),
        ("{\n", "invalid_manifest_json", 1),
        (
            '{"record":"batch","record":"batch","schema_version":1,'
            '"media":"image","operation":"remove"}\n',
            "invalid_manifest_json",
            1,
        ),
        (
            '{"record":"batch","schema_version":NaN,"media":"image","operation":"remove"}\n',
            "invalid_manifest_json",
            1,
        ),
        ("[]\n", "invalid_manifest_record", 1),
    ],
)
def test_manifest_adapter_rejects_invalid_json_lines(
    tmp_path: Path,
    content: str,
    code: str,
    line_number: int | None,
) -> None:
    manifest_path = tmp_path / "batch.jsonl"
    manifest_path.write_text(content, encoding="utf-8")
    request = ManifestBatchRequest(
        manifest_path=manifest_path,
        output_directory=tmp_path,
    )

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == code
    assert captured.value.line_number == line_number


@pytest.mark.parametrize(
    ("header", "code"),
    [
        (_header(extra=True), "unknown_manifest_field"),
        ({"record": "batch"}, "missing_manifest_field"),
        (_header(record="item"), "invalid_manifest_record"),
        (_header(schema_version=True), "unsupported_schema_version"),
        (_header(schema_version=1.0), "unsupported_schema_version"),
        (_header(schema_version=2), "unsupported_schema_version"),
        (_header(media="video"), "unsupported_media"),
        (_header(operation="detect"), "unsupported_operation"),
        (_header(defaults=[]), "invalid_manifest_defaults"),
        (_header(defaults={"unknown": 1}), "unknown_manifest_field"),
        (_header(defaults={"method": 1}), "invalid_manifest_method"),
        (_header(defaults={"method": "unknown"}), "invalid_manifest_method"),
        (_header(defaults={"radius": True}), "invalid_manifest_radius"),
        (_header(defaults={"radius": "three"}), "invalid_manifest_radius"),
        (_header(defaults={"radius": float("inf")}), "invalid_manifest_json"),
        (_header(defaults={"radius": 10**1000}), "invalid_manifest_radius"),
        (_header(defaults={"radius": 0}), "invalid_manifest_radius"),
        (_header(defaults={"dilate": True}), "invalid_manifest_dilation"),
        (_header(defaults={"dilate": "one"}), "invalid_manifest_dilation"),
        (_header(defaults={"dilate": -1}), "invalid_manifest_dilation"),
        (_header(defaults={"mask_threshold": True}), "invalid_manifest_threshold"),
        (_header(defaults={"mask_threshold": "high"}), "invalid_manifest_threshold"),
        (_header(defaults={"mask_threshold": -1}), "invalid_manifest_threshold"),
        (_header(defaults={"mask_threshold": 256}), "invalid_manifest_threshold"),
    ],
)
def test_manifest_adapter_rejects_invalid_batch_record(
    tmp_path: Path,
    header: dict[str, Any],
    code: str,
) -> None:
    request = _request(tmp_path, [header, _item()])

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == code
    assert captured.value.line_number == 1


@pytest.mark.parametrize(
    ("item", "code"),
    [
        (_item(extra=True), "unknown_manifest_field"),
        ({"record": "item", "id": "a", "input": "a.png"}, "missing_manifest_field"),
        (_item(record="batch"), "invalid_manifest_record"),
        (_item(id=1), "invalid_manifest_item_id"),
        (_item(id="  "), "invalid_manifest_item_id"),
        (
            {
                "record": "item",
                "id": "a",
                "input": "a.png",
                "output": "a.png",
            },
            "invalid_manifest_localization",
        ),
        (_item(mask="mask.png"), "invalid_manifest_localization"),
        (_item(method=1), "invalid_manifest_method"),
        (_item(method="unknown"), "invalid_manifest_method"),
        (_item(radius=True), "invalid_manifest_radius"),
        (_item(radius="three"), "invalid_manifest_radius"),
        (_item(radius=float("inf")), "invalid_manifest_json"),
        (_item(radius=0), "invalid_manifest_radius"),
        (_item(dilate=True), "invalid_manifest_dilation"),
        (_item(dilate="one"), "invalid_manifest_dilation"),
        (_item(dilate=-1), "invalid_manifest_dilation"),
        (_item(mask_threshold=True), "invalid_manifest_threshold"),
        (_item(mask_threshold="high"), "invalid_manifest_threshold"),
        (_item(mask_threshold=-1), "invalid_manifest_threshold"),
        (_item(mask_threshold=256), "invalid_manifest_threshold"),
        (_item(box="1,2,3,4"), "invalid_manifest_box"),
        (_item(box=[1, 2, 3]), "invalid_manifest_box"),
        (_item(box=[1, 2, True, 4]), "invalid_manifest_box"),
        (_item(box=[-1, 2, 3, 4]), "invalid_manifest_box"),
    ],
)
def test_manifest_adapter_rejects_invalid_item_record(
    tmp_path: Path,
    item: dict[str, Any],
    code: str,
) -> None:
    request = _request(tmp_path, [_header(), item])

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == code
    assert captured.value.line_number == 2


@pytest.mark.parametrize(
    "value",
    [1, "", "a\\b.png", "bad\x00name.png", "/absolute.png", "C:/absolute.png", "."],
)
def test_manifest_adapter_rejects_nonportable_paths(tmp_path: Path, value: object) -> None:
    request = _request(tmp_path, [_header(), _item(input=value)])

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "invalid_manifest_path"
    assert captured.value.line_number == 2


def test_manifest_adapter_rejects_invalid_mask_path(tmp_path: Path) -> None:
    item = _item(mask="masks\\a.png")
    item.pop("box")
    request = _request(tmp_path, [_header(), item])

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "invalid_manifest_path"


def test_manifest_adapter_rejects_invalid_output_path(tmp_path: Path) -> None:
    request = _request(tmp_path, [_header(), _item(output="/output.png")])

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "invalid_manifest_path"


def test_manifest_adapter_rejects_duplicate_item_ids(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        [_header(), _item(), _item(input="inputs/b.png", output="b.png")],
    )

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "duplicate_item_id"
    assert captured.value.line_number == 3


def test_manifest_adapter_rejects_batch_without_items(tmp_path: Path) -> None:
    request = _request(tmp_path, [_header()])

    with pytest.raises(BatchInputError) as captured:
        build_manifest_batch_spec(request)

    assert captured.value.code == "empty_batch"


def test_manifest_plan_rejects_path_traversal(tmp_path: Path) -> None:
    request = _request(tmp_path, [_header(), _item(input="../outside.png")])

    with pytest.raises(BatchPreflightError) as captured:
        plan_manifest_batch(request, run_id="traversal")

    assert captured.value.code == "path_outside_root"


def test_manifest_plan_rejects_duplicate_outputs(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        [
            _header(),
            _item(id="a"),
            _item(id="b", input="inputs/b.png"),
        ],
    )

    with pytest.raises(BatchPreflightError) as captured:
        plan_manifest_batch(request, run_id="collision")

    assert captured.value.code == "duplicate_output"


def test_manifest_plan_protects_manifest_from_media_output(tmp_path: Path) -> None:
    manifest_path = tmp_path / "batch.jsonl"
    _write_manifest(
        manifest_path,
        [_header(), _item(input="input.png", output="batch.jsonl")],
    )
    (tmp_path / "input.png").write_bytes(b"input")
    request = ManifestBatchRequest(
        manifest_path=manifest_path,
        output_directory=tmp_path,
        overwrite_policy=OverwritePolicy.REPLACE,
    )

    with pytest.raises(BatchPreflightError) as captured:
        plan_manifest_batch(request, run_id="protect-media")

    assert captured.value.code == "in_place_output"


def test_manifest_plan_protects_manifest_from_results_output(tmp_path: Path) -> None:
    request = _request(tmp_path, [_header(), _item()])
    protected = replace(
        request,
        results_path=request.manifest_path,
        overwrite_policy=OverwritePolicy.REPLACE,
    )

    with pytest.raises(BatchPreflightError) as captured:
        plan_manifest_batch(protected, run_id="protect-results")

    assert captured.value.code == "metadata_path_collision"
