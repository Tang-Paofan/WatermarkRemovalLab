"""Tests for deterministic directory-mode batch input."""

from pathlib import Path
from typing import cast

import pytest

from watermark_removal_lab.application import (
    BatchFailurePolicy,
    BatchInputError,
    BoxMaskSource,
    DirectoryBatchRequest,
    DirectoryOutputFormat,
    MaskFileSource,
    OverwritePolicy,
    build_directory_batch_spec,
    plan_directory_batch,
)
from watermark_removal_lab.common import Box
from watermark_removal_lab.image import OpenCVInpaintMethod


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    return input_directory, output_directory


def _request(tmp_path: Path, **updates: object) -> DirectoryBatchRequest:
    input_directory, output_directory = _roots(tmp_path)
    arguments: dict[str, object] = {
        "input_directory": input_directory,
        "output_directory": output_directory,
        "box": Box.from_xywh(1, 2, 3, 4),
    }
    arguments.update(updates)
    return DirectoryBatchRequest(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"input_directory": cast(Path, "input")}, "invalid_directory_path"),
        ({"output_directory": cast(Path, "output")}, "invalid_directory_path"),
        ({"mask_directory": cast(Path, "masks"), "box": None}, "invalid_directory_path"),
        ({"results_path": cast(Path, "results.jsonl")}, "invalid_results_path"),
        ({"box": None}, "invalid_localization"),
        ({"mask_directory": Path("masks")}, "invalid_localization"),
        ({"box": cast(Box, object())}, "invalid_localization"),
        ({"recursive": cast(bool, 1)}, "invalid_recursive"),
        ({"method": cast(OpenCVInpaintMethod, "telea")}, "invalid_method"),
        ({"radius": cast(float, True)}, "invalid_radius"),
        ({"radius": cast(float, "three")}, "invalid_radius"),
        ({"radius": float("inf")}, "invalid_radius"),
        ({"radius": 10**10000}, "invalid_radius"),
        ({"radius": 0}, "invalid_radius"),
        ({"dilation_radius": cast(int, True)}, "invalid_dilation"),
        ({"dilation_radius": cast(int, "one")}, "invalid_dilation"),
        ({"dilation_radius": -1}, "invalid_dilation"),
        (
            {"output_format": cast(DirectoryOutputFormat, "preserve")},
            "invalid_output_format",
        ),
        ({"overwrite_policy": cast(OverwritePolicy, "error")}, "invalid_overwrite"),
        (
            {"failure_policy": cast(BatchFailurePolicy, "continue")},
            "invalid_failure_policy",
        ),
    ],
)
def test_directory_request_rejects_invalid_contract(
    tmp_path: Path,
    updates: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(BatchInputError) as captured:
        _request(tmp_path, **updates)

    assert captured.value.code == code


def test_directory_request_accepts_mask_mode_and_all_options(tmp_path: Path) -> None:
    input_directory, output_directory = _roots(tmp_path)
    mask_directory = tmp_path / "masks"
    mask_directory.mkdir()

    request = DirectoryBatchRequest(
        input_directory=input_directory,
        output_directory=output_directory,
        mask_directory=mask_directory,
        recursive=True,
        method=OpenCVInpaintMethod.NAVIER_STOKES,
        radius=2,
        dilation_radius=1,
        output_format=DirectoryOutputFormat.PNG,
        overwrite_policy=OverwritePolicy.SKIP,
        failure_policy=BatchFailurePolicy.FAIL_FAST,
        results_path=Path("reports/results.jsonl"),
    )

    assert request.recursive


def test_directory_adapter_discovers_top_level_images_deterministically(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    (request.input_directory / "B.JPG").write_bytes(b"b")
    (request.input_directory / "a.PNG").write_bytes(b"a")
    (request.input_directory / "ignored.webp").write_bytes(b"ignored")
    nested = request.input_directory / "nested"
    nested.mkdir()
    (nested / "nested.jpeg").write_bytes(b"nested")

    spec = build_directory_batch_spec(request)

    assert tuple(item.item_id for item in spec.items) == ("a.PNG", "B.JPG")
    assert tuple(item.request.output_path for item in spec.items) == (
        Path("a.PNG"),
        Path("B.JPG"),
    )
    assert all(isinstance(item.request.mask_source, BoxMaskSource) for item in spec.items)
    assert spec.source_root == request.input_directory
    assert spec.output_root == request.output_directory
    assert spec.mask_root is None


def test_directory_adapter_recurses_maps_png_and_ignores_batch_state(
    tmp_path: Path,
) -> None:
    request = _request(
        tmp_path,
        recursive=True,
        output_format=DirectoryOutputFormat.PNG,
    )
    nested = request.input_directory / "nested"
    nested.mkdir()
    (nested / "photo.JPEG").write_bytes(b"photo")
    state = request.input_directory / ".WRL-BATCH" / "old-run"
    state.mkdir(parents=True)
    (state / "result.png").write_bytes(b"state")

    spec = build_directory_batch_spec(request)

    assert tuple(item.item_id for item in spec.items) == ("nested/photo.JPEG",)
    assert spec.items[0].request.output_path == Path("nested/photo.png")


def test_directory_adapter_pairs_mirrored_masks(tmp_path: Path) -> None:
    input_directory, output_directory = _roots(tmp_path)
    mask_directory = tmp_path / "masks"
    mask_directory.mkdir()
    nested_input = input_directory / "nested"
    nested_mask = mask_directory / "nested"
    nested_input.mkdir()
    nested_mask.mkdir()
    (nested_input / "photo.jpg").write_bytes(b"photo")
    (nested_mask / "photo.png").write_bytes(b"mask")
    request = DirectoryBatchRequest(
        input_directory=input_directory,
        output_directory=output_directory,
        mask_directory=mask_directory,
        recursive=True,
    )

    plan = plan_directory_batch(request, run_id="mask-pair")

    assert plan.validated_count == 1
    item = plan.planned_items[0]
    assert item.mask_reference == "nested/photo.png"
    assert isinstance(item.request.mask_source, MaskFileSource)
    assert item.request.mask_source.path == nested_mask / "photo.png"


def test_directory_adapter_records_missing_mirrored_mask_as_item_failure(
    tmp_path: Path,
) -> None:
    input_directory, output_directory = _roots(tmp_path)
    mask_directory = tmp_path / "masks"
    mask_directory.mkdir()
    (input_directory / "photo.png").write_bytes(b"photo")
    request = DirectoryBatchRequest(
        input_directory=input_directory,
        output_directory=output_directory,
        mask_directory=mask_directory,
    )

    plan = plan_directory_batch(request, run_id="missing-mask")

    assert plan.validated_count == 0
    assert plan.invalid_items[0].validation_error is not None
    assert plan.invalid_items[0].validation_error.code == "mask_not_found"


def test_directory_adapter_applies_batch_policies(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        overwrite_policy=OverwritePolicy.REPLACE,
        failure_policy=BatchFailurePolicy.FAIL_FAST,
        results_path=Path("reports/items.jsonl"),
        method=OpenCVInpaintMethod.NAVIER_STOKES,
        radius=2,
        dilation_radius=1,
    )
    (request.input_directory / "photo.png").write_bytes(b"photo")

    spec = build_directory_batch_spec(request)

    assert spec.overwrite_policy is OverwritePolicy.REPLACE
    assert spec.failure_policy is BatchFailurePolicy.FAIL_FAST
    assert spec.results_path == Path("reports/items.jsonl")
    assert spec.items[0].request.method is OpenCVInpaintMethod.NAVIER_STOKES
    assert spec.items[0].request.radius == 2.0
    assert spec.items[0].request.dilation_radius == 1


def test_directory_adapter_rejects_non_request() -> None:
    with pytest.raises(BatchInputError) as captured:
        build_directory_batch_spec(cast(DirectoryBatchRequest, object()))

    assert captured.value.code == "invalid_directory_request"


@pytest.mark.parametrize("role", ["input", "output", "mask"])
def test_directory_adapter_requires_existing_directories(tmp_path: Path, role: str) -> None:
    input_directory, output_directory = _roots(tmp_path)
    missing = tmp_path / f"missing-{role}"
    arguments: dict[str, object] = {
        "input_directory": missing if role == "input" else input_directory,
        "output_directory": missing if role == "output" else output_directory,
        "box": Box.from_xywh(0, 0, 1, 1),
    }
    if role == "mask":
        arguments["box"] = None
        arguments["mask_directory"] = missing
    request = DirectoryBatchRequest(**arguments)  # type: ignore[arg-type]

    with pytest.raises(BatchInputError) as captured:
        build_directory_batch_spec(request)

    assert captured.value.code == f"invalid_{role}_directory"


def test_directory_adapter_translates_resolution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)

    def fail_resolve(self: Path, *, strict: bool = False) -> Path:
        del self, strict
        raise OSError("simulated")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    with pytest.raises(BatchInputError) as captured:
        build_directory_batch_spec(request)

    assert captured.value.code == "directory_resolution_failed"
    assert isinstance(captured.value.__cause__, OSError)


def test_directory_adapter_translates_discovery_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)

    def fail_iterdir(self: Path) -> object:
        del self
        raise OSError("simulated")

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)

    with pytest.raises(BatchInputError) as captured:
        build_directory_batch_spec(request)

    assert captured.value.code == "directory_discovery_failed"
    assert isinstance(captured.value.__cause__, OSError)


def test_directory_adapter_rejects_output_inside_input(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    output_directory = input_directory / "output"
    output_directory.mkdir()
    request = DirectoryBatchRequest(
        input_directory=input_directory,
        output_directory=output_directory,
        box=Box.from_xywh(0, 0, 1, 1),
    )

    with pytest.raises(BatchInputError) as captured:
        build_directory_batch_spec(request)

    assert captured.value.code == "output_inside_input"


def test_directory_adapter_allows_output_outside_input(tmp_path: Path) -> None:
    request = _request(tmp_path)
    (request.input_directory / "photo.png").write_bytes(b"photo")

    assert build_directory_batch_spec(request).items


def test_directory_adapter_rejects_empty_discovery(tmp_path: Path) -> None:
    request = _request(tmp_path)
    (request.input_directory / "ignored.txt").write_text("ignored", encoding="utf-8")

    with pytest.raises(BatchInputError) as captured:
        build_directory_batch_spec(request)

    assert captured.value.code == "no_input_files"
