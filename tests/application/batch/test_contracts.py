"""Tests for framework-neutral B1 batch contracts."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from watermark_removal_lab.application import (
    BatchContractError,
    BatchFailurePolicy,
    BatchItemError,
    BatchMedia,
    BatchOperation,
    BatchSpec,
    BoxMaskSource,
    ImageBatchItemSpec,
    ImageRemovalRequest,
    MaskFileSource,
    OverwritePolicy,
)
from watermark_removal_lab.common import Box
from watermark_removal_lab.image import OpenCVInpaintMethod


def _request(tmp_path: Path) -> ImageRemovalRequest:
    return ImageRemovalRequest(
        input_path=tmp_path / "input.png",
        output_path=tmp_path / "output.png",
        mask_source=BoxMaskSource(Box.from_xywh(0, 0, 1, 1)),
    )


def _item(tmp_path: Path) -> ImageBatchItemSpec:
    return ImageBatchItemSpec(item_id="item-a", request=_request(tmp_path))


def test_batch_item_error_has_stable_machine_shape() -> None:
    error = BatchItemError(code="invalid", message="bad item", category="validation")

    assert error.to_dict() == {
        "code": "invalid",
        "message": "bad item",
        "category": "validation",
    }


@pytest.mark.parametrize("item_id", ["", "   ", cast(str, 12)])
def test_image_batch_item_rejects_invalid_id(tmp_path: Path, item_id: str) -> None:
    with pytest.raises(BatchContractError) as captured:
        ImageBatchItemSpec(item_id=item_id, request=_request(tmp_path))

    assert captured.value.code == "invalid_item_id"


def test_image_batch_item_rejects_non_request(tmp_path: Path) -> None:
    with pytest.raises(BatchContractError) as captured:
        ImageBatchItemSpec(
            item_id="item-a",
            request=cast(ImageRemovalRequest, object()),
        )

    assert captured.value.code == "invalid_item_request"


@pytest.mark.parametrize(
    ("transform", "code"),
    [
        (
            lambda request: replace(request, input_path=cast(Path, "input.png")),
            "invalid_path",
        ),
        (
            lambda request: replace(request, output_path=cast(Path, "output.png")),
            "invalid_path",
        ),
        (
            lambda request: replace(
                request,
                mask_source=cast(BoxMaskSource, object()),
            ),
            "invalid_localization",
        ),
        (
            lambda request: replace(
                request,
                mask_source=BoxMaskSource(cast(Box, object())),
            ),
            "invalid_localization",
        ),
        (
            lambda request: replace(
                request,
                mask_source=MaskFileSource(cast(Path, "mask.png")),
            ),
            "invalid_path",
        ),
        (
            lambda request: replace(
                request,
                method=cast(OpenCVInpaintMethod, "telea"),
            ),
            "invalid_method",
        ),
        (
            lambda request: replace(
                request,
                save_mask_path=cast(Path, "mask.png"),
            ),
            "invalid_path",
        ),
    ],
)
def test_image_batch_item_rejects_invalid_request_structure(
    tmp_path: Path,
    transform: Callable[[ImageRemovalRequest], ImageRemovalRequest],
    code: str,
) -> None:
    with pytest.raises(BatchContractError) as captured:
        ImageBatchItemSpec(
            item_id="item-a",
            request=transform(_request(tmp_path)),
        )

    assert captured.value.code == code


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"schema_version": 2}, "unsupported_schema_version"),
        ({"schema_version": True}, "unsupported_schema_version"),
        ({"schema_version": 1.0}, "unsupported_schema_version"),
        ({"media": cast(BatchMedia, "video")}, "unsupported_media"),
        ({"operation": cast(BatchOperation, "detect")}, "unsupported_operation"),
        ({"source_root": cast(Path, ".")}, "invalid_path"),
        ({"output_root": cast(Path, ".")}, "invalid_path"),
        ({"mask_root": cast(Path, ".")}, "invalid_path"),
        ({"protected_paths": []}, "invalid_path"),
        ({"protected_paths": (cast(Path, "manifest.jsonl"),)}, "invalid_path"),
        ({"results_path": cast(Path, "results.jsonl")}, "invalid_path"),
        ({"items": ()}, "empty_batch"),
        ({"items": []}, "empty_batch"),
        ({"items": (cast(ImageBatchItemSpec, object()),)}, "invalid_batch_item"),
        ({"overwrite_policy": cast(OverwritePolicy, "error")}, "invalid_overwrite"),
        (
            {"failure_policy": cast(BatchFailurePolicy, "continue")},
            "invalid_failure_policy",
        ),
        ({"worker_count": True}, "unsupported_worker_count"),
        ({"worker_count": 1.0}, "unsupported_worker_count"),
        ({"worker_count": 2}, "unsupported_worker_count"),
    ],
)
def test_batch_spec_rejects_invalid_contract(
    tmp_path: Path,
    updates: dict[str, object],
    code: str,
) -> None:
    arguments: dict[str, object] = {
        "source_root": tmp_path,
        "output_root": tmp_path,
        "items": (_item(tmp_path),),
    }
    arguments.update(updates)

    with pytest.raises(BatchContractError) as captured:
        BatchSpec(**arguments)  # type: ignore[arg-type]

    assert captured.value.code == code


def test_batch_spec_accepts_all_b1_contract_values(tmp_path: Path) -> None:
    spec = BatchSpec(
        source_root=tmp_path,
        output_root=tmp_path,
        items=(_item(tmp_path),),
        results_path=tmp_path / "results.jsonl",
        media=BatchMedia.IMAGE,
        operation=BatchOperation.REMOVE,
        overwrite_policy=OverwritePolicy.SKIP,
        failure_policy=BatchFailurePolicy.FAIL_FAST,
    )

    assert spec.worker_count == 1
