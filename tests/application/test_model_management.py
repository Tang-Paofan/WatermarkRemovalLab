"""Tests for reviewed-model application services."""

from pathlib import Path
from typing import BinaryIO

import pytest

import watermark_removal_lab.application.model_management as model_management_module
from watermark_removal_lab.application import (
    ModelManagementError,
    UnknownModelError,
    inspect_reviewed_model,
    install_reviewed_model,
    reviewed_model_descriptor,
    reviewed_model_notice,
)
from watermark_removal_lab.models import (
    LAMA_ONNX_FP32,
    ModelArtifactState,
    ModelArtifactStatus,
    ModelCacheError,
)


def _state(path: Path) -> ModelArtifactState:
    return ModelArtifactState(
        descriptor=LAMA_ONNX_FP32,
        path=path,
        status=ModelArtifactStatus.VERIFIED,
        size_bytes=LAMA_ONNX_FP32.size_bytes,
        sha256=LAMA_ONNX_FP32.sha256,
    )


def test_reviewed_model_descriptor_rejects_unknown_id() -> None:
    with pytest.raises(UnknownModelError) as captured:
        reviewed_model_descriptor("unknown-model")

    assert captured.value.code == "unknown_model"
    assert reviewed_model_descriptor(LAMA_ONNX_FP32.model_id) is LAMA_ONNX_FP32


def test_reviewed_model_notice_is_json_compatible() -> None:
    notice = reviewed_model_notice(LAMA_ONNX_FP32.model_id)

    assert notice.to_dict() == {
        "model_id": LAMA_ONNX_FP32.model_id,
        "source_url": LAMA_ONNX_FP32.source_url,
        "model_card_url": LAMA_ONNX_FP32.model_card_url,
        "declared_license": LAMA_ONNX_FP32.declared_license,
        "dataset_notice": LAMA_ONNX_FP32.dataset_notice,
        "dataset_terms_url": LAMA_ONNX_FP32.dataset_terms_url,
        "expected_size_bytes": LAMA_ONNX_FP32.size_bytes,
        "expected_sha256": LAMA_ONNX_FP32.sha256,
    }


def test_inspect_reviewed_model_returns_structured_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_state = _state(tmp_path / "lama_fp32.onnx")
    monkeypatch.setattr(
        model_management_module,
        "inspect_model_artifact",
        lambda descriptor, cache_root: expected_state,
    )

    result = inspect_reviewed_model(LAMA_ONNX_FP32.model_id, cache_root=tmp_path)

    assert result.to_dict()["status"] == "verified"
    assert result.to_dict()["path"] == str(expected_state.path)
    assert result.to_dict()["size_bytes"] == LAMA_ONNX_FP32.size_bytes


def test_inspect_reviewed_model_translates_store_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_inspection(descriptor: object, cache_root: object) -> ModelArtifactState:
        del descriptor, cache_root
        raise ModelCacheError("simulated cache failure")

    monkeypatch.setattr(model_management_module, "inspect_model_artifact", fail_inspection)

    with pytest.raises(ModelManagementError) as captured:
        inspect_reviewed_model(LAMA_ONNX_FP32.model_id, cache_root=tmp_path)

    assert captured.value.code == "model_cache_failed"
    assert isinstance(captured.value.__cause__, ModelCacheError)


def test_install_reviewed_model_forwards_explicit_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_state = _state(tmp_path / "lama_fp32.onnx")
    observed: dict[str, object] = {}

    def fake_install(
        descriptor: object,
        cache_root: object,
        *,
        terms_accepted: bool,
        downloader: object,
    ) -> ModelArtifactState:
        observed.update(
            descriptor=descriptor,
            cache_root=cache_root,
            terms_accepted=terms_accepted,
            downloader=downloader,
        )
        return expected_state

    def downloader(url: str, destination: BinaryIO) -> None:
        del url, destination

    monkeypatch.setattr(model_management_module, "install_model_artifact", fake_install)

    result = install_reviewed_model(
        LAMA_ONNX_FP32.model_id,
        cache_root=tmp_path,
        terms_accepted=True,
        downloader=downloader,
    )

    assert result.status == "verified"
    assert observed == {
        "descriptor": LAMA_ONNX_FP32,
        "cache_root": tmp_path,
        "terms_accepted": True,
        "downloader": downloader,
    }


def test_install_reviewed_model_translates_store_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_install(*args: object, **kwargs: object) -> ModelArtifactState:
        del args, kwargs
        raise ModelCacheError("simulated install failure")

    monkeypatch.setattr(model_management_module, "install_model_artifact", fail_install)

    with pytest.raises(ModelManagementError) as captured:
        install_reviewed_model(
            LAMA_ONNX_FP32.model_id,
            terms_accepted=True,
        )

    assert captured.value.code == "model_cache_failed"
