"""Application services for one reviewed model artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from watermark_removal_lab.models import (
    LAMA_ONNX_FP32,
    ArtifactDownloader,
    ModelArtifactDescriptor,
    ModelArtifactError,
    ModelArtifactState,
    inspect_model_artifact,
    install_model_artifact,
)


class ModelManagementError(RuntimeError):
    """Stable application-layer failure for model management commands."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class UnknownModelError(ModelManagementError):
    """Raised when an adapter requests a model outside the reviewed registry."""


@dataclass(frozen=True, slots=True)
class ReviewedModelNotice:
    """Exact source, integrity, and terms shown before model installation."""

    model_id: str
    source_url: str
    model_card_url: str
    declared_license: str
    dataset_notice: str
    dataset_terms_url: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible notice."""

        return {
            "model_id": self.model_id,
            "source_url": self.source_url,
            "model_card_url": self.model_card_url,
            "declared_license": self.declared_license,
            "dataset_notice": self.dataset_notice,
            "dataset_terms_url": self.dataset_terms_url,
            "expected_size_bytes": self.size_bytes,
            "expected_sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ModelManagementResult:
    """JSON-compatible state for a reviewed model at its exact cache path."""

    notice: ReviewedModelNotice
    path: Path
    status: str
    size_bytes: int | None
    sha256: str | None

    def to_dict(self) -> dict[str, object]:
        """Return the stable model-management result."""

        return {
            **self.notice.to_dict(),
            "path": str(self.path),
            "status": self.status,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def reviewed_model_descriptor(model_id: str) -> ModelArtifactDescriptor:
    """Return the exact reviewed descriptor selected by a public model ID."""

    if model_id != LAMA_ONNX_FP32.model_id:
        raise UnknownModelError(
            f"Model {model_id!r} is not a reviewed model ID.",
            code="unknown_model",
        )
    return LAMA_ONNX_FP32


def reviewed_model_notice(model_id: str) -> ReviewedModelNotice:
    """Return the notice that must be shown before an explicit installation."""

    descriptor = reviewed_model_descriptor(model_id)
    return ReviewedModelNotice(
        model_id=descriptor.model_id,
        source_url=descriptor.source_url,
        model_card_url=descriptor.model_card_url,
        declared_license=descriptor.declared_license,
        dataset_notice=descriptor.dataset_notice,
        dataset_terms_url=descriptor.dataset_terms_url,
        size_bytes=descriptor.size_bytes,
        sha256=descriptor.sha256,
    )


def _result_from_state(state: ModelArtifactState) -> ModelManagementResult:
    return ModelManagementResult(
        notice=reviewed_model_notice(state.descriptor.model_id),
        path=state.path,
        status=state.status.value,
        size_bytes=state.size_bytes,
        sha256=state.sha256,
    )


def inspect_reviewed_model(
    model_id: str,
    *,
    cache_root: Path | None = None,
) -> ModelManagementResult:
    """Inspect one reviewed artifact without creating directories or downloading."""

    descriptor = reviewed_model_descriptor(model_id)
    try:
        state = inspect_model_artifact(descriptor, cache_root)
    except ModelArtifactError as exc:
        raise ModelManagementError(str(exc), code=exc.code) from exc
    return _result_from_state(state)


def install_reviewed_model(
    model_id: str,
    *,
    cache_root: Path | None = None,
    terms_accepted: bool,
    downloader: ArtifactDownloader | None = None,
) -> ModelManagementResult:
    """Explicitly install and verify one reviewed artifact."""

    descriptor = reviewed_model_descriptor(model_id)
    try:
        state = install_model_artifact(
            descriptor,
            cache_root,
            terms_accepted=terms_accepted,
            downloader=downloader,
        )
    except ModelArtifactError as exc:
        raise ModelManagementError(str(exc), code=exc.code) from exc
    return _result_from_state(state)
