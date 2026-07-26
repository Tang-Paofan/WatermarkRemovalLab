"""Tests for exact reviewed model artifact descriptors."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from watermark_removal_lab.models import (
    LAMA_ONNX_FP32,
    ModelArtifactDescriptor,
    ModelDescriptorError,
)


def test_lama_onnx_fp32_descriptor_is_pinned() -> None:
    descriptor = LAMA_ONNX_FP32

    assert descriptor.model_id == "lama-onnx-fp32"
    assert descriptor.filename == "lama_fp32.onnx"
    assert "a3ee2fca54baebec351b8fa7786154ffa7555aa6" in descriptor.download_url
    assert "a3ee2fca54baebec351b8fa7786154ffa7555aa6" in descriptor.source_url
    assert "c3c0c9e468934d62e79c329e35d82dd09ff8c444" in descriptor.model_card_url
    assert descriptor.size_bytes == 208_044_816
    assert descriptor.sha256 == "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
    assert descriptor.declared_license.startswith("Apache-2.0")
    assert "non-commercial research and education" in descriptor.dataset_notice
    assert descriptor.dataset_terms_url == "https://places2.csail.mit.edu/download-private.html"


@pytest.mark.parametrize("model_id", ["", "Uppercase", "-leading", "two--hyphens", "space id"])
def test_descriptor_rejects_invalid_model_id(model_id: str) -> None:
    with pytest.raises(ModelDescriptorError, match="model_id"):
        replace(LAMA_ONNX_FP32, model_id=model_id)


@pytest.mark.parametrize("filename", ["", ".", "..", "nested/model.onnx", r"nested\model.onnx"])
def test_descriptor_rejects_non_leaf_filename(filename: str) -> None:
    with pytest.raises(ModelDescriptorError, match="filename"):
        replace(LAMA_ONNX_FP32, filename=filename)


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (
            lambda: replace(
                LAMA_ONNX_FP32,
                download_url="http://example.test/model.onnx",
            ),
            "download_url",
        ),
        (
            lambda: replace(LAMA_ONNX_FP32, source_url="https:///missing-host"),
            "source_url",
        ),
        (
            lambda: replace(LAMA_ONNX_FP32, model_card_url="relative/model-card"),
            "model_card_url",
        ),
        (
            lambda: replace(LAMA_ONNX_FP32, dataset_terms_url=""),
            "dataset_terms_url",
        ),
    ],
)
def test_descriptor_rejects_non_https_metadata_url(
    factory: Callable[[], ModelArtifactDescriptor],
    field_name: str,
) -> None:
    with pytest.raises(ModelDescriptorError, match=field_name):
        factory()


@pytest.mark.parametrize("size_bytes", [True, 0, -1])
def test_descriptor_rejects_invalid_size(size_bytes: int) -> None:
    with pytest.raises(ModelDescriptorError, match="size_bytes"):
        replace(LAMA_ONNX_FP32, size_bytes=size_bytes)


@pytest.mark.parametrize(
    "sha256",
    [
        "",
        "0" * 63,
        "0" * 65,
        "A" * 64,
        "z" * 64,
    ],
)
def test_descriptor_rejects_invalid_sha256(sha256: str) -> None:
    with pytest.raises(ModelDescriptorError, match="sha256"):
        replace(LAMA_ONNX_FP32, sha256=sha256)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: replace(LAMA_ONNX_FP32, declared_license=" \t"),
            "declared_license",
        ),
        (
            lambda: replace(LAMA_ONNX_FP32, dataset_notice=" \t"),
            "dataset_notice",
        ),
    ],
)
def test_descriptor_rejects_empty_required_notice(
    factory: Callable[[], ModelArtifactDescriptor],
    message: str,
) -> None:
    with pytest.raises(ModelDescriptorError, match=message):
        factory()


def test_descriptor_can_represent_a_small_reviewed_fixture() -> None:
    descriptor = ModelArtifactDescriptor(
        model_id="fixture-model",
        filename="fixture.onnx",
        download_url="https://example.test/fixture.onnx",
        source_url="https://example.test/source/fixture.onnx",
        model_card_url="https://example.test/model-card",
        size_bytes=7,
        sha256="0" * 64,
        declared_license="Test-only",
        dataset_notice="Synthetic test bytes only.",
        dataset_terms_url="https://example.test/terms",
    )

    assert descriptor.filename == "fixture.onnx"
