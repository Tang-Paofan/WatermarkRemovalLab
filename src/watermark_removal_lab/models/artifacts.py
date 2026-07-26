"""Exact descriptors for model artifacts reviewed by the project."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final
from urllib.parse import urlsplit

_MODEL_ID_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class ModelDescriptorError(ValueError):
    """Raised when a model artifact descriptor violates storage invariants."""


@dataclass(frozen=True, slots=True)
class ModelArtifactDescriptor:
    """Immutable identity, integrity, and terms metadata for one model artifact."""

    model_id: str
    filename: str
    download_url: str
    source_url: str
    model_card_url: str
    size_bytes: int
    sha256: str
    declared_license: str
    dataset_notice: str
    dataset_terms_url: str

    def __post_init__(self) -> None:
        if _MODEL_ID_PATTERN.fullmatch(self.model_id) is None:
            raise ModelDescriptorError(
                "model_id must use lowercase letters, digits, and single hyphen separators."
            )

        if (
            not self.filename
            or self.filename in {".", ".."}
            or PurePosixPath(self.filename).name != self.filename
            or PureWindowsPath(self.filename).name != self.filename
        ):
            raise ModelDescriptorError("filename must be a single non-empty path component.")

        for field_name, value in (
            ("download_url", self.download_url),
            ("source_url", self.source_url),
            ("model_card_url", self.model_card_url),
            ("dataset_terms_url", self.dataset_terms_url),
        ):
            parsed = urlsplit(value)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ModelDescriptorError(f"{field_name} must be an absolute HTTPS URL.")

        if isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            raise ModelDescriptorError("size_bytes must be a positive integer.")

        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ModelDescriptorError(
                "sha256 must be exactly 64 lowercase hexadecimal characters."
            )

        if not self.declared_license.strip():
            raise ModelDescriptorError("declared_license must not be empty.")
        if not self.dataset_notice.strip():
            raise ModelDescriptorError("dataset_notice must not be empty.")


LAMA_ONNX_FP32: Final = ModelArtifactDescriptor(
    model_id="lama-onnx-fp32",
    filename="lama_fp32.onnx",
    download_url=(
        "https://huggingface.co/Carve/LaMa-ONNX/resolve/"
        "a3ee2fca54baebec351b8fa7786154ffa7555aa6/lama_fp32.onnx?download=true"
    ),
    source_url=(
        "https://huggingface.co/Carve/LaMa-ONNX/blob/"
        "a3ee2fca54baebec351b8fa7786154ffa7555aa6/lama_fp32.onnx"
    ),
    model_card_url=(
        "https://huggingface.co/Carve/LaMa-ONNX/blob/"
        "c3c0c9e468934d62e79c329e35d82dd09ff8c444/README.md"
    ),
    size_bytes=208_044_816,
    sha256="1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
    declared_license="Apache-2.0 (declared by the pinned model card)",
    dataset_notice=(
        "The model card identifies big-lama as trained on Places2. The official Places image "
        "terms limit the data to non-commercial research and education and prohibit image "
        "redistribution. Weight redistribution and commercial use are not approved by this project."
    ),
    dataset_terms_url="https://places2.csail.mit.edu/download-private.html",
)
