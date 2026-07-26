"""Atomic, integrity-checked storage for reviewed model artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Final, Protocol, cast
from urllib.request import urlopen

from watermark_removal_lab.models.artifacts import ModelArtifactDescriptor

MODEL_CACHE_ENV_VAR: Final = "WRL_MODEL_CACHE"
_PROJECT_CACHE_DIRECTORY: Final = "watermark-removal-lab"
_MODEL_CACHE_DIRECTORY: Final = "models"
_DOWNLOAD_CHUNK_SIZE: Final = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS: Final = 60


class ArtifactDownloader(Protocol):
    """Write the artifact at ``url`` to an already-open binary destination."""

    def __call__(self, url: str, destination: BinaryIO) -> None:
        """Download one artifact without choosing its final filesystem path."""


class ModelArtifactError(RuntimeError):
    """Base class for stable model-storage failures."""

    code = "model_artifact_error"


class ModelTermsNotAcceptedError(ModelArtifactError):
    """Raised before installation when required model terms were not accepted."""

    code = "model_terms_not_accepted"


class ModelIntegrityError(ModelArtifactError):
    """Raised when downloaded bytes do not match the reviewed descriptor."""

    code = "model_integrity_failed"

    def __init__(
        self,
        descriptor: ModelArtifactDescriptor,
        path: Path,
        *,
        actual_size_bytes: int | None,
        actual_sha256: str | None,
    ) -> None:
        self.descriptor = descriptor
        self.path = path
        self.actual_size_bytes = actual_size_bytes
        self.actual_sha256 = actual_sha256
        super().__init__(
            f"Artifact {descriptor.model_id!r} failed integrity verification at {path}: "
            f"expected {descriptor.size_bytes} bytes and SHA-256 {descriptor.sha256}, "
            f"received {actual_size_bytes} bytes and SHA-256 {actual_sha256}."
        )


class ModelDownloadError(ModelArtifactError):
    """Raised when artifact transport fails before verification."""

    code = "model_download_failed"


class ModelCacheError(ModelArtifactError):
    """Raised when the selected model cache cannot be inspected or updated."""

    code = "model_cache_failed"


class ModelArtifactStatus(StrEnum):
    """Integrity state of an artifact at its expected cache path."""

    MISSING = "missing"
    VERIFIED = "verified"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ModelArtifactState:
    """Inspection result for one descriptor at one exact cache path."""

    descriptor: ModelArtifactDescriptor
    path: Path
    status: ModelArtifactStatus
    size_bytes: int | None
    sha256: str | None

    @property
    def is_verified(self) -> bool:
        """Return whether size and digest match the reviewed descriptor."""

        return self.status is ModelArtifactStatus.VERIFIED


def _platform_cache_root(
    *,
    os_name: str,
    environ: Mapping[str, str],
    home: Path,
) -> Path:
    if os_name == "nt":
        platform_root = Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    else:
        platform_root = Path(environ.get("XDG_CACHE_HOME") or home / ".cache")
    return platform_root / _PROJECT_CACHE_DIRECTORY / _MODEL_CACHE_DIRECTORY


def resolve_model_cache_root(cache_root: Path | None = None) -> Path:
    """Resolve an explicit, environment, or platform-specific model cache root."""

    if cache_root is not None:
        return cache_root.expanduser()

    environment_root = os.environ.get(MODEL_CACHE_ENV_VAR)
    if environment_root:
        return Path(environment_root).expanduser()

    return _platform_cache_root(os_name=os.name, environ=os.environ, home=Path.home())


def artifact_path(
    descriptor: ModelArtifactDescriptor,
    cache_root: Path | None = None,
) -> Path:
    """Return the exact path owned by ``descriptor`` without touching the filesystem."""

    return resolve_model_cache_root(cache_root) / descriptor.model_id / descriptor.filename


def _inspect_path(
    descriptor: ModelArtifactDescriptor,
    path: Path,
) -> ModelArtifactState:
    if not path.exists():
        return ModelArtifactState(
            descriptor=descriptor,
            path=path,
            status=ModelArtifactStatus.MISSING,
            size_bytes=None,
            sha256=None,
        )

    if not path.is_file():
        return ModelArtifactState(
            descriptor=descriptor,
            path=path,
            status=ModelArtifactStatus.INVALID,
            size_bytes=None,
            sha256=None,
        )

    try:
        size_bytes = path.stat().st_size
        with path.open("rb") as artifact_file:
            sha256 = hashlib.file_digest(artifact_file, "sha256").hexdigest()
    except OSError as exc:
        raise ModelCacheError(f"Could not inspect model artifact at {path}.") from exc

    status = (
        ModelArtifactStatus.VERIFIED
        if size_bytes == descriptor.size_bytes and sha256 == descriptor.sha256
        else ModelArtifactStatus.INVALID
    )
    return ModelArtifactState(
        descriptor=descriptor,
        path=path,
        status=status,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def inspect_model_artifact(
    descriptor: ModelArtifactDescriptor,
    cache_root: Path | None = None,
) -> ModelArtifactState:
    """Inspect an artifact without creating cache directories or downloading bytes."""

    return _inspect_path(descriptor, artifact_path(descriptor, cache_root))


def _download_http(url: str, destination: BinaryIO) -> None:
    with urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        shutil.copyfileobj(response, destination, length=_DOWNLOAD_CHUNK_SIZE)


def _remove_temporary_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def install_model_artifact(
    descriptor: ModelArtifactDescriptor,
    cache_root: Path | None = None,
    *,
    terms_accepted: bool,
    downloader: ArtifactDownloader | None = None,
) -> ModelArtifactState:
    """Install one reviewed artifact atomically after explicit terms acceptance."""

    if terms_accepted is not True:
        raise ModelTermsNotAcceptedError(
            f"Installation of {descriptor.model_id!r} requires explicit model-terms acceptance."
        )

    target_path = artifact_path(descriptor, cache_root)
    target_directory = target_path.parent
    try:
        target_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelCacheError(
            f"Could not create model cache directory {target_directory}."
        ) from exc

    current_state = _inspect_path(descriptor, target_path)
    if current_state.is_verified:
        return current_state
    if target_path.exists() and not target_path.is_file():
        raise ModelCacheError(f"Model artifact path is not a regular file: {target_path}.")

    selected_downloader = downloader or _download_http
    temporary_path: Path | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{descriptor.filename}.",
                suffix=".download",
                dir=target_directory,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                try:
                    selected_downloader(
                        descriptor.download_url,
                        cast(BinaryIO, temporary_file),
                    )
                except ModelArtifactError:
                    raise
                except Exception as exc:
                    raise ModelDownloadError(
                        f"Could not download model artifact {descriptor.model_id!r}."
                    ) from exc
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        except ModelArtifactError:
            raise
        except OSError as exc:
            raise ModelCacheError(
                f"Could not write a temporary model artifact in {target_directory}."
            ) from exc

        temporary_state = _inspect_path(descriptor, temporary_path)
        if not temporary_state.is_verified:
            raise ModelIntegrityError(
                descriptor,
                temporary_path,
                actual_size_bytes=temporary_state.size_bytes,
                actual_sha256=temporary_state.sha256,
            )

        try:
            os.replace(temporary_path, target_path)
        except OSError as exc:
            raise ModelCacheError(f"Could not publish model artifact at {target_path}.") from exc
        temporary_path = None
        return ModelArtifactState(
            descriptor=descriptor,
            path=target_path,
            status=ModelArtifactStatus.VERIFIED,
            size_bytes=temporary_state.size_bytes,
            sha256=temporary_state.sha256,
        )
    finally:
        _remove_temporary_file(temporary_path)
