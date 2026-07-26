"""Offline tests for model cache resolution and atomic artifact installation."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, Never

import pytest

import watermark_removal_lab.models.store as store_module
from watermark_removal_lab.models import (
    LAMA_ONNX_FP32,
    MODEL_CACHE_ENV_VAR,
    ModelArtifactDescriptor,
    ModelArtifactError,
    ModelArtifactStatus,
    ModelCacheError,
    ModelDownloadError,
    ModelIntegrityError,
    ModelTermsNotAcceptedError,
    artifact_path,
    inspect_model_artifact,
    install_model_artifact,
    resolve_model_cache_root,
)


def _fixture_descriptor(payload: bytes = b"fixture-model") -> ModelArtifactDescriptor:
    return replace(
        LAMA_ONNX_FP32,
        model_id="fixture-model",
        filename="fixture.onnx",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _copy_downloader(
    payload: bytes,
    observed_urls: list[str] | None = None,
) -> store_module.ArtifactDownloader:
    def download(url: str, destination: BinaryIO) -> None:
        if observed_urls is not None:
            observed_urls.append(url)
        destination.write(payload)

    return download


def _temporary_downloads(directory: Path) -> list[Path]:
    return list(directory.rglob("*.download"))


def test_model_error_codes_are_stable() -> None:
    assert ModelArtifactError.code == "model_artifact_error"
    assert ModelTermsNotAcceptedError.code == "model_terms_not_accepted"
    assert ModelIntegrityError.code == "model_integrity_failed"
    assert ModelDownloadError.code == "model_download_failed"
    assert ModelCacheError.code == "model_cache_failed"


def test_resolve_model_cache_root_prefers_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv(MODEL_CACHE_ENV_VAR, str(environment_root))

    assert resolve_model_cache_root(explicit_root) == explicit_root


def test_resolve_model_cache_root_uses_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_root = tmp_path / "environment"
    monkeypatch.setenv(MODEL_CACHE_ENV_VAR, str(environment_root))

    assert resolve_model_cache_root() == environment_root


def test_resolve_model_cache_root_uses_platform_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MODEL_CACHE_ENV_VAR, raising=False)
    expected = store_module._platform_cache_root(
        os_name=os.name,
        environ=os.environ,
        home=Path.home(),
    )

    assert resolve_model_cache_root() == expected


@pytest.mark.parametrize(
    ("os_name", "environment", "home", "expected_root"),
    [
        (
            "nt",
            {"LOCALAPPDATA": "C:/cache"},
            Path("C:/home"),
            Path("C:/cache/watermark-removal-lab/models"),
        ),
        (
            "nt",
            {},
            Path("C:/home"),
            Path("C:/home/AppData/Local/watermark-removal-lab/models"),
        ),
        (
            "posix",
            {"XDG_CACHE_HOME": "/cache"},
            Path("/home/test"),
            Path("/cache/watermark-removal-lab/models"),
        ),
        (
            "posix",
            {},
            Path("/home/test"),
            Path("/home/test/.cache/watermark-removal-lab/models"),
        ),
    ],
)
def test_platform_cache_root(
    os_name: str,
    environment: dict[str, str],
    home: Path,
    expected_root: Path,
) -> None:
    assert (
        store_module._platform_cache_root(
            os_name=os_name,
            environ=environment,
            home=home,
        )
        == expected_root
    )


def test_artifact_path_namespaces_model_id(tmp_path: Path) -> None:
    descriptor = _fixture_descriptor()

    assert artifact_path(descriptor, tmp_path) == (tmp_path / "fixture-model" / "fixture.onnx")


def test_inspect_model_artifact_reports_missing_without_creating_cache(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "missing-cache"

    state = inspect_model_artifact(_fixture_descriptor(), cache_root)

    assert state.status is ModelArtifactStatus.MISSING
    assert not state.is_verified
    assert state.size_bytes is None
    assert state.sha256 is None
    assert not cache_root.exists()


def test_inspect_model_artifact_reports_non_file_as_invalid(tmp_path: Path) -> None:
    descriptor = _fixture_descriptor()
    target = artifact_path(descriptor, tmp_path)
    target.mkdir(parents=True)

    state = inspect_model_artifact(descriptor, tmp_path)

    assert state.status is ModelArtifactStatus.INVALID
    assert state.size_bytes is None
    assert state.sha256 is None


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        (b"fixture-model", ModelArtifactStatus.VERIFIED),
        (b"fixture-modeL", ModelArtifactStatus.INVALID),
        (b"short", ModelArtifactStatus.INVALID),
    ],
)
def test_inspect_model_artifact_checks_size_and_digest(
    tmp_path: Path,
    payload: bytes,
    expected_status: ModelArtifactStatus,
) -> None:
    descriptor = _fixture_descriptor()
    target = artifact_path(descriptor, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    state = inspect_model_artifact(descriptor, tmp_path)

    assert state.status is expected_status
    assert state.size_bytes == len(payload)
    assert state.sha256 == hashlib.sha256(payload).hexdigest()
    assert state.is_verified is (expected_status is ModelArtifactStatus.VERIFIED)


def test_inspect_model_artifact_translates_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _fixture_descriptor()
    target = artifact_path(descriptor, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fixture-model")

    def fail_target_open(path: Path, *args: object, **kwargs: object) -> Never:
        del args, kwargs
        assert path == target
        raise OSError("simulated read failure")

    monkeypatch.setattr(Path, "open", fail_target_open)

    with pytest.raises(ModelCacheError, match="Could not inspect") as captured:
        inspect_model_artifact(descriptor, tmp_path)

    assert isinstance(captured.value.__cause__, OSError)


@pytest.mark.parametrize("accepted", [False, 1])
def test_install_model_artifact_requires_literal_true(
    tmp_path: Path,
    accepted: bool,
) -> None:
    cache_root = tmp_path / "cache"

    with pytest.raises(ModelTermsNotAcceptedError, match="explicit"):
        install_model_artifact(
            _fixture_descriptor(),
            cache_root,
            terms_accepted=accepted,
            downloader=_copy_downloader(b"fixture-model"),
        )

    assert not cache_root.exists()


def test_install_model_artifact_downloads_verifies_and_publishes(
    tmp_path: Path,
) -> None:
    payload = b"fixture-model"
    descriptor = _fixture_descriptor(payload)
    observed_urls: list[str] = []

    state = install_model_artifact(
        descriptor,
        tmp_path,
        terms_accepted=True,
        downloader=_copy_downloader(payload, observed_urls),
    )

    assert state.status is ModelArtifactStatus.VERIFIED
    assert state.path.read_bytes() == payload
    assert state.size_bytes == len(payload)
    assert state.sha256 == hashlib.sha256(payload).hexdigest()
    assert observed_urls == [descriptor.download_url]
    assert not _temporary_downloads(tmp_path)


def test_install_model_artifact_reuses_verified_cache(tmp_path: Path) -> None:
    payload = b"fixture-model"
    descriptor = _fixture_descriptor(payload)
    target = artifact_path(descriptor, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    def unexpected_download(url: str, destination: BinaryIO) -> None:
        del url, destination
        raise AssertionError("verified cache must not download")

    state = install_model_artifact(
        descriptor,
        tmp_path,
        terms_accepted=True,
        downloader=unexpected_download,
    )

    assert state.is_verified
    assert state.path == target


def test_install_model_artifact_replaces_invalid_regular_file(tmp_path: Path) -> None:
    payload = b"fixture-model"
    descriptor = _fixture_descriptor(payload)
    target = artifact_path(descriptor, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"invalid")

    install_model_artifact(
        descriptor,
        tmp_path,
        terms_accepted=True,
        downloader=_copy_downloader(payload),
    )

    assert target.read_bytes() == payload


def test_install_model_artifact_rejects_non_file_target(tmp_path: Path) -> None:
    descriptor = _fixture_descriptor()
    target = artifact_path(descriptor, tmp_path)
    target.mkdir(parents=True)

    with pytest.raises(ModelCacheError, match="not a regular file"):
        install_model_artifact(
            descriptor,
            tmp_path,
            terms_accepted=True,
            downloader=_copy_downloader(b"fixture-model"),
        )


def test_install_model_artifact_translates_directory_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        del path, args, kwargs
        raise OSError("simulated mkdir failure")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(ModelCacheError, match="create model cache") as captured:
        install_model_artifact(
            _fixture_descriptor(),
            tmp_path,
            terms_accepted=True,
            downloader=_copy_downloader(b"fixture-model"),
        )

    assert isinstance(captured.value.__cause__, OSError)


def test_install_model_artifact_translates_temporary_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_temporary_file(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("simulated temporary-file failure")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_temporary_file)

    with pytest.raises(ModelCacheError, match="temporary model artifact") as captured:
        install_model_artifact(
            _fixture_descriptor(),
            tmp_path,
            terms_accepted=True,
            downloader=_copy_downloader(b"fixture-model"),
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert not _temporary_downloads(tmp_path)


def test_install_model_artifact_cleans_up_after_download_failure(
    tmp_path: Path,
) -> None:
    def fail_download(url: str, destination: BinaryIO) -> None:
        del url
        destination.write(b"partial")
        raise OSError("simulated transport failure")

    with pytest.raises(ModelDownloadError, match="Could not download") as captured:
        install_model_artifact(
            _fixture_descriptor(),
            tmp_path,
            terms_accepted=True,
            downloader=fail_download,
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert not _temporary_downloads(tmp_path)


def test_install_model_artifact_preserves_model_error_from_downloader(
    tmp_path: Path,
) -> None:
    expected = ModelDownloadError("already translated")

    def fail_download(url: str, destination: BinaryIO) -> None:
        del url, destination
        raise expected

    with pytest.raises(ModelDownloadError) as captured:
        install_model_artifact(
            _fixture_descriptor(),
            tmp_path,
            terms_accepted=True,
            downloader=fail_download,
        )

    assert captured.value is expected
    assert not _temporary_downloads(tmp_path)


def test_install_model_artifact_translates_flush_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fsync(file_descriptor: int) -> None:
        del file_descriptor
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(ModelCacheError, match="temporary model artifact") as captured:
        install_model_artifact(
            _fixture_descriptor(),
            tmp_path,
            terms_accepted=True,
            downloader=_copy_downloader(b"fixture-model"),
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert not _temporary_downloads(tmp_path)


def test_install_model_artifact_rejects_wrong_bytes_without_replacing_existing(
    tmp_path: Path,
) -> None:
    descriptor = _fixture_descriptor()
    target = artifact_path(descriptor, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing-invalid")

    with pytest.raises(ModelIntegrityError, match="failed integrity") as captured:
        install_model_artifact(
            descriptor,
            tmp_path,
            terms_accepted=True,
            downloader=_copy_downloader(b"wrong"),
        )

    assert captured.value.descriptor is descriptor
    assert captured.value.path.suffix == ".download"
    assert captured.value.actual_size_bytes == 5
    assert captured.value.actual_sha256 == hashlib.sha256(b"wrong").hexdigest()
    assert target.read_bytes() == b"existing-invalid"
    assert not _temporary_downloads(tmp_path)


def test_install_model_artifact_cleans_temporary_file_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_replace(source: object, target: object) -> None:
        del source, target
        raise OSError("simulated publish failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(ModelCacheError, match="Could not publish") as captured:
        install_model_artifact(
            _fixture_descriptor(),
            tmp_path,
            terms_accepted=True,
            downloader=_copy_downloader(b"fixture-model"),
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert not _temporary_downloads(tmp_path)


def test_install_model_artifact_preserves_original_error_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_unlink = Path.unlink

    def fail_download(url: str, destination: BinaryIO) -> None:
        del url, destination
        raise OSError("simulated transport failure")

    def fail_temporary_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".download":
            raise OSError("simulated cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", fail_temporary_unlink)
        with pytest.raises(ModelDownloadError) as captured:
            install_model_artifact(
                _fixture_descriptor(),
                tmp_path,
                terms_accepted=True,
                downloader=fail_download,
            )

    assert str(captured.value.__cause__) == "simulated transport failure"
    for temporary in _temporary_downloads(tmp_path):
        temporary.unlink()


def test_default_http_downloader_streams_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"downloaded bytes"
    destination = io.BytesIO()
    observed: list[tuple[str, int]] = []

    def fake_urlopen(url: str, *, timeout: int) -> object:
        observed.append((url, timeout))
        return nullcontext(io.BytesIO(payload))

    monkeypatch.setattr(store_module, "urlopen", fake_urlopen)

    store_module._download_http("https://example.test/model", destination)

    assert destination.getvalue() == payload
    assert observed == [("https://example.test/model", 60)]
