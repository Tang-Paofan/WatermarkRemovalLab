"""Offline tests for lazy ONNX Runtime loading and LaMa session ownership."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

import watermark_removal_lab.models.runtime as runtime_module
from watermark_removal_lab.models import (
    LAMA_INPUT_CONTRACTS,
    LAMA_ONNX_FP32,
    LAMA_OUTPUT_CONTRACTS,
    InsufficientMemoryError,
    LamaOnnxSessionOwner,
    ModelArtifactDescriptor,
    ModelContractMismatchError,
    ModelIntegrityError,
    ModelNotInstalledError,
    ModelRuntimeError,
    OnnxRuntimeModule,
    OnnxSession,
    OnnxTensorMetadata,
    ProviderUnavailableError,
    RuntimeNotInstalledError,
    RuntimeProvider,
    SessionCreationError,
    SessionOwnerClosedError,
    artifact_path,
    load_onnx_runtime,
    validate_lama_session_contract,
)

_CPU = "CPUExecutionProvider"
_CUDA = "CUDAExecutionProvider"


@dataclass(frozen=True, slots=True)
class FakeTensorMetadata:
    """Small NodeArg-compatible metadata object."""

    name: str
    type: str
    shape: tuple[object, ...]


class BrokenNameMetadata:
    """Metadata whose tensor name cannot be inspected."""

    @property
    def name(self) -> str:
        raise RuntimeError("simulated name failure")

    @property
    def type(self) -> str:
        return "tensor(float)"

    @property
    def shape(self) -> Sequence[object]:
        return ("batch", 3, 512, 512)


class BrokenShapeMetadata:
    """Metadata whose tensor shape cannot be inspected."""

    @property
    def name(self) -> str:
        return "image"

    @property
    def type(self) -> str:
        return "tensor(float)"

    @property
    def shape(self) -> Sequence[object]:
        raise RuntimeError("simulated shape failure")


class FakeSession:
    """Configurable fake inference session with no model dependency."""

    def __init__(
        self,
        *,
        inputs: Sequence[OnnxTensorMetadata] | None = None,
        outputs: Sequence[OnnxTensorMetadata] | None = None,
        providers: Sequence[str] = (_CPU,),
        metadata_error: str | None = None,
        provider_error: BaseException | None = None,
    ) -> None:
        self._inputs = tuple(inputs if inputs is not None else _valid_inputs("batch"))
        self._outputs = tuple(outputs if outputs is not None else _valid_outputs("batch"))
        self._providers = tuple(providers)
        self._metadata_error = metadata_error
        self._provider_error = provider_error

    def get_inputs(self) -> Sequence[OnnxTensorMetadata]:
        if self._metadata_error == "inputs":
            raise RuntimeError("simulated input metadata failure")
        return self._inputs

    def get_outputs(self) -> Sequence[OnnxTensorMetadata]:
        if self._metadata_error == "outputs":
            raise RuntimeError("simulated output metadata failure")
        return self._outputs

    def get_providers(self) -> Sequence[str]:
        if self._provider_error is not None:
            raise self._provider_error
        return self._providers

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, object],
    ) -> Sequence[object]:
        del output_names, input_feed
        raise AssertionError("runtime-boundary tests must not execute inference")


class FakeRuntime:
    """Configurable ONNX Runtime module substitute."""

    def __init__(
        self,
        session: OnnxSession,
        *,
        version: object = "1.26.0",
        available_providers: Sequence[object] = (_CPU,),
        version_error: BaseException | None = None,
        available_error: BaseException | None = None,
        creation_error: BaseException | None = None,
    ) -> None:
        self._session = session
        self._version = version
        self._available_providers = tuple(available_providers)
        self._version_error = version_error
        self._available_error = available_error
        self._creation_error = creation_error
        self.session_calls: list[tuple[str, tuple[runtime_module.ProviderConfiguration, ...]]] = []

    @property
    def __version__(self) -> str:
        if self._version_error is not None:
            raise self._version_error
        return cast(str, self._version)

    def get_available_providers(self) -> Sequence[str]:
        if self._available_error is not None:
            raise self._available_error
        return cast(Sequence[str], self._available_providers)

    def InferenceSession(
        self,
        path_or_bytes: str,
        *,
        providers: Sequence[runtime_module.ProviderConfiguration],
    ) -> OnnxSession:
        self.session_calls.append((path_or_bytes, tuple(providers)))
        if self._creation_error is not None:
            raise self._creation_error
        return self._session


def _valid_inputs(batch_dimension: object) -> tuple[FakeTensorMetadata, ...]:
    return (
        FakeTensorMetadata(
            name="image",
            type="tensor(float)",
            shape=(batch_dimension, 3, 512, 512),
        ),
        FakeTensorMetadata(
            name="mask",
            type="tensor(float)",
            shape=(batch_dimension, 1, 512, 512),
        ),
    )


def _valid_outputs(batch_dimension: object) -> tuple[FakeTensorMetadata, ...]:
    return (
        FakeTensorMetadata(
            name="output",
            type="tensor(float)",
            shape=(batch_dimension, 3, 512, 512),
        ),
    )


def _fixture_descriptor(payload: bytes = b"fixture-model") -> ModelArtifactDescriptor:
    return replace(
        LAMA_ONNX_FP32,
        model_id="runtime-fixture",
        filename="runtime-fixture.onnx",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _install_fixture(
    tmp_path: Path,
    payload: bytes = b"fixture-model",
) -> ModelArtifactDescriptor:
    descriptor = _fixture_descriptor(payload)
    target = artifact_path(descriptor, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    return descriptor


def _owner(
    tmp_path: Path,
    descriptor: ModelArtifactDescriptor,
    runtime: OnnxRuntimeModule,
    *,
    provider: RuntimeProvider = RuntimeProvider.CPU,
    observed_providers: list[RuntimeProvider] | None = None,
) -> LamaOnnxSessionOwner:
    def load(requested_provider: RuntimeProvider) -> OnnxRuntimeModule:
        if observed_providers is not None:
            observed_providers.append(requested_provider)
        return runtime

    return LamaOnnxSessionOwner(
        tmp_path,
        provider=provider,
        descriptor=descriptor,
        runtime_loader=load,
    )


def test_runtime_error_codes_are_stable() -> None:
    assert ModelRuntimeError.code == "model_runtime_error"
    assert ModelNotInstalledError.code == "model_not_installed"
    assert RuntimeNotInstalledError.code == "runtime_not_installed"
    assert ProviderUnavailableError.code == "provider_unavailable"
    assert ModelContractMismatchError.code == "model_contract_mismatch"
    assert SessionCreationError.code == "session_creation_failed"
    assert InsufficientMemoryError.code == "insufficient_memory"
    assert SessionOwnerClosedError.code == "session_owner_closed"


def test_runtime_provider_maps_exact_execution_provider_names() -> None:
    assert RuntimeProvider.CPU.execution_provider == _CPU
    assert RuntimeProvider.CUDA.execution_provider == _CUDA


def test_lama_tensor_contracts_are_fixed() -> None:
    assert [(item.name, item.data_type, item.shape) for item in LAMA_INPUT_CONTRACTS] == [
        ("image", "tensor(float)", (None, 3, 512, 512)),
        ("mask", "tensor(float)", (None, 1, 512, 512)),
    ]
    assert [(item.name, item.data_type, item.shape) for item in LAMA_OUTPUT_CONTRACTS] == [
        ("output", "tensor(float)", (None, 3, 512, 512))
    ]


def test_load_onnx_runtime_imports_only_when_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = FakeRuntime(FakeSession())
    observed_names: list[str] = []

    def fake_import(name: str) -> object:
        observed_names.append(name)
        return expected

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert load_onnx_runtime(RuntimeProvider.CPU) is expected
    assert observed_names == ["onnxruntime"]


@pytest.mark.parametrize(
    ("provider", "expected_extra"),
    [
        (RuntimeProvider.CPU, "lama-onnx-cpu"),
        (RuntimeProvider.CUDA, "lama-onnx-cuda"),
    ],
)
@pytest.mark.parametrize("error", [ImportError("missing"), OSError("broken binary")])
def test_load_onnx_runtime_translates_import_failure(
    provider: RuntimeProvider,
    expected_extra: str,
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str) -> object:
        assert name == "onnxruntime"
        raise error

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeNotInstalledError, match=expected_extra) as captured:
        load_onnx_runtime(provider)

    assert captured.value.provider is provider
    assert captured.value.__cause__ is error


@pytest.mark.parametrize("batch_dimension", [None, "batch_size", 1])
def test_validate_lama_session_contract_accepts_supported_batch_metadata(
    batch_dimension: object,
) -> None:
    validate_lama_session_contract(
        FakeSession(
            inputs=_valid_inputs(batch_dimension),
            outputs=_valid_outputs(batch_dimension),
        )
    )


@pytest.mark.parametrize("metadata_error", ["inputs", "outputs"])
def test_validate_lama_session_contract_translates_metadata_access_failure(
    metadata_error: str,
) -> None:
    with pytest.raises(ModelContractMismatchError, match="could not expose") as captured:
        validate_lama_session_contract(FakeSession(metadata_error=metadata_error))

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_validate_lama_session_contract_rejects_unreadable_name() -> None:
    inputs = (cast(OnnxTensorMetadata, BrokenNameMetadata()), _valid_inputs("batch")[1])

    with pytest.raises(ModelContractMismatchError, match="readable name") as captured:
        validate_lama_session_contract(FakeSession(inputs=inputs))

    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize("invalid_name", ["", cast(str, 7)])
def test_validate_lama_session_contract_rejects_invalid_name(
    invalid_name: str,
) -> None:
    inputs = (
        replace(_valid_inputs("batch")[0], name=invalid_name),
        _valid_inputs("batch")[1],
    )

    with pytest.raises(ModelContractMismatchError, match="invalid tensor name"):
        validate_lama_session_contract(FakeSession(inputs=inputs))


def test_validate_lama_session_contract_rejects_duplicate_name() -> None:
    image = _valid_inputs("batch")[0]

    with pytest.raises(ModelContractMismatchError, match="duplicate tensor"):
        validate_lama_session_contract(FakeSession(inputs=(image, image)))


def test_validate_lama_session_contract_rejects_wrong_names() -> None:
    inputs = (
        replace(_valid_inputs("batch")[0], name="wrong"),
        _valid_inputs("batch")[1],
    )

    with pytest.raises(ModelContractMismatchError, match="names are"):
        validate_lama_session_contract(FakeSession(inputs=inputs))


def test_validate_lama_session_contract_rejects_unreadable_shape() -> None:
    inputs = (cast(OnnxTensorMetadata, BrokenShapeMetadata()), _valid_inputs("batch")[1])

    with pytest.raises(ModelContractMismatchError, match="unreadable type or shape") as captured:
        validate_lama_session_contract(FakeSession(inputs=inputs))

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_validate_lama_session_contract_rejects_wrong_type() -> None:
    inputs = (
        replace(_valid_inputs("batch")[0], type="tensor(double)"),
        _valid_inputs("batch")[1],
    )

    with pytest.raises(ModelContractMismatchError, match="has type"):
        validate_lama_session_contract(FakeSession(inputs=inputs))


def test_validate_lama_session_contract_rejects_wrong_rank() -> None:
    inputs = (
        replace(_valid_inputs("batch")[0], shape=("batch", 3, 512)),
        _valid_inputs("batch")[1],
    )

    with pytest.raises(ModelContractMismatchError, match="has rank"):
        validate_lama_session_contract(FakeSession(inputs=inputs))


@pytest.mark.parametrize("batch_dimension", ["", True, 2])
def test_validate_lama_session_contract_rejects_invalid_batch_dimension(
    batch_dimension: object,
) -> None:
    inputs = (
        replace(
            _valid_inputs("batch")[0],
            shape=(batch_dimension, 3, 512, 512),
        ),
        _valid_inputs("batch")[1],
    )

    with pytest.raises(ModelContractMismatchError, match="axis 0"):
        validate_lama_session_contract(FakeSession(inputs=inputs))


@pytest.mark.parametrize("dimension", [True, "512", 511])
def test_validate_lama_session_contract_rejects_invalid_fixed_dimension(
    dimension: object,
) -> None:
    inputs = (
        replace(
            _valid_inputs("batch")[0],
            shape=("batch", 3, dimension, 512),
        ),
        _valid_inputs("batch")[1],
    )

    with pytest.raises(ModelContractMismatchError, match="axis 2"):
        validate_lama_session_contract(FakeSession(inputs=inputs))


def test_session_owner_is_lazy_and_missing_model_does_not_load_runtime(
    tmp_path: Path,
) -> None:
    loader_calls: list[RuntimeProvider] = []

    def unexpected_load(provider: RuntimeProvider) -> OnnxRuntimeModule:
        loader_calls.append(provider)
        raise AssertionError("missing model must be checked before runtime import")

    owner = LamaOnnxSessionOwner(tmp_path, runtime_loader=unexpected_load)

    assert not owner.is_open
    assert owner.diagnostics is None
    with pytest.raises(ModelNotInstalledError, match="not installed"):
        owner.get_session()
    assert loader_calls == []


def test_session_owner_rejects_invalid_model_before_runtime_load(
    tmp_path: Path,
) -> None:
    descriptor = _fixture_descriptor()
    target = artifact_path(descriptor, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"invalid")
    loader_calls: list[RuntimeProvider] = []

    def unexpected_load(provider: RuntimeProvider) -> OnnxRuntimeModule:
        loader_calls.append(provider)
        raise AssertionError("invalid model must be checked before runtime import")

    owner = LamaOnnxSessionOwner(
        tmp_path,
        descriptor=descriptor,
        runtime_loader=unexpected_load,
    )

    with pytest.raises(ModelIntegrityError):
        owner.get_session()
    assert loader_calls == []


def test_session_owner_creates_and_reuses_one_cpu_session(tmp_path: Path) -> None:
    descriptor = _install_fixture(tmp_path)
    session = FakeSession(providers=(_CPU,))
    runtime = FakeRuntime(session, available_providers=(_CPU,))
    loader_calls: list[RuntimeProvider] = []
    owner = _owner(
        tmp_path,
        descriptor,
        runtime,
        observed_providers=loader_calls,
    )

    assert owner.get_session() is session
    assert owner.get_session() is session
    assert owner.is_open
    assert loader_calls == [RuntimeProvider.CPU]
    assert runtime.session_calls == [(str(artifact_path(descriptor, tmp_path)), (_CPU,))]
    diagnostics = owner.diagnostics
    assert diagnostics is not None
    assert diagnostics.model_id == descriptor.model_id
    assert diagnostics.model_sha256 == descriptor.sha256
    assert diagnostics.runtime_version == "1.26.0"
    assert diagnostics.requested_provider is RuntimeProvider.CPU
    assert diagnostics.available_providers == (_CPU,)
    assert diagnostics.session_providers == (_CPU,)
    assert diagnostics.effective_provider == _CPU

    owner.close()
    owner.close()
    assert not owner.is_open
    assert owner.diagnostics is None
    with pytest.raises(SessionOwnerClosedError, match="closed"):
        owner.get_session()


@pytest.mark.parametrize(
    ("available_providers", "expected_configuration"),
    [
        (
            (_CUDA, _CPU),
            ((_CUDA, {"device_id": 0}), _CPU),
        ),
        (
            (_CUDA,),
            ((_CUDA, {"device_id": 0}),),
        ),
    ],
)
def test_session_owner_configures_cuda_without_silent_fallback(
    tmp_path: Path,
    available_providers: tuple[str, ...],
    expected_configuration: tuple[runtime_module.ProviderConfiguration, ...],
) -> None:
    descriptor = _install_fixture(tmp_path)
    session = FakeSession(providers=available_providers)
    runtime = FakeRuntime(session, available_providers=available_providers)
    owner = _owner(
        tmp_path,
        descriptor,
        runtime,
        provider=RuntimeProvider.CUDA,
    )

    owner.get_session()

    assert runtime.session_calls[0][1] == expected_configuration
    diagnostics = owner.diagnostics
    assert diagnostics is not None
    assert diagnostics.requested_provider is RuntimeProvider.CUDA
    assert diagnostics.effective_provider == _CUDA


@pytest.mark.parametrize(
    ("provider", "available_providers"),
    [
        (RuntimeProvider.CPU, ()),
        (RuntimeProvider.CUDA, (_CPU,)),
    ],
)
def test_session_owner_rejects_unregistered_provider(
    tmp_path: Path,
    provider: RuntimeProvider,
    available_providers: tuple[str, ...],
) -> None:
    descriptor = _install_fixture(tmp_path)
    runtime = FakeRuntime(
        FakeSession(providers=available_providers),
        available_providers=available_providers,
    )
    owner = _owner(
        tmp_path,
        descriptor,
        runtime,
        provider=provider,
    )

    with pytest.raises(ProviderUnavailableError) as captured:
        owner.get_session()

    assert captured.value.provider is provider
    assert captured.value.available_providers == available_providers
    assert runtime.session_calls == []


def test_session_owner_preserves_runtime_loader_domain_error(tmp_path: Path) -> None:
    descriptor = _install_fixture(tmp_path)
    expected = RuntimeNotInstalledError(RuntimeProvider.CPU)

    def fail_load(provider: RuntimeProvider) -> OnnxRuntimeModule:
        assert provider is RuntimeProvider.CPU
        raise expected

    owner = LamaOnnxSessionOwner(
        tmp_path,
        descriptor=descriptor,
        runtime_loader=fail_load,
    )

    with pytest.raises(RuntimeNotInstalledError) as captured:
        owner.get_session()

    assert captured.value is expected


def test_session_owner_translates_unexpected_loader_error(tmp_path: Path) -> None:
    descriptor = _install_fixture(tmp_path)
    expected = RuntimeError("simulated loader failure")

    def fail_load(provider: RuntimeProvider) -> OnnxRuntimeModule:
        assert provider is RuntimeProvider.CPU
        raise expected

    owner = LamaOnnxSessionOwner(
        tmp_path,
        descriptor=descriptor,
        runtime_loader=fail_load,
    )

    with pytest.raises(RuntimeNotInstalledError) as captured:
        owner.get_session()

    assert captured.value.__cause__ is expected


def test_session_owner_translates_unreadable_runtime_version(tmp_path: Path) -> None:
    descriptor = _install_fixture(tmp_path)
    expected = RuntimeError("simulated version failure")
    runtime = FakeRuntime(
        FakeSession(),
        version_error=expected,
    )

    with pytest.raises(SessionCreationError, match="does not expose") as captured:
        _owner(tmp_path, descriptor, runtime).get_session()

    assert captured.value.__cause__ is expected


@pytest.mark.parametrize("version", ["", " ", cast(str, 126)])
def test_session_owner_rejects_invalid_runtime_version(
    tmp_path: Path,
    version: str,
) -> None:
    descriptor = _install_fixture(tmp_path)
    runtime = FakeRuntime(FakeSession(), version=version)

    with pytest.raises(SessionCreationError, match="invalid version"):
        _owner(tmp_path, descriptor, runtime).get_session()


def test_session_owner_translates_provider_inspection_failure(tmp_path: Path) -> None:
    descriptor = _install_fixture(tmp_path)
    expected = RuntimeError("simulated provider inspection failure")
    runtime = FakeRuntime(
        FakeSession(),
        available_error=expected,
    )

    with pytest.raises(SessionCreationError, match="available providers") as captured:
        _owner(tmp_path, descriptor, runtime).get_session()

    assert captured.value.__cause__ is expected


@pytest.mark.parametrize(
    "available_providers",
    [
        ("",),
        cast(tuple[object, ...], (7,)),
    ],
)
def test_session_owner_rejects_invalid_available_provider_metadata(
    tmp_path: Path,
    available_providers: tuple[object, ...],
) -> None:
    descriptor = _install_fixture(tmp_path)
    runtime = FakeRuntime(
        FakeSession(),
        available_providers=available_providers,
    )

    with pytest.raises(SessionCreationError, match="invalid provider metadata"):
        _owner(tmp_path, descriptor, runtime).get_session()


def test_session_owner_translates_session_creation_failure(tmp_path: Path) -> None:
    descriptor = _install_fixture(tmp_path)
    expected = RuntimeError("simulated initialization failure")
    runtime = FakeRuntime(
        FakeSession(),
        creation_error=expected,
    )

    with pytest.raises(SessionCreationError, match="could not create") as captured:
        _owner(tmp_path, descriptor, runtime).get_session()

    assert captured.value.__cause__ is expected


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("host memory exhausted"),
        RuntimeError("CUDNN_STATUS_ALLOC_FAILED"),
        RuntimeError("std::bad_alloc"),
    ],
)
def test_session_owner_classifies_memory_failure(
    tmp_path: Path,
    error: BaseException,
) -> None:
    descriptor = _install_fixture(tmp_path)
    runtime = FakeRuntime(FakeSession(), creation_error=error)

    with pytest.raises(InsufficientMemoryError, match="allocate memory") as captured:
        _owner(tmp_path, descriptor, runtime).get_session()

    assert captured.value.__cause__ is error


def test_session_owner_translates_active_provider_inspection_failure(
    tmp_path: Path,
) -> None:
    descriptor = _install_fixture(tmp_path)
    expected = RuntimeError("simulated active-provider failure")
    session = FakeSession(provider_error=expected)
    runtime = FakeRuntime(session)

    with pytest.raises(SessionCreationError, match="active providers") as captured:
        _owner(tmp_path, descriptor, runtime).get_session()

    assert captured.value.__cause__ is expected


@pytest.mark.parametrize(
    "session_providers",
    [
        ("",),
        cast(tuple[str, ...], (7,)),
    ],
)
def test_session_owner_rejects_invalid_active_provider_metadata(
    tmp_path: Path,
    session_providers: tuple[str, ...],
) -> None:
    descriptor = _install_fixture(tmp_path)
    session = FakeSession(providers=session_providers)
    runtime = FakeRuntime(session)

    with pytest.raises(SessionCreationError, match="invalid providers"):
        _owner(tmp_path, descriptor, runtime).get_session()


@pytest.mark.parametrize("session_providers", [(), (_CPU,)])
def test_cuda_session_owner_rejects_missing_or_inactive_cuda(
    tmp_path: Path,
    session_providers: tuple[str, ...],
) -> None:
    descriptor = _install_fixture(tmp_path)
    session = FakeSession(providers=session_providers)
    runtime = FakeRuntime(
        session,
        available_providers=(_CUDA, _CPU),
    )
    owner = _owner(
        tmp_path,
        descriptor,
        runtime,
        provider=RuntimeProvider.CUDA,
    )

    with pytest.raises(ProviderUnavailableError) as captured:
        owner.get_session()

    assert captured.value.available_providers == session_providers
    assert not owner.is_open
    assert owner.diagnostics is None


def test_session_owner_does_not_cache_contract_mismatch(tmp_path: Path) -> None:
    descriptor = _install_fixture(tmp_path)
    bad_inputs = (
        replace(_valid_inputs("batch")[0], type="tensor(double)"),
        _valid_inputs("batch")[1],
    )
    session = FakeSession(inputs=bad_inputs)
    runtime = FakeRuntime(session)
    owner = _owner(tmp_path, descriptor, runtime)

    with pytest.raises(ModelContractMismatchError):
        owner.get_session()

    assert not owner.is_open
    assert owner.diagnostics is None


def test_session_owner_context_manager_stays_lazy_and_closes(
    tmp_path: Path,
) -> None:
    descriptor = _install_fixture(tmp_path)
    session = FakeSession()
    runtime = FakeRuntime(session)
    owner = _owner(tmp_path, descriptor, runtime)

    with owner as entered:
        assert entered is owner
        assert not owner.is_open
        assert owner.get_session() is session

    assert not owner.is_open
    with pytest.raises(SessionOwnerClosedError):
        owner.get_session()
