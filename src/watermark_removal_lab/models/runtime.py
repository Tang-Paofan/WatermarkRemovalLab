"""Lazy ONNX Runtime session ownership for the reviewed LaMa artifact."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, TypeAlias, cast

from watermark_removal_lab.models.artifacts import (
    LAMA_ONNX_FP32,
    ModelArtifactDescriptor,
)
from watermark_removal_lab.models.store import (
    ModelArtifactStatus,
    ModelIntegrityError,
    inspect_model_artifact,
)

_CPU_EXECUTION_PROVIDER: Final = "CPUExecutionProvider"
_CUDA_EXECUTION_PROVIDER: Final = "CUDAExecutionProvider"
_TENSOR_FLOAT_TYPE: Final = "tensor(float)"
_OUT_OF_MEMORY_MARKERS: Final = (
    "out of memory",
    "cudaerrormemoryallocation",
    "cudnn_status_alloc_failed",
    "std::bad_alloc",
)

ProviderOptionValue: TypeAlias = str | int
ProviderConfiguration: TypeAlias = str | tuple[str, dict[str, ProviderOptionValue]]


class RuntimeProvider(StrEnum):
    """Supported ONNX Runtime execution-provider policies."""

    CPU = "cpu"
    CUDA = "cuda"

    @property
    def execution_provider(self) -> str:
        """Return the exact ONNX Runtime provider name."""

        if self is RuntimeProvider.CPU:
            return _CPU_EXECUTION_PROVIDER
        return _CUDA_EXECUTION_PROVIDER


class OnnxTensorMetadata(Protocol):
    """Subset of ONNX Runtime NodeArg metadata used for contract validation."""

    @property
    def name(self) -> str:
        """Return the graph tensor name."""

    @property
    def type(self) -> str:
        """Return the ONNX Runtime tensor type string."""

    @property
    def shape(self) -> Sequence[object]:
        """Return graph dimensions, including symbolic batch dimensions."""


class OnnxSession(Protocol):
    """Subset of an ONNX Runtime inference session required by this slice."""

    def get_inputs(self) -> Sequence[OnnxTensorMetadata]:
        """Return model input metadata."""

    def get_outputs(self) -> Sequence[OnnxTensorMetadata]:
        """Return model output metadata."""

    def get_providers(self) -> Sequence[str]:
        """Return the providers active for this session in priority order."""

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, object],
    ) -> Sequence[object]:
        """Run inference for named outputs and input tensors."""


class OnnxRuntimeModule(Protocol):
    """Subset of the lazily imported ONNX Runtime module."""

    @property
    def __version__(self) -> str:
        """Return the runtime version."""

    def get_available_providers(self) -> Sequence[str]:
        """Return providers registered by the installed runtime package."""

    def InferenceSession(
        self,
        path_or_bytes: str,
        *,
        providers: Sequence[ProviderConfiguration],
    ) -> OnnxSession:
        """Create an inference session for one verified model path."""


RuntimeLoader: TypeAlias = Callable[[RuntimeProvider], OnnxRuntimeModule]


class ModelRuntimeError(RuntimeError):
    """Base class for stable model-runtime failures."""

    code = "model_runtime_error"


class ModelNotInstalledError(ModelRuntimeError):
    """Raised when the reviewed artifact is absent from the selected cache."""

    code = "model_not_installed"


class RuntimeNotInstalledError(ModelRuntimeError):
    """Raised when the selected optional ONNX Runtime package cannot load."""

    code = "runtime_not_installed"

    def __init__(self, provider: RuntimeProvider) -> None:
        self.provider = provider
        extra = "lama-onnx-cpu" if provider is RuntimeProvider.CPU else "lama-onnx-cuda"
        super().__init__(
            f"ONNX Runtime for provider {provider.value!r} is unavailable. "
            f"Install it with 'uv sync --extra {extra}'."
        )


class ProviderUnavailableError(ModelRuntimeError):
    """Raised when the requested execution provider is unavailable or inactive."""

    code = "provider_unavailable"

    def __init__(
        self,
        provider: RuntimeProvider,
        available_providers: Sequence[str],
    ) -> None:
        self.provider = provider
        self.available_providers = tuple(available_providers)
        super().__init__(
            f"Requested provider {provider.execution_provider!r} is unavailable; "
            f"reported providers: {self.available_providers!r}."
        )


class ModelContractMismatchError(ModelRuntimeError):
    """Raised when graph metadata differs from the reviewed LaMa tensor contract."""

    code = "model_contract_mismatch"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"The LaMa ONNX tensor contract does not match: {detail}")


class SessionCreationError(ModelRuntimeError):
    """Raised when ONNX Runtime cannot inspect or create a valid session."""

    code = "session_creation_failed"


class InsufficientMemoryError(ModelRuntimeError):
    """Raised when a runtime operation reports host or device memory exhaustion."""

    code = "insufficient_memory"


class SessionOwnerClosedError(ModelRuntimeError):
    """Raised when a closed session owner is reused."""

    code = "session_owner_closed"


@dataclass(frozen=True, slots=True)
class TensorContract:
    """Expected ONNX metadata for one graph input or output."""

    name: str
    data_type: str
    shape: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    """Normalized runtime and provider evidence for one validated session."""

    model_id: str
    model_sha256: str
    runtime_version: str
    requested_provider: RuntimeProvider
    available_providers: tuple[str, ...]
    session_providers: tuple[str, ...]

    @property
    def effective_provider(self) -> str:
        """Return the highest-priority provider active in the session."""

        return self.session_providers[0]


LAMA_INPUT_CONTRACTS: Final = (
    TensorContract("image", _TENSOR_FLOAT_TYPE, (None, 3, 512, 512)),
    TensorContract("mask", _TENSOR_FLOAT_TYPE, (None, 1, 512, 512)),
)
LAMA_OUTPUT_CONTRACTS: Final = (TensorContract("output", _TENSOR_FLOAT_TYPE, (None, 3, 512, 512)),)


def load_onnx_runtime(provider: RuntimeProvider) -> OnnxRuntimeModule:
    """Import the selected optional runtime with an actionable failure."""

    try:
        module = importlib.import_module("onnxruntime")
    except (ImportError, OSError) as exc:
        raise RuntimeNotInstalledError(provider) from exc
    return cast(OnnxRuntimeModule, module)


def _provider_configuration(
    provider: RuntimeProvider,
    available_providers: Sequence[str],
) -> tuple[ProviderConfiguration, ...]:
    required_provider = provider.execution_provider
    if required_provider not in available_providers:
        raise ProviderUnavailableError(provider, available_providers)

    if provider is RuntimeProvider.CPU:
        return (_CPU_EXECUTION_PROVIDER,)

    configuration: list[ProviderConfiguration] = [(_CUDA_EXECUTION_PROVIDER, {"device_id": 0})]
    if _CPU_EXECUTION_PROVIDER in available_providers:
        configuration.append(_CPU_EXECUTION_PROVIDER)
    return tuple(configuration)


def _dimension_matches(expected: int | None, actual: object) -> bool:
    if expected is None:
        return (
            actual is None
            or (isinstance(actual, str) and bool(actual))
            or (isinstance(actual, int) and not isinstance(actual, bool) and actual == 1)
        )
    return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected


def _validate_tensor_group(
    actual_metadata: Sequence[OnnxTensorMetadata],
    expected_contracts: Sequence[TensorContract],
    *,
    group_name: str,
) -> None:
    expected_by_name = {contract.name: contract for contract in expected_contracts}
    actual_by_name: dict[str, OnnxTensorMetadata] = {}
    for metadata in actual_metadata:
        try:
            name = metadata.name
        except Exception as exc:
            raise ModelContractMismatchError(
                f"{group_name} metadata does not expose a readable name."
            ) from exc
        if not isinstance(name, str) or not name:
            raise ModelContractMismatchError(
                f"{group_name} metadata contains an invalid tensor name."
            )
        if name in actual_by_name:
            raise ModelContractMismatchError(
                f"{group_name} metadata contains duplicate tensor {name!r}."
            )
        actual_by_name[name] = metadata

    if actual_by_name.keys() != expected_by_name.keys():
        raise ModelContractMismatchError(
            f"{group_name} names are {sorted(actual_by_name)!r}; "
            f"expected {sorted(expected_by_name)!r}."
        )

    for name, contract in expected_by_name.items():
        metadata = actual_by_name[name]
        try:
            data_type = metadata.type
            shape = tuple(metadata.shape)
        except Exception as exc:
            raise ModelContractMismatchError(
                f"{group_name} tensor {name!r} has unreadable type or shape metadata."
            ) from exc
        if data_type != contract.data_type:
            raise ModelContractMismatchError(
                f"{group_name} tensor {name!r} has type {data_type!r}; "
                f"expected {contract.data_type!r}."
            )
        if len(shape) != len(contract.shape):
            raise ModelContractMismatchError(
                f"{group_name} tensor {name!r} has rank {len(shape)}; "
                f"expected {len(contract.shape)}."
            )
        for axis, (expected, actual) in enumerate(zip(contract.shape, shape, strict=True)):
            if not _dimension_matches(expected, actual):
                raise ModelContractMismatchError(
                    f"{group_name} tensor {name!r} axis {axis} is {actual!r}; "
                    f"expected {expected!r}."
                )


def validate_lama_session_contract(session: OnnxSession) -> None:
    """Validate exact names, types, ranks, and fixed spatial dimensions."""

    try:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
    except Exception as exc:
        raise ModelContractMismatchError(
            "the session could not expose graph input/output metadata."
        ) from exc
    _validate_tensor_group(inputs, LAMA_INPUT_CONTRACTS, group_name="input")
    _validate_tensor_group(outputs, LAMA_OUTPUT_CONTRACTS, group_name="output")


def is_out_of_memory_error(error: BaseException) -> bool:
    """Return whether a backend exception reports host or device memory exhaustion."""

    if isinstance(error, MemoryError):
        return True
    normalized_message = str(error).lower().replace("_", "")
    return any(marker.replace("_", "") in normalized_message for marker in _OUT_OF_MEMORY_MARKERS)


def _runtime_version(runtime: OnnxRuntimeModule) -> str:
    try:
        version = runtime.__version__
    except Exception as exc:
        raise SessionCreationError("ONNX Runtime does not expose a version.") from exc
    if not isinstance(version, str) or not version.strip():
        raise SessionCreationError("ONNX Runtime reported an invalid version.")
    return version


def _available_providers(runtime: OnnxRuntimeModule) -> tuple[str, ...]:
    try:
        providers = tuple(runtime.get_available_providers())
    except Exception as exc:
        raise SessionCreationError("ONNX Runtime could not report available providers.") from exc
    if any(not isinstance(provider, str) or not provider for provider in providers):
        raise SessionCreationError("ONNX Runtime reported invalid provider metadata.")
    return providers


def _create_session(
    runtime: OnnxRuntimeModule,
    model_path: Path,
    provider_configuration: Sequence[ProviderConfiguration],
) -> OnnxSession:
    try:
        return runtime.InferenceSession(
            str(model_path),
            providers=provider_configuration,
        )
    except Exception as exc:
        if is_out_of_memory_error(exc):
            raise InsufficientMemoryError(
                "ONNX Runtime could not allocate memory while creating the model session."
            ) from exc
        raise SessionCreationError(
            "ONNX Runtime could not create the reviewed LaMa session."
        ) from exc


def _session_providers(session: OnnxSession) -> tuple[str, ...]:
    try:
        providers = tuple(session.get_providers())
    except Exception as exc:
        raise SessionCreationError(
            "The ONNX Runtime session could not report active providers."
        ) from exc
    if any(not isinstance(provider, str) or not provider for provider in providers):
        raise SessionCreationError("The ONNX Runtime session reported invalid providers.")
    return providers


class LamaOnnxSessionOwner:
    """Own at most one lazily created, validated LaMa ONNX Runtime session."""

    def __init__(
        self,
        cache_root: Path | None = None,
        *,
        provider: RuntimeProvider = RuntimeProvider.CPU,
        descriptor: ModelArtifactDescriptor = LAMA_ONNX_FP32,
        runtime_loader: RuntimeLoader = load_onnx_runtime,
    ) -> None:
        self._cache_root = cache_root
        self._provider = provider
        self._descriptor = descriptor
        self._runtime_loader = runtime_loader
        self._session: OnnxSession | None = None
        self._diagnostics: RuntimeDiagnostics | None = None
        self._closed = False

    @property
    def diagnostics(self) -> RuntimeDiagnostics | None:
        """Return normalized evidence after successful lazy initialization."""

        return self._diagnostics

    @property
    def is_open(self) -> bool:
        """Return whether this owner currently retains a validated session."""

        return self._session is not None

    def get_session(self) -> OnnxSession:
        """Return the cached session or create it after all preflight checks."""

        if self._closed:
            raise SessionOwnerClosedError("The LaMa ONNX session owner is closed.")
        if self._session is not None:
            return self._session

        artifact_state = inspect_model_artifact(self._descriptor, self._cache_root)
        if artifact_state.status is ModelArtifactStatus.MISSING:
            raise ModelNotInstalledError(
                f"Model artifact {self._descriptor.model_id!r} is not installed at "
                f"{artifact_state.path}."
            )
        if artifact_state.status is ModelArtifactStatus.INVALID:
            raise ModelIntegrityError(
                self._descriptor,
                artifact_state.path,
                actual_size_bytes=artifact_state.size_bytes,
                actual_sha256=artifact_state.sha256,
            )

        try:
            runtime = self._runtime_loader(self._provider)
        except ModelRuntimeError:
            raise
        except Exception as exc:
            raise RuntimeNotInstalledError(self._provider) from exc

        runtime_version = _runtime_version(runtime)
        available_providers = _available_providers(runtime)
        provider_configuration = _provider_configuration(
            self._provider,
            available_providers,
        )
        session = _create_session(
            runtime,
            artifact_state.path,
            provider_configuration,
        )
        session_providers = _session_providers(session)
        if not session_providers or session_providers[0] != self._provider.execution_provider:
            raise ProviderUnavailableError(self._provider, session_providers)
        validate_lama_session_contract(session)

        self._session = session
        self._diagnostics = RuntimeDiagnostics(
            model_id=self._descriptor.model_id,
            model_sha256=self._descriptor.sha256,
            runtime_version=runtime_version,
            requested_provider=self._provider,
            available_providers=available_providers,
            session_providers=session_providers,
        )
        return session

    def close(self) -> None:
        """Release this owner's session reference and permanently close the owner."""

        self._session = None
        self._diagnostics = None
        self._closed = True

    def __enter__(self) -> LamaOnnxSessionOwner:
        """Enter without eagerly loading the optional runtime."""

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the owned session reference."""

        del exception_type, exception, traceback
        self.close()
