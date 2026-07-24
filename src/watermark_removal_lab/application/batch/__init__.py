"""Public B1 batch planning and result contracts."""

from watermark_removal_lab.application.batch.contracts import (
    BATCH_SCHEMA_VERSION,
    BatchCancellationReason,
    BatchContractError,
    BatchFailurePolicy,
    BatchInputError,
    BatchItemError,
    BatchItemStatus,
    BatchMedia,
    BatchOperation,
    BatchPlan,
    BatchSpec,
    ImageBatchItemSpec,
    PlannedImageBatchItem,
)
from watermark_removal_lab.application.batch.directory import (
    DirectoryBatchRequest,
    DirectoryOutputFormat,
    build_directory_batch_spec,
    plan_directory_batch,
)
from watermark_removal_lab.application.batch.manifest import (
    ManifestBatchRequest,
    build_manifest_batch_spec,
    plan_manifest_batch,
)
from watermark_removal_lab.application.batch.planning import (
    BatchPreflightError,
    plan_batch,
)
from watermark_removal_lab.application.batch.results import (
    BatchItemResult,
    BatchSummary,
)

__all__ = [
    "BATCH_SCHEMA_VERSION",
    "BatchCancellationReason",
    "BatchContractError",
    "BatchFailurePolicy",
    "BatchInputError",
    "BatchItemError",
    "BatchItemResult",
    "BatchItemStatus",
    "BatchMedia",
    "BatchOperation",
    "BatchPlan",
    "BatchPreflightError",
    "BatchSpec",
    "BatchSummary",
    "DirectoryBatchRequest",
    "DirectoryOutputFormat",
    "ImageBatchItemSpec",
    "ManifestBatchRequest",
    "PlannedImageBatchItem",
    "build_directory_batch_spec",
    "build_manifest_batch_spec",
    "plan_batch",
    "plan_directory_batch",
    "plan_manifest_batch",
]
