"""Framework-neutral application services."""

from watermark_removal_lab.application.image_removal import (
    BoxMaskSource,
    ImageRemovalError,
    ImageRemovalInputError,
    ImageRemovalOutputError,
    ImageRemovalProcessingError,
    ImageRemovalRequest,
    ImageRemovalResult,
    ImageRemovalStatus,
    MaskFileSource,
    OverwritePolicy,
    build_failed_image_removal_result,
    remove_image,
)

__all__ = [
    "BoxMaskSource",
    "ImageRemovalError",
    "ImageRemovalInputError",
    "ImageRemovalOutputError",
    "ImageRemovalProcessingError",
    "ImageRemovalRequest",
    "ImageRemovalResult",
    "ImageRemovalStatus",
    "MaskFileSource",
    "OverwritePolicy",
    "build_failed_image_removal_result",
    "remove_image",
]
