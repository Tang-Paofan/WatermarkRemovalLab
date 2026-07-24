"""Shared domain contracts for Watermark Removal Lab."""

from watermark_removal_lab.common.masks import (
    box_to_mask,
    dilate_mask,
    mask_to_uint8,
    threshold_mask,
)
from watermark_removal_lab.common.types import (
    BinaryMask,
    BoolArray,
    Box,
    DataContractError,
    ImageData,
    UInt8Array,
)

__all__ = [
    "BinaryMask",
    "BoolArray",
    "Box",
    "DataContractError",
    "ImageData",
    "UInt8Array",
    "box_to_mask",
    "dilate_mask",
    "mask_to_uint8",
    "threshold_mask",
]
