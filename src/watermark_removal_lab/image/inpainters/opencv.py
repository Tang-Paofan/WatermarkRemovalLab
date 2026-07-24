"""OpenCV Telea and Navier-Stokes image inpainting adapter."""

import math
from enum import StrEnum
from typing import cast

import cv2

from watermark_removal_lab.common import (
    BinaryMask,
    DataContractError,
    ImageData,
    UInt8Array,
    mask_to_uint8,
)


class OpenCVInpaintMethod(StrEnum):
    """Supported OpenCV inpainting algorithms."""

    TELEA = "telea"
    NAVIER_STOKES = "ns"


class OpenCVInpaintError(RuntimeError):
    """Raised when OpenCV cannot produce an inpainting candidate."""


def _validate_radius(radius: object) -> float:
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise DataContractError("inpaint radius must be a real number")

    normalized = float(radius)
    if not math.isfinite(normalized):
        raise DataContractError("inpaint radius must be finite")
    if normalized <= 0:
        raise DataContractError("inpaint radius must be positive")
    return normalized


def inpaint_opencv(
    image: ImageData,
    mask: BinaryMask,
    *,
    method: OpenCVInpaintMethod = OpenCVInpaintMethod.TELEA,
    radius: float = 3.0,
) -> ImageData:
    """Generate a full-frame OpenCV inpainting candidate.

    The canonical RGB input is converted explicitly to OpenCV BGR and back.
    This function returns a backend candidate; callers must use
    :func:`watermark_removal_lab.image.composite_masked` to apply only pixels
    selected by the mask.
    """
    if not isinstance(method, OpenCVInpaintMethod):
        raise DataContractError("inpaint method must be an OpenCVInpaintMethod")

    normalized_radius = _validate_radius(radius)
    mask.validate_for(image)
    if mask.is_empty:
        return ImageData(rgb=image.rgb, alpha=image.alpha)

    flag = cv2.INPAINT_TELEA if method is OpenCVInpaintMethod.TELEA else cv2.INPAINT_NS

    try:
        bgr = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(
            bgr,
            mask_to_uint8(mask),
            normalized_radius,
            flag,
        )
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error as error:
        raise OpenCVInpaintError("OpenCV inpainting failed") from error

    return ImageData(rgb=cast(UInt8Array, inpainted_rgb), alpha=image.alpha)
