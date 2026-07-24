"""Image-specific IO and processing components."""

from watermark_removal_lab.image.composite import composite_masked
from watermark_removal_lab.image.inpainters import (
    OpenCVInpaintError,
    OpenCVInpaintMethod,
    inpaint_opencv,
)
from watermark_removal_lab.image.io import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageReadError,
    MaskReadError,
    UnsupportedImageFormatError,
    read_image,
    read_mask_intensity,
)

__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "ImageReadError",
    "MaskReadError",
    "OpenCVInpaintError",
    "OpenCVInpaintMethod",
    "UnsupportedImageFormatError",
    "composite_masked",
    "inpaint_opencv",
    "read_image",
    "read_mask_intensity",
]
