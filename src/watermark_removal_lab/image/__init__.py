"""Image-specific IO and processing components."""

from watermark_removal_lab.image.composite import composite_masked
from watermark_removal_lab.image.inpainters import (
    CropTransformError,
    CropWindow,
    Float32Array,
    LamaCropPlan,
    LamaInferenceError,
    LamaInpaintResult,
    LamaPreparedInput,
    OpenCVInpaintError,
    OpenCVInpaintMethod,
    PixelPadding,
    inpaint_lama,
    inpaint_opencv,
    plan_lama_crop,
    prepare_lama_input,
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
    "CropTransformError",
    "CropWindow",
    "Float32Array",
    "ImageReadError",
    "LamaCropPlan",
    "LamaInferenceError",
    "LamaInpaintResult",
    "LamaPreparedInput",
    "MaskReadError",
    "OpenCVInpaintError",
    "OpenCVInpaintMethod",
    "PixelPadding",
    "UnsupportedImageFormatError",
    "composite_masked",
    "inpaint_lama",
    "inpaint_opencv",
    "plan_lama_crop",
    "prepare_lama_input",
    "read_image",
    "read_mask_intensity",
]
