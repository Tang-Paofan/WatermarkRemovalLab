"""Image inpainting backend adapters."""

from watermark_removal_lab.image.inpainters.lama import (
    CropTransformError,
    CropWindow,
    Float32Array,
    LamaCropPlan,
    LamaInferenceError,
    LamaInpaintResult,
    LamaPreparedInput,
    PixelPadding,
    inpaint_lama,
    plan_lama_crop,
    prepare_lama_input,
)
from watermark_removal_lab.image.inpainters.opencv import (
    OpenCVInpaintError,
    OpenCVInpaintMethod,
    inpaint_opencv,
)

__all__ = [
    "CropTransformError",
    "CropWindow",
    "Float32Array",
    "LamaCropPlan",
    "LamaInferenceError",
    "LamaInpaintResult",
    "LamaPreparedInput",
    "OpenCVInpaintError",
    "OpenCVInpaintMethod",
    "PixelPadding",
    "inpaint_lama",
    "inpaint_opencv",
    "plan_lama_crop",
    "prepare_lama_input",
]
