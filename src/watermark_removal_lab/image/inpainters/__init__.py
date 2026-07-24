"""Image inpainting backend adapters."""

from watermark_removal_lab.image.inpainters.opencv import (
    OpenCVInpaintError,
    OpenCVInpaintMethod,
    inpaint_opencv,
)

__all__ = [
    "OpenCVInpaintError",
    "OpenCVInpaintMethod",
    "inpaint_opencv",
]
