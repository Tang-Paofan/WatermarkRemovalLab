"""Mask-constrained image compositing."""

import numpy as np

from watermark_removal_lab.common import BinaryMask, DataContractError, ImageData


def composite_masked(
    original: ImageData,
    candidate: ImageData,
    mask: BinaryMask,
) -> ImageData:
    """Copy candidate RGB pixels into the original only where ``mask`` is selected.

    The original alpha channel is preserved byte-for-byte. Candidate pixels
    outside the mask, including any changes introduced by an inpainting
    backend, are deliberately ignored.
    """
    mask.validate_for(original)
    if candidate.spatial_shape != original.spatial_shape:
        raise DataContractError(
            f"candidate shape {candidate.spatial_shape} "
            f"must match original shape {original.spatial_shape}"
        )

    composed = np.array(original.rgb, copy=True)
    composed[mask.data] = candidate.rgb[mask.data]
    return ImageData(rgb=composed, alpha=original.alpha)
