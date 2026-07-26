"""Local-crop LaMa ONNX inpainting with exact mask-only compositing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from watermark_removal_lab.common import (
    BinaryMask,
    Box,
    DataContractError,
    ImageData,
    UInt8Array,
)
from watermark_removal_lab.image.composite import composite_masked
from watermark_removal_lab.models import (
    InsufficientMemoryError,
    ModelRuntimeError,
    OnnxSession,
    is_out_of_memory_error,
)

Float32Array: TypeAlias = NDArray[np.float32]

_MODEL_SIDE = 512
_DEFAULT_CROP_PADDING = 64
_IMAGE_INPUT_NAME = "image"
_MASK_INPUT_NAME = "mask"
_OUTPUT_NAME = "output"


class CropTransformError(ModelRuntimeError):
    """Raised when the crop or inverse mapping cannot preserve its contract."""

    code = "crop_transform_failed"


class LamaInferenceError(ModelRuntimeError):
    """Raised when inference or its returned tensor violates the runtime contract."""

    code = "inference_failed"


@dataclass(frozen=True, slots=True)
class CropWindow:
    """Half-open crop coordinates that may extend outside the image."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        """Return the window width."""

        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Return the window height."""

        return self.y_max - self.y_min


@dataclass(frozen=True, slots=True)
class PixelPadding:
    """Synthetic pixels added around an image-clipped crop."""

    top: int
    bottom: int
    left: int
    right: int


@dataclass(frozen=True, slots=True)
class LamaCropPlan:
    """Recorded forward and inverse geometry for one local model crop."""

    source_box: Box
    square_window: CropWindow
    clipped_box: Box
    padding: PixelPadding
    context_side: int
    scale: float
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class LamaPreparedInput:
    """One immutable batch-one LaMa input and its inverse transform."""

    plan: LamaCropPlan
    image_tensor: Float32Array
    mask_tensor: Float32Array


@dataclass(frozen=True, slots=True, eq=False)
class LamaInpaintResult:
    """Safely composited image plus reproducible crop evidence."""

    image: ImageData
    plan: LamaCropPlan | None
    warnings: tuple[str, ...]


def _validate_crop_padding(crop_padding: object) -> int:
    if isinstance(crop_padding, bool) or not isinstance(crop_padding, int):
        raise DataContractError("crop padding must be an integer")
    if crop_padding < 0:
        raise DataContractError("crop padding must be non-negative")
    return crop_padding


def _mask_box(mask: BinaryMask) -> Box:
    selected_rows, selected_columns = np.nonzero(mask.data)
    return Box(
        x_min=int(selected_columns.min()),
        y_min=int(selected_rows.min()),
        x_max=int(selected_columns.max()) + 1,
        y_max=int(selected_rows.max()) + 1,
    )


def _square_window(source_box: Box, crop_padding: int) -> CropWindow:
    expanded_x_min = source_box.x_min - crop_padding
    expanded_y_min = source_box.y_min - crop_padding
    expanded_x_max = source_box.x_max + crop_padding
    expanded_y_max = source_box.y_max + crop_padding
    expanded_width = expanded_x_max - expanded_x_min
    expanded_height = expanded_y_max - expanded_y_min
    side = max(expanded_width, expanded_height)

    horizontal_extra = side - expanded_width
    vertical_extra = side - expanded_height
    left_extra = horizontal_extra // 2
    top_extra = vertical_extra // 2
    return CropWindow(
        x_min=expanded_x_min - left_extra,
        y_min=expanded_y_min - top_extra,
        x_max=expanded_x_max + horizontal_extra - left_extra,
        y_max=expanded_y_max + vertical_extra - top_extra,
    )


def plan_lama_crop(
    image: ImageData,
    mask: BinaryMask,
    *,
    crop_padding: int = _DEFAULT_CROP_PADDING,
) -> LamaCropPlan | None:
    """Plan a deterministic padded square, or return ``None`` for an empty mask."""

    normalized_padding = _validate_crop_padding(crop_padding)
    mask.validate_for(image)
    if mask.is_empty:
        return None

    source_box = _mask_box(mask)
    square_window = _square_window(source_box, normalized_padding)
    clipped_box = Box(
        x_min=max(0, square_window.x_min),
        y_min=max(0, square_window.y_min),
        x_max=min(image.width, square_window.x_max),
        y_max=min(image.height, square_window.y_max),
    )
    padding = PixelPadding(
        top=max(0, -square_window.y_min),
        bottom=max(0, square_window.y_max - image.height),
        left=max(0, -square_window.x_min),
        right=max(0, square_window.x_max - image.width),
    )
    context_side = square_window.width
    warnings: list[str] = []
    if context_side > _MODEL_SIDE:
        warnings.append("crop_downscaled")
    if mask.is_full:
        warnings.append("full_frame_mask")

    return LamaCropPlan(
        source_box=source_box,
        square_window=square_window,
        clipped_box=clipped_box,
        padding=padding,
        context_side=context_side,
        scale=_MODEL_SIDE / context_side,
        warnings=tuple(warnings),
    )


def _pad_rgb(crop: UInt8Array, padding: PixelPadding) -> UInt8Array:
    padded = crop
    if padding.top or padding.bottom:
        vertical_width = ((padding.top, padding.bottom), (0, 0), (0, 0))
        if padded.shape[0] > 1:
            padded = np.pad(padded, vertical_width, mode="reflect")
        else:
            padded = np.pad(padded, vertical_width, mode="edge")
    if padding.left or padding.right:
        horizontal_width = ((0, 0), (padding.left, padding.right), (0, 0))
        if padded.shape[1] > 1:
            padded = np.pad(padded, horizontal_width, mode="reflect")
        else:
            padded = np.pad(padded, horizontal_width, mode="edge")
    return padded


def _extract_context(
    image: ImageData,
    mask: BinaryMask,
    plan: LamaCropPlan,
) -> tuple[UInt8Array, NDArray[np.bool_]]:
    box = plan.clipped_box
    rgb_crop = np.array(
        image.rgb[box.y_min : box.y_max, box.x_min : box.x_max],
        copy=True,
    )
    mask_crop = np.array(
        mask.data[box.y_min : box.y_max, box.x_min : box.x_max],
        copy=True,
    )
    rgb_context = _pad_rgb(rgb_crop, plan.padding)
    mask_context = np.pad(
        mask_crop,
        (
            (plan.padding.top, plan.padding.bottom),
            (plan.padding.left, plan.padding.right),
        ),
        mode="constant",
        constant_values=False,
    )
    expected_shape = (plan.context_side, plan.context_side)
    if rgb_context.shape[:2] != expected_shape or mask_context.shape != expected_shape:
        raise CropTransformError("The padded crop does not match the planned square context.")
    return rgb_context, mask_context


def _resize_rgb(rgb: UInt8Array, target_side: int) -> UInt8Array:
    source_side = int(rgb.shape[0])
    if source_side == target_side:
        return np.array(rgb, copy=True)
    interpolation = cv2.INTER_AREA if target_side < source_side else cv2.INTER_CUBIC
    resized = cv2.resize(
        rgb,
        (target_side, target_side),
        interpolation=interpolation,
    )
    return cast(UInt8Array, resized)


def _resize_mask(mask: NDArray[np.bool_], target_side: int) -> NDArray[np.bool_]:
    if mask.shape == (target_side, target_side):
        return np.array(mask, copy=True)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_side, target_side),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def _readonly_float32(array: Float32Array) -> Float32Array:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    contiguous.flags.writeable = False
    return contiguous


def prepare_lama_input(
    image: ImageData,
    mask: BinaryMask,
    *,
    crop_padding: int = _DEFAULT_CROP_PADDING,
) -> LamaPreparedInput | None:
    """Create batch-one NCHW float32 tensors without importing a model runtime."""

    plan = plan_lama_crop(image, mask, crop_padding=crop_padding)
    if plan is None:
        return None

    try:
        rgb_context, mask_context = _extract_context(image, mask, plan)
        model_rgb = _resize_rgb(rgb_context, _MODEL_SIDE)
        model_mask = _resize_mask(mask_context, _MODEL_SIDE)
    except (cv2.error, ValueError) as exc:
        raise CropTransformError("Could not prepare the LaMa model crop.") from exc

    if not bool(model_mask.any()):
        raise CropTransformError("The selected mask disappeared while resizing to model space.")

    image_tensor = np.transpose(model_rgb.astype(np.float32), (2, 0, 1))[None, ...]
    image_tensor /= np.float32(255.0)
    mask_tensor = model_mask.astype(np.float32)[None, None, ...]
    return LamaPreparedInput(
        plan=plan,
        image_tensor=_readonly_float32(image_tensor),
        mask_tensor=_readonly_float32(mask_tensor),
    )


def _normalize_model_output(outputs: Sequence[object]) -> Float32Array:
    try:
        normalized_outputs = tuple(outputs)
    except TypeError as exc:
        raise LamaInferenceError("LaMa inference did not return an output sequence.") from exc
    if len(normalized_outputs) != 1:
        raise LamaInferenceError(
            f"LaMa inference returned {len(normalized_outputs)} outputs; expected one."
        )

    output = normalized_outputs[0]
    if not isinstance(output, np.ndarray):
        raise LamaInferenceError("LaMa output must be a NumPy array.")
    if output.dtype != np.float32:
        raise LamaInferenceError(f"LaMa output dtype must be float32, got {output.dtype}.")
    if output.shape != (1, 3, _MODEL_SIDE, _MODEL_SIDE):
        raise LamaInferenceError(f"LaMa output shape must be (1, 3, 512, 512), got {output.shape}.")
    if not bool(np.isfinite(output).all()):
        raise LamaInferenceError("LaMa output contains non-finite values.")
    return output


def _restore_candidate(
    image: ImageData,
    plan: LamaCropPlan,
    model_output: Float32Array,
) -> ImageData:
    model_rgb = np.transpose(model_output[0], (1, 2, 0))
    if plan.context_side == _MODEL_SIDE:
        square = np.array(model_rgb, copy=True)
    else:
        interpolation = cv2.INTER_AREA if plan.context_side < _MODEL_SIDE else cv2.INTER_CUBIC
        try:
            square = cv2.resize(
                model_rgb,
                (plan.context_side, plan.context_side),
                interpolation=interpolation,
            )
        except cv2.error as exc:
            raise CropTransformError(
                "Could not restore the LaMa model output to crop space."
            ) from exc

    y_start = plan.padding.top
    y_stop = plan.context_side - plan.padding.bottom
    x_start = plan.padding.left
    x_stop = plan.context_side - plan.padding.right
    restored_crop = square[y_start:y_stop, x_start:x_stop]
    expected_shape = (plan.clipped_box.height, plan.clipped_box.width, 3)
    if restored_crop.shape != expected_shape:
        raise CropTransformError(
            f"Restored crop shape {restored_crop.shape} does not match {expected_shape}."
        )

    clipped = np.clip(restored_crop, 0.0, 255.0)
    rounded = np.floor(clipped + np.float32(0.5)).astype(np.uint8)
    candidate_rgb = np.array(image.rgb, copy=True)
    box = plan.clipped_box
    candidate_rgb[box.y_min : box.y_max, box.x_min : box.x_max] = rounded
    return ImageData(rgb=candidate_rgb, alpha=image.alpha)


def inpaint_lama(
    image: ImageData,
    mask: BinaryMask,
    session: OnnxSession,
    *,
    crop_padding: int = _DEFAULT_CROP_PADDING,
) -> LamaInpaintResult:
    """Run a prepared local crop and composite only original-mask pixels."""

    prepared = prepare_lama_input(image, mask, crop_padding=crop_padding)
    if prepared is None:
        return LamaInpaintResult(
            image=ImageData(rgb=image.rgb, alpha=image.alpha),
            plan=None,
            warnings=(),
        )

    input_feed: Mapping[str, object] = {
        _IMAGE_INPUT_NAME: prepared.image_tensor,
        _MASK_INPUT_NAME: prepared.mask_tensor,
    }
    try:
        outputs = session.run((_OUTPUT_NAME,), input_feed)
    except ModelRuntimeError:
        raise
    except Exception as exc:
        if is_out_of_memory_error(exc):
            raise InsufficientMemoryError(
                "ONNX Runtime could not allocate memory during LaMa inference."
            ) from exc
        raise LamaInferenceError("ONNX Runtime failed during LaMa inference.") from exc

    model_output = _normalize_model_output(outputs)
    candidate = _restore_candidate(image, prepared.plan, model_output)
    output = composite_masked(image, candidate, mask)
    return LamaInpaintResult(
        image=output,
        plan=prepared.plan,
        warnings=prepared.plan.warnings,
    )
