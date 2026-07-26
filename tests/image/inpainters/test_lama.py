"""Offline tests for LaMa crop transforms, tensors, inference, and compositing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Never, cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

import watermark_removal_lab.image.inpainters.lama as lama_module
from watermark_removal_lab.common import (
    BinaryMask,
    DataContractError,
    ImageData,
)
from watermark_removal_lab.image import (
    CropTransformError,
    CropWindow,
    LamaInferenceError,
    PixelPadding,
    inpaint_lama,
    plan_lama_crop,
    prepare_lama_input,
)
from watermark_removal_lab.models import (
    InsufficientMemoryError,
    OnnxSession,
    OnnxTensorMetadata,
)

_MODEL_SIDE = 512


class FakeSession:
    """Session stub that records inference without importing ONNX Runtime."""

    def __init__(
        self,
        outputs: object,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._outputs = outputs
        self._error = error
        self.run_calls: list[tuple[tuple[str, ...], Mapping[str, object]]] = []

    def get_inputs(self) -> Sequence[OnnxTensorMetadata]:
        return ()

    def get_outputs(self) -> Sequence[OnnxTensorMetadata]:
        return ()

    def get_providers(self) -> Sequence[str]:
        return ("CPUExecutionProvider",)

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, object],
    ) -> Sequence[object]:
        self.run_calls.append((tuple(output_names), input_feed))
        if self._error is not None:
            raise self._error
        return cast(Sequence[object], self._outputs)


def _image(
    *,
    height: int,
    width: int,
    alpha: bool = False,
) -> ImageData:
    values = np.arange(height * width * 3, dtype=np.uint32)
    rgb = (values % 256).astype(np.uint8).reshape(height, width, 3)
    alpha_data = (
        np.arange(height * width, dtype=np.uint32).astype(np.uint8).reshape(height, width)
        if alpha
        else None
    )
    return ImageData(rgb, alpha_data)


def _mask(
    *,
    height: int,
    width: int,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
) -> BinaryMask:
    data = np.zeros((height, width), dtype=np.bool_)
    data[y_min:y_max, x_min:x_max] = True
    return BinaryMask(data)


def _constant_output(
    red: float = 10.0,
    green: float = 20.0,
    blue: float = 30.0,
) -> NDArray[np.float32]:
    output = np.empty((1, 3, _MODEL_SIDE, _MODEL_SIDE), dtype=np.float32)
    output[:, 0] = red
    output[:, 1] = green
    output[:, 2] = blue
    return output


def test_lama_error_codes_are_stable() -> None:
    assert CropTransformError.code == "crop_transform_failed"
    assert LamaInferenceError.code == "inference_failed"


def test_crop_window_reports_half_open_dimensions() -> None:
    window = CropWindow(x_min=-2, y_min=3, x_max=8, y_max=10)

    assert window.width == 10
    assert window.height == 7


@pytest.mark.parametrize(
    ("crop_padding", "message"),
    [
        (True, "must be an integer"),
        (1.5, "must be an integer"),
        (-1, "must be non-negative"),
    ],
)
def test_plan_lama_crop_rejects_invalid_padding(
    crop_padding: object,
    message: str,
) -> None:
    image = _image(height=2, width=3)
    mask = _mask(height=2, width=3, y_min=0, y_max=1, x_min=0, x_max=1)

    with pytest.raises(DataContractError, match=message):
        plan_lama_crop(
            image,
            mask,
            crop_padding=cast(int, crop_padding),
        )


def test_plan_lama_crop_rejects_mismatched_mask() -> None:
    with pytest.raises(DataContractError, match="mask shape"):
        plan_lama_crop(
            _image(height=2, width=3),
            BinaryMask(np.zeros((3, 2), dtype=np.bool_)),
        )


def test_plan_lama_crop_returns_none_for_empty_mask() -> None:
    image = _image(height=2, width=3)

    assert (
        plan_lama_crop(
            image,
            BinaryMask(np.zeros((2, 3), dtype=np.bool_)),
        )
        is None
    )


def test_plan_lama_crop_centers_non_square_context() -> None:
    image = _image(height=10, width=20)
    mask = _mask(
        height=10,
        width=20,
        y_min=4,
        y_max=6,
        x_min=8,
        x_max=12,
    )

    plan = plan_lama_crop(image, mask, crop_padding=2)

    assert plan is not None
    assert plan.source_box.x_min == 8
    assert plan.source_box.y_min == 4
    assert plan.source_box.x_max == 12
    assert plan.source_box.y_max == 6
    assert plan.square_window == CropWindow(x_min=6, y_min=1, x_max=14, y_max=9)
    assert plan.clipped_box.x_min == 6
    assert plan.clipped_box.y_min == 1
    assert plan.clipped_box.x_max == 14
    assert plan.clipped_box.y_max == 9
    assert plan.padding == PixelPadding(top=0, bottom=0, left=0, right=0)
    assert plan.context_side == 8
    assert plan.scale == 64.0
    assert plan.warnings == ()


def test_plan_lama_crop_allocates_odd_square_remainder_to_right() -> None:
    image = _image(height=5, width=5)
    mask = _mask(height=5, width=5, y_min=1, y_max=4, x_min=2, x_max=4)

    plan = plan_lama_crop(image, mask, crop_padding=0)

    assert plan is not None
    assert plan.square_window == CropWindow(x_min=2, y_min=1, x_max=5, y_max=4)


def test_plan_lama_crop_records_edge_padding() -> None:
    image = _image(height=5, width=7)
    mask = _mask(height=5, width=7, y_min=0, y_max=1, x_min=0, x_max=1)

    plan = plan_lama_crop(image, mask, crop_padding=2)

    assert plan is not None
    assert plan.square_window == CropWindow(x_min=-2, y_min=-2, x_max=3, y_max=3)
    assert plan.clipped_box.x_min == 0
    assert plan.clipped_box.y_min == 0
    assert plan.clipped_box.x_max == 3
    assert plan.clipped_box.y_max == 3
    assert plan.padding == PixelPadding(top=2, bottom=0, left=2, right=0)


def test_plan_lama_crop_warns_for_downscale_and_full_frame() -> None:
    image = _image(height=600, width=400)
    mask = BinaryMask(np.ones((600, 400), dtype=np.bool_))

    plan = plan_lama_crop(image, mask, crop_padding=64)

    assert plan is not None
    assert plan.context_side == 728
    assert plan.scale == pytest.approx(512 / 728)
    assert plan.warnings == ("crop_downscaled", "full_frame_mask")


def test_extract_context_reflects_rgb_and_zero_pads_mask() -> None:
    image = _image(height=3, width=3)
    mask = _mask(height=3, width=3, y_min=0, y_max=1, x_min=0, x_max=1)
    plan = plan_lama_crop(image, mask, crop_padding=2)
    assert plan is not None

    rgb_context, mask_context = lama_module._extract_context(image, mask, plan)

    expected_rgb = np.pad(
        image.rgb,
        ((2, 0), (2, 0), (0, 0)),
        mode="reflect",
    )
    expected_mask = np.pad(
        mask.data,
        ((2, 0), (2, 0)),
        mode="constant",
        constant_values=False,
    )
    assert np.array_equal(rgb_context, expected_rgb)
    assert np.array_equal(mask_context, expected_mask)
    assert not mask_context[:2].any()
    assert not mask_context[:, :2].any()


def test_extract_context_uses_edge_padding_for_one_pixel_height() -> None:
    image = _image(height=1, width=5)
    mask = _mask(height=1, width=5, y_min=0, y_max=1, x_min=2, x_max=3)
    plan = plan_lama_crop(image, mask, crop_padding=2)
    assert plan is not None

    rgb_context, _ = lama_module._extract_context(image, mask, plan)

    assert rgb_context.shape == (5, 5, 3)
    for row in rgb_context:
        assert np.array_equal(row, image.rgb[0])


def test_extract_context_uses_edge_padding_for_one_pixel_width() -> None:
    image = _image(height=5, width=1)
    mask = _mask(height=5, width=1, y_min=2, y_max=3, x_min=0, x_max=1)
    plan = plan_lama_crop(image, mask, crop_padding=2)
    assert plan is not None

    rgb_context, _ = lama_module._extract_context(image, mask, plan)

    assert rgb_context.shape == (5, 5, 3)
    for column in range(rgb_context.shape[1]):
        assert np.array_equal(rgb_context[:, column], image.rgb[:, 0])


def test_extract_context_detects_corrupt_plan() -> None:
    image = _image(height=3, width=3)
    mask = _mask(height=3, width=3, y_min=1, y_max=2, x_min=1, x_max=2)
    plan = plan_lama_crop(image, mask, crop_padding=0)
    assert plan is not None

    with pytest.raises(CropTransformError, match="planned square"):
        lama_module._extract_context(
            image,
            mask,
            replace(plan, context_side=plan.context_side + 1),
        )


def test_prepare_lama_input_builds_normalized_read_only_tensors() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[:, :, 0] = 0
    rgb[:, :, 1] = 128
    rgb[:, :, 2] = 255
    image = ImageData(rgb)
    mask = BinaryMask(np.ones((2, 2), dtype=np.bool_))
    original_rgb = image.rgb.copy()
    original_mask = mask.data.copy()

    prepared = prepare_lama_input(image, mask, crop_padding=0)

    assert prepared is not None
    assert prepared.image_tensor.shape == (1, 3, 512, 512)
    assert prepared.image_tensor.dtype == np.float32
    assert prepared.image_tensor.flags.c_contiguous
    assert not prepared.image_tensor.flags.writeable
    assert np.all(prepared.image_tensor[:, 0] == 0.0)
    assert np.allclose(prepared.image_tensor[:, 1], 128 / 255)
    assert np.all(prepared.image_tensor[:, 2] == 1.0)
    assert prepared.mask_tensor.shape == (1, 1, 512, 512)
    assert prepared.mask_tensor.dtype == np.float32
    assert prepared.mask_tensor.flags.c_contiguous
    assert not prepared.mask_tensor.flags.writeable
    assert np.array_equal(np.unique(prepared.mask_tensor), [1.0])
    assert np.array_equal(image.rgb, original_rgb)
    assert np.array_equal(mask.data, original_mask)


def test_prepare_lama_input_returns_none_for_empty_mask() -> None:
    assert (
        prepare_lama_input(
            _image(height=2, width=2),
            BinaryMask(np.zeros((2, 2), dtype=np.bool_)),
        )
        is None
    )


@pytest.mark.parametrize(
    ("image_side", "expected_rgb_interpolation"),
    [
        (10, cv2.INTER_CUBIC),
        (600, cv2.INTER_AREA),
    ],
)
def test_prepare_lama_input_uses_directional_rgb_and_nearest_mask_resize(
    image_side: int,
    expected_rgb_interpolation: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resize = cv2.resize
    observed_interpolations: list[int] = []

    def observe_resize(
        source: NDArray[np.integer[Any] | np.floating[Any]],
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> NDArray[np.integer[Any] | np.floating[Any]]:
        observed_interpolations.append(interpolation)
        return original_resize(source, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", observe_resize)
    image = _image(height=image_side, width=image_side)
    mask = BinaryMask(np.ones((image_side, image_side), dtype=np.bool_))

    assert prepare_lama_input(image, mask, crop_padding=0) is not None
    assert observed_interpolations == [expected_rgb_interpolation, cv2.INTER_NEAREST]


def test_prepare_lama_input_skips_resize_for_exact_model_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_resize(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise AssertionError("exact 512 by 512 context must not resize")

    monkeypatch.setattr(cv2, "resize", unexpected_resize)
    image = _image(height=512, width=512)
    mask = BinaryMask(np.ones((512, 512), dtype=np.bool_))

    prepared = prepare_lama_input(image, mask, crop_padding=0)

    assert prepared is not None
    assert prepared.image_tensor.shape == (1, 3, 512, 512)


def test_prepare_lama_input_translates_resize_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = cv2.error("simulated resize failure")

    def fail_resize(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise expected

    monkeypatch.setattr(cv2, "resize", fail_resize)

    with pytest.raises(CropTransformError, match="prepare") as captured:
        prepare_lama_input(
            _image(height=2, width=2),
            BinaryMask(np.ones((2, 2), dtype=np.bool_)),
            crop_padding=0,
        )

    assert captured.value.__cause__ is expected


def test_prepare_lama_input_rejects_mask_lost_during_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_resized_mask(
        mask: NDArray[np.bool_],
        target_side: int,
    ) -> NDArray[np.bool_]:
        del mask
        return np.zeros((target_side, target_side), dtype=np.bool_)

    monkeypatch.setattr(lama_module, "_resize_mask", empty_resized_mask)

    with pytest.raises(CropTransformError, match="disappeared"):
        prepare_lama_input(
            _image(height=2, width=2),
            _mask(height=2, width=2, y_min=0, y_max=1, x_min=0, x_max=1),
            crop_padding=0,
        )


def test_inpaint_lama_runs_expected_tensors_and_composites_only_mask() -> None:
    image = _image(height=4, width=6, alpha=True)
    mask = _mask(height=4, width=6, y_min=1, y_max=3, x_min=2, x_max=5)
    original_rgb = image.rgb.copy()
    original_alpha = None if image.alpha is None else image.alpha.copy()
    original_mask = mask.data.copy()
    session = FakeSession([_constant_output(-10.0, 127.5, 300.0)])

    result = inpaint_lama(image, mask, cast(OnnxSession, session), crop_padding=1)

    assert result.plan is not None
    assert result.warnings == result.plan.warnings
    assert len(session.run_calls) == 1
    output_names, input_feed = session.run_calls[0]
    assert output_names == ("output",)
    assert set(input_feed) == {"image", "mask"}
    image_tensor = cast(NDArray[np.float32], input_feed["image"])
    mask_tensor = cast(NDArray[np.float32], input_feed["mask"])
    assert image_tensor.shape == (1, 3, 512, 512)
    assert image_tensor.dtype == np.float32
    assert float(image_tensor.min()) >= 0.0
    assert float(image_tensor.max()) <= 1.0
    assert mask_tensor.shape == (1, 1, 512, 512)
    assert set(np.unique(mask_tensor)) <= {0.0, 1.0}
    assert np.all(result.image.rgb[mask.data] == [0, 128, 255])
    assert np.array_equal(result.image.rgb[~mask.data], image.rgb[~mask.data])
    assert result.image.alpha is not None
    assert original_alpha is not None
    assert np.array_equal(result.image.alpha, original_alpha)
    assert np.array_equal(image.rgb, original_rgb)
    assert np.array_equal(mask.data, original_mask)


def test_inpaint_lama_bypasses_session_for_empty_mask() -> None:
    image = _image(height=2, width=3, alpha=True)
    session = FakeSession([_constant_output()])

    result = inpaint_lama(
        image,
        BinaryMask(np.zeros((2, 3), dtype=np.bool_)),
        cast(OnnxSession, session),
    )

    assert session.run_calls == []
    assert result.plan is None
    assert result.warnings == ()
    assert np.array_equal(result.image.rgb, image.rgb)
    assert result.image.rgb is not image.rgb
    assert result.image.alpha is not None
    assert image.alpha is not None
    assert np.array_equal(result.image.alpha, image.alpha)


def test_inpaint_lama_uses_exact_model_shape_without_inverse_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_resize(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise AssertionError("exact model-sized crop must not resize")

    monkeypatch.setattr(cv2, "resize", unexpected_resize)
    image = _image(height=512, width=512)
    mask = BinaryMask(np.ones((512, 512), dtype=np.bool_))

    result = inpaint_lama(
        image,
        mask,
        cast(OnnxSession, FakeSession([_constant_output()])),
        crop_padding=0,
    )

    assert result.plan is not None
    assert result.plan.context_side == 512


def test_inpaint_lama_upscales_large_context_with_cubic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resize = cv2.resize
    observed_interpolations: list[int] = []

    def observe_resize(
        source: NDArray[np.integer[Any] | np.floating[Any]],
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> NDArray[np.integer[Any] | np.floating[Any]]:
        observed_interpolations.append(interpolation)
        return original_resize(source, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", observe_resize)
    image = _image(height=600, width=600)
    mask = BinaryMask(np.ones((600, 600), dtype=np.bool_))

    result = inpaint_lama(
        image,
        mask,
        cast(OnnxSession, FakeSession([_constant_output()])),
        crop_padding=0,
    )

    assert result.plan is not None
    assert result.plan.warnings == ("crop_downscaled", "full_frame_mask")
    assert observed_interpolations == [
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_CUBIC,
    ]


def test_inpaint_lama_preserves_domain_error_from_session() -> None:
    expected = InsufficientMemoryError("already translated")
    session = FakeSession([], error=expected)

    with pytest.raises(InsufficientMemoryError) as captured:
        inpaint_lama(
            _image(height=2, width=2),
            BinaryMask(np.ones((2, 2), dtype=np.bool_)),
            cast(OnnxSession, session),
            crop_padding=0,
        )

    assert captured.value is expected


def test_inpaint_lama_classifies_memory_failure() -> None:
    expected = RuntimeError("CUDA out of memory")
    session = FakeSession([], error=expected)

    with pytest.raises(InsufficientMemoryError, match="during LaMa inference") as captured:
        inpaint_lama(
            _image(height=2, width=2),
            BinaryMask(np.ones((2, 2), dtype=np.bool_)),
            cast(OnnxSession, session),
            crop_padding=0,
        )

    assert captured.value.__cause__ is expected


def test_inpaint_lama_translates_backend_failure() -> None:
    expected = RuntimeError("simulated inference failure")
    session = FakeSession([], error=expected)

    with pytest.raises(LamaInferenceError, match="failed during") as captured:
        inpaint_lama(
            _image(height=2, width=2),
            BinaryMask(np.ones((2, 2), dtype=np.bool_)),
            cast(OnnxSession, session),
            crop_padding=0,
        )

    assert captured.value.__cause__ is expected


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (cast(Sequence[object], 7), "output sequence"),
        ([], "returned 0 outputs"),
        ([_constant_output(), _constant_output()], "returned 2 outputs"),
        ([object()], "must be a NumPy array"),
        ([_constant_output().astype(np.float64)], "dtype must be float32"),
        ([np.zeros((1, 3, 511, 512), dtype=np.float32)], "output shape"),
    ],
)
def test_inpaint_lama_rejects_invalid_output(
    outputs: Sequence[object],
    message: str,
) -> None:
    with pytest.raises(LamaInferenceError, match=message):
        inpaint_lama(
            _image(height=2, width=2),
            BinaryMask(np.ones((2, 2), dtype=np.bool_)),
            cast(OnnxSession, FakeSession(outputs)),
            crop_padding=0,
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_inpaint_lama_rejects_non_finite_output(non_finite: float) -> None:
    output = _constant_output()
    output[0, 0, 0, 0] = non_finite

    with pytest.raises(LamaInferenceError, match="non-finite"):
        inpaint_lama(
            _image(height=2, width=2),
            BinaryMask(np.ones((2, 2), dtype=np.bool_)),
            cast(OnnxSession, FakeSession([output])),
            crop_padding=0,
        )


def test_inpaint_lama_translates_inverse_resize_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resize = cv2.resize
    call_count = 0
    expected = cv2.error("simulated inverse resize failure")

    def fail_third_resize(
        source: NDArray[np.integer[Any] | np.floating[Any]],
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> NDArray[np.integer[Any] | np.floating[Any]]:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise expected
        return original_resize(source, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", fail_third_resize)

    with pytest.raises(CropTransformError, match="restore") as captured:
        inpaint_lama(
            _image(height=2, width=2),
            BinaryMask(np.ones((2, 2), dtype=np.bool_)),
            cast(OnnxSession, FakeSession([_constant_output()])),
            crop_padding=0,
        )

    assert captured.value.__cause__ is expected


def test_restore_candidate_detects_corrupt_inverse_plan() -> None:
    image = _image(height=3, width=3)
    mask = BinaryMask(np.ones((3, 3), dtype=np.bool_))
    plan = plan_lama_crop(image, mask, crop_padding=0)
    assert plan is not None
    corrupt_plan = replace(
        plan,
        padding=PixelPadding(top=1, bottom=0, left=0, right=0),
    )

    with pytest.raises(CropTransformError, match="Restored crop shape"):
        lama_module._restore_candidate(
            image,
            corrupt_plan,
            _constant_output(),
        )
