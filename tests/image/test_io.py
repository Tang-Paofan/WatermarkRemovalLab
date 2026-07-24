"""Integration tests for Pillow-backed image and mask decoding."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from watermark_removal_lab.image import (
    ImageReadError,
    MaskReadError,
    UnsupportedImageFormatError,
    read_image,
    read_mask_intensity,
)


def test_read_image_decodes_rgb_png_exactly(tmp_path: Path) -> None:
    path = tmp_path / "sample.PNG"
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    Image.fromarray(pixels, mode="RGB").save(path)

    decoded = read_image(path)

    assert np.array_equal(decoded.rgb, pixels)
    assert decoded.alpha is None


def test_read_image_separates_rgba_alpha_exactly(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    Image.fromarray(pixels, mode="RGBA").save(path)

    decoded = read_image(path)

    assert np.array_equal(decoded.rgb, pixels[:, :, :3])
    assert decoded.alpha is not None
    assert np.array_equal(decoded.alpha, pixels[:, :, 3])


def test_read_image_normalizes_grayscale_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    pixels = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)

    decoded = read_image(path)

    assert decoded.rgb.shape == (2, 2, 3)
    assert np.array_equal(decoded.rgb[:, :, 0], pixels)
    assert np.array_equal(decoded.rgb[:, :, 1], pixels)
    assert np.array_equal(decoded.rgb[:, :, 2], pixels)


def test_read_image_preserves_palette_transparency(tmp_path: Path) -> None:
    path = tmp_path / "palette.png"
    image = Image.new("P", (2, 1))
    image.putpalette([255, 0, 0, 0, 255, 0] + [0, 0, 0] * 254)
    image.putdata([0, 1])
    image.info["transparency"] = 0
    image.save(path)

    decoded = read_image(path)

    assert decoded.alpha is not None
    assert np.array_equal(decoded.alpha, [[0, 255]])


def test_read_image_applies_exif_orientation_before_reporting_shape(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (3, 2), color=(10, 20, 30))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)

    decoded = read_image(path)

    assert decoded.spatial_shape == (3, 2)


@pytest.mark.parametrize("extension", [".jpg", ".jpeg"])
def test_read_image_supports_both_jpeg_extensions(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"sample{extension}"
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(path)

    decoded = read_image(path)

    assert decoded.spatial_shape == (2, 3)
    assert decoded.alpha is None


def test_read_image_rejects_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedImageFormatError, match="expected one of"):
        read_image(tmp_path / "sample.webp")


def test_read_image_translates_decode_failure_and_preserves_cause(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    with pytest.raises(ImageReadError, match=r"broken\.png") as captured:
        read_image(path)

    assert captured.value.__cause__ is not None


def test_read_mask_intensity_prefers_alpha(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    pixels = np.full((2, 2, 4), 255, dtype=np.uint8)
    pixels[:, :, 3] = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    Image.fromarray(pixels, mode="RGBA").save(path)

    intensity = read_mask_intensity(path)

    assert np.array_equal(intensity, pixels[:, :, 3])


def test_read_mask_intensity_converts_non_alpha_image_to_grayscale(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    pixels = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)

    intensity = read_mask_intensity(path)

    assert np.array_equal(intensity, pixels)


def test_read_mask_intensity_does_not_apply_orientation_metadata(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    image = Image.new("L", (3, 2), color=255)
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)

    intensity = read_mask_intensity(path)

    assert intensity.shape == (2, 3)


def test_read_mask_intensity_translates_decode_failure(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    with pytest.raises(MaskReadError, match=r"broken\.png") as captured:
        read_mask_intensity(path)

    assert captured.value.__cause__ is not None
