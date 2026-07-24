"""Integration tests for atomic image and mask output."""

import os
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from watermark_removal_lab.common import BinaryMask, ImageData, UInt8Array
from watermark_removal_lab.image import UnsupportedImageFormatError, read_image
from watermark_removal_lab.image.output import (
    ImageWriteError,
    OutputExistsError,
    output_is_lossy,
    write_image_atomic,
    write_mask_atomic,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("output.png", False),
        ("output.PNG", False),
        ("output.jpg", True),
        ("output.JPEG", True),
    ],
)
def test_output_is_lossy_identifies_supported_formats(name: str, expected: bool) -> None:
    assert output_is_lossy(Path(name)) is expected


@pytest.mark.parametrize("name", ["output", "output.webp"])
def test_output_is_lossy_rejects_unsupported_formats(name: str) -> None:
    with pytest.raises(UnsupportedImageFormatError, match="unsupported output extension"):
        output_is_lossy(Path(name))


def test_write_image_atomic_preserves_png_rgb_and_alpha(tmp_path: Path) -> None:
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    alpha = np.array([[0, 64, 128], [192, 224, 255]], dtype=np.uint8)
    destination = tmp_path / "result.png"

    write_image_atomic(ImageData(rgb, alpha), destination)

    decoded = read_image(destination)
    assert np.array_equal(decoded.rgb, rgb)
    assert decoded.alpha is not None
    assert np.array_equal(decoded.alpha, alpha)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("extension", [".jpg", ".jpeg"])
def test_write_image_atomic_encodes_rgb_jpeg(tmp_path: Path, extension: str) -> None:
    destination = tmp_path / f"result{extension}"
    image = ImageData(np.full((4, 5, 3), 128, dtype=np.uint8))

    write_image_atomic(image, destination)

    decoded = read_image(destination)
    assert decoded.spatial_shape == (4, 5)
    assert decoded.alpha is None


def test_write_image_atomic_rejects_alpha_for_jpeg(tmp_path: Path) -> None:
    destination = tmp_path / "result.jpg"
    image = ImageData(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2), 255, dtype=np.uint8),
    )

    with pytest.raises(ImageWriteError, match="cannot preserve"):
        write_image_atomic(image, destination)

    assert not destination.exists()


def test_write_mask_atomic_serializes_binary_png(tmp_path: Path) -> None:
    destination = tmp_path / "mask.png"
    mask = BinaryMask(np.array([[False, True], [True, False]], dtype=np.bool_))

    write_mask_atomic(mask, destination)

    with Image.open(destination) as encoded:
        pixels = np.asarray(encoded.convert("L"), dtype=np.uint8)
    assert np.array_equal(pixels, [[0, 255], [255, 0]])


def test_write_mask_atomic_rejects_non_png_output(tmp_path: Path) -> None:
    mask = BinaryMask(np.zeros((2, 2), dtype=np.bool_))

    with pytest.raises(UnsupportedImageFormatError, match=r"expected \.png"):
        write_mask_atomic(mask, tmp_path / "mask.jpg")


def test_write_image_atomic_preserves_existing_output_without_replace(tmp_path: Path) -> None:
    destination = tmp_path / "result.png"
    destination.write_bytes(b"original")

    with pytest.raises(OutputExistsError, match="already exists"):
        write_image_atomic(
            ImageData(np.zeros((2, 2, 3), dtype=np.uint8)),
            destination,
        )

    assert destination.read_bytes() == b"original"


def test_write_image_atomic_replaces_existing_output(tmp_path: Path) -> None:
    destination = tmp_path / "result.png"
    destination.write_bytes(b"original")
    expected = np.full((2, 2, 3), 42, dtype=np.uint8)

    write_image_atomic(ImageData(expected), destination, replace=True)

    assert np.array_equal(read_image(destination).rgb, expected)


def test_write_image_atomic_rejects_a_missing_output_directory(tmp_path: Path) -> None:
    destination = tmp_path / "missing" / "result.png"

    with pytest.raises(ImageWriteError, match="does not exist"):
        write_image_atomic(
            ImageData(np.zeros((2, 2, 3), dtype=np.uint8)),
            destination,
        )


def test_write_image_atomic_translates_temporary_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.png"

    def fail_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        del args, kwargs
        raise OSError("simulated temporary-file failure")

    monkeypatch.setattr(tempfile, "mkstemp", fail_mkstemp)

    with pytest.raises(ImageWriteError, match="could not create temporary output") as captured:
        write_image_atomic(
            ImageData(np.zeros((2, 2, 3), dtype=np.uint8)),
            destination,
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert not destination.exists()


def test_write_image_atomic_cleans_temporary_file_after_encode_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.png"

    def fail_save(
        self: Image.Image,
        fp: object,
        format: str | None = None,
        **params: object,
    ) -> None:
        del self, fp, format, params
        raise OSError("simulated encode failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)

    with pytest.raises(ImageWriteError, match="could not write output") as captured:
        write_image_atomic(
            ImageData(np.zeros((2, 2, 3), dtype=np.uint8)),
            destination,
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_image_atomic_preserves_encode_error_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.png"
    original_unlink = Path.unlink

    def fail_save(
        self: Image.Image,
        fp: object,
        format: str | None = None,
        **params: object,
    ) -> None:
        del self, fp, format, params
        raise OSError("simulated encode failure")

    def fail_temporary_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".tmp":
            raise OSError("simulated cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(Image.Image, "save", fail_save)
        context.setattr(Path, "unlink", fail_temporary_unlink)
        with pytest.raises(ImageWriteError) as captured:
            write_image_atomic(
                ImageData(np.zeros((2, 2, 3), dtype=np.uint8)),
                destination,
            )

    assert str(captured.value.__cause__) == "simulated encode failure"
    for temporary in tmp_path.glob("*.tmp"):
        temporary.unlink()


def test_write_image_atomic_cleans_temporary_file_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.png"

    def fail_replace(source: object, target: object) -> None:
        del source, target
        raise OSError("simulated publish failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(ImageWriteError, match="could not write output") as captured:
        write_image_atomic(
            ImageData(np.zeros((2, 2, 3), dtype=np.uint8)),
            destination,
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_image_atomic_detects_a_late_output_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.png"
    original_exists = Path.exists
    destination_checks = 0

    def collide_after_encoding(path: Path) -> bool:
        nonlocal destination_checks
        if path == destination:
            destination_checks += 1
            return destination_checks > 1
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", collide_after_encoding)

    with pytest.raises(OutputExistsError, match="already exists"):
        write_image_atomic(
            ImageData(cast(UInt8Array, np.zeros((2, 2, 3), dtype=np.uint8))),
            destination,
        )

    assert not list(tmp_path.glob("*.tmp"))
