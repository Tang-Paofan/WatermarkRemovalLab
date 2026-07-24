"""Unit and integration tests for the CLI adapter."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import watermark_removal_lab.cli as cli_module
from watermark_removal_lab.application import (
    ImageRemovalError,
    ImageRemovalOutputError,
    ImageRemovalProcessingError,
)


def _save_rgb(path: Path) -> None:
    pixels = np.full((5, 5, 3), 100, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_cli_box_workflow_succeeds_with_human_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _save_rgb(input_path)

    exit_code = cli_module.main(
        [
            "image",
            "remove",
            str(input_path),
            str(output_path),
            "--box",
            "2,2,1,1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == f"succeeded: {output_path}"
    assert captured.err == ""
    assert output_path.exists()


def test_cli_mask_workflow_emits_one_json_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "output.jpg"
    _save_rgb(input_path)
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8), mode="L").save(mask_path)

    exit_code = cli_module.main(
        [
            "image",
            "remove",
            str(input_path),
            str(output_path),
            "--mask",
            str(mask_path),
            "--method",
            "ns",
            "--radius",
            "1.5",
            "--dilate",
            "1",
            "--mask-threshold",
            "100",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert payload["status"] == "succeeded"
    assert payload["method"] == "ns"
    assert payload["options"] == {
        "radius": 1.5,
        "dilate": 1,
        "mask_threshold": 100,
    }
    assert payload["lossy_output"] is True
    assert payload["warnings"] == ["empty_mask", "lossy_output"]


def test_cli_skip_reports_human_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _save_rgb(input_path)
    output_path.write_bytes(b"existing")

    exit_code = cli_module.main(
        [
            "image",
            "remove",
            str(input_path),
            str(output_path),
            "--box",
            "1,1,1,1",
            "--overwrite",
            "skip",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == f"skipped: {output_path}"
    assert captured.err.strip() == "warning: output_exists"


def test_cli_input_failure_emits_json_and_exit_code_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _save_rgb(input_path)
    output_path.write_bytes(b"existing")

    exit_code = cli_module.main(
        [
            "image",
            "remove",
            str(input_path),
            str(output_path),
            "--box",
            "1,1,1,1",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert captured.err == ""
    assert payload["status"] == "failed"
    assert payload["error_code"] == "output_exists"


@pytest.mark.parametrize(
    ("error", "expected_exit"),
    [
        (ImageRemovalProcessingError("failed", code="processing"), 3),
        (ImageRemovalOutputError("failed", code="output"), 4),
        (ImageRemovalError("failed", code="unknown"), 4),
    ],
)
def test_cli_maps_service_failures_to_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error: ImageRemovalError,
    expected_exit: int,
) -> None:
    input_path = tmp_path / "input.png"
    _save_rgb(input_path)

    def fail_service(request: object) -> object:
        del request
        raise error

    monkeypatch.setattr(cli_module, "remove_image", fail_service)

    exit_code = cli_module.main(
        [
            "image",
            "remove",
            str(input_path),
            str(tmp_path / "output.png"),
            "--box",
            "1,1,1,1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == expected_exit
    assert captured.out == ""
    assert captured.err.strip() == "failed: failed"


@pytest.mark.parametrize(
    "argv",
    [
        ["image", "remove", "input.png", "output.png"],
        [
            "image",
            "remove",
            "input.png",
            "output.png",
            "--box",
            "0,0,1,1",
            "--mask",
            "mask.png",
        ],
        [
            "image",
            "remove",
            "input.png",
            "output.png",
            "--box",
            "0,0,1,1",
            "--method",
            "unknown",
        ],
    ],
)
def test_cli_rejects_invalid_argument_combinations(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured_exit:
        cli_module.main(argv)

    captured = capsys.readouterr()
    assert captured_exit.value.code == 2
    assert "error:" in captured.err


def test_cli_help_documents_authorized_use_and_m1_limitations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured_exit:
        cli_module.main(["--help"])

    captured = capsys.readouterr()
    normalized_help = " ".join(captured.out.split())
    assert captured_exit.value.code == 0
    assert "authorized to edit" in normalized_help
    assert "automatic detection is not included" in normalized_help


def test_cli_remove_help_documents_defaults_and_output_limits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured_exit:
        cli_module.main(["image", "remove", "--help"])

    captured = capsys.readouterr()
    normalized_help = " ".join(captured.out.split())
    assert captured_exit.value.code == 0
    assert "authorized to edit" in normalized_help
    assert "PNG preserves alpha" in normalized_help
    assert "JPEG is lossy" in normalized_help
    assert "default: telea" in normalized_help
    assert "default: 127" in normalized_help


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1,2,3", "X,Y,WIDTH,HEIGHT"),
        ("a,2,3,4", "invalid literal"),
        ("1,2,0,4", "width must be positive"),
    ],
)
def test_parse_box_rejects_invalid_values(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        cli_module._parse_box(value)


def test_parse_box_creates_half_open_box() -> None:
    box = cli_module._parse_box("10,20,30,40")

    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (10, 20, 40, 60)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("value", "must be a number"),
        ("0", "finite and positive"),
        ("nan", "finite and positive"),
    ],
)
def test_parse_positive_float_rejects_invalid_values(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        cli_module._parse_positive_float(value)


def test_parse_positive_float_accepts_a_positive_number() -> None:
    assert cli_module._parse_positive_float("2.5") == 2.5


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("value", "must be an integer"),
        ("-1", "must be non-negative"),
    ],
)
def test_parse_non_negative_int_rejects_invalid_values(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        cli_module._parse_non_negative_int(value)


def test_parse_non_negative_int_accepts_zero() -> None:
    assert cli_module._parse_non_negative_int("0") == 0


def test_parse_mask_threshold_rejects_values_above_255() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="between 0 and 255"):
        cli_module._parse_mask_threshold("256")


def test_parse_mask_threshold_accepts_255() -> None:
    assert cli_module._parse_mask_threshold("255") == 255
