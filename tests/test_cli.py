"""Unit and integration tests for the CLI adapter."""

import argparse
import json
import signal
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

import watermark_removal_lab.cli as cli_module
from watermark_removal_lab.application import (
    BatchContractError,
    BatchPlan,
    BatchRunError,
    BatchSummary,
    CancellationToken,
    ImageRemovalError,
    ImageRemovalOutputError,
    ImageRemovalProcessingError,
    ModelManagementError,
    ModelManagementResult,
    ReviewedModelNotice,
    run_batch,
)
from watermark_removal_lab.models import LAMA_ONNX_FP32


def _save_rgb(path: Path) -> None:
    pixels = np.full((5, 5, 3), 100, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _batch_state_files(output_directory: Path) -> tuple[Path, Path, Path]:
    state_directories = tuple((output_directory / ".wrl-batch").iterdir())
    assert len(state_directories) == 1
    state_directory = state_directories[0]
    return (
        state_directory / "run.json",
        state_directory / "results.jsonl",
        state_directory / "summary.json",
    )


def _result_records(path: Path) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


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
        [
            "batch",
            "image",
            "--input-dir",
            "input",
            "--output-dir",
            "output",
        ],
        [
            "batch",
            "image",
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--box",
            "0,0,1,1",
            "--mask-dir",
            "masks",
        ],
        ["batch", "run", "manifest.jsonl"],
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


def test_cli_directory_batch_processes_recursive_inputs_and_custom_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    (input_directory / "nested").mkdir(parents=True)
    output_directory.mkdir()
    _save_rgb(input_directory / "root.jpg")
    _save_rgb(input_directory / "nested" / "child.png")

    exit_code = cli_module.main(
        [
            "batch",
            "image",
            "--input-dir",
            str(input_directory),
            "--output-dir",
            str(output_directory),
            "--box",
            "1,1,2,2",
            "--recursive",
            "--method",
            "ns",
            "--radius",
            "1.5",
            "--dilate",
            "1",
            "--output-format",
            "png",
            "--results",
            "reports/items.jsonl",
        ]
    )

    captured = capsys.readouterr()
    run_file, default_results, summary_file = _batch_state_files(output_directory)
    custom_results = output_directory / "reports/items.jsonl"
    assert exit_code == 0
    assert "discovered=2 validated=2 succeeded=2 skipped=0 failed=0 cancelled=0" in captured.out
    assert f"results: {custom_results}" in captured.out
    assert f"summary: {summary_file}" in captured.out
    assert captured.err == ""
    assert (output_directory / "root.png").is_file()
    assert (output_directory / "nested/child.png").is_file()
    assert run_file.is_file()
    assert not default_results.exists()
    assert [record["status"] for record in _result_records(custom_results)] == [
        "succeeded",
        "succeeded",
    ]


def test_cli_directory_batch_records_missing_mask_and_continues(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    mask_directory = tmp_path / "masks"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    mask_directory.mkdir()
    output_directory.mkdir()
    _save_rgb(input_directory / "a.png")
    _save_rgb(input_directory / "b.png")
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8), mode="L").save(mask_directory / "a.png")

    exit_code = cli_module.main(
        [
            "batch",
            "image",
            "--input-dir",
            str(input_directory),
            "--output-dir",
            str(output_directory),
            "--mask-dir",
            str(mask_directory),
        ]
    )

    captured = capsys.readouterr()
    _, results_file, _ = _batch_state_files(output_directory)
    records = _result_records(results_file)
    assert exit_code == 3
    assert "succeeded=1" in captured.out
    assert "failed=1" in captured.out
    assert captured.err == ""
    assert [record["status"] for record in records] == ["succeeded", "failed"]
    assert cast(dict[str, object], records[1]["error"])["code"] == "mask_not_found"
    assert (output_directory / "a.png").is_file()
    assert not (output_directory / "b.png").exists()


def test_cli_manifest_batch_applies_item_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    _save_rgb(tmp_path / "input.png")
    manifest = tmp_path / "batch.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "record": "batch",
                "schema_version": 1,
                "media": "image",
                "operation": "remove",
                "defaults": {"method": "telea", "radius": 3},
            },
            {
                "record": "item",
                "id": "sample",
                "input": "input.png",
                "output": "result.png",
                "box": [1, 1, 2, 2],
                "method": "ns",
                "radius": 1.0,
            },
        ],
    )

    exit_code = cli_module.main(
        [
            "batch",
            "run",
            str(manifest),
            "--output-dir",
            str(output_directory),
        ]
    )

    captured = capsys.readouterr()
    _, results_file, _ = _batch_state_files(output_directory)
    record = _result_records(results_file)[0]
    assert exit_code == 0
    assert "succeeded=1" in captured.out
    assert captured.err == ""
    assert record["status"] == "succeeded"
    assert cast(dict[str, object], record["normalized_request"])["method"] == "ns"
    assert (output_directory / "result.png").is_file()


def test_cli_manifest_fail_fast_cancels_remaining_items(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    _save_rgb(tmp_path / "valid.png")
    manifest = tmp_path / "batch.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "record": "batch",
                "schema_version": 1,
                "media": "image",
                "operation": "remove",
            },
            {
                "record": "item",
                "id": "missing",
                "input": "missing.png",
                "output": "missing-output.png",
                "box": [1, 1, 1, 1],
            },
            {
                "record": "item",
                "id": "valid",
                "input": "valid.png",
                "output": "valid-output.png",
                "box": [1, 1, 1, 1],
            },
        ],
    )

    exit_code = cli_module.main(
        [
            "batch",
            "run",
            str(manifest),
            "--output-dir",
            str(output_directory),
            "--fail-fast",
        ]
    )

    captured = capsys.readouterr()
    _, results_file, _ = _batch_state_files(output_directory)
    records = _result_records(results_file)
    assert exit_code == 3
    assert "failed=1 cancelled=1" in captured.out
    assert captured.err == ""
    assert [record["status"] for record in records] == ["failed", "cancelled"]
    assert cast(dict[str, object], records[1]["error"])["code"] == "fail_fast"
    assert not (output_directory / "valid-output.png").exists()


def test_cli_batch_skip_is_a_successful_terminal_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    _save_rgb(input_directory / "input.png")
    (output_directory / "input.png").write_bytes(b"existing")

    exit_code = cli_module.main(
        [
            "batch",
            "image",
            "--input-dir",
            str(input_directory),
            "--output-dir",
            str(output_directory),
            "--box",
            "1,1,1,1",
            "--overwrite",
            "skip",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "succeeded=0 skipped=1 failed=0" in captured.out
    assert captured.err == ""
    assert (output_directory / "input.png").read_bytes() == b"existing"


def test_cli_batch_reports_manifest_line_errors_as_configuration_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    manifest = tmp_path / "invalid.jsonl"
    manifest.write_text(
        '{"record":"batch","schema_version":1,"media":"image","operation":"remove"}\n{invalid}\n',
        encoding="utf-8",
    )

    exit_code = cli_module.main(
        [
            "batch",
            "run",
            str(manifest),
            "--output-dir",
            str(output_directory),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("failed: line 2:")
    assert not (output_directory / ".wrl-batch").exists()


def test_cli_batch_reports_preflight_errors_with_exit_code_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    _save_rgb(input_directory / "input.png")
    (output_directory / "input.png").write_bytes(b"existing")

    exit_code = cli_module.main(
        [
            "batch",
            "image",
            "--input-dir",
            str(input_directory),
            "--output-dir",
            str(output_directory),
            "--box",
            "1,1,1,1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("failed:")
    assert "already exists" in captured.err


def test_cli_batch_maps_contract_error_to_exit_code_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    _save_rgb(input_directory / "input.png")
    argv = [
        "batch",
        "image",
        "--input-dir",
        str(input_directory),
        "--output-dir",
        str(output_directory),
        "--box",
        "1,1,1,1",
    ]

    monkeypatch.setattr(
        cli_module,
        "plan_directory_batch",
        lambda request: (_ for _ in ()).throw(
            BatchContractError("invalid contract", code="invalid_contract")
        ),
    )
    assert cli_module.main(argv) == 2
    assert capsys.readouterr().err.strip() == "failed: invalid contract"


def test_cli_batch_reports_adapter_error_without_manifest_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()

    exit_code = cli_module.main(
        [
            "batch",
            "image",
            "--input-dir",
            str(input_directory),
            "--output-dir",
            str(output_directory),
            "--box",
            "1,1,1,1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.strip() == "failed: input directory contains no supported images"


def test_cli_batch_maps_fatal_orchestration_error_and_restores_sigint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    _save_rgb(input_directory / "input.png")
    previous_handler = signal.getsignal(signal.SIGINT)

    def fail_run(
        plan: BatchPlan,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> BatchSummary:
        del plan, cancellation_token
        raise BatchRunError("state failed", code="state_failed")

    monkeypatch.setattr(cli_module, "run_batch", fail_run)
    exit_code = cli_module.main(
        [
            "batch",
            "image",
            "--input-dir",
            str(input_directory),
            "--output-dir",
            str(output_directory),
            "--box",
            "1,1,1,1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert captured.err.strip() == "failed: state failed"
    assert signal.getsignal(signal.SIGINT) == previous_handler


def test_cli_batch_ctrl_c_marks_items_cancelled_and_returns_130(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    _save_rgb(input_directory / "input.png")
    previous_handler = signal.getsignal(signal.SIGINT)

    def cancel_before_run(
        plan: BatchPlan,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> BatchSummary:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert cancellation_token is not None
        return run_batch(plan, cancellation_token=cancellation_token)

    monkeypatch.setattr(cli_module, "run_batch", cancel_before_run)
    exit_code = cli_module.main(
        [
            "batch",
            "image",
            "--input-dir",
            str(input_directory),
            "--output-dir",
            str(output_directory),
            "--box",
            "1,1,1,1",
        ]
    )

    captured = capsys.readouterr()
    _, results_file, _ = _batch_state_files(output_directory)
    record = _result_records(results_file)[0]
    assert exit_code == 130
    assert "cancelled=1" in captured.out
    assert captured.err == ""
    assert record["status"] == "cancelled"
    assert cast(dict[str, object], record["error"])["code"] == "user_cancelled"
    assert signal.getsignal(signal.SIGINT) == previous_handler
    assert not (output_directory / "input.png").exists()


@pytest.mark.parametrize("failure_stage", ["install", "restore"])
def test_cli_batch_maps_signal_handler_failures_to_exit_code_four(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    _save_rgb(input_directory / "input.png")
    calls = 0

    def fail_signal(sig: int, handler: object) -> object:
        nonlocal calls
        del sig, handler
        calls += 1
        if failure_stage == "install" or calls == 2:
            raise ValueError(f"{failure_stage} failed")
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", fail_signal)
    exit_code = cli_module.main(
        [
            "batch",
            "image",
            "--input-dir",
            str(input_directory),
            "--output-dir",
            str(output_directory),
            "--box",
            "1,1,1,1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert f"{failure_stage} failed" in captured.err


def test_cli_batch_help_documents_authorized_use_and_b1_limits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured_exit:
        cli_module.main(["batch", "image", "--help"])

    captured = capsys.readouterr()
    normalized_help = " ".join(captured.out.split())
    assert captured_exit.value.code == 0
    assert "authorized to edit" in normalized_help
    assert "one worker" in normalized_help
    assert "does not support resume or retry" in normalized_help
    assert "--mask-dir" in normalized_help
    assert "--fail-fast" in normalized_help


def _model_notice() -> ReviewedModelNotice:
    return ReviewedModelNotice(
        model_id=LAMA_ONNX_FP32.model_id,
        source_url=LAMA_ONNX_FP32.source_url,
        model_card_url=LAMA_ONNX_FP32.model_card_url,
        declared_license=LAMA_ONNX_FP32.declared_license,
        dataset_notice=LAMA_ONNX_FP32.dataset_notice,
        dataset_terms_url=LAMA_ONNX_FP32.dataset_terms_url,
        size_bytes=LAMA_ONNX_FP32.size_bytes,
        sha256=LAMA_ONNX_FP32.sha256,
    )


def _model_result(path: Path) -> ModelManagementResult:
    return ModelManagementResult(
        notice=_model_notice(),
        path=path,
        status="verified",
        size_bytes=LAMA_ONNX_FP32.size_bytes,
        sha256=LAMA_ONNX_FP32.sha256,
    )


def test_cli_lama_empty_mask_parses_model_options_without_loading_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "output.png"
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
            "lama",
            "--provider",
            "cuda",
            "--crop-padding",
            "8",
            "--model-dir",
            str(tmp_path / "models"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["warnings"] == ["empty_mask"]
    assert payload["method"] == "lama"
    assert payload["options"]["radius"] is None
    assert payload["options"]["crop_padding"] == 8
    assert payload["backend"]["requested_provider"] == "cuda"
    assert payload["backend"]["effective_providers"] == []


@pytest.mark.parametrize(
    "extra_options",
    [
        ["--method", "lama", "--radius", "2"],
        ["--method", "telea", "--provider", "cpu"],
        ["--method", "telea", "--crop-padding", "8"],
        ["--method", "telea", "--model-dir", "models"],
    ],
)
def test_cli_rejects_backend_specific_option_mismatches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_options: list[str],
) -> None:
    input_path = tmp_path / "input.png"
    _save_rgb(input_path)

    exit_code = cli_module.main(
        [
            "image",
            "remove",
            str(input_path),
            str(tmp_path / "output.png"),
            "--box",
            "1,1,1,1",
            *extra_options,
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["error_code"] == "invalid_inpaint_options"


def test_cli_model_status_emits_json_without_installing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _model_result(tmp_path / "lama_fp32.onnx")
    observed: list[tuple[str, Path | None]] = []

    def fake_inspect(model_id: str, *, cache_root: Path | None) -> ModelManagementResult:
        observed.append((model_id, cache_root))
        return expected

    monkeypatch.setattr(cli_module, "inspect_reviewed_model", fake_inspect)

    exit_code = cli_module.main(
        [
            "model",
            "status",
            LAMA_ONNX_FP32.model_id,
            "--cache-dir",
            str(tmp_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["status"] == "verified"
    assert payload["expected_sha256"] == LAMA_ONNX_FP32.sha256
    assert observed == [(LAMA_ONNX_FP32.model_id, tmp_path)]


def test_cli_model_install_displays_terms_before_human_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notice = _model_notice()
    expected = _model_result(tmp_path / "lama_fp32.onnx")
    observed: list[tuple[str, Path | None, bool]] = []
    monkeypatch.setattr(cli_module, "reviewed_model_notice", lambda model_id: notice)

    def fake_install(
        model_id: str,
        *,
        cache_root: Path | None,
        terms_accepted: bool,
    ) -> ModelManagementResult:
        observed.append((model_id, cache_root, terms_accepted))
        return expected

    monkeypatch.setattr(cli_module, "install_reviewed_model", fake_install)

    exit_code = cli_module.main(
        [
            "model",
            "install",
            LAMA_ONNX_FP32.model_id,
            "--accept-model-terms",
            "--cache-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: verified" in captured.out
    assert f"source: {LAMA_ONNX_FP32.source_url}" in captured.err
    assert f"declared license: {LAMA_ONNX_FP32.declared_license}" in captured.err
    assert f"dataset notice: {LAMA_ONNX_FP32.dataset_notice}" in captured.err
    assert f"dataset terms: {LAMA_ONNX_FP32.dataset_terms_url}" in captured.err
    assert f"expected size: {LAMA_ONNX_FP32.size_bytes}" in captured.err
    assert f"expected SHA-256: {LAMA_ONNX_FP32.sha256}" in captured.err
    assert observed == [(LAMA_ONNX_FP32.model_id, tmp_path, True)]


def test_cli_model_install_terms_failure_emits_json_and_notice(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "reviewed_model_notice", lambda model_id: _model_notice())

    def fail_install(
        model_id: str,
        *,
        cache_root: Path | None,
        terms_accepted: bool,
    ) -> ModelManagementResult:
        del model_id, cache_root, terms_accepted
        raise ModelManagementError("terms required", code="model_terms_not_accepted")

    monkeypatch.setattr(cli_module, "install_reviewed_model", fail_install)

    exit_code = cli_module.main(["model", "install", LAMA_ONNX_FP32.model_id, "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload == {
        "status": "failed",
        "error_code": "model_terms_not_accepted",
        "error_message": "terms required",
    }
    assert f"model: {LAMA_ONNX_FP32.model_id}" in captured.err


def test_cli_model_status_failure_uses_processing_exit_code(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_inspect(model_id: str, *, cache_root: Path | None) -> ModelManagementResult:
        del model_id, cache_root
        raise ModelManagementError("cache unavailable", code="model_cache_failed")

    monkeypatch.setattr(cli_module, "inspect_reviewed_model", fail_inspect)

    exit_code = cli_module.main(["model", "status", LAMA_ONNX_FP32.model_id])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert captured.err.strip() == "failed: cache unavailable"


def test_cli_model_help_documents_explicit_terms(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured_exit:
        cli_module.main(["model", "install", "--help"])

    normalized_help = " ".join(capsys.readouterr().out.split())
    assert captured_exit.value.code == 0
    assert "--accept-model-terms" in normalized_help
    assert LAMA_ONNX_FP32.model_id in normalized_help


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
