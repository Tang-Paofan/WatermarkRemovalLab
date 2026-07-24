"""Command-line adapter for Watermark Removal Lab."""

import argparse
import json
import math
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from types import FrameType
from typing import cast

from watermark_removal_lab.application import (
    BatchCancellationToken,
    BatchContractError,
    BatchFailurePolicy,
    BatchInputError,
    BatchPlan,
    BatchPreflightError,
    BatchRunError,
    BatchSummary,
    BoxMaskSource,
    DirectoryBatchRequest,
    DirectoryOutputFormat,
    ImageRemovalError,
    ImageRemovalInputError,
    ImageRemovalOutputError,
    ImageRemovalProcessingError,
    ImageRemovalRequest,
    ManifestBatchRequest,
    MaskFileSource,
    OverwritePolicy,
    build_failed_image_removal_result,
    plan_directory_batch,
    plan_manifest_batch,
    remove_image,
    run_batch,
)
from watermark_removal_lab.common import Box, DataContractError
from watermark_removal_lab.image import OpenCVInpaintMethod


def _parse_box(value: str) -> Box:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must use X,Y,WIDTH,HEIGHT")
    try:
        x, y, width, height = (int(part) for part in parts)
        return Box.from_xywh(x=x, y=y, width=width, height=height)
    except (ValueError, DataContractError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_mask_threshold(value: str) -> int:
    parsed = _parse_non_negative_int(value)
    if parsed > 255:
        raise argparse.ArgumentTypeError("mask threshold must be between 0 and 255")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wrl",
        description=(
            "Remove a user-selected visible overlay from media you own or are authorized to edit."
        ),
        epilog=(
            "M1 requires a user-provided box or mask. JPEG output is lossy; "
            "automatic detection is not included."
        ),
    )
    command_parsers = parser.add_subparsers(dest="command", required=True)
    image_parser = command_parsers.add_parser("image", help="Image operations")
    image_commands = image_parser.add_subparsers(dest="image_command", required=True)
    remove_parser = image_commands.add_parser(
        "remove",
        help="Remove a selected visible overlay",
        description=(
            "Remove a user-selected visible overlay from an image you own or "
            "are authorized to edit."
        ),
        epilog=(
            "Provide exactly one box or mask. PNG preserves alpha; JPEG is lossy "
            "and cannot preserve alpha. Automatic detection is not included."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    remove_parser.add_argument("input", type=Path, help="Input PNG or JPEG path")
    remove_parser.add_argument("output", type=Path, help="Output PNG or JPEG path")
    selection = remove_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--box", type=_parse_box, help="Selection as X,Y,WIDTH,HEIGHT")
    selection.add_argument("--mask", type=Path, help="External mask PNG or JPEG path")
    remove_parser.add_argument(
        "--method",
        choices=tuple(OpenCVInpaintMethod),
        type=OpenCVInpaintMethod,
        default=OpenCVInpaintMethod.TELEA,
        help="OpenCV inpainting algorithm",
    )
    remove_parser.add_argument(
        "--radius",
        type=_parse_positive_float,
        default=3.0,
        help="Inpainting radius in pixels",
    )
    remove_parser.add_argument(
        "--dilate",
        type=_parse_non_negative_int,
        default=0,
        help="Elliptical mask dilation radius in pixels",
    )
    remove_parser.add_argument(
        "--mask-threshold",
        type=_parse_mask_threshold,
        default=127,
        help="Select external-mask intensities strictly above this value",
    )
    remove_parser.add_argument(
        "--save-mask",
        type=Path,
        help="Optional path for the final 0/255 PNG mask",
    )
    remove_parser.add_argument(
        "--overwrite",
        choices=tuple(OverwritePolicy),
        type=OverwritePolicy,
        default=OverwritePolicy.ERROR,
        help="Existing-output policy",
    )
    remove_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write one machine-readable result object",
    )
    batch_parser = command_parsers.add_parser("batch", help="Sequential image batches")
    batch_commands = batch_parser.add_subparsers(dest="batch_command", required=True)
    directory_parser = batch_commands.add_parser(
        "image",
        help="Process images discovered in a directory",
        description=(
            "Sequentially process images you own or are authorized to edit, "
            "using one shared box or mirrored mask files."
        ),
        epilog=(
            "Input, output, and optional mask directories must already exist. "
            "B1 uses one worker and does not support resume or retry."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    directory_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Existing input image directory",
    )
    directory_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Existing output directory",
    )
    directory_selection = directory_parser.add_mutually_exclusive_group(required=True)
    directory_selection.add_argument(
        "--box",
        type=_parse_box,
        help="Shared selection as X,Y,WIDTH,HEIGHT",
    )
    directory_selection.add_argument(
        "--mask-dir",
        type=Path,
        help="Existing directory of mirrored PNG masks",
    )
    directory_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Discover images below nested input directories",
    )
    directory_parser.add_argument(
        "--method",
        choices=tuple(OpenCVInpaintMethod),
        type=OpenCVInpaintMethod,
        default=OpenCVInpaintMethod.TELEA,
        help="OpenCV inpainting algorithm",
    )
    directory_parser.add_argument(
        "--radius",
        type=_parse_positive_float,
        default=3.0,
        help="Inpainting radius in pixels",
    )
    directory_parser.add_argument(
        "--dilate",
        type=_parse_non_negative_int,
        default=0,
        help="Elliptical mask dilation radius in pixels",
    )
    directory_parser.add_argument(
        "--output-format",
        choices=tuple(DirectoryOutputFormat),
        type=DirectoryOutputFormat,
        default=DirectoryOutputFormat.PRESERVE,
        help="Output extension policy",
    )
    _add_batch_execution_options(directory_parser)

    manifest_parser = batch_commands.add_parser(
        "run",
        help="Process a versioned JSON Lines manifest",
        description=(
            "Sequentially process a strict image-removal manifest containing "
            "only media you own or are authorized to edit."
        ),
        epilog=(
            "The output directory must already exist. B1 uses one worker and "
            "does not support resume or retry."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    manifest_parser.add_argument(
        "manifest",
        type=Path,
        help="Versioned JSON Lines manifest",
    )
    manifest_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Existing output directory",
    )
    _add_batch_execution_options(manifest_parser)
    return parser


def _add_batch_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results",
        type=Path,
        help="Optional JSON Lines result path",
    )
    parser.add_argument(
        "--overwrite",
        choices=tuple(OverwritePolicy),
        type=OverwritePolicy,
        default=OverwritePolicy.ERROR,
        help="Existing-output policy",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Cancel unscheduled items after the first failure",
    )


def _request_from_args(args: argparse.Namespace) -> ImageRemovalRequest:
    box = cast(Box | None, args.box)
    mask_path = cast(Path | None, args.mask)
    mask_source = (
        BoxMaskSource(box)
        if box is not None
        else MaskFileSource(cast(Path, mask_path), threshold=cast(int, args.mask_threshold))
    )
    return ImageRemovalRequest(
        input_path=cast(Path, args.input),
        output_path=cast(Path, args.output),
        mask_source=mask_source,
        method=cast(OpenCVInpaintMethod, args.method),
        radius=cast(float, args.radius),
        dilation_radius=cast(int, args.dilate),
        save_mask_path=cast(Path | None, args.save_mask),
        overwrite=cast(OverwritePolicy, args.overwrite),
    )


def _emit_result(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return

    status = result["status"]
    output = result["output"]
    if status == "failed":
        print(f"failed: {result['error_message']}", file=sys.stderr)
    elif status == "skipped":
        print(f"skipped: {output}")
    else:
        print(f"succeeded: {output}")
    for warning in cast(list[str], result["warnings"]):
        print(f"warning: {warning}", file=sys.stderr)


def _exit_code(error: ImageRemovalError) -> int:
    if isinstance(error, ImageRemovalInputError):
        return 2
    if isinstance(error, ImageRemovalProcessingError):
        return 3
    if isinstance(error, ImageRemovalOutputError):
        return 4
    return 4


def _failure_policy(args: argparse.Namespace) -> BatchFailurePolicy:
    return (
        BatchFailurePolicy.FAIL_FAST if cast(bool, args.fail_fast) else BatchFailurePolicy.CONTINUE
    )


def _plan_from_batch_args(args: argparse.Namespace) -> BatchPlan:
    if args.batch_command == "image":
        directory_request = DirectoryBatchRequest(
            input_directory=cast(Path, args.input_dir),
            output_directory=cast(Path, args.output_dir),
            box=cast(Box | None, args.box),
            mask_directory=cast(Path | None, args.mask_dir),
            recursive=cast(bool, args.recursive),
            method=cast(OpenCVInpaintMethod, args.method),
            radius=cast(float, args.radius),
            dilation_radius=cast(int, args.dilate),
            output_format=cast(DirectoryOutputFormat, args.output_format),
            overwrite_policy=cast(OverwritePolicy, args.overwrite),
            failure_policy=_failure_policy(args),
            results_path=cast(Path | None, args.results),
        )
        return plan_directory_batch(directory_request)

    manifest_request = ManifestBatchRequest(
        manifest_path=cast(Path, args.manifest),
        output_directory=cast(Path, args.output_dir),
        results_path=cast(Path | None, args.results),
        overwrite_policy=cast(OverwritePolicy, args.overwrite),
        failure_policy=_failure_policy(args),
    )
    return plan_manifest_batch(manifest_request)


def _run_batch_with_sigint(
    plan: BatchPlan,
    *,
    cancellation_token: BatchCancellationToken,
) -> BatchSummary:
    def request_cancellation(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        cancellation_token.cancel()

    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_cancellation)
    except (OSError, ValueError) as error:
        raise BatchRunError(
            f"could not install the Ctrl+C cancellation handler: {error}",
            code="signal_handler_failed",
        ) from error

    try:
        return run_batch(plan, cancellation_token=cancellation_token)
    finally:
        try:
            signal.signal(signal.SIGINT, previous_handler)
        except (OSError, ValueError) as error:
            raise BatchRunError(
                f"could not restore the Ctrl+C cancellation handler: {error}",
                code="signal_restore_failed",
            ) from error


def _emit_batch_error(error: Exception) -> None:
    if isinstance(error, BatchInputError) and error.line_number is not None:
        print(f"failed: line {error.line_number}: {error}", file=sys.stderr)
    else:
        print(f"failed: {error}", file=sys.stderr)


def _emit_batch_summary(plan: BatchPlan, summary: BatchSummary) -> None:
    print(
        f"batch {summary.run_id}: "
        f"discovered={summary.discovered} "
        f"validated={summary.validated} "
        f"succeeded={summary.succeeded} "
        f"skipped={summary.skipped} "
        f"failed={summary.failed} "
        f"cancelled={summary.cancelled}"
    )
    print(f"results: {plan.result_file}")
    print(f"summary: {plan.summary_file}")


def _run_single_image_command(args: argparse.Namespace) -> int:
    request = _request_from_args(args)
    started = perf_counter()
    try:
        result = remove_image(request)
    except ImageRemovalError as error:
        result = build_failed_image_removal_result(
            request,
            error,
            duration_ms=(perf_counter() - started) * 1000,
        )
        _emit_result(result.to_dict(), json_output=cast(bool, args.json_output))
        return _exit_code(error)

    _emit_result(result.to_dict(), json_output=cast(bool, args.json_output))
    return 0


def _run_batch_command(args: argparse.Namespace) -> int:
    try:
        plan = _plan_from_batch_args(args)
    except (BatchContractError, BatchInputError, BatchPreflightError) as error:
        _emit_batch_error(error)
        return 2

    cancellation_token = BatchCancellationToken()
    try:
        summary = _run_batch_with_sigint(
            plan,
            cancellation_token=cancellation_token,
        )
    except BatchRunError as error:
        _emit_batch_error(error)
        return 4

    _emit_batch_summary(plan, summary)
    if cancellation_token.is_cancelled():
        return 130
    if summary.failed:
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "image":
        return _run_single_image_command(args)
    return _run_batch_command(args)
