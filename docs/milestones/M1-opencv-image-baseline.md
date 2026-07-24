# M1 Specification: OpenCV Image Baseline

[English](M1-opencv-image-baseline.md) | [简体中文](M1-opencv-image-baseline.zh-CN.md)

This document is the implementation and acceptance contract for M1. It defines a model-free, offline image-removal core, a scriptable CLI, and the B1 sequential batch foundation.

## 1. Goals

M1 must:

- remove a user-selected visible overlay from one image using a box or binary mask;
- support OpenCV Telea and Navier-Stokes inpainting;
- preserve alpha and pixels outside the final refined mask before lossy encoding;
- expose the same single-image application service to the CLI and batch orchestrator;
- process deterministic sequential image batches without duplicating algorithm logic;
- run offline without model weights, GPU runtimes, or network access;
- provide focused tests and actionable failures.

## 2. Non-goals

M1 does not include:

- automatic watermark detection;
- SAM/SAM 2, LaMa, ONNX Runtime, or other model inference;
- interactive clicks or graphical mask editing;
- desktop, web, or API adapters;
- video decoding or frame processing;
- concurrent workers, GPU scheduling, resume, or automatic retry;
- temporal consistency;
- automatic preservation of all EXIF, ICC, or vendor-specific metadata;
- platform-specific paid-preview watermark logic.

## 3. Architecture slice

```text
CLI or B1 batch orchestrator
            ↓
Single-image removal service
            ↓
Image IO + mask preparation + OpenCV inpainter
            ↓
Mask-only compositing + atomic output
```

The batch orchestrator creates single-image requests and collects results. It must not read image arrays, call OpenCV directly, or implement separate removal behavior.

## 4. Canonical data contract

- decoded image: `uint8`, `H × W × 3`, RGB;
- optional alpha: separate `uint8`, `H × W`;
- internal binary mask: `bool`, `H × W`;
- serialized mask: 0 background, 255 selected;
- internal box: `(x_min, y_min, x_max, y_max)`, right/bottom exclusive;
- CLI box: `x,y,width,height`, converted to the internal box;
- OpenCV adapter: explicit RGB-to-BGR and BGR-to-RGB conversion.

The service must not mutate input arrays. The result contains a new RGB array and the original alpha array.

## 5. Single-image CLI

Command shape:

```text
wrl image remove INPUT OUTPUT (--box X,Y,W,H | --mask MASK)
    [--method telea|ns]
    [--radius PIXELS]
    [--dilate PIXELS]
    [--mask-threshold 0..255]
    [--save-mask PATH]
    [--overwrite error|skip|replace]
    [--json]
```

Examples:

```powershell
wrl image remove input.png output.png --box 100,50,240,80 --method telea

wrl image remove input.png output.png --mask watermark-mask.png `
  --method ns --radius 3 --dilate 2 --save-mask final-mask.png
```

Required behavior:

- `--box` and `--mask` are mutually exclusive and one is required;
- `--method` defaults to `telea`;
- `--radius` is a positive pixel radius and defaults to 3;
- `--dilate` is a non-negative pixel radius and defaults to 0;
- `--mask-threshold` defaults to 127;
- `--overwrite` defaults to `error`;
- `--json` writes one machine-readable result object to stdout;
- human progress and diagnostics go to stderr when JSON output is enabled.

## 6. Box and mask behavior

### Box input

- `x`, `y`, `width`, and `height` are integers;
- width and height must be positive;
- the complete box must be inside the decoded, orientation-normalized image;
- touching an image edge is valid;
- out-of-bounds boxes fail instead of being silently clipped.

### External mask input

- the mask must match the decoded image width and height;
- if the mask has alpha, alpha is used as mask intensity;
- otherwise the mask is converted to grayscale intensity;
- pixels greater than `--mask-threshold` are selected;
- an empty mask is a successful no-op with a warning;
- a full-frame mask is valid but may produce poor visual quality;
- mask orientation is not guessed or rotated automatically.

### Dilation

`--dilate N` expands the binary mask with an elliptical `(2N + 1) × (2N + 1)` kernel. The dilated mask is the final refined mask and defines the only pixels the algorithm may change.

## 7. Image IO and output safety

### Required format support

- M1 must decode and encode PNG, JPEG `.jpg`, and JPEG `.jpeg`.
- Directory discovery matches `.png`, `.jpg`, and `.jpeg` case-insensitively.
- PNG is the required lossless and alpha-preserving output format.
- JPEG output is supported only for images without alpha and is always reported as lossy.
- BMP, WebP, TIFF, and other formats are outside the M1 exit gate unless they are explicitly documented and covered by integration tests.

- Decode image orientation before box validation.
- Normalize output orientation; full metadata preservation is not an M1 guarantee.
- Preserve the original alpha channel exactly when the output format supports alpha.
- Reject an alpha-bearing input when the chosen output format cannot preserve alpha unless the user explicitly opts into alpha loss in a future interface.
- Never overwrite the input path.
- Write to a temporary file in the destination directory, validate completion, then atomically replace the final path.
- Remove temporary files after failures where safe.
- Do not leave a partial final output.

Overwrite policies:

| Policy | Behavior |
|---|---|
| `error` | Fail before processing if the output exists |
| `skip` | Return a successful skipped result without changing the output |
| `replace` | Atomically replace an existing output after successful processing |

JPEG and other lossy encoders may change pixels outside the mask during encoding. The JSON result must identify lossy output, and correctness tests compare in-memory pixels or lossless PNG output.

## 8. OpenCV inpainting contract

- Telea maps to the OpenCV Telea algorithm.
- `ns` maps to the OpenCV Navier-Stokes algorithm.
- The OpenCV adapter receives RGB plus a boolean mask and performs required conversions internally.
- The adapter may calculate a full-frame candidate, but the service composites candidate pixels only where the final mask is true.
- An empty mask bypasses OpenCV and returns an unchanged copy.
- Backend failures are translated into a domain error while preserving the original exception.

## 9. B1 sequential batch CLI

B1 has two input modes that compile into the same batch-item contract.

### Directory mode

```text
wrl batch image --input-dir INPUT_DIR --output-dir OUTPUT_DIR
    (--box X,Y,W,H | --mask-dir MASK_DIR)
    [--recursive]
    [--method telea|ns]
    [--radius PIXELS]
    [--dilate PIXELS]
    [--output-format preserve|png]
    [--overwrite error|skip|replace]
    [--results PATH]
    [--fail-fast]
```

Rules:

- supported images are discovered in deterministic relative-path order;
- default discovery is non-recursive;
- `--recursive` preserves relative directory structure in the output;
- a shared box is applied to every image and validated independently;
- `--mask-dir` mirrors relative paths and replaces the input extension with `.png`;
- missing paired masks are item failures;
- `--output-format preserve` keeps each input extension;
- `--output-format png` replaces each output extension with `.png`;
- directory and output paths are preflighted before processing.

### Manifest mode

```text
wrl batch run MANIFEST.jsonl --output-dir OUTPUT_DIR
    [--results PATH]
    [--overwrite error|skip|replace]
    [--fail-fast]
```

The versioned JSON Lines manifest begins with one batch record, followed by item records:

```json
{"record":"batch","schema_version":1,"media":"image","operation":"remove","defaults":{"method":"telea","radius":3,"dilate":1}}
{"record":"item","id":"sample-a","input":"inputs/a.png","output":"a.png","box":[10,20,120,40]}
{"record":"item","id":"sample-b","input":"inputs/b.png","output":"b.png","mask":"masks/b.png","method":"ns"}
```

Manifest rules:

- the first record requires `media` set to `image` and `operation` set to `remove`;
- item IDs are unique non-empty strings;
- paths are relative to the manifest directory;
- output paths are relative to `--output-dir`;
- each item provides exactly one of `box` or `mask`;
- item fields override batch defaults;
- unknown schema versions, missing required fields, and unsupported media/operation values fail before processing;
- duplicate output paths and in-place input/output paths fail preflight;
- traversal outside declared roots is rejected;
- item execution order matches manifest order.

The complete batch contract is defined in [../BATCH_PROCESSING.md](../BATCH_PROCESSING.md).

## 10. B1 execution behavior

- B1 uses exactly one worker.
- Preflight creates one stable `RUN_ID` for the run.
- When `--results` is omitted, results default to `<OUTPUT_DIR>/.wrl-batch/<RUN_ID>/results.jsonl`; `run.json` and `summary.json` use the same state directory.
- A custom `--results` path overrides only the JSON Lines result location and must not alias an input, mask, media output, or batch metadata file.
- Default behavior continues after an item failure.
- `--fail-fast` stops scheduling after the first failed item.
- Completed outputs remain valid when later items fail.
- Each item produces a result with `succeeded`, `skipped`, `failed`, or `cancelled` status.
- Results are appended to a JSON Lines result file after each item.
- The CLI prints totals for discovered, succeeded, skipped, failed, and cancelled items.
- Fail-fast marks every remaining unscheduled item as `cancelled` with reason `fail_fast`.
- Ctrl+C stops after the active atomic step, marks every remaining unscheduled item as `cancelled` with reason `user_cancelled`, and exits with 130.
- B1 does not resume an interrupted run; B2 introduces verified resume.

## 11. Result fields

Single and batch item results use a shared machine-readable shape containing at least:

- schema version;
- item ID when applicable;
- input and output paths;
- status;
- method and normalized options;
- image width and height;
- final-mask selected-pixel count;
- duration in milliseconds;
- whether output encoding is lossy;
- warnings;
- stable error code and message on failure.

Do not include image pixels, credentials, stack traces, or unnecessary absolute private paths in standard result output.

## 12. Exit codes

| Code | Meaning |
|---|---|
| `0` | Single item succeeded/skipped, or all batch items succeeded/skipped |
| `2` | Invalid CLI arguments, manifest, box, mask, or preflight configuration |
| `3` | Processing completed with one or more item failures |
| `4` | Fatal orchestration, IO, or output-commit failure |
| `130` | Cancelled by user |

Batch item failures must be visible in both the exit code and result file.

## 13. Test matrix

### Unit tests

- CLI box parsing and half-open conversion;
- zero, negative, edge-touching, and out-of-bounds boxes;
- mask loading, thresholding, empty/full masks, and mismatched dimensions;
- elliptical dilation for 0 and positive radii;
- RGB/BGR round trip;
- mask-only compositing;
- immutable inputs and preserved alpha;
- overwrite-policy decisions;
- deterministic directory and manifest ordering;
- manifest schema, duplicate IDs, collisions, and traversal rejection.

### Integration tests

- Telea and NS on a synthetic RGB PNG;
- RGBA PNG with byte-identical alpha;
- grayscale input normalized to RGB;
- empty-mask no-op;
- lossless output with identical final-mask exterior pixels;
- JPEG output marked as lossy;
- atomic replacement and cleanup after simulated write failure;
- single-image CLI success and invalid-input exit codes;
- directory batch with shared box;
- directory batch with paired masks;
- manifest batch with per-item overrides;
- partial batch failure, fail-fast, skip, and replace behavior;
- Ctrl+C or cancellation boundary without a partial final output.

All default tests are CPU-only, offline, and use synthetic fixtures.

## 14. Acceptance checklist

M1 is complete only when:

- [x] single-image box and mask workflows run through one application service;
- [x] Telea and NS are selectable and tested;
- [x] final-mask exterior pixels are unchanged before lossy encoding;
- [x] alpha is preserved for supported outputs;
- [x] inputs are never modified in place;
- [x] output writes are atomic and overwrite behavior is explicit;
- [x] directory and manifest B1 batches reuse the single-image service;
- [x] batch ordering and results are deterministic;
- [x] partial failures are visible and do not invalidate completed items;
- [x] all required unit and integration tests pass offline on CPU;
- [x] CLI help and examples document limitations and responsible use;
- [x] no model, GUI, video, or automatic-detection dependency is introduced.

Acceptance evidence is recorded in the
[M1/B1 acceptance report](../acceptance/M1-B1-acceptance.md).

## 15. Suggested implementation slices

1. package and test-tool scaffold;
2. image, alpha, box, and mask types;
3. image IO and mask utilities;
4. OpenCV inpainter and mask-only compositing;
5. single-image application service and CLI;
6. B1 batch contracts, preflight, and results;
7. directory and manifest batch adapters;
8. integration tests and M1 documentation.

Each slice should be independently reviewable and use the repository commit-message rules.
