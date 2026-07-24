# CLI Guide

[English](CLI.md) | [简体中文](CLI.zh-CN.md)

Use Watermark Removal Lab only with images you own, created, or are authorized to edit. The CLI
requires a user-provided box or mask and does not automatically detect watermarks.

## Run the CLI

Set up the repository environment as described in the
[development guide](DEVELOPMENT.md), then run:

```powershell
uv run wrl --help
```

All examples below use synthetic filenames. Create output directories before running a command.

## Remove an overlay from one image

Use a half-open box expressed as `X,Y,WIDTH,HEIGHT`:

```powershell
uv run wrl image remove input.png output.png --box 620,420,160,60
```

Or provide an external mask:

```powershell
uv run wrl image remove input.png output.png `
    --mask mask.png `
    --mask-threshold 127 `
    --method ns `
    --radius 3 `
    --dilate 1 `
    --save-mask final-mask.png
```

Mask intensities strictly above the threshold are selected. `--method` accepts `telea` or `ns`.
Use `--json` for one machine-readable single-image result.

## Process an image directory

Create the output directory, then apply one shared box to every discovered image:

```powershell
New-Item -ItemType Directory -Force output | Out-Null

uv run wrl batch image `
    --input-dir input `
    --output-dir output `
    --box 620,420,160,60 `
    --recursive `
    --method telea `
    --radius 3 `
    --dilate 1 `
    --output-format preserve
```

Without `--recursive`, only the immediate input directory is scanned. Supported images are processed
in deterministic relative-path order. `--output-format png` changes every output extension to
`.png`; `preserve` retains the input extension.

To use masks, replace `--box` with `--mask-dir`:

```powershell
uv run wrl batch image `
    --input-dir input `
    --output-dir output `
    --mask-dir masks `
    --recursive
```

The mask directory mirrors input relative paths and replaces each input extension with `.png`.
For example, `input/nested/photo.jpg` uses `masks/nested/photo.png`. A missing mask is recorded as
an item failure; other items continue unless `--fail-fast` is set. Directory masks use threshold
127; manifests can override `mask_threshold` in batch defaults or individual items.

## Process a manifest

A version 1 JSON Lines manifest begins with one batch record followed by item records:

```json
{"record":"batch","schema_version":1,"media":"image","operation":"remove","defaults":{"method":"telea","radius":3,"dilate":1}}
{"record":"item","id":"sample-a","input":"inputs/a.png","output":"a.png","box":[10,20,120,40]}
{"record":"item","id":"sample-b","input":"inputs/b.png","output":"b.png","mask":"masks/b.png","method":"ns"}
```

Run it with:

```powershell
New-Item -ItemType Directory -Force output | Out-Null

uv run wrl batch run batch.jsonl `
    --output-dir output `
    --overwrite error
```

Manifest input and mask paths are relative to the manifest directory. Output paths are relative to
the declared output directory. Every item must provide exactly one `box` or `mask`.

## Batch results and cancellation

By default, one run writes:

```text
OUTPUT_DIR/
└── .wrl-batch/
    └── RUN_ID/
        ├── run.json
        ├── results.jsonl
        └── summary.json
```

`results.jsonl` is flushed after every terminal item. `summary.json` is published atomically after
all discovered items become terminal. `--results PATH` changes only the JSON Lines result location;
a relative path is resolved below the output directory.

The default failure policy continues after an item failure. `--fail-fast` cancels every remaining
unscheduled item with reason `fail_fast`.

Pressing Ctrl+C requests cancellation at a safe boundary. The active atomic image operation is
allowed to finish, completed outputs and flushed records remain valid, and remaining unscheduled
items are recorded as `cancelled` with reason `user_cancelled`.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All applicable items succeeded or were intentionally skipped |
| `2` | Invalid arguments, manifest, paths, or preflight configuration |
| `3` | Processing completed with one or more item failures |
| `4` | Fatal orchestration, state, IO, or output-commit failure |
| `130` | Cancelled by the user |

## Current limitations

- Localization is manual: provide a box or mask.
- Telea and Navier-Stokes are model-free baselines and may produce visible artifacts on textured,
  structured, or semantically complex backgrounds.
- Batch execution uses exactly one worker.
- Batch resume and automatic retry are not supported.
- PNG preserves supported alpha data. JPEG output is lossy and cannot preserve alpha.
