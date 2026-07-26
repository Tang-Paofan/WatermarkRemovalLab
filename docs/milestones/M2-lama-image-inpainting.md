# M2 Specification: LaMa Image Inpainting

[English](M2-lama-image-inpainting.md) | [简体中文](M2-lama-image-inpainting.zh-CN.md)

This document is the implementation and acceptance contract for M2. It adds an optional LaMa
ONNX inpainter while preserving the M1 image, mask, output-safety, and adapter boundaries.

The initial model artifact is conditionally approved for non-bundled research integration. Model
redistribution and commercial-use conclusions remain outside that approval. See
[MODEL_LICENSES.md](../../MODEL_LICENSES.md) and
[ADR 0002](../adr/0002-m2-lama-onnx-runtime.md). The local/AutoDL validation split and portable
reproduction procedure are defined in the
[model execution guide](../MODEL_EXECUTION.md).

## 1. Goals

M2 must:

- add a model-based inpainter behind the same public single-image application service as M1;
- derive a padded local crop from the final refined mask;
- adapt that crop deterministically to the reviewed fixed-shape LaMa ONNX contract;
- composite generated pixels only where the original-resolution final mask is true;
- preserve input immutability, alpha, atomic output, and overwrite behavior;
- keep CPU and CUDA providers explicit and report the provider actually used;
- install and cache model artifacts outside Git with mandatory SHA-256 verification;
- keep OpenCV Telea and Navier-Stokes usable without model dependencies;
- keep ordinary local development complete without installing or running the real model;
- compare LaMa against the M1 baselines on reproducible synthetic or authorized inputs;
- introduce the M2 portion of B2: verified resume and provider-aware bounded scheduling.

## 2. Non-goals

M2 does not include:

- automatic watermark detection;
- SAM/SAM 2 prompting or segmentation;
- arbitrary third-party LaMa files or a general model registry;
- training, fine-tuning, or converting LaMa weights inside the normal application;
- tiling or a claim of native arbitrary-resolution ONNX inference;
- remote inference, a workstation-to-AutoDL upload API, or any other hosted model service;
- desktop, web, or API adapters;
- video processing or temporal consistency;
- automatic quality claims that LaMa is better for every image;
- bundling model weights in the repository, wheel, source archive, or release asset;
- approving model-weight redistribution or commercial use where upstream evidence is incomplete.

## 3. Architecture slice

```text
CLI / B2 batch adapter
          ↓
Single-image removal service
          ↓
Final mask + local crop planner
          ↓
Inpainter interface ────────────────┐
    ↓                               ↓
OpenCV adapter              LaMa ONNX adapter
                                    ↓
                       Model store + session owner
                                    ↓
                         ONNX Runtime provider
          ↓
Original-resolution mask-only composite
          ↓
Atomic output
```

The LaMa adapter must not perform file discovery, output encoding, CLI rendering, batch state
transitions, or network downloads. The model store owns installation and integrity checks. The
application service owns the session lifecycle and passes an already validated model path to the
adapter.

### Execution environment split

The current maintainer workstation is not a real-model execution target. This is an environment
constraint, not a core architecture dependency.

- The local workstation and ordinary CI run all default offline tests, OpenCV integration,
  crop-transform tests, CLI tests, and fake ONNX session tests.
- AutoDL is the initial reference host for the pinned model, CPU real-model smoke test, CUDA
  provider test, and resource benchmarks.
- Any compatible Linux host may replace AutoDL when it uses the same Git commit, lockfile, model
  SHA-256, normalized configuration, and evidence fields.
- The project CLI runs on the compute host beside the model and authorized media. M2 does not
  turn AutoDL into an inference server.

The full procedure is in [MODEL_EXECUTION.md](../MODEL_EXECUTION.md).

## 4. Preserved data contract

M1 invariants remain mandatory:

- decoded image: `uint8`, `H × W × 3`, RGB;
- optional alpha: separate `uint8`, `H × W`;
- final mask: `bool`, `H × W`;
- boxes: `(x_min, y_min, x_max, y_max)`, right/bottom exclusive;
- serialized masks: 0 background, 255 selected;
- inputs are never mutated;
- alpha is byte-identical unless an explicit future operation edits alpha;
- before lossy encoding, pixels outside the original-resolution final mask are exactly unchanged.

The crop planner and model adapter return new arrays. Model padding, resizing, and generated
pixels outside the model-space mask never expand the final compositing region.

## 5. Initial reviewed artifact

The first implementation targets one exact artifact:

| Field | Required value |
|---|---|
| Backend ID | `lama-onnx-fp32` |
| Repository | `Carve/LaMa-ONNX` |
| Artifact | `lama_fp32.onnx` |
| Artifact commit | `a3ee2fca54baebec351b8fa7786154ffa7555aa6` |
| Model-card commit | `c3c0c9e468934d62e79c329e35d82dd09ff8c444` |
| Size | `208044816` bytes |
| SHA-256 | `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6` |
| ONNX opset | 17 |
| Spatial shape | fixed `512 × 512` |

The model card identifies this file as the recommended export of the original PyTorch
`big-lama` model. The linked export notebook downloads `smartywu/big-lama`, applies the
`Carve-Photos/lama` Fourier export changes, and uses `torch.onnx.export`.

This evidence is sufficient for a pinned, hash-verified research integration, but it is not a
complete reproducible conversion chain: the notebook clones the exporter branch without a Git
revision and does not pin the source checkpoint SHA-256. The model repository declares
Apache-2.0 in model-card metadata, while the official Places2 download terms restrict the
training images to non-commercial research and educational use. These limitations must remain
visible in the model record and installer notice.

`lama.onnx` is not an M2 artifact. Its own model card marks it not recommended, assigns opset 18,
and reports lower performance.

## 6. Model installation and cache

Image processing never downloads a model implicitly.

The headless CLI will expose an explicit equivalent of:

```text
wrl model install lama-onnx-fp32 --accept-model-terms [--cache-dir DIR]
wrl model status lama-onnx-fp32 [--cache-dir DIR] [--json]
```

Installation rules:

- show the exact source, declared license, dataset restriction note, size, and SHA-256;
- require explicit acceptance in non-interactive use;
- download from the artifact commit, never an unpinned `main` URL;
- write to a temporary file inside the cache directory;
- calculate SHA-256 while downloading or immediately afterward;
- atomically publish the final cache file only after the hash and size match;
- remove an invalid temporary artifact where safe;
- never treat filename or download success as an integrity check;
- never write weights inside the repository by default;
- processing in offline mode succeeds when the verified artifact already exists;
- a missing artifact produces an actionable error and never starts a download.

The model store owns the cache path and descriptor. A user-supplied path is accepted only when
its contents match the reviewed size and SHA-256. Support for arbitrary descriptors is deferred.
On AutoDL, the reference cache is outside the Git checkout under
`/root/autodl-tmp/wrl-models`; other hosts use an equivalent user-controlled cache.

The first local implementation slice provides exactly one descriptor,
`lama-onnx-fp32`, and a headless store with these boundaries:

- cache-root precedence is an adapter-provided path, `WRL_MODEL_CACHE`, then the platform user
  cache;
- the final path is `<cache-root>/lama-onnx-fp32/lama_fp32.onnx`;
- inspection is read-only and never downloads;
- installation requires the acceptance value to be the Boolean `true`;
- a verified final file is reused without transport;
- an invalid final file remains untouched unless a newly downloaded temporary file passes both
  checks and is atomically published;
- the downloader is injectable so default tests remain offline.

The fifth local implementation slice exposes these commands through a model-management
application service. `status` remains read-only, while `install` displays the reviewed notice,
requires explicit acceptance, and delegates to the atomic verified store. Processing commands
still never download a model.

## 7. Runtime dependencies and providers

ONNX Runtime remains optional and is imported lazily.

- CPU uses the `lama-onnx-cpu` optional dependency group.
- CUDA uses the `lama-onnx-cuda` optional dependency group.
- The CPU and CUDA groups are mutually exclusive in one environment.
- Installing neither group leaves all OpenCV commands operational.
- `cpu` maps to `CPUExecutionProvider`.
- `cuda` requires `CUDAExecutionProvider` to be present before creating the session.
- A CUDA session may list CPU after CUDA for unsupported nodes, but a missing CUDA provider is
  an error rather than a silent full-CPU fallback.
- M2 defaults to `cpu`; automatic provider selection is outside the initial contract.
- The normalized result records requested and effective providers.

The session owner:

- verifies the artifact once before creating a session;
- validates model input/output names, types, and shapes;
- keys sessions by artifact SHA-256, provider, device, and relevant provider options;
- gives caches an explicit process-scoped owner and bounded lifetime;
- does not use an import-time singleton or unbounded global mutable cache;
- converts provider initialization and memory failures into stable domain errors.

The second local implementation slice locks ONNX Runtime 1.26.0 in two conflicting extras.
`lama-onnx-cpu` selects `onnxruntime`; `lama-onnx-cuda` selects `onnxruntime-gpu` on Linux or
Windows. Version 1.26 is the official CUDA 12.8 / cuDNN 9 GPU line. Moving to the CUDA 13 default
introduced in 1.27 requires a new host-compatibility review.

This slice also provides a lazy `LamaOnnxSessionOwner`. It checks the artifact before importing
`onnxruntime`, requires the requested provider to be registered and first in the created session,
validates the reviewed input/output metadata, retains at most one session, and drops its reference
when closed. CPU and CUDA use the same graph contract. No runtime is installed by default and all
default tests use injected fake modules and sessions.

The third local implementation slice adds the model-independent LaMa inference core. It plans the
local crop from the original-resolution final mask, prepares immutable input arrays, invokes an
injected `session.run`, validates and restores its output, and composites only inside the final
mask. Fake-session tests cover the full loop without importing ONNX Runtime or reading a model
file.

The fifth local implementation slice connects this core to the public single-image service. The
service owns one lazy session owner for a non-empty LaMa request, closes it before returning, and
records runtime diagnostics and the crop plan in its result. The CLI only constructs the request
and invokes that service; it does not create a runtime session.

Execution code for a verified real artifact is now present. The optional pinned-artifact tests,
AutoDL CPU/CUDA evidence, and quality/resource benchmarks remain later slices.

## 8. Local crop contract

Given the original image and final refined mask:

1. An empty mask bypasses crop planning and inference and returns an unchanged copy.
2. Calculate the minimal half-open bounding box containing every selected mask pixel.
3. Expand the box by `crop_padding` pixels on all sides. The initial default is 64 and the value
   must be a non-negative integer.
4. Convert the expanded region into a square context window centered on the expanded box. The
   side length is the larger expanded dimension.
5. When the square extends outside the image, synthesize RGB context with reflection padding.
   Degenerate one-pixel RGB axes use edge replication. Mask context outside the image is always
   false.
6. Resize the square isotropically to `512 × 512`. RGB downscaling uses area interpolation and RGB
   upscaling uses cubic interpolation. Masks always use nearest-neighbor interpolation and are
   re-binarized.
7. Run inference with batch size one.
8. Invert the resize and padding transform to produce an original-crop-sized candidate.
9. Composite the candidate only where the original-resolution final mask is true.

This transform never stretches width and height by different factors. It records the source box,
square window, padding, scale, and warnings so the inverse mapping is testable.

If the context side exceeds 512, the input is downscaled and the result includes a
`crop_downscaled` warning. Full-frame masks are valid but warned and may have poor quality.
Tiling and overlapping-window fusion are deferred until a measured need justifies them.

## 9. Expected tensor contract

The loader must validate the exact artifact before inference:

| Tensor | Name | Type | Shape | Range |
|---|---|---|---|---|
| Image input | `image` | `float32` | `1 × 3 × 512 × 512` | RGB normalized to `[0, 1]` |
| Mask input | `mask` | `float32` | `1 × 1 × 512 × 512` | binary `0` or `1`; `1` is the hole |
| Output | `output` | `float32` | `1 × 3 × 512 × 512` | expected `[0, 255]` |

Although the export declares a dynamic batch axis, M2 submits batch size one. Spatial dimensions
are fixed. Output values are checked for finite values, clipped to `[0, 255]`, rounded with
round-half-up (`floor(value + 0.5)` after clipping), and converted to `uint8`.

Any name, type, rank, or spatial-shape mismatch fails with `model_contract_mismatch` before
processing user media.

## 10. CLI extension

The single-image command extends the existing shape:

```text
wrl image remove INPUT OUTPUT (--box X,Y,W,H | --mask MASK)
    --method lama
    [--provider cpu|cuda]
    [--crop-padding PIXELS]
    [--model-dir DIR]
    [existing mask, output, overwrite, and JSON options]
```

Rules:

- `--provider` defaults to `cpu` for LaMa;
- `--crop-padding` defaults to 64 and is LaMa-specific;
- OpenCV `--radius` does not change LaMa behavior and an explicitly incompatible option fails
  instead of being silently ignored;
- `--dilate` remains a mask-refinement option shared by all inpainters;
- JSON results include backend ID, artifact SHA-256, crop transform, requested/effective
  providers, duration, warnings, and stable errors;
- human diagnostics remain on stderr when JSON is enabled.

The CLI continues to call the public single-image service. It never creates an ONNX Runtime
session directly.

The fifth local implementation slice provides these options and rejects explicit backend
mismatches. An empty final mask remains a successful no-op and does not initialize a model
session. B1 directory and manifest batches continue to accept only OpenCV methods until the B2
fingerprint, resume, and scheduling contract is implemented.

## 11. Stable error contract

M2 adds at least these error codes:

| Code | Meaning |
|---|---|
| `model_not_installed` | The reviewed artifact is absent from the selected cache |
| `model_integrity_failed` | Size or SHA-256 does not match the descriptor |
| `model_terms_not_accepted` | An install was requested without explicit acceptance |
| `runtime_not_installed` | The selected optional ONNX Runtime package is unavailable |
| `provider_unavailable` | The requested execution provider is not registered |
| `model_contract_mismatch` | Model metadata differs from the reviewed tensor contract |
| `session_creation_failed` | ONNX Runtime could not create the requested session |
| `inference_failed` | The provider failed while running a valid request |
| `insufficient_memory` | A classified host or device out-of-memory failure occurred |
| `crop_transform_failed` | Crop or inverse mapping could not preserve the contract |

Errors retain their original exceptions for debugging, provide an actionable public message, and
do not leave a final output or partial model file.

## 12. B2 behavior introduced with M2

M2 begins B2 without duplicating the single-image pipeline:

- manifests may select `method: "lama"`, `provider`, and `crop_padding`;
- planning checks the model descriptor and provider availability before expensive work;
- successful item records include input, mask, normalized configuration, software, model
  artifact, and output fingerprints;
- resume reuses an output only when every identity in
  [BATCH_PROCESSING.md](../BATCH_PROCESSING.md) matches;
- changing model SHA-256, provider, crop transform, mask, or application version marks the item
  stale;
- GPU defaults to one active inference item per device;
- CPU concurrency is bounded by an explicit worker and memory budget;
- one validated session may be reused only under its documented thread-safety policy;
- queue depth is bounded and scheduling never silently reduces crop quality;
- deterministic model, integrity, contract, and validation failures are never auto-retried.

Batch execution performs no model download. Installation is a separate preflight action.

## 13. Test matrix

### Default offline tests

- empty, one-pixel, edge-touching, non-square, and full-frame masks;
- half-open bounding-box derivation and configurable padding;
- square-window construction and reflection/edge padding;
- forward and inverse coordinate mapping;
- RGB interpolation versus nearest-neighbor mask interpolation;
- re-binarization after model-space transformation;
- exact mask-exterior pixel preservation and byte-identical alpha;
- immutable image and mask inputs;
- fake-session input normalization, names, dtypes, and shapes;
- non-finite and out-of-range output handling;
- missing runtime, provider, model, and integrity failures;
- atomic model installation using a local fake transport;
- cache keys, session ownership, and bounded reuse;
- CLI option compatibility and non-zero failure exit codes;
- B2 fingerprints, stale classification, resume, and provider scheduling.

Default tests use fakes or tiny synthetic fixtures. They do not import a heavy runtime unless its
optional test environment is selected, access the network, or download a model.

### Optional real-artifact tests

An explicitly marked suite must:

- use the pinned, hash-verified artifact;
- validate ONNX metadata against the expected tensor contract;
- run CPU inference on a synthetic RGB image and binary mask;
- verify finite `uint8` output and exact exterior compositing;
- run CUDA only when the optional environment declares compatible hardware and runtime;
- report the runtime and provider versions used.

The `model` pytest marker selects pinned real-artifact tests. The `gpu` marker additionally
requires a compatible GPU provider. These markers are registered only when their implementation
slice is added and are never part of the default offline suite.

The reference run occurs on AutoDL, but the tests must not import AutoDL APIs or depend on its
filesystem outside an adapter-provided cache path.

The pinned artifact must complete one real CPU smoke test before M2 acceptance.

## 14. Benchmark and review evidence

Compare OpenCV Telea, OpenCV Navier-Stokes, and LaMa with identical decoded inputs and final masks.

The M2 benchmark records:

- synthetic or authorized source and clean reference provenance;
- mask category and selected-pixel count;
- crop size, scale, and downscale warning;
- backend, artifact SHA-256, runtime version, provider, and hardware;
- wall-clock latency after a separately reported warm-up;
- peak RAM and, where applicable, VRAM;
- masked-region error against a clean synthetic reference;
- exact exterior-pixel-change count before encoding;
- failures and excluded cases.

Report distributions and representative failures, not only a mean or a curated success. The
benchmark may show that LaMa is better on some textured regions and worse on others; M2 does not
set a universal superiority claim as an exit gate.

## 15. Acceptance checklist

M2 is complete only when:

- [ ] the pinned artifact and conditional-use limitations remain recorded in
      `MODEL_LICENSES.md`;
- [ ] model installation is explicit, atomic, commit-pinned, and SHA-256 verified;
- [ ] OpenCV commands work in an environment with no model optional dependencies;
- [ ] CPU and CUDA requests have explicit provider behavior without silent fallback;
- [ ] local crop and inverse transforms cover edge and full-frame cases;
- [ ] the real artifact passes its tensor-contract and CPU smoke tests;
- [ ] final-mask exterior pixels and alpha preserve the M1 invariants;
- [ ] failures never publish partial outputs or unverified model files;
- [ ] B2 resume includes the exact model and transform fingerprints;
- [ ] provider-aware concurrency and memory limits are tested;
- [ ] OpenCV/LaMa comparison evidence records quality and resource limitations;
- [ ] AutoDL or an equivalent host produces a reviewed environment record following
      `MODEL_EXECUTION.md`;
- [ ] both language versions of affected documentation are updated;
- [ ] redistribution remains disabled unless a later license review explicitly approves it.

## 16. Suggested implementation slices

1. model descriptor, cache paths, atomic installer, and integrity tests;
2. optional ONNX Runtime dependency groups and provider diagnostics;
3. crop-transform data contract and pure unit tests;
4. LaMa ONNX adapter with a fake-session boundary;
5. single-image service and CLI extension;
6. optional pinned-artifact CPU smoke test;
7. B2 fingerprints, verified resume, and bounded provider scheduling;
8. benchmarks, acceptance evidence, and documentation review.

Each slice must remain independently reviewable and use the repository commit-message rules.
