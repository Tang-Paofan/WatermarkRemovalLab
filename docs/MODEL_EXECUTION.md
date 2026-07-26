# Model Execution and Reproduction Environments

[English](MODEL_EXECUTION.md) | [简体中文](MODEL_EXECUTION.zh-CN.md)

This guide explains how Watermark Removal Lab separates ordinary local development from real
model execution, and how another contributor can reproduce M2 LaMa validation on AutoDL or an
equivalent compatible host.

It complements the
[M2 specification](milestones/M2-lama-image-inpainting.md) and
[ADR 0002](adr/0002-m2-lama-onnx-runtime.md). It does not define a hosted inference service.

## 1. Why the environments are split

The current maintainer workstation is not a real-model execution target. It remains the required
environment for the headless core, OpenCV baselines, CLI behavior, crop transformations, fake
runtime tests, and all default offline checks.

The pinned LaMa artifact and CUDA provider are validated on a separate Linux compute host. AutoDL
is the project's initial reference host because it can provide an NVIDIA GPU and persistent
workspace storage. AutoDL is not a product dependency: any host that satisfies the same recorded
software, model, provider, and integrity contracts may reproduce the result.

This split must never weaken the default test baseline. Contributors without a GPU or model file
must still be able to develop and verify the non-model package.

## 2. Environment responsibilities

| Environment | Required work | Must not be required for |
|---|---|---|
| Local workstation or ordinary CI | formatting, linting, types, unit/integration tests, OpenCV, fake model sessions, packaging | model downloads, ONNX Runtime GPU, CUDA, network access |
| AutoDL reference instance | pinned model installation, CPU model smoke test, CUDA provider smoke test, real inference, RAM/VRAM and latency evidence | source-of-truth code changes or permanent artifact storage |
| Other compatible Linux GPU host | reproduce the same descriptor, provider, tests, and evidence fields | AutoDL-specific APIs or filesystem paths |

The model and authorized test media are local to the compute host while an experiment runs. The
developer invokes the ordinary project CLI on that host. A workstation-to-AutoDL image-upload API
is not part of M2; remote service transport belongs to M6.

## 3. Reference host requirements

Use a Linux host with:

- an NVIDIA GPU when validating `CUDAExecutionProvider`;
- a driver compatible with the ONNX Runtime GPU version locked by the repository;
- a repository-supported Python version;
- Git and the repository-supported `uv` version;
- enough disk for the environment, the 208044816-byte model, authorized fixtures, and outputs;
- outbound access to the pinned Hugging Face artifact only during explicit installation;
- a writable cache outside the Git checkout.

Do not choose an environment by the GPU name alone. ONNX Runtime requires compatible CUDA and
cuDNN libraries. Also, `nvidia-smi` reports the highest CUDA version supported by the driver, not
the CUDA toolkit actually installed in the environment. Record both the driver information and
the runtime provider that ONNX Runtime registers.

Use official release packages locked by the repository. Do not substitute a nightly build or
manually install an unrecorded latest package when producing acceptance evidence.

References:

- [ONNX Runtime installation matrix](https://onnxruntime.ai/docs/install/)
- [ONNX Runtime CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [AutoDL CUDA/cuDNN explanation](https://www.autodl.com/docs/cuda/)
- [AutoDL base image guidance](https://www.autodl.com/docs/base_config/)

## 4. AutoDL storage and session rules

The reference layout is:

```text
/root/autodl-tmp/
├── wrl-models/       Verified model cache root
│   └── lama-onnx-fp32/
│       └── lama_fp32.onnx
├── wrl-work/         Authorized or synthetic inputs and temporary outputs
└── wrl-evidence/     Logs and machine-readable acceptance evidence

<system or chosen project path>/
└── WatermarkRemovalLab/   Git checkout; no weights or user media committed
```

AutoDL documents `/root/autodl-tmp` as its instance data-disk path. The platform also warns that
local system and data disks have no redundant copy, and that released instances lose their data.
The model can be downloaded again from its pinned source, but reviewed evidence and any valuable
authorized inputs must be backed up separately.

Rules:

- never store Git credentials, SSH private keys, access tokens, or service passwords in the
  repository or evidence directory;
- do not commit the model, input media, generated outputs, caches, or environment files;
- use only synthetic, self-created, or explicitly authorized media;
- back up small reviewed evidence before releasing or resetting the instance;
- delete sensitive authorized media after the review when retention is not required;
- use `tmux` or `screen` for a long SSH-run command and save logs to a file;
- stop the paid GPU instance after checks complete.

References:

- [AutoDL instance data retention](https://www.autodl.com/docs/instance_data/)
- [AutoDL local data disk warning](https://www.autodl.com/docs/local_disk/)
- [AutoDL SSH guidance](https://www.autodl.com/docs/ssh/)

## 5. Bootstrap the repository

Clone the same repository and checkout the exact commit being reviewed:

```bash
git clone https://github.com/Tang-Paofan/WatermarkRemovalLab.git
cd WatermarkRemovalLab
git checkout <REVIEWED_COMMIT>
git status --short
```

Install the default development environment and prove that the ordinary baseline passes before
adding a model runtime:

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build --no-sources
```

These commands must remain model-free and offline after dependency synchronization.

## 6. Install the optional runtime and model

### Current implementation boundary

The repository currently contains the fixed `lama-onnx-fp32` descriptor, atomic model store,
mutually exclusive CPU/CUDA runtime extras, a lazy session owner, and the model-independent LaMa
crop/inference core. Offline tests prove that model integrity is checked before importing the
optional runtime, CPU and CUDA provider policies share one tensor contract, unavailable CUDA
never silently falls back to CPU, graph metadata is validated, and one owner retains at most one
session.

The local inference core now plans a padded square crop from the final mask, prepares immutable
`float32` NCHW image and mask tensors, calls an injected `session.run`, validates the fixed output,
inverts the crop transform, and replaces only original-resolution final-mask pixels. RGB context
uses reflection padding, with edge replication for a one-pixel axis; mask context outside the
image is always false. RGB downscaling uses area interpolation, RGB upscaling uses cubic
interpolation, and masks use nearest-neighbor interpolation. Output conversion clips to
`[0, 255]` and uses round-half-up. Default tests exercise this loop with fake sessions and do not
install ONNX Runtime or read a model file.

The `wrl model` commands, application/CLI integration, execution through a real ONNX Runtime
session, and real-model pytest markers remain later M2 slices. Until those slices form the minimum
user-facing inference loop, do not start real-model acceptance on AutoDL and do not treat the
command examples below as available. Another contributor can review the implemented slices with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build --no-sources
```

The model cache root is selected in this order: an adapter-provided `--cache-dir` or `Path`, the
`WRL_MODEL_CACHE` environment variable, then the platform user cache. Artifacts are namespaced as
`<cache-root>/<model-id>/<filename>`. Inspection is read-only and never creates a directory or
starts a download.

M2 defines separate optional groups:

```bash
# CPU-only real-model validation
uv sync --extra lama-onnx-cpu

# CUDA validation; do not install both runtime extras in one environment
uv sync --extra lama-onnx-cuda
```

Both extras lock ONNX Runtime 1.26.0. The GPU package is the official CUDA 12.8 / cuDNN 9 build;
ONNX Runtime 1.27 and later changed the default PyPI GPU build to CUDA 13, so upgrading requires a
new compatibility review and AutoDL environment record. The project declares the extras as
conflicting because the CPU and GPU distributions expose the same `onnxruntime` import package.
Use a separate environment for each acceptance run.

Use only an extra that exists in the checked-out revision and is documented by `uv` and project
metadata. A CLI command described by this contract may still be unavailable during staged M2
implementation; `wrl --help` is authoritative for that revision.

Install the model explicitly into the data-disk cache:

```bash
uv run wrl model install lama-onnx-fp32 \
  --accept-model-terms \
  --cache-dir /root/autodl-tmp/wrl-models

uv run wrl model status lama-onnx-fp32 \
  --cache-dir /root/autodl-tmp/wrl-models \
  --json
```

The installer must download the artifact-commit URL, verify its 208044816-byte size and expected
SHA-256, and publish it atomically. Do not manually rename an unverified download into the cache.

Expected SHA-256:

```text
1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6
```

Processing commands never download a missing model.

## 7. Capture the environment before inference

Save output from equivalent commands:

```bash
git rev-parse HEAD
uv run python --version
uv --version
nvidia-smi

uv run python -c \
  "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

For CUDA acceptance, `CUDAExecutionProvider` must appear in the ONNX Runtime list before creating
the model session. A CUDA request that reports only `CPUExecutionProvider` is a failed CUDA
environment, not a successful fallback.

The evidence must distinguish:

- selected GPU model and driver;
- Python and ONNX Runtime versions;
- requested provider;
- registered and effective providers;
- model descriptor ID, byte size, and SHA-256;
- repository commit and clean/dirty state.

Do not include an AutoDL instance ID, IP address, SSH command, token, username, or private absolute
media path in committed evidence.

## 8. Run real-artifact validation

M2 reserves two explicit pytest markers:

```bash
# Pinned real model; CPU is sufficient
uv run pytest -m model

# Pinned real model with a compatible CUDA provider and GPU
uv run pytest -m "model and gpu"
```

Only run markers declared by the checked-out `pyproject.toml`. Default `uv run pytest` must
continue to exclude network, model, and GPU requirements.

The real-model sequence is:

1. verify the model status and hash;
2. validate model tensor names, dtypes, and fixed 512-by-512 shapes;
3. run a synthetic CPU smoke test;
4. verify original-resolution mask-exterior pixels and alpha;
5. run the same synthetic request through CUDA;
6. confirm the requested and effective provider records;
7. run authorized manual CLI cases;
8. compare Telea, Navier-Stokes, and LaMa with identical decoded inputs and masks;
9. save latency, peak RAM, peak VRAM, crop transform, warnings, and failures.

CPU and CUDA outputs may have small backend-dependent numeric differences. Tests must use
documented tolerances for model pixels while preserving the exact exterior-compositing invariant.

## 9. Run the CLI on the compute host

The intended execution shape is:

```bash
uv run wrl image remove INPUT.png OUTPUT.png \
  --mask MASK.png \
  --method lama \
  --provider cuda \
  --crop-padding 64 \
  --model-dir /root/autodl-tmp/wrl-models \
  --overwrite error \
  --json
```

All three paths are paths on the compute host. M2 does not send them over an HTTP API.

For comparison, run the same decoded input and final mask with `telea` and `ns`. Use PNG when
checking exact pixel invariants so JPEG encoding does not contaminate the result.

## 10. Evidence record

Keep raw logs outside Git. A small reviewed acceptance record may later be committed under
`docs/acceptance/` in both languages. It must include at least:

```json
{
  "schema_version": 1,
  "repository_commit": "<FULL_GIT_SHA>",
  "worktree_clean": true,
  "environment": {
    "kind": "autodl",
    "os": "<OS_VERSION>",
    "python": "<PYTHON_VERSION>",
    "gpu": "<GPU_MODEL>",
    "driver": "<DRIVER_VERSION>",
    "onnxruntime": "<ORT_VERSION>"
  },
  "provider": {
    "requested": "cuda",
    "registered": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "effective": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "model": {
    "id": "lama-onnx-fp32",
    "size_bytes": 208044816,
    "sha256": "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
  },
  "checks": {
    "default_suite": "<PASS_OR_FAIL>",
    "model_cpu": "<PASS_OR_FAIL>",
    "model_cuda": "<PASS_OR_FAIL>"
  }
}
```

Add benchmark sample counts, warm-up policy, input dimensions, mask categories, latency
distribution, RAM/VRAM peaks, exterior-change counts, and failures when claiming performance or
quality results.

## 11. Portability

To reproduce outside AutoDL:

1. keep the same Git commit and lockfile;
2. install exactly one reviewed ONNX Runtime optional group;
3. use the same model artifact SHA-256;
4. preserve the crop and CLI configuration;
5. run the same default, model, and provider-specific tests;
6. capture the same evidence fields;
7. explain any operating-system, driver, provider, or hardware difference.

Comparable evidence depends on identities and recorded configuration, not on the AutoDL brand.
An environment that cannot report the exact model hash or effective provider has not reproduced
the M2 result.
