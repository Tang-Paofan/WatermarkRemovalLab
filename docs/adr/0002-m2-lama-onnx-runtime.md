# ADR 0002: Pinned LaMa ONNX Runtime for M2

[English](0002-m2-lama-onnx-runtime.md) |
[简体中文](0002-m2-lama-onnx-runtime.zh-CN.md)

- Status: Accepted for non-bundled research integration
- Date: 2026-07-24

## Context

M1 established OpenCV Telea and Navier-Stokes as model-free baselines. M2 needs a materially
stronger inpainting backend without making PyTorch, a model download, or a GPU runtime mandatory
for the core package.

The initial candidate is `Carve/LaMa-ONNX` `lama_fp32.onnx`. Its model card identifies it as the
recommended opset-17 export of `big-lama`, with fixed 512-by-512 spatial inputs. Hugging Face
publishes an exact artifact size and SHA-256.

The audit also found limits:

- the exporter notebook does not pin the exporter Git revision;
- it does not record the source `big-lama.zip` SHA-256;
- the model card declares Apache-2.0, but the official Places2 image-download terms restrict the
  training data to non-commercial research and educational use;
- the project has no need to redistribute the weight to implement a local learning platform.

The current maintainer workstation cannot host the real model runtime. That limitation must not
make a hosted API part of M2 or make default development depend on a paid GPU. A separate compute
host can run the same headless CLI without changing the application boundary.

## Decision

M2 will first integrate one pinned LaMa ONNX FP32 artifact through ONNX Runtime.

- The exact descriptor, artifact commit, byte size, and SHA-256 live in
  [MODEL_LICENSES.md](../../MODEL_LICENSES.md).
- The model is optional, never committed or bundled, and installed only through an explicit
  user action.
- The installer shows the declared model license and dataset restriction note before download.
- Processing never downloads a missing model.
- The loader accepts only the reviewed artifact hash during M2.
- CPU and CUDA ONNX Runtime distributions are separate optional dependency groups.
- Both initial groups lock ONNX Runtime 1.26.0. Its GPU distribution targets CUDA 12.8 and
  cuDNN 9; the CUDA 13 default introduced in 1.27 requires a separate compatibility review.
- CPU is the default provider; requesting CUDA without a registered CUDA provider fails.
- A deterministic padded square crop adapts original media to the fixed 512-by-512 model input.
- Only pixels selected by the original-resolution final mask are composited back.
- The model and runtime session have explicit application-layer owners and bounded caches.
- OpenCV remains fully operational when model dependencies are absent.
- Local workstations and ordinary CI remain model-free; AutoDL is the initial reference host for
  real-model and CUDA acceptance.
- AutoDL is interchangeable with another compatible Linux host and is not imported, called, or
  represented in core contracts.
- The CLI, model, and authorized media run together on the compute host. M2 does not add a
  workstation-to-host upload or inference API.

The artifact is approved for non-bundled research integration and execution-host-local
evaluation. This decision does not approve redistribution of the weight or make a commercial-use
determination. Those actions require a new review.

## Consequences

Positive:

- M2 can test LaMa without introducing the original PyTorch training stack;
- a pinned commit and SHA-256 make user downloads reproducible and tamper-evident;
- optional runtime groups preserve the lightweight OpenCV baseline;
- the crop and provider contracts are reusable by CLI, batch, desktop, and later API adapters;
- a fixed artifact makes benchmarks and B2 resume fingerprints comparable.
- contributors can reproduce acceptance on AutoDL or another compatible host without changing
  the core architecture.

Costs and limitations:

- every crop must be transformed to and from a fixed 512-by-512 space;
- large masks may lose detail when downscaled;
- CPU and CUDA packaging need separate verification;
- the conversion cannot currently be reproduced from a fully pinned source chain;
- model license metadata and training-data terms must both remain visible;
- releases cannot bundle the weight under this decision.
- default and real-model validation results come from different declared environments and must be
  joined through an exact repository commit and evidence record.

## Alternatives considered

### Integrate original PyTorch LaMa first

Rejected for M2 because it introduces a larger dependency and runtime surface before the CLI
model boundary, crop behavior, and benchmarks are proven. It remains a future comparison option.

### Re-export the model in this repository

Deferred because a trustworthy derived artifact would require pinning the original checkpoint,
exporter code, toolchain, conversion procedure, equivalence tests, and a new artifact hash and
license record. M2 does not need to publish a converted weight.

### Accept arbitrary ONNX paths

Rejected for M2 because names, shapes, normalization, output range, licenses, and integrity would
be unbounded. A future descriptor format may support reviewed additional artifacts.

### Bundle the reviewed ONNX file

Rejected because the repository is intended to remain small and because redistribution and
training-data implications are not approved by this review.

## Revisit criteria

Revisit this decision when:

- an upstream release supplies a more complete pinned provenance chain;
- a different reviewed artifact materially improves quality or runtime behavior;
- dynamic spatial shapes or tiling become an accepted requirement;
- model redistribution, commercial packaging, or a hosted service is proposed;
- ONNX Runtime provider packaging changes materially;
- benchmarks show the fixed-shape transform is not fit for the intended images.
