# Engineering Roadmap

[English](ROADMAP.md) | [简体中文](ROADMAP.zh-CN.md)

This document records the planned engineering sequence for Watermark Removal Lab. It is not a release schedule or a statement that a milestone is complete. Public README files intentionally omit roadmap and development-status sections.

## Delivery principles

- Complete milestones in order unless an explicit decision records why work moves ahead.
- Keep localization, mask refinement, inpainting, evaluation, and presentation replaceable.
- Build the headless core and CLI before a desktop application.
- Use OpenCV as the required model-free baseline before adding model runtimes.
- Add model and data licenses before integrating or distributing exact artifacts.
- Treat batch processing as orchestration over the same single-item pipelines, never as a second algorithm implementation.
- Require synthetic, self-created, or explicitly licensed fixtures and examples.

## Capability tracks

The roadmap has eight primary milestones plus cross-cutting batch-processing and desktop tracks:

| Track | Purpose |
|---|---|
| M1–M6 | Image localization, inpainting, evaluation, and deployment |
| M7–M8 | Baseline and temporally consistent video processing |
| B1–B4 | Batch orchestration that incrementally reuses milestone pipelines |
| D1 | Desktop adapter after the M1/B1 CLI contracts are validated |

Batch processing does not become a separate M9 because it must evolve with each media pipeline:

- **B1, aligned with M1:** deterministic sequential image batches;
- **B2, aligned with M2–M4:** resumable and resource-aware model batches;
- **B3, aligned with M5–M6:** aggregate evaluation and queued service jobs;
- **B4, aligned with M7–M8:** long-running video and mixed-media batches.

The detailed batch contract is defined in [BATCH_PROCESSING.md](BATCH_PROCESSING.md).

The desktop track begins only after the M1 single-image and B1 batch CLI exit gates pass. Qt, Electron, or another framework is selected through a focused prototype and a new ADR, not by changing the headless core.

## M1: OpenCV image baseline

### Objective

Deliver a model-free image-removal CLI using a user-provided box or binary mask and OpenCV Telea/Navier-Stokes inpainting.

### Required scope

- image IO with explicit RGB/BGR handling;
- separate alpha handling;
- box-to-mask and external-mask input;
- mask validation and configurable dilation;
- Telea and Navier-Stokes backends;
- mask-only compositing;
- safe output behavior and structured failures;
- deterministic sequential image batch processing through B1;
- focused unit and CLI integration tests.

### Exit gate

- single-image and B1 batch commands satisfy [M1 specification](milestones/M1-opencv-image-baseline.md);
- pixels outside the final refined mask are unchanged before lossy encoding;
- alpha is preserved;
- invalid boxes, masks, collisions, and output failures are tested;
- no network access or model download is required;
- documentation contains only authorized or synthetic examples.

## M2: LaMa image inpainting

### Objective

Add a model-based image inpainter while preserving the M1 request/result and mask contracts.

### Required scope

- derive the final mask bounding box;
- expand a configurable padded local crop;
- run a reviewed PyTorch or ONNX LaMa artifact;
- composite only inside the final mask;
- support explicit CPU/GPU runtime providers and model caching;
- compare OpenCV and LaMa quality, latency, RAM, and VRAM;
- extend B2 with provider-aware scheduling and resumable model jobs.

### Exit gate

- implementation satisfies the
  [M2 specification](milestones/M2-lama-image-inpainting.md);
- exact model and weight licenses are recorded in `MODEL_LICENSES.md`;
- model download and integrity verification are explicit;
- missing providers and memory failures have actionable errors;
- crop and full-image edge cases are tested;
- OpenCV remains available without model dependencies.

## M3: Interactive SAM/SAM 2 image segmentation

### Objective

Convert user points and boxes into editable mask candidates before inpainting.

### Required scope

- positive and negative points;
- box prompts;
- multiple mask candidates with scores;
- preview, selection, and refinement contracts;
- cached image embeddings;
- `Prompt → Segment → Refine → Inpaint` integration;
- manifest support for previously saved prompts or masks in B2.

### Exit gate

- SAM/SAM 2 is not described or implemented as an automatic semantic watermark detector;
- repeated prompts reuse image embeddings;
- prompt coordinates and transformed image coordinates are tested;
- exact code/checkpoint licenses and hashes are recorded;
- interactive adapters remain separate from the core segmentation contract.

## M4: Automatic image localization

### Objective

Produce watermark region proposals that can feed mask generation and inpainting.

### Required sequence

1. template matching;
2. multi-scale normalized cross-correlation;
3. translucent text and edge-response heuristics;
4. general box detector;
5. box/point proposals refined by SAM.

Candidate visual classes include `corner_logo`, `text_strip`, `timestamp`, `centered_wordmark`, `translucent_badge`, and `custom_overlay`.

### Exit gate

- automatic proposals expose confidence and provenance;
- false-positive behavior is measured on clean images;
- batch inference can reuse detector and embedding caches safely;
- no platform-specific paid-preview detector is introduced;
- users can review, override, or reject proposals.

## M5: Multi-pipeline evaluation

### Objective

Compare replaceable localization, segmentation, and inpainting combinations under reproducible conditions.

### Required comparisons

- ManualBox + RuleMask + OpenCV;
- ManualBox + SAM + LaMa;
- TemplateDetector + SAM + LaMa;
- Detector + SAM + LaMa.

### Required metrics

- mask IoU;
- watermark residual or residual-detection rate;
- clean-image false-positive rate;
- pixels changed outside the final mask;
- boundary color error;
- SSIM and LPIPS where methodologically valid;
- latency and throughput;
- peak RAM and VRAM.

### Exit gate

- metric definitions, data splits, exclusions, and hardware context are documented;
- batch B3 aggregates item metrics without hiding failed items;
- raw per-item results remain available;
- benchmark comparisons use equivalent inputs and mask conditions.

## M6: Image deployment

### Objective

Expose reviewed image pipelines through service and interactive adapters.

### Required scope

- FastAPI image endpoints;
- Gradio or web interaction;
- CPU and CUDA container options;
- ONNX Runtime providers;
- model cache, queue, concurrency, timeout, cancellation, and resource limits;
- asynchronous jobs and status endpoints;
- B3 queue integration using the same batch request/result contracts.
- production packaging and distribution for the selected D1 desktop adapter, if the desktop track has been activated.

Suggested endpoints:

- `/v1/image/detect`;
- `/v1/image/segment`;
- `/v1/image/inpaint`;
- `/v1/image/remove`;
- asynchronous job submission and status endpoints.

### Exit gate

- synchronous and queued behavior share application services;
- desktop, web, and API adapters reuse the same application and batch services;
- uploads, outputs, timeouts, and cancellations have bounded resource behavior;
- the API does not expose server-local arbitrary paths;
- deployment packages include applicable licenses and notices.

## M7: Baseline video watermark removal

### Objective

Provide an explicitly labeled engineering baseline for fixed video overlays.

### Required scope

- FFmpeg decoding;
- first-frame fixed box or static mask;
- per-frame OpenCV or LaMa processing;
- original audio passthrough when compatible;
- output re-encoding;
- FPS, resolution, pixel format, rotation metadata, codec, and failure handling;
- checkpointable long tasks and B4 video batch jobs.

### Exit gate

- fixed corner marks, timestamps, and static text have covered examples;
- output duration, frame count, timing, rotation, and audio are validated;
- interruption does not present a partial file as final output;
- documentation clearly states that temporal flicker may occur;
- the implementation is not described as temporally consistent.

## M8: Temporally consistent video pipeline

### Objective

Add mask propagation and video inpainting behavior designed for cross-frame consistency.

### Required scope

- first-frame prompts;
- SAM 2 propagation or tracking;
- scene-cut detection;
- temporal mask smoothing;
- occlusion and re-entry behavior;
- temporally consistent video inpainting;
- bounded GPU memory and cache lifecycle;
- B4 resume and mixed-duration scheduling.

### Required metrics

- adjacent-frame mask IoU;
- track jitter;
- optical-flow-aligned error;
- flicker;
- processing FPS and real-time factor;
- peak RAM and VRAM.

### Exit gate

- scene cuts, occlusion, mask drift, and cache reset behavior are tested;
- temporal quality is measured rather than inferred from per-frame quality;
- long-running jobs can cancel and resume at defined checkpoints;
- limitations and unsupported content are documented.

## Batch-processing track

### B1: Sequential image batches

- deterministic input ordering;
- directory discovery and versioned manifest input;
- shared defaults with per-item box or mask overrides;
- preflight validation and output-collision detection;
- item-level failure isolation;
- atomic outputs, machine-readable results, and final summary;
- one worker only.

### B2: Resumable model batches

- provider-aware bounded concurrency;
- model and embedding cache reuse;
- input/config fingerprints;
- explicit resume, retry, fail-fast, and overwrite policies;
- no automatic retry for deterministic validation failures.

### B3: Evaluation and service batches

- aggregate metrics with successful, failed, skipped, and cancelled counts;
- raw item-level metrics;
- asynchronous queue and status contracts;
- quotas, timeouts, cancellation, and resource limits;
- CLI, API, and desktop reuse of the same orchestration service.

### B4: Video and long-running batches

- video checkpoint records;
- duration- and resource-aware scheduling;
- safe cancellation and restart;
- partial-work cleanup and final-output atomicity;
- mixed image/video manifests only after both individual pipelines are stable.

## Desktop application track

### D1: Desktop adapter after CLI validation

Entry gate:

- M1 single-image and B1 batch CLI exit gates pass;
- request, result, progress, cancellation, and error contracts have integration coverage;
- the headless core can run without importing a GUI framework.

Required scope:

- perform a focused Qt/Electron/alternative prototype covering packaging, image interaction, Python/model integration, distribution size, updates, and crash isolation;
- record the selected framework and trade-offs in a new ADR;
- open authorized images and select a box or mask;
- preview the final mask and before/after result;
- run single-image and B1 batch operations through public application services;
- expose progress, cancellation, structured errors, overwrite policy, and batch summary;
- add SAM/LaMa/video UI only after the corresponding milestone exit gate passes.

Exit gate:

- no algorithm or batch orchestration logic is duplicated in the desktop adapter;
- at least one target-platform package is reproducibly built and documented;
- model caches and optional runtimes remain owned by the application layer;
- applicable licenses and third-party notices are included in the package;
- closing or cancelling the UI cannot publish a partial final output.

## Change control

- Detailed milestone specifications live under `docs/milestones/`.
- Significant architecture changes require a new ADR.
- A milestone exit gate may be strengthened without an ADR.
- Reordering milestones, weakening safety boundaries, or changing core dependency direction requires explicit review.
