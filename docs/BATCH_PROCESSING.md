# Batch Processing Design

[English](BATCH_PROCESSING.md) | [简体中文](BATCH_PROCESSING.zh-CN.md)

This document defines batch processing as a reusable orchestration capability across image and video milestones. It complements the [engineering roadmap](ROADMAP.md) and the [M1 specification](milestones/M1-opencv-image-baseline.md).

## 1. Design goals

Batch processing must:

- call the same single-item application services used by CLI, desktop, and API adapters;
- support deterministic directory discovery and versioned manifest input;
- validate the full plan before expensive work begins;
- isolate item failures while preserving completed outputs;
- produce incremental machine-readable results and a final summary;
- support safe cancellation and, after B1, verified resume;
- schedule CPU, GPU, model-cache, and video resources within explicit limits;
- expose stable contracts that future desktop and service queues can reuse.

## 2. Non-goals

The batch layer does not:

- implement detection, segmentation, mask refinement, inpainting, encoding, or evaluation algorithms;
- silently infer boxes or masks when an item does not provide an approved localization source;
- overwrite inputs;
- use output existence alone as proof that an item is complete;
- retry deterministic validation or unsupported-content failures;
- hide failed, skipped, or cancelled items inside aggregate metrics;
- introduce distributed execution during B1 or B2;
- support mixed image/video manifests before both individual pipelines are stable.

## 3. Architecture

```text
Directory / Manifest / API / Desktop adapter
                    ↓
              Batch planner
                    ↓
      Validated immutable BatchPlan
                    ↓
     Resource-aware batch orchestrator
                    ↓
          Single-item application service
                    ↓
       Existing image or video pipeline
```

The batch planner resolves defaults, paths, output mappings, and collisions. The orchestrator controls item lifecycle, resources, cancellation, results, and summary. Only the single-item service accesses algorithm pipelines.

## 4. Core contracts

The eventual Python API should express framework-neutral equivalents of:

```text
BatchSpec
  schema_version
  operation
  defaults
  items
  overwrite_policy
  failure_policy
  execution_policy

BatchItem
  index
  id
  media_type
  input
  output
  localization
  options

BatchPlan
  run_id
  normalized_spec
  validated_items
  required_resources
  warnings

ItemResult
  schema_version
  run_id
  item_index
  item_id
  status
  normalized_request
  fingerprints
  output
  metrics
  warnings
  error
  timing

BatchSummary
  run_id
  discovered
  validated
  succeeded
  skipped
  failed
  cancelled
  aggregate_metrics
  result_file
```

Requests and results must not contain Qt, Electron, HTTP, terminal, or framework-specific objects.

## 5. Input modes

### Directory adapter

Directory mode is a convenience adapter. It:

1. discovers supported files;
2. normalizes relative paths;
3. sorts them deterministically;
4. pairs shared boxes or mirrored masks;
5. maps output paths;
6. produces ordinary batch items.

The directory adapter must not execute processing itself.

Rules:

- non-recursive by default;
- explicit recursion;
- ignore hidden batch-state directories;
- reject output roots inside input roots by default;
- normalize case when checking collisions on case-insensitive filesystems;
- resolve symlinks before containment checks;
- never follow a symlink outside the declared input root unless an explicit future policy permits it.

### Manifest adapter

The canonical manifest is versioned JSON Lines. The first record describes the batch and defaults; following records describe items.

```json
{"record":"batch","schema_version":1,"media":"image","operation":"remove","defaults":{"method":"telea","radius":3,"dilate":1}}
{"record":"item","id":"corner-a","input":"inputs/a.png","output":"a.png","box":[20,20,160,48]}
{"record":"item","id":"corner-b","input":"inputs/b.png","output":"b.png","mask":"masks/b.png","method":"ns"}
```

Manifest v1 rules:

- exactly one `batch` record appears first;
- every later record is an `item`;
- item IDs are unique non-empty strings;
- paths use forward slashes in serialized form;
- input and mask paths are relative to the manifest directory;
- output paths are relative to the declared output root;
- each image-removal item has exactly one localization source;
- CLI overrides apply to batch defaults, never silently replace explicit item fields;
- unknown schema versions and unknown required fields fail preflight;
- duplicate canonical output paths fail preflight;
- path traversal outside declared roots fails preflight.

Future manifest versions may add video or mixed media without changing v1 behavior.

## 6. Localization in batches

Batch localization sources may be:

- one shared box validated separately for every image;
- one per-item box;
- a mirrored mask directory;
- one per-item mask;
- later, a reviewed detector configuration or saved prompt set.

Rules:

- coordinates remain `x,y,width,height` in user-facing manifests;
- mask dimensions are validated after image orientation normalization;
- shared boxes may fail on smaller images without invalidating other items;
- automatic localization, when introduced, records detector name, version, confidence, and proposal provenance;
- interactive prompts are batchable only after they have been saved as explicit item data.

## 7. Planning and preflight

Planning completes before the first item runs.

Preflight checks:

- manifest/schema validity;
- unique item IDs;
- supported media and operation;
- input and auxiliary file existence;
- path containment and symlink policy;
- output-path uniqueness;
- no input/output aliasing;
- overwrite-policy conflicts;
- required localization fields;
- option types and ranges;
- known backend/provider availability when it can be checked cheaply;
- estimated resource class and unsupported combinations.

Preflight produces an immutable `BatchPlan`. Item-specific dimensions that require decoding may still fail at item validation time and must be reported as item failures.

## 8. Output mapping and atomicity

The output root contains user outputs. Batch state is isolated under:

```text
OUTPUT_ROOT/
├── <mapped user outputs>
└── .wrl-batch/
    └── RUN_ID/
        ├── run.json
        ├── results.jsonl
        └── summary.json
```

Rules:

- final output paths are known before execution;
- temporary files are created beside their final destination;
- a single-item service validates and atomically commits each output;
- failed items do not leave a final output;
- existing outputs follow `error`, `skip`, or `replace`;
- `error` conflicts are detected during preflight;
- state files never count as input media during directory discovery;
- batch metadata contains paths and metrics, not media pixels.

## 9. Item lifecycle

Each item follows:

```text
discovered
    → validated
    → queued
    → running
    → succeeded | skipped | failed | cancelled
```

After B1, resume may classify a previously successful item as `stale` before it is queued again.

Terminal statuses are immutable within one run. A retry is a new attempt recorded under the same item result history, not a silent status rewrite.

## 10. Failure policy

Default behavior is `continue`:

- item validation or processing failure is recorded;
- other valid items continue;
- the final process exit code indicates partial failure.

Optional `fail-fast` behavior:

- stop scheduling new items after the first failure;
- allow the active atomic step to finish safely;
- retain successful outputs;
- mark unscheduled items as not started or cancelled in the summary.

Fatal batch failures include an unreadable manifest, invalid output root, corrupted state store, or inability to commit result records. Fatal failures stop the run.

## 11. Results and observability

`results.jsonl` is append-only during a run. Each record includes:

- schema and run versions;
- item index and ID;
- attempt number;
- normalized input/output references;
- status;
- normalized pipeline configuration;
- input, auxiliary, config, model, and software fingerprints when available;
- output reference and optional integrity hash;
- duration and resource metrics;
- warnings;
- stable error code, user-facing message, and backend category.

Stack traces belong in debug logs, not standard result records.

`summary.json` is written atomically at the end and includes all terminal counts. Aggregate metrics must include their denominator and never exclude failures without stating the exclusion.

## 12. Exit behavior

CLI exit codes follow the active milestone specification:

- success only when all applicable items succeeded or were intentionally skipped;
- a distinct partial-failure code when one or more items failed;
- a distinct preflight/configuration code;
- a fatal orchestration/IO code;
- 130 for user cancellation.

API and desktop adapters map the same batch summary into their own transport status without changing item semantics.

## 13. Cancellation and crash safety

- Cancellation is a shared framework-neutral token.
- New work stops when cancellation is observed.
- The active item stops only at a backend-defined safe boundary.
- Completed outputs and flushed result records remain valid.
- Temporary output files are cleaned when safe.
- Batch state is flushed after every terminal item result.
- A process crash must not make a partially written media file appear final.

B4 video jobs define additional frame/time checkpoints without weakening final-output atomicity.

## 14. Resume design

Verified resume begins in B2. `--resume` never means “skip if output exists.”

An item may be reused only when all required identities match:

- item ID and normalized output mapping;
- input content fingerprint;
- mask, prompt, template, or auxiliary-content fingerprint;
- normalized pipeline configuration hash;
- application/pipeline version;
- exact model artifact hash when applicable;
- previously recorded successful status;
- expected output exists and passes the configured integrity check.

If any identity differs, the item is `stale` and must be reprocessed or explicitly rejected by policy.

Content hashes use a cryptographic hash suitable for integrity. Release MD5 requirements remain separate and do not replace runtime integrity fingerprints.

## 15. Retry design

Automatic retry is disabled in B1.

Later bounded retry may apply only to classified transient failures such as:

- temporary provider unavailability;
- retryable service transport failures;
- temporary resource exhaustion after reducing concurrency.

Do not automatically retry:

- invalid boxes, masks, manifests, or paths;
- unsupported formats or providers;
- deterministic decode failures;
- license or model-integrity failures;
- output collisions under `error`;
- failures that would repeat without changed input or configuration.

Every attempt is recorded with its cause and delay.

## 16. Concurrency and resource scheduling

B1 uses one worker.

B2 introduces explicit resource-aware concurrency:

- CPU workers are bounded by configuration and memory budget;
- GPU inference defaults to one active worker per device unless a backend proves safe parallelism;
- model sessions and embeddings are shared only through declared thread-safe caches;
- queue depth is bounded;
- items declare resource class before scheduling when possible;
- memory pressure can pause scheduling but must not silently change quality settings;
- output encoding may use a separate bounded pool only when ordering and cancellation remain correct.

Result ordering under concurrency may follow completion order, but every result includes the original item index and the final summary is canonicalized by item index.

## 17. Evaluation aggregation

B3 batch evaluation reports:

- total discovered, valid, successful, failed, skipped, and cancelled items;
- per-pipeline counts;
- metric denominator and missing-value counts;
- latency distribution rather than only an average;
- peak batch RAM/VRAM and configured concurrency;
- raw per-item metrics for audit.

Failed items never receive invented quality scores. Cross-pipeline comparison uses matched successful item sets or explicitly reports differing denominators.

## 18. CLI, API, and desktop reuse

The headless application layer should eventually expose equivalents of:

```text
plan_batch(spec) -> BatchPlan
run_batch(plan, progress_sink, cancellation_token) -> BatchSummary
resume_batch(state, policy, progress_sink, cancellation_token) -> BatchSummary
```

Adapters are responsible for:

- CLI parsing and terminal rendering;
- HTTP serialization and authentication;
- desktop dialogs, tables, and progress views.

Adapters are not responsible for:

- item state transitions;
- resource scheduling;
- retry or resume decisions;
- output atomicity;
- aggregate metric calculation.

## 19. Security and privacy

- Reject path traversal and canonical output collisions.
- Do not expose arbitrary server-local paths through API manifests.
- Do not log media content or credentials.
- Avoid full private paths in portable result files where relative references are sufficient.
- Bound manifest size, item count, queue depth, output size, time, RAM, and VRAM in service environments.
- Validate archive inputs before extraction if archive support is ever added.
- Do not send user media to remote services without an explicit feature and consent.

## 20. Delivery stages

### B1 with M1

- sequential only;
- directory and manifest adapters;
- preflight, atomic outputs, incremental results, summary, fail-fast, cancellation;
- no resume or retry.

### B2 with M2–M4

- verified resume;
- bounded resource-aware concurrency;
- model/session/embedding cache reuse;
- transient retry classification;
- detector and saved-prompt item data.

### B3 with M5–M6

- evaluation aggregation;
- asynchronous job queues;
- status, timeout, quota, and service-safe path contracts;
- desktop/API reuse.

### B4 with M7–M8

- video checkpoints;
- duration-aware scheduling;
- safe restart of long-running jobs;
- mixed media only after individual pipelines meet their exit gates.

## 21. Acceptance principles

A batch stage is complete only when:

- it reuses public single-item services;
- planning is deterministic and validated;
- output mappings are collision-free and never overwrite inputs;
- item failures remain visible;
- final outputs are atomic;
- cancellation preserves completed work and does not publish partial files;
- machine-readable results can reconstruct what happened;
- resource use is bounded for that stage;
- tests cover partial failure and recovery behavior introduced by that stage.
