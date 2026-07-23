# Coding Standards

[English](CODING_STANDARDS.md) | [简体中文](CODING_STANDARDS.zh-CN.md)

These standards apply to production code, tests, scripts, benchmarks, APIs, CLI commands, and desktop adapters in Watermark Removal Lab. They define stable project rules before implementation begins; tool-specific configuration may be added later without weakening these rules.

## 1. Language and formatting

- Source identifiers, type names, code comments, docstrings, log fields, CLI flags, API fields, and machine-readable messages use English.
- User-facing prose may be localized, but languages remain in separate files or resource catalogs.
- Use UTF-8, LF line endings, a final newline, and no trailing whitespace.
- Python uses four spaces and a maximum line length of 100 characters.
- Format-only changes must not be mixed with behavior changes.
- Automated formatting, linting, and type-checking configuration is authoritative once added to the repository.

## 2. Architecture and dependency direction

The system follows:

```text
CLI / Desktop / API adapters
            ↓
Application pipelines
            ↓
Detector / Segmenter / MaskRefiner / Inpainter / QualityEvaluator
            ↓
Image, mask, video, model, and runtime utilities
```

Rules:

- Core and pipeline modules must not import CLI, desktop, web, or GUI modules.
- UI framework types must not appear in core interfaces.
- Algorithm implementations must not parse command-line arguments or render UI.
- File IO, model inference, mask processing, evaluation, and presentation remain separate concerns.
- Localization and inpainting remain independently replaceable.
- Do not add a registry or plugin abstraction until at least two real implementations require it.
- Avoid global mutable state. Model caches and runtime providers must have explicit ownership and lifecycle.

## 3. Public contracts

- Public functions, classes, protocols, dataclasses, and non-obvious array transformations require type hints.
- Prefer small immutable request/result objects over long parameter lists.
- Paths use `pathlib.Path` at Python boundaries.
- Public behavior must document accepted shapes, dtypes, color spaces, coordinate systems, units, and failure modes.
- Do not return unstructured tuples when fields have distinct meanings.
- Compatibility-breaking changes require an architecture decision or a documented migration path after the first public release.

## 4. Image and mask invariants

Unless an adapter explicitly documents otherwise:

- images are `uint8`, `H × W × 3`, RGB;
- alpha is stored separately as `uint8`, `H × W`;
- binary masks are `bool`, `H × W`;
- serialized masks use 0 for background and 255 for selected pixels;
- bounding boxes use `(x_min, y_min, x_max, y_max)` with left/top inclusive and right/bottom exclusive coordinates;
- array coordinates are `(row, column)`, while user-facing points are `(x, y)`;
- mask and image spatial dimensions must match before processing;
- the original input object must not be mutated;
- only pixels inside the final refined mask may change before output encoding;
- the original alpha channel remains unchanged unless a future operation explicitly declares alpha editing.

Color conversion must happen at adapter boundaries. OpenCV-specific code converts RGB/BGR explicitly; model adapters document normalization and tensor layout explicitly.

Lossy formats such as JPEG may change pixels during encoding. Pixel-consistency tests must compare in-memory results or lossless outputs and must distinguish algorithm changes from codec changes.

## 5. Mask processing

- Every mask transformation returns a new mask or clearly documents controlled mutation.
- Dilation, erosion, feathering, thresholding, and padding use explicit units and parameters.
- Empty masks are valid inputs unless an operation explicitly requires selected pixels.
- Full-frame masks, border-touching masks, and out-of-bounds prompts require defined behavior.
- The pipeline records or can reproduce the final mask used for inpainting.
- Refinement must not silently expand beyond configured bounds.

## 6. Errors, logging, and observability

- Validate user input at adapter boundaries and domain invariants in the core.
- Raise specific domain exceptions with actionable messages; do not use bare `except`.
- Preserve the original exception when translating backend failures.
- Core modules use structured logging and never print directly.
- CLI adapters own terminal output and exit-code mapping.
- Logs must not include credentials, full private paths when avoidable, or embedded user media.
- Long-running operations expose progress and cancellation boundaries without depending on a UI framework.

## 7. Dependencies and models

- Add a dependency only when the standard library or an existing dependency cannot reasonably satisfy the requirement.
- Separate runtime, development, optional model, and platform-specific dependencies.
- Pin or lock environments through repository-approved tooling once the package scaffold is introduced.
- Import optional heavy dependencies lazily and report a clear installation remedy when absent.
- Model downloads require an explicit user action, cache location, source URL, expected license, and integrity check.
- Do not commit model weights, datasets, generated outputs, secrets, or local caches.
- Audit code licenses separately from model-weight and dataset licenses.
- Update [MODEL_LICENSES.md](../MODEL_LICENSES.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) before distributing third-party artifacts.

## 8. Tests

Every behavior change requires tests proportional to risk.

- Unit tests cover coordinate conversion, mask operations, validation, and pure transformations.
- Integration tests cover IO, inference adapters, pipelines, CLI behavior, and provider selection.
- Regression tests reproduce fixed defects with the smallest safe fixture.
- Quality tests use synthetic or authorized datasets and record metric definitions.
- Tests must be deterministic or document tolerances and nondeterministic sources.
- CPU-only tests form the required baseline; GPU/model tests must be separately marked.
- Network access and model downloads are disabled in default unit tests.
- Temporary outputs use isolated test directories and are cleaned up by the test framework.

Critical baseline assertions include:

- final-mask dimensions match the image;
- pixels outside the final mask are identical before lossy encoding;
- alpha is preserved;
- input arrays and files are not modified;
- invalid coordinates and mismatched shapes fail clearly;
- CLI failures return non-zero exit codes and do not leave partial final outputs.

## 9. Evaluation and benchmarks

- Benchmarks state hardware, OS, runtime provider, model version/hash, input dimensions, warm-up policy, sample count, and aggregation method.
- Quality metric definitions include masks, crop policy, color space, denominator, and relevant exclusions.
- Compare OpenCV baselines and model methods under the same input and mask conditions.
- Do not present per-frame video inpainting as temporally consistent.
- Performance optimizations must preserve quality and correctness checks or document an intentional trade-off.
- Generated benchmark results are not committed unless they are small, reviewed reference artifacts.

## 10. CLI, API, and desktop adapters

- The CLI is the first product adapter and must call public application services rather than private backend details.
- CLI commands are scriptable: stable exit codes, explicit overwrite behavior, and optional machine-readable output.
- Desktop applications reuse the same request/result contracts, progress events, cancellation, and model cache.
- Qt, Electron, or another UI choice must not change the core pipeline.
- APIs and desktop workers must not create a second implementation of algorithm logic.

## 11. Documentation and responsible use

- Public features require documentation, limitations, and at least one authorized or synthetic example.
- English and Simplified Chinese documents remain separate and structurally aligned when both exist.
- Model/provider names and capabilities must be stated accurately.
- Experimental and planned features must not be described as production-ready.
- Examples must not contain unverified copyrighted media or target paid-preview watermark services.
- Licensing and responsible-use statements are product requirements, not optional marketing text.

## 12. Definition of done

A change is complete only when:

- the intended behavior and non-goals are clear;
- architecture and data invariants remain valid;
- tests and static checks pass;
- documentation and translations are updated where applicable;
- model and third-party licenses are recorded where applicable;
- no generated artifacts, credentials, unrelated changes, or large binaries are included;
- the commit message follows [CONTRIBUTING.md](../CONTRIBUTING.md).
