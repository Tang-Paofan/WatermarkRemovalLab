# Agent Development Rules

This file applies to the entire repository. It is the operational contract for coding agents working on Watermark Removal Lab.

## 1. Read before changing files

Before implementation or repository changes:

1. inspect the working tree and existing files;
2. read [CONTRIBUTING.md](CONTRIBUTING.md);
3. read [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md);
4. read [docs/ROADMAP.md](docs/ROADMAP.md) and the relevant specification under `docs/milestones/`;
5. read relevant architecture decisions under `docs/adr/`;
6. read [docs/BATCH_PROCESSING.md](docs/BATCH_PROCESSING.md) when batch behavior is involved;
7. inspect [MODEL_LICENSES.md](MODEL_LICENSES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) when models, datasets, copied code, media, or binaries are involved.

Do not overwrite, discard, reformat, stage, or commit unrelated user changes. If the requested scope conflicts with existing work, stop and explain the conflict.

## 2. Scope and sequencing

- Implement only the milestone or infrastructure explicitly requested by the user.
- A request to discuss, review, diagnose, or plan does not authorize implementation.
- Do not introduce model runtimes, GUI frameworks, web services, video pipelines, or automatic detectors during an earlier milestone unless explicitly requested.
- Prefer the smallest end-to-end slice that is testable and reviewable.
- Do not add speculative abstractions. Add registries and plugin systems only when multiple real implementations require them.
- Keep generated artifacts, model weights, datasets, downloaded media, caches, credentials, and local environment files out of Git.

## 3. Product and safety boundary

This project is for visible-watermark research on media the user owns, created, or is authorized to edit.

Agents must not:

- add platform-specific detectors intended to remove stock-media paid-preview or copyright-protection watermarks;
- market features as bypassing attribution, provenance, platform disclosure, or legally required AI labels;
- add unverified copyrighted examples or datasets;
- present experimental behavior as production-ready;
- present frame-by-frame video inpainting as temporally consistent.

Use synthetic, self-created, or explicitly licensed fixtures and examples.

## 4. Architecture contract

The dependency direction is:

```text
CLI / Desktop / API adapters
            ↓
Application pipelines
            ↓
Detector / Segmenter / MaskRefiner / Inpainter / QualityEvaluator
            ↓
Image, mask, video, model, and runtime utilities
```

Required rules:

- The Python core is headless and independent of Qt, Electron, web, and CLI frameworks.
- Core and pipeline modules never import presentation adapters.
- UI framework objects never appear in core request/result contracts.
- The CLI is the first product adapter and calls public application services.
- Desktop and API adapters reuse the same pipelines; they must not reimplement algorithm logic.
- Localization and inpainting stay independently replaceable.
- File IO, mask processing, inference, evaluation, and presentation remain separate concerns.
- Model caches and runtime providers have explicit owners and lifecycles; avoid global mutable state.

Follow [ADR 0001](docs/adr/0001-cli-first-core.md). A future choice between Qt and Electron must not change the core architecture.

## 5. Pipeline contract

The conceptual pipeline is:

```text
Detect / Prompt
    → Segment / Localize
    → Refine Mask
    → Inpaint
    → Evaluate
    → Output
```

- SAM/SAM 2 are prompted segmenters, mask refiners, or video propagation tools—not default semantic watermark detectors.
- Known overlays may use template matching, NCC, or a detector to generate boxes/points before segmentation.
- General-purpose MVPs prioritize user boxes, clicks, or masks over unsupported automatic claims.
- LaMa-style inpainting should operate on padded local crops and composite only inside the final mask.
- OpenCV Telea and Navier-Stokes remain required model-free baselines.

## 6. Image and mask invariants

Unless an adapter explicitly documents otherwise:

- images: `uint8`, `H × W × 3`, RGB;
- alpha: separate `uint8`, `H × W`;
- binary masks: `bool`, `H × W`;
- serialized masks: 0 background, 255 selected;
- boxes: `(x_min, y_min, x_max, y_max)`, left/top inclusive and right/bottom exclusive;
- array indexing: `(row, column)`; user-facing points: `(x, y)`.

Agents must preserve these invariants:

- mask and image spatial dimensions match;
- input arrays and files are not mutated;
- before lossy encoding, pixels outside the final refined mask are exactly unchanged;
- alpha remains unchanged unless an operation explicitly edits alpha;
- OpenCV RGB/BGR conversion is explicit at its adapter boundary;
- model normalization and tensor layout are documented at the model boundary.

JPEG and other lossy encoders may change pixels globally. Tests must distinguish codec changes from pipeline changes.

## 7. Code quality

- Use English for identifiers, comments, docstrings, logs, CLI flags, API fields, and machine-readable output.
- Use UTF-8, LF, a final newline, four-space Python indentation, and a 100-character Python line limit.
- Public contracts and non-obvious array transformations require type hints.
- Prefer small request/result dataclasses over long parameter lists and unstructured tuples.
- Use `pathlib.Path` at Python boundaries.
- Validate user input at adapter boundaries and domain invariants in the core.
- Raise specific exceptions with actionable messages; never use bare `except`.
- Core code uses logging and never prints directly.
- Optional heavy dependencies are imported lazily with clear installation guidance.
- Do not mix formatting-only changes with behavior changes.

## 8. Tests and verification

Every behavior change requires tests proportional to risk.

- Unit tests cover coordinates, masks, validation, and pure transformations.
- Integration tests cover IO, adapters, pipelines, CLI behavior, and runtime providers.
- Fixes require a focused regression test.
- Default unit tests are CPU-only, offline, and do not download models.
- GPU and model tests are explicitly marked and optional.
- Temporary outputs use isolated test directories.

Critical assertions include:

- masks and images have matching dimensions;
- final-mask exterior pixels are unchanged before lossy encoding;
- alpha is preserved;
- inputs are not modified;
- invalid coordinates and shapes fail clearly;
- CLI failures return non-zero exit codes;
- failed operations do not leave partial final outputs.

Before handoff, run the repository-approved formatter, linter, type checker, unit tests, and relevant integration tests. If the repository does not yet define a command, do not invent a successful result—state what was and was not available.

## 9. Models, dependencies, and licenses

- Add only necessary dependencies and separate runtime, development, optional-model, and platform-specific groups.
- Do not assume a code repository license covers weights or datasets.
- Before integrating an exact model artifact, verify its source, version, code license, weight license, dataset restrictions, redistribution terms, and integrity hash.
- Prefer explicit first-run downloads over committing weights.
- Record model artifacts in `MODEL_LICENSES.md`.
- Record copied code, binaries, assets, and required notices in `THIRD_PARTY_NOTICES.md`.
- Preserve copyright, attribution, NOTICE, patent, trademark, and modification requirements.
- Apache-2.0 for this repository does not relicense third-party artifacts.

## 10. Documentation rules

- English documents use the base filename; Simplified Chinese translations use `.zh-CN.md`.
- Keep languages in separate files and add language navigation when both versions exist.
- Update both language versions when shared meaning changes.
- README files are product-facing overviews. Do not add development roadmaps, milestone plans, or development-status sections to README.
- Put architecture decisions in `docs/adr/` and detailed engineering plans under `docs/`.
- Public features require limitations and at least one authorized or synthetic example.

## 11. Git and review

Commit messages use:

```text
TYPE:English description
```

Allowed types: `FEAT`, `FIX`, `ENHANCE`, `DOC`, `CHORE`, `BUILD`, `SCRIPT`, `REFACTOR`, `STYLE`, and `OTHER`.

- The English description starts with a capital letter.
- Do not place spaces around the colon.
- Keep commits small, coherent, and free of unrelated changes.
- Stage explicit paths when the worktree contains unrelated changes.
- Complete the pull request checklist when a PR is requested.
- Do not push, tag, publish, or create a release unless the user authorizes that action.
- Every release uses an annotated tag and lists downloadable resource MD5 hashes.

## 12. Definition of done

Do not claim completion until:

- requested behavior is implemented and non-goals remain out of scope;
- architecture and data invariants are preserved;
- relevant checks pass, with unavailable checks stated explicitly;
- documentation and translations are updated where applicable;
- model and third-party records are updated where applicable;
- no credentials, caches, large artifacts, or unrelated changes are included;
- the final diff and repository status have been reviewed.
