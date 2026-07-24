# Contributing to Watermark Removal Lab

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

Thank you for helping improve Watermark Removal Lab. Contributions must preserve the project's learning focus, replaceable pipeline architecture, reproducible evaluation, and authorized-content-only boundary.

## Before contributing

- Use only media that you own, created, or are authorized to modify.
- Do not contribute platform-specific logic intended to remove paid-preview or copyright-protection watermarks.
- Do not commit model weights, datasets, generated media, credentials, or local environment files.
- Discuss large architecture changes before implementation.
- Read the [coding standards](docs/CODING_STANDARDS.md), [engineering roadmap](docs/ROADMAP.md), relevant milestone specification, and accepted [CLI-first architecture decision](docs/adr/0001-cli-first-core.md).
- Read the [batch-processing design](docs/BATCH_PROCESSING.md) before changing directory, manifest, queue, resume, retry, or multi-item behavior.

## Change workflow

1. Define one reviewable objective.
2. Create a short-lived branch using a clear prefix such as `feature/`, `fix/`, `docs/`, `refactor/`, or `chore/`.
3. Keep changes focused and add tests or documentation appropriate to the risk.
4. Run the repository checks documented for the affected component.
5. Open a pull request and complete every applicable checklist item.

Do not mix unrelated refactoring, formatting, generated files, or dependency updates into a feature or fix.

## Commit messages

Every commit must use:

```text
TYPE:English description
```

The description must be in English and start with a capital letter. Do not insert a space before or after the colon.

| Type | Use |
|---|---|
| `FEAT` | New behavior or capability |
| `FIX` | Bug fix |
| `ENHANCE` | Performance or resource optimization |
| `DOC` | Documentation |
| `CHORE` | Maintenance, tests, or file organization |
| `BUILD` | Build, packaging, or dependency logic |
| `SCRIPT` | Development or operational scripts |
| `REFACTOR` | Internal restructuring without intended behavior change |
| `STYLE` | Formatting-only changes |
| `OTHER` | Changes that cannot reasonably use another type |

Examples:

```text
FEAT:Add OpenCV image inpainting
FIX:Preserve alpha channel during export
DOC:Document binary mask conventions
```

Prefer several small, coherent commits over one large commit.

## Code and test expectations

- Follow [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md).
- Preserve the dependency direction from adapters toward the headless core.
- Add regression coverage for fixes and focused tests for new behavior.
- Use synthetic, self-created, or explicitly licensed fixtures.
- Treat exact pixels outside the final refined mask as an invariant before lossy encoding.
- Report quality, latency, RAM, and VRAM measurements with enough context to reproduce them.
- Follow the setup and verification commands in the [development guide](docs/DEVELOPMENT.md).

## Models and third-party material

Before adding a model, weight, dataset, copied implementation, or media asset:

1. verify its source and exact license;
2. verify that the license applies to the specific artifact, not only its code repository;
3. record it in [MODEL_LICENSES.md](MODEL_LICENSES.md) or [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md);
4. preserve required copyright, attribution, NOTICE, and modification statements;
5. document the download method and integrity hash without committing large artifacts.

An Apache-2.0 project license does not automatically relicense third-party weights or datasets.

## Documentation languages

English documents use the base filename. Simplified Chinese translations use the `.zh-CN.md` suffix. Keep translations in separate files, add language navigation at the top, and update both versions when changing shared meaning.

Identifiers, commit subjects, code comments, docstrings, log field names, and machine-readable output use English.

## Releases

Every release must:

- use an annotated Git tag;
- reference that tag in the release notes;
- list downloadable resources and their MD5 hashes;
- include applicable model licenses and third-party notices;
- state supported platforms, providers, known limitations, and whether video processing is a baseline or temporally consistent implementation.
