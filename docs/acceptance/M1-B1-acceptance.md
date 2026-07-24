# M1/B1 Acceptance Report

[English](M1-B1-acceptance.md) | [简体中文](M1-B1-acceptance.zh-CN.md)

Related contracts: [M1 specification](../milestones/M1-opencv-image-baseline.md),
[batch-processing design](../BATCH_PROCESSING.md), and [CLI guide](../CLI.md).

## Decision

M1, including the B1 sequential image-batch slice, is accepted for its defined engineering scope.
The accepted baseline may be used as the comparison and compatibility boundary for M2.

This decision confirms the model-free image core, single-image CLI, sequential batch contracts,
state persistence, cancellation boundaries, and documented exit behavior. It does not claim that
OpenCV Telea or Navier-Stokes provides model-level restoration quality.

## Acceptance context

| Field | Value |
|---|---|
| Acceptance date | 2026-07-24 |
| Accepted commit | `73e7cde` (`FEAT:Add B1 batch CLI`) |
| Operating system | Windows |
| Default acceptance runtime | CPython 3.11.15 |
| Additional tested runtimes | CPython 3.12.13 and 3.13.9 |
| Network/model requirement | None |
| Test media | Self-created synthetic `AUTHORIZED DEMO` image and binary mask |

The accepted commit includes all M1 and B1 implementation commits through `73e7cde`. Generated
manual outputs remain under the Git-ignored `tmp/` directory and are intentionally not repository
fixtures.

## Automated verification

The repository-approved checks passed:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build --no-sources
```

Results:

| Check | Result |
|---|---|
| Ruff formatting | Passed |
| Ruff linting | Passed |
| mypy strict checking | Passed |
| CPython 3.11.15 | 441 tests passed |
| CPython 3.12.13 | 441 tests passed |
| CPython 3.13.9 | 441 tests passed |
| Statement coverage | 100% |
| Branch coverage | 100% |
| Source distribution | Built successfully |
| Wheel | Built successfully |

The default suite remained CPU-only, offline, deterministic, and free of model downloads.

## Manual CLI scenarios

The following scenarios were executed through the installed `wrl` command, not by calling OpenCV
or batch-private functions directly.

| Scenario | Observed terminal counts | Exit code | Result |
|---|---:|---:|---|
| Recursive directory, shared box, NS, PNG mapping, custom JSONL | 2 succeeded | `0` | Passed |
| Mirrored mask directory with one missing mask | 1 succeeded, 1 failed | `3` | Passed |
| Manifest with per-item method and option overrides | 2 succeeded | `0` | Passed |
| Manifest fail-fast after an invalid first item | 1 failed, 1 cancelled | `3` | Passed |
| Existing outputs with `error` | Preflight rejection | `2` | Passed |
| Existing outputs with `skip` | 2 skipped | `0` | Passed |

The partial-failure scenario recorded `mask_not_found` without publishing the failed item's output.
The fail-fast scenario recorded `input_not_found` for the first item and `fail_fast` for the
unscheduled item; neither output was published.

The CLI's Ctrl+C-to-`user_cancelled` mapping and exit code `130`, along with fatal orchestration
exit code `4`, are covered by end-to-end CLI tests. The manual terminal session did not synthesize
an operating-system Ctrl+C event because force-terminating a child process would not exercise the
cooperative cancellation contract.

## State and integrity observations

- Directory discovery and result ordering followed deterministic relative-path order.
- `run.json`, every `results.jsonl` record, and `summary.json` parsed as valid JSON.
- `run.json` recorded `worker_count` as 1 and retained normalized portable requests.
- Custom `--results` changed only the JSON Lines location.
- Result records were present for succeeded, failed, skipped, and cancelled terminal states.
- Summary counts and aggregate denominators included every discovered item.
- No `.tmp` files remained after successful or failed scenarios.
- SHA-256 hashes of copied acceptance inputs matched the original synthetic source after execution.
- Failed and cancelled items did not leave final media outputs.
- The Git worktree remained clean because acceptance artifacts were stored under ignored `tmp/`.

## Acceptance traceability

| M1 exit condition | Evidence |
|---|---|
| Box and mask workflows share one application service | Application and CLI integration tests; manual box and mask runs |
| Telea and NS are selectable | Backend, service, CLI, directory, and manifest tests; manual runs of both methods |
| Final-mask exterior pixels remain unchanged before lossy encoding | Mask-only compositing and lossless output tests |
| Supported alpha is preserved | RGBA IO, compositing, and output integration tests |
| Inputs are never modified in place | Immutability tests and manual SHA-256 comparison |
| Output publication is atomic and overwrite behavior is explicit | Atomic-write failure tests; manual `error` and `skip` runs |
| Directory and manifest batches reuse the single-image service | Orchestrator service-injection tests and CLI integration tests |
| Batch order and results are deterministic | Directory, manifest, planning, JSONL, and manual ordering checks |
| Partial failures remain visible without invalidating completed work | Continue and fail-fast tests; manual partial-failure scenarios |
| Required tests pass offline on CPU | 441 tests on each supported Python version |
| CLI help and examples document limits and responsible use | CLI help tests and bilingual CLI guide |
| No model, GUI, video, or automatic detection dependency is introduced | Dependency and final-diff review |

## Accepted limitations

- Localization requires a user-provided box or mask.
- Telea and Navier-Stokes are transparent engineering baselines, not semantic restoration models.
- Large rectangular masks can leave visible seams or bands, especially with Telea.
- Structured, textured, or semantically complex backgrounds can produce obvious artifacts.
- B1 uses one worker and has no retry or resume.
- JPEG output is lossy and cannot preserve alpha.

These limitations are expected M1 behavior and are not reclassified as defects. M2 must retain the
OpenCV baseline while evaluating whether LaMa improves quality under equivalent masks and inputs.

## M2 entry conditions

Acceptance of M1 does not by itself authorize a model artifact. Before M2 implementation:

1. select the exact LaMa code and weight artifact;
2. verify code, weight, and training-data terms separately;
3. record source, version, redistribution terms, and integrity hash in `MODEL_LICENSES.md`;
4. record copied code, binaries, and notices in `THIRD_PARTY_NOTICES.md`;
5. define the padded local-crop and mask-only compositing contract;
6. define CPU/GPU provider, model-cache, offline-test, and benchmark boundaries.
