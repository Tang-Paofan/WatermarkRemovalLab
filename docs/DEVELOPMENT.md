# Development Guide

[English](DEVELOPMENT.md) | [简体中文](DEVELOPMENT.zh-CN.md)

This guide defines the local Python environment and repository-approved verification commands.
Product architecture and behavior remain governed by the coding standards, accepted ADRs, and
milestone specifications.

## Requirements

- [uv](https://docs.astral.sh/uv/) 0.11 or later
- Git

Python 3.11 is the default development interpreter in `.python-version`. The package supports
Python 3.11 through 3.13. Runtime, optional model, and platform-specific dependencies will remain
in separate dependency groups as they are introduced by an approved milestone.

## Set up the environment

From the repository root:

```powershell
uv sync
```

`uv sync` creates or updates `.venv` from `pyproject.toml` and `uv.lock`. The default environment
contains the package in editable form and the development tools. No model weights or datasets are
downloaded by this command.

M2 exposes mutually exclusive ONNX Runtime extras without adding either package to the default
environment:

```powershell
# CPU package
uv sync --extra lama-onnx-cpu

# CUDA 12.8 / cuDNN 9 package; use a compatible Linux or Windows host
uv sync --extra lama-onnx-cuda
```

Both extras lock ONNX Runtime 1.26.0. Use separate environments for CPU and CUDA acceptance
evidence; requesting both extras in one environment is rejected by the project configuration.
Installing an extra does not install or download the LaMa model.

## Run the required checks

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build --no-sources
```

These commands check formatting, lint rules, static types, tests with coverage, and source/wheel
packaging. `uv build --no-sources` verifies that the package does not rely on undeclared local
workspace sources.

To apply the repository formatter before rerunning the checks:

```powershell
uv run ruff format .
```

Default tests must stay CPU-only, offline, deterministic, and free of model downloads. Add
explicitly marked optional suites when a later milestone introduces GPU or model coverage.

## Model execution environments

The current maintainer workstation is not a real-model execution target. This does not block M2:

- default local development and CI remain model-free;
- AutoDL is the initial reference host for pinned LaMa CPU/CUDA validation;
- another compatible Linux host may reproduce the same evidence;
- the project CLI runs directly on the compute host rather than calling a premature remote API.

Follow [MODEL_EXECUTION.md](MODEL_EXECUTION.md) for the environment split, AutoDL storage rules,
optional runtime groups, model integrity check, pytest markers, CLI shape, and acceptance evidence
fields. Commands documented there are available only after the corresponding M2 implementation
slice exists in the checked-out revision.

## Source layout

```text
src/watermark_removal_lab/  Installable headless Python package
tests/                      Unit and integration tests
docs/                       Engineering contracts and decisions
```

Presentation adapters must depend on public application services; the headless package must not
depend on CLI, desktop, web, or framework-specific objects.
