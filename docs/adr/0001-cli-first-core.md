# ADR 0001: CLI-First, UI-Agnostic Core

[English](0001-cli-first-core.md) | [简体中文](0001-cli-first-core.zh-CN.md)

- Status: Accepted
- Date: 2026-07-23

## Context

Watermark Removal Lab must support image and video experiments across OpenCV, SAM/SAM 2, LaMa, ONNX Runtime, and FFmpeg. The first usable interface will be a CLI. A desktop application may follow after the CLI proves the pipelines, but the choice between Qt, Electron, or another UI framework is intentionally deferred.

If algorithm code is coupled to the first interface, later desktop, API, and evaluation work would duplicate behavior and make correctness difficult to compare.

## Decision

The project will build a headless, UI-agnostic Python core before any desktop application.

- The CLI is the first adapter and calls public application services.
- Pipelines and algorithm components do not import CLI, desktop, web, or GUI modules.
- Requests, results, progress events, cancellation, and errors use framework-neutral contracts.
- Image and video processing behavior is implemented once in the core.
- A future Qt desktop application may call the Python services directly.
- A future Electron application may call the same services through a local worker or API boundary.
- The desktop framework will be selected only after interaction requirements and packaging constraints are demonstrated by the CLI and early prototypes.

## Initial boundary

```text
CLI
 ↓
Application service
 ↓
Image pipeline
 ↓
Mask utilities + OpenCV inpainter
```

SAM/SAM 2, LaMa, automatic detectors, web APIs, and video processing extend this boundary later without changing the direction of dependency.

## Consequences

Positive:

- algorithm behavior is independently testable;
- CLI, desktop, API, and benchmarks can share the same implementation;
- Qt versus Electron remains a product and distribution decision;
- model caches, progress, cancellation, and evaluation can use common contracts.

Costs:

- adapter boundaries and data contracts must be designed explicitly;
- the first CLI cannot take shortcuts by calling backend-private details;
- GUI-specific image and event types require translation at the desktop boundary.

## Revisit criteria

Revisit this decision only if a proven requirement cannot be represented without a framework-specific core dependency. Convenience alone is not sufficient.
