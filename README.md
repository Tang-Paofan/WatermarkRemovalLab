# Watermark Removal Lab

[English](README.md) | [简体中文](README.zh-CN.md)

A learning-oriented image and video watermark removal platform built with OpenCV, SAM/SAM 2, LaMa, ONNX Runtime, and FFmpeg.

> [!IMPORTANT]
> This project is intended only for images and videos you own, created, or are authorized to edit. It is not designed to bypass copyright protection, provenance systems, platform disclosures, or paid-preview watermarks.

## Media support

| Media | Status | Planned pipeline |
|---|---|---|
| Images | In development | Detection / Prompt → SAM → LaMa |
| Videos: fixed marks | Planned | Static mask → frame inpainting |
| Videos: moving marks | Planned | SAM 2 tracking → temporal inpainting |

## Project goals

Watermark Removal Lab is an engineering and learning project for experimenting with:

- visible-watermark detection and interactive prompting;
- mask segmentation, refinement, and propagation;
- image and temporally consistent video inpainting;
- OpenCV and model-based quality baselines;
- ONNX Runtime model deployment and performance tuning;
- FFmpeg video decoding, encoding, and audio passthrough;
- reproducible evaluation of quality, latency, and resource usage.

The project uses a replaceable pipeline:

```text
Detect / Prompt
    → Segment / Localize
    → Refine Mask
    → Inpaint
    → Evaluate
    → Output
```

Localization and inpainting remain separate so that detectors, segmenters, mask refiners, inpainters, and quality evaluators can evolve independently.

## Design principles

- Prefer user clicks, boxes, or masks for the first general-purpose MVP.
- Use SAM/SAM 2 for prompted segmentation, refinement, and tracking—not as a default semantic watermark detector.
- Run LaMa on padded local crops and composite only pixels inside the refined mask.
- Keep OpenCV Telea and Navier-Stokes as transparent, model-free baselines.
- Describe frame-by-frame video processing honestly as a baseline until temporal consistency is implemented and evaluated.
- Cache model assets outside Git and audit code, weight, and dataset licenses separately.

## Responsible use

Use this project only with content you own or have permission to modify. Examples and test assets should be self-created, synthetic, or explicitly licensed. The project does not provide platform-specific detectors for removing stock-media preview watermarks or tools marketed for removing legally required AI labels, provenance metadata, or source attribution.

This statement describes the project's intended use and is not legal advice.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md), the [coding standards](docs/CODING_STANDARDS.md), and the accepted [CLI-first architecture decision](docs/adr/0001-cli-first-core.md) before implementation. Model and third-party artifacts must be recorded in [MODEL_LICENSES.md](MODEL_LICENSES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Source code in this repository is licensed under the [Apache License 2.0](LICENSE). Third-party models, weights, datasets, and bundled assets may use different licenses and will be documented separately before distribution.
