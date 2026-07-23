# Watermark Removal Lab

A learning-oriented image and video watermark removal platform built with OpenCV, SAM/SAM 2, LaMa, ONNX Runtime, and FFmpeg.

基于 OpenCV、SAM/SAM 2、LaMa、ONNX Runtime 和 FFmpeg 的图片与视频可见水印定位、分割、补全、评测和部署实验平台。

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

## Roadmap

1. OpenCV image baseline with manual regions and Telea/Navier-Stokes inpainting.
2. Local-crop LaMa image inpainting with mask-only compositing.
3. Interactive SAM/SAM 2 image segmentation with cached embeddings.
4. General image-watermark localization from templates, visual cues, and detectors.
5. Multi-model pipelines and reproducible quality/performance evaluation.
6. FastAPI and web deployment with CPU/CUDA and ONNX Runtime options.
7. Fixed-region video baseline with FFmpeg and audio passthrough.
8. SAM 2 propagation and temporally consistent video inpainting.

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

## Development status

The repository is being initialized. The first implementation milestone will provide an OpenCV image CLI, binary-mask utilities, and minimal automated tests.

## License

Source code in this repository is licensed under the [MIT License](LICENSE). Third-party models, weights, datasets, and bundled assets may use different licenses and will be documented separately before distribution.
