# Model Licenses

This file records the license and distribution status of every model package, checkpoint, converted weight, and model-specific dataset used by Watermark Removal Lab.

No third-party model weights are currently bundled or distributed by this repository.

## Required record

Before a model artifact is integrated or distributed, add a reviewed row containing:

| Component | Exact artifact/version | Upstream source | Code license | Weight license | Dataset restrictions | Distribution method | Integrity hash | Review status |
|---|---|---|---|---|---|---|---|---|
| LaMa ONNX FP32 | `lama_fp32.onnx`; artifact commit `a3ee2fca54baebec351b8fa7786154ffa7555aa6`; 208044816 bytes; opset 17; fixed 512 × 512 spatial shape | [Artifact](https://huggingface.co/Carve/LaMa-ONNX/blob/a3ee2fca54baebec351b8fa7786154ffa7555aa6/lama_fp32.onnx), [model card](https://huggingface.co/Carve/LaMa-ONNX/blob/c3c0c9e468934d62e79c329e35d82dd09ff8c444/README.md), [export fork](https://github.com/Carve-Photos/lama), and [original LaMa](https://github.com/advimman/lama) | Original and export-fork repositories carry Apache-2.0 | The Hugging Face model-card metadata declares Apache-2.0; the model repository does not contain a separate weight license file | `big-lama` is identified as trained on Places2/Places365 Challenge; the [official image terms](https://places2.csail.mit.edu/download-private.html) limit data use to non-commercial research and education and prohibit image redistribution | Not bundled; the explicit `wrl model install/status` commands use the fixed descriptor, require terms acceptance for installation, verify size and SHA-256, and publish to a user cache | SHA-256 `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6` | Conditionally approved for non-bundled, non-commercial research integration; weight redistribution and commercial use are not approved |

## LaMa ONNX FP32 review notes

Review date: 2026-07-24.

The reviewed file is the model card's recommended `torch.onnx.export` artifact. The expected
interface is two `float32` inputs named `image` and `mask`, one `float32` output named `output`,
batch size one in this project, and fixed 512 × 512 spatial dimensions. The exact runtime
contract is defined in the
[M2 specification](docs/milestones/M2-lama-image-inpainting.md).

Evidence that supports integration:

- the original LaMa and the `Carve-Photos/lama` export fork carry Apache-2.0;
- the model repository declares Apache-2.0 in its model-card metadata;
- the model card links the original implementation and export fork;
- the export notebook downloads `smartywu/big-lama` and documents the ONNX input/output names,
  fixed spatial shape, opset, normalization, and output range;
- Hugging Face exposes the artifact byte size and SHA-256.

Open issues that limit approval:

- the export notebook clones `Carve-Photos/lama` without pinning a commit;
- the notebook downloads `smartywu/big-lama` from an unpinned `main` URL and does not record the
  source ZIP hash;
- the model repository has license metadata but no dedicated weight license or NOTICE file;
- the identified Places2/Places365 Challenge training images have non-commercial research and
  educational terms, and this project does not make a legal determination about how those terms
  apply to the converted weight.

Therefore:

- M2 code may integrate and locally evaluate this exact hash for research use;
- the repository, wheel, source distribution, and release assets must not contain the weight;
- processing must not download the weight implicitly;
- the installer must show the declared model license and dataset restriction note;
- any redistribution, commercial packaging, hosted service, different artifact, or re-export
  requires a new review.

This record documents engineering and release constraints and is not legal advice.

## Review rules

- Verify the exact upstream repository, release, model card, and artifact URL.
- Do not assume a repository's code license also covers checkpoints or training data.
- Record local conversions such as PyTorch-to-ONNX as derived artifacts and preserve the upstream license.
- Prefer first-run downloads to committing weights into Git.
- Record a cryptographic integrity hash for every downloadable artifact.
- Preserve attribution, NOTICE, acceptable-use, redistribution, and modification requirements.
- Re-review a model when the upstream version, checkpoint, source URL, or license changes.

Candidate technologies named in the roadmap, including SAM, SAM 2, LaMa, and converted ONNX artifacts, are not approved for distribution merely because they are mentioned in project documentation. Each exact artifact must complete this review before integration or release.
