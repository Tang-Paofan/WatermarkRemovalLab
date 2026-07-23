# Model Licenses

This file records the license and distribution status of every model package, checkpoint, converted weight, and model-specific dataset used by Watermark Removal Lab.

No third-party model weights are currently bundled or distributed by this repository.

## Required record

Before a model artifact is integrated or distributed, add a reviewed row containing:

| Component | Exact artifact/version | Upstream source | Code license | Weight license | Dataset restrictions | Distribution method | Integrity hash | Review status |
|---|---|---|---|---|---|---|---|---|
| _None yet_ | — | — | — | — | — | — | — | Not applicable |

## Review rules

- Verify the exact upstream repository, release, model card, and artifact URL.
- Do not assume a repository's code license also covers checkpoints or training data.
- Record local conversions such as PyTorch-to-ONNX as derived artifacts and preserve the upstream license.
- Prefer first-run downloads to committing weights into Git.
- Record a cryptographic integrity hash for every downloadable artifact.
- Preserve attribution, NOTICE, acceptable-use, redistribution, and modification requirements.
- Re-review a model when the upstream version, checkpoint, source URL, or license changes.

Candidate technologies named in the roadmap, including SAM, SAM 2, LaMa, and converted ONNX artifacts, are not approved for distribution merely because they are mentioned in project documentation. Each exact artifact must complete this review before integration or release.
