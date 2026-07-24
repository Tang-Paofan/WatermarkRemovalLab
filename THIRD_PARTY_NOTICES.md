# Third-Party Notices

Watermark Removal Lab is licensed under Apache License 2.0. Third-party source code, models, weights, datasets, media, fonts, binaries, and tools retain their own licenses and copyright notices.

No third-party source code or media assets are copied directly into this repository. Runtime
dependencies listed below are resolved through `uv.lock` and installed separately by the package
manager; they retain their upstream licenses.

## Notice policy

When third-party material is added:

1. identify the exact source, version, author, and license;
2. retain all applicable copyright, patent, trademark, attribution, and NOTICE text;
3. mark copied or modified files with prominent modification notices when required;
4. record model-specific artifacts in [MODEL_LICENSES.md](MODEL_LICENSES.md);
5. include the relevant license text or a compliant distribution reference;
6. verify that redistribution is allowed for every shipped package and platform.

Add reviewed notices below. Do not remove an existing notice while the corresponding material remains in the repository or distributed artifacts.

## Notices

### NumPy

- Purpose: canonical image and mask arrays plus pure mask transformations.
- Locked versions: 2.4.6 for Python 3.11; 2.5.1 for Python 3.12–3.13.
- Source: <https://github.com/numpy/numpy>
- License: BSD 3-Clause License.
- License texts:
  [NumPy 2.4.6](https://github.com/numpy/numpy/blob/v2.4.6/LICENSE.txt) and
  [NumPy 2.5.1](https://github.com/numpy/numpy/blob/v2.5.1/LICENSE.txt).
- Distribution: external runtime dependency; not copied into this repository or project wheel.

### Pillow

- Purpose: PNG/JPEG decoding, color-mode normalization, transparency extraction, and EXIF
  orientation handling.
- Locked version: 12.3.0.
- Source: <https://github.com/python-pillow/Pillow>
- License: MIT-CMU License.
- License text:
  [Pillow 12.3.0](https://github.com/python-pillow/Pillow/blob/12.3.0/LICENSE).
- Distribution: external runtime dependency; not copied into this repository or project wheel.

### OpenCV Python Headless

- Purpose: CPU-only Telea and Navier-Stokes image inpainting without GUI bindings.
- Locked version: 4.13.0.92.
- Package/source: <https://github.com/opencv/opencv-python/tree/92>.
- Packaging scripts: MIT License
  ([license text](https://github.com/opencv/opencv-python/blob/92/LICENSE.txt)).
- OpenCV: Apache License 2.0
  ([OpenCV 4.13.0 license](https://github.com/opencv/opencv/blob/4.13.0/LICENSE)).
- Bundled binary notices: the wheel includes FFmpeg under LGPL-2.1 and other third-party
  components documented in the exact package
  [third-party license file](https://github.com/opencv/opencv-python/blob/92/LICENSE-3RD-PARTY.txt).
- Distribution: external runtime dependency; not copied into this repository or project wheel.
