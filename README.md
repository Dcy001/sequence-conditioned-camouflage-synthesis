# Hybrid cGAN-StyleGAN2 Fused-Background Synthesis

This repository provides the public companion implementation of the hybrid deep model for sequence-conditioned fused-background synthesis, together with the deterministic camouflage renderer. The canonical model entrypoint is `HybridCGANStyleGAN2`.

## Environment

Python 3.9 and PyTorch are expected.

```bash
pip install -r requirements.txt
```

Dependency installation is left to the user.

## Repository Structure

```text
configs/default.yaml
models/
rendering/
utils/
infer_demo.py
render_demo.py
tests/
IMPLEMENTATION_CHECKLIST.md
RELEASE_STATEMENT.md
```

## Public Scope

This public repository contains:

- the released model architecture code
- configuration files
- the deterministic renderer
- runnable demos
- basic tests

This public repository does **not** include:

- trained weights
- dataset files
- restricted training and dataset-preparation workflow components coupled to institution-managed data handling

## Model And Rendering

`HybridCGANStyleGAN2` is the canonical model entrypoint for the deep-learning component.

The high-level semantic branch uses a frozen pretrained ResNet50 under the paper-aligned configuration. This repository does not download pretrained weights automatically; set `model.resnet.weights_path` to a local checkpoint for paper-aligned runs. Empty or random ResNet50 weights are available only through an explicit demo or structural-validation fallback.

The deterministic renderer is provided as a separate post-processing component for fused RGB background images. Rendering defaults are configured under `rendering` in `configs/default.yaml`. `cm_per_pixel` is a user-set scale conversion used to derive the pixel block size for a given image scale.

## Commands

Forward demo:

```bash
python infer_demo.py --config configs/default.yaml --quick --output outputs/demo_sample.png
```

Rendering demo:

```bash
python render_demo.py --config configs/default.yaml --input /path/to/fused_rgb.png --output outputs/rendered_camouflage.png
```

## Notes

Users should provide their own permitted data and local checkpoints where needed. This repository is intended as a public companion codebase for the released model and renderer components.
