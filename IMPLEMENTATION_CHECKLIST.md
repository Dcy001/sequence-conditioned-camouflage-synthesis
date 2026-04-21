# Implementation Checklist

## Random Seed Handling

- Default seed: `2025`.
- Python, NumPy, and PyTorch seeds are set through `utils.seed.set_seed`.
- Deterministic PyTorch settings can be enabled from configuration.

## Sequence-Level Data Split

- Splitting is performed at sequence level, not frame level.
- Default protocol follows the environment-level sequence split:
  - 20 sequences per environment.
  - 14 training sequences per environment.
  - 3 validation sequences per environment.
  - 3 test sequences per environment.
- Seasonal coverage is preserved as closely as possible within each environment.

## Augmentation Policy

- Augmentation is applied only to the training subset.
- Validation and test subsets receive only common preprocessing.
- Training augmentations:
  - rotation
  - horizontal flip
  - vertical flip

## Input Size And Sequence Length

- Frame size: `512 x 512`.
- Sequence length: `64`.
- Frames are RGB for the generator and discriminator.
- HSV conversion is used only for low-level conditioning features.
- Paper-aligned runs require a local pretrained ResNet50 checkpoint; empty or random ResNet50 weights are only for demo or structural validation.

## Key Training Hyperparameters

- Objective: WGAN-GP.
- Optimizer: Adam.
- Generator learning rate: `1e-4`.
- Discriminator learning rate: `1e-4`.
- Adam beta1: `0.0`.
- Adam beta2: `0.99`.
- Effective batch size: 4 sequences per parameter update.
- Gradient accumulation: 2 steps when using 2 sequences per device step.
- Epochs: 300.
- Discriminator updates per generator update: 1.
- Gradient penalty weight: 10.
- Auxiliary heads: environment and season.
- Auxiliary loss weights: `0.25` for environment and `0.25` for season.
- Logging frequency: once per epoch.

## Train, Validation, And Test Conventions

- Training uses sequence-level batches and training-only augmentation.
- Validation and test data remain unchanged apart from common preprocessing.
- Model examples, training, inference, and tests use `HybridCGANStyleGAN2` as the canonical entrypoint.

## Rendering Configuration

- The renderer is a separate post-processing component for fused RGB background images.
- Default dominant-colour count: `5`.
- Default physical minimum unit: `35 cm`.
- `cm_per_pixel` is a configurable scale conversion for deriving pixel block size from the image scale.
- Boundary post-processing regularises class labels before final centroid colour back-filling.
