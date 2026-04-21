from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from models import HybridCGANStyleGAN2
from utils import load_config, set_seed
from utils.visualise import save_image_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forward demo for HybridCGANStyleGAN2.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/demo_sample.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def apply_quick_overrides(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    demo = cfg.get("demo", {})
    model = cfg["model"]
    image_size = demo.get("quick_image_size", 128)
    sequence_length = demo.get("quick_sequence_length", 4)
    model["image_size"] = image_size
    model["sequence_length"] = sequence_length
    model["resnet"]["feature_batch_size"] = 1
    model["resnet"]["allow_empty_weights"] = True
    model["generator"]["channel_schedule"] = demo.get("quick_generator_channels")
    model["discriminator"]["channel_schedule"] = demo.get("quick_discriminator_channels")
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)
    set_seed(cfg.get("seed", 2025), deterministic=False)

    device = torch.device(args.device)
    model = HybridCGANStyleGAN2.from_config(cfg).to(device)
    model.eval()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)

    batch = 1
    image_size = cfg["model"]["image_size"]
    sequence_length = cfg["model"]["sequence_length"]
    sequence = torch.rand(batch, sequence_length, 3, image_size, image_size, device=device) * 2.0 - 1.0
    environment_id = torch.tensor([0], dtype=torch.long, device=device)
    season_id = torch.tensor([0], dtype=torch.long, device=device)
    z = torch.randn(batch, cfg["model"]["latent_dim"], device=device)

    with torch.no_grad():
        out = model(sequence, environment_id, season_id, z=z)

    print(f"sequence: {tuple(sequence.shape)}")
    print(f"c_feat: {tuple(out['c_feat'].shape)}")
    print(f"c_cls: {tuple(out['c_cls'].shape)}")
    print(f"fake_image: {tuple(out['fake_image'].shape)}")
    print(f"adv_score: {tuple(out['adv_score'].shape)}")
    print(f"env_logits: {tuple(out['env_logits'].shape)}")
    print(f"season_logits: {tuple(out['season_logits'].shape)}")

    save_image_tensor(out["fake_image"][0], Path(args.output))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
