from __future__ import annotations

import torch

from models import HybridCGANStyleGAN2


def tiny_model() -> HybridCGANStyleGAN2:
    model = HybridCGANStyleGAN2(
        image_size=32,
        sequence_length=2,
        allow_empty_resnet_weights=True,
        feature_batch_size=1,
        generator_config={
            "style_dim": 512,
            "mapping_layers": 2,
            "channel_schedule": {4: 32, 8: 32, 16: 16, 32: 8},
            "start_resolution": 4,
            "use_noise": False,
        },
        discriminator_config={
            "channel_schedule": {32: 16, 16: 32, 8: 64, 4: 64},
        },
    )
    model.eval()
    return model


@torch.no_grad()
def test_end_to_end_hybrid_forward() -> None:
    model = tiny_model()
    sequence = torch.rand(1, 2, 3, 32, 32)
    environment_id = torch.tensor([0])
    season_id = torch.tensor([1])
    z = torch.randn(1, 512)

    out = model(sequence, environment_id, season_id, z=z)

    assert out["c_feat"].shape == (1, 512)
    assert out["c_cls"].shape == (1, 8)
    assert out["fake_image"].shape == (1, 3, 32, 32)
    assert out["adv_score"].shape == (1, 1)
    assert out["env_logits"].shape == (1, 4)
    assert out["season_logits"].shape == (1, 4)
