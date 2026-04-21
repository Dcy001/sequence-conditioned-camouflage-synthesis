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
def test_condition_shapes() -> None:
    model = tiny_model()
    sequence = torch.rand(2, 2, 3, 32, 32)
    environment_id = torch.tensor([0, 1])
    season_id = torch.tensor([2, 3])

    c_feat, c_cls = model.condition(sequence, environment_id, season_id)

    assert c_feat.shape == (2, 512)
    assert c_cls.shape == (2, 8)


@torch.no_grad()
def test_generator_and_discriminator_shapes() -> None:
    model = tiny_model()
    c_feat = torch.randn(2, 512)
    c_cls = torch.zeros(2, 8)
    c_cls[:, 0] = 1.0
    z = torch.randn(2, 512)

    fake = model.generate(z, c_feat)
    disc_out = model.discriminate(fake, c_feat, c_cls)

    assert fake.shape == (2, 3, 32, 32)
    assert disc_out["adv_score"].shape == (2, 1)
    assert disc_out["env_logits"].shape == (2, 4)
    assert disc_out["season_logits"].shape == (2, 4)
