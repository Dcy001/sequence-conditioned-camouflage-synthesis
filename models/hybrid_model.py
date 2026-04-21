from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .conditional_control import ConditionalControlModule
from .discriminator import ConditionalDiscriminator
from .stylegan2_generator import StyleGAN2Generator


class HybridCGANStyleGAN2(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        sequence_length: int = 64,
        latent_dim: int = 512,
        condition_dim: int = 512,
        class_dim: int = 8,
        environment_classes: int = 4,
        season_classes: int = 4,
        low_level_dim: int = 864,
        high_level_dim: int = 2048,
        fused_dim: int = 2912,
        activation_negative_slope: float = 0.2,
        resnet_weights_path: Optional[str] = None,
        allow_empty_resnet_weights: bool = False,
        freeze_resnet: bool = True,
        feature_batch_size: int = 2,
        hsv_hist_bins: int = 32,
        lbp_bins: int = 256,
        generator_config: Optional[Dict[str, Any]] = None,
        discriminator_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim

        self.conditional_control = ConditionalControlModule(
            condition_dim=condition_dim,
            high_level_dim=high_level_dim,
            low_level_dim=low_level_dim,
            fused_dim=fused_dim,
            environment_classes=environment_classes,
            season_classes=season_classes,
            hsv_hist_bins=hsv_hist_bins,
            lbp_bins=lbp_bins,
            resnet_weights_path=resnet_weights_path,
            allow_empty_weights=allow_empty_resnet_weights,
            freeze_resnet=freeze_resnet,
            feature_batch_size=feature_batch_size,
            negative_slope=activation_negative_slope,
        )

        generator_config = generator_config or {}
        self.generator = StyleGAN2Generator(
            image_size=image_size,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            negative_slope=activation_negative_slope,
            **generator_config,
        )

        discriminator_config = discriminator_config or {}
        self.discriminator = ConditionalDiscriminator(
            image_size=image_size,
            condition_dim=condition_dim,
            class_dim=class_dim,
            environment_classes=environment_classes,
            season_classes=season_classes,
            negative_slope=activation_negative_slope,
            **discriminator_config,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "HybridCGANStyleGAN2":
        model_cfg = cfg.get("model", {})
        resnet_cfg = model_cfg.get("resnet", {})
        low_cfg = model_cfg.get("low_level", {})
        gen_cfg = model_cfg.get("generator", {}).copy()
        disc_cfg = model_cfg.get("discriminator", {}).copy()

        generator_config = {
            "style_dim": gen_cfg.get("style_dim", model_cfg.get("condition_dim", 512)),
            "mapping_layers": gen_cfg.get("mapping_layers", 8),
            "channel_schedule": gen_cfg.get("channel_schedule"),
            "start_resolution": gen_cfg.get("start_resolution", 4),
            "use_noise": gen_cfg.get("noise", True),
        }
        discriminator_config = {
            "channel_schedule": disc_cfg.get("channel_schedule"),
        }

        return cls(
            image_size=model_cfg.get("image_size", 512),
            sequence_length=model_cfg.get("sequence_length", 64),
            latent_dim=model_cfg.get("latent_dim", 512),
            condition_dim=model_cfg.get("condition_dim", 512),
            class_dim=model_cfg.get("class_dim", 8),
            environment_classes=model_cfg.get("environment_classes", 4),
            season_classes=model_cfg.get("season_classes", 4),
            low_level_dim=model_cfg.get("low_level_dim", 864),
            high_level_dim=model_cfg.get("high_level_dim", 2048),
            fused_dim=model_cfg.get("fused_dim", 2912),
            activation_negative_slope=model_cfg.get("activation_negative_slope", 0.2),
            resnet_weights_path=resnet_cfg.get("weights_path"),
            allow_empty_resnet_weights=resnet_cfg.get("allow_empty_weights", False),
            freeze_resnet=resnet_cfg.get("freeze", True),
            feature_batch_size=resnet_cfg.get("feature_batch_size", 2),
            hsv_hist_bins=low_cfg.get("hsv_hist_bins", 32),
            lbp_bins=low_cfg.get("lbp_bins", 256),
            generator_config=generator_config,
            discriminator_config=discriminator_config,
        )

    def condition(
        self,
        sequence: torch.Tensor,
        environment_id: torch.Tensor,
        season_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conditional_control(sequence, environment_id, season_id)

    def generate(self, z: torch.Tensor, c_feat: torch.Tensor) -> torch.Tensor:
        return self.generator(z, c_feat)

    def discriminate(self, image: torch.Tensor, c_feat: torch.Tensor, c_cls: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.discriminator(image, c_feat, c_cls)

    def forward(
        self,
        sequence: torch.Tensor,
        environment_id: torch.Tensor,
        season_id: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        real_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        c_feat, c_cls = self.condition(sequence, environment_id, season_id)
        if z is None:
            z = torch.randn(sequence.size(0), self.latent_dim, device=sequence.device)
        fake_image = self.generate(z, c_feat)
        disc_image = real_image if real_image is not None else fake_image
        disc_out = self.discriminate(disc_image, c_feat, c_cls)
        return {
            "c_feat": c_feat,
            "c_cls": c_cls,
            "fake_image": fake_image,
            **disc_out,
        }
