from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from .layers import ResidualDownBlock


def _discriminator_resolutions(image_size: int) -> List[int]:
    if image_size < 4 or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two and at least 4.")
    resolutions = []
    value = image_size
    while value >= 4:
        resolutions.append(value)
        value //= 2
    return resolutions


def _normalize_schedule(schedule: Optional[Dict[int | str, int]], resolutions: List[int]) -> Dict[int, int]:
    if schedule is None:
        schedule = {
            512: 64,
            256: 128,
            128: 256,
            64: 512,
            32: 512,
            16: 512,
            8: 512,
            4: 512,
        }
    normalized = {int(k): int(v) for k, v in schedule.items()}
    missing = [res for res in resolutions if res not in normalized]
    if missing:
        raise ValueError(f"Missing discriminator channel settings for resolutions: {missing}.")
    return normalized


class ConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        condition_dim: int = 512,
        class_dim: int = 8,
        environment_classes: int = 4,
        season_classes: int = 4,
        channel_schedule: Optional[Dict[int | str, int]] = None,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.condition_dim = condition_dim
        self.class_dim = class_dim
        self.environment_classes = environment_classes
        self.season_classes = season_classes
        self.resolutions = _discriminator_resolutions(image_size)
        self.channels = _normalize_schedule(channel_schedule, self.resolutions)

        first_channels = self.channels[image_size]
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, first_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

        blocks = []
        for in_res, out_res in zip(self.resolutions[:-1], self.resolutions[1:]):
            blocks.append(
                ResidualDownBlock(
                    self.channels[in_res],
                    self.channels[out_res],
                    negative_slope=negative_slope,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        final_channels = self.channels[4]
        self.image_readout = nn.Linear(final_channels, 512)

        self.feature_embed = nn.Linear(condition_dim, 512)
        self.class_embed = nn.Linear(class_dim, 512)
        self.critic = nn.Linear(1536, 1)
        self.environment_head = nn.Linear(512, environment_classes)
        self.season_head = nn.Linear(512, season_classes)

    def image_features(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.size(1) != 3:
            raise ValueError("Expected image shape [B, 3, H, W].")
        x = self.from_rgb(image)
        x = self.blocks(x)
        pooled = x.mean(dim=(2, 3))
        return self.image_readout(pooled)

    def forward(self, image: torch.Tensor, c_feat: torch.Tensor, c_cls: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_img = self.image_features(image)
        v_feat = self.feature_embed(c_feat)
        v_cls = self.class_embed(c_cls)
        fused = torch.cat((v_img, v_feat, v_cls), dim=1)
        return {
            "adv_score": self.critic(fused),
            "env_logits": self.environment_head(v_img),
            "season_logits": self.season_head(v_img),
            "v_img": v_img,
            "v_feat": v_feat,
            "v_cls": v_cls,
        }
