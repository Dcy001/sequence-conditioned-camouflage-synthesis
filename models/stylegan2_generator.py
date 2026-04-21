from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .layers import EqualLinear, StyledConv, ToRGB


def _resolution_list(image_size: int, start_resolution: int = 4) -> List[int]:
    if image_size < start_resolution or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two and at least start_resolution.")
    resolutions = []
    value = start_resolution
    while value <= image_size:
        resolutions.append(value)
        value *= 2
    return resolutions


def _normalize_schedule(schedule: Optional[Dict[int | str, int]], resolutions: Iterable[int]) -> Dict[int, int]:
    if schedule is None:
        default = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }
        schedule = default
    normalized = {int(k): int(v) for k, v in schedule.items()}
    missing = [res for res in resolutions if res not in normalized]
    if missing:
        raise ValueError(f"Missing generator channel settings for resolutions: {missing}.")
    return normalized


class MappingNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        style_dim: int = 512,
        num_layers: int = 8,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(EqualLinear(in_dim, style_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            in_dim = style_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleGAN2Generator(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        latent_dim: int = 512,
        condition_dim: int = 512,
        style_dim: int = 512,
        mapping_layers: int = 8,
        channel_schedule: Optional[Dict[int | str, int]] = None,
        start_resolution: int = 4,
        negative_slope: float = 0.2,
        use_noise: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.style_dim = style_dim
        self.start_resolution = start_resolution
        self.resolutions = _resolution_list(image_size, start_resolution)
        self.channels = _normalize_schedule(channel_schedule, self.resolutions)

        self.mapping = MappingNetwork(
            input_dim=latent_dim + condition_dim,
            style_dim=style_dim,
            num_layers=mapping_layers,
            negative_slope=negative_slope,
        )

        first_channels = self.channels[start_resolution]
        self.constant = nn.Parameter(torch.randn(1, first_channels, start_resolution, start_resolution))

        convs = nn.ModuleList()
        to_rgbs = nn.ModuleList()
        in_channels = first_channels
        for resolution in self.resolutions:
            out_channels = self.channels[resolution]
            if resolution == start_resolution:
                convs.append(
                    nn.ModuleList(
                        [
                            StyledConv(in_channels, out_channels, 3, style_dim, negative_slope, use_noise),
                            StyledConv(out_channels, out_channels, 3, style_dim, negative_slope, use_noise),
                        ]
                    )
                )
            else:
                convs.append(
                    nn.ModuleList(
                        [
                            StyledConv(in_channels, out_channels, 3, style_dim, negative_slope, use_noise),
                            StyledConv(out_channels, out_channels, 3, style_dim, negative_slope, use_noise),
                        ]
                    )
                )
            to_rgbs.append(ToRGB(out_channels, style_dim))
            in_channels = out_channels
        self.convs = convs
        self.to_rgbs = to_rgbs

    def forward(self, z: torch.Tensor, c_feat: torch.Tensor, return_latent: bool = False) -> torch.Tensor | Dict[str, torch.Tensor]:
        if z.ndim != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"Expected z shape [B, {self.latent_dim}].")
        if c_feat.ndim != 2 or c_feat.size(1) != self.condition_dim:
            raise ValueError(f"Expected c_feat shape [B, {self.condition_dim}].")

        style_input = torch.cat((z, c_feat), dim=1)
        w = self.mapping(style_input)
        batch = z.size(0)
        x = self.constant.repeat(batch, 1, 1, 1)
        rgb = None

        for index, resolution in enumerate(self.resolutions):
            if index > 0:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            block = self.convs[index]
            x = block[0](x, w)
            x = block[1](x, w)
            rgb = self.to_rgbs[index](x, w, rgb)

        image = torch.tanh(rgb)
        if return_latent:
            return {"image": image, "w": w}
        return image
