from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class EqualLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        bias_init: float = 0.0,
        lr_mul: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.scale = (1.0 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        if bias:
            self.bias = nn.Parameter(torch.full((out_dim,), bias_init))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.eps = eps

        weight = torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight / math.sqrt(in_channels * kernel_size * kernel_size))
        self.affine = EqualLinear(style_dim, in_channels, bias_init=1.0)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch, in_channels, height, width = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {in_channels}.")

        style = self.affine(style).view(batch, 1, self.in_channels, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + self.eps)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        x = x.view(1, batch * self.in_channels, height, width)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        out = out.view(batch, self.out_channels, height, width)
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = x.new_empty(x.size(0), 1, x.size(2), x.size(3)).normal_()
        return x + self.weight * noise


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        negative_slope: float = 0.2,
        use_noise: bool = True,
    ) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, style_dim)
        self.noise = NoiseInjection(out_channels) if use_noise else None
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, style)
        if self.noise is not None:
            out = self.noise(out)
        out = out + self.bias
        return self.activation(out)


class ToRGB(nn.Module):
    def __init__(self, in_channels: int, style_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels,
            3,
            kernel_size=1,
            style_dim=style_dim,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, style: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.conv(x, style) + self.bias
        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2, mode="nearest")
            out = out + skip
        return out


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        out = F.avg_pool2d(out, kernel_size=2)
        skip = F.avg_pool2d(self.skip(x), kernel_size=2)
        return (out + skip) / math.sqrt(2.0)
