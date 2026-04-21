from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    r, g, b = image.unbind(dim=1)
    maxc = image.max(dim=1).values
    minc = image.min(dim=1).values
    delta = maxc - minc

    hue = torch.zeros_like(maxc)
    mask = delta > 1e-6
    hue = torch.where((maxc == r) & mask, ((g - b) / delta).remainder(6.0), hue)
    hue = torch.where((maxc == g) & mask, ((b - r) / delta) + 2.0, hue)
    hue = torch.where((maxc == b) & mask, ((r - g) / delta) + 4.0, hue)
    hue = hue / 6.0

    saturation = torch.where(maxc > 1e-6, delta / maxc.clamp_min(1e-6), torch.zeros_like(maxc))
    value = maxc
    return torch.stack((hue, saturation, value), dim=1)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).float()


class ConditionalControlModule(nn.Module):
    def __init__(
        self,
        condition_dim: int = 512,
        high_level_dim: int = 2048,
        low_level_dim: int = 864,
        fused_dim: int = 2912,
        environment_classes: int = 4,
        season_classes: int = 4,
        hsv_hist_bins: int = 32,
        lbp_bins: int = 256,
        resnet_weights_path: Optional[str] = None,
        allow_empty_weights: bool = False,
        freeze_resnet: bool = True,
        feature_batch_size: int = 2,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if low_level_dim != 3 * hsv_hist_bins + 3 * lbp_bins:
            raise ValueError("low_level_dim must equal 3 * hsv_hist_bins + 3 * lbp_bins.")
        if fused_dim != high_level_dim + low_level_dim:
            raise ValueError("fused_dim must equal high_level_dim + low_level_dim.")

        self.condition_dim = condition_dim
        self.high_level_dim = high_level_dim
        self.low_level_dim = low_level_dim
        self.fused_dim = fused_dim
        self.environment_classes = environment_classes
        self.season_classes = season_classes
        self.hsv_hist_bins = hsv_hist_bins
        self.lbp_bins = lbp_bins
        self.feature_batch_size = feature_batch_size
        self.freeze_resnet = freeze_resnet

        self.backbone = self._build_resnet50(resnet_weights_path, allow_empty_weights)
        if freeze_resnet:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        self.projector = nn.Sequential(
            nn.Linear(fused_dim, condition_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("resnet_mean", mean, persistent=False)
        self.register_buffer("resnet_std", std, persistent=False)

    @staticmethod
    def _build_resnet50(weights_path: Optional[str], allow_empty_weights: bool) -> nn.Module:
        from torchvision.models import resnet50

        if not weights_path and not allow_empty_weights:
            raise ValueError(
                "A local pretrained ResNet50 checkpoint is required for the high-level semantic branch. "
                "Set model.resnet.weights_path, or set model.resnet.allow_empty_weights=true only for "
                "demo or structural validation runs."
            )

        model = resnet50(weights=None)
        if weights_path:
            checkpoint_path = Path(weights_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"ResNet50 checkpoint not found: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                cleaned: Dict[str, torch.Tensor] = {}
                for key, value in state.items():
                    key = key.replace("module.", "")
                    key = key.replace("backbone.", "")
                    cleaned[key] = value
                model.load_state_dict(cleaned, strict=False)
        model.fc = nn.Identity()
        return model

    @staticmethod
    def _to_unit_rgb(sequence: torch.Tensor) -> torch.Tensor:
        if sequence.min().detach() < 0.0:
            sequence = (sequence + 1.0) * 0.5
        return sequence.clamp(0.0, 1.0)

    def _resnet_features(self, frames: torch.Tensor) -> torch.Tensor:
        outputs = []
        grad_enabled = not self.freeze_resnet
        with torch.set_grad_enabled(grad_enabled):
            for chunk in frames.split(self.feature_batch_size, dim=0):
                x = (chunk - self.resnet_mean) / self.resnet_std
                outputs.append(self.backbone(x))
        return torch.cat(outputs, dim=0)

    def _histogram_features(self, hsv: torch.Tensor) -> torch.Tensor:
        feats = []
        for sample in hsv:
            parts = []
            for channel in sample:
                hist = torch.histc(channel, bins=self.hsv_hist_bins, min=0.0, max=1.0)
                hist = hist / hist.sum().clamp_min(1.0)
                parts.append(hist)
            feats.append(torch.cat(parts, dim=0))
        return torch.stack(feats, dim=0)

    def _lbp_features(self, hsv: torch.Tensor) -> torch.Tensor:
        padded = F.pad(hsv, (1, 1, 1, 1), mode="reflect")
        center = hsv
        offsets = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (2, 1),
            (2, 0),
            (1, 0),
        ]
        codes = torch.zeros_like(center, dtype=torch.long)
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[:, :, dy : dy + hsv.size(2), dx : dx + hsv.size(3)]
            codes = codes + (neighbor >= center).long() * (1 << bit)

        feats = []
        for sample in codes:
            parts = []
            for channel in sample:
                hist = torch.bincount(channel.reshape(-1), minlength=self.lbp_bins).float()
                hist = hist[: self.lbp_bins]
                hist = hist / hist.sum().clamp_min(1.0)
                parts.append(hist)
            feats.append(torch.cat(parts, dim=0))
        return torch.stack(feats, dim=0)

    def _low_level_features(self, frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hsv = rgb_to_hsv(frames)
            colour = self._histogram_features(hsv)
            texture = self._lbp_features(hsv)
            return torch.cat((colour, texture), dim=1)

    def forward(
        self,
        sequence: torch.Tensor,
        environment_id: torch.Tensor,
        season_id: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if sequence.ndim != 5:
            raise ValueError("Expected sequence shape [B, T, 3, H, W].")
        batch, steps, channels, height, width = sequence.shape
        if channels != 3:
            raise ValueError("Expected RGB frames with 3 channels.")

        frames = self._to_unit_rgb(sequence).reshape(batch * steps, channels, height, width)
        high = self._resnet_features(frames)
        low = self._low_level_features(frames).to(high.device)
        fused = torch.cat((high, low), dim=1)
        fused = fused.view(batch, steps, self.fused_dim)
        aggregated = fused.mean(dim=1)
        c_feat = self.projector(aggregated)

        env = one_hot(environment_id.to(sequence.device), self.environment_classes)
        season = one_hot(season_id.to(sequence.device), self.season_classes)
        c_cls = torch.cat((env, season), dim=1)

        if return_intermediate:
            extras = {
                "frame_high": high.view(batch, steps, self.high_level_dim),
                "frame_low": low.view(batch, steps, self.low_level_dim),
                "aggregated": aggregated,
            }
            return c_feat, c_cls, extras
        return c_feat, c_cls

    def train(self, mode: bool = True) -> "ConditionalControlModule":
        super().train(mode)
        if self.freeze_resnet:
            self.backbone.eval()
        return self
