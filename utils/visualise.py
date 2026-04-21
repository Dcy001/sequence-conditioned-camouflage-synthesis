from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 3 or tensor.size(0) != 3:
        raise ValueError("Expected image tensor [3, H, W].")
    tensor = tensor.detach().cpu().float()
    if tensor.min() < 0.0:
        tensor = (tensor + 1.0) * 0.5
    tensor = tensor.clamp(0.0, 1.0)
    array = tensor.permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def save_image_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)
