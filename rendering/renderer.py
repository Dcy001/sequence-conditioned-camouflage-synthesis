from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class RendererConfig:
    k: int = 5
    minimum_unit_cm: float = 35.0
    cm_per_pixel: float = 1.0
    n_init: int = 10
    max_iter: int = 300
    mean_filter_passes: int = 1
    closing_kernel_size: int = 3
    max_shape_iterations: int = 3
    label_change_threshold: float = 0.005
    a_min_factor: float = 0.25
    random_seed: int = 2025

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "RendererConfig":
        render_cfg = cfg.get("rendering", cfg)
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in render_cfg.items() if key in allowed}
        return cls(**values)

    @property
    def s_px(self) -> int:
        if self.cm_per_pixel <= 0:
            raise ValueError("cm_per_pixel must be positive.")
        return max(1, int(round(self.minimum_unit_cm / self.cm_per_pixel)))

    @property
    def a_min(self) -> float:
        return self.a_min_factor * float(self.s_px**2)


@dataclass
class RendererOutput:
    image: np.ndarray
    class_map: np.ndarray
    regularized_class_map: np.ndarray
    centroids_lab: np.ndarray
    s_px: int
    a_min: float
    preliminary_image: Optional[np.ndarray] = None


def _as_rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    elif torch is not None and torch.is_tensor(image):
        tensor = image.detach().cpu().float()
        if tensor.ndim == 4:
            if tensor.size(0) != 1:
                raise ValueError("Batched tensor input must have batch size 1.")
            tensor = tensor[0]
        array = tensor.numpy()
    else:
        array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError("Expected RGB image with 3 dimensions.")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError("Expected RGB image with 3 channels.")

    array = array.astype(np.float64, copy=False)
    if array.min() < 0.0:
        array = (array + 1.0) * 0.5
    elif array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0.0) ** (1.0 / 2.4) - 0.055)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = xyz / white
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f_xyz = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    l = 116.0 * f_xyz[..., 1] - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return np.stack((l, a, b), axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def inverse_f(value: np.ndarray) -> np.ndarray:
        value3 = value**3
        return np.where(value3 > epsilon, value3, (116.0 * value - 16.0) / kappa)

    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = np.stack((inverse_f(fx), inverse_f(fy), inverse_f(fz)), axis=-1) * white
    matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    linear_rgb = xyz @ matrix.T
    return np.clip(_linear_to_srgb(linear_rgb), 0.0, 1.0)


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def _kmeans_plus_plus(values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, values.shape[1]), dtype=np.float64)
    centers[0] = values[rng.integers(0, values.shape[0])]
    closest = _squared_distances(values, centers[:1]).reshape(-1)
    for idx in range(1, k):
        total = closest.sum()
        if total <= 0.0:
            centers[idx] = values[rng.integers(0, values.shape[0])]
        else:
            centers[idx] = values[rng.choice(values.shape[0], p=closest / total)]
        closest = np.minimum(closest, _squared_distances(values, centers[idx : idx + 1]).reshape(-1))
    return centers


def _run_kmeans(
    values: np.ndarray,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected flattened Lab values with shape [N, 3].")
    k = min(k, values.shape[0])
    best_inertia = np.inf
    best_centers: Optional[np.ndarray] = None
    best_labels: Optional[np.ndarray] = None
    rng = np.random.default_rng(seed)

    for _ in range(max(1, n_init)):
        centers = _kmeans_plus_plus(values, k, rng)
        labels = np.zeros(values.shape[0], dtype=np.int64)
        for _ in range(max(1, max_iter)):
            distances = _squared_distances(values, centers)
            new_labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            min_distances = distances[np.arange(values.shape[0]), new_labels]
            for cls in range(k):
                members = values[new_labels == cls]
                if len(members):
                    new_centers[cls] = members.mean(axis=0)
                else:
                    new_centers[cls] = values[min_distances.argmax()]
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                labels = new_labels
                centers = new_centers
                break
            labels = new_labels
            centers = new_centers
        distances = _squared_distances(values, centers)
        labels = distances.argmin(axis=1)
        inertia = distances[np.arange(values.shape[0]), labels].sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()

    if best_centers is None or best_labels is None:
        raise RuntimeError("K-means failed to produce centroids.")
    return best_centers, best_labels


def _mode_label(values: np.ndarray, k: int) -> int:
    counts = np.bincount(values.reshape(-1), minlength=k)
    return int(counts.argmax())


def _block_refill(labels: np.ndarray, k: int, block_size: int) -> np.ndarray:
    out = labels.copy()
    height, width = labels.shape
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = labels[y : y + block_size, x : x + block_size]
            out[y : y + block_size, x : x + block_size] = _mode_label(block, k)
    return out


def _majority_label_smoothing(labels: np.ndarray, k: int) -> np.ndarray:
    height, width = labels.shape
    padded = np.pad(labels, 1, mode="edge")
    counts = np.zeros((k, height, width), dtype=np.int32)
    for dy in range(3):
        for dx in range(3):
            view = padded[dy : dy + height, dx : dx + width]
            for cls in range(k):
                counts[cls] += view == cls
    return counts.argmax(axis=0).astype(labels.dtype, copy=False)


def _binary_dilation(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _binary_erosion(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=True)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _binary_erosion(_binary_dilation(mask, kernel_size), kernel_size)


def _neighbour_support(labels: np.ndarray, k: int) -> np.ndarray:
    height, width = labels.shape
    padded = np.pad(labels, 1, mode="edge")
    support = np.zeros((k, height, width), dtype=np.int32)
    for dy in range(3):
        for dx in range(3):
            view = padded[dy : dy + height, dx : dx + width]
            for cls in range(k):
                support[cls] += view == cls
    return support


def _close_label_map(labels: np.ndarray, k: int, kernel_size: int) -> np.ndarray:
    closed = np.stack([_closing(labels == cls, kernel_size) for cls in range(k)], axis=0)
    support = _neighbour_support(labels, k)
    masked_support = np.where(closed, support, -1)
    any_closed = closed.any(axis=0)
    out = labels.copy()
    out[any_closed] = masked_support.argmax(axis=0)[any_closed]
    return out


def _component_pixels(mask: np.ndarray, start: Tuple[int, int], visited: np.ndarray) -> list[Tuple[int, int]]:
    height, width = mask.shape
    queue: deque[Tuple[int, int]] = deque([start])
    visited[start] = True
    pixels = []
    while queue:
        y, x = queue.popleft()
        pixels.append((y, x))
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    return pixels


def _merge_small_components(labels: np.ndarray, k: int, min_area: float) -> np.ndarray:
    out = labels.copy()
    height, width = labels.shape
    for cls in range(k):
        mask = out == cls
        visited = np.zeros_like(mask, dtype=bool)
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            if visited[y, x]:
                continue
            pixels = _component_pixels(mask, (int(y), int(x)), visited)
            if len(pixels) >= min_area:
                continue
            neighbours = []
            component = set(pixels)
            for py, px in pixels:
                for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                    if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in component:
                        neighbours.append(int(out[ny, nx]))
            if neighbours:
                replacement = int(np.bincount(np.array(neighbours), minlength=k).argmax())
                for py, px in pixels:
                    out[py, px] = replacement
    return out


def _lab_backfill_to_uint8(centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = lab_to_rgb(centers[labels])
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


class DeterministicCamouflageRenderer:
    def __init__(self, config: RendererConfig | Dict[str, Any] | None = None) -> None:
        if config is None:
            self.config = RendererConfig()
        elif isinstance(config, RendererConfig):
            self.config = config
        else:
            self.config = RendererConfig.from_dict(config)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DeterministicCamouflageRenderer":
        return cls(RendererConfig.from_dict(cfg))

    def render(self, image: Any) -> RendererOutput:
        rgb = _as_rgb_array(image)
        lab = rgb_to_lab(rgb)
        height, width, _ = lab.shape
        values = lab.reshape(-1, 3)
        centers, labels = _run_kmeans(
            values,
            k=self.config.k,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            seed=self.config.random_seed,
        )
        class_map = labels.reshape(height, width)
        s_px = self.config.s_px
        regularized = _block_refill(class_map, centers.shape[0], s_px)
        preliminary_image = _lab_backfill_to_uint8(centers, regularized)

        for _ in range(self.config.max_shape_iterations):
            previous = regularized.copy()
            for _ in range(self.config.mean_filter_passes):
                regularized = _majority_label_smoothing(regularized, centers.shape[0])
            regularized = _close_label_map(regularized, centers.shape[0], self.config.closing_kernel_size)
            regularized = _merge_small_components(regularized, centers.shape[0], self.config.a_min)
            change_ratio = float(np.mean(regularized != previous))
            if change_ratio < self.config.label_change_threshold:
                break

        image_uint8 = _lab_backfill_to_uint8(centers, regularized)
        return RendererOutput(
            image=image_uint8,
            class_map=class_map.astype(np.int64),
            regularized_class_map=regularized.astype(np.int64),
            centroids_lab=centers,
            s_px=s_px,
            a_min=self.config.a_min,
            preliminary_image=preliminary_image,
        )
