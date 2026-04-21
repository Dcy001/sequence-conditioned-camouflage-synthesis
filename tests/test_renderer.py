from __future__ import annotations

import numpy as np

from rendering import DeterministicCamouflageRenderer, RendererConfig
from rendering.renderer import lab_to_rgb
import rendering.renderer as renderer_module


def dummy_image() -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:8, :8] = [32, 96, 48]
    image[:8, 8:] = [128, 116, 64]
    image[8:, :8] = [210, 210, 220]
    image[8:, 8:] = [32, 64, 128]
    return image


def test_renderer_output_shape_and_range() -> None:
    config = RendererConfig(k=4, cm_per_pixel=10.0, n_init=2, max_iter=20, random_seed=7)
    renderer = DeterministicCamouflageRenderer(config)

    result = renderer.render(dummy_image())

    assert result.image.shape == (16, 16, 3)
    assert result.image.dtype == np.uint8
    assert result.image.min() >= 0
    assert result.image.max() <= 255
    assert result.preliminary_image is not None
    assert result.preliminary_image.shape == (16, 16, 3)


def test_renderer_operability_and_scale_values() -> None:
    config = RendererConfig(k=3, minimum_unit_cm=35.0, cm_per_pixel=10.0, n_init=2, max_iter=10)
    renderer = DeterministicCamouflageRenderer(config)

    result = renderer.render(dummy_image().astype(np.float32) / 255.0)

    assert result.s_px == 4
    assert result.a_min == 4.0
    assert result.class_map.shape == (16, 16)
    assert result.regularized_class_map.shape == (16, 16)
    assert result.centroids_lab.shape[1] == 3


def test_final_output_uses_centroid_backfilling() -> None:
    config = RendererConfig(
        k=4,
        cm_per_pixel=10.0,
        n_init=2,
        max_iter=20,
        mean_filter_passes=1,
        max_shape_iterations=2,
        random_seed=11,
    )
    renderer = DeterministicCamouflageRenderer(config)

    result = renderer.render(dummy_image())
    palette_rgb = (lab_to_rgb(result.centroids_lab) * 255.0).round().astype(np.uint8)
    expected = palette_rgb[result.regularized_class_map]

    assert np.array_equal(result.image, expected)


def test_renderer_has_no_final_rgb_mean_filter_helper() -> None:
    helper_name = "_mean_filter" + "_rgb"
    assert not hasattr(renderer_module, helper_name)
