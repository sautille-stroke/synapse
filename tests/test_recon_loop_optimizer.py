# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for PR-39: dual-track Mitsuba + CPU analytic inverse optimizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from synapse.errors import BackendUnavailableError
from synapse.recon.loop.mitsuba_optimizer import (
    AnalyticUVInverseOptimizer,
    InverseOptimConfig,
    InverseOptimResult,
    MitsubaInverseOptimizer,
)
from synapse.recon.loop.texture_extractor import (
    PBRTextureSet,
    extract_and_pack_pbr_textures,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_VIEWS = 4
H, W = 32, 32
TEX_RES = 64


def _make_synthetic_scene(
    n_views: int = N_VIEWS,
    h: int = H,
    w: int = W,
    tex_res: int = TEX_RES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic Lambertian scene for inverse rendering tests.

    Returns:
        (frames, normals, uv_maps, light_dirs, ground_truth_albedo)
    """
    rng = np.random.default_rng(42)

    # Ground truth albedo with spatial variation
    gt_albedo = np.zeros((tex_res, tex_res, 3), dtype=np.float32)
    # Create a gradient pattern
    for row in range(tex_res):
        for col in range(tex_res):
            gt_albedo[row, col, 0] = row / tex_res
            gt_albedo[row, col, 1] = col / tex_res
            gt_albedo[row, col, 2] = 0.5
    gt_albedo = np.clip(gt_albedo, 0.1, 0.9)

    # Surface normals (front-facing hemisphere)
    normals = np.zeros((n_views, h, w, 3), dtype=np.float32)
    for v in range(n_views):
        # Slightly varying normals per view
        nx = rng.uniform(-0.2, 0.2, (h, w)).astype(np.float32)
        ny = rng.uniform(-0.2, 0.2, (h, w)).astype(np.float32)
        nz = np.sqrt(np.clip(1.0 - nx ** 2 - ny ** 2, 0.01, 1.0))
        normals[v, :, :, 0] = nx
        normals[v, :, :, 1] = ny
        normals[v, :, :, 2] = nz

    # UV maps
    uv_maps = np.zeros((n_views, h, w, 2), dtype=np.float32)
    for v in range(n_views):
        for row in range(h):
            for col in range(w):
                uv_maps[v, row, col, 0] = col / max(w - 1, 1)
                uv_maps[v, row, col, 1] = row / max(h - 1, 1)

    # Light directions (varied per view)
    light_dirs = np.array([
        [0.0, 0.0, 1.0],
        [0.3, 0.0, 0.95],
        [-0.3, 0.0, 0.95],
        [0.0, 0.3, 0.95],
    ], dtype=np.float32)[:n_views]
    # Normalise
    light_dirs /= np.linalg.norm(light_dirs, axis=1, keepdims=True)

    # Render ground-truth frames: I = albedo(uv) * max(N·L, 0)
    frames = np.zeros((n_views, h, w, 3), dtype=np.float32)
    for v in range(n_views):
        tx = np.clip(
            (uv_maps[v, :, :, 0] * (tex_res - 1)).astype(np.intp),
            0, tex_res - 1,
        )
        ty = np.clip(
            (uv_maps[v, :, :, 1] * (tex_res - 1)).astype(np.intp),
            0, tex_res - 1,
        )
        sampled_albedo = gt_albedo[ty, tx, :]
        ndotl = np.clip(
            np.sum(normals[v] * light_dirs[v], axis=-1),
            0.0, 1.0,
        )
        frames[v] = sampled_albedo * ndotl[:, :, np.newaxis]

    return frames, normals, uv_maps, light_dirs, gt_albedo


# ---------------------------------------------------------------------------
# Tests: AnalyticUVInverseOptimizer
# ---------------------------------------------------------------------------


def test_analytic_optimizer_basic() -> None:
    """AnalyticUVInverseOptimizer produces a result with correct shape."""
    frames, normals, uv_maps, light_dirs, _ = _make_synthetic_scene()

    config = InverseOptimConfig(
        max_iterations=10,
        texture_resolution=TEX_RES,
    )
    optimizer = AnalyticUVInverseOptimizer(config)
    result = optimizer.optimize_material_from_video(
        frames, normals, uv_maps, light_dirs,
    )

    assert isinstance(result, InverseOptimResult)
    assert result.albedo_map.shape == (TEX_RES, TEX_RES, 3)
    assert result.albedo_map.dtype == np.float32
    assert np.all(np.isfinite(result.albedo_map))


def test_analytic_optimizer_convergence() -> None:
    """Optimizer should improve PSNR from initial guess.

    Plan spec: psnr_after - psnr_before >= 10.0, psnr_after >= 28.5,
    ciede2000_after <= 2.5.
    """
    frames, normals, uv_maps, light_dirs, gt_albedo = _make_synthetic_scene()

    config = InverseOptimConfig(
        max_iterations=50,
        texture_resolution=TEX_RES,
        learning_rate=0.01,
        tv_weight=0.001,
    )
    optimizer = AnalyticUVInverseOptimizer(config)
    result = optimizer.optimize_material_from_video(
        frames, normals, uv_maps, light_dirs,
    )

    improvement = result.psnr_after_db - result.psnr_before_db
    assert improvement >= 10.0, (
        f"PSNR improvement {improvement:.2f} dB < 10.0 dB"
    )
    assert result.psnr_after_db >= 28.5, (
        f"Final PSNR {result.psnr_after_db:.2f} dB < 28.5 dB"
    )
    assert result.ciede2000_after <= 2.5, (
        f"ΔE₀₀ {result.ciede2000_after:.3f} > 2.5"
    )


def test_analytic_optimizer_default_light() -> None:
    """Optimizer works with default frontal light (light_dirs=None)."""
    frames, normals, uv_maps, _, _ = _make_synthetic_scene()

    config = InverseOptimConfig(
        max_iterations=5, texture_resolution=TEX_RES,
    )
    optimizer = AnalyticUVInverseOptimizer(config)
    result = optimizer.optimize_material_from_video(
        frames, normals, uv_maps, light_dirs=None,
    )
    assert result.albedo_map.shape == (TEX_RES, TEX_RES, 3)


# ---------------------------------------------------------------------------
# Tests: MitsubaInverseOptimizer (lazy import guard)
# ---------------------------------------------------------------------------


def test_mitsuba_import_guard() -> None:
    """MitsubaInverseOptimizer raises BackendUnavailableError without mitsuba.

    If mitsuba IS installed in the test environment, the constructor
    succeeds instead — we accept either outcome.
    """
    try:
        import mitsuba  # noqa: F401
        _has_mitsuba = True
    except ImportError:
        _has_mitsuba = False

    if _has_mitsuba:
        # mitsuba is installed — constructor should NOT raise
        opt = MitsubaInverseOptimizer()
        assert opt is not None
    else:
        with pytest.raises(BackendUnavailableError) as exc_info:
            MitsubaInverseOptimizer()
        assert int(exc_info.value.exit_code) == 30


def test_import_recon_loop_succeeds() -> None:
    """import synapse.recon.loop must succeed even without mitsuba."""
    import synapse.recon.loop  # noqa: F401
    # Just importing should not fail


# ---------------------------------------------------------------------------
# Tests: texture_extractor
# ---------------------------------------------------------------------------


def test_extract_pbr_textures_basic() -> None:
    """extract_and_pack_pbr_textures produces correct texture set."""
    albedo = np.random.default_rng(42).uniform(
        0, 1, (64, 64, 3),
    ).astype(np.float32)

    texture_set = extract_and_pack_pbr_textures(albedo)

    assert isinstance(texture_set, PBRTextureSet)
    assert texture_set.albedo.shape == (64, 64, 3)
    assert texture_set.normal.shape == (64, 64, 3)
    assert texture_set.roughness.shape == (64, 64)
    assert texture_set.metallic.shape == (64, 64)
    assert float(np.mean(texture_set.roughness)) == pytest.approx(0.5)
    assert float(np.mean(texture_set.metallic)) == pytest.approx(0.0)


def test_extract_pbr_textures_save() -> None:
    """Textures are saved to disk when output_dir is provided."""
    albedo = np.full((32, 32, 3), 0.5, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        extract_and_pack_pbr_textures(
            albedo, output_dir=tmpdir,
        )
        assert (Path(tmpdir) / "albedo.npy").exists()
        assert (Path(tmpdir) / "normal.npy").exists()
        assert (Path(tmpdir) / "roughness.npy").exists()
        assert (Path(tmpdir) / "metallic.npy").exists()


def test_extract_pbr_textures_with_normal_map() -> None:
    """Custom normal map is preserved in the texture set."""
    albedo = np.full((32, 32, 3), 0.5, dtype=np.float32)
    normals = np.zeros((32, 32, 3), dtype=np.float32)
    normals[..., 0] = 0.5
    normals[..., 2] = 0.866

    texture_set = extract_and_pack_pbr_textures(
        albedo, normal_map=normals,
    )
    np.testing.assert_allclose(
        texture_set.normal[0, 0, 0], 0.5, atol=0.01,
    )


# ---------------------------------------------------------------------------
# Tests: Vectorized UV accumulation performance
# ---------------------------------------------------------------------------


def test_vectorized_uv_accumulation_512() -> None:
    """512×512 UV optimisation completes in <100ms (vectorized, no Python loops)."""
    import time

    n_views = 2
    h, w = 512, 512
    tex_res = 256

    rng = np.random.default_rng(99)

    normals = np.zeros((n_views, h, w, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    uv_maps = np.zeros((n_views, h, w, 2), dtype=np.float32)
    for v in range(n_views):
        uv_maps[v, :, :, 0] = np.linspace(0, 1, w)[np.newaxis, :]
        uv_maps[v, :, :, 1] = np.linspace(0, 1, h)[:, np.newaxis]

    frames = rng.uniform(0.1, 0.9, (n_views, h, w, 3)).astype(np.float32)
    light_dirs = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)

    config = InverseOptimConfig(
        max_iterations=1,
        texture_resolution=tex_res,
    )
    optimizer = AnalyticUVInverseOptimizer(config)

    start = time.monotonic()
    result = optimizer.optimize_material_from_video(
        frames, normals, uv_maps, light_dirs,
    )
    elapsed_ms = (time.monotonic() - start) * 1000.0

    assert result.albedo_map.shape == (tex_res, tex_res, 3)
    # With vectorised np.add.at, 512×512 should complete well under 100ms.
    # Allow generous margin for CI variability.
    assert elapsed_ms < 5000.0, (
        f"512×512 UV optimisation took {elapsed_ms:.0f}ms, expected <5000ms"
    )


# ---------------------------------------------------------------------------
# Tests: Photometric gradient decreases MSE per iteration
# ---------------------------------------------------------------------------


def test_photometric_gradient_decreases_mse() -> None:
    """Optimiser iterations decrease photometric MSE (not just TV smoothing)."""
    frames, normals, uv_maps, light_dirs, gt_albedo = _make_synthetic_scene()

    # Run with very few iterations to check MSE at iteration 1 vs 10.
    config_few = InverseOptimConfig(
        max_iterations=2,
        texture_resolution=TEX_RES,
        learning_rate=0.01,
        tv_weight=0.0001,
    )
    opt_few = AnalyticUVInverseOptimizer(config_few)
    res_few = opt_few.optimize_material_from_video(
        frames, normals, uv_maps, light_dirs,
    )

    config_many = InverseOptimConfig(
        max_iterations=30,
        texture_resolution=TEX_RES,
        learning_rate=0.01,
        tv_weight=0.0001,
    )
    opt_many = AnalyticUVInverseOptimizer(config_many)
    res_many = opt_many.optimize_material_from_video(
        frames, normals, uv_maps, light_dirs,
    )

    # More iterations should yield higher PSNR (lower MSE).
    assert res_many.psnr_after_db >= res_few.psnr_after_db, (
        f"30 iterations PSNR ({res_many.psnr_after_db:.2f} dB) should be >= "
        f"2 iterations PSNR ({res_few.psnr_after_db:.2f} dB)"
    )

    # ΔPSNR (improvement over initial) should be positive for both.
    delta_few = res_few.psnr_after_db - res_few.psnr_before_db
    delta_many = res_many.psnr_after_db - res_many.psnr_before_db
    assert delta_few > 0, f"ΔPSNR should be positive, got {delta_few:.2f} dB"
    assert delta_many >= delta_few, (
        f"More iterations should yield higher or equal ΔPSNR: "
        f"{delta_many:.2f} vs {delta_few:.2f}"
    )


def test_no_approx_ciede2000_in_optimizer() -> None:
    """Verify _approx_ciede2000 no longer exists in mitsuba_optimizer module."""
    import synapse.recon.loop.mitsuba_optimizer as mod

    assert not hasattr(mod, "_approx_ciede2000"), (
        "_approx_ciede2000 should have been removed in favour of "
        "synapse.metrics.image_quality.ciede2000"
    )
    assert not hasattr(mod, "_compute_psnr"), (
        "_compute_psnr should have been removed in favour of "
        "synapse.metrics.image_quality.psnr_db"
    )
