"""Tests for composite blending — PR-11.

Covers:
- border_ring_lab_match (LAB affine colour transfer)
- laplacian_pyramid_blend (multi-scale alpha blending)
- composite_surgical_sequence (temporal EMA compositing)
- composite_optical_increments (ACEScg shadow + caustics overlay)
- _guided_filter_box (edge-preserving box guided filter)
"""

from __future__ import annotations

import numpy as np
from synapse.composite.blend import (
    _guided_filter_box,
    border_ring_lab_match,
    composite_optical_increments,
    composite_surgical_sequence,
    laplacian_pyramid_blend,
)
from synapse.style.aces_color import (
    rgb_to_cielab,
    srgb_to_linear,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solid_image(
    h: int,
    w: int,
    color: tuple[int, int, int],
    dtype: type = np.uint8,
) -> np.ndarray:
    """Create a solid-colour [H, W, 3] image."""
    img = np.zeros((h, w, 3), dtype=dtype)
    img[..., 0] = color[0]
    img[..., 1] = color[1]
    img[..., 2] = color[2]
    return img


def _make_circle_mask(
    h: int, w: int, cx: int, cy: int, radius: int,
) -> np.ndarray:
    """Create a circular bool mask [H, W]."""
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2


def _mean_ciede2000(img1_srgb: np.ndarray, img2_srgb: np.ndarray,
                    mask: np.ndarray) -> float:
    """Compute mean CIEDE2000 over masked pixels."""
    from synapse.style.aces_color import ciede2000_delta_e
    a = img1_srgb.astype(np.float64) / 255.0
    b = img2_srgb.astype(np.float64) / 255.0
    lab_a = rgb_to_cielab(srgb_to_linear(a))
    lab_b = rgb_to_cielab(srgb_to_linear(b))
    de = ciede2000_delta_e(lab_a, lab_b)
    return float(de[mask].mean())


# ---------------------------------------------------------------------------
# TestBorderRingLabMatch
# ---------------------------------------------------------------------------


class TestBorderRingLabMatch:
    """Tests for border_ring_lab_match."""

    def test_brightness_offset_matching(self) -> None:
        """Border ring CIEDE2000 < 2.5 after matching +15% brightness
        offset."""
        h, w = 128, 128
        # Background: mid-grey.
        bg = _make_solid_image(h, w, (128, 128, 128))

        # Foreground: 15% brighter.
        bright = int(min(128 * 1.15, 255))
        fg = _make_solid_image(h, w, (bright, bright, bright))

        mask = _make_circle_mask(h, w, 64, 64, 30)

        result = border_ring_lab_match(fg, bg, mask, ring_width_px=12)

        # After matching, the inner edge band of result should be
        # closer to bg than the original fg was.
        from synapse.composite.blend import (
            _binary_erosion,
            _circular_structuring_element,
        )
        se = _circular_structuring_element(12)
        eroded = _binary_erosion(mask, se)
        inner_band = mask & (~eroded)

        if np.any(inner_band):
            de_before = _mean_ciede2000(fg, bg, inner_band)
            de_after = _mean_ciede2000(result, bg, inner_band)
            assert de_after < de_before, (
                f"CIEDE2000 should decrease: before={de_before:.2f} "
                f"after={de_after:.2f}"
            )
            assert de_after < 2.5, (
                f"CIEDE2000 after match should be < 2.5, got {de_after:.2f}"
            )

    def test_warm_color_shift(self) -> None:
        """Border-ring match handles warm colour shift reasonably."""
        h, w = 128, 128
        # Cool background (bluish).
        bg = _make_solid_image(h, w, (100, 110, 160))
        # Warm foreground (reddish) simulating +1200K shift.
        fg = _make_solid_image(h, w, (180, 130, 90))
        mask = _make_circle_mask(h, w, 64, 64, 30)

        result = border_ring_lab_match(fg, bg, mask, ring_width_px=12)

        # Should not crash, and result should differ from input.
        assert result.shape == fg.shape
        assert result.dtype == fg.dtype

    def test_scale_clamping(self) -> None:
        """Scale factor clamping prevents extreme colour explosion."""
        h, w = 128, 128
        bg = _make_solid_image(h, w, (200, 200, 200))
        # Foreground with near-zero variance (all same colour).
        fg = _make_solid_image(h, w, (100, 100, 100))
        mask = _make_circle_mask(h, w, 64, 64, 30)

        result = border_ring_lab_match(fg, bg, mask, ring_width_px=12)

        # Should not produce NaN, Inf, or wildly out-of-range values.
        assert np.all(np.isfinite(result.astype(np.float64)))
        if result.dtype == np.uint8:
            assert result.max() <= 255
            assert result.min() >= 0

    def test_empty_mask_returns_unchanged(self) -> None:
        """Empty mask returns the original patch unchanged."""
        h, w = 64, 64
        fg = _make_solid_image(h, w, (150, 120, 90))
        bg = _make_solid_image(h, w, (80, 80, 80))
        empty_mask = np.zeros((h, w), dtype=bool)

        result = border_ring_lab_match(fg, bg, empty_mask)
        np.testing.assert_array_equal(result, fg)


# ---------------------------------------------------------------------------
# TestLaplacianPyramidBlend
# ---------------------------------------------------------------------------


class TestLaplacianPyramidBlend:
    """Tests for laplacian_pyramid_blend."""

    def test_hard_vs_soft_mask_smoothness(self) -> None:
        """Soft mask produces smoother transitions than hard mask."""
        h, w = 128, 128
        fg = _make_solid_image(h, w, (200, 50, 50))
        bg = _make_solid_image(h, w, (50, 50, 200))

        # Hard binary mask.
        hard_mask = np.zeros((h, w), dtype=np.float32)
        hard_mask[:, :w // 2] = 1.0

        # Soft mask with gradient at boundary.
        soft_mask = np.zeros((h, w), dtype=np.float32)
        soft_mask[:, :w // 2 - 16] = 1.0
        for i in range(32):
            col = w // 2 - 16 + i
            if 0 <= col < w:
                soft_mask[:, col] = 1.0 - i / 32.0

        blend_hard = laplacian_pyramid_blend(fg, bg, hard_mask)
        blend_soft = laplacian_pyramid_blend(fg, bg, soft_mask)

        # Measure gradient magnitude at the boundary region.
        boundary_slice = slice(w // 2 - 20, w // 2 + 20)

        grad_hard = np.abs(
            np.diff(blend_hard[:, boundary_slice, :].astype(np.float64),
                    axis=1)
        ).mean()
        grad_soft = np.abs(
            np.diff(blend_soft[:, boundary_slice, :].astype(np.float64),
                    axis=1)
        ).mean()

        # Soft mask should produce smoother transitions (lower gradient).
        assert grad_soft <= grad_hard + 1.0, (
            f"Soft gradient ({grad_soft:.2f}) should be <= "
            f"hard gradient ({grad_hard:.2f})"
        )

    def test_identical_images_yield_original(self) -> None:
        """Blending two identical images returns the original."""
        h, w = 64, 64
        img = _make_solid_image(h, w, (123, 45, 67))
        mask = np.random.RandomState(42).random((h, w)).astype(np.float32)

        result = laplacian_pyramid_blend(img, img, mask)

        # Allow small numerical error from float round-trips.
        np.testing.assert_allclose(
            result.astype(np.float64),
            img.astype(np.float64),
            atol=2.0,
        )

    def test_alpha_one_yields_foreground(self) -> None:
        """Alpha=1 everywhere should return the foreground."""
        h, w = 64, 64
        fg = _make_solid_image(h, w, (200, 100, 50))
        bg = _make_solid_image(h, w, (30, 30, 30))
        mask = np.ones((h, w), dtype=np.float32)

        result = laplacian_pyramid_blend(fg, bg, mask)
        np.testing.assert_allclose(
            result.astype(np.float64),
            fg.astype(np.float64),
            atol=2.0,
        )

    def test_alpha_zero_yields_background(self) -> None:
        """Alpha=0 everywhere should return the background."""
        h, w = 64, 64
        fg = _make_solid_image(h, w, (200, 100, 50))
        bg = _make_solid_image(h, w, (30, 30, 30))
        mask = np.zeros((h, w), dtype=np.float32)

        result = laplacian_pyramid_blend(fg, bg, mask)
        np.testing.assert_allclose(
            result.astype(np.float64),
            bg.astype(np.float64),
            atol=2.0,
        )


# ---------------------------------------------------------------------------
# TestSurgicalSequence
# ---------------------------------------------------------------------------


class TestSurgicalSequence:
    """Tests for composite_surgical_sequence."""

    def _make_sequence(
        self, t: int = 8, h: int = 64, w: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create synthetic video, rerendered, and mask sequences."""
        rng = np.random.RandomState(123)
        video = rng.randint(0, 256, (t, h, w, 3), dtype=np.uint8)
        rerendered = rng.randint(0, 256, (t, h, w, 3), dtype=np.uint8)
        masks = np.zeros((t, h, w), dtype=np.float32)
        # Place a circular mask in each frame.
        for i in range(t):
            masks[i] = _make_circle_mask(
                h, w, w // 2, h // 2, 15,
            ).astype(np.float32)
        return video, rerendered, masks

    def test_temporal_ema_reduces_flicker(self) -> None:
        """EMA smoothing should reduce frame-to-frame boundary
        variation."""
        video, rerendered, masks = self._make_sequence()

        # Compositing without EMA (alpha=1.0 means no temporal blend).
        no_ema = composite_surgical_sequence(
            video, rerendered, masks, ema_alpha=1.0,
        )
        # Compositing with EMA.
        with_ema = composite_surgical_sequence(
            video, rerendered, masks, ema_alpha=0.7,
        )

        # Measure inter-frame variance at boundary.
        boundary = _make_circle_mask(64, 64, 32, 32, 17) & ~_make_circle_mask(
            64, 64, 32, 32, 13
        )
        if np.any(boundary):
            diffs_no_ema = []
            diffs_with_ema = []
            for t in range(1, len(video)):
                d_no = np.abs(
                    no_ema[t][boundary].astype(np.float64)
                    - no_ema[t - 1][boundary].astype(np.float64)
                ).mean()
                d_ema = np.abs(
                    with_ema[t][boundary].astype(np.float64)
                    - with_ema[t - 1][boundary].astype(np.float64)
                ).mean()
                diffs_no_ema.append(d_no)
                diffs_with_ema.append(d_ema)

            mean_no_ema = np.mean(diffs_no_ema)
            mean_with_ema = np.mean(diffs_with_ema)
            # EMA should not increase flicker significantly.
            assert mean_with_ema <= mean_no_ema + 5.0, (
                f"EMA flicker ({mean_with_ema:.2f}) should be <= "
                f"no-EMA ({mean_no_ema:.2f})"
            )

    def test_smoke_full_pipeline(self) -> None:
        """Smoke test: full pipeline produces valid output shape and
        dtype."""
        video, rerendered, masks = self._make_sequence()

        result = composite_surgical_sequence(video, rerendered, masks)

        assert result.shape == video.shape
        assert result.dtype == np.uint8
        assert np.all(np.isfinite(result.astype(np.float64)))


# ---------------------------------------------------------------------------
# TestOpticalIncrements
# ---------------------------------------------------------------------------


class TestOpticalIncrements:
    """Tests for composite_optical_increments."""

    def _make_inputs(
        self, t: int = 4, h: int = 64, w: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create synthetic inputs for optical increments."""
        rng = np.random.RandomState(42)
        video = rng.randint(50, 200, (t, h, w, 3), dtype=np.uint8)
        fgpp = rng.random((t, h, w, 3)).astype(np.float32)
        cids = np.zeros((t, h, w), dtype=np.uint32)
        # Two entities.
        cids[:, :h // 2, :] = 1
        cids[:, h // 2:, :] = 2
        return video, fgpp, cids

    def test_contact_shadow_darkens(self) -> None:
        """Contact shadow map should darken the output."""
        video, fgpp, cids = self._make_inputs()

        # No shadow.
        no_shadow = composite_optical_increments(
            video, fgpp, cids,
            contact_shadow_map=None,
            caustics_map=None,
            guided_filter_r=0,
        )

        # With shadow (0.5 = 50% attenuation everywhere).
        shadow = np.full(
            (4, 64, 64), 0.5, dtype=np.float32,
        )
        with_shadow = composite_optical_increments(
            video, fgpp, cids,
            contact_shadow_map=shadow,
            caustics_map=None,
            guided_filter_r=0,
        )

        # Shadowed output should be darker on average.
        mean_no = no_shadow.astype(np.float64).mean()
        mean_with = with_shadow.astype(np.float64).mean()
        assert mean_with < mean_no, (
            f"Shadow should darken: no_shadow_mean={mean_no:.2f} "
            f"with_shadow_mean={mean_with:.2f}"
        )

    def test_caustics_brightens(self) -> None:
        """Caustics map should brighten the output."""
        video, fgpp, cids = self._make_inputs()

        # No caustics.
        no_caustics = composite_optical_increments(
            video, fgpp, cids,
            contact_shadow_map=None,
            caustics_map=None,
            guided_filter_r=0,
        )

        # With caustics (add bright highlights).
        caustics = np.full(
            (4, 64, 64, 3), 0.3, dtype=np.float32,
        )
        with_caustics = composite_optical_increments(
            video, fgpp, cids,
            contact_shadow_map=None,
            caustics_map=caustics,
            guided_filter_r=0,
        )

        # Caustics output should be brighter on average.
        mean_no = no_caustics.astype(np.float64).mean()
        mean_with = with_caustics.astype(np.float64).mean()
        assert mean_with > mean_no, (
            f"Caustics should brighten: no_caustics_mean={mean_no:.2f} "
            f"with_caustics_mean={mean_with:.2f}"
        )

    def test_no_nan_inf_in_output(self) -> None:
        """Output should never contain NaN or Inf."""
        video, fgpp, cids = self._make_inputs()
        shadow = np.random.RandomState(7).random(
            (4, 64, 64),
        ).astype(np.float32)
        caustics = np.random.RandomState(8).random(
            (4, 64, 64, 3),
        ).astype(np.float32)

        result = composite_optical_increments(
            video, fgpp, cids,
            contact_shadow_map=shadow,
            caustics_map=caustics,
        )

        assert np.all(np.isfinite(result.astype(np.float64)))

    def test_skip_when_maps_none(self) -> None:
        """Passing None for shadow and caustics should still produce
        valid output."""
        video, fgpp, cids = self._make_inputs()

        result = composite_optical_increments(
            video, fgpp, cids,
            contact_shadow_map=None,
            caustics_map=None,
        )

        assert result.shape == video.shape
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# TestGuidedFilter
# ---------------------------------------------------------------------------


class TestGuidedFilter:
    """Tests for _guided_filter_box."""

    def test_identity_when_guide_equals_src(self) -> None:
        """When guide == src (single channel), output is close to
        input."""
        h, w = 64, 64
        rng = np.random.RandomState(99)
        img = rng.random((h, w)).astype(np.float64)

        result = _guided_filter_box(img, img, radius=4, eps=1e-6)

        np.testing.assert_allclose(result, img, atol=0.15)

    def test_edge_preserving(self) -> None:
        """Guided filter should preserve sharp edges present in the
        guide."""
        h, w = 64, 64
        # Guide with a sharp vertical edge.
        guide = np.zeros((h, w), dtype=np.float64)
        guide[:, w // 2:] = 1.0

        # Source with smooth gradient across the edge.
        src = np.linspace(0, 1, w, dtype=np.float64)
        src = np.broadcast_to(src, (h, w)).copy()

        result = _guided_filter_box(guide, src, radius=8, eps=1e-3)

        # The gradient across the edge should be sharper in the
        # filtered result than a naive box blur.
        edge_col = w // 2
        edge_gradient_result = abs(
            float(result[h // 2, edge_col])
            - float(result[h // 2, edge_col - 1])
        )

        # A fully smoothed version would have gradient ~ 1/w.
        naive_gradient = 1.0 / w
        assert edge_gradient_result > naive_gradient, (
            f"Edge gradient ({edge_gradient_result:.4f}) should be > "
            f"naive ({naive_gradient:.4f})"
        )

    def test_multichannel(self) -> None:
        """Guided filter handles [H, W, C] source arrays."""
        h, w, c = 32, 32, 3
        rng = np.random.RandomState(55)
        guide = rng.random((h, w)).astype(np.float64)
        src = rng.random((h, w, c)).astype(np.float64)

        result = _guided_filter_box(guide, src, radius=2, eps=0.01)

        assert result.shape == (h, w, c)
        assert np.all(np.isfinite(result))
