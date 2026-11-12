"""Tests for composite detector & surgical mask — PR-10.

Covers:
- VideoDefectRegion dataclass construction
- detect_video_artifacts (flicker, silhouette-drift, ground-violation)
- generate_surgical_mask (union, dilation, motion, feathering)
- apply_surgical_mask (alpha blending)
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.composite.detector import (
    VideoDefectRegion,
    detect_video_artifacts,
)
from synapse.composite.mask import (
    apply_surgical_mask,
    generate_surgical_mask,
)
from synapse.fgpp.frame import CameraParams, FGPPFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(w: int = 64, h: int = 64) -> CameraParams:
    """Create a simple pinhole camera for testing."""
    return CameraParams(
        fx=float(w), fy=float(h),
        cx=w / 2.0, cy=h / 2.0,
        width=w, height=h,
    )


def _make_fgpp_frame(
    w: int = 64,
    h: int = 64,
    frame_index: int = 0,
    motion_vector: np.ndarray | None = None,
    cryptomatte_id: np.ndarray | None = None,
) -> FGPPFrame:
    """Create a minimal FGPPFrame for testing."""
    return FGPPFrame(
        camera=_make_camera(w, h),
        depth=np.ones((h, w), dtype=np.float32) * 5.0,
        frame_index=frame_index,
        motion_vector=motion_vector,
        cryptomatte_id=cryptomatte_id,
    )


def _uniform_video(
    t: int, h: int, w: int, color: tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """Create a uniform-colour video [T, H, W, 3] uint8."""
    vid = np.empty((t, h, w, 3), dtype=np.uint8)
    vid[..., 0] = color[0]
    vid[..., 1] = color[1]
    vid[..., 2] = color[2]
    return vid


# =========================================================================
# TestDetector
# =========================================================================


class TestDetector:
    """Tests for detect_video_artifacts."""

    def test_detect_no_artifacts_clean(self) -> None:
        """Identical consecutive frames → no temporal flicker."""
        h, w, t = 64, 64, 4
        video = _uniform_video(t, h, w)
        frames = [_make_fgpp_frame(w, h, i) for i in range(t)]
        defects = detect_video_artifacts(video, frames)
        flickers = [d for d in defects if d.defect_type == "temporal_flicker"]
        assert len(flickers) == 0

    def test_detect_flicker(self) -> None:
        """Inject brightness spike → detects temporal_flicker."""
        h, w, t = 64, 64, 4
        video = _uniform_video(t, h, w, color=(100, 100, 100))
        # Spike frame 2 to white in a large block.
        video[2, :32, :32, :] = 255
        frames = [_make_fgpp_frame(w, h, i) for i in range(t)]
        defects = detect_video_artifacts(video, frames)
        flickers = [d for d in defects if d.defect_type == "temporal_flicker"]
        assert len(flickers) > 0
        assert all(d.severity_score > 0 for d in flickers)

    def test_detect_returns_sorted(self) -> None:
        """Results are sorted by severity descending."""
        h, w, t = 64, 64, 4
        video = _uniform_video(t, h, w, color=(100, 100, 100))
        # Spike multiple frames with different intensities.
        video[1, :32, :32, :] = 200
        video[2, :48, :48, :] = 255
        frames = [_make_fgpp_frame(w, h, i) for i in range(t)]
        defects = detect_video_artifacts(video, frames)
        if len(defects) >= 2:
            for i in range(len(defects) - 1):
                assert defects[i].severity_score >= defects[i + 1].severity_score

    def test_defect_region_fields(self) -> None:
        """VideoDefectRegion dataclass fields are accessible."""
        d = VideoDefectRegion(
            defect_type="temporal_flicker",
            frame_range=(0, 1),
            bbox_xyxy=(10, 20, 30, 40),
            entity_id=0,
            severity_score=0.5,
        )
        assert d.defect_type == "temporal_flicker"
        assert d.frame_range == (0, 1)
        assert d.entity_id == 0
        assert d.details == {}

    def test_invalid_video_shape(self) -> None:
        """Non-4D input raises ValueError."""
        with pytest.raises(ValueError, match="Expected video_srgb shape"):
            detect_video_artifacts(
                np.zeros((64, 64, 3), dtype=np.uint8), []
            )


# =========================================================================
# TestSurgicalMask
# =========================================================================


class TestSurgicalMask:
    """Tests for generate_surgical_mask and apply_surgical_mask."""

    def test_union_masks(self) -> None:
        """mask_3d ∪ mask_sam2 covers both regions."""
        h, w = 64, 64
        mask_3d = np.zeros((h, w), dtype=bool)
        mask_sam2 = np.zeros((h, w), dtype=bool)
        mask_3d[10:20, 10:20] = True
        mask_sam2[30:40, 30:40] = True

        alpha = generate_surgical_mask(
            mask_3d, mask_sam2, r_base=0, feather_width=0.0,
        )
        # Both regions should be fully covered.
        assert np.all(alpha[10:20, 10:20] == 1.0)
        assert np.all(alpha[30:40, 30:40] == 1.0)

    def test_isotropic_dilation(self) -> None:
        """Dilated mask is strictly larger than original."""
        h, w = 64, 64
        mask_3d = np.zeros((h, w), dtype=bool)
        mask_3d[25:35, 25:35] = True
        original_count = int(mask_3d.sum())

        alpha = generate_surgical_mask(
            mask_3d, r_base=4, feather_width=0.0,
        )
        dilated_count = int((alpha > 0).sum())
        assert dilated_count > original_count

    def test_motion_directed_dilation(self) -> None:
        """With rightward motion, mask extends more to the right."""
        h, w = 128, 128
        mask_3d = np.zeros((h, w), dtype=bool)
        mask_3d[50:70, 50:70] = True

        # Strong rightward motion.
        mv = np.zeros((h, w, 2), dtype=np.float32)
        mv[..., 0] = 20.0  # dx = +20 pixels

        alpha = generate_surgical_mask(
            mask_3d, motion_vectors=mv, r_base=2,
            motion_scale=1.5, feather_width=0.0,
        )

        # Count non-zero pixels to the right vs left of original mask.
        original_right_edge = 70
        original_left_edge = 50
        right_extent = int((alpha[:, original_right_edge:] > 0).sum())
        left_extent = int((alpha[:, :original_left_edge] > 0).sum())
        assert right_extent > left_extent

    def test_soft_alpha_range(self) -> None:
        """Output alpha is in [0.0, 1.0]."""
        h, w = 64, 64
        mask_3d = np.zeros((h, w), dtype=bool)
        mask_3d[20:40, 20:40] = True

        alpha = generate_surgical_mask(mask_3d, r_base=4, feather_width=6.0)
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_soft_alpha_interior_one(self) -> None:
        """Interior of the dilated mask has alpha = 1.0."""
        h, w = 64, 64
        mask_3d = np.zeros((h, w), dtype=bool)
        mask_3d[20:40, 20:40] = True

        alpha = generate_surgical_mask(mask_3d, r_base=4, feather_width=6.0)
        # The original mask interior should remain 1.0.
        assert np.all(alpha[20:40, 20:40] == 1.0)

    def test_apply_composite_foreground(self) -> None:
        """alpha=1 → foreground, alpha=0 → background."""
        h, w = 32, 32
        bg = np.full((h, w, 3), 50, dtype=np.uint8)
        fg = np.full((h, w, 3), 200, dtype=np.uint8)

        alpha_full = np.ones((h, w), dtype=np.float32)
        out = apply_surgical_mask(bg, fg, alpha_full)
        np.testing.assert_array_equal(out, fg)

        alpha_zero = np.zeros((h, w), dtype=np.float32)
        out = apply_surgical_mask(bg, fg, alpha_zero)
        np.testing.assert_array_equal(out, bg)

    def test_apply_composite_half(self) -> None:
        """alpha=0.5 → average of fg and bg."""
        h, w = 16, 16
        bg = np.full((h, w, 3), 100, dtype=np.uint8)
        fg = np.full((h, w, 3), 200, dtype=np.uint8)
        alpha = np.full((h, w), 0.5, dtype=np.float32)
        out = apply_surgical_mask(bg, fg, alpha)
        # Should be ~150.
        assert np.allclose(out, 150, atol=1)

    def test_no_mask_returns_zeros(self) -> None:
        """Empty input mask → all-zero alpha."""
        h, w = 32, 32
        mask_3d = np.zeros((h, w), dtype=bool)
        alpha = generate_surgical_mask(mask_3d, r_base=4, feather_width=6.0)
        assert np.all(alpha == 0.0)


# =========================================================================
# TestCoverageSmokeTest
# =========================================================================


class TestCoverageSmokeTest:
    """End-to-end coverage / regression tests."""

    def test_residual_covered(self) -> None:
        """Old mask 10px right of new mask with rightward motion.

        Motion-directed dilation should cover 100% of old mask area.
        """
        h, w = 128, 128
        mask_new = np.zeros((h, w), dtype=bool)
        mask_old = np.zeros((h, w), dtype=bool)
        mask_new[40:80, 40:60] = True   # 40×20 block
        mask_old[40:80, 50:70] = True   # 10px to the right

        mv = np.zeros((h, w, 2), dtype=np.float32)
        mv[..., 0] = 12.0  # rightward 12 px/frame

        alpha = generate_surgical_mask(
            mask_new, mask_old,
            motion_vectors=mv,
            r_base=4, motion_scale=1.5,
            feather_width=0.0,
        )

        # All old-mask pixels should be covered (alpha > 0).
        coverage = float(alpha[mask_old].sum()) / float(mask_old.sum())
        assert coverage >= 1.0, (
            f"Old mask coverage {coverage:.2%} < 100%"
        )

    def test_full_pipeline_smoke(self) -> None:
        """Smoke test: detector + mask + composite all execute."""
        h, w, t = 64, 64, 3
        video = _uniform_video(t, h, w)
        frames = [_make_fgpp_frame(w, h, i) for i in range(t)]

        defects = detect_video_artifacts(video, frames)
        assert isinstance(defects, list)

        mask_3d = np.zeros((h, w), dtype=bool)
        mask_3d[10:30, 10:30] = True
        alpha = generate_surgical_mask(mask_3d, r_base=4)
        assert alpha.shape == (h, w)
        assert alpha.dtype == np.float32

        bg = np.zeros((h, w, 3), dtype=np.uint8)
        fg = np.ones((h, w, 3), dtype=np.uint8) * 255
        out = apply_surgical_mask(bg, fg, alpha)
        assert out.shape == (h, w, 3)
        assert out.dtype == np.uint8
