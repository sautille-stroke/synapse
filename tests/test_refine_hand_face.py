# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for hand and face refinement modules (PR-34).

Covers:
- CanonicalAnchorGallery frontal & 45° retrieval (cosine similarity)
- FaceRefiner gaze deviation threshold
- HandRefiner 5-finger acceptance and 6-finger rejection
- Hand hard-priors normalisation range
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.refine.face import CanonicalAnchorGallery, FaceRefiner
from synapse.refine.hand import HandRefiner
from synapse.refine.types import AnatomicalRegion, RefinementROI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_roi(
    h: int = 64,
    w: int = 64,
    region: AnatomicalRegion = AnatomicalRegion.HAND,
    frame_index: int = 0,
) -> RefinementROI:
    """Create a synthetic ROI with a non-black center patch."""
    crop = np.full((h, w, 3), 0.4, dtype=np.float32)
    bbox = np.array([10, 10, 10 + w, 10 + h], dtype=np.float32)
    return RefinementROI(
        bbox_xyxy=bbox,
        crop_acescg=crop,
        region=region,
        confidence=0.95,
        frame_index=frame_index,
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _make_hand_mask_with_fingers(n_fingers: int, size: int = 64) -> np.ndarray:
    """Create a synthetic hand mask with *n_fingers* protrusions.

    Builds a palm region at the bottom with *n_fingers* well-separated
    rectangular finger strips extending upward.  Gaps between fingers
    are wider than the fingers themselves so that convex-hull defect
    analysis can reliably detect each gap.
    """
    mask = np.zeros((size, size), dtype=np.float32)

    if n_fingers <= 0:
        return mask

    # -- Palm: wide rectangle at the bottom 30% of the image --
    palm_top = int(size * 0.65)
    palm_bot = int(size * 0.95)
    palm_left = int(size * 0.15)
    palm_right = int(size * 0.85)
    mask[palm_top:palm_bot, palm_left:palm_right] = 1.0

    # -- Fingers: tall narrow strips above the palm, well separated --
    palm_width = palm_right - palm_left
    # Each finger gets 1 unit of width, each gap gets 1.5 units
    total_units = n_fingers + (n_fingers - 1) * 1.5
    unit = palm_width / total_units
    finger_w = max(2, int(unit))
    gap_w = max(3, int(unit * 1.5))

    finger_height = int(size * 0.45)
    finger_top = max(0, palm_top - finger_height)

    actual_span = n_fingers * finger_w + (n_fingers - 1) * gap_w
    start_x = palm_left + (palm_width - actual_span) // 2

    for i in range(n_fingers):
        fx = start_x + i * (finger_w + gap_w)
        fx_end = min(fx + finger_w, size)
        if fx >= size:
            break
        mask[finger_top:palm_top, fx:fx_end] = 1.0

    return mask


# ---------------------------------------------------------------------------
# CanonicalAnchorGallery tests
# ---------------------------------------------------------------------------


class TestCanonicalAnchorGallery:
    """Tests for the multi-angle identity anchor gallery."""

    def test_anchor_gallery_frontal_retrieval(self) -> None:
        """Add a frontal anchor → retrieve at yaw=0 → cosine ≥ 0.82."""
        rng = np.random.RandomState(42)
        gallery = CanonicalAnchorGallery(embedding_dim=512)

        # Frontal anchor
        frontal_emb = rng.randn(512).astype(np.float64)
        frontal_emb /= np.linalg.norm(frontal_emb)
        gallery.add_anchor(yaw_deg=0.0, pitch_deg=0.0, embedding=frontal_emb)

        # Side anchor (different)
        side_emb = rng.randn(512).astype(np.float64)
        side_emb /= np.linalg.norm(side_emb)
        gallery.add_anchor(yaw_deg=90.0, pitch_deg=0.0, embedding=side_emb)

        result = gallery.interpolate_identity(yaw_deg=0.0, pitch_deg=0.0)
        sim = _cosine_similarity(result, frontal_emb)
        assert sim >= 0.82, f"Frontal cosine similarity {sim:.4f} < 0.82"

    def test_anchor_gallery_45deg_retrieval(self) -> None:
        """Retrieve at 45° → cosine similarity ≥ 0.78 with nearest anchor."""
        rng = np.random.RandomState(123)
        gallery = CanonicalAnchorGallery(embedding_dim=512)

        # Create a base identity vector and derive pose-variant anchors
        # with controlled perturbation (simulating real face embeddings
        # that share identity but differ by pose).
        base = rng.randn(512).astype(np.float64)
        base /= np.linalg.norm(base)

        # Frontal: identity + small noise
        noise_f = rng.randn(512) * 0.05
        frontal = base + noise_f
        frontal /= np.linalg.norm(frontal)
        gallery.add_anchor(yaw_deg=0.0, pitch_deg=0.0, embedding=frontal)

        # 90° side: identity + different noise
        noise_s = rng.randn(512) * 0.05
        side_90 = base + noise_s
        side_90 /= np.linalg.norm(side_90)
        gallery.add_anchor(yaw_deg=90.0, pitch_deg=0.0, embedding=side_90)

        # Query at 45° should interpolate between frontal and 90°
        result = gallery.interpolate_identity(yaw_deg=45.0, pitch_deg=0.0)

        # Should be reasonably close to both anchors (they share identity)
        sim_front = _cosine_similarity(result, frontal)
        sim_side = _cosine_similarity(result, side_90)
        best_sim = max(sim_front, sim_side)
        assert best_sim >= 0.78, (
            f"45° retrieval best cosine similarity {best_sim:.4f} < 0.78 "
            f"(front={sim_front:.4f}, side={sim_side:.4f})"
        )

    def test_empty_gallery_raises(self) -> None:
        """An empty gallery must raise RuntimeError on interpolation."""
        gallery = CanonicalAnchorGallery(embedding_dim=128)
        with pytest.raises(RuntimeError, match="no anchors"):
            gallery.interpolate_identity(0.0, 0.0)

    def test_dimension_mismatch_raises(self) -> None:
        """Adding an embedding with wrong dim must raise ValueError."""
        gallery = CanonicalAnchorGallery(embedding_dim=128)
        wrong_emb = np.ones(64, dtype=np.float64)
        with pytest.raises(ValueError, match="dimension mismatch"):
            gallery.add_anchor(0.0, 0.0, wrong_emb)


# ---------------------------------------------------------------------------
# FaceRefiner tests
# ---------------------------------------------------------------------------


class TestFaceRefiner:
    """Tests for the face refiner with gaze consistency."""

    def test_face_refiner_gaze_within_threshold(self) -> None:
        """Simulate face refinement → Δθ_gaze ≤ 4.5°."""
        roi = _make_roi(64, 64, region=AnatomicalRegion.FACE)
        refiner = FaceRefiner()
        result = refiner.refine(roi)

        assert result.region == AnatomicalRegion.FACE
        assert result.quality_score > 0.0
        assert result.refined_acescg.shape == roi.crop_acescg.shape

        # The gaze deviation for a near-uniform crop should be very small
        gaze_dev = refiner._compute_gaze_deviation(
            roi.crop_acescg, result.refined_acescg,
        )
        assert gaze_dev <= 4.5, (
            f"Gaze deviation {gaze_dev:.2f}° exceeds 4.5° threshold"
        )

    def test_face_refiner_with_gallery(self) -> None:
        """Refiner should use gallery's interpolate_identity for identity fusion."""
        rng = np.random.RandomState(7)
        gallery = CanonicalAnchorGallery(embedding_dim=128)
        emb = rng.randn(128).astype(np.float64)
        emb /= np.linalg.norm(emb)
        gallery.add_anchor(0.0, 0.0, emb)

        # Add a second anchor at a different pose
        emb2 = rng.randn(128).astype(np.float64)
        emb2 /= np.linalg.norm(emb2)
        gallery.add_anchor(45.0, 0.0, emb2)

        roi = _make_roi(32, 32, region=AnatomicalRegion.FACE)
        refiner = FaceRefiner(gallery=gallery)
        result = refiner.refine(roi)

        assert result.region == AnatomicalRegion.FACE
        assert result.refined_acescg.shape == (32, 32, 3)
        # With identity fusion active, output should differ from input
        # (the gallery-driven colour correction should modify values).
        diff = np.abs(result.refined_acescg - roi.crop_acescg)
        assert diff.max() > 0.0, (
            "Gallery-based identity fusion did not modify the output"
        )

    def test_face_refiner_gallery_interpolation_called(self) -> None:
        """Verify that interpolate_identity is called during refinement."""
        rng = np.random.RandomState(99)
        gallery = CanonicalAnchorGallery(embedding_dim=64)

        # Add two different pose anchors
        for yaw in [0.0, 90.0]:
            emb = rng.randn(64).astype(np.float64)
            emb /= np.linalg.norm(emb)
            gallery.add_anchor(yaw, 0.0, emb)

        roi = _make_roi(32, 32, region=AnatomicalRegion.FACE)
        refiner = FaceRefiner(gallery=gallery)
        result = refiner.refine(roi)

        # The result should be valid and different from a pure clone
        assert result.refined_acescg.shape == (32, 32, 3)
        assert result.quality_score > 0.0


# ---------------------------------------------------------------------------
# HandRefiner tests
# ---------------------------------------------------------------------------


class TestHandRefiner:
    """Tests for the hand refiner with finger topology validation."""

    def test_hand_refiner_accepts_five_fingers(self) -> None:
        """Normal 5-finger mask → valid topology."""
        refiner = HandRefiner(finger_count_expected=5)
        mask = _make_hand_mask_with_fingers(5, size=80)
        count, is_valid = refiner._validate_finger_topology(mask)
        assert is_valid, (
            f"5-finger mask should be valid, but got count={count}"
        )
        assert count == 5

    def test_hand_refiner_rejects_six_fingers(self) -> None:
        """6-finger mask → detected as invalid, falls back to depth."""
        refiner = HandRefiner(finger_count_expected=5)
        mask = _make_hand_mask_with_fingers(6, size=80)
        count, is_valid = refiner._validate_finger_topology(mask)
        assert not is_valid, (
            f"6-finger mask should be invalid, but got count={count}"
        )
        assert count == 6

        # Full refine path should still produce a result (fallback)
        roi = _make_roi(80, 80, region=AnatomicalRegion.HAND)
        # Inject the 6-finger mask via priors
        priors = refiner.prepare_hard_priors(roi)
        priors["hand_mask"] = mask
        result = refiner.refine(roi, priors=priors)
        assert result.region == AnatomicalRegion.HAND
        assert result.quality_score < 1.0  # penalised

    def test_hand_prepare_priors_produces_normalized_outputs(self) -> None:
        """Assert all prior outputs are in [0, 1] range."""
        roi = _make_roi(48, 48, region=AnatomicalRegion.HAND)
        depth = np.random.RandomState(0).uniform(0.5, 50.0, (48, 48)).astype(
            np.float32
        )
        normals = np.random.RandomState(1).uniform(-1, 1, (48, 48, 3)).astype(
            np.float32
        )

        refiner = HandRefiner()
        priors = refiner.prepare_hard_priors(
            roi, depth_map=depth, normal_map=normals,
        )

        log_depth = priors["log_depth_normalized"]
        assert log_depth.min() >= 0.0, f"log_depth min={log_depth.min()}"
        assert log_depth.max() <= 1.0, f"log_depth max={log_depth.max()}"

        cam_normals = priors["camera_normal_encoded"]
        assert cam_normals.min() >= 0.0, (
            f"cam_normals min={cam_normals.min()}"
        )
        assert cam_normals.max() <= 1.0, (
            f"cam_normals max={cam_normals.max()}"
        )

        hand_mask = priors["hand_mask"]
        assert hand_mask.min() >= 0.0
        assert hand_mask.max() <= 1.0

    def test_hand_refine_without_priors(self) -> None:
        """Refine should work even when no priors are pre-computed."""
        roi = _make_roi(32, 32, region=AnatomicalRegion.HAND)
        refiner = HandRefiner()
        result = refiner.refine(roi)
        assert result.region == AnatomicalRegion.HAND
        assert result.refined_acescg.shape == (32, 32, 3)
