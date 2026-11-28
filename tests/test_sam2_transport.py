"""Tests for SAM 2.1 Video Object Segmentation transport and Cryptomatte fusion.

Covers:
- SAM2PromptType / SAM2Prompt dataclass construction
- SAM2InstanceMask / SAM2FrameResult / SAM2SequenceResult
- MockSAM2Backend VOS inference
- FoundarySAM2Backend raises when unconnected
- fuse_sam2_cryptomatte mask fusion
- compute_mask_iou metric
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.bridge.sam2_transport import (
    FoundarySAM2Backend,
    MockSAM2Backend,
    SAM2FrameResult,
    SAM2InstanceMask,
    SAM2Prompt,
    SAM2PromptType,
    SAM2SequenceResult,
    compute_mask_iou,
    fuse_sam2_cryptomatte,
)
from synapse.errors import BackendUnavailableError, ContractValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T, H, W = 8, 32, 32


def _make_video() -> np.ndarray:
    """Create a synthetic test video [T, H, W, 3] uint8."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (T, H, W, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# SAM2PromptType
# ---------------------------------------------------------------------------


class TestPromptType:
    """Prompt type enum values."""

    def test_values(self) -> None:
        assert SAM2PromptType.POINT.value == "point"
        assert SAM2PromptType.BOX.value == "box"
        assert SAM2PromptType.MASK.value == "mask"
        assert SAM2PromptType.AUTO.value == "auto"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Dataclass construction."""

    def test_prompt_defaults(self) -> None:
        p = SAM2Prompt()
        assert p.prompt_type == SAM2PromptType.AUTO
        assert p.instance_id == 0

    def test_prompt_with_points(self) -> None:
        p = SAM2Prompt(
            prompt_type=SAM2PromptType.POINT,
            points=np.array([[10.0, 20.0]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            instance_id=1,
        )
        assert p.points is not None
        assert p.points.shape == (1, 2)

    def test_instance_mask(self) -> None:
        mask = np.ones((H, W), dtype=bool)
        inst = SAM2InstanceMask(instance_id=0, mask=mask, confidence=0.95)
        assert inst.mask.shape == (H, W)
        assert not inst.is_occluded

    def test_frame_result_combined_mask(self) -> None:
        m1 = np.zeros((H, W), dtype=bool)
        m1[:H // 2, :] = True
        m2 = np.zeros((H, W), dtype=bool)
        m2[H // 2:, :] = True
        fr = SAM2FrameResult(
            frame_index=0,
            instances=[
                SAM2InstanceMask(instance_id=0, mask=m1),
                SAM2InstanceMask(instance_id=1, mask=m2),
            ],
        )
        combined = fr.combined_mask
        assert combined is not None
        assert np.all(combined)  # Full image covered

    def test_frame_result_empty(self) -> None:
        fr = SAM2FrameResult(frame_index=0)
        assert fr.combined_mask is None

    def test_sequence_result(self) -> None:
        sr = SAM2SequenceResult(num_instances=2, scene_cuts=[10, 50])
        assert sr.num_instances == 2
        assert len(sr.scene_cuts) == 2


# ---------------------------------------------------------------------------
# MockSAM2Backend
# ---------------------------------------------------------------------------


class TestMockBackend:
    """MockSAM2Backend produces deterministic results."""

    def test_always_available(self) -> None:
        assert MockSAM2Backend().is_available()

    def test_segment_video_auto(self) -> None:
        backend = MockSAM2Backend()
        video = _make_video()
        result = backend.segment_video(video)
        assert len(result.frames) == T
        assert result.num_instances == 1
        assert result.metadata["backend"] == "mock"

    def test_segment_video_with_prompts(self) -> None:
        backend = MockSAM2Backend()
        prompts = [
            SAM2Prompt(prompt_type=SAM2PromptType.POINT, instance_id=0),
            SAM2Prompt(prompt_type=SAM2PromptType.BOX, instance_id=1),
        ]
        result = backend.segment_video(_make_video(), prompts=prompts)
        assert result.num_instances == 2
        for frame in result.frames:
            assert len(frame.instances) == 2

    def test_masks_are_boolean(self) -> None:
        backend = MockSAM2Backend()
        result = backend.segment_video(_make_video())
        for frame in result.frames:
            for inst in frame.instances:
                assert inst.mask.dtype == bool
                assert inst.mask.shape == (H, W)

    def test_masks_have_nonzero_coverage(self) -> None:
        backend = MockSAM2Backend()
        result = backend.segment_video(_make_video())
        for frame in result.frames:
            for inst in frame.instances:
                assert np.any(inst.mask), f"Empty mask at frame {frame.frame_index}"

    def test_deterministic(self) -> None:
        backend = MockSAM2Backend()
        video = _make_video()
        r1 = backend.segment_video(video)
        r2 = backend.segment_video(video)
        for f1, f2 in zip(r1.frames, r2.frames, strict=True):
            for i1, i2 in zip(f1.instances, f2.instances, strict=True):
                np.testing.assert_array_equal(i1.mask, i2.mask)

    def test_confidence_positive(self) -> None:
        backend = MockSAM2Backend()
        result = backend.segment_video(_make_video())
        for frame in result.frames:
            for inst in frame.instances:
                assert inst.confidence > 0.0

    def test_invalid_input_shape_raises(self) -> None:
        backend = MockSAM2Backend()
        with pytest.raises(ContractValidationError, match="Expected"):
            backend.segment_video(np.zeros((H, W, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# FoundarySAM2Backend
# ---------------------------------------------------------------------------


class TestFoundaryBackend:
    """FoundarySAM2Backend raises when unconnected."""

    def test_not_available(self) -> None:
        assert not FoundarySAM2Backend().is_available()

    def test_segment_raises(self) -> None:
        backend = FoundarySAM2Backend()
        with pytest.raises(BackendUnavailableError):
            backend.segment_video(_make_video())


# ---------------------------------------------------------------------------
# Cryptomatte fusion
# ---------------------------------------------------------------------------


class TestCryptomatteFusion:
    """SAM2 ⊕ Cryptomatte mask fusion."""

    def test_union_of_masks(self) -> None:
        sam2 = np.zeros((H, W), dtype=bool)
        sam2[:H // 2, :] = True
        crypto = np.zeros((H, W), dtype=np.uint32)
        crypto[H // 2:, :] = 1
        fused = fuse_sam2_cryptomatte(sam2, crypto, dilation_px=0)
        # Union should cover entire image
        assert np.all(fused)

    def test_target_instance_filtering(self) -> None:
        sam2 = np.zeros((H, W), dtype=bool)
        crypto = np.zeros((H, W), dtype=np.uint32)
        crypto[:H // 2, :] = 1  # Instance 1
        crypto[H // 2:, :] = 2  # Instance 2
        fused = fuse_sam2_cryptomatte(
            sam2, crypto, target_instance_ids=[1], dilation_px=0,
        )
        # Only instance 1 region should be True
        assert np.all(fused[:H // 2, :])
        assert not np.any(fused[H // 2:, :])

    def test_dilation_expands_mask(self) -> None:
        sam2 = np.zeros((H, W), dtype=bool)
        sam2[H // 2, W // 2] = True
        crypto = np.zeros((H, W), dtype=np.uint32)
        no_dilate = fuse_sam2_cryptomatte(sam2, crypto, dilation_px=0)
        with_dilate = fuse_sam2_cryptomatte(sam2, crypto, dilation_px=3)
        assert np.sum(with_dilate) > np.sum(no_dilate)

    def test_shape_mismatch_raises(self) -> None:
        sam2 = np.zeros((H, W), dtype=bool)
        crypto = np.zeros((H + 1, W), dtype=np.uint32)
        with pytest.raises(ContractValidationError, match="shape"):
            fuse_sam2_cryptomatte(sam2, crypto)

    def test_all_zero_crypto(self) -> None:
        """Zero Cryptomatte with SAM2 mask → result is SAM2 mask."""
        sam2 = np.zeros((H, W), dtype=bool)
        sam2[5:10, 5:10] = True
        crypto = np.zeros((H, W), dtype=np.uint32)
        fused = fuse_sam2_cryptomatte(sam2, crypto, dilation_px=0)
        np.testing.assert_array_equal(fused, sam2)


# ---------------------------------------------------------------------------
# IoU metric
# ---------------------------------------------------------------------------


class TestMaskIoU:
    """compute_mask_iou returns correct IoU values."""

    def test_identical_masks(self) -> None:
        mask = np.ones((H, W), dtype=bool)
        assert compute_mask_iou(mask, mask) == 1.0

    def test_disjoint_masks(self) -> None:
        a = np.zeros((H, W), dtype=bool)
        b = np.zeros((H, W), dtype=bool)
        a[:H // 2, :] = True
        b[H // 2:, :] = True
        assert compute_mask_iou(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = np.zeros((H, W), dtype=bool)
        b = np.zeros((H, W), dtype=bool)
        a[:20, :] = True
        b[10:, :] = True
        iou = compute_mask_iou(a, b)
        # Overlap = 10 rows, union = 32 rows
        expected = (10 * W) / (32 * W)
        assert abs(iou - expected) < 0.01

    def test_empty_masks(self) -> None:
        a = np.zeros((H, W), dtype=bool)
        b = np.zeros((H, W), dtype=bool)
        assert compute_mask_iou(a, b) == 0.0
