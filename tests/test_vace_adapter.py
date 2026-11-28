"""Tests for the Wan2.1-VACE unified adapter (PR-216 / PR-217).

Covers:
- VaceAdapter registration and identity
- VCU payload construction for R2V / V2V / MV2V modes
- MockVaceBackend deterministic output
- Condition mode selection (FGPP_7CH vs VCU_UNIFIED)
- Spatiotemporal mask generation from Cryptomatte IDs
- Reference frame stacking and validation
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.adapters.base import (
    PreparedConditionPayload,
    VideoGenerationConfig,
    create_adapter,
    list_adapters,
)
from synapse.adapters.vace import (
    ConditionMode,
    MockVaceBackend,
    VaceAdapter,
    VaceTaskMode,
    VCUPayload,
    _build_spatiotemporal_mask,
    _build_vcu_reference_tensor,
)
from synapse.errors import BackendUnavailableError, ContractValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fgpp_sequence(
    n_frames: int = 4,
    height: int = 32,
    width: int = 32,
    *,
    with_cryptomatte: bool = True,
) -> object:
    """Create a minimal FGPPSequence-like object for testing."""
    from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

    frames = []
    for i in range(n_frames):
        cam = CameraParams(
            fx=500.0,
            fy=500.0,
            cx=float(width // 2),
            cy=float(height // 2),
            width=width,
            height=height,
            view_matrix=np.eye(4, dtype=np.float64),
            z_near=0.1,
            z_far=100.0,
        )
        crypto = None
        if with_cryptomatte:
            crypto = np.full((height, width), i + 1, dtype=np.uint32)
            # Set some background pixels
            crypto[:height // 4, :] = 0

        frame = FGPPFrame(
            camera=cam,
            depth=np.full((height, width), 5.0 + i * 0.1, dtype=np.float32),
            world_normal=np.tile(
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                (height, width, 1),
            ),
            direct_radiance=np.full(
                (height, width, 3), 0.5, dtype=np.float32,
            ),
            base_color=np.full(
                (height, width, 3), 0.7, dtype=np.float32,
            ),
            roughness_metallic=np.zeros(
                (height, width, 2), dtype=np.float32,
            ),
            motion_vector=np.zeros(
                (height, width, 2), dtype=np.float32,
            ),
            cryptomatte_id=crypto,
            frame_index=i,
        )
        frames.append(frame)

    return FGPPSequence(frames=frames, fps=24.0)


@pytest.fixture
def fgpp_seq() -> object:
    """4-frame FGPP sequence with Cryptomatte."""
    return _make_fgpp_sequence(n_frames=4)


@pytest.fixture
def config() -> VideoGenerationConfig:
    """Standard generation config for testing."""
    return VideoGenerationConfig(
        seed=42,
        fps=24.0,
        width=32,
        height=32,
        num_frames=4,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestVaceAdapterRegistration:
    """VaceAdapter is correctly registered in the global adapter registry."""

    def test_adapter_id(self) -> None:
        adapter = VaceAdapter()
        assert adapter.adapter_id == "vace"

    def test_tier(self) -> None:
        adapter = VaceAdapter()
        assert adapter.tier == "whitebox"

    def test_listed_in_registry(self) -> None:
        adapters = list_adapters()
        assert "vace" in adapters

    def test_create_by_name(self) -> None:
        adapter = create_adapter("vace")
        assert isinstance(adapter, VaceAdapter)


# ---------------------------------------------------------------------------
# ConditionMode and VaceTaskMode enums
# ---------------------------------------------------------------------------


class TestEnums:
    """Enum values are correctly defined."""

    def test_condition_mode_values(self) -> None:
        assert ConditionMode.FGPP_7CH.value == "fgpp_7ch"
        assert ConditionMode.VCU_UNIFIED.value == "vcu_unified"

    def test_vace_task_mode_values(self) -> None:
        assert VaceTaskMode.REFERENCE_TO_VIDEO.value == "r2v"
        assert VaceTaskMode.VIDEO_TO_VIDEO.value == "v2v"
        assert VaceTaskMode.MASKED_VIDEO_TO_VIDEO.value == "mv2v"


# ---------------------------------------------------------------------------
# VCU Payload construction
# ---------------------------------------------------------------------------


class TestVCUPayloadConstruction:
    """VCU payload is correctly built from FGPP sequences."""

    def test_r2v_with_references(self, fgpp_seq: object, config: VideoGenerationConfig) -> None:
        ref_imgs = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
        adapter = VaceAdapter(
            backend=MockVaceBackend(),
            task_mode=VaceTaskMode.REFERENCE_TO_VIDEO,
            reference_images=ref_imgs,
        )
        vcu = adapter.build_vcu_payload(fgpp_seq, config)  # type: ignore[arg-type]
        assert vcu.task_mode == VaceTaskMode.REFERENCE_TO_VIDEO
        assert vcu.reference_frames is not None
        assert vcu.reference_frames.shape == (2, 32, 32, 3)
        assert vcu.spatiotemporal_mask is None

    def test_v2v_no_mask_no_reference(
        self, fgpp_seq: object, config: VideoGenerationConfig
    ) -> None:
        adapter = VaceAdapter(
            backend=MockVaceBackend(),
            task_mode=VaceTaskMode.VIDEO_TO_VIDEO,
        )
        vcu = adapter.build_vcu_payload(fgpp_seq, config)  # type: ignore[arg-type]
        assert vcu.task_mode == VaceTaskMode.VIDEO_TO_VIDEO
        assert vcu.reference_frames is None
        assert vcu.spatiotemporal_mask is None

    def test_mv2v_builds_mask(self, fgpp_seq: object, config: VideoGenerationConfig) -> None:
        adapter = VaceAdapter(
            backend=MockVaceBackend(),
            task_mode=VaceTaskMode.MASKED_VIDEO_TO_VIDEO,
        )
        vcu = adapter.build_vcu_payload(fgpp_seq, config)  # type: ignore[arg-type]
        assert vcu.task_mode == VaceTaskMode.MASKED_VIDEO_TO_VIDEO
        assert vcu.spatiotemporal_mask is not None
        assert vcu.spatiotemporal_mask.shape == (4, 32, 32)
        assert vcu.spatiotemporal_mask.dtype == np.float32
        # Top quarter should be 0 (background), rest should be 1
        assert np.all(vcu.spatiotemporal_mask[:, :8, :] == 0.0)
        assert np.all(vcu.spatiotemporal_mask[:, 8:, :] == 1.0)

    def test_mv2v_target_entity_ids(
        self, fgpp_seq: object, config: VideoGenerationConfig
    ) -> None:
        adapter = VaceAdapter(
            backend=MockVaceBackend(),
            task_mode=VaceTaskMode.MASKED_VIDEO_TO_VIDEO,
            target_entity_ids=[2],  # Only entity ID 2
        )
        vcu = adapter.build_vcu_payload(fgpp_seq, config)  # type: ignore[arg-type]
        assert vcu.spatiotemporal_mask is not None
        # Only frame 1 (entity ID=2) should have non-zero mask in fg area
        assert np.any(vcu.spatiotemporal_mask[1, 8:, :] > 0)
        # Frame 0 (entity ID=1) should be all zeros
        assert np.all(vcu.spatiotemporal_mask[0] == 0.0)


# ---------------------------------------------------------------------------
# prepare_conditions
# ---------------------------------------------------------------------------


class TestPrepareConditions:
    """prepare_conditions embeds VCU payload correctly."""

    def test_vcu_in_extra(self, fgpp_seq: object, config: VideoGenerationConfig) -> None:
        adapter = VaceAdapter(backend=MockVaceBackend())
        payload = adapter.prepare_conditions(fgpp_seq, config)  # type: ignore[arg-type]
        assert "vcu_payload" in payload.extra
        assert isinstance(payload.extra["vcu_payload"], VCUPayload)
        assert payload.extra["condition_mode"] == "vcu_unified"
        assert payload.extra["task_mode"] == "r2v"

    def test_cryptomatte_passthrough(
        self, fgpp_seq: object, config: VideoGenerationConfig
    ) -> None:
        adapter = VaceAdapter(backend=MockVaceBackend())
        payload = adapter.prepare_conditions(fgpp_seq, config)  # type: ignore[arg-type]
        assert payload.cryptomatte_id is not None
        assert payload.cryptomatte_id.shape == (4, 32, 32)
        assert payload.cryptomatte_id.dtype == np.uint32


# ---------------------------------------------------------------------------
# MockVaceBackend
# ---------------------------------------------------------------------------


class TestMockVaceBackend:
    """MockVaceBackend produces deterministic, mode-encoded output."""

    def test_deterministic_output(self, config: VideoGenerationConfig) -> None:
        backend = MockVaceBackend()
        vcu = VCUPayload(task_mode=VaceTaskMode.REFERENCE_TO_VIDEO)
        conditions = PreparedConditionPayload()
        frames1 = backend.submit(vcu, conditions, config)
        frames2 = backend.submit(vcu, conditions, config)
        np.testing.assert_array_equal(frames1, frames2)

    def test_output_shape(self, config: VideoGenerationConfig) -> None:
        backend = MockVaceBackend()
        vcu = VCUPayload(task_mode=VaceTaskMode.VIDEO_TO_VIDEO)
        conditions = PreparedConditionPayload()
        frames = backend.submit(vcu, conditions, config)
        assert frames.shape == (4, 32, 32, 3)
        assert frames.dtype == np.uint8

    def test_mode_encoded_in_blue_channel(self, config: VideoGenerationConfig) -> None:
        backend = MockVaceBackend()
        conditions = PreparedConditionPayload()

        for mode, expected_blue in [
            (VaceTaskMode.REFERENCE_TO_VIDEO, 64),
            (VaceTaskMode.VIDEO_TO_VIDEO, 128),
            (VaceTaskMode.MASKED_VIDEO_TO_VIDEO, 192),
        ]:
            vcu = VCUPayload(task_mode=mode)
            frames = backend.submit(vcu, conditions, config)
            assert frames[0, 0, 0, 2] == expected_blue, f"Mode {mode} blue mismatch"

    def test_always_available(self) -> None:
        assert MockVaceBackend().is_available()


# ---------------------------------------------------------------------------
# generate end-to-end
# ---------------------------------------------------------------------------


class TestGenerateEndToEnd:
    """Full generate() flow through VaceAdapter."""

    def test_r2v_generate(self, fgpp_seq: object, config: VideoGenerationConfig) -> None:
        adapter = VaceAdapter(
            backend=MockVaceBackend(),
            task_mode=VaceTaskMode.REFERENCE_TO_VIDEO,
        )
        payload = adapter.prepare_conditions(fgpp_seq, config)  # type: ignore[arg-type]
        result = adapter.generate(payload, config)
        assert result.frames_srgb_u8.shape == (4, 32, 32, 3)
        assert result.adapter_id == "vace"
        assert result.metadata["task_mode"] == "r2v"

    def test_mv2v_generate(self, fgpp_seq: object, config: VideoGenerationConfig) -> None:
        adapter = VaceAdapter(
            backend=MockVaceBackend(),
            task_mode=VaceTaskMode.MASKED_VIDEO_TO_VIDEO,
        )
        payload = adapter.prepare_conditions(fgpp_seq, config)  # type: ignore[arg-type]
        result = adapter.generate(payload, config)
        assert result.metadata["task_mode"] == "mv2v"
        assert result.metadata["has_mask"]

    def test_no_backend_raises(self, fgpp_seq: object, config: VideoGenerationConfig) -> None:
        adapter = VaceAdapter(backend=None)
        payload = adapter.prepare_conditions(fgpp_seq, config)  # type: ignore[arg-type]
        with pytest.raises(BackendUnavailableError):
            adapter.generate(payload, config)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for VCU helper functions."""

    def test_build_reference_tensor_consistent(self) -> None:
        imgs = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
        result = _build_vcu_reference_tensor(imgs)
        assert result.shape == (3, 16, 16, 3)

    def test_build_reference_tensor_empty(self) -> None:
        result = _build_vcu_reference_tensor([])
        assert result.shape[0] == 0

    def test_build_reference_tensor_inconsistent_raises(self) -> None:
        imgs = [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
        ]
        with pytest.raises(ContractValidationError):
            _build_vcu_reference_tensor(imgs)

    def test_build_spatiotemporal_mask_all_foreground(self) -> None:
        crypto = np.ones((4, 8, 8), dtype=np.uint32) * 5
        mask = _build_spatiotemporal_mask(crypto)
        assert mask is not None
        assert np.all(mask == 1.0)

    def test_build_spatiotemporal_mask_target_ids(self) -> None:
        crypto = np.zeros((4, 8, 8), dtype=np.uint32)
        crypto[0, :4, :4] = 1
        crypto[1, :4, :4] = 2
        mask = _build_spatiotemporal_mask(crypto, target_entity_ids=[1])
        assert mask is not None
        assert np.all(mask[0, :4, :4] == 1.0)
        assert np.all(mask[1] == 0.0)

    def test_build_spatiotemporal_mask_none_input(self) -> None:
        assert _build_spatiotemporal_mask(None) is None
