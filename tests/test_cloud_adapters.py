"""Tests for Tier-2 blackbox cloud adapters with FGPP anchor degradation.

Tests cover:
- ``degrade_fgpp_for_blackbox`` segment splitting and anchor extraction
- ``MockCloudTransport`` determinism and validation
- Cloud adapter registry, generation, and concatenation
- Idempotency key determinism
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.adapters.base import (
    VideoGenerationConfig,
    create_adapter,
    list_adapters,
)
from synapse.adapters.cloud import (
    DegradedSegment,
    KlingAdapter,
    MockCloudTransport,
    SeedanceAdapter,
    VeoAdapter,
    _compute_idempotency_key,
    degrade_fgpp_for_blackbox,
)
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_camera(
    frame_idx: int,
    total_frames: int,
) -> CameraParams:
    """Create a camera with slight lateral pan over time."""
    # Simulate a gentle pan: translate X linearly
    t = frame_idx / max(total_frames - 1, 1)
    world_mat = np.eye(4, dtype=np.float64)
    world_mat[0, 3] = t * 2.0  # pan right by 2 units over sequence
    view_mat = np.linalg.inv(world_mat)
    return CameraParams(
        fx=500.0, fy=500.0, cx=32.0, cy=32.0,
        width=64, height=64,
        view_matrix=view_mat,
        world_matrix=world_mat,
    )


def _make_fgpp_sequence(
    num_frames: int,
    h: int = 64,
    w: int = 64,
) -> FGPPSequence:
    """Build a synthetic FGPP sequence with all channels populated."""
    rng = np.random.Generator(np.random.PCG64(12345))
    frames: list[FGPPFrame] = []
    for i in range(num_frames):
        cam = _make_camera(i, num_frames)
        frame = FGPPFrame(
            camera=cam,
            depth=rng.uniform(0.1, 100.0, (h, w)).astype(np.float32),
            direct_radiance=rng.uniform(
                0.0, 2.0, (h, w, 3)
            ).astype(np.float16),
            world_normal=_random_normals(rng, h, w),
            base_color=rng.uniform(
                0.0, 1.0, (h, w, 3)
            ).astype(np.float16),
            frame_index=i,
        )
        frames.append(frame)
    return FGPPSequence(frames=frames, fps=24.0, scene_id="test_scene")


def _random_normals(
    rng: np.random.Generator,
    h: int,
    w: int,
) -> np.ndarray:
    """Generate random unit normals ``[H, W, 3]`` float16."""
    raw = rng.standard_normal((h, w, 3))
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (raw / norms).astype(np.float16)


# ---------------------------------------------------------------------------
# TestDegradeFGPP
# ---------------------------------------------------------------------------


class TestDegradeFGPP:
    """Tests for ``degrade_fgpp_for_blackbox``."""

    def test_97_frames_split_into_2_segments(self) -> None:
        """97 frames -> anchors at 0, 48, 96 -> 2 segments."""
        seq = _make_fgpp_sequence(97)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)

        assert len(segments) == 2
        assert segments[0].start_frame == 0
        assert segments[0].end_frame == 48
        assert segments[1].start_frame == 48
        assert segments[1].end_frame == 96

    def test_segments_share_anchor_frame_48(self) -> None:
        """Frame 48 is the end of segment 0 and start of segment 1."""
        seq = _make_fgpp_sequence(97)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)

        # Segment 0 ends at 48, segment 1 starts at 48
        assert segments[0].end_frame == segments[1].start_frame

    def test_merge_total_frames_no_duplicates(self) -> None:
        """After merging with first-frame removal, total = 97."""
        seq = _make_fgpp_sequence(97)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)

        config = VideoGenerationConfig(
            seed=42, width=64, height=64, num_frames=97,
        )
        transport = MockCloudTransport()

        all_frames: list[np.ndarray] = []
        for i, seg in enumerate(segments):
            frames = transport.submit_segment(seg, config)
            if i == 0:
                all_frames.append(frames)
            else:
                all_frames.append(frames[1:])

        result = np.concatenate(all_frames, axis=0)
        assert result.shape[0] == 97

    def test_anchor_pngs_non_empty(self) -> None:
        """First and last anchor PNGs are non-empty bytes."""
        seq = _make_fgpp_sequence(97)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)

        for seg in segments:
            assert isinstance(seg.first_anchor_png, bytes)
            assert len(seg.first_anchor_png) > 0
            assert isinstance(seg.last_anchor_png, bytes)
            assert len(seg.last_anchor_png) > 0

    def test_camera_motion_enum_valid(self) -> None:
        """Camera motion enum is a valid string."""
        seq = _make_fgpp_sequence(97)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)

        valid_motions = {
            "pan_right", "pan_left",
            "dolly_in", "dolly_out",
            "orbit_cw", "orbit_ccw",
            "static",
        }
        for seg in segments:
            assert seg.camera_motion_enum in valid_motions

    def test_motion_intensity_in_range(self) -> None:
        """Motion intensity is in [0, 10]."""
        seq = _make_fgpp_sequence(97)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)

        for seg in segments:
            assert 0.0 <= seg.motion_intensity <= 10.0

    def test_trajectory_shape(self) -> None:
        """Camera trajectory has correct shape [T_seg, 4, 4]."""
        seq = _make_fgpp_sequence(97)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)

        # Segment 0: frames 0..48 inclusive = 49 frames
        assert segments[0].camera_trajectory.shape == (49, 4, 4)
        # Segment 1: frames 48..96 inclusive = 49 frames
        assert segments[1].camera_trajectory.shape == (49, 4, 4)

    def test_single_frame_returns_empty(self) -> None:
        """A single-frame sequence has no segments (no interval)."""
        seq = _make_fgpp_sequence(1)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)
        # Only 1 anchor point -> no pair -> 0 segments
        assert len(segments) == 0

    def test_depth_and_normal_guides_present(self) -> None:
        """Depth and normal edge guides are populated."""
        seq = _make_fgpp_sequence(97)
        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)

        for seg in segments:
            assert seg.depth_guide is not None
            assert seg.depth_guide.shape[0] == (
                seg.end_frame - seg.start_frame + 1
            )
            assert seg.normal_edge_guide is not None
            assert seg.normal_edge_guide.shape[0] == (
                seg.end_frame - seg.start_frame + 1
            )

    def test_static_camera_returns_static_enum(self) -> None:
        """A sequence with identical cameras returns 'static'."""
        seq = _make_fgpp_sequence(50)
        # Override all cameras to identity
        for f in seq.frames:
            f.camera.world_matrix = np.eye(4, dtype=np.float64)
            f.camera.view_matrix = np.eye(4, dtype=np.float64)

        segments = degrade_fgpp_for_blackbox(seq, anchor_interval=48)
        if segments:
            assert segments[0].camera_motion_enum == "static"
            assert segments[0].motion_intensity == 0.0


# ---------------------------------------------------------------------------
# TestMockCloudTransport
# ---------------------------------------------------------------------------


class TestMockCloudTransport:
    """Tests for ``MockCloudTransport``."""

    def test_returns_correct_shape(self) -> None:
        """Output shape matches segment frames and config resolution."""
        transport = MockCloudTransport()
        seg = DegradedSegment(
            start_frame=0,
            end_frame=48,
            first_anchor_png=b"\x89PNG_fake",
            last_anchor_png=b"\x89PNG_fake",
            camera_trajectory=np.eye(4, dtype=np.float64)[
                np.newaxis
            ].repeat(49, axis=0),
            camera_motion_enum="static",
            motion_intensity=0.0,
        )
        config = VideoGenerationConfig(
            seed=42, width=64, height=64,
        )
        result = transport.submit_segment(seg, config)
        assert result.shape == (49, 64, 64, 3)
        assert result.dtype == np.uint8

    def test_deterministic_same_seed(self) -> None:
        """Same seed + segment produces identical output."""
        transport = MockCloudTransport()
        seg = DegradedSegment(
            start_frame=0,
            end_frame=10,
            first_anchor_png=b"anchor1",
            last_anchor_png=b"anchor2",
            camera_trajectory=np.eye(4, dtype=np.float64)[
                np.newaxis
            ].repeat(11, axis=0),
            camera_motion_enum="pan_right",
            motion_intensity=5.0,
        )
        config = VideoGenerationConfig(seed=99, width=32, height=32)

        r1 = transport.submit_segment(seg, config)
        r2 = transport.submit_segment(seg, config)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_output(self) -> None:
        """Different seeds produce different outputs."""
        transport = MockCloudTransport()
        seg = DegradedSegment(
            start_frame=0,
            end_frame=10,
            first_anchor_png=b"anchor1",
            last_anchor_png=b"anchor2",
            camera_trajectory=np.eye(4, dtype=np.float64)[
                np.newaxis
            ].repeat(11, axis=0),
            camera_motion_enum="pan_right",
            motion_intensity=5.0,
        )
        config1 = VideoGenerationConfig(seed=1, width=32, height=32)
        config2 = VideoGenerationConfig(seed=2, width=32, height=32)

        r1 = transport.submit_segment(seg, config1)
        r2 = transport.submit_segment(seg, config2)
        assert not np.array_equal(r1, r2)

    def test_is_available(self) -> None:
        """Mock transport is always available."""
        assert MockCloudTransport().is_available()


# ---------------------------------------------------------------------------
# TestCloudAdapters
# ---------------------------------------------------------------------------


class TestCloudAdapters:
    """Tests for SeedanceAdapter, KlingAdapter, VeoAdapter."""

    def test_all_three_in_registry(self) -> None:
        """All 3 adapters are registered."""
        adapters = list_adapters()
        assert "seedance2.5" in adapters
        assert "kling1.5" in adapters
        assert "veo" in adapters

    def test_create_adapters_by_name(self) -> None:
        """Adapters can be instantiated from the registry."""
        for name in ("seedance2.5", "kling1.5", "veo"):
            adapter = create_adapter(name)
            assert adapter.adapter_id == name
            assert adapter.tier == "blackbox"

    @pytest.mark.parametrize(
        "adapter_cls",
        [SeedanceAdapter, KlingAdapter, VeoAdapter],
    )
    def test_generate_correct_total_frames(
        self,
        adapter_cls: type,
    ) -> None:
        """Each adapter generates correct total frames (97)."""
        transport = MockCloudTransport()
        adapter = adapter_cls(transport=transport)

        seq = _make_fgpp_sequence(97)
        config = VideoGenerationConfig(
            seed=42, width=64, height=64, num_frames=97,
        )

        conditions = adapter.prepare_conditions(seq, config)
        result = adapter.generate(conditions, config)

        assert result.frames_srgb_u8.shape[0] == 97
        assert result.frames_srgb_u8.shape[1] == 64
        assert result.frames_srgb_u8.shape[2] == 64
        assert result.frames_srgb_u8.shape[3] == 3
        assert result.frames_srgb_u8.dtype == np.uint8

    def test_concatenation_removes_duplicates(self) -> None:
        """Concatenation of 2 segments of 49 frames -> 97 total."""
        transport = MockCloudTransport()
        adapter = SeedanceAdapter(transport=transport)

        seq = _make_fgpp_sequence(97)
        config = VideoGenerationConfig(
            seed=42, width=64, height=64, num_frames=97,
        )

        conditions = adapter.prepare_conditions(seq, config)
        segments = conditions.extra["degraded_segments"]
        assert len(segments) == 2

        result = adapter.generate(conditions, config)
        # 49 + 49 - 1 (duplicate anchor at frame 48) = 97
        assert result.frames_srgb_u8.shape[0] == 97

    def test_seedance_supports_all_guides(self) -> None:
        """Seedance adapter preserves depth and normal guides."""
        transport = MockCloudTransport()
        adapter = SeedanceAdapter(transport=transport)

        seq = _make_fgpp_sequence(97)
        config = VideoGenerationConfig(
            seed=42, width=64, height=64, num_frames=97,
        )
        conditions = adapter.prepare_conditions(seq, config)
        segments = conditions.extra["degraded_segments"]
        for seg in segments:
            assert seg.depth_guide is not None
            assert seg.normal_edge_guide is not None

    def test_kling_strips_all_guides(self) -> None:
        """Kling adapter strips both depth and normal guides."""
        transport = MockCloudTransport()
        adapter = KlingAdapter(transport=transport)

        seq = _make_fgpp_sequence(97)
        config = VideoGenerationConfig(
            seed=42, width=64, height=64, num_frames=97,
        )
        conditions = adapter.prepare_conditions(seq, config)
        segments = conditions.extra["degraded_segments"]
        for seg in segments:
            assert seg.depth_guide is None
            assert seg.normal_edge_guide is None

    def test_veo_supports_depth_only(self) -> None:
        """Veo adapter keeps depth guide but strips normal guide."""
        transport = MockCloudTransport()
        adapter = VeoAdapter(transport=transport)

        seq = _make_fgpp_sequence(97)
        config = VideoGenerationConfig(
            seed=42, width=64, height=64, num_frames=97,
        )
        conditions = adapter.prepare_conditions(seq, config)
        segments = conditions.extra["degraded_segments"]
        for seg in segments:
            assert seg.depth_guide is not None
            assert seg.normal_edge_guide is None

    def test_missing_token_raises_backend_error(self) -> None:
        """Missing API token raises BackendUnavailableError."""
        from synapse.errors import BackendUnavailableError

        adapter = SeedanceAdapter()  # no transport, no token
        seq = _make_fgpp_sequence(50)
        config = VideoGenerationConfig(
            seed=42, width=64, height=64, num_frames=50,
        )
        conditions = adapter.prepare_conditions(seq, config)

        with pytest.raises(BackendUnavailableError) as exc_info:
            adapter.generate(conditions, config)
        assert exc_info.value.exit_code == 30
        assert exc_info.value.agent_fix_hint


# ---------------------------------------------------------------------------
# TestIdempotencyKey
# ---------------------------------------------------------------------------


class TestIdempotencyKey:
    """Tests for idempotency key computation."""

    def test_same_inputs_same_key(self) -> None:
        """Same inputs produce the same key."""
        k1 = _compute_idempotency_key("scene_a", 0, 48, 42)
        k2 = _compute_idempotency_key("scene_a", 0, 48, 42)
        assert k1 == k2

    def test_different_seed_different_key(self) -> None:
        """Different seeds produce different keys."""
        k1 = _compute_idempotency_key("scene_a", 0, 48, 42)
        k2 = _compute_idempotency_key("scene_a", 0, 48, 99)
        assert k1 != k2

    def test_different_scene_different_key(self) -> None:
        """Different scene IDs produce different keys."""
        k1 = _compute_idempotency_key("scene_a", 0, 48, 42)
        k2 = _compute_idempotency_key("scene_b", 0, 48, 42)
        assert k1 != k2

    def test_different_range_different_key(self) -> None:
        """Different frame ranges produce different keys."""
        k1 = _compute_idempotency_key("scene_a", 0, 48, 42)
        k2 = _compute_idempotency_key("scene_a", 48, 96, 42)
        assert k1 != k2

    def test_key_is_hex_sha256(self) -> None:
        """Key is a 64-char hex string (SHA-256)."""
        key = _compute_idempotency_key("scene", 0, 10, 1)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_appears_in_result_metadata(self) -> None:
        """Idempotency keys appear in adapter result metadata."""
        transport = MockCloudTransport()
        adapter = SeedanceAdapter(transport=transport)

        seq = _make_fgpp_sequence(97)
        config = VideoGenerationConfig(
            seed=42, width=64, height=64, num_frames=97,
        )

        conditions = adapter.prepare_conditions(seq, config)
        result = adapter.generate(conditions, config)

        keys = result.metadata.get("idempotency_keys", [])
        assert len(keys) == 2
        for k in keys:
            assert len(k) == 64
