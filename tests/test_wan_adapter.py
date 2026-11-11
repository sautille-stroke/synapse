"""Tests for the Wan 2.1 whitebox adapter (PR-07).

Covers:
* Camera-space normal rotation (identity, 90-deg, background, magnitude)
* Condition preparation (tensor shape, depth range, normal normalisation)
* Generation via ``LocalCallableBackend``
* Backend transport error handling (``BackendUnavailableError``)
* Adapter registry presence
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from synapse.adapters.base import (
    PreparedConditionPayload,
    VideoGenerationConfig,
    VideoResult,
    list_adapters,
)
from synapse.adapters.wan import (
    FoundaryHttpTransport,
    LocalCallableBackend,
    WanAdapter,
    WanBackendTransport,
    _build_control_tensor,
    _world_normal_to_camera,
)
from synapse.errors import BackendUnavailableError
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(
    w: int = 8,
    h: int = 8,
    view_matrix: np.ndarray | None = None,
) -> CameraParams:
    """Create a minimal CameraParams with optional view_matrix override."""
    vm = view_matrix if view_matrix is not None else np.eye(4, dtype=np.float64)
    return CameraParams(
        fx=100.0, fy=100.0,
        cx=w / 2.0, cy=h / 2.0,
        width=w, height=h,
        view_matrix=vm,
        world_matrix=np.linalg.inv(vm),
        z_near=0.1, z_far=100.0,
    )


def _make_fgpp_sequence(
    n_frames: int = 4,
    h: int = 8,
    w: int = 8,
    view_matrix: np.ndarray | None = None,
) -> FGPPSequence:
    """Build a synthetic FGPPSequence with filled channels."""
    rng = np.random.RandomState(42)
    frames: list[FGPPFrame] = []
    for i in range(n_frames):
        cam = _make_camera(w, h, view_matrix)
        depth = rng.uniform(0.5, 50.0, (h, w)).astype(np.float32)
        # Normalised random world normals
        wn = rng.randn(h, w, 3).astype(np.float32)
        mag = np.linalg.norm(wn, axis=-1, keepdims=True)
        mag = np.where(mag < 1e-6, 1.0, mag)
        wn = wn / mag
        radiance = rng.uniform(0.0, 2.0, (h, w, 3)).astype(np.float16)
        frames.append(FGPPFrame(
            camera=cam,
            depth=depth,
            world_normal=wn,
            direct_radiance=radiance,
            frame_index=i,
        ))
    return FGPPSequence(frames=frames, fps=24.0, scene_id="test")


# ===================================================================
# TestWorldToCameraNormal
# ===================================================================


class TestWorldToCameraNormal:
    """Tests for ``_world_normal_to_camera``."""

    def test_identity_preserves_normals(self) -> None:
        """Identity rotation should preserve normals exactly."""
        h, w = 4, 4
        rng = np.random.RandomState(0)
        normals = rng.randn(h, w, 3).astype(np.float32)
        mag = np.linalg.norm(normals, axis=-1, keepdims=True)
        mag = np.where(mag < 1e-6, 1.0, mag)
        normals = normals / mag

        view = np.eye(4, dtype=np.float64)
        result = _world_normal_to_camera(normals, view)

        np.testing.assert_allclose(result, normals, atol=1e-5)

    def test_90_degree_rotation(self) -> None:
        """90-degree rotation about Y should swap X and Z."""
        h, w = 1, 1
        # Normal pointing along +X
        normals = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)

        # 90-deg rotation about Y: X -> -Z, Z -> X
        angle = math.pi / 2.0
        c, s = math.cos(angle), math.sin(angle)
        view = np.eye(4, dtype=np.float64)
        view[0, 0] = c
        view[0, 2] = s
        view[2, 0] = -s
        view[2, 2] = c

        result = _world_normal_to_camera(normals, view)

        # After rotation: (1,0,0) -> (cos90, 0, -sin90) = (0, 0, -1)
        expected = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_zero_normal_becomes_background(self) -> None:
        """Zero-magnitude normals should map to [0, 0, 1]."""
        h, w = 2, 2
        normals = np.zeros((h, w, 3), dtype=np.float32)
        view = np.eye(4, dtype=np.float64)

        result = _world_normal_to_camera(normals, view)

        expected = np.broadcast_to(
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            (h, w, 3),
        )
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_magnitude_preserved(self) -> None:
        """Rotation should preserve unit magnitude (within tolerance)."""
        h, w = 8, 8
        rng = np.random.RandomState(7)
        normals = rng.randn(h, w, 3).astype(np.float32)
        mag = np.linalg.norm(normals, axis=-1, keepdims=True)
        mag = np.where(mag < 1e-6, 1.0, mag)
        normals = normals / mag

        # Arbitrary rotation
        angle = 0.73
        c, s = math.cos(angle), math.sin(angle)
        view = np.eye(4, dtype=np.float64)
        view[0, 0] = c
        view[0, 1] = -s
        view[1, 0] = s
        view[1, 1] = c

        result = _world_normal_to_camera(normals, view)
        result_mag = np.linalg.norm(result, axis=-1)

        np.testing.assert_allclose(
            result_mag, 1.0, atol=1e-3,
            err_msg="Rotated normals should have unit magnitude",
        )

    def test_camera_facing_sphere_center(self) -> None:
        """Camera-facing sphere centre normal should have Nz > 0.99."""
        # Normal pointing towards camera (along -Z in world = +Z in camera)
        normals = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        view = np.eye(4, dtype=np.float64)

        result = _world_normal_to_camera(normals, view)
        assert result[0, 0, 2] > 0.99


# ===================================================================
# TestWanPrepareConditions
# ===================================================================


class TestWanPrepareConditions:
    """Tests for ``WanAdapter.prepare_conditions``."""

    def test_output_tensor_shape(self) -> None:
        """Control tensor shape should be [C_cond, T, H, W]."""
        seq = _make_fgpp_sequence(n_frames=4, h=8, w=8)
        adapter = WanAdapter()
        config = VideoGenerationConfig(
            num_frames=4, width=8, height=8,
        )
        payload = adapter.prepare_conditions(seq, config)

        assert payload.control_tensor is not None
        # C_cond = 7 (1 depth + 3 normal + 3 radiance)
        assert payload.control_tensor.shape == (7, 4, 8, 8)

    def test_depth_values_in_unit_range(self) -> None:
        """Depth channel should be normalised to [0, 1]."""
        seq = _make_fgpp_sequence(n_frames=2, h=4, w=4)
        adapter = WanAdapter()
        config = VideoGenerationConfig(num_frames=2, width=4, height=4)
        payload = adapter.prepare_conditions(seq, config)

        assert payload.control_tensor is not None
        depth_channel = payload.control_tensor[0]  # [T, H, W]
        assert depth_channel.min() >= 0.0
        assert depth_channel.max() <= 1.0

    def test_normals_are_normalised(self) -> None:
        """Camera-space normals should have unit magnitude."""
        seq = _make_fgpp_sequence(n_frames=2, h=4, w=4)
        adapter = WanAdapter()
        config = VideoGenerationConfig(num_frames=2, width=4, height=4)
        payload = adapter.prepare_conditions(seq, config)

        assert payload.control_tensor is not None
        # Normal channels: indices 1, 2, 3 -> [3, T, H, W]
        normals = payload.control_tensor[1:4]  # [3, T, H, W]
        # Transpose to [T, H, W, 3] to compute magnitude per pixel
        normals_thw3 = np.transpose(normals, (1, 2, 3, 0))
        mag = np.linalg.norm(normals_thw3, axis=-1)
        np.testing.assert_allclose(
            mag, 1.0, atol=1e-2,
            err_msg="Camera-space normals should be unit length",
        )

    def test_default_flow_bias_zero(self) -> None:
        """Default gamma_flow_bias should be 0.0."""
        adapter = WanAdapter()
        assert adapter.gamma_flow_bias == 0.0

        seq = _make_fgpp_sequence(n_frames=1, h=4, w=4)
        config = VideoGenerationConfig(num_frames=1, width=4, height=4)
        payload = adapter.prepare_conditions(seq, config)
        assert payload.extra["flow_bias_gamma"] == 0.0

    def test_flow_bias_clamped(self) -> None:
        """gamma_flow_bias should be clamped to [0.0, 2.5]."""
        adapter_high = WanAdapter(gamma_flow_bias=5.0)
        assert adapter_high.gamma_flow_bias == 2.5

        adapter_neg = WanAdapter(gamma_flow_bias=-1.0)
        assert adapter_neg.gamma_flow_bias == 0.0

    def test_empty_sequence(self) -> None:
        """Empty FGPP sequence should return empty payload."""
        seq = FGPPSequence(frames=[], fps=24.0)
        adapter = WanAdapter()
        config = VideoGenerationConfig()
        payload = adapter.prepare_conditions(seq, config)
        assert payload.control_tensor is None

    def test_missing_channels_fallback(self) -> None:
        """Frames with None radiance/normal should not crash."""
        cam = _make_camera(4, 4)
        frame = FGPPFrame(
            camera=cam,
            depth=np.ones((4, 4), dtype=np.float32),
            world_normal=None,
            direct_radiance=None,
            frame_index=0,
        )
        seq = FGPPSequence(frames=[frame], fps=24.0)
        adapter = WanAdapter()
        config = VideoGenerationConfig(num_frames=1, width=4, height=4)
        payload = adapter.prepare_conditions(seq, config)
        assert payload.control_tensor is not None
        assert payload.control_tensor.shape == (7, 1, 4, 4)


# ===================================================================
# TestWanGenerate
# ===================================================================


class TestWanGenerate:
    """Tests for ``WanAdapter.generate``."""

    def test_generate_returns_video_result(self) -> None:
        """generate() should return a correctly populated VideoResult."""
        rng = np.random.RandomState(99)

        def fake_pipeline(
            payload: PreparedConditionPayload,
            config: VideoGenerationConfig,
        ) -> np.ndarray:
            return rng.randint(
                0, 256,
                (config.num_frames, config.height, config.width, 3),
                dtype=np.uint8,
            )

        backend = LocalCallableBackend(pipeline_fn=fake_pipeline)
        adapter = WanAdapter(backend=backend)

        seq = _make_fgpp_sequence(n_frames=4, h=16, w=16)
        config = VideoGenerationConfig(
            seed=123, fps=30.0,
            width=16, height=16,
            num_frames=4,
        )
        payload = adapter.prepare_conditions(seq, config)
        result = adapter.generate(payload, config)

        assert isinstance(result, VideoResult)
        assert result.adapter_id == "wan2.1"
        assert result.fps == 30.0
        assert result.seed == 123
        assert result.frames_srgb_u8.shape == (4, 16, 16, 3)
        assert result.frames_srgb_u8.dtype == np.uint8

    def test_generate_without_backend_raises(self) -> None:
        """generate() should raise RuntimeError if no backend is set."""
        adapter = WanAdapter()
        config = VideoGenerationConfig()
        payload = PreparedConditionPayload()

        with pytest.raises(RuntimeError, match="no backend"):
            adapter.generate(payload, config)


# ===================================================================
# TestBackendTransport
# ===================================================================


class TestBackendTransport:
    """Tests for backend transport implementations."""

    def test_local_callable_invokes_fn(self) -> None:
        """LocalCallableBackend should call the provided callable."""
        call_log: list[bool] = []

        def my_fn(
            payload: PreparedConditionPayload,
            config: VideoGenerationConfig,
        ) -> np.ndarray:
            call_log.append(True)
            return np.zeros(
                (config.num_frames, config.height, config.width, 3),
                dtype=np.uint8,
            )

        backend = LocalCallableBackend(pipeline_fn=my_fn)
        config = VideoGenerationConfig(
            num_frames=2, width=4, height=4,
        )
        result = backend.submit(PreparedConditionPayload(), config)

        assert len(call_log) == 1
        assert result.shape == (2, 4, 4, 3)

    def test_local_callable_is_available(self) -> None:
        """LocalCallableBackend.is_available() should return True."""
        backend = LocalCallableBackend(pipeline_fn=lambda p, c: None)
        assert backend.is_available() is True

    def test_foundary_http_raises_on_connection_refused(self) -> None:
        """FoundaryHttpTransport should raise BackendUnavailableError."""
        # Use a port that is (almost certainly) not listening
        transport = FoundaryHttpTransport(
            endpoint_url="http://127.0.0.1:1/v1/generate",
            timeout_s=1.0,
        )
        config = VideoGenerationConfig(num_frames=1, width=4, height=4)

        with pytest.raises(BackendUnavailableError) as exc_info:
            transport.submit(PreparedConditionPayload(), config)

        assert exc_info.value.exit_code == 30
        assert exc_info.value.agent_fix_hint != ""


# ===================================================================
# TestRegistration
# ===================================================================


class TestRegistration:
    """Tests for adapter registry integration."""

    def test_wan_in_registry(self) -> None:
        """'wan2.1' should appear in list_adapters()."""
        adapters = list_adapters()
        assert "wan2.1" in adapters

    def test_wan_properties(self) -> None:
        """WanAdapter should have correct adapter_id and tier."""
        adapter = WanAdapter()
        assert adapter.adapter_id == "wan2.1"
        assert adapter.tier == "whitebox"
