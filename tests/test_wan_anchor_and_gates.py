"""Tests for Wan 48-frame physical anchors, float64 depth, non-degenerate gates,
and PR-76 FoundaryHttpTransport serialisation.

Covers:
- RFC 0044 §3.2 & RFC 0063 §3.1, §4.1 (AUD-P1-08)
- 48-frame physical anchors in ConditionScheduler and WanAdapter.prepare_conditions
- Float64 log-normalized depth round-trip precision (< 1e-12)
- WanAdapter.reanchor_physical_frame ground-truth conditioning slice
- validate_output_non_degenerate Tier-2 variance gate enforcement (exit code 40)
- PR-76(c): FoundaryHttpTransport serialises control_tensor and forwards traceparent
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, ClassVar

import numpy as np
import pytest
from synapse.adapters.base import PreparedConditionPayload, VideoGenerationConfig
from synapse.adapters.scheduler import (
    CausalWindowPlan,
    ConditionScheduler,
    DiscontinuityDetector,
)
from synapse.adapters.wan import FoundaryHttpTransport, LocalCallableBackend, WanAdapter
from synapse.errors import ExitCode, QualityGateError
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.fgpp.schema import (
    linear_to_log_normalized_depth,
    log_normalized_to_linear_depth,
)
from synapse.metrics.image_quality import validate_output_non_degenerate


def _make_dummy_frame(idx: int, h: int = 32, w: int = 32) -> FGPPFrame:
    """Create a dummy FGPPFrame."""
    cam = CameraParams(
        fx=32.0, fy=32.0, cx=16.0, cy=16.0,
        width=w, height=h, z_near=0.1, z_far=100.0,
    )
    return FGPPFrame(
        camera=cam,
        depth=np.full((h, w), 2.5 + idx * 0.01, dtype=np.float32),
        direct_radiance=np.full((h, w, 3), 0.5, dtype=np.float16),
        world_normal=np.full((h, w, 3), 0.0, dtype=np.float16),
        motion_vector=np.zeros((h, w, 2), dtype=np.float16),
        base_color=np.full((h, w, 3), 0.8, dtype=np.float16),
        roughness_metallic=np.zeros((h, w, 2), dtype=np.float16),
        cryptomatte_id=np.ones((h, w), dtype=np.uint32),
        frame_index=idx,
    )


class TestSchedulerPhysicalAnchors:
    """Test 48-frame physical anchor scheduling."""

    def test_scheduler_periodic_48_frame_anchors(self) -> None:
        """ConditionScheduler should mark frames 0, 48, 96 as physical anchors."""
        frames = [_make_dummy_frame(i) for i in range(100)]
        seq = FGPPSequence(frames=frames, fps=24.0, scene_id="anchor_test")

        plan = CausalWindowPlan(window_frames=97, stride_frames=64, total_frames=100)
        detector = DiscontinuityDetector()
        scheduler = ConditionScheduler(plan, detector, anchor_interval=48)

        windows = scheduler.schedule(seq)
        assert len(windows) > 0

        # Collect all physical anchor requests across all windows
        anchor_indices = set()
        for win in windows:
            for req in win.reanchor_requests:
                if req.is_physical_anchor:
                    anchor_indices.add(req.frame_index)

        assert 0 in anchor_indices
        assert 48 in anchor_indices
        assert 96 in anchor_indices


class TestWanAdapterPhysicalAnchors:
    """Test WanAdapter 48-frame physical anchor metadata and re-anchoring."""

    def test_prepare_conditions_anchor_indices(self) -> None:
        """prepare_conditions should annotate physical anchor indices."""
        frames = [_make_dummy_frame(i) for i in range(97)]
        seq = FGPPSequence(frames=frames, fps=24.0, scene_id="wan_anchor_test")

        adapter = WanAdapter()
        config = VideoGenerationConfig(num_frames=97)
        payload = adapter.prepare_conditions(seq, config)

        assert payload.control_tensor is not None
        assert payload.control_tensor.shape == (7, 97, 32, 32)
        assert "physical_anchor_indices" in payload.extra
        assert payload.extra["physical_anchor_indices"] == [0, 48, 96]
        assert payload.extra["anchor_interval"] == 48

    def test_reanchor_physical_frame(self) -> None:
        """reanchor_physical_frame extracts ground-truth 7-channel condition slice."""
        frame = _make_dummy_frame(48, h=64, w=64)
        adapter = WanAdapter()
        control_slice = adapter.reanchor_physical_frame(frame, frame_index=48)

        assert control_slice.shape == (7, 64, 64)
        assert control_slice.dtype == np.float32
        assert np.all(np.isfinite(control_slice))


class TestFloat64DepthPrecision:
    """Test float64 log-normalized depth numerical accuracy."""

    def test_log_depth_float64_roundtrip_precision(self) -> None:
        """Float64 depth conversion round-trip should have relative error < 1e-12."""
        z_near = 0.1
        z_far = 1000.0
        depths = np.linspace(z_near, z_far, 1000, dtype=np.float64)

        log_norm = linear_to_log_normalized_depth(depths, z_near_m=z_near, z_far_m=z_far)
        recovered = log_normalized_to_linear_depth(log_norm, z_near_m=z_near, z_far_m=z_far)

        rel_err = float(np.max(np.abs(depths - recovered) / depths))
        assert rel_err < 1e-12


class TestNonDegenerateQualityGate:
    """Test Tier-2 non-degenerate variance quality gate."""

    def test_textured_dynamic_frames_pass(self) -> None:
        """Varied frames should pass the gate without error."""
        rng = np.random.default_rng(42)
        frames = rng.integers(0, 256, size=(8, 64, 64, 3), dtype=np.uint8)
        validate_output_non_degenerate(frames)

    def test_constant_gray_frames_fail_code_40(self) -> None:
        """Flat solid gray frames must trigger QualityGateError (exit code 40)."""
        frames = np.full((8, 64, 64, 3), 128, dtype=np.uint8)
        with pytest.raises(QualityGateError) as exc_info:
            validate_output_non_degenerate(frames)
        assert exc_info.value.exit_code == ExitCode.CIRCUIT_BREAKER  # 40
        assert "degenerate" in str(exc_info.value)

    def test_all_black_frames_fail_code_40(self) -> None:
        """All zero frames must trigger QualityGateError."""
        frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
        with pytest.raises(QualityGateError) as exc_info:
            validate_output_non_degenerate(frames)
        assert exc_info.value.exit_code == ExitCode.CIRCUIT_BREAKER  # 40
        assert "all-zero" in str(exc_info.value)

    def test_nan_frames_fail_code_40(self) -> None:
        """Frames with NaN must trigger QualityGateError."""
        frames = np.full((8, 64, 64, 3), np.nan, dtype=np.float32)
        with pytest.raises(QualityGateError) as exc_info:
            validate_output_non_degenerate(frames)
        assert exc_info.value.exit_code == ExitCode.CIRCUIT_BREAKER  # 40
        assert "NaN or Inf" in str(exc_info.value)

    def test_wan_adapter_generate_fails_closed_on_flat_frames(self) -> None:
        """WanAdapter.generate must fail closed if backend produces flat frames."""
        flat_backend = LocalCallableBackend(
            lambda payload, config: np.full((8, 32, 32, 3), 128, dtype=np.uint8)
        )
        adapter = WanAdapter(backend=flat_backend)

        frames = [_make_dummy_frame(i) for i in range(8)]
        seq = FGPPSequence(frames=frames, fps=24.0, scene_id="flat_test")
        config = VideoGenerationConfig(num_frames=8)
        payload = adapter.prepare_conditions(seq, config)

        with pytest.raises(QualityGateError):
            adapter.generate(payload, config, check_non_degenerate=True)

        # When check_non_degenerate is disabled, it returns normally
        result = adapter.generate(payload, config, check_non_degenerate=False)
        assert result.frames_srgb_u8.shape == (8, 32, 32, 3)


# ===================================================================
# PR-76(c): FoundaryHttpTransport serialisation test
# ===================================================================


class _CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures request body and headers for test inspection."""

    captured_body: ClassVar[dict[str, Any]] = {}
    captured_headers: ClassVar[dict[str, str]] = {}

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        _CaptureHandler.captured_body = json.loads(body.decode("utf-8"))
        # Store headers with lowercase keys for case-insensitive lookup
        _CaptureHandler.captured_headers = {
            k.lower(): v for k, v in self.headers.items()
        }

        # Respond with valid frames JSON
        rng = np.random.default_rng(42)
        frames = rng.integers(10, 240, size=(2, 8, 8, 3), dtype=np.uint8)
        response = json.dumps({"frames": frames.tolist()}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress request logging during tests."""


class TestFoundaryHttpTransportSerialization:
    """PR-76(c): Mock HTTP service receives control_tensor and traceparent."""

    def test_submit_serialises_control_tensor_and_traceparent(self) -> None:
        """FoundaryHttpTransport.submit must include control_tensor in the
        request body and forward the upstream traceparent header."""
        # Start a mock HTTP server
        server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
        port = server.server_address[1]
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            upstream_traceparent = "00-abcdef0123456789abcdef0123456789-1234567890abcdef-01"
            transport = FoundaryHttpTransport(
                endpoint_url=f"http://127.0.0.1:{port}/v1/generate",
                timeout_s=5.0,
                traceparent=upstream_traceparent,
            )

            # Build a payload with a real control_tensor
            control = np.random.default_rng(0).random(
                (7, 2, 8, 8), dtype=np.float64,
            ).astype(np.float32)
            payload = PreparedConditionPayload(
                control_tensor=control,
                text_prompt="test prompt",
                extra={"anchor_interval": 48},
            )
            config = VideoGenerationConfig(
                num_frames=2, width=8, height=8, seed=42,
            )

            frames = transport.submit(payload, config)

            # Verify response
            assert frames.shape == (2, 8, 8, 3)
            assert frames.dtype == np.uint8

            # Verify request body contains control_tensor
            body = _CaptureHandler.captured_body
            assert "control_tensor" in body
            assert body["control_tensor"] is not None
            assert isinstance(body["control_tensor"], str)  # base64 string
            assert len(body["control_tensor"]) > 0

            # Verify extra dict is merged
            assert body["extra"]["anchor_interval"] == 48

            # Verify traceparent header was forwarded
            hdrs = _CaptureHandler.captured_headers
            assert "traceparent" in hdrs
            assert hdrs["traceparent"] == upstream_traceparent

        finally:
            server.server_close()
            thread.join(timeout=2.0)


class TestPrepareConditionsCryptomattePassthrough:
    """Test that prepare_conditions passes through cryptomatte_id."""

    def test_cryptomatte_id_in_payload(self) -> None:
        """prepare_conditions should include cryptomatte_id in the payload."""
        frames = [_make_dummy_frame(i) for i in range(4)]
        seq = FGPPSequence(frames=frames, fps=24.0, scene_id="crypt_test")

        adapter = WanAdapter()
        config = VideoGenerationConfig(num_frames=4)
        payload = adapter.prepare_conditions(seq, config)

        assert payload.cryptomatte_id is not None
        assert payload.cryptomatte_id.shape == (4, 32, 32)
        assert payload.cryptomatte_id.dtype == np.uint32
