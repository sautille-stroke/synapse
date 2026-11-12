"""Tests for synapse.bridge — vg-core bridge, foundary client, and HITL breaker."""

from __future__ import annotations

import numpy as np
import pytest
from synapse.bridge.foundary_client import (
    FoundaryRequest,
    FoundaryResponse,
    HttpFoundaryTransport,
    MockFoundaryTransport,
)
from synapse.bridge.hitl import (
    CommandIRRecord,
    OperatorIdentity,
    evaluate_hitl_circuit_breaker,
)
from synapse.bridge.vg_bridge import (
    MockVgBackend,
    NativeVgBackend,
    enforce_distillation_geometry_gate,
    get_vg_backend,
)
from synapse.errors import (
    BackendUnavailableError,
    CircuitBreakerError,
    ContractValidationError,
)

# ===================================================================
# TestVgBridge
# ===================================================================


class TestVgBridge:
    """Tests for MockVgBackend, NativeVgBackend, and factory."""

    def test_mock_backend_delight(self) -> None:
        """Mock delight returns array with same shape as input."""
        backend = MockVgBackend()
        albedo = np.random.rand(64, 64, 3).astype(np.float32)
        normal = np.random.rand(64, 64, 3).astype(np.float32)
        radiance = np.random.rand(64, 64, 3).astype(np.float32)
        result = backend.delight(albedo, normal, radiance)
        assert result.shape == albedo.shape
        np.testing.assert_array_equal(result, albedo)

    def test_mock_backend_quad_remesh(self) -> None:
        """Mock quad_remesh returns input mesh unchanged."""
        backend = MockVgBackend()
        verts = np.random.rand(100, 3).astype(np.float32)
        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        out_v, out_f = backend.quad_remesh(verts, faces, target_faces=50)
        np.testing.assert_array_equal(out_v, verts)
        np.testing.assert_array_equal(out_f, faces)

    def test_mock_backend_check_manifold(self) -> None:
        """Mock check_manifold always returns True."""
        backend = MockVgBackend()
        verts = np.random.rand(10, 3)
        faces = np.array([[0, 1, 2]])
        assert backend.check_manifold(verts, faces) is True

    def test_mock_backend_solve_zero_slip_ik(self) -> None:
        """Mock solve_zero_slip_ik returns skeleton data unchanged."""
        backend = MockVgBackend()
        skel = {"joints": [1, 2, 3]}
        contacts = np.zeros((10, 2), dtype=np.float32)
        result = backend.solve_zero_slip_ik(skel, contacts)
        assert result is skel

    def test_mock_backend_refine_pose_6dof(self) -> None:
        """Mock refine_pose_6dof returns 4x4 identity."""
        backend = MockVgBackend()
        src = np.random.rand(50, 3)
        tgt = np.random.rand(30, 3)
        result = backend.refine_pose_6dof(src, tgt)
        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result, np.eye(4))

    def test_get_backend_mock(self) -> None:
        """get_vg_backend(mock=True) returns MockVgBackend."""
        backend = get_vg_backend(mock=True)
        assert isinstance(backend, MockVgBackend)

    def test_native_backend_import_error(self) -> None:
        """NativeVgBackend methods raise BackendUnavailableError."""
        backend = NativeVgBackend()
        albedo = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises(BackendUnavailableError) as exc_info:
            backend.delight(albedo, albedo, albedo)
        assert exc_info.value.exit_code == 30

    def test_native_backend_quad_remesh_import_error(self) -> None:
        """NativeVgBackend.quad_remesh raises on missing vg."""
        backend = NativeVgBackend()
        v = np.zeros((3, 3))
        f = np.array([[0, 1, 2]])
        with pytest.raises(BackendUnavailableError):
            backend.quad_remesh(v, f, 10)

    def test_native_backend_check_manifold_import_error(self) -> None:
        """NativeVgBackend.check_manifold raises on missing vg."""
        backend = NativeVgBackend()
        with pytest.raises(BackendUnavailableError):
            backend.check_manifold(np.zeros((3, 3)), np.array([[0, 1, 2]]))

    def test_native_backend_solve_ik_import_error(self) -> None:
        """NativeVgBackend.solve_zero_slip_ik raises on missing vg."""
        backend = NativeVgBackend()
        with pytest.raises(BackendUnavailableError):
            backend.solve_zero_slip_ik({}, np.zeros((1, 2)))

    def test_native_backend_refine_pose_import_error(self) -> None:
        """NativeVgBackend.refine_pose_6dof raises on missing vg."""
        backend = NativeVgBackend()
        with pytest.raises(BackendUnavailableError):
            backend.refine_pose_6dof(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_get_backend_native_unavailable(self) -> None:
        """get_vg_backend(mock=False) raises when vg not installed."""
        with pytest.raises(BackendUnavailableError) as exc_info:
            get_vg_backend(mock=False)
        assert "vg-core" in str(exc_info.value)


# ===================================================================
# TestDistillationGate
# ===================================================================


class TestDistillationGate:
    """Tests for enforce_distillation_geometry_gate."""

    def test_track_b_without_gen3d_blocked(self) -> None:
        """Track-B without gen3d highpoly is blocked."""
        with pytest.raises(ContractValidationError):
            enforce_distillation_geometry_gate(
                track="track_b_whitebox",
                has_gen3d_highpoly=False,
                vertex_count=2000,
                has_articulated_limbs=True,
                is_character=False,
            )

    def test_track_b_character_low_vertex_blocked(self) -> None:
        """Track-B character with low vertex count is blocked."""
        with pytest.raises(ContractValidationError):
            enforce_distillation_geometry_gate(
                track="track_b_whitebox",
                has_gen3d_highpoly=True,
                vertex_count=500,
                has_articulated_limbs=True,
                is_character=True,
            )

    def test_track_b_character_no_limbs_blocked(self) -> None:
        """Track-B character without articulated limbs is blocked."""
        with pytest.raises(ContractValidationError):
            enforce_distillation_geometry_gate(
                track="track_b_whitebox",
                has_gen3d_highpoly=True,
                vertex_count=2000,
                has_articulated_limbs=False,
                is_character=True,
            )

    def test_track_b_with_gen3d_passes(self) -> None:
        """Track-B with gen3d and sufficient vertices passes."""
        # Should not raise
        enforce_distillation_geometry_gate(
            track="track_b_whitebox",
            has_gen3d_highpoly=True,
            vertex_count=2000,
            has_articulated_limbs=True,
            is_character=True,
        )

    def test_track_b_non_character_with_gen3d_passes(self) -> None:
        """Track-B non-character asset with gen3d passes."""
        enforce_distillation_geometry_gate(
            track="track_b_whitebox",
            has_gen3d_highpoly=True,
            vertex_count=100,
            has_articulated_limbs=False,
            is_character=False,
        )

    def test_non_track_b_always_passes(self) -> None:
        """Non Track-B always passes regardless of params."""
        enforce_distillation_geometry_gate(
            track="track_a",
            has_gen3d_highpoly=False,
            vertex_count=0,
            has_articulated_limbs=False,
            is_character=True,
        )

    def test_error_exit_code(self) -> None:
        """ContractValidationError has exit_code=10."""
        with pytest.raises(ContractValidationError) as exc_info:
            enforce_distillation_geometry_gate(
                track="track_b_whitebox",
                has_gen3d_highpoly=False,
                vertex_count=0,
                has_articulated_limbs=False,
                is_character=False,
            )
        assert exc_info.value.exit_code == 10

    def test_error_agent_fix_hint(self) -> None:
        """Fix hint mentions TRELLIS or Hunyuan3D."""
        with pytest.raises(ContractValidationError) as exc_info:
            enforce_distillation_geometry_gate(
                track="track_b_whitebox",
                has_gen3d_highpoly=False,
                vertex_count=2000,
                has_articulated_limbs=True,
                is_character=False,
            )
        hint = exc_info.value.agent_fix_hint
        assert "TRELLIS" in hint or "Hunyuan3D" in hint


# ===================================================================
# TestFoundaryClient
# ===================================================================


class TestFoundaryClient:
    """Tests for foundary transport models."""

    def test_mock_transport_returns_ok(self) -> None:
        """MockFoundaryTransport.submit returns status='ok'."""
        transport = MockFoundaryTransport(latency_seconds=0.5)
        req = FoundaryRequest(service="trellis", job_id="j-001")
        resp = transport.submit(req)
        assert isinstance(resp, FoundaryResponse)
        assert resp.status == "ok"
        assert resp.service == "trellis"
        assert resp.outputs == {"mock": True}
        assert resp.elapsed_seconds == 0.5

    def test_http_transport_raises(self) -> None:
        """HttpFoundaryTransport raises BackendUnavailableError."""
        transport = HttpFoundaryTransport(
            base_url="http://localhost:9999"
        )
        req = FoundaryRequest(service="sam2", job_id="j-002")
        with pytest.raises(BackendUnavailableError) as exc_info:
            transport.submit(req)
        assert exc_info.value.exit_code == 30

    def test_request_model_defaults(self) -> None:
        """FoundaryRequest has sensible defaults."""
        req = FoundaryRequest(service="hunyuan3d")
        assert req.job_id == ""
        assert req.inputs == {}
        assert req.params == {}

    def test_response_model_defaults(self) -> None:
        """FoundaryResponse has sensible defaults."""
        resp = FoundaryResponse()
        assert resp.status == "ok"
        assert resp.service == ""
        assert resp.outputs == {}
        assert resp.error_message == ""
        assert resp.elapsed_seconds == 0.0


# ===================================================================
# TestHitlCircuitBreaker
# ===================================================================


class TestHitlCircuitBreaker:
    """Tests for evaluate_hitl_circuit_breaker."""

    def test_within_budget_no_trigger(self) -> None:
        """899s is within the 900s budget — returns None."""
        result = evaluate_hitl_circuit_breaker(
            elapsed_seconds=899.0, domain="video_shot"
        )
        assert result is None

    def test_exact_budget_no_trigger(self) -> None:
        """Exactly 900s does NOT trigger the breaker."""
        result = evaluate_hitl_circuit_breaker(
            elapsed_seconds=900.0, domain="video_shot"
        )
        assert result is None

    def test_over_budget_video_raises(self) -> None:
        """900.1s on video_shot triggers CircuitBreakerError."""
        with pytest.raises(CircuitBreakerError) as exc_info:
            evaluate_hitl_circuit_breaker(
                elapsed_seconds=900.1, domain="video_shot"
            )
        assert "circuit breaker" in str(exc_info.value).lower()

    def test_over_budget_asset_raises(self) -> None:
        """901s on asset_3d triggers CircuitBreakerError."""
        with pytest.raises(CircuitBreakerError) as exc_info:
            evaluate_hitl_circuit_breaker(
                elapsed_seconds=901.0, domain="asset_3d"
            )
        assert "circuit breaker" in str(exc_info.value).lower()

    def test_exit_code_is_40(self) -> None:
        """CircuitBreakerError exit_code is 40."""
        with pytest.raises(CircuitBreakerError) as exc_info:
            evaluate_hitl_circuit_breaker(
                elapsed_seconds=1000.0, domain="video_shot"
            )
        assert exc_info.value.exit_code == 40

    def test_custom_max_seconds(self) -> None:
        """Custom max_seconds=60 triggers at 61s."""
        result = evaluate_hitl_circuit_breaker(
            elapsed_seconds=60.0,
            domain="asset_3d",
            max_seconds=60.0,
        )
        assert result is None

        with pytest.raises(CircuitBreakerError):
            evaluate_hitl_circuit_breaker(
                elapsed_seconds=61.0,
                domain="asset_3d",
                max_seconds=60.0,
            )

    def test_video_shot_hint_mentions_fgpp(self) -> None:
        """Video shot fallback hint mentions FGPP."""
        with pytest.raises(CircuitBreakerError) as exc_info:
            evaluate_hitl_circuit_breaker(
                elapsed_seconds=1000.0, domain="video_shot"
            )
        assert "FGPP" in exc_info.value.agent_fix_hint

    def test_asset_hint_mentions_decal(self) -> None:
        """Asset 3D fallback hint mentions decal or procedural."""
        with pytest.raises(CircuitBreakerError) as exc_info:
            evaluate_hitl_circuit_breaker(
                elapsed_seconds=1000.0, domain="asset_3d"
            )
        hint = exc_info.value.agent_fix_hint
        assert "decal" in hint.lower() or "procedural" in hint.lower()


# ===================================================================
# TestCommandIRRecord
# ===================================================================


class TestCommandIRRecord:
    """Tests for CommandIRRecord and OperatorIdentity dataclasses."""

    def test_create_record(self) -> None:
        """Can create a CommandIRRecord with all fields."""
        record = CommandIRRecord(
            intent_group_id="grp-001",
            intent="fix UV seam on left arm",
            state_before_uri="s3://bucket/before.usd",
            actions=[{"tool": "uv_unwrap", "params": {"island": 3}}],
            state_after_uri="s3://bucket/after.usd",
            feedback="approved",
            elapsed_seconds=45.2,
        )
        assert record.intent_group_id == "grp-001"
        assert record.intent == "fix UV seam on left arm"
        assert len(record.actions) == 1
        assert record.elapsed_seconds == 45.2

    def test_record_defaults(self) -> None:
        """CommandIRRecord has sensible defaults."""
        record = CommandIRRecord(
            intent_group_id="grp-002",
            intent="adjust material roughness",
        )
        assert record.state_before_uri == ""
        assert record.actions == []
        assert record.state_after_uri == ""
        assert record.feedback == ""
        assert record.elapsed_seconds == 0.0

    def test_operator_identity_human_l1(self) -> None:
        """Can create human_l1 operator."""
        op = OperatorIdentity(
            kind="human_l1",
            operator_id="usr-001",
        )
        assert op.kind == "human_l1"
        assert op.model_urn is None

    def test_operator_identity_human_l2(self) -> None:
        """Can create human_l2 operator."""
        op = OperatorIdentity(
            kind="human_l2",
            operator_id="usr-002",
        )
        assert op.kind == "human_l2"

    def test_operator_identity_agent(self) -> None:
        """Can create agent operator with model URN."""
        op = OperatorIdentity(
            kind="agent",
            operator_id="agent-gemini-4",
            model_urn="urn:google:gemini:4.0",
        )
        assert op.kind == "agent"
        assert op.model_urn == "urn:google:gemini:4.0"
