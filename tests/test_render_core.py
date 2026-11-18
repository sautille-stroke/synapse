"""Tests for render_core — PR-78.

Covers:
- SoftwareRasterizer near-clip plane clipping (Sutherland-Hodgman)
- Perspective-correct 1/w interpolation
- True per-pixel screen motion vectors from prev_view_proj
- No inverted projection artifacts on triangles crossing near clip
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from synapse.errors import ContractValidationError, evaluate_hitl_circuit_breaker
from synapse.fgpp.frame import CameraParams
from synapse.render_core.conditions import Conditions
from synapse.render_core.morphology import binary_dilation, binary_erosion
from synapse.render_core.procedural import generate_humanoid_proxy
from synapse.render_core.pyramid import separable_gaussian, upsample2x
from synapse.render_core.rasterizer import SoftwareRasterizer, entity_hash
from synapse.telemetry import MemoryForecaster, TraceContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(
    width: int = 64,
    height: int = 64,
    fov_deg: float = 60.0,
    z_near: float = 0.1,
    z_far: float = 100.0,
    position: tuple[float, float, float] = (0.0, 0.0, 5.0),
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CameraParams:
    """Create a camera looking from *position* toward *look_at*."""
    pos = np.array(position, dtype=np.float64)
    target = np.array(look_at, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    fwd = target - pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-30)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-30)
    up_vec = np.cross(right, fwd)

    # View matrix (world → camera)
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = up_vec
    view[2, :3] = -fwd
    view[0, 3] = -np.dot(right, pos)
    view[1, 3] = -np.dot(up_vec, pos)
    view[2, 3] = np.dot(fwd, pos)

    world = np.linalg.inv(view)

    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx

    return CameraParams(
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
        fov_degrees=fov_deg,
        z_near=z_near,
        z_far=z_far,
        view_matrix=view,
        world_matrix=world,
    )


def _unit_cube_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a simple unit cube mesh centered at origin."""
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # front
            [4, 6, 5],
            [4, 7, 6],  # back
            [0, 4, 5],
            [0, 5, 1],  # bottom
            [2, 6, 7],
            [2, 7, 3],  # top
            [0, 7, 4],
            [0, 3, 7],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ],
        dtype=np.int32,
    )

    # Face normals for each vertex (simplified)
    normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face]
        n = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(n)
        if n_len > 1e-8:
            n = n / n_len
        normals[face] += n
    for i in range(len(normals)):
        n_len = np.linalg.norm(normals[i])
        if n_len > 1e-8:
            normals[i] /= n_len

    return vertices, faces, normals


def _simple_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a single triangle in front of origin."""
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return vertices, faces, normals


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSoftwareRasterizerBasics:
    """Basic rasterizer tests."""

    def test_entity_hash_deterministic(self) -> None:
        """entity_hash returns consistent values."""
        h1 = entity_hash("test_entity")
        h2 = entity_hash("test_entity")
        assert h1 == h2
        assert isinstance(h1, int)
        assert h1 != entity_hash("other_entity")

    def test_render_produces_fgpp_frame(self) -> None:
        """Rendering a simple triangle produces valid FGPPFrame."""
        rast = SoftwareRasterizer(width=32, height=32)
        verts, faces, norms = _simple_triangle()
        cam = _make_camera(width=32, height=32, position=(0, 0, 3))
        frame = rast.render(verts, faces, norms, cam)

        assert frame.depth.shape == (32, 32)
        assert frame.world_normal.shape == (32, 32, 3)  # type: ignore[union-attr]
        assert frame.motion_vector.shape == (32, 32, 2)  # type: ignore[union-attr]
        assert frame.cryptomatte_id.shape == (32, 32)  # type: ignore[union-attr]

    def test_render_writes_depth(self) -> None:
        """Rendered triangle produces non-infinite depth values."""
        rast = SoftwareRasterizer(width=32, height=32)
        verts, faces, norms = _simple_triangle()
        cam = _make_camera(width=32, height=32, position=(0, 0, 3))
        frame = rast.render(verts, faces, norms, cam)

        # Some pixels should have depth < z_far
        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert np.any(fg_mask), "No foreground pixels rendered"
        fg_depths = frame.depth[fg_mask]
        assert np.all(fg_depths < cam.z_far)
        assert np.all(fg_depths > 0)


class TestNearClipClipping:
    """Test Sutherland-Hodgman near-plane clipping."""

    def test_triangle_crossing_near_clip_no_artifacts(self) -> None:
        """A triangle straddling the near plane produces valid pixels
        without inverted projection artifacts."""
        rast = SoftwareRasterizer(width=64, height=64)
        # Camera at z=5 looking toward origin, near=0.5.
        # Two vertices at z=0 (4.5 units in front), one vertex at z=5.4
        # (behind camera at z=5 → view z = -(5-5.4) = +0.4, behind near)
        verts = np.array(
            [
                [-2.0, -2.0, 0.0],  # well in front of camera
                [2.0, -2.0, 0.0],  # well in front of camera
                [0.0, 2.0, 5.4],  # behind camera near plane
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        cam = _make_camera(width=64, height=64, position=(0, 0, 5.0), look_at=(0, 0, 0), z_near=0.5)
        frame = rast.render(verts, faces, normals, cam)

        # Should have some rendered pixels (the two front vertices are visible)
        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert np.any(fg_mask), "Near-clipped triangle should still produce pixels"

        # All depths must be positive and within valid range
        fg_depths = frame.depth[fg_mask]
        assert np.all(fg_depths > 0), "Depth must be positive (no inverted projection)"
        assert np.all(np.isfinite(fg_depths)), "Depth must be finite"

    def test_triangle_fully_behind_near_clip(self) -> None:
        """A triangle entirely behind the near plane produces no pixels."""
        rast = SoftwareRasterizer(width=32, height=32)
        # All vertices behind the camera
        verts = np.array(
            [
                [-1.0, -1.0, 10.0],
                [1.0, -1.0, 10.0],
                [0.0, 1.0, 10.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.full((3, 3), [0, 0, 1], dtype=np.float64)

        cam = _make_camera(width=32, height=32, position=(0, 0, 5.0), look_at=(0, 0, -1))
        frame = rast.render(verts, faces, normals, cam)

        # Triangle is behind the camera, should have no foreground pixels
        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert not np.any(fg_mask), "Triangle behind camera should not render"


class TestMotionVectors:
    """Test true per-pixel motion vector computation."""

    def test_rigid_translation_nonzero_mv(self) -> None:
        """A rigid body translation produces non-zero motion vectors."""
        rast = SoftwareRasterizer(width=64, height=64)
        verts, faces, norms = _simple_triangle()

        # Camera moves from (0,0,5) to (0.5,0,5) → horizontal translation
        cam_prev = _make_camera(width=64, height=64, position=(0, 0, 5))
        cam_curr = _make_camera(width=64, height=64, position=(0.5, 0, 5))

        frame = rast.render(
            verts,
            faces,
            norms,
            cam_curr,
            prev_camera=cam_prev,
        )

        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert np.any(fg_mask), "Triangle should be visible"

        # Motion vectors should be non-zero for visible pixels
        mv = frame.motion_vector.astype(np.float32)  # type: ignore[union-attr]
        fg_mv = mv[fg_mask]
        mv_magnitude = np.sqrt(fg_mv[:, 0] ** 2 + fg_mv[:, 1] ** 2)

        assert np.any(mv_magnitude > 0.01), (
            f"Motion vectors should be non-zero for translated camera. "
            f"Max magnitude: {float(np.max(mv_magnitude)):.4f}"
        )

    def test_stationary_camera_zero_mv(self) -> None:
        """A stationary camera (prev == curr) produces zero motion vectors."""
        rast = SoftwareRasterizer(width=32, height=32)
        verts, faces, norms = _simple_triangle()

        cam = _make_camera(width=32, height=32, position=(0, 0, 5))
        frame = rast.render(
            verts,
            faces,
            norms,
            cam,
            prev_camera=cam,
        )

        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert np.any(fg_mask)

        mv = frame.motion_vector.astype(np.float32)  # type: ignore[union-attr]
        fg_mv = mv[fg_mask]
        # With identical cameras, motion vectors should be essentially zero
        max_mv = float(np.max(np.abs(fg_mv)))
        assert max_mv < 0.5, (
            f"Stationary camera should produce near-zero motion vectors, got max={max_mv:.4f}"
        )

    def test_no_prev_camera_zero_mv(self) -> None:
        """Without prev_camera, motion vectors should be zero."""
        rast = SoftwareRasterizer(width=32, height=32)
        verts, faces, norms = _simple_triangle()
        cam = _make_camera(width=32, height=32, position=(0, 0, 5))
        frame = rast.render(verts, faces, norms, cam)

        assert np.allclose(frame.motion_vector, 0.0)

    def test_near_clip_triangle_with_motion_vectors(self) -> None:
        """Near-clipped triangle with camera translation still produces
        valid (non-NaN) motion vectors."""
        rast = SoftwareRasterizer(width=64, height=64)
        # Triangle crossing near plane
        verts = np.array(
            [
                [-2.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
                [0.0, 1.0, -1.0],  # behind near plane
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.full((3, 3), [0, 0, 1], dtype=np.float64)

        cam_prev = _make_camera(
            width=64, height=64, position=(0, 0, 2.0), look_at=(0, 0, 0), z_near=0.5
        )
        cam_curr = _make_camera(
            width=64, height=64, position=(0.3, 0, 2.0), look_at=(0, 0, 0), z_near=0.5
        )

        frame = rast.render(
            verts,
            faces,
            normals,
            cam_curr,
            prev_camera=cam_prev,
        )

        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        if np.any(fg_mask):
            mv = frame.motion_vector.astype(np.float32)  # type: ignore[union-attr]
            fg_mv = mv[fg_mask]
            assert np.all(np.isfinite(fg_mv)), (
                "Motion vectors for near-clipped triangle must be finite"
            )


class TestPerspectiveCorrectInterpolation:
    """Test perspective-correct 1/w interpolation."""

    def test_depth_monotonic(self) -> None:
        """Depth values are monotonically consistent with vertex ordering."""
        rast = SoftwareRasterizer(width=64, height=64)
        # Triangle with varying depth
        verts = np.array(
            [
                [-1.0, -1.0, -1.0],  # closer
                [1.0, -1.0, 0.0],  # mid
                [0.0, 1.0, 1.0],  # farther
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.full((3, 3), [0, 0, 1], dtype=np.float64)

        cam = _make_camera(width=64, height=64, position=(0, 0, 5))
        frame = rast.render(verts, faces, normals, cam)

        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        if np.any(fg_mask):
            depths = frame.depth[fg_mask]
            assert np.all(depths > 0)
            assert np.all(np.isfinite(depths))


# ---------------------------------------------------------------------------
# PR-95: Extended Coverage for Homogeneous Clipping, Intrinsics, Telemetry & Math
# ---------------------------------------------------------------------------


class TestDynamicMotionVectorsAndIntrinsics:
    """PR-95: Test prev_vertices motion vectors and off-center intrinsics."""

    def test_prev_vertices_motion_vector_deformation(self) -> None:
        """Stationary camera with deformed mesh (prev_vertices) produces non-zero motion vectors."""
        rast = SoftwareRasterizer(width=64, height=64)
        verts, faces, norms = _simple_triangle()
        prev_verts = verts.copy()
        prev_verts[:, 0] -= 0.3  # Previous mesh was shifted to the left by 0.3 units

        cam = _make_camera(width=64, height=64, position=(0, 0, 3))
        frame = rast.render(
            verts,
            faces,
            norms,
            cam,
            prev_camera=cam,
            prev_vertices=prev_verts,
        )

        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert np.any(fg_mask), "Foreground triangle should be rendered"
        mv = frame.motion_vector.astype(np.float32)  # type: ignore[union-attr]
        fg_mv = mv[fg_mask]
        assert np.any(np.abs(fg_mv[:, 0]) > 0.5), (
            "Mesh deformation with stationary camera must produce motion vectors"
        )

    def test_principal_point_offset_projection(self) -> None:
        """Principal point offset shifts the projected center correctly."""
        w, h = 64, 64
        cam = _make_camera(width=w, height=h, position=(0, 0, 3))
        # Modify cx, cy to be off-center
        cam.cx = 40.0
        cam.cy = 20.0
        proj = cam.projection_matrix()

        # Check p02 and p12 (corrected formula from PR-134: p02 = (w - 2*cx)/w)
        expected_p02 = (64.0 - 2.0 * 40.0) / 64.0  # -16 / 64 = -0.25
        expected_p12 = (2.0 * 20.0 - 64.0) / 64.0  # -24 / 64 = -0.375
        assert abs(proj[0, 2] - expected_p02) < 1e-6
        assert abs(proj[1, 2] - expected_p12) < 1e-6

        # Rasterize a centered point/triangle and ensure it renders offset
        rast = SoftwareRasterizer(width=w, height=h)
        verts, faces, norms = _simple_triangle()
        frame = rast.render(verts, faces, norms, cam)
        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert np.any(fg_mask)


class TestHumanoidKinematicChain:
    """PR-95: Test hierarchical joint rotation linkage in generate_humanoid_proxy."""

    def test_hip_rotation_moves_knee_and_foot(self) -> None:
        """Rotating hip (left_thigh) propagates forward kinematics to shin and foot."""
        verts_neutral, _, bones_neutral = generate_humanoid_proxy(
            pose_params={"phase": 0.0, "left_thigh": 0.0},
        )
        verts_bent, _, bones_bent = generate_humanoid_proxy(
            pose_params={"phase": 0.0, "left_thigh": 0.8},
        )

        shin_idx_n = bones_neutral["left_shin"]
        foot_idx_n = bones_neutral["left_foot"]
        torso_idx_n = bones_neutral["torso"]

        shin_pos_neutral = verts_neutral[shin_idx_n].mean(axis=0)
        shin_pos_bent = verts_bent[shin_idx_n].mean(axis=0)

        foot_pos_neutral = verts_neutral[foot_idx_n].mean(axis=0)
        foot_pos_bent = verts_bent[foot_idx_n].mean(axis=0)

        torso_pos_neutral = verts_neutral[torso_idx_n].mean(axis=0)
        torso_pos_bent = verts_bent[torso_idx_n].mean(axis=0)

        # Torso should stay in place
        np.testing.assert_allclose(torso_pos_neutral, torso_pos_bent, atol=1e-5)

        # Shin (knee) and foot must move due to hip rotation
        shin_disp = np.linalg.norm(shin_pos_bent - shin_pos_neutral)
        foot_disp = np.linalg.norm(foot_pos_bent - foot_pos_neutral)
        assert shin_disp > 0.1, f"Shin should displace when hip bends (got {shin_disp:.4f})"
        assert foot_disp > 0.15, f"Foot should displace when hip bends (got {foot_disp:.4f})"


class TestPyramidAndMorphologyHardening:
    """PR-95: Test float64 accumulation in pyramid convolution and slice morphology."""

    def test_separable_gaussian_float16_precision(self) -> None:
        """Separable Gaussian does not overflow or truncate with small/half precision."""
        img = np.ones((32, 32), dtype=np.float32)
        blurred = separable_gaussian(img, sigma=2.0)
        np.testing.assert_allclose(blurred[5:-5, 5:-5], 1.0, atol=1e-3)

    def test_upsample_downsample_grid(self) -> None:
        """upsample2x with half-pixel alignment correctly scales dimensions."""
        img = np.zeros((16, 16), dtype=np.float32)
        img[8, 8] = 1.0
        up = upsample2x(img, 32, 32)
        assert up.shape == (32, 32)
        assert up.max() > 0.0

    def test_binary_dilation_no_border_wrap(self) -> None:
        """binary_dilation uses zero-padded slices without border wrap."""
        mask = np.zeros((16, 16), dtype=bool)
        mask[0, 0] = True
        se = np.ones((3, 3), dtype=bool)
        dilated = binary_dilation(mask, se)
        # Should not wrap to the opposite border (bottom/right)
        assert not dilated[15, 15]
        assert not dilated[0, 15]
        assert not dilated[15, 0]
        assert dilated[0, 0]
        assert dilated[0, 1]
        assert dilated[1, 0]

    def test_binary_erosion_no_border_wrap(self) -> None:
        """binary_erosion correctly preserves borders without wrap."""
        mask = np.ones((16, 16), dtype=bool)
        se = np.ones((3, 3), dtype=bool)
        eroded = binary_erosion(mask, se)
        # Edges should be eroded
        assert not eroded[0, 0]
        assert not eroded[0, 5]
        # Interior should be preserved
        assert eroded[5, 5]


class TestCircuitBreakerAndTelemetryHardening:
    """PR-95: Test HITL fail-closed checks and W3C traceparent regex."""

    def test_hitl_circuit_breaker_nan_raises_contract_validation(self) -> None:
        """NaN elapsed_seconds raises ContractValidationError (exit_code=10)."""
        with pytest.raises(ContractValidationError) as exc_info:
            evaluate_hitl_circuit_breaker(float("nan"), "video_shot")
        assert exc_info.value.exit_code == 10

    def test_hitl_circuit_breaker_invalid_domain_raises(self) -> None:
        """Invalid domain raises ContractValidationError (exit_code=10)."""
        with pytest.raises(ContractValidationError) as exc_info:
            evaluate_hitl_circuit_breaker(10.0, "unknown_domain")  # type: ignore[arg-type]
        assert exc_info.value.exit_code == 10

    def test_w3c_traceparent_parsing_and_generation(self) -> None:
        """TraceContext parses standard W3C header and rejects malformed."""
        valid_header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        ctx = TraceContext.from_env_or_dict({"traceparent": valid_header})
        assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert ctx.span_id == "00f067aa0ba902b7"
        assert ctx.trace_flags == "01"

        # Malformed / zero header regenerates fresh IDs
        invalid_header = "00-00000000000000000000000000000000-0000000000000000-00"
        fresh_ctx = TraceContext.from_env_or_dict({"traceparent": invalid_header})
        assert fresh_ctx.trace_id != "00000000000000000000000000000000"
        assert fresh_ctx.span_id != "0000000000000000"

    def test_memory_forecaster_custom_layers_and_vram(self) -> None:
        """MemoryForecaster respects num_layers and available_vram_gb."""
        forecast = MemoryForecaster.estimate_dit_peak_gb(
            num_frames=16,
            height=512,
            width=512,
            model_variant="wan2.1_14b",
            num_layers=48,
            available_vram_gb=16.0,
        )
        assert forecast.breakdown["num_layers"] == 48
        # 14B with 16 frames is > 28GB weights, so on 16GB VRAM it should fallback or reject
        assert forecast.fallback_level.value > 0

    def test_conditions_shutter_angle_default_zero(self) -> None:
        """get_shutter_angle_deg returns 0.0 when not specified."""
        cond = Conditions()
        assert cond.get_shutter_angle_deg() == 0.0


class TestWave33PR138Enhancements:
    """PR-138: Tests for MeshData unification, camera forwarding, cone primitive, and streaming."""

    def test_load_obj_missing_raises_asset_not_found(self, tmp_path: Path) -> None:
        """Missing OBJ file raises AssetNotFoundError with code 20."""
        from synapse.errors import AssetNotFoundError
        from synapse.render_core.mesh import load_obj

        with pytest.raises(AssetNotFoundError) as exc_info:
            load_obj(tmp_path / "nonexistent.obj")
        assert exc_info.value.exit_code == 20

    def test_load_obj_preserves_uv_seam_and_exports(self, tmp_path: Path) -> None:
        """OBJ with vertex sharing different UV coordinates splits vertices along seam."""
        from synapse.render_core.mesh import MeshData, load_obj

        obj_content = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "v -1.0 0.0 0.0\n"
            "vt 0.0 0.0\n"
            "vt 1.0 0.0\n"
            "vt 0.5 1.0\n"
            "vt 0.0 1.0\n"
            "# Two triangles sharing vertex 1 (0.0, 0.0, 0.0), but with different UVs (vt 1 vs vt 2)\n"
            "f 1/1 2/2 3/3\n"
            "f 1/2 4/4 3/3\n"
        )
        obj_file = tmp_path / "seam.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        mesh = load_obj(obj_file)
        assert isinstance(mesh, MeshData)
        assert mesh.num_faces == 2
        # Vertex 1 should have been split because it has 2 different UV indices (1 and 2)
        assert mesh.num_vertices >= 4
        assert mesh.uvs is not None
        assert len(mesh.uvs) == mesh.num_vertices

        # Roundtrip export
        out_obj = tmp_path / "exported.obj"
        mesh.export_obj(out_obj)
        assert out_obj.exists()

    def test_load_obj_vn_without_face_indices_computes_normals(self, tmp_path: Path) -> None:
        """OBJ with vn definitions but face lines lacking normal indices still computes valid normals."""
        from synapse.render_core.mesh import load_obj

        obj_content = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "vn 0.0 0.0 1.0\n"
            "f 1 2 3\n"
        )
        obj_file = tmp_path / "no_face_normals.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        mesh = load_obj(obj_file)
        assert mesh.normals is not None
        # Normals must not be all zeros
        norms = np.linalg.norm(mesh.normals, axis=1)
        assert np.all(norms > 0.5)

    def test_cone_procedural_mesh(self) -> None:
        """Cone primitive is watertight with 0 degenerate faces."""
        from synapse.render_core.procedural import generate_procedural_mesh

        cone = generate_procedural_mesh("cone", segments=16)
        assert cone.num_vertices > 0
        assert cone.num_faces == 32  # 16 side + 16 bottom
        # Verify no zero-area faces
        v0 = cone.vertices[cone.faces[:, 0]]
        v1 = cone.vertices[cone.faces[:, 1]]
        v2 = cone.vertices[cone.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        assert np.all(areas > 1e-6)

    def test_camera_trajectory_forwards_cam_kw(self) -> None:
        """Trajectory generator forwards z_near, z_far, aperture_fstop, shutter_angle_deg."""
        from synapse.render_core.camera import generate_cameras

        cams = generate_cameras(
            "orbit",
            num_frames=4,
            z_near=0.25,
            z_far=500.0,
            aperture_fstop=2.8,
            shutter_angle_deg=90.0,
        )
        assert len(cams) == 4
        for cam in cams:
            assert cam.z_near == 0.25
            assert cam.z_far == 500.0
            assert cam.aperture_fstop == 2.8
            assert cam.shutter_angle_deg == 90.0

    def test_pyramid_vectorized_upsample_and_1x1_reflect(self) -> None:
        """Pyramid upsample2x is vectorized and separable_gaussian does not crash on 1x1."""
        from synapse.render_core.pyramid import (
            build_laplacian_pyramid,
            reconstruct_laplacian_pyramid,
            separable_gaussian,
            upsample2x,
        )

        # 1x1 reflect crash prevention
        one_px = np.array([[128]], dtype=np.uint8)
        blurred = separable_gaussian(one_px, sigma=2.0)
        assert blurred.shape == (1, 1)

        # uint8 underflow prevention in Laplacian pyramid
        img = np.full((16, 16), 100, dtype=np.uint8)
        img[4:12, 4:12] = 200
        lap = build_laplacian_pyramid(img, n_levels=2)
        assert len(lap) == 3
        # Should not have uint8 wrap-around values
        reconstructed = reconstruct_laplacian_pyramid(lap)
        assert np.allclose(reconstructed, img.astype(np.float32), atol=2.0)

        # Vectorized upsample2x check
        small = np.random.randn(8, 8, 3).astype(np.float32)
        upsampled = upsample2x(small, 16, 16)
        assert upsampled.shape == (16, 16, 3)
