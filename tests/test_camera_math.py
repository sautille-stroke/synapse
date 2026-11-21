"""Unit tests for synapse.render_core.camera_math (SSOT module).

Covers:
- look_at: basic, zenith/nadir singularity, eye==target guard, orthonormality
- look_at_rotation: basic, singularity fallback consistency
- perspective_projection: basic properties
- view_matrix_from_pose / pose_from_view_matrix: roundtrip
- quat_to_matrix: identity, normalization, zero quaternion
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.errors import ContractValidationError
from synapse.render_core.camera_math import (
    look_at,
    look_at_rotation,
    perspective_projection,
    pose_from_view_matrix,
    quat_to_matrix,
    view_matrix_from_pose,
)


class TestLookAt:
    """Tests for look_at() view matrix computation."""

    def test_basic_look_along_neg_z(self) -> None:
        """Camera at origin looking at (0, 0, -1) should produce identity-like view."""
        eye = np.array([0.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, -1.0])
        view = look_at(eye, target)
        assert view.shape == (4, 4)
        # Rotation sub-matrix should be close to identity (looking along -Z with Y up)
        rot = view[:3, :3]
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-8)

    def test_orthonormality(self) -> None:
        """Rotation sub-matrix must be orthonormal (det ≈ 1)."""
        eye = np.array([3.0, 4.0, 5.0])
        target = np.array([1.0, 2.0, 0.0])
        view = look_at(eye, target)
        rot = view[:3, :3]
        det = np.linalg.det(rot)
        np.testing.assert_allclose(det, 1.0, atol=1e-6)
        # R @ R^T should be identity
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-8)

    def test_zenith_singularity(self) -> None:
        """Camera looking straight up should not produce NaN."""
        eye = np.array([0.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 0.0])
        view = look_at(eye, target)
        assert not np.any(np.isnan(view))
        rot = view[:3, :3]
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-4)

    def test_nadir_singularity(self) -> None:
        """Camera looking straight down should not produce NaN."""
        eye = np.array([0.0, 10.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        view = look_at(eye, target)
        assert not np.any(np.isnan(view))

    def test_eye_equals_target_raises(self) -> None:
        """eye == target should raise ContractValidationError, not silently NaN."""
        eye = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ContractValidationError, match="coincident"):
            look_at(eye, eye.copy())

    def test_custom_up_vector(self) -> None:
        """Non-default up vector should produce valid view matrix."""
        eye = np.array([5.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])  # Z-up (perpendicular to forward)
        view = look_at(eye, target, up=up)
        assert not np.any(np.isnan(view))
        np.testing.assert_allclose(np.linalg.det(view[:3, :3]), 1.0, atol=1e-6)

    def test_new_up_is_normalized(self) -> None:
        """The computed up vector in the view matrix must be unit length."""
        eye = np.array([1.0, 2.0, 3.0])
        target = np.array([4.0, 5.0, 6.0])
        view = look_at(eye, target)
        new_up = view[1, :3]
        np.testing.assert_allclose(np.linalg.norm(new_up), 1.0, atol=1e-10)


class TestLookAtRotation:
    """Tests for look_at_rotation() 3×3 rotation."""

    def test_basic(self) -> None:
        """Should return a valid 3×3 rotation."""
        eye = np.array([0.0, 0.0, 5.0])
        target = np.array([0.0, 0.0, 0.0])
        rot = look_at_rotation(eye, target)
        assert rot.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-6)

    def test_coincident_returns_identity(self) -> None:
        """eye == target should return identity (graceful degradation)."""
        eye = np.array([1.0, 1.0, 1.0])
        rot = look_at_rotation(eye, eye.copy())
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-10)

    def test_singularity_fallback_uses_z_axis(self) -> None:
        """Singularity (looking straight up) should use Z-axis fallback, consistent with look_at()."""
        eye = np.array([0.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 0.0])
        rot = look_at_rotation(eye, target)
        assert not np.any(np.isnan(rot))
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-4)


class TestPerspectiveProjection:
    """Tests for perspective_projection()."""

    def test_shape(self) -> None:
        proj = perspective_projection(60.0, 1920, 1080)
        assert proj.shape == (4, 4)

    def test_zero_height_safe(self) -> None:
        """Should not crash with height=0 (clamped to 1 internally)."""
        proj = perspective_projection(60.0, 1920, 0)
        assert not np.any(np.isnan(proj))

    def test_w_component(self) -> None:
        """proj[3,2] should be -1 for right-handed OpenGL NDC."""
        proj = perspective_projection(60.0, 1920, 1080)
        assert proj[3, 2] == pytest.approx(-1.0)


class TestViewMatrixRoundtrip:
    """Tests for view_matrix_from_pose / pose_from_view_matrix roundtrip."""

    def test_rotation_9_roundtrip(self) -> None:
        """Column-major 9-float rotation should roundtrip through view matrix."""
        pos = [1.0, 2.0, 3.0]
        # Identity rotation in column-major
        rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        view = view_matrix_from_pose(pos, rot)
        pos_out, rot_out = pose_from_view_matrix(view)
        np.testing.assert_allclose(pos_out, pos, atol=1e-10)
        np.testing.assert_allclose(rot_out, rot, atol=1e-10)

    def test_quaternion_roundtrip(self) -> None:
        """Quaternion [x,y,z,w] should produce a valid view matrix."""
        pos = [0.0, 0.0, 5.0]
        q = [0.0, 0.0, 0.0, 1.0]  # identity quaternion
        view = view_matrix_from_pose(pos, q)
        pos_out, rot_out = pose_from_view_matrix(view)
        np.testing.assert_allclose(pos_out, pos, atol=1e-10)


class TestQuatToMatrix:
    """Tests for quat_to_matrix()."""

    def test_identity(self) -> None:
        """Identity quaternion [0,0,0,1] should give identity matrix."""
        rot = quat_to_matrix([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-10)

    def test_normalization(self) -> None:
        """Non-unit quaternion should be normalized before conversion."""
        # Scale identity quaternion by 2
        rot = quat_to_matrix([0.0, 0.0, 0.0, 2.0])
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-10)

    def test_zero_quaternion(self) -> None:
        """Zero quaternion should return identity (graceful degradation)."""
        rot = quat_to_matrix([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-10)

    def test_90_degree_rotation_z(self) -> None:
        """90° rotation around Z axis."""
        import math

        angle = math.pi / 2.0
        q = [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]
        rot = quat_to_matrix(q)
        expected = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(rot, expected, atol=1e-10)

    def test_det_is_one(self) -> None:
        """Any quaternion should produce det(R) = 1."""
        q = [0.1, 0.2, 0.3, 0.9]
        rot = quat_to_matrix(q)
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-6)


class TestNativeNumpyParity:
    """Verify exact elementwise parity between aether_geo C++ and pure NumPy fallback."""

    def test_non_symmetric_pose_view_parity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-symmetric rotation (9-float col-major, 3x3, and quat) must match native C++."""
        import math

        from synapse.bridge import aether_bridge
        from synapse.render_core import camera_math

        angle = math.radians(37.0)
        q = np.array([0.2, -0.35, 0.5, 0.77], dtype=np.float64)
        pos = np.array([1.5, -2.25, 4.1], dtype=np.float64)

        # Native (if available) vs forced NumPy
        has_native = aether_bridge.has_aether_geo()
        if has_native:
            native_q_mat = quat_to_matrix(q)
            native_view_q = view_matrix_from_pose(pos, q)
            native_rot_col = native_q_mat.flatten(order="F")
            native_view_9 = view_matrix_from_pose(pos, native_rot_col)
            native_view_33 = view_matrix_from_pose(pos, native_q_mat)
            native_pos_out, native_rot_out = pose_from_view_matrix(native_view_q)
            native_look_zenith = look_at(np.array([0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]))
            native_look_nadir = look_at(np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 0.0]))
            native_look_rot_coinc = look_at_rotation(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
            native_proj = perspective_projection(53.0, 1280, 720, 0.2, 500.0)

        monkeypatch.setattr(camera_math, "get_aether_geo", lambda: None)

        np_q_mat = quat_to_matrix(q)
        np_view_q = view_matrix_from_pose(pos, q)
        np_rot_col = np_q_mat.flatten(order="F")
        np_view_9 = view_matrix_from_pose(pos, np_rot_col)
        np_view_33 = view_matrix_from_pose(pos, np_q_mat)
        np_pos_out, np_rot_out = pose_from_view_matrix(np_view_q)
        np_look_zenith = look_at(np.array([0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]))
        np_look_nadir = look_at(np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        np_look_rot_coinc = look_at_rotation(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        np_proj = perspective_projection(53.0, 1280, 720, 0.2, 500.0)

        # Roundtrip self-consistency for non-symmetric rotation
        np.testing.assert_allclose(np_pos_out, pos, atol=1e-10)
        np.testing.assert_allclose(np_rot_out, np_rot_col, atol=1e-10)
        np.testing.assert_allclose(np_view_q, np_view_9, atol=1e-12)
        np.testing.assert_allclose(np_view_q, np_view_33, atol=1e-12)
        np.testing.assert_allclose(np_look_rot_coinc, np.eye(3), atol=1e-12)

        # Verify non-zero skew-symmetric part (r != r.T)
        r_yaw = np.array(
            [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]],
            dtype=np.float64,
        )
        v_yaw = view_matrix_from_pose(pos, r_yaw.flatten(order="F"))
        np.testing.assert_allclose(v_yaw[:3, :3], r_yaw, atol=1e-12)

        if has_native:
            np.testing.assert_allclose(np_q_mat, native_q_mat, atol=1e-12)
            np.testing.assert_allclose(np_view_q, native_view_q, atol=1e-12)
            np.testing.assert_allclose(np_view_9, native_view_9, atol=1e-12)
            np.testing.assert_allclose(np_view_33, native_view_33, atol=1e-12)
            np.testing.assert_allclose(np_pos_out, native_pos_out, atol=1e-12)
            np.testing.assert_allclose(np_rot_out, native_rot_out, atol=1e-12)
            np.testing.assert_allclose(np_look_zenith, native_look_zenith, atol=1e-12)
            np.testing.assert_allclose(np_look_nadir, native_look_nadir, atol=1e-12)
            np.testing.assert_allclose(np_look_rot_coinc, native_look_rot_coinc, atol=1e-12)
            np.testing.assert_allclose(np_proj, native_proj, atol=1e-12)

    def test_numpy_primitives_outward_winding_and_uvs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NumPy fallback closed primitives must have positive signed volume (outward CCW) and UVs."""
        from synapse.render_core import procedural

        monkeypatch.setattr(procedural, "_create_primitive_mesh", lambda *a, **kw: None)
        for prim in ("cube", "sphere", "cylinder", "torus", "cone"):
            mesh = procedural.generate_procedural_mesh(prim)
            v = mesh.vertices
            f = mesh.faces
            v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            signed_vol = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)
            assert signed_vol > 0.0, f"{prim} NumPy fallback has inverted CW winding (vol={signed_vol})"
            assert mesh.uvs is not None and mesh.uvs.shape == (mesh.num_vertices, 2)
            assert mesh.vertices.dtype == np.float64

        # Humanoid sphere head must also have positive signed volume
        head_v, head_f = procedural._generate_segment_mesh("sphere", (0.12,), segments=16)
        hv0, hv1, hv2 = head_v[head_f[:, 0]], head_v[head_f[:, 1]], head_v[head_f[:, 2]]
        head_vol = float(np.sum(np.einsum("ij,ij->i", hv0, np.cross(hv1, hv2))) / 6.0)
        assert head_vol > 0.0, f"Humanoid head sphere has inverted winding (vol={head_vol})"

    def test_pan_camera_positive_elevation_looks_down_toward_target(self) -> None:
        """Positive elevation_deg places pan camera above target and pitches downward to look at it."""
        from synapse.render_core.camera import generate_cameras

        cams = generate_cameras(
            "pan",
            num_frames=3,
            distance=4.0,
            elevation_deg=15.0,
            azimuth_start_deg=-30.0,
            azimuth_end_deg=30.0,
        )
        mid_cam = cams[1]  # pan_deg == 0.0 -> looking directly through origin
        # Camera forward in world space is -view[2, :3]
        forward = -mid_cam.view_matrix[2, :3]
        assert forward[1] < -0.2, f"Expected downward pitch for +15 deg elevation, got forward={forward}"
        # Ray from camera position along forward should pass close to origin (0, 0, 0)
        eye = -(mid_cam.view_matrix[:3, :3].T @ mid_cam.view_matrix[:3, 3])
        closest = eye + float(np.dot(-eye, forward)) * forward
        np.testing.assert_allclose(closest, [0.0, 0.0, 0.0], atol=1e-6)
