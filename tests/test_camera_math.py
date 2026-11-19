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
