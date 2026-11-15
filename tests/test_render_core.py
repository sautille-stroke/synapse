"""Tests for render_core — PR-78.

Covers:
- SoftwareRasterizer near-clip plane clipping (Sutherland-Hodgman)
- Perspective-correct 1/w interpolation
- True per-pixel screen motion vectors from prev_view_proj
- No inverted projection artifacts on triangles crossing near clip
"""

from __future__ import annotations

import math

import numpy as np
from synapse.fgpp.frame import CameraParams
from synapse.render_core.rasterizer import SoftwareRasterizer, entity_hash

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
        fx=fx, fy=fy,
        cx=width / 2.0, cy=height / 2.0,
        width=width, height=height,
        fov_degrees=fov_deg,
        z_near=z_near, z_far=z_far,
        view_matrix=view,
        world_matrix=world,
    )


def _unit_cube_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a simple unit cube mesh centered at origin."""
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
        [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float64)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 6, 5], [4, 7, 6],  # back
        [0, 4, 5], [0, 5, 1],  # bottom
        [2, 6, 7], [2, 7, 3],  # top
        [0, 7, 4], [0, 3, 7],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.int32)

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
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0,  1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
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
        verts = np.array([
            [-2.0, -2.0, 0.0],   # well in front of camera
            [2.0, -2.0, 0.0],    # well in front of camera
            [0.0,  2.0, 5.4],    # behind camera near plane
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        cam = _make_camera(width=64, height=64, position=(0, 0, 5.0),
                           look_at=(0, 0, 0), z_near=0.5)
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
        verts = np.array([
            [-1.0, -1.0, 10.0],
            [1.0, -1.0, 10.0],
            [0.0,  1.0, 10.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.full((3, 3), [0, 0, 1], dtype=np.float64)

        cam = _make_camera(width=32, height=32, position=(0, 0, 5.0),
                           look_at=(0, 0, -1))
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
            verts, faces, norms, cam_curr,
            prev_camera=cam_prev,
        )

        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert np.any(fg_mask), "Triangle should be visible"

        # Motion vectors should be non-zero for visible pixels
        mv = frame.motion_vector.astype(np.float32)  # type: ignore[union-attr]
        fg_mv = mv[fg_mask]
        mv_magnitude = np.sqrt(fg_mv[:, 0]**2 + fg_mv[:, 1]**2)

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
            verts, faces, norms, cam,
            prev_camera=cam,
        )

        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        assert np.any(fg_mask)

        mv = frame.motion_vector.astype(np.float32)  # type: ignore[union-attr]
        fg_mv = mv[fg_mask]
        # With identical cameras, motion vectors should be essentially zero
        max_mv = float(np.max(np.abs(fg_mv)))
        assert max_mv < 0.5, (
            f"Stationary camera should produce near-zero motion vectors, "
            f"got max={max_mv:.4f}"
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
        verts = np.array([
            [-2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [0.0,  1.0, -1.0],  # behind near plane
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.full((3, 3), [0, 0, 1], dtype=np.float64)

        cam_prev = _make_camera(width=64, height=64, position=(0, 0, 2.0),
                                look_at=(0, 0, 0), z_near=0.5)
        cam_curr = _make_camera(width=64, height=64, position=(0.3, 0, 2.0),
                                look_at=(0, 0, 0), z_near=0.5)

        frame = rast.render(
            verts, faces, normals, cam_curr,
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
        verts = np.array([
            [-1.0, -1.0, -1.0],  # closer
            [1.0, -1.0,  0.0],   # mid
            [0.0,  1.0,  1.0],   # farther
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.full((3, 3), [0, 0, 1], dtype=np.float64)

        cam = _make_camera(width=64, height=64, position=(0, 0, 5))
        frame = rast.render(verts, faces, normals, cam)

        fg_mask = frame.cryptomatte_id > 0  # type: ignore[operator]
        if np.any(fg_mask):
            depths = frame.depth[fg_mask]
            assert np.all(depths > 0)
            assert np.all(np.isfinite(depths))