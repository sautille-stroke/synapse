"""Tests for synapse.recon.multiview — MultiViewInitializer.

All tests use pure numpy synthetic data. No GPU or heavy third-party
dependencies required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from synapse.errors import ContractValidationError
from synapse.recon.multiview import MultiViewInitializer
from synapse.recon.types import CameraExtrinsics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ring_cameras(
    n_views: int,
    radius: float = 2.0,
    height: float = 0.0,
) -> list[CameraExtrinsics]:
    """Create *n_views* cameras arranged in a horizontal ring.

    Each camera looks inward toward the world origin.
    """
    cameras: list[CameraExtrinsics] = []
    for i in range(n_views):
        angle = 2.0 * math.pi * i / n_views
        cx = radius * math.cos(angle)
        cz = radius * math.sin(angle)
        pos = np.array([cx, height, cz], dtype=np.float64)

        # Look-at → build a view matrix (world-to-camera).
        forward = -pos / np.linalg.norm(pos)  # toward origin
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
        right /= right_norm
        up = np.cross(right, forward)

        R = np.eye(3, dtype=np.float64)
        R[0, :] = right
        R[1, :] = up
        R[2, :] = forward  # camera looks along +Z in camera space
        t = -R @ pos

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R
        transform[:3, 3] = t

        cameras.append(
            CameraExtrinsics(
                frame=i,
                transform=transform,
                position=pos,
                rotation_matrix=R,
                yaw_rad=angle,
            )
        )
    return cameras


def _make_sphere_depth_maps(
    cameras: list[CameraExtrinsics],
    h: int = 64,
    w: int = 64,
    sphere_radius: float = 0.5,
) -> np.ndarray:
    """Render synthetic depth maps of a unit sphere centred at the origin.

    Returns shape ``(N, H, W)`` float32 depth in metres.
    """
    depths = np.zeros((len(cameras), h, w), dtype=np.float32)
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0

    for ci, cam in enumerate(cameras):
        T = cam.transform.astype(np.float64)
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t  # camera position in world

        for py in range(h):
            for px in range(w):
                # Ray in camera space.
                d_cam = np.array(
                    [(px - cx) / fx, (py - cy) / fy, 1.0], dtype=np.float64
                )
                d_cam /= np.linalg.norm(d_cam)
                # Transform ray to world.
                d_world = R_inv @ d_cam
                o_world = t_inv

                # Ray-sphere intersection (sphere at origin, radius).
                a = np.dot(d_world, d_world)
                b = 2.0 * np.dot(o_world, d_world)
                c = np.dot(o_world, o_world) - sphere_radius ** 2
                disc = b * b - 4.0 * a * c
                if disc >= 0:
                    sqrt_disc = math.sqrt(disc)
                    t_hit = (-b - sqrt_disc) / (2.0 * a)
                    if t_hit > 0:
                        # Depth in camera space = z-component after transform.
                        hit_world = o_world + t_hit * d_world
                        hit_cam = R @ hit_world + t
                        depths[ci, py, px] = float(hit_cam[2])
    return depths


def _make_sphere_masks(
    cameras: list[CameraExtrinsics],
    h: int = 64,
    w: int = 64,
    sphere_radius: float = 0.5,
) -> np.ndarray:
    """Render binary silhouette masks of a sphere at the origin."""
    depths = _make_sphere_depth_maps(cameras, h, w, sphere_radius)
    return (depths > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiView12Views:
    """TSDF fusion with 12 ring cameras + sphere depth."""

    def test_multiview_12_views_produces_non_box(self) -> None:
        n_views = 12
        cameras = _make_ring_cameras(n_views, radius=2.0)
        h, w = 32, 32
        frames = np.zeros((n_views, h, w, 3), dtype=np.uint8)
        depths = _make_sphere_depth_maps(cameras, h, w, sphere_radius=0.5)

        init = MultiViewInitializer(
            voxel_size=0.05, sdf_trunc=0.1, volume_bounds=(2.0, 2.0, 2.0)
        )
        mesh = init.reconstruct(frames, cameras, depths=depths)

        # Must produce a non-trivial mesh, not the old 8-vertex box.
        assert mesh.vertex_count > 100, (
            f"Expected > 100 vertices from 12-view TSDF, got {mesh.vertex_count}"
        )
        assert mesh.face_count > 0


class TestDegenerateSingleView:
    """A single camera must be rejected."""

    def test_multiview_degenerate_single_view_raises(self) -> None:
        cameras = _make_ring_cameras(1)
        frames = np.zeros((1, 32, 32, 3), dtype=np.uint8)

        init = MultiViewInitializer()
        with pytest.raises(ContractValidationError) as exc_info:
            init.reconstruct(frames, cameras)

        assert exc_info.value.exit_code == 10
        assert exc_info.value.agent_fix_hint


class TestCoplanarViews:
    """Cameras with angular span < 15° must be rejected."""

    def test_multiview_coplanar_views_raises(self) -> None:
        # Place 4 cameras very close together (within ~5° arc).
        cameras: list[CameraExtrinsics] = []
        for i in range(4):
            angle = math.radians(i * 2.0)  # 0°, 2°, 4°, 6° — span = 6°
            cx = 2.0 * math.cos(angle)
            cz = 2.0 * math.sin(angle)
            pos = np.array([cx, 0.0, cz], dtype=np.float64)
            R = np.eye(3, dtype=np.float64)
            t = np.zeros(3, dtype=np.float64)
            transform = np.eye(4, dtype=np.float64)
            cameras.append(
                CameraExtrinsics(
                    frame=i,
                    transform=transform,
                    position=pos,
                    rotation_matrix=R,
                    yaw_rad=angle,
                )
            )

        frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        init = MultiViewInitializer()
        with pytest.raises(ContractValidationError) as exc_info:
            init.reconstruct(frames, cameras)

        assert exc_info.value.exit_code == 10
        assert "angular span" in str(exc_info.value).lower()


class TestVisualHull:
    """Visual hull from 8 views with sphere silhouette masks."""

    def test_multiview_with_masks_visual_hull(self) -> None:
        n_views = 8
        cameras = _make_ring_cameras(n_views, radius=2.0)
        h, w = 32, 32
        frames = np.zeros((n_views, h, w, 3), dtype=np.uint8)
        masks = _make_sphere_masks(cameras, h, w, sphere_radius=0.5)

        init = MultiViewInitializer(
            voxel_size=0.05, sdf_trunc=0.1, volume_bounds=(2.0, 2.0, 2.0)
        )
        mesh = init.reconstruct(frames, cameras, masks=masks)

        assert mesh.vertex_count > 50, (
            f"Expected > 50 vertices from visual hull, got {mesh.vertex_count}"
        )
        assert mesh.face_count > 0
