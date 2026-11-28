"""Persistent 3D point-cloud spatial memory for long-sequence consistency.

Maintains an incrementally updated scene point cloud built from FGPP
depth maps and camera intrinsics / extrinsics.  Each causal-window step
back-projects visible pixels into world space and merges them into a
voxel-hashed structure suitable for reference depth / normal
re-rendering during overlap fusion.

Coordinate convention (matching :class:`synapse.fgpp.CameraParams`):
  - Right-handed, Y-up, camera looks along ``-Z``.
  - Top of image (``y = 0``) corresponds to ``+Y_cam``.
  - Right of image (``x = W - 1``) corresponds to ``+X_cam``.

Design goals (Horizon 14 M57 — PR-228):
  - Incremental update latency ≤ 50 ms / frame (pure CPU NumPy).
  - Memory budget ≤ 200 MB for 1000-frame sequences.
  - Deterministic: identical inputs → identical point cloud state.

All heavy dependencies are deferred to function scope; module-level
imports are limited to ``numpy`` and stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Bias for packing signed 32-bit voxel coordinates into a single collision-free
# Python integer key: (vx + bias) << 64 | (vy + bias) << 32 | (vz + bias).
_VOXEL_INT32_MIN = -(1 << 31)
_VOXEL_INT32_MAX = (1 << 31) - 1
_VOXEL_INT32_BIAS = 1 << 31


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------


@dataclass
class SpatialMemoryConfig:
    """Tunables for the spatial memory.

    Attributes:
        voxel_size: Side length of each voxel cell (metres).
            Smaller → more detail, more memory.  Default 0.05 m.
        max_points: Hard cap on stored points.  Oldest points are
            evicted when this limit is reached.  Default 4_000_000
            (≈ 192 MB at 48 B/point).
        subsample_stride: Pixel stride for depth back-projection.
            ``1`` = every pixel, ``2`` = every other pixel, etc.
            Default 4 (16× reduction).
        depth_min_m: Minimum valid depth in metres.  Pixels with
            depth below this value are discarded (sky / background).
        depth_max_m: Maximum valid depth in metres.
        normal_estimation: Whether to estimate per-point normals from
            the depth map.  Adds ~20% overhead per frame.
    """

    voxel_size: float = 0.05
    max_points: int = 4_000_000
    subsample_stride: int = 4
    depth_min_m: float = 0.1
    depth_max_m: float = 100.0
    normal_estimation: bool = True


# -----------------------------------------------------------------------
# Point data record
# -----------------------------------------------------------------------


@dataclass
class SpatialPointCloud:
    """Accumulated world-space point cloud with optional normals.

    All arrays share the first dimension ``(N,)``.

    Attributes:
        positions: ``(N, 3)`` float32 world-space XYZ.
        normals: ``(N, 3)`` float32 world-space normals, or ``None``.
        frame_ids: ``(N,)`` int32 frame index each point was added from.
        count: Current number of points ``N``.
    """

    positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
    )
    normals: np.ndarray | None = None
    frame_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32),
    )

    @property
    def count(self) -> int:
        """Current number of stored points."""
        return int(self.positions.shape[0])

    @property
    def memory_bytes(self) -> int:
        """Approximate memory footprint in bytes."""
        total = self.positions.nbytes + self.frame_ids.nbytes
        if self.normals is not None:
            total += self.normals.nbytes
        return total


# -----------------------------------------------------------------------
# Core: SpatialMemory
# -----------------------------------------------------------------------


class SpatialMemory:
    """Incrementally updated scene point cloud for long-sequence drift suppression.

    Usage::

        mem = SpatialMemory(SpatialMemoryConfig())
        for frame_idx, fgpp_frame in enumerate(sequence.frames):
            mem.ingest_frame(frame_idx, fgpp_frame)
        ref_depth, ref_normal = mem.render_reference(camera_params)

    Args:
        config: ``SpatialMemoryConfig`` instance.
    """

    def __init__(self, config: SpatialMemoryConfig | None = None) -> None:
        self._config = config or SpatialMemoryConfig()
        if self._config.voxel_size <= 0.0:
            raise ValueError(
                f"voxel_size must be > 0, got {self._config.voxel_size}."
            )
        if self._config.max_points < 0:
            raise ValueError(
                f"max_points must be >= 0, got {self._config.max_points}."
            )
        self._cloud = SpatialPointCloud()
        self._ingested_frames: set[int] = set()
        # Voxel occupancy: packed int (or fallback 3-tuple) -> True for dedup,
        # paired in insertion order with self._cloud points for eviction sync.
        self._voxel_keys: list[int | tuple[int, int, int]] = []
        self._voxel_set: set[int | tuple[int, int, int]] = set()

    # -- Properties -------------------------------------------------------

    @property
    def config(self) -> SpatialMemoryConfig:
        """Active configuration."""
        return self._config

    @property
    def cloud(self) -> SpatialPointCloud:
        """Current accumulated point cloud (read-only snapshot)."""
        return self._cloud

    @property
    def num_points(self) -> int:
        """Number of points currently stored."""
        return self._cloud.count

    @property
    def ingested_frame_count(self) -> int:
        """Number of distinct frames ingested so far."""
        return len(self._ingested_frames)

    def has_frame(self, frame_index: int) -> bool:
        """Return ``True`` if *frame_index* has already been ingested."""
        return frame_index in self._ingested_frames

    # -- Core: ingest a frame ---------------------------------------------

    def ingest_frame(
        self,
        frame_index: int,
        fgpp_frame: Any,
    ) -> int:
        """Back-project a single FGPP frame into the spatial memory.

        Args:
            frame_index: Zero-based frame index (must be unique).
            fgpp_frame: An ``FGPPFrame`` with ``.depth`` ``(H, W)``
                float32 linear depth in metres, and ``.camera``
                ``CameraParams`` with intrinsics and extrinsics.

        Returns:
            Number of *new* points added after voxel dedup.

        Raises:
            ValueError: If ``frame_index`` was already ingested or
                the frame lacks required depth / camera data.
        """
        if frame_index in self._ingested_frames:
            raise ValueError(
                f"Frame {frame_index} already ingested into SpatialMemory."
            )

        depth = getattr(fgpp_frame, "depth", None)
        camera = getattr(fgpp_frame, "camera", None)
        if depth is None or camera is None:
            raise ValueError(
                "FGPPFrame must have .depth (H,W) and .camera (CameraParams)."
            )

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(
                f"Depth must be 2-D (H, W), got shape {depth.shape}."
            )

        h, w = depth.shape
        cfg = self._config
        stride = max(1, cfg.subsample_stride)

        # Build subsampled pixel-center grid (H', W')
        ys = np.arange(0, h, stride, dtype=np.float32) + 0.5
        xs = np.arange(0, w, stride, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        depth_sub = depth[::stride, ::stride]

        gx = grid_x.ravel()
        gy = grid_y.ravel()
        dz = depth_sub.ravel()

        # Filter by valid depth range
        valid = (dz >= cfg.depth_min_m) & (dz <= cfg.depth_max_m) & np.isfinite(dz)
        if not np.any(valid):
            self._ingested_frames.add(frame_index)
            return 0

        # Back-project full subsampled grid in OpenGL convention:
        # right = +X, up = +Y (top pixel y=0 -> +Y), forward = -Z.
        fx = float(camera.fx) if camera.fx > 0 else 1.0
        fy = float(camera.fy) if camera.fy > 0 else 1.0
        cx = float(camera.cx)
        cy = float(camera.cy)

        # Transform only valid pixels for world positions
        gx_v = gx[valid]
        gy_v = gy[valid]
        dz_v = dz[valid]

        cam_x = (gx_v - cx) * dz_v / fx
        cam_y = -(gy_v - cy) * dz_v / fy
        cam_z = -dz_v

        cam_pts = np.stack(
            [cam_x, cam_y, cam_z, np.ones_like(dz_v)],
            axis=-1,
        )  # (M, 4)
        world_matrix = np.asarray(camera.world_matrix, dtype=np.float64)
        world_pts = (world_matrix @ cam_pts.T).T[:, :3].astype(np.float32)

        # Optional normal estimation from the full subsampled grid,
        # then aligned with ``valid`` mask so indices match ``world_pts``.
        normals_valid: np.ndarray | None = None
        if cfg.normal_estimation:
            normals_grid = self._estimate_normals_from_depth(
                depth_sub, camera, world_matrix, stride=stride,
            )
            normals_valid = normals_grid[valid]

        # Vectorised intra-frame voxel deduplication, then global voxel set check.
        voxel_indices = np.floor(
            world_pts.astype(np.float64) / cfg.voxel_size
        ).astype(np.int64)

        contiguous_voxels = np.ascontiguousarray(voxel_indices)
        voxel_view = contiguous_voxels.view(np.dtype((np.void, 24))).ravel()
        _, first_idx = np.unique(voxel_view, return_index=True)
        # Preserve spatial scan order deterministically
        first_idx.sort()

        unique_voxels = voxel_indices[first_idx]
        candidate_keys = _pack_voxel_keys(unique_voxels)

        voxel_set = self._voxel_set
        kept_indices: list[int] = []
        new_keys: list[int | tuple[int, int, int]] = []
        for pos_idx, key in zip(first_idx.tolist(), candidate_keys, strict=True):
            if key not in voxel_set:
                voxel_set.add(key)
                kept_indices.append(pos_idx)
                new_keys.append(key)

        n_new = len(kept_indices)
        if n_new == 0:
            self._ingested_frames.add(frame_index)
            return 0

        keep_arr = np.asarray(kept_indices, dtype=np.intp)
        new_pts = world_pts[keep_arr]
        new_fids = np.full(n_new, frame_index, dtype=np.int32)

        self._voxel_keys.extend(new_keys)
        self._cloud.positions = np.concatenate(
            [self._cloud.positions, new_pts], axis=0,
        )
        self._cloud.frame_ids = np.concatenate(
            [self._cloud.frame_ids, new_fids], axis=0,
        )

        if cfg.normal_estimation and normals_valid is not None:
            new_normals_filtered = normals_valid[keep_arr]
            if self._cloud.normals is not None:
                self._cloud.normals = np.concatenate(
                    [self._cloud.normals, new_normals_filtered], axis=0,
                )
            else:
                self._cloud.normals = new_normals_filtered

        # Evict oldest if over budget
        self._evict_if_needed()

        self._ingested_frames.add(frame_index)
        return n_new

    # -- Render reference depth / normal from memory ----------------------

    def render_reference(
        self,
        camera: Any,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Render reference depth and normal maps from the spatial memory.

        Projects all stored points into the given camera and rasterises
        a nearest-point depth map (z-buffer) and optional normal map.

        Args:
            camera: ``CameraParams`` with intrinsics and extrinsics.
            width: Output width (defaults to ``camera.width``).
            height: Output height (defaults to ``camera.height``).

        Returns:
            ``(depth_map, normal_map)`` where ``depth_map`` is ``(H, W)``
            float32 with 0.0 for empty pixels, and ``normal_map`` is
            ``(H, W, 3)`` float32 or ``None`` if normals are unavailable.
        """
        w = width or int(camera.width)
        h = height or int(camera.height)

        depth_out = np.zeros((h, w), dtype=np.float32)
        normal_out: np.ndarray | None = None
        if self._cloud.normals is not None:
            normal_out = np.zeros((h, w, 3), dtype=np.float32)

        if self._cloud.count == 0:
            return depth_out, normal_out

        # World → camera
        view_matrix = np.asarray(camera.view_matrix, dtype=np.float64)
        pts_h = np.concatenate(
            [
                self._cloud.positions.astype(np.float64),
                np.ones((self._cloud.count, 1), dtype=np.float64),
            ],
            axis=-1,
        )
        cam_pts = (view_matrix @ pts_h.T).T  # (N, 4)
        cam_z = -cam_pts[:, 2]  # Positive depth (camera looks along -Z)

        # Filter behind camera
        in_front = cam_z > 0
        cam_pts = cam_pts[in_front]
        cam_z = cam_z[in_front]
        normals_valid = None
        if self._cloud.normals is not None:
            normals_valid = self._cloud.normals[in_front]

        if len(cam_z) == 0:
            return depth_out, normal_out

        # Project to pixel coordinates in OpenGL Y-up convention:
        # y_px = cy - y_cam * fy / z
        fx = float(camera.fx) if camera.fx > 0 else 1.0
        fy = float(camera.fy) if camera.fy > 0 else 1.0
        cx = float(camera.cx)
        cy = float(camera.cy)

        px = np.floor(cam_pts[:, 0] * fx / cam_z + cx).astype(np.int32)
        py = np.floor(cy - cam_pts[:, 1] * fy / cam_z).astype(np.int32)

        # Clip to image bounds
        in_bounds = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px = px[in_bounds]
        py = py[in_bounds]
        cam_z_valid = cam_z[in_bounds].astype(np.float32)
        if normals_valid is not None:
            normals_valid = normals_valid[in_bounds]

        # Z-buffer rasterisation (closest point wins)
        # Sort by depth descending so closer points overwrite farther ones
        sort_idx = np.argsort(-cam_z_valid)
        px = px[sort_idx]
        py = py[sort_idx]
        cam_z_valid = cam_z_valid[sort_idx]

        depth_out[py, px] = cam_z_valid
        if normal_out is not None and normals_valid is not None:
            normals_sorted = normals_valid[sort_idx]
            normal_out[py, px] = normals_sorted

        return depth_out, normal_out

    # -- Reset / stats ----------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._cloud = SpatialPointCloud()
        self._ingested_frames.clear()
        self._voxel_keys.clear()
        self._voxel_set.clear()

    def stats(self) -> dict[str, Any]:
        """Return diagnostic statistics."""
        return {
            "num_points": self._cloud.count,
            "num_frames_ingested": len(self._ingested_frames),
            "num_voxels": len(self._voxel_set),
            "memory_bytes": self._cloud.memory_bytes,
            "memory_mb": round(self._cloud.memory_bytes / (1024 * 1024), 2),
        }

    # -- Private helpers --------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Remove oldest points if over the max_points budget."""
        if self._cloud.count <= self._config.max_points:
            return
        excess = self._cloud.count - self._config.max_points
        evicted_keys = self._voxel_keys[:excess]
        self._voxel_set.difference_update(evicted_keys)
        self._voxel_keys = self._voxel_keys[excess:]
        self._cloud.positions = self._cloud.positions[excess:]
        self._cloud.frame_ids = self._cloud.frame_ids[excess:]
        if self._cloud.normals is not None:
            self._cloud.normals = self._cloud.normals[excess:]

    def _estimate_normals_from_depth(
        self,
        depth_sub: np.ndarray,
        camera: Any,
        world_matrix: np.ndarray,
        stride: int = 1,
    ) -> np.ndarray:
        """Estimate outward surface normals from depth gradients.

        Uses OpenGL camera convention (``+Y`` up, ``+X`` right, looks ``-Z``)
        and accounts for pixel stride scaling ``dx_cam = stride * z / fx``.

        Returns ``(H' * W', 3)`` float32 normals matching the flattened
        ``depth_sub`` grid.
        """
        h, w = depth_sub.shape
        dz_dx = np.zeros_like(depth_sub, dtype=np.float32)
        dz_dy = np.zeros_like(depth_sub, dtype=np.float32)
        if w > 1:
            dz_dx[:, :-1] = depth_sub[:, 1:] - depth_sub[:, :-1]
        if h > 1:
            dz_dy[:-1, :] = depth_sub[1:, :] - depth_sub[:-1, :]

        fx = float(camera.fx) if camera.fx > 0 else 1.0
        fy = float(camera.fy) if camera.fy > 0 else 1.0
        safe_z = np.maximum(np.abs(depth_sub), 1e-4)

        # Tangent_x = (stride * z / fx, 0, -dz_dx)
        # Tangent_y = (0, -stride * z / fy, -dz_dy)
        # Normal facing camera (+Z_cam) = ty x tx -> (dz_dx*fx/(stride*z), -dz_dy*fy/(stride*z), 1)
        nx_cam = dz_dx * fx / (stride * safe_z)
        ny_cam = -dz_dy * fy / (stride * safe_z)
        nz_cam = np.ones_like(depth_sub, dtype=np.float32)

        length = np.sqrt(nx_cam**2 + ny_cam**2 + nz_cam**2)
        length = np.maximum(length, 1e-8)
        nx_cam /= length
        ny_cam /= length
        nz_cam /= length

        normals_cam = np.stack(
            [nx_cam.ravel(), ny_cam.ravel(), nz_cam.ravel()],
            axis=-1,
        ).astype(np.float32)

        # Rotate to world space (upper-left 3x3 of world_matrix)
        rot = world_matrix[:3, :3].astype(np.float32)
        normals_world = (rot @ normals_cam.T).T

        norms = np.linalg.norm(normals_world, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals_world = normals_world / norms

        return np.asarray(normals_world, dtype=np.float32)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _pack_voxel_keys(
    voxel_indices: np.ndarray,
) -> list[int | tuple[int, int, int]]:
    """Pack ``(K, 3)`` int64 voxel coords into collision-free hashable keys."""
    if len(voxel_indices) == 0:
        return []
    vmin = int(voxel_indices.min())
    vmax = int(voxel_indices.max())
    if vmin >= _VOXEL_INT32_MIN and vmax <= _VOXEL_INT32_MAX:
        shifted = voxel_indices + _VOXEL_INT32_BIAS
        rows = shifted.tolist()
        return [(r[0] << 64) | (r[1] << 32) | r[2] for r in rows]
    return [(int(r[0]), int(r[1]), int(r[2])) for r in voxel_indices.tolist()]


def depth_to_world_points(
    depth: np.ndarray,
    camera: Any,
    stride: int = 1,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
) -> np.ndarray:
    """Back-project a depth map to world-space points.

    Utility function for external consumers who want to perform custom
    point cloud operations without going through ``SpatialMemory``.

    Args:
        depth: ``(H, W)`` float32 linear depth in metres.
        camera: ``CameraParams`` with intrinsics and ``world_matrix``.
        stride: Pixel sampling stride (default 1 = every pixel).
        depth_min: Minimum valid depth.
        depth_max: Maximum valid depth.

    Returns:
        ``(N, 3)`` float32 world-space points.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth, got shape {depth.shape}.")

    h, w = depth.shape
    stride = max(1, stride)

    ys = np.arange(0, h, stride, dtype=np.float32) + 0.5
    xs = np.arange(0, w, stride, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    dz = depth[::stride, ::stride].ravel()
    gx = grid_x.ravel()
    gy = grid_y.ravel()

    valid = (dz >= depth_min) & (dz <= depth_max) & np.isfinite(dz)
    gx = gx[valid]
    gy = gy[valid]
    dz = dz[valid]

    if len(dz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    fx = float(camera.fx) if camera.fx > 0 else 1.0
    fy = float(camera.fy) if camera.fy > 0 else 1.0
    cx = float(camera.cx)
    cy = float(camera.cy)

    cam_x = (gx - cx) * dz / fx
    cam_y = -(gy - cy) * dz / fy
    cam_z = -dz

    cam_pts = np.stack([cam_x, cam_y, cam_z, np.ones_like(dz)], axis=-1)
    world_matrix = np.asarray(camera.world_matrix, dtype=np.float64)
    world_pts = (world_matrix @ cam_pts.T).T[:, :3].astype(np.float32)

    return np.asarray(world_pts)


def compute_reference_consistency_score(
    reference_depth: np.ndarray,
    actual_depth: np.ndarray,
    threshold_m: float = 0.5,
) -> float:
    """Compute consistency between reference depth and actual depth.

    Measures how well the actual depth map matches the spatial memory's
    reference.  Useful for detecting drift or scene changes.

    Args:
        reference_depth: ``(H, W)`` reference depth from ``render_reference``.
        actual_depth: ``(H, W)`` actual FGPP depth.
        threshold_m: Maximum absolute depth error considered consistent.

    Returns:
        Fraction of valid pixels within the threshold ∈ ``[0.0, 1.0]``.
    """
    ref = np.asarray(reference_depth, dtype=np.float32)
    act = np.asarray(actual_depth, dtype=np.float32)

    # Only compare where both have valid (> 0) depth
    valid = (ref > 0) & (act > 0) & np.isfinite(ref) & np.isfinite(act)
    if not np.any(valid):
        return 0.0

    diff = np.abs(ref[valid] - act[valid])
    consistent = float(np.mean(diff <= threshold_m))
    return consistent
