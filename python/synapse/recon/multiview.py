"""Multi-view 3D mesh initialization via TSDF fusion or visual hull.

Provides ``MultiViewInitializer`` which replaces the hardcoded 8-vertex box
fallback in :class:`~synapse.recon.pipeline.ReconPipeline` with a proper
volumetric reconstruction from multi-view depth maps or silhouette masks.

All heavy dependencies (``torch``, ``trimesh``, ``open3d``) are lazy-imported
inside methods to keep module-level import lightweight (numpy + synapse only).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from synapse.errors import ContractValidationError
from synapse.recon.types import CameraExtrinsics, MeshGeometry

if TYPE_CHECKING:
    pass  # reserved for optional heavy type stubs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CAMERAS = 2
_MIN_ANGULAR_SPAN_DEG = 15.0


# ---------------------------------------------------------------------------
# Pure-numpy marching cubes (simplified surface-net extraction)
# ---------------------------------------------------------------------------


def _marching_cubes_naive(
    volume: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an iso-surface mesh from a signed-distance volume.

    Uses a simplified dual-contouring / surface-net approach: vertices are
    placed at zero-crossing midpoints between neighbouring voxels, and faces
    are emitted for each edge that straddles the zero level-set.

    Args:
        volume: 3-D SDF array, shape ``(Nx, Ny, Nz)``.
        voxel_size: Physical size of a single voxel in metres.
        origin: World-space origin of the ``[0, 0, 0]`` voxel corner, shape ``(3,)``.

    Returns:
        ``(vertices, faces)`` – ``float32 (V, 3)`` and ``int32 (F, 3)``.
    """
    nx, ny, nz = volume.shape

    # Find zero-crossings along each axis.
    vertex_map: dict[tuple[int, int, int], int] = {}
    vertices: list[np.ndarray] = []

    def _vert_id(ix: int, iy: int, iz: int) -> int:
        key = (ix, iy, iz)
        if key not in vertex_map:
            pos = origin + np.array([ix, iy, iz], dtype=np.float32) * voxel_size
            vertex_map[key] = len(vertices)
            vertices.append(pos)
        return vertex_map[key]

    faces: list[tuple[int, int, int]] = []

    # Scan interior voxels and emit quads (as two triangles) on sign changes.
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                v0 = volume[ix, iy, iz]
                # Check x-edge
                if volume[ix + 1, iy, iz] * v0 < 0:
                    a = _vert_id(ix, iy, iz)
                    b = _vert_id(ix, iy + 1, iz)
                    c = _vert_id(ix, iy + 1, iz + 1)
                    d = _vert_id(ix, iy, iz + 1)
                    if v0 < 0:
                        faces.append((a, b, c))
                        faces.append((a, c, d))
                    else:
                        faces.append((a, c, b))
                        faces.append((a, d, c))
                # Check y-edge
                if volume[ix, iy + 1, iz] * v0 < 0:
                    a = _vert_id(ix, iy, iz)
                    b = _vert_id(ix + 1, iy, iz)
                    c = _vert_id(ix + 1, iy, iz + 1)
                    d = _vert_id(ix, iy, iz + 1)
                    if v0 < 0:
                        faces.append((a, c, b))
                        faces.append((a, d, c))
                    else:
                        faces.append((a, b, c))
                        faces.append((a, c, d))
                # Check z-edge
                if volume[ix, iy, iz + 1] * v0 < 0:
                    a = _vert_id(ix, iy, iz)
                    b = _vert_id(ix + 1, iy, iz)
                    c = _vert_id(ix + 1, iy + 1, iz)
                    d = _vert_id(ix, iy + 1, iz)
                    if v0 < 0:
                        faces.append((a, b, c))
                        faces.append((a, c, d))
                    else:
                        faces.append((a, c, b))
                        faces.append((a, d, c))

    if len(vertices) == 0:
        # Degenerate: return minimal valid mesh (single triangle)
        verts_arr = np.zeros((3, 3), dtype=np.float32)
        faces_arr = np.array([[0, 1, 2]], dtype=np.int32)
        return verts_arr, faces_arr

    verts_arr = np.array(vertices, dtype=np.float32)
    faces_arr = np.array(faces, dtype=np.int32)
    return verts_arr, faces_arr


# ---------------------------------------------------------------------------
# MultiViewInitializer
# ---------------------------------------------------------------------------


class MultiViewInitializer:
    """Multi-view depth-based 3D mesh initializer using TSDF fusion / visual hull.

    Provides two reconstruction strategies:

    1. **TSDF fusion** (when ``depths`` are provided): integrates per-view
       metric depth maps into a truncated signed-distance volume, then
       extracts the iso-surface via marching cubes.
    2. **Visual hull** (when only ``masks`` are provided): carves a voxel
       grid using binary silhouette masks from each camera view.

    Both paths are implemented in pure numpy — no GPU required.
    """

    def __init__(
        self,
        voxel_size: float = 0.01,
        sdf_trunc: float = 0.04,
        volume_bounds: tuple[float, float, float] = (2.0, 2.0, 2.0),
    ) -> None:
        """Initialise the multi-view reconstructor.

        Args:
            voxel_size: Edge length of a single voxel in metres.
            sdf_trunc: Truncation distance for TSDF integration (metres).
            volume_bounds: ``(x, y, z)`` extent of the reconstruction volume
                centred on the world origin, in metres.
        """
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.volume_bounds = volume_bounds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        frames: np.ndarray,
        cameras: list[CameraExtrinsics],
        masks: np.ndarray | None = None,
        depths: np.ndarray | None = None,
    ) -> MeshGeometry:
        """Reconstruct a 3D mesh from multi-view observations.

        Args:
            frames: RGB frames, shape ``(N, H, W, 3)``, ``uint8`` or ``float32``.
            cameras: Per-frame camera extrinsics (length ``N``).
            masks: Optional binary silhouette masks, shape ``(N, H, W)``.
            depths: Optional metric depth maps, shape ``(N, H, W)`` in metres.

        Returns:
            Reconstructed :class:`MeshGeometry`.

        Raises:
            ContractValidationError: If fewer than 2 cameras are supplied or
                the angular span across cameras is below 15°.
        """
        self._validate_cameras(cameras)

        if depths is not None:
            return self._reconstruct_tsdf(frames, cameras, depths)

        if masks is not None:
            return self._reconstruct_visual_hull(frames, cameras, masks)

        # No depths or masks provided — attempt to derive foreground masks
        # from frames via adaptive Otsu thresholding on grayscale variance,
        # then fall back to visual hull reconstruction.
        derived_masks = self._derive_foreground_masks(frames)
        if derived_masks is not None:
            return self._reconstruct_visual_hull(frames, cameras, derived_masks)

        # Foreground separation failed completely — raise structured error
        # instead of silently producing a flat plane from constant depth.
        raise ContractValidationError(
            "Cannot reconstruct 3D mesh: neither depth maps nor silhouette "
            "masks were provided, and automatic foreground estimation from "
            f"{len(cameras)} input frames failed (frames may be uniform or "
            "lack sufficient contrast for foreground/background separation).",
            agent_fix_hint=(
                "Provide explicit `depths` (metric depth maps) or `masks` "
                "(binary silhouette masks) to MultiViewInitializer.reconstruct(). "
                "Alternatively, ensure input frames have visible foreground "
                "objects with sufficient contrast against the background."
            ),
        )

    # ------------------------------------------------------------------
    # Automatic foreground mask derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_foreground_masks(frames: np.ndarray) -> np.ndarray | None:
        """Attempt to derive binary foreground masks from RGB frames.

        Uses a combination of per-frame Otsu thresholding on grayscale
        intensity and cross-frame variance to separate foreground from
        background.  Returns ``None`` if the frames are too uniform
        (e.g. all-black, all-white) to produce meaningful masks.

        Args:
            frames: RGB frames, shape ``(N, H, W, 3)``, uint8 or float32.

        Returns:
            Binary masks ``(N, H, W)`` float32 or ``None`` on failure.
        """
        n, h, w, _ = frames.shape

        # Convert to float [0, 1] grayscale
        if frames.dtype == np.uint8:
            gray = np.mean(frames.astype(np.float32) / 255.0, axis=-1)
        else:
            gray = np.mean(frames.astype(np.float32), axis=-1)

        # Cross-frame variance — pixels that change across views are
        # more likely foreground (parallax).
        if n > 1:
            frame_variance = np.var(gray, axis=0)  # (H, W)
            variance_useful = float(np.max(frame_variance)) > 1e-4
        else:
            frame_variance = np.zeros((h, w), dtype=np.float32)
            variance_useful = False

        masks = np.zeros((n, h, w), dtype=np.float32)
        any_foreground = False

        for i in range(n):
            g = gray[i]

            # Otsu's method: find threshold that minimises intra-class
            # variance on a 256-bin histogram.
            hist_bins = 256
            g_uint8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
            hist = np.bincount(g_uint8.ravel(), minlength=hist_bins).astype(
                np.float64
            )
            total = g_uint8.size
            if total == 0:
                continue

            sum_total = np.dot(np.arange(hist_bins, dtype=np.float64), hist)
            sum_bg = 0.0
            weight_bg = 0.0
            max_var = 0.0
            threshold = 0

            for t in range(hist_bins):
                weight_bg += hist[t]
                if weight_bg == 0:
                    continue
                weight_fg = total - weight_bg
                if weight_fg == 0:
                    break
                sum_bg += t * hist[t]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg
                between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
                if between_var > max_var:
                    max_var = between_var
                    threshold = t

            # Apply threshold — foreground is typically the brighter
            # object on a dark background, or vice versa.
            mask_bright = g_uint8 > threshold
            mask_dark = g_uint8 <= threshold

            # Heuristic: pick the smaller region as foreground
            # (foreground object usually occupies < 50% of image).
            bright_frac = np.mean(mask_bright)
            if 0.05 < bright_frac < 0.95:
                if bright_frac < 0.5:
                    mask = mask_bright
                else:
                    mask = mask_dark
            else:
                # Nearly uniform — Otsu cannot separate.
                # Try cross-frame variance if available.
                if variance_useful:
                    var_thresh = float(np.percentile(frame_variance, 75))
                    mask = frame_variance > var_thresh
                else:
                    continue

            # Combine with cross-frame variance when both are useful
            if variance_useful and 0.05 < bright_frac < 0.95:
                var_thresh = float(np.percentile(frame_variance, 50))
                var_mask = frame_variance > var_thresh
                mask = mask | var_mask

            fg_fraction = float(np.mean(mask))
            if 0.01 < fg_fraction < 0.99:
                masks[i] = mask.astype(np.float32)
                any_foreground = True

        if not any_foreground:
            return None

        return masks

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_cameras(cameras: list[CameraExtrinsics]) -> None:
        """Ensure camera configuration meets minimum requirements."""
        if len(cameras) < _MIN_CAMERAS:
            raise ContractValidationError(
                f"Multi-view reconstruction requires at least {_MIN_CAMERAS} "
                f"cameras, but only {len(cameras)} supplied.",
                agent_fix_hint=(
                    "Provide at least 2 calibrated camera views with "
                    "sufficient baseline. Re-run tracking with more frames "
                    "or a wider camera orbit."
                ),
            )

        # Compute angular span from camera positions (directions from origin).
        positions = np.array([c.position for c in cameras], dtype=np.float64)
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        # Avoid division by zero for cameras at the origin.
        norms = np.clip(norms, 1e-12, None)
        dirs_normalised = positions / norms

        # Maximum pairwise angular separation.
        max_angle_deg = 0.0
        n = len(dirs_normalised)
        for i in range(n):
            for j in range(i + 1, n):
                cos_val = float(np.dot(dirs_normalised[i], dirs_normalised[j]))
                cos_val = np.clip(cos_val, -1.0, 1.0)
                angle = math.degrees(math.acos(cos_val))
                if angle > max_angle_deg:
                    max_angle_deg = angle

        if max_angle_deg < _MIN_ANGULAR_SPAN_DEG:
            raise ContractValidationError(
                f"Camera angular span is {max_angle_deg:.1f}°, which is "
                f"below the minimum {_MIN_ANGULAR_SPAN_DEG}° required for "
                f"reliable multi-view reconstruction.",
                agent_fix_hint=(
                    "Widen the camera orbit or add views from different "
                    "angles. The maximum pairwise angular separation must "
                    f"be >= {_MIN_ANGULAR_SPAN_DEG}°."
                ),
            )

    # ------------------------------------------------------------------
    # TSDF fusion
    # ------------------------------------------------------------------

    def _reconstruct_tsdf(
        self,
        frames: np.ndarray,
        cameras: list[CameraExtrinsics],
        depths: np.ndarray,
    ) -> MeshGeometry:
        """Volumetric TSDF fusion followed by marching cubes."""
        bx, by, bz = self.volume_bounds
        origin = np.array([-bx / 2, -by / 2, -bz / 2], dtype=np.float32)
        nx = int(bx / self.voxel_size)
        ny = int(by / self.voxel_size)
        nz = int(bz / self.voxel_size)

        # Limit resolution for memory safety (pure-numpy, must fit in RAM).
        max_dim = 128
        if max(nx, ny, nz) > max_dim:
            scale = max_dim / max(nx, ny, nz)
            nx = max(int(nx * scale), 2)
            ny = max(int(ny * scale), 2)
            nz = max(int(nz * scale), 2)
            effective_voxel = max(bx / nx, by / ny, bz / nz)
        else:
            effective_voxel = self.voxel_size

        tsdf = np.ones((nx, ny, nz), dtype=np.float32)
        weight = np.zeros((nx, ny, nz), dtype=np.float32)

        # Pre-compute voxel centres.
        xs = origin[0] + (np.arange(nx, dtype=np.float32) + 0.5) * effective_voxel
        ys = origin[1] + (np.arange(ny, dtype=np.float32) + 0.5) * effective_voxel
        zs = origin[2] + (np.arange(nz, dtype=np.float32) + 0.5) * effective_voxel

        # Meshgrid of world-space voxel centres → (nx*ny*nz, 3)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        world_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)  # (M, 3)

        n_views = len(cameras)
        h, w = depths.shape[1], depths.shape[2]

        for vi in range(n_views):
            cam = cameras[vi]
            depth_map = depths[vi]

            # World → camera transform.
            T = cam.transform.astype(np.float64)
            R = T[:3, :3]
            t = T[:3, 3]
            # Camera-space points.
            cam_pts = (R @ world_pts.T).T + t  # (M, 3)

            # Simple pinhole: assume fx=fy=w, cx=w/2, cy=h/2.
            fx = fy = float(w)
            cx, cy = w / 2.0, h / 2.0

            z_cam = cam_pts[:, 2]
            valid_z = z_cam > 1e-6
            u = np.full(len(cam_pts), -1.0)
            v = np.full(len(cam_pts), -1.0)
            u[valid_z] = fx * cam_pts[valid_z, 0] / z_cam[valid_z] + cx
            v[valid_z] = fy * cam_pts[valid_z, 1] / z_cam[valid_z] + cy

            # Round to pixel indices.
            ui = np.round(u).astype(np.int64)
            vi_arr = np.round(v).astype(np.int64)

            in_bounds = valid_z & (ui >= 0) & (ui < w) & (vi_arr >= 0) & (vi_arr < h)

            # Sample depth.
            depth_sampled = np.zeros(len(cam_pts), dtype=np.float32)
            if np.any(in_bounds):
                depth_sampled[in_bounds] = depth_map[vi_arr[in_bounds], ui[in_bounds]]

            # SDF = observed depth - voxel depth along ray.
            sdf = depth_sampled - z_cam.astype(np.float32)

            # Truncation.
            valid = in_bounds & (depth_sampled > 0) & (np.abs(sdf) < self.sdf_trunc)

            tsdf_update = np.clip(sdf / self.sdf_trunc, -1.0, 1.0)

            w_old = weight.ravel()
            tsdf_old = tsdf.ravel()
            w_new = np.where(valid, w_old + 1.0, w_old)
            tsdf_new = np.where(
                valid,
                (tsdf_old * w_old + tsdf_update) / np.clip(w_new, 1e-8, None),
                tsdf_old,
            )
            tsdf = tsdf_new.reshape(nx, ny, nz)
            weight = w_new.reshape(nx, ny, nz)

        verts, faces = _marching_cubes_naive(tsdf, effective_voxel, origin)
        return MeshGeometry(vertices=verts, faces=faces)

    # ------------------------------------------------------------------
    # Visual hull
    # ------------------------------------------------------------------

    def _reconstruct_visual_hull(
        self,
        frames: np.ndarray,
        cameras: list[CameraExtrinsics],
        masks: np.ndarray,
    ) -> MeshGeometry:
        """Silhouette-based space carving (visual hull)."""
        bx, by, bz = self.volume_bounds
        origin = np.array([-bx / 2, -by / 2, -bz / 2], dtype=np.float32)
        nx = int(bx / self.voxel_size)
        ny = int(by / self.voxel_size)
        nz = int(bz / self.voxel_size)

        max_dim = 64
        if max(nx, ny, nz) > max_dim:
            scale = max_dim / max(nx, ny, nz)
            nx = max(int(nx * scale), 2)
            ny = max(int(ny * scale), 2)
            nz = max(int(nz * scale), 2)
            effective_voxel = max(bx / nx, by / ny, bz / nz)
        else:
            effective_voxel = self.voxel_size

        # Start with all voxels occupied.
        occupied = np.ones((nx, ny, nz), dtype=bool)

        xs = origin[0] + (np.arange(nx, dtype=np.float32) + 0.5) * effective_voxel
        ys = origin[1] + (np.arange(ny, dtype=np.float32) + 0.5) * effective_voxel
        zs = origin[2] + (np.arange(nz, dtype=np.float32) + 0.5) * effective_voxel
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        world_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        n_views = len(cameras)
        h, w = masks.shape[1], masks.shape[2]

        for vi in range(n_views):
            cam = cameras[vi]
            mask = masks[vi]

            T = cam.transform.astype(np.float64)
            R = T[:3, :3]
            t = T[:3, 3]
            cam_pts = (R @ world_pts.T).T + t

            fx = fy = float(w)
            cx, cy = w / 2.0, h / 2.0

            z_cam = cam_pts[:, 2]
            valid_z = z_cam > 1e-6
            u = np.full(len(cam_pts), -1.0)
            v = np.full(len(cam_pts), -1.0)
            u[valid_z] = fx * cam_pts[valid_z, 0] / z_cam[valid_z] + cx
            v[valid_z] = fy * cam_pts[valid_z, 1] / z_cam[valid_z] + cy

            ui = np.round(u).astype(np.int64)
            vi_arr = np.round(v).astype(np.int64)

            in_bounds = valid_z & (ui >= 0) & (ui < w) & (vi_arr >= 0) & (vi_arr < h)

            # Carve: if projected pixel is NOT inside silhouette → remove voxel.
            in_silhouette = np.zeros(len(cam_pts), dtype=bool)
            if np.any(in_bounds):
                in_silhouette[in_bounds] = mask[vi_arr[in_bounds], ui[in_bounds]] > 0.5

            # Voxels outside image bounds are also carved.
            carve = ~in_silhouette
            occupied_flat = occupied.ravel()
            occupied_flat &= ~carve
            occupied = occupied_flat.reshape(nx, ny, nz)

        # Convert occupied voxel grid to SDF-like field for marching cubes.
        # occupied=True → SDF < 0 (inside), occupied=False → SDF > 0 (outside).
        sdf = np.where(occupied, -1.0, 1.0).astype(np.float32)
        verts, faces = _marching_cubes_naive(sdf, effective_voxel, origin)
        return MeshGeometry(vertices=verts, faces=faces)
