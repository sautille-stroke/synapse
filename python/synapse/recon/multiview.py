"""Multi-view 3D mesh initialization via TSDF fusion or visual hull.

Provides ``MultiViewInitializer`` which replaces the hardcoded 8-vertex box
fallback in :class:`~synapse.recon.pipeline.ReconPipeline` with a proper
volumetric reconstruction from multi-view depth maps or silhouette masks.

All heavy dependencies (``torch``, ``trimesh``, ``open3d``) are lazy-imported
inside methods to keep module-level import lightweight (numpy + synapse only).

World-to-Camera (w2c) convention
---------------------------------
All ``CameraExtrinsics.transform`` matrices consumed by this module follow
the **world-to-camera (w2c)** convention:

    p_cam = R @ p_world + t          (3-vector form)
    p_cam = T @ [p_world; 1]         (4x4 homogeneous form)

Where ``T[:3,:3] = R`` (rotation) and ``T[:3,3] = t`` (translation).
The camera position in world space is ``CameraExtrinsics.position = -R^T @ t``.
This convention is consistent with ``synapse.recon.tracking`` output.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from synapse.errors import ContractValidationError
from synapse.recon._tri_table import TRI_TABLE
from synapse.recon.types import CameraExtrinsics, MeshGeometry

if TYPE_CHECKING:
    pass  # reserved for optional heavy type stubs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CAMERAS = 2
_MIN_ANGULAR_SPAN_DEG = 15.0

_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int32,
)

_EDGE_CORNERS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


# ---------------------------------------------------------------------------
# 256 Marching Cubes with canonical edge sharing and linear interpolation
# ---------------------------------------------------------------------------


def _marching_cubes(
    volume: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    isovalue: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an iso-surface mesh from a signed-distance volume.

    Uses full 256-entry triangulation table (Bourke 1994) with linear
    interpolation along intersected edges and canonical edge deduplication,
    guaranteeing a closed manifold surface.

    Args:
        volume: 3-D SDF array, shape ``(Nx, Ny, Nz)``. Inside < isovalue.
        voxel_size: Physical size of a single voxel in metres.
        origin: World-space origin of the ``[0, 0, 0]`` voxel corner, shape ``(3,)``.
        isovalue: Iso-surface threshold (default 0.0).

    Returns:
        ``(vertices, faces)`` – ``float32 (V, 3)`` and ``int32 (F, 3)``.
    """
    nx, ny, nz = volume.shape
    vertex_map: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []

    # NumPy active-voxel mask pre-filter: skip cells where all 8 corners
    # share the same sign (~400x speedup for sparse grids).
    inside = volume < isovalue
    active = np.zeros((max(nx - 1, 0), max(ny - 1, 0), max(nz - 1, 0)), dtype=bool)
    for di, dj, dk in [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]:
        active |= (
            inside[di : di + active.shape[0], dj : dj + active.shape[1], dk : dk + active.shape[2]]
            != inside[: active.shape[0], : active.shape[1], : active.shape[2]]
        )
    active_cells = np.argwhere(active)

    for cell in active_cells:
        ix, iy, iz = int(cell[0]), int(cell[1]), int(cell[2])
        c_vals = [
            float(volume[ix, iy, iz]),
            float(volume[ix + 1, iy, iz]),
            float(volume[ix + 1, iy + 1, iz]),
            float(volume[ix, iy + 1, iz]),
            float(volume[ix, iy, iz + 1]),
            float(volume[ix + 1, iy, iz + 1]),
            float(volume[ix + 1, iy + 1, iz + 1]),
            float(volume[ix, iy + 1, iz + 1]),
        ]
        cubeindex = 0
        for k in range(8):
            if c_vals[k] < isovalue:
                cubeindex |= 1 << k

        if cubeindex == 0 or cubeindex == 255:
            continue

        tris = TRI_TABLE[cubeindex]
        if not tris:
            continue

        edge_vids: dict[int, int] = {}
        for edge_idx in set(tris):
            ca, cb = _EDGE_CORNERS[edge_idx]
            ga = (ix + int(_CORNERS[ca][0]), iy + int(_CORNERS[ca][1]), iz + int(_CORNERS[ca][2]))
            gb = (ix + int(_CORNERS[cb][0]), iy + int(_CORNERS[cb][1]), iz + int(_CORNERS[cb][2]))
            edge_key = (min(ga, gb), max(ga, gb))

            if edge_key not in vertex_map:
                va = c_vals[ca]
                vb = c_vals[cb]
                denom = vb - va
                mu = (isovalue - va) / denom if abs(denom) > 1e-12 else 0.5
                mu = float(np.clip(mu, 0.0, 1.0))
                pos_a = origin + np.array(ga, dtype=np.float32) * voxel_size
                pos_b = origin + np.array(gb, dtype=np.float32) * voxel_size
                pos = pos_a + mu * (pos_b - pos_a)
                vid = len(vertices)
                vertices.append(pos)
                vertex_map[edge_key] = vid
            edge_vids[edge_idx] = vertex_map[edge_key]

        for t in range(0, len(tris), 3):
            # CCW winding: swap indices 1 and 2 for outward-facing normals
            faces.append((edge_vids[tris[t]], edge_vids[tris[t + 2]], edge_vids[tris[t + 1]]))

    if not vertices or not faces:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def _weld_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Weld duplicate vertices within epsilon distance.

    Merges vertices whose positions are within *epsilon* Euclidean distance,
    remaps face indices to the merged vertex set, and removes degenerate faces.

    Args:
        vertices: Vertex positions, shape ``(V, 3)``, ``float32``.
        faces: Triangle indices, shape ``(F, 3)``, ``int32``.
        epsilon: Maximum distance to consider vertices identical.

    Returns:
        ``(welded_vertices, remapped_faces)`` with compacted vertex array.
    """
    if len(vertices) == 0:
        return vertices, faces

    # Quantize positions to grid cells of size epsilon and merge
    quantized = np.round(vertices / max(epsilon, 1e-12)).astype(np.int64)
    # Create unique keys from quantized coordinates
    keys = quantized[:, 0] * 1_000_000_007 + quantized[:, 1] * 1_000_003 + quantized[:, 2]

    # Find unique vertices
    _, unique_indices, inverse_map = np.unique(keys, return_index=True, return_inverse=True)
    welded_verts = vertices[unique_indices]

    # Remap face indices
    remapped_faces = inverse_map[faces.ravel()].reshape(faces.shape).astype(np.int32)

    # Remove degenerate faces (where welding collapsed a triangle)
    if remapped_faces.ndim == 2 and remapped_faces.shape[1] == 3:
        non_degen = (
            (remapped_faces[:, 0] != remapped_faces[:, 1])
            & (remapped_faces[:, 1] != remapped_faces[:, 2])
            & (remapped_faces[:, 0] != remapped_faces[:, 2])
        )
        remapped_faces = remapped_faces[non_degen]

    return welded_verts.astype(np.float32), remapped_faces


# Keep backward-compatible alias
_marching_cubes_naive = _marching_cubes


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
            hist = np.bincount(g_uint8.ravel(), minlength=hist_bins).astype(np.float64)
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

            # Remove isolated 1-pixel salt-and-pepper noise via 3x3 majority vote
            padded = np.pad(mask.astype(np.float32), 1, mode="edge")
            neighbor_sum = np.zeros((h, w), dtype=np.float32)
            for dy in range(3):
                for dx in range(3):
                    neighbor_sum += padded[dy : dy + h, dx : dx + w]
            mask = neighbor_sum >= 5.0

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
        """Ensure camera configuration meets minimum requirements.

        Validates:
        1. Minimum camera count >= 2.
        2. Angular span >= 15° using direction from volume center (origin)
           to each camera position, rejecting zero-baseline configurations.
        """
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

        positions = np.array([c.position for c in cameras], dtype=np.float64)

        # Zero-baseline rejection: check all cameras are not at the same point
        pos_spread = np.max(np.ptp(positions, axis=0))
        if pos_spread < 1e-6:
            raise ContractValidationError(
                "All cameras are at the same position (zero-baseline). "
                "Multi-view reconstruction requires parallax between views.",
                agent_fix_hint=(
                    "Ensure cameras have different positions with sufficient "
                    "baseline separation. Zero-baseline setups cannot "
                    "triangulate 3D geometry."
                ),
            )

        # Compute angular span using camera view directions (optical axes).
        # Using directions from the centroid to each camera fails when cameras
        # are arranged symmetrically (e.g., ring orbit around origin) because
        # directions from the centroid are zero or near-zero.  Instead, we use
        # the camera's forward (Z-axis) direction from the rotation matrix.
        view_dirs = np.array(
            [c.rotation_matrix[2, :] for c in cameras], dtype=np.float64,
        )
        view_norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_norms = np.clip(view_norms, 1e-12, None)
        view_dirs_normalised = view_dirs / view_norms

        # Maximum pairwise angular separation between view directions.
        max_angle_deg = 0.0
        n = len(view_dirs_normalised)
        for i in range(n):
            for j in range(i + 1, n):
                cos_val = float(np.dot(view_dirs_normalised[i], view_dirs_normalised[j]))
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

    @staticmethod
    def _compute_volume_origin(
        cameras: list[CameraExtrinsics],
        volume_bounds: tuple[float, float, float],
    ) -> np.ndarray:
        """Compute world-space origin of the reconstruction volume.

        For convergent orbits (e.g. ring or hemisphere), centers the volume
        at the cameras' optical focal intersection point (e.g. [0,0,0]).
        For forward-translating cameras (e.g. video tracking), places the
        volume in front of the forward-most camera along the average optical axis.
        """
        bx, by, bz = volume_bounds
        half_extents = np.array([bx / 2.0, by / 2.0, bz / 2.0], dtype=np.float32)

        if not cameras:
            return -half_extents

        positions = np.array([c.position for c in cameras], dtype=np.float64)
        dirs = np.array([c.rotation_matrix[2, :] for c in cameras], dtype=np.float64)
        n = len(cameras)

        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        for C, d in zip(positions, dirs, strict=False):
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-8:
                d = d / d_norm
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ C

        eigvals = np.linalg.eigvalsh(A)
        # If cameras converge (e.g. ring orbit), all eigenvalues are large
        if eigvals[0] >= 0.05 * n:
            try:
                center = np.linalg.solve(A, b)
                return center.astype(np.float32) - half_extents
            except np.linalg.LinAlgError:
                pass

        # Parallel or near-parallel rays (forward motion)
        d_sum = np.sum(dirs, axis=0)
        d_norm = np.linalg.norm(d_sum)
        d_avg = d_sum / d_norm if d_norm > 1e-4 else np.array([0.0, 0.0, 1.0])

        max_bound = max(bx, by, bz)
        d_focus = max(1.5, max_bound * 0.75)
        max_proj = float(np.max(np.dot(positions, d_avg)))
        z_target = max_proj + d_focus

        reg = 0.1 * n
        A_reg = A + reg * np.outer(d_avg, d_avg)
        b_reg = b + reg * z_target * d_avg
        try:
            center = np.linalg.solve(A_reg, b_reg)
        except np.linalg.LinAlgError:
            c_avg = np.mean(positions, axis=0)
            center = c_avg + z_target * d_avg

        return center.astype(np.float32) - half_extents

    # ------------------------------------------------------------------
    # TSDF fusion
    # ------------------------------------------------------------------

    def _reconstruct_tsdf(
        self,
        frames: np.ndarray,
        cameras: list[CameraExtrinsics],
        depths: np.ndarray,
    ) -> MeshGeometry:
        """Volumetric TSDF fusion followed by marching cubes.

        Improvements over naive integration:
        - Only voxels with valid observations (weight > 0) are considered
          inside. Unobserved interior voxels are flood-filled to -1.0
          (inside sentinel) to prevent double-wall internal cavities.
        - Attempts to use PR-141 C++20 ``synapse._c_geometry.marching_cubes``
          for performance; falls back to pure-Python MC on ImportError.
        """
        bx, by, bz = self.volume_bounds
        origin = self._compute_volume_origin(cameras, self.volume_bounds)
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

            # Asymmetric truncation: only gate behind-surface side.
            # Free-space (sdf > 0) is always valid; behind-surface (sdf < 0)
            # is only valid within truncation distance.
            valid = in_bounds & (depth_sampled > 0) & (sdf > -self.sdf_trunc)

            tsdf_update = np.clip(sdf / self.sdf_trunc, -1.0, 1.0)

            # Confidence weighting: weight by inverse depth (closer = more reliable)
            confidence = np.where(
                valid,
                1.0 / np.clip(z_cam.astype(np.float32), 0.1, None),
                0.0,
            )

            w_old = weight.ravel()
            tsdf_old = tsdf.ravel()
            w_new = np.where(valid, w_old + confidence, w_old)
            tsdf_new = np.where(
                valid,
                (tsdf_old * w_old + tsdf_update * confidence) / np.clip(w_new, 1e-8, None),
                tsdf_old,
            )
            tsdf = tsdf_new.reshape(nx, ny, nz)
            weight = w_new.reshape(nx, ny, nz)

        # Flood-fill unobserved interior voxels: voxels with weight=0
        # that are surrounded by negative-SDF voxels should be -1.0 (inside),
        # not +1.0 (outside) which causes double-wall internal cavities.
        observed = weight > 0
        surface_inside = observed & (tsdf < 0)
        if np.any(surface_inside):
            # Simple 6-connected flood fill from surface-inside to unobserved
            unobserved = ~observed
            fill_mask = np.zeros_like(unobserved)
            # Seed: unobserved voxels adjacent to observed-inside voxels.
            # Use slicing instead of np.roll to avoid wrap-around leakage
            # across opposite volume boundaries.
            def _shift_no_wrap(
                arr: np.ndarray, direction: int, axis: int,
            ) -> np.ndarray:
                """Shift array along axis without wrapping (fill with False)."""
                result = np.zeros_like(arr)
                slices_src: list[slice] = [slice(None)] * arr.ndim
                slices_dst: list[slice] = [slice(None)] * arr.ndim
                if direction == 1:
                    slices_src[axis] = slice(None, -1)
                    slices_dst[axis] = slice(1, None)
                else:
                    slices_src[axis] = slice(1, None)
                    slices_dst[axis] = slice(None, -1)
                result[tuple(slices_dst)] = arr[tuple(slices_src)]
                return result

            for axis in range(3):
                for direction in (-1, 1):
                    shifted = _shift_no_wrap(surface_inside, direction, axis)
                    fill_mask |= shifted & unobserved

            # Iterative propagation (bounded by grid size)
            changed = True
            max_iters = max(nx, ny, nz)
            it = 0
            while changed and it < max_iters:
                changed = False
                it += 1
                new_fill = np.copy(fill_mask)
                for axis in range(3):
                    for direction in (-1, 1):
                        shifted = _shift_no_wrap(fill_mask, direction, axis)
                        expansion = shifted & unobserved & (~fill_mask)
                        if np.any(expansion):
                            new_fill |= expansion
                            changed = True
                fill_mask = new_fill

            # Set interior unobserved voxels to -1.0 (inside)
            tsdf[fill_mask] = -1.0

        # Pad outer faces with +1.0 (outside) to ensure watertight closure.
        tsdf = np.pad(tsdf, pad_width=1, mode="constant", constant_values=1.0)

        # Origin for MC: shift back by one voxel (padding) + half voxel (centre).
        mc_origin = (
            origin
            - np.array([1.0, 1.0, 1.0]) * effective_voxel
            + np.array([0.5, 0.5, 0.5]) * effective_voxel
        )

        # Prefer C++20 marching cubes from aether_geo if available
        try:
            from synapse.bridge.aether_bridge import get_aether_geo

            _cg = get_aether_geo()
            if _cg is not None and hasattr(_cg, "marching_cubes"):
                # C++ marching_cubes expects Z-major layout (nz, ny, nx).
                # Our volume is stored as (nx, ny, nz), so transpose axes.
                tsdf_zyx = np.ascontiguousarray(np.transpose(tsdf, (2, 1, 0)))
                mc_result = _cg.marching_cubes(
                    tsdf_zyx, 0.0, effective_voxel, mc_origin.tolist(),
                )
                # C++ binding returns dict {"vertices": ..., "faces": ...}
                if isinstance(mc_result, dict):
                    verts = np.asarray(mc_result["vertices"], dtype=np.float32)
                    faces = np.asarray(mc_result["faces"], dtype=np.int32)
                else:
                    verts, faces = mc_result
                    verts = np.asarray(verts, dtype=np.float32)
                    faces = np.asarray(faces, dtype=np.int32)

                # C++ outputs face soup (V=3F) with CW winding. Fix both:
                if verts.ndim == 2 and verts.shape[1] == 3 and len(faces) > 0:
                    # 1. Flip triangle winding CW → CCW for outward normals
                    if faces.ndim == 2 and faces.shape[1] == 3:
                        faces = faces[:, [0, 2, 1]]

                    # 2. Weld duplicate vertices (merge within tight float32 tolerance;
                    # coarse epsilon like 0.01*voxel collapses distinct nearby MC crossings
                    # on noisy TSDFs into non-manifold pinch edges).
                    if len(verts) > 0:
                        verts, faces = _weld_vertices(verts, faces, epsilon=effective_voxel * 1e-4)
            else:
                verts, faces = _marching_cubes(tsdf, effective_voxel, mc_origin, isovalue=0.0)
        except AttributeError:
            verts, faces = _marching_cubes(tsdf, effective_voxel, mc_origin, isovalue=0.0)

        if verts.size == 0 or faces.size == 0:
            raise ContractValidationError(
                "TSDF fusion produced an empty mesh: marching cubes extracted "
                f"no valid iso-surface from {n_views} input depth views.",
                agent_fix_hint=(
                    "Ensure depth maps have non-zero values within the "
                    "volume bounds. Consider adjusting `sdf_trunc` or "
                    "`volume_bounds` parameters."
                ),
            )

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
        origin = self._compute_volume_origin(cameras, self.volume_bounds)
        nx = int(bx / self.voxel_size)
        ny = int(by / self.voxel_size)
        nz = int(bz / self.voxel_size)

        max_dim = 32
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

            # Carve: only carve voxels within the camera frustum that project onto background!
            carve = np.zeros(len(cam_pts), dtype=bool)
            if np.any(in_bounds):
                carve[in_bounds] = mask[vi_arr[in_bounds], ui[in_bounds]] <= 0.5

            occupied_flat = occupied.ravel()
            occupied_flat[carve] = False
            occupied = occupied_flat.reshape(nx, ny, nz)

        # Convert occupied voxel grid to SDF-like field for marching cubes.
        # occupied=True → SDF < 0 (inside), occupied=False → SDF > 0 (outside).
        sdf = np.where(occupied, -1.0, 1.0).astype(np.float32)

        # Pad outer faces with +1.0 (outside) to ensure watertight closure.
        sdf = np.pad(sdf, pad_width=1, mode="constant", constant_values=1.0)

        # Origin for MC: shift back by one voxel (padding) + half voxel (centre).
        mc_origin = (
            origin
            - np.array([1.0, 1.0, 1.0]) * effective_voxel
            + np.array([0.5, 0.5, 0.5]) * effective_voxel
        )

        verts, faces = _marching_cubes(sdf, effective_voxel, mc_origin, isovalue=0.0)

        if verts.size == 0 or faces.size == 0:
            raise ContractValidationError(
                "Visual hull reconstruction produced an empty mesh: no "
                "voxels survived silhouette carving from "
                f"{n_views} input views.",
                agent_fix_hint=(
                    "Ensure silhouette masks contain visible foreground "
                    "pixels and that camera extrinsics are calibrated "
                    "correctly."
                ),
            )

        return MeshGeometry(vertices=verts, faces=faces)
