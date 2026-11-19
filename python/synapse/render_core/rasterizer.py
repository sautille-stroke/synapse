"""Pure NumPy vectorized software rasterizer for Synapse-Bench.

.. important:: **Test-only mock rasterizer** — NOT for production FGPP generation.

    This is a zero-external-dependency CPU rasterization path intended
    **exclusively** for ``@pytest`` / CI smoke tests and offline benchmarking
    where GPU access is unavailable.  Production FGPP rendering must use:

    - **Track A** (offline ground truth): ``caelum-core`` GPU path tracer
    - **Track B** (real-time): ``fulgur`` headless renderer

    See https://github.com/sautille-stroke/synapse/issues/34 (area ④)

Provides a zero-external-dependency rasterization path that produces
complete :class:`~synapse.fgpp.FGPPFrame` output including all 7 logical
channel groups.  If ``numba`` is available, JIT-accelerated kernels are
used for inner loops; otherwise falls back to pure NumPy vectorised code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from synapse.fgpp.frame import CameraParams, FGPPFrame

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Cryptomatte entity ID hashing (Psyop Cryptomatte spec: MD5 → float32 bits)
# ---------------------------------------------------------------------------


def _cryptomatte_name_hash(name: str) -> int:
    """Compute a Cryptomatte entity ID per the Psyop Cryptomatte specification.

    Algorithm: MD5(UTF-8 name) → take first 4 bytes LE → uint32 →
    sanitize IEEE 754 exponent (clamp NaN/Inf/denormals) → store as uint32.

    This matches ``aether::core::cryptomatteNameHash`` in
    ``aether/core/include/aether/core/hash.h`` and is compatible with
    Cryptomatte IDs produced by ``fulgur`` and ``caelum-core``.

    Returns a ``uint32`` integer representing the float32 bit-pattern.
    """
    import hashlib
    import struct

    digest = hashlib.md5(name.encode("utf-8")).digest()

    # First 4 bytes, little-endian → uint32
    h = struct.unpack("<I", digest[:4])[0]

    # Sanitize IEEE 754 exponent field
    exponent = (h >> 23) & 0xFF
    if exponent == 0xFF:
        # NaN or Inf — clear one exponent bit to make it finite
        h &= ~(1 << 23)
    elif exponent == 0x00:
        # Denormal — set one exponent bit to make it normal
        h |= 1 << 23

    return int(h)


def entity_hash(name: str) -> int:
    """Compute a 32-bit Cryptomatte entity ID for *name*.

    Uses the Psyop Cryptomatte specification (MD5-based) for cross-engine
    compatibility with ``fulgur`` and ``caelum-core``.

    Returns a non-zero hash; background ``0`` is reserved.
    """
    h = _cryptomatte_name_hash(name)
    return h if h != 0 else 1


# ---------------------------------------------------------------------------
# Sutherland-Hodgman near-plane clipping in homogeneous clip space
# ---------------------------------------------------------------------------

_CLIP_EPSILON: float = 1e-7


@dataclass
class _ClipVertex:
    """Vertex representation in homogeneous clip space carrying all attributes."""

    clip: np.ndarray  # [4] homogeneous coordinate (x, y, z, w)
    normal: np.ndarray  # [3] surface normal
    bc: np.ndarray  # [3] base color
    view_z: float  # view-space z coordinate
    prev_clip: np.ndarray | None = None  # [4] previous frame homogeneous coordinate


def _clip_triangle_near_plane(
    v0: _ClipVertex,
    v1: _ClipVertex,
    v2: _ClipVertex,
) -> list[_ClipVertex]:
    """Clip a triangle against the near plane ``z + w >= 0`` in clip space.

    Directly interpolates all attributes in 4D homogeneous space before
    perspective division.
    """
    input_verts = [v0, v1, v2]
    output: list[_ClipVertex] = []

    for i in range(len(input_verts)):
        curr = input_verts[i]
        prev = input_verts[i - 1]

        # Inside test: z + w >= 0 (near plane in homogeneous clip space)
        d_curr = float(curr.clip[2] + curr.clip[3])
        d_prev = float(prev.clip[2] + prev.clip[3])

        curr_inside = d_curr >= -_CLIP_EPSILON
        prev_inside = d_prev >= -_CLIP_EPSILON

        if curr_inside:
            if not prev_inside:
                # Entering: emit intersection then current
                denom = d_prev - d_curr
                t = 0.0 if abs(denom) < 1e-30 else d_prev / denom
                t = max(0.0, min(1.0, float(t)))
                inter_clip = prev.clip + t * (curr.clip - prev.clip)
                inter_norm = prev.normal + t * (curr.normal - prev.normal)
                n_len = float(np.linalg.norm(inter_norm))
                if n_len > 1e-8:
                    inter_norm = inter_norm / n_len
                inter_bc = prev.bc + t * (curr.bc - prev.bc)
                inter_vz = prev.view_z + t * (curr.view_z - prev.view_z)
                inter_prev_clip = None
                if prev.prev_clip is not None and curr.prev_clip is not None:
                    inter_prev_clip = prev.prev_clip + t * (curr.prev_clip - prev.prev_clip)

                output.append(
                    _ClipVertex(
                        clip=inter_clip,
                        normal=inter_norm,
                        bc=inter_bc,
                        view_z=inter_vz,
                        prev_clip=inter_prev_clip,
                    )
                )
            output.append(curr)
        elif prev_inside:
            # Leaving: emit intersection only
            denom = d_prev - d_curr
            t = 0.0 if abs(denom) < 1e-30 else d_prev / denom
            t = max(0.0, min(1.0, float(t)))
            inter_clip = prev.clip + t * (curr.clip - prev.clip)
            inter_norm = prev.normal + t * (curr.normal - prev.normal)
            n_len = float(np.linalg.norm(inter_norm))
            if n_len > 1e-8:
                inter_norm = inter_norm / n_len
            inter_bc = prev.bc + t * (curr.bc - prev.bc)
            inter_vz = prev.view_z + t * (curr.view_z - prev.view_z)
            inter_prev_clip = None
            if prev.prev_clip is not None and curr.prev_clip is not None:
                inter_prev_clip = prev.prev_clip + t * (curr.prev_clip - prev.prev_clip)

            output.append(
                _ClipVertex(
                    clip=inter_clip,
                    normal=inter_norm,
                    bc=inter_bc,
                    view_z=inter_vz,
                    prev_clip=inter_prev_clip,
                )
            )

    return output


# ---------------------------------------------------------------------------
# Software Rasterizer
# ---------------------------------------------------------------------------


@dataclass
class SoftwareRasterizer:
    """Pure NumPy bounding-box + barycentric software rasterizer.

    Produces complete :class:`~synapse.fgpp.FGPPFrame` output for each
    rendered frame.

    Parameters
    ----------
    width, height:
        Output image dimensions.
    light_dir:
        World-space directional light direction (normalised).
    light_color:
        Light RGB intensity.
    ambient:
        Ambient light level.
    """

    width: int = 512
    height: int = 512
    light_dir: np.ndarray | None = None
    light_color: np.ndarray | None = None
    ambient: float = 0.15

    def __post_init__(self) -> None:
        if self.light_dir is None:
            self.light_dir = np.array([0.5, 0.8, 0.6], dtype=np.float64)
            self.light_dir /= np.linalg.norm(self.light_dir)
        if self.light_color is None:
            self.light_color = np.array([1.0, 0.95, 0.9], dtype=np.float64)

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray,
        camera: CameraParams,
        *,
        base_color: np.ndarray | None = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        entity_name: str = "default",
        prev_camera: CameraParams | None = None,
        prev_vertices: np.ndarray | None = None,
    ) -> FGPPFrame:
        """Render a mesh and produce a full FGPPFrame.

        Parameters
        ----------
        vertices:
            ``[N, 3]`` world-space vertex positions.
        faces:
            ``[M, 3]`` triangle face indices.
        normals:
            ``[N, 3]`` vertex normals.
        camera:
            Camera parameters for this frame.
        base_color:
            ``[N, 3]`` per-vertex base colour (linear sRGB).
            If ``None``, uses uniform ``[0.8, 0.8, 0.8]``.
        entity_name:
            Name used for Cryptomatte ID hashing.
        prev_camera:
            Previous frame camera for motion vector computation.
        prev_vertices:
            Optional ``[N, 3]`` previous frame vertex positions for deforming
            or skeletal animation motion vectors.
        """
        W, H = self.width, self.height
        if camera.width > 0 and camera.width != W:
            raise ValueError(f"Camera width {camera.width} != rasterizer width {W}")
        if camera.height > 0 and camera.height != H:
            raise ValueError(f"Camera height {camera.height} != rasterizer height {H}")
        eid = np.uint32(entity_hash(entity_name))

        # Per-vertex base color default
        N = vertices.shape[0]
        if base_color is None:
            base_color = np.full((N, 3), 0.8, dtype=np.float32)
        else:
            base_color = np.asarray(base_color, dtype=np.float32)
            if base_color.ndim == 1 and base_color.shape[0] == 3:
                base_color = np.tile(base_color, (N, 1))
            elif base_color.ndim == 3:
                mean_color = np.mean(base_color, axis=(0, 1))[:3]
                base_color = np.tile(mean_color, (N, 1))
            elif base_color.shape != (N, 3):
                base_color = np.full((N, 3), 0.8, dtype=np.float32)

        # Build MVP: view_matrix then projection
        view = camera.view_matrix.astype(np.float64)
        proj = self._build_projection(camera)
        mvp = proj @ view

        # Build previous frame MVP for motion vectors
        prev_mvp: np.ndarray | None = None
        if prev_camera is not None:
            prev_view = prev_camera.view_matrix.astype(np.float64)
            prev_proj = self._build_projection(prev_camera)
            prev_mvp = prev_proj @ prev_view

        # Transform vertices to clip space
        verts_h = np.hstack([vertices.astype(np.float64), np.ones((N, 1), dtype=np.float64)])
        verts_clip = (mvp @ verts_h.T).T  # [N, 4]
        verts_view = (view @ verts_h.T).T  # [N, 4]

        # Pre-compute previous clip-space positions for motion vectors
        prev_verts_clip: np.ndarray | None = None
        if prev_mvp is not None:
            if prev_vertices is not None:
                p_verts = prev_vertices.astype(np.float64)
                p_verts_h = np.hstack([p_verts, np.ones((N, 1), dtype=np.float64)])
            else:
                p_verts_h = verts_h
            prev_verts_clip = (prev_mvp @ p_verts_h.T).T  # [N, 4]

        # Init buffers
        depth_buf = np.full((H, W), np.inf, dtype=np.float64)
        color_buf = np.zeros((H, W, 3), dtype=np.float64)
        normal_buf = np.tile(
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            (H, W, 1),
        )
        radiance_buf = np.zeros((H, W, 3), dtype=np.float64)
        bcolor_buf = np.zeros((H, W, 3), dtype=np.float64)
        mask_buf = np.zeros((H, W), dtype=np.uint32)
        rm_buf = np.zeros((H, W, 2), dtype=np.float64)
        mv_buf = np.zeros((H, W, 2), dtype=np.float32)

        # Rasterise each triangle
        assert self.light_dir is not None and self.light_color is not None
        norms = normals.astype(np.float64)
        light_d = self.light_dir.astype(np.float64)
        light_c = self.light_color.astype(np.float64)
        amb = self.ambient

        for fi in range(faces.shape[0]):
            i0, i1, i2 = faces[fi]

            v0 = _ClipVertex(
                clip=verts_clip[i0],
                normal=norms[i0],
                bc=base_color[i0].astype(np.float64),
                view_z=float(verts_view[i0, 2]),
                prev_clip=prev_verts_clip[i0] if prev_verts_clip is not None else None,
            )
            v1 = _ClipVertex(
                clip=verts_clip[i1],
                normal=norms[i1],
                bc=base_color[i1].astype(np.float64),
                view_z=float(verts_view[i1, 2]),
                prev_clip=prev_verts_clip[i1] if prev_verts_clip is not None else None,
            )
            v2 = _ClipVertex(
                clip=verts_clip[i2],
                normal=norms[i2],
                bc=base_color[i2].astype(np.float64),
                view_z=float(verts_view[i2, 2]),
                prev_clip=prev_verts_clip[i2] if prev_verts_clip is not None else None,
            )

            # Sutherland-Hodgman near-plane clipping
            clipped_poly = _clip_triangle_near_plane(v0, v1, v2)
            if len(clipped_poly) < 3:
                continue

            # Triangulate the clipped polygon and rasterise each sub-triangle
            for ti in range(1, len(clipped_poly) - 1):
                self._rasterize_clipped_triangle(
                    W,
                    H,
                    clipped_poly[0],
                    clipped_poly[ti],
                    clipped_poly[ti + 1],
                    depth_buf,
                    color_buf,
                    normal_buf,
                    radiance_buf,
                    bcolor_buf,
                    mask_buf,
                    rm_buf,
                    mv_buf,
                    eid,
                    roughness,
                    metallic,
                    light_d,
                    light_c,
                    amb,
                    camera,
                )

        # Clamp depth: replace inf with z_far
        depth_out = depth_buf.copy()
        depth_out[depth_out == np.inf] = camera.z_far
        depth_out = np.clip(depth_out, camera.z_near, camera.z_far)

        return FGPPFrame(
            camera=camera,
            depth=depth_out.astype(np.float32),
            direct_radiance=radiance_buf.astype(np.float16),
            world_normal=normal_buf.astype(np.float16),
            motion_vector=mv_buf.astype(np.float16),
            base_color=bcolor_buf.astype(np.float16),
            roughness_metallic=rm_buf.astype(np.float16),
            cryptomatte_id=mask_buf,
        )

    def _rasterize_clipped_triangle(
        self,
        W: int,
        H: int,
        v0: _ClipVertex,
        v1: _ClipVertex,
        v2: _ClipVertex,
        depth_buf: np.ndarray,
        color_buf: np.ndarray,
        normal_buf: np.ndarray,
        radiance_buf: np.ndarray,
        bcolor_buf: np.ndarray,
        mask_buf: np.ndarray,
        rm_buf: np.ndarray,
        mv_buf: np.ndarray,
        eid: np.uint32,
        roughness: float,
        metallic: float,
        light_d: np.ndarray,
        light_c: np.ndarray,
        amb: float,
        camera: CameraParams,
    ) -> None:
        """Rasterize a single clipped triangle with perspective-correct 1/w interpolation."""
        w0, w1, w2 = float(v0.clip[3]), float(v1.clip[3]), float(v2.clip[3])
        if abs(w0) < _CLIP_EPSILON or abs(w1) < _CLIP_EPSILON or abs(w2) < _CLIP_EPSILON:
            return

        inv_w0, inv_w1, inv_w2 = 1.0 / w0, 1.0 / w1, 1.0 / w2

        # Perspective divide for screen coordinates
        sx0 = (v0.clip[0] * inv_w0 * 0.5 + 0.5) * W
        sy0 = (1.0 - (v0.clip[1] * inv_w0 * 0.5 + 0.5)) * H
        sx1 = (v1.clip[0] * inv_w1 * 0.5 + 0.5) * W
        sy1 = (1.0 - (v1.clip[1] * inv_w1 * 0.5 + 0.5)) * H
        sx2 = (v2.clip[0] * inv_w2 * 0.5 + 0.5) * W
        sy2 = (1.0 - (v2.clip[1] * inv_w2 * 0.5 + 0.5)) * H

        z0 = -float(v0.view_z)
        z1 = -float(v1.view_z)
        z2 = -float(v2.view_z)

        min_x = max(0, int(math.floor(min(sx0, sx1, sx2))))
        max_x = min(W - 1, int(math.ceil(max(sx0, sx1, sx2))))
        min_y = max(0, int(math.floor(min(sy0, sy1, sy2))))
        max_y = min(H - 1, int(math.ceil(max(sy0, sy1, sy2))))

        if min_x > max_x or min_y > max_y:
            return

        denom = (sy1 - sy2) * (sx0 - sx2) + (sx2 - sx1) * (sy0 - sy2)
        if abs(denom) < 1e-10:
            return
        inv_denom = 1.0 / denom

        # Sample at pixel centers (px + 0.5, py + 0.5)
        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        bw0 = ((sy1 - sy2) * (grid_x - sx2) + (sx2 - sx1) * (grid_y - sy2)) * inv_denom
        bw1 = ((sy2 - sy0) * (grid_x - sx2) + (sx0 - sx2) * (grid_y - sy2)) * inv_denom
        bw2 = 1.0 - bw0 - bw1

        inside = (bw0 >= 0.0) & (bw1 >= 0.0) & (bw2 >= 0.0)
        if not np.any(inside):
            return

        interp_inv_w = bw0 * inv_w0 + bw1 * inv_w1 + bw2 * inv_w2
        valid_w = inside & (abs(interp_inv_w) > 1e-20)
        if not np.any(valid_w):
            return

        w_correct = np.zeros_like(interp_inv_w)
        w_correct[valid_w] = 1.0 / interp_inv_w[valid_w]

        linear_depth = np.zeros_like(interp_inv_w)
        linear_depth[valid_w] = (
            bw0[valid_w] * z0 * inv_w0 + bw1[valid_w] * z1 * inv_w1 + bw2[valid_w] * z2 * inv_w2
        ) * w_correct[valid_w]

        cur_depth_box = depth_buf[min_y : max_y + 1, min_x : max_x + 1]
        closer = (
            valid_w
            & (linear_depth > 0.0)
            & (linear_depth <= camera.z_far)
            & (linear_depth < cur_depth_box)
        )
        if not np.any(closer):
            return

        cur_depth_box[closer] = linear_depth[closer]

        # Normal interpolation
        n0_w = v0.normal * inv_w0
        n1_w = v1.normal * inv_w1
        n2_w = v2.normal * inv_w2
        nx = (bw0 * n0_w[0] + bw1 * n1_w[0] + bw2 * n2_w[0]) * w_correct
        ny = (bw0 * n0_w[1] + bw1 * n1_w[1] + bw2 * n2_w[1]) * w_correct
        nz = (bw0 * n0_w[2] + bw1 * n1_w[2] + bw2 * n2_w[2]) * w_correct

        n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
        n_mag_safe = np.maximum(n_mag, 1e-8)
        nx /= n_mag_safe
        ny /= n_mag_safe
        nz /= n_mag_safe

        norm_box = normal_buf[min_y : max_y + 1, min_x : max_x + 1]
        norm_box[closer, 0] = nx[closer]
        norm_box[closer, 1] = ny[closer]
        norm_box[closer, 2] = nz[closer]

        # Base color interpolation
        bc0_w = v0.bc * inv_w0
        bc1_w = v1.bc * inv_w1
        bc2_w = v2.bc * inv_w2
        bc_r = (bw0 * bc0_w[0] + bw1 * bc1_w[0] + bw2 * bc2_w[0]) * w_correct
        bc_g = (bw0 * bc0_w[1] + bw1 * bc1_w[1] + bw2 * bc2_w[1]) * w_correct
        bc_b = (bw0 * bc0_w[2] + bw1 * bc1_w[2] + bw2 * bc2_w[2]) * w_correct

        bc_box = bcolor_buf[min_y : max_y + 1, min_x : max_x + 1]
        bc_box[closer, 0] = bc_r[closer]
        bc_box[closer, 1] = bc_g[closer]
        bc_box[closer, 2] = bc_b[closer]

        # Diffuse shading
        ndotl = np.maximum(0.0, nx * light_d[0] + ny * light_d[1] + nz * light_d[2])
        diff_r = bc_r * (amb + ndotl * light_c[0])
        diff_g = bc_g * (amb + ndotl * light_c[1])
        diff_b = bc_b * (amb + ndotl * light_c[2])

        rad_box = radiance_buf[min_y : max_y + 1, min_x : max_x + 1]
        rad_box[closer, 0] = diff_r[closer]
        rad_box[closer, 1] = diff_g[closer]
        rad_box[closer, 2] = diff_b[closer]

        col_box = color_buf[min_y : max_y + 1, min_x : max_x + 1]
        col_box[closer, 0] = diff_r[closer]
        col_box[closer, 1] = diff_g[closer]
        col_box[closer, 2] = diff_b[closer]

        mask_box = mask_buf[min_y : max_y + 1, min_x : max_x + 1]
        mask_box[closer] = eid

        rm_box = rm_buf[min_y : max_y + 1, min_x : max_x + 1]
        rm_box[closer, 0] = roughness
        rm_box[closer, 1] = metallic

        # Motion vectors
        if v0.prev_clip is not None and v1.prev_clip is not None and v2.prev_clip is not None:
            pc0_w = v0.prev_clip * inv_w0
            pc1_w = v1.prev_clip * inv_w1
            pc2_w = v2.prev_clip * inv_w2

            prev_cx = (bw0 * pc0_w[0] + bw1 * pc1_w[0] + bw2 * pc2_w[0]) * w_correct
            prev_cy = (bw0 * pc0_w[1] + bw1 * pc1_w[1] + bw2 * pc2_w[1]) * w_correct
            prev_cw = (bw0 * pc0_w[3] + bw1 * pc1_w[3] + bw2 * pc2_w[3]) * w_correct

            valid_prev = closer & (prev_cw > _CLIP_EPSILON)
            mv_box = mv_buf[min_y : max_y + 1, min_x : max_x + 1]

            if np.any(valid_prev):
                prev_ndc_x = prev_cx[valid_prev] / prev_cw[valid_prev]
                prev_ndc_y = prev_cy[valid_prev] / prev_cw[valid_prev]
                prev_sx = (prev_ndc_x * 0.5 + 0.5) * W
                prev_sy = (1.0 - (prev_ndc_y * 0.5 + 0.5)) * H

                mv_box[valid_prev, 0] = grid_x[valid_prev] - prev_sx
                mv_box[valid_prev, 1] = grid_y[valid_prev] - prev_sy

            invalid_prev = closer & ~valid_prev
            if np.any(invalid_prev):
                mv_box[invalid_prev, :] = 0.0

    @staticmethod
    def _build_projection(camera: CameraParams) -> np.ndarray:
        """Build a 4×4 perspective projection matrix from camera intrinsics."""
        return camera.projection_matrix()
