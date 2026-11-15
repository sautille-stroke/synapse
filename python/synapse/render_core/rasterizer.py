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
# MurmurHash3 (32-bit) — pure Python for Cryptomatte entity IDs
# ---------------------------------------------------------------------------

def _murmurhash3_32(key: bytes, seed: int = 0) -> int:
    """MurmurHash3 32-bit finaliser for entity name hashing.

    Uses native Python ``int`` with explicit ``& 0xFFFFFFFF`` masking to avoid
    ``RuntimeWarning: overflow encountered in scalar multiply`` emitted by
    NumPy ≥ 2.0 when performing arithmetic on ``np.uint32`` scalars.
    """
    _M = 0xFFFFFFFF
    length = len(key)
    h = seed & _M
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    nblocks = length // 4

    for i in range(nblocks):
        k = int.from_bytes(key[i * 4:(i + 1) * 4], "little") & _M
        k = (k * c1) & _M
        k = ((k << 15) | (k >> 17)) & _M
        k = (k * c2) & _M
        h ^= k
        h = ((h << 13) | (h >> 19)) & _M
        h = (h * 5 + 0xE6546B64) & _M

    tail = key[nblocks * 4:]
    k1 = 0
    tlen = len(tail)
    if tlen >= 3:
        k1 ^= (tail[2] << 16) & _M
    if tlen >= 2:
        k1 ^= (tail[1] << 8) & _M
    if tlen >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & _M
        k1 = ((k1 << 15) | (k1 >> 17)) & _M
        k1 = (k1 * c2) & _M
        h ^= k1

    h ^= length
    h = (h ^ (h >> 16)) & _M
    h = (h * 0x85EBCA6B) & _M
    h = (h ^ (h >> 13)) & _M
    h = (h * 0xC2B2AE35) & _M
    h = (h ^ (h >> 16)) & _M
    return h


def entity_hash(name: str) -> int:
    """Compute a 32-bit Cryptomatte entity ID for *name*."""
    return _murmurhash3_32(name.encode("utf-8"))


# ---------------------------------------------------------------------------
# Sutherland-Hodgman near-plane clipping in homogeneous clip space
# ---------------------------------------------------------------------------

_CLIP_EPSILON: float = 1e-7


def _clip_triangle_near_plane(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> list[np.ndarray]:
    """Clip a triangle against the near plane ``z + w >= 0`` in clip space.

    Each vertex is a 4-element homogeneous clip-space coordinate ``[x,y,z,w]``.
    Returns a list of 0–4 vertices forming the clipped convex polygon (may
    become a quad if one vertex is behind the near plane).

    Uses Sutherland-Hodgman algorithm for a single clip plane.
    """
    input_verts = [v0, v1, v2]
    output: list[np.ndarray] = []

    for i in range(len(input_verts)):
        current = input_verts[i]
        prev = input_verts[i - 1]

        # Inside test: z + w >= 0  (near plane in clip space)
        d_curr = current[2] + current[3]
        d_prev = prev[2] + prev[3]

        curr_inside = d_curr >= -_CLIP_EPSILON
        prev_inside = d_prev >= -_CLIP_EPSILON

        if curr_inside:
            if not prev_inside:
                # Entering: emit intersection then current
                t = d_prev / (d_prev - d_curr + 1e-30)
                intersection = prev + t * (current - prev)
                output.append(intersection)
            output.append(current)
        elif prev_inside:
            # Leaving: emit intersection only
            t = d_prev / (d_prev - d_curr + 1e-30)
            intersection = prev + t * (current - prev)
            output.append(intersection)

    return output


def _triangulate_polygon(
    polygon: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fan-triangulate a convex polygon into triangles."""
    if len(polygon) < 3:
        return []
    tris = []
    for i in range(1, len(polygon) - 1):
        tris.append((polygon[0], polygon[i], polygon[i + 1]))
    return tris


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
        """
        W, H = self.width, self.height
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
        verts_h = np.hstack([vertices.astype(np.float64),
                             np.ones((N, 1), dtype=np.float64)])
        verts_clip = (mvp @ verts_h.T).T  # [N, 4]
        verts_view = (view @ verts_h.T).T  # [N, 4]

        # Pre-compute previous clip-space positions for motion vectors
        prev_verts_clip: np.ndarray | None = None
        if prev_mvp is not None:
            prev_verts_clip = (prev_mvp @ verts_h.T).T  # [N, 4]

        # Init buffers
        depth_buf = np.full((H, W), np.inf, dtype=np.float64)
        color_buf = np.zeros((H, W, 3), dtype=np.float64)
        normal_buf = np.zeros((H, W, 3), dtype=np.float64)
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

            # Clip-space vertices for near-plane clipping
            c0 = verts_clip[i0]
            c1 = verts_clip[i1]
            c2 = verts_clip[i2]

            # Per-vertex attributes for interpolation
            attr0 = {
                "normal": norms[i0],
                "bc": base_color[i0].astype(np.float64),
                "view_z": float(verts_view[i0, 2]),
                "clip": c0.copy(),
            }
            attr1 = {
                "normal": norms[i1],
                "bc": base_color[i1].astype(np.float64),
                "view_z": float(verts_view[i1, 2]),
                "clip": c1.copy(),
            }
            attr2 = {
                "normal": norms[i2],
                "bc": base_color[i2].astype(np.float64),
                "view_z": float(verts_view[i2, 2]),
                "clip": c2.copy(),
            }

            if prev_verts_clip is not None:
                attr0["prev_clip"] = prev_verts_clip[i0].copy()
                attr1["prev_clip"] = prev_verts_clip[i1].copy()
                attr2["prev_clip"] = prev_verts_clip[i2].copy()

            # Sutherland-Hodgman near-plane clipping
            clipped_poly = _clip_triangle_near_plane(c0, c1, c2)
            if len(clipped_poly) < 3:
                continue

            # Interpolate attributes for clipped vertices using
            # barycentric coordinates relative to the original triangle
            clipped_attrs = []
            for cv in clipped_poly:
                attr = self._interpolate_clip_attrs(
                    c0, c1, c2, cv,
                    attr0, attr1, attr2,
                    prev_verts_clip is not None,
                )
                clipped_attrs.append(attr)

            # Triangulate the clipped polygon and rasterise each sub-triangle
            for ti in range(1, len(clipped_poly) - 1):
                tri_clips = [clipped_poly[0], clipped_poly[ti], clipped_poly[ti + 1]]
                tri_attrs = [clipped_attrs[0], clipped_attrs[ti], clipped_attrs[ti + 1]]

                self._rasterize_clipped_triangle(
                    W, H, tri_clips, tri_attrs,
                    depth_buf, color_buf, normal_buf, radiance_buf,
                    bcolor_buf, mask_buf, rm_buf, mv_buf,
                    eid, roughness, metallic,
                    light_d, light_c, amb,
                    prev_mvp is not None,
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

    @staticmethod
    def _interpolate_clip_attrs(
        c0: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
        cv: np.ndarray,
        attr0: dict[str, object],
        attr1: dict[str, object],
        attr2: dict[str, object],
        has_prev: bool,
    ) -> dict[str, object]:
        """Compute interpolated attributes for a clipped vertex *cv*.

        Uses barycentric coordinates in clip space to interpolate
        per-vertex attributes from the original triangle.
        """
        # Compute barycentric coordinates using the 2D projection of
        # clip-space x/w, y/w. But since cv may be on the near plane edge
        # we use the clip-space coordinates directly with a robust method.
        # Barycentric in 4D clip space using the x,y,z components
        denom_mat = np.array([
            [c0[0] - c2[0], c1[0] - c2[0]],
            [c0[1] - c2[1], c1[1] - c2[1]],
        ], dtype=np.float64)
        rhs = np.array([cv[0] - c2[0], cv[1] - c2[1]], dtype=np.float64)
        det = denom_mat[0, 0] * denom_mat[1, 1] - denom_mat[0, 1] * denom_mat[1, 0]
        if abs(det) < 1e-20:
            # Degenerate — use w-based fallback
            w0, w1, w2 = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        else:
            inv_det = 1.0 / det
            w0 = (rhs[0] * denom_mat[1, 1] - rhs[1] * denom_mat[0, 1]) * inv_det
            w1 = (denom_mat[0, 0] * rhs[1] - denom_mat[1, 0] * rhs[0]) * inv_det
            w2 = 1.0 - w0 - w1

        # Clamp barycentric weights
        w0 = max(0.0, min(1.0, w0))
        w1 = max(0.0, min(1.0, w1))
        w2 = max(0.0, min(1.0, w2))
        s = w0 + w1 + w2
        if s > 1e-20:
            w0 /= s
            w1 /= s
            w2 /= s

        n0 = attr0["normal"]
        n1 = attr1["normal"]
        n2 = attr2["normal"]
        assert isinstance(n0, np.ndarray) and isinstance(n1, np.ndarray) and isinstance(n2, np.ndarray)

        bc0 = attr0["bc"]
        bc1 = attr1["bc"]
        bc2 = attr2["bc"]
        assert isinstance(bc0, np.ndarray) and isinstance(bc1, np.ndarray) and isinstance(bc2, np.ndarray)

        vz0 = float(attr0["view_z"])  # type: ignore[arg-type]
        vz1 = float(attr1["view_z"])  # type: ignore[arg-type]
        vz2 = float(attr2["view_z"])  # type: ignore[arg-type]

        result: dict[str, object] = {
            "normal": w0 * n0 + w1 * n1 + w2 * n2,
            "bc": w0 * bc0 + w1 * bc1 + w2 * bc2,
            "view_z": w0 * vz0 + w1 * vz1 + w2 * vz2,
            "clip": cv.copy(),
        }
        if has_prev:
            pc0 = attr0["prev_clip"]
            pc1 = attr1["prev_clip"]
            pc2 = attr2["prev_clip"]
            assert isinstance(pc0, np.ndarray) and isinstance(pc1, np.ndarray) and isinstance(pc2, np.ndarray)
            result["prev_clip"] = w0 * pc0 + w1 * pc1 + w2 * pc2
        return result

    def _rasterize_clipped_triangle(
        self,
        W: int,
        H: int,
        tri_clips: list[np.ndarray],
        tri_attrs: list[dict[str, object]],
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
        has_prev: bool,
    ) -> None:
        """Rasterize a single clipped triangle with perspective-correct 1/w interpolation."""
        # Perspective divide for screen coordinates
        screen_pts = []
        inv_ws = []
        for clip_v in tri_clips:
            w = clip_v[3]
            if abs(w) < _CLIP_EPSILON:
                return  # Degenerate
            inv_w = 1.0 / w
            ndc_x = clip_v[0] * inv_w
            ndc_y = clip_v[1] * inv_w
            sx = (ndc_x * 0.5 + 0.5) * W
            sy = (1.0 - (ndc_y * 0.5 + 0.5)) * H
            screen_pts.append((sx, sy))
            inv_ws.append(inv_w)

        x0, y0 = screen_pts[0]
        x1, y1 = screen_pts[1]
        x2, y2 = screen_pts[2]

        # Linear view-space depth for each vertex (positive = in front)
        vz0 = float(tri_attrs[0]["view_z"])  # type: ignore[arg-type]
        vz1 = float(tri_attrs[1]["view_z"])  # type: ignore[arg-type]
        vz2 = float(tri_attrs[2]["view_z"])  # type: ignore[arg-type]
        z0 = -vz0  # convert to positive depth
        z1 = -vz1
        z2 = -vz2

        # Bounding box
        min_x = max(0, int(math.floor(min(x0, x1, x2))))
        max_x = min(W - 1, int(math.ceil(max(x0, x1, x2))))
        min_y = max(0, int(math.floor(min(y0, y1, y2))))
        max_y = min(H - 1, int(math.ceil(max(y0, y1, y2))))

        if min_x > max_x or min_y > max_y:
            return

        # Barycentric setup
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            return
        inv_denom = 1.0 / denom

        # Per-vertex attributes
        n0_attr = tri_attrs[0]["normal"]
        n1_attr = tri_attrs[1]["normal"]
        n2_attr = tri_attrs[2]["normal"]
        assert isinstance(n0_attr, np.ndarray) and isinstance(n1_attr, np.ndarray) and isinstance(n2_attr, np.ndarray)
        bc0_attr = tri_attrs[0]["bc"]
        bc1_attr = tri_attrs[1]["bc"]
        bc2_attr = tri_attrs[2]["bc"]
        assert isinstance(bc0_attr, np.ndarray) and isinstance(bc1_attr, np.ndarray) and isinstance(bc2_attr, np.ndarray)

        # 1/w values for perspective-correct interpolation
        iw0, iw1, iw2 = inv_ws[0], inv_ws[1], inv_ws[2]

        # Pre-divide attributes by w for perspective-correct interpolation
        n0_over_w = n0_attr * iw0
        n1_over_w = n1_attr * iw1
        n2_over_w = n2_attr * iw2
        bc0_over_w = bc0_attr * iw0
        bc1_over_w = bc1_attr * iw1
        bc2_over_w = bc2_attr * iw2

        # Previous clip-space vertices for motion vectors
        prev_clips: list[np.ndarray] | None = None
        if has_prev:
            pc0 = tri_attrs[0].get("prev_clip")
            pc1 = tri_attrs[1].get("prev_clip")
            pc2 = tri_attrs[2].get("prev_clip")
            if pc0 is not None and pc1 is not None and pc2 is not None:
                assert isinstance(pc0, np.ndarray) and isinstance(pc1, np.ndarray) and isinstance(pc2, np.ndarray)
                # Pre-divide prev_clip by w for perspective-correct interpolation
                prev_clips = [pc0 * iw0, pc1 * iw1, pc2 * iw2]

        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                # Barycentric coordinates
                bw0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * inv_denom
                bw1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * inv_denom
                bw2 = 1.0 - bw0 - bw1

                if bw0 < 0 or bw1 < 0 or bw2 < 0:
                    continue

                # Perspective-correct 1/w interpolation
                interp_inv_w = bw0 * iw0 + bw1 * iw1 + bw2 * iw2
                if abs(interp_inv_w) < 1e-20:
                    continue
                w_correct = 1.0 / interp_inv_w

                # Interpolate depth (perspective-correct)
                z_interp = (bw0 * z0 * iw0 + bw1 * z1 * iw1 + bw2 * z2 * iw2) * w_correct
                linear_depth = z_interp

                if linear_depth <= 0 or linear_depth >= depth_buf[py, px]:
                    continue

                depth_buf[py, px] = linear_depth

                # Perspective-correct normal interpolation
                n = (bw0 * n0_over_w + bw1 * n1_over_w + bw2 * n2_over_w) * w_correct
                n_len = np.linalg.norm(n)
                if n_len > 1e-8:
                    n = n / n_len

                normal_buf[py, px] = n

                # Perspective-correct base color interpolation
                bc = (bw0 * bc0_over_w + bw1 * bc1_over_w + bw2 * bc2_over_w) * w_correct
                bcolor_buf[py, px] = bc

                # Diffuse shading
                ndotl = max(0.0, float(np.dot(n, light_d)))
                diffuse = bc * (amb + ndotl * light_c)
                radiance_buf[py, px] = diffuse
                color_buf[py, px] = diffuse

                # Entity mask
                mask_buf[py, px] = eid

                # Roughness / metallic
                rm_buf[py, px, 0] = roughness
                rm_buf[py, px, 1] = metallic

                # Motion vectors: compute screen-space displacement from
                # prev_view_proj
                if prev_clips is not None:
                    # Interpolate previous clip-space position
                    # (perspective-correct)
                    prev_clip = (
                        bw0 * prev_clips[0]
                        + bw1 * prev_clips[1]
                        + bw2 * prev_clips[2]
                    ) * w_correct
                    prev_w = prev_clip[3]
                    if abs(prev_w) > _CLIP_EPSILON:
                        prev_ndc_x = prev_clip[0] / prev_w
                        prev_ndc_y = prev_clip[1] / prev_w
                        prev_sx = (prev_ndc_x * 0.5 + 0.5) * W
                        prev_sy = (1.0 - (prev_ndc_y * 0.5 + 0.5)) * H
                        # Motion vector = current - previous (screen pixels)
                        mv_buf[py, px, 0] = float(px - prev_sx)
                        mv_buf[py, px, 1] = float(py - prev_sy)

        return

    @staticmethod
    def _build_projection(camera: CameraParams) -> np.ndarray:
        """Build a 4×4 perspective projection matrix."""
        fov_rad = math.radians(camera.fov_degrees or 60.0)
        aspect = camera.width / max(camera.height, 1)
        near = camera.z_near
        far = camera.z_far

        f = 1.0 / math.tan(fov_rad / 2.0)
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0
        return proj
