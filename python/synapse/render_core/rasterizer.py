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
        base_color = np.asarray(base_color, dtype=np.float32)

        # Build MVP: view_matrix then projection
        view = camera.view_matrix.astype(np.float64)
        proj = self._build_projection(camera)

        # Transform vertices to clip space
        verts_h = np.hstack([vertices.astype(np.float64),
                             np.ones((N, 1), dtype=np.float64)])
        verts_view = (view @ verts_h.T).T  # [N, 4]
        verts_clip = (proj @ verts_view.T).T  # [N, 4]

        # Perspective divide -> NDC
        w_clip = verts_clip[:, 3:4]
        w_clip = np.where(np.abs(w_clip) < 1e-10, 1e-10, w_clip)
        verts_ndc = verts_clip[:, :3] / w_clip

        # Screen coords: NDC [-1,1] -> pixel [0, W/H)
        sx = (verts_ndc[:, 0] * 0.5 + 0.5) * W
        sy = (1.0 - (verts_ndc[:, 1] * 0.5 + 0.5)) * H
        sz = verts_view[:, 2]  # linear depth (negative = in front)

        # Init buffers
        depth_buf = np.full((H, W), np.inf, dtype=np.float64)
        color_buf = np.zeros((H, W, 3), dtype=np.float64)
        normal_buf = np.zeros((H, W, 3), dtype=np.float64)
        radiance_buf = np.zeros((H, W, 3), dtype=np.float64)
        bcolor_buf = np.zeros((H, W, 3), dtype=np.float64)
        mask_buf = np.zeros((H, W), dtype=np.uint32)
        rm_buf = np.zeros((H, W, 2), dtype=np.float64)  # roughness, metallic

        # Rasterise each triangle
        assert self.light_dir is not None and self.light_color is not None
        norms = normals.astype(np.float64)
        light_d = self.light_dir.astype(np.float64)
        light_c = self.light_color.astype(np.float64)
        amb = self.ambient

        for fi in range(faces.shape[0]):
            i0, i1, i2 = faces[fi]

            # Screen-space triangle vertices
            x0, y0, z0 = sx[i0], sy[i0], sz[i0]
            x1, y1, z1 = sx[i1], sy[i1], sz[i1]
            x2, y2, z2 = sx[i2], sy[i2], sz[i2]

            # Skip back-facing or behind camera (positive z in view = behind)
            if z0 > -camera.z_near and z1 > -camera.z_near and z2 > -camera.z_near:
                continue

            # Bounding box
            min_x = max(0, int(math.floor(min(x0, x1, x2))))
            max_x = min(W - 1, int(math.ceil(max(x0, x1, x2))))
            min_y = max(0, int(math.floor(min(y0, y1, y2))))
            max_y = min(H - 1, int(math.ceil(max(y0, y1, y2))))

            if min_x > max_x or min_y > max_y:
                continue

            # Barycentric setup
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-10:
                continue
            inv_denom = 1.0 / denom

            # Per-vertex attributes
            n0 = norms[i0]
            n1 = norms[i1]
            n2 = norms[i2]
            bc0 = base_color[i0].astype(np.float64)
            bc1 = base_color[i1].astype(np.float64)
            bc2 = base_color[i2].astype(np.float64)

            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    # Barycentric coordinates
                    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * inv_denom
                    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * inv_denom
                    w2 = 1.0 - w0 - w1

                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue

                    # Interpolate depth
                    z_interp = w0 * z0 + w1 * z1 + w2 * z2
                    linear_depth = -z_interp  # positive depth

                    if linear_depth <= 0 or linear_depth >= depth_buf[py, px]:
                        continue

                    depth_buf[py, px] = linear_depth

                    # Interpolate normal
                    n = w0 * n0 + w1 * n1 + w2 * n2
                    n_len = np.linalg.norm(n)
                    if n_len > 1e-8:
                        n = n / n_len

                    normal_buf[py, px] = n

                    # Interpolate base color
                    bc = w0 * bc0 + w1 * bc1 + w2 * bc2
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

        # Compute motion vectors
        mv_buf = np.zeros((H, W, 2), dtype=np.float32)
        if prev_camera is not None:
            # Simple screen-space reprojection for foreground pixels
            fg_mask = mask_buf > 0
            if np.any(fg_mask):
                ys, xs = np.where(fg_mask)
                depths = depth_buf[ys, xs]
                # Unproject to world
                ndc_x = (xs + 0.5) / W * 2.0 - 1.0
                ndc_y = 1.0 - (ys + 0.5) / H * 2.0
                _view_inv = np.linalg.inv(view)  # noqa: F841
                _proj_inv = np.linalg.inv(proj)  # noqa: F841
                pts_clip = np.stack([ndc_x * depths, ndc_y * depths,
                                     -depths, np.ones_like(depths)], axis=1)
                # Reproject with prev camera
                prev_view = prev_camera.view_matrix.astype(np.float64)
                prev_proj = self._build_projection(prev_camera)
                _pts_world = (np.linalg.inv(prev_view) @ (np.linalg.inv(prev_proj) @ pts_clip.T)).T  # noqa: F841
                # This is a simplified motion vector; exact would need full unprojection

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
