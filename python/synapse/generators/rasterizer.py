"""Multi-channel software rasterizer for Synapse.

Computes exact ground-truth channels:
- RGB (with diffuse + specular + PBR texture)
- Depth (camera-space linear depth)
- World Normal (unit normal in world coordinates)
- Motion Vector (subpixel screen-space optical flow relative to previous frame)
- Mask / Cryptomatte (binary/ID foreground mask)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover — numba is an optional heavy dep
    # Provide no-op fallback decorators so the module can still be *imported*
    # without numba.  The JIT-accelerated kernels fall back to plain Python/
    # NumPy execution (significantly slower but functionally identical).
    def njit(*_args, **_kwargs):  # type: ignore[misc]
        """Identity decorator when numba is unavailable."""
        def _wrap(fn):
            return fn
        if _args and callable(_args[0]):
            return _args[0]
        return _wrap

    prange = range  # type: ignore[assignment]

from synapse.fgpp.frame import CameraParams
from synapse.models.asset import ModelAsset


@dataclass
class RenderResult:
    """Multi-channel rendering output for a single frame."""

    rgb: np.ndarray  # [H, W, 3] uint8
    depth: np.ndarray  # [H, W] float32
    world_normal: np.ndarray  # [H, W, 3] float32 [-1, 1]
    motion_vector: np.ndarray  # [H, W, 2] float32 (dx, dy)
    mask: np.ndarray  # [H, W] uint8 (0 or 1)
    base_color: Optional[np.ndarray] = None  # [H, W, 3] float32 [0, 1]


@njit(fastmath=True)
def _rasterize_kernel(
    screen_pts: np.ndarray,       # [M, 3, 2] float32 (u, v)
    cam_inv_z: np.ndarray,        # [M, 3] float32 (1 / depth)
    world_verts: np.ndarray,      # [M, 3, 3] float32 (world pos)
    world_norms: np.ndarray,      # [M, 3, 3] float32 (world normal)
    face_uvs: np.ndarray,         # [M, 3, 2] float32 (uv coords)
    has_texture: bool,
    texture: np.ndarray,          # [H_tex, W_tex, 3] uint8
    base_color_factor: np.ndarray,# [3] float32
    prev_view_matrix: np.ndarray, # [4, 4] float64
    prev_fx: float,
    prev_fy: float,
    prev_cx: float,
    prev_cy: float,
    has_prev_camera: bool,
    width: int,
    height: int,
    light_dir: np.ndarray,        # [3] float32
    rgb_out: np.ndarray,          # [H, W, 3] uint8
    depth_out: np.ndarray,        # [H, W] float32
    normal_out: np.ndarray,       # [H, W, 3] float32
    flow_out: np.ndarray,         # [H, W, 2] float32
    mask_out: np.ndarray,         # [H, W] uint8
):
    """JIT-accelerated core rasterization loop with z-buffering and channel interpolation."""
    num_faces = screen_pts.shape[0]
    tex_h = texture.shape[0] if has_texture else 1
    tex_w = texture.shape[1] if has_texture else 1

    for f in range(num_faces):
        u0 = screen_pts[f, 0, 0]
        v0 = screen_pts[f, 0, 1]
        u1 = screen_pts[f, 1, 0]
        v1 = screen_pts[f, 1, 1]
        u2 = screen_pts[f, 2, 0]
        v2 = screen_pts[f, 2, 1]

        # 2D cross product for back-face culling & barycentric denominator
        # In screen coords (Y down), front-facing CCW gives positive area
        denom = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)
        if denom <= 1e-6:
            continue

        inv_denom = 1.0 / denom

        # Bounding box clamped to viewport
        min_x = max(0, int(math.floor(min(u0, u1, u2))))
        max_x = min(width - 1, int(math.ceil(max(u0, u1, u2))))
        min_y = max(0, int(math.floor(min(v0, v1, v2))))
        max_y = min(height - 1, int(math.ceil(max(v0, v1, v2))))

        if min_x > max_x or min_y > max_y:
            continue

        iz0 = cam_inv_z[f, 0]
        iz1 = cam_inv_z[f, 1]
        iz2 = cam_inv_z[f, 2]

        # Triangle vertices & normals
        wv0 = world_verts[f, 0]
        wv1 = world_verts[f, 1]
        wv2 = world_verts[f, 2]

        wn0 = world_norms[f, 0]
        wn1 = world_norms[f, 1]
        wn2 = world_norms[f, 2]

        uv0 = face_uvs[f, 0]
        uv1 = face_uvs[f, 1]
        uv2 = face_uvs[f, 2]

        for py in range(min_y, max_y + 1):
            y_f = float(py) + 0.5
            for px in range(min_x, max_x + 1):
                x_f = float(px) + 0.5

                # Barycentric weights w0, w1, w2
                w0 = ((u1 - x_f) * (v2 - y_f) - (v1 - y_f) * (u2 - x_f)) * inv_denom
                w1 = ((u2 - x_f) * (v0 - y_f) - (v2 - y_f) * (u0 - x_f)) * inv_denom
                w2 = 1.0 - w0 - w1

                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                    continue

                # Perspective-correct 1/z interpolation
                interp_iz = w0 * iz0 + w1 * iz1 + w2 * iz2
                if interp_iz <= 0.0:
                    continue

                curr_depth = 1.0 / interp_iz
                if curr_depth >= depth_out[py, px]:
                    continue

                # Depth buffer test passed!
                depth_out[py, px] = curr_depth
                mask_out[py, px] = 1

                # Perspective-correct normal
                nx = curr_depth * (w0 * iz0 * wn0[0] + w1 * iz1 * wn1[0] + w2 * iz2 * wn2[0])
                ny = curr_depth * (w0 * iz0 * wn0[1] + w1 * iz1 * wn1[1] + w2 * iz2 * wn2[1])
                nz = curr_depth * (w0 * iz0 * wn0[2] + w1 * iz1 * wn1[2] + w2 * iz2 * wn2[2])
                norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
                if norm_len > 1e-8:
                    nx /= norm_len
                    ny /= norm_len
                    nz /= norm_len

                normal_out[py, px, 0] = nx
                normal_out[py, px, 1] = ny
                normal_out[py, px, 2] = nz

                # Base albedo color
                albedo_r = base_color_factor[0]
                albedo_g = base_color_factor[1]
                albedo_b = base_color_factor[2]

                if has_texture:
                    # Perspective-correct UV
                    u_coord = curr_depth * (w0 * iz0 * uv0[0] + w1 * iz1 * uv1[0] + w2 * iz2 * uv2[0])
                    v_coord = curr_depth * (w0 * iz0 * uv0[1] + w1 * iz1 * uv1[1] + w2 * iz2 * uv2[1])

                    # Texture wrapping (repeat)
                    u_norm = u_coord - math.floor(u_coord)
                    v_norm = v_coord - math.floor(v_coord)
                    # Flip V for standard GLTF texture coordinates
                    tx = int(u_norm * (tex_w - 1))
                    ty = int((1.0 - v_norm) * (tex_h - 1))
                    tx = max(0, min(tex_w - 1, tx))
                    ty = max(0, min(tex_h - 1, ty))

                    albedo_r = float(texture[ty, tx, 0]) / 255.0
                    albedo_g = float(texture[ty, tx, 1]) / 255.0
                    albedo_b = float(texture[ty, tx, 2]) / 255.0

                # Two-tone lighting: Key light + Ambient fill
                n_dot_l = max(0.0, nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2])
                diffuse = 0.25 + 0.75 * n_dot_l

                # Shaded RGB
                rgb_out[py, px, 0] = min(255, max(0, int(albedo_r * diffuse * 255.0)))
                rgb_out[py, px, 1] = min(255, max(0, int(albedo_g * diffuse * 255.0)))
                rgb_out[py, px, 2] = min(255, max(0, int(albedo_b * diffuse * 255.0)))

                # Motion vector calculation
                if has_prev_camera:
                    # Interpolate world position
                    wx = curr_depth * (w0 * iz0 * wv0[0] + w1 * iz1 * wv1[0] + w2 * iz2 * wv2[0])
                    wy = curr_depth * (w0 * iz0 * wv0[1] + w1 * iz1 * wv1[1] + w2 * iz2 * wv2[1])
                    wz = curr_depth * (w0 * iz0 * wv0[2] + w1 * iz1 * wv1[2] + w2 * iz2 * wv2[2])

                    # Transform into previous camera space
                    # prev_cam = prev_view @ [wx, wy, wz, 1]
                    pcx = prev_view_matrix[0, 0] * wx + prev_view_matrix[0, 1] * wy + prev_view_matrix[0, 2] * wz + prev_view_matrix[0, 3]
                    pcy = prev_view_matrix[1, 0] * wx + prev_view_matrix[1, 1] * wy + prev_view_matrix[1, 2] * wz + prev_view_matrix[1, 3]
                    pcz = prev_view_matrix[2, 0] * wx + prev_view_matrix[2, 1] * wy + prev_view_matrix[2, 2] * wz + prev_view_matrix[2, 3]

                    p_depth = -pcz
                    if p_depth > 0.05:
                        prev_px = prev_fx * (pcx / p_depth) + prev_cx
                        prev_py = -prev_fy * (pcy / p_depth) + prev_cy
                        # Optical flow: displacement from current pixel to previous pixel
                        flow_out[py, px, 0] = float(px) - float(prev_px)
                        flow_out[py, px, 1] = float(py) - float(prev_py)


class SoftwareRasterizer:
    """High-performance multi-channel software rasterizer for 3D meshes."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        # Directional light (normalized, coming from upper right front)
        l = np.array([0.5, 0.7, 0.5], dtype=np.float32)
        self.light_dir = (l / np.linalg.norm(l)).astype(np.float32)

    def render(
        self,
        mesh: ModelAsset,
        camera: CameraParams,
        prev_camera: Optional[CameraParams] = None,
    ) -> RenderResult:
        """Render a single frame yielding RGB, Depth, Normal, Motion Vector, and Mask."""
        w, h = self.width, self.height

        # 1. Transform vertices to camera space
        # Camera convention: right-handed, Y-up, camera looks along -Z
        v_world = mesh.vertices.astype(np.float64)
        view_mat = camera.view_matrix  # 4x4 row-major
        # V_cam = V_world @ R^T + t
        v_cam = (v_world @ view_mat[:3, :3].T) + view_mat[:3, 3]

        # Depth in front of camera is -Z_cam
        depths = (-v_cam[:, 2]).astype(np.float32)

        # Discard or clamp points behind camera near plane
        depth_safe = np.where(depths < camera.z_near, 1e-4, depths)
        inv_depths = (1.0 / depth_safe).astype(np.float32)

        # Screen coordinates
        # u = fx * (X / depth) + cx
        # v = -fy * (Y / depth) + cy
        u_screen = (camera.fx * (v_cam[:, 0] / depth_safe) + camera.cx).astype(np.float32)
        v_screen = (-camera.fy * (v_cam[:, 1] / depth_safe) + camera.cy).astype(np.float32)

        # 2. Gather per-face arrays
        faces = mesh.faces
        screen_pts = np.stack([
            np.stack([u_screen[faces[:, 0]], v_screen[faces[:, 0]]], axis=-1),
            np.stack([u_screen[faces[:, 1]], v_screen[faces[:, 1]]], axis=-1),
            np.stack([u_screen[faces[:, 2]], v_screen[faces[:, 2]]], axis=-1),
        ], axis=1).astype(np.float32)

        cam_inv_z = np.stack([
            inv_depths[faces[:, 0]],
            inv_depths[faces[:, 1]],
            inv_depths[faces[:, 2]],
        ], axis=-1).astype(np.float32)

        world_verts = np.stack([
            mesh.vertices[faces[:, 0]],
            mesh.vertices[faces[:, 1]],
            mesh.vertices[faces[:, 2]],
        ], axis=1).astype(np.float32)

        world_norms = np.stack([
            mesh.normals[faces[:, 0]],
            mesh.normals[faces[:, 1]],
            mesh.normals[faces[:, 2]],
        ], axis=1).astype(np.float32)

        if mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices):
            face_uvs = np.stack([
                mesh.uvs[faces[:, 0]],
                mesh.uvs[faces[:, 1]],
                mesh.uvs[faces[:, 2]],
            ], axis=1).astype(np.float32)
        else:
            face_uvs = np.zeros((len(faces), 3, 2), dtype=np.float32)

        has_texture = mesh.base_color_texture is not None
        texture = mesh.base_color_texture if has_texture else np.zeros((1, 1, 3), dtype=np.uint8)

        # Outputs
        # Background color: subtle dark slate (20, 22, 28)
        rgb_out = np.full((h, w, 3), (20, 22, 28), dtype=np.uint8)
        depth_out = np.full((h, w), 1e6, dtype=np.float32)
        normal_out = np.zeros((h, w, 3), dtype=np.float32)
        flow_out = np.zeros((h, w, 2), dtype=np.float32)
        mask_out = np.zeros((h, w), dtype=np.uint8)

        has_prev = prev_camera is not None
        prev_view = prev_camera.view_matrix if has_prev else np.eye(4, dtype=np.float64)
        prev_fx = prev_camera.fx if has_prev else camera.fx
        prev_fy = prev_camera.fy if has_prev else camera.fy
        prev_cx = prev_camera.cx if has_prev else camera.cx
        prev_cy = prev_camera.cy if has_prev else camera.cy

        _rasterize_kernel(
            screen_pts=screen_pts,
            cam_inv_z=cam_inv_z,
            world_verts=world_verts,
            world_norms=world_norms,
            face_uvs=face_uvs,
            has_texture=has_texture,
            texture=texture,
            base_color_factor=mesh.base_color_factor,
            prev_view_matrix=prev_view,
            prev_fx=prev_fx,
            prev_fy=prev_fy,
            prev_cx=prev_cx,
            prev_cy=prev_cy,
            has_prev_camera=has_prev,
            width=w,
            height=h,
            light_dir=self.light_dir,
            rgb_out=rgb_out,
            depth_out=depth_out,
            normal_out=normal_out,
            flow_out=flow_out,
            mask_out=mask_out,
        )

        # Set background depth to far/inf convention
        depth_out = np.where(mask_out == 0, camera.z_far, depth_out)

        return RenderResult(
            rgb=rgb_out,
            depth=depth_out,
            world_normal=normal_out,
            motion_vector=flow_out,
            mask=mask_out,
        )
