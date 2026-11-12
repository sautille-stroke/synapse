"""Fibonacci-sphere camera generation and silhouette-IoU pose alignment.

Implements the coarse-to-fine camera alignment pipeline described in
RFC 0044 Appendix E:

1. **Coarse search** — Fibonacci-spiral cameras on a bounding sphere,
   rasterise silhouettes, rank by IoU against a reference mask.
2. **Refinement** — Nelder-Mead optimisation around the Top-3 coarse
   candidates, maximising silhouette IoU.
3. **Quality gate** — reject if best ``silhouette_iou < 0.88``.
"""

from __future__ import annotations

import math

import numpy as np

from synapse.errors import QualityGateError
from synapse.inspect.schemas import CameraPose

# ------------------------------------------------------------------
# Fibonacci sphere camera generation
# ------------------------------------------------------------------

def generate_fibonacci_sphere_cameras(
    n_cameras: int = 48,
    radius_scale: float = 2.0,
    fov_deg: float = 50.0,
) -> list[CameraPose]:
    """Generate uniformly distributed cameras on a sphere.

    Uses the Fibonacci spiral (golden-angle) method to place *n_cameras*
    points on a unit sphere, scaled by *radius_scale*.  Every camera
    looks at the origin.

    Parameters
    ----------
    n_cameras:
        Number of camera viewpoints to generate.
    radius_scale:
        Multiplier applied to the unit sphere radius.
    fov_deg:
        Vertical field of view in degrees for each camera.

    Returns
    -------
    list[CameraPose]
        Camera poses uniformly covering the sphere.
    """
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    cameras: list[CameraPose] = []

    for i in range(n_cameras):
        # Latitude: arccosine of linearly spaced z in [-1, 1]
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / n_cameras)
        # Longitude: golden-angle increment
        phi = 2.0 * math.pi * i / golden_ratio

        x = radius_scale * math.sin(theta) * math.cos(phi)
        y = radius_scale * math.cos(theta)
        z = radius_scale * math.sin(theta) * math.sin(phi)

        position = [x, y, z]

        # Build look-at rotation (camera looks at origin, Y-up)
        rot = _look_at_rotation(
            np.array(position, dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        # Store as column-major 3x3 (9 floats)
        rot_list = rot.T.flatten().tolist()

        cameras.append(CameraPose(
            position=position,
            rotation=rot_list,
            fov_deg=fov_deg,
        ))

    return cameras


# ------------------------------------------------------------------
# Silhouette rendering & IoU
# ------------------------------------------------------------------

def compute_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_pose: CameraPose,
    resolution: int = 128,
) -> np.ndarray:
    """Render a binary silhouette mask via Z-buffer rasterisation.

    Parameters
    ----------
    vertices:
        ``[N, 3]`` mesh vertex positions.
    faces:
        ``[F, 3]`` triangle face indices.
    camera_pose:
        Camera from which to render.
    resolution:
        Output square image size.

    Returns
    -------
    np.ndarray
        ``[resolution, resolution]`` binary mask (``True`` where mesh
        is visible).
    """
    view = _camera_pose_to_view_matrix(camera_pose)
    proj = _build_projection(camera_pose.fov_deg, resolution, resolution)

    verts = np.asarray(vertices, dtype=np.float64)
    n = verts.shape[0]

    # Homogeneous coords
    ones = np.ones((n, 1), dtype=np.float64)
    verts_h = np.hstack([verts, ones])  # [N, 4]

    # View transform
    verts_view = (view @ verts_h.T).T  # [N, 4]
    # Projection
    verts_clip = (proj @ verts_view.T).T  # [N, 4]

    # Perspective divide
    w_clip = verts_clip[:, 3:4].copy()
    w_clip[np.abs(w_clip) < 1e-10] = 1e-10
    verts_ndc = verts_clip[:, :3] / w_clip

    # Screen coords
    sx = (verts_ndc[:, 0] * 0.5 + 0.5) * resolution
    sy = (1.0 - (verts_ndc[:, 1] * 0.5 + 0.5)) * resolution
    sz = verts_view[:, 2]  # negative = in front of camera

    W = H = resolution
    depth_buf = np.full((H, W), np.inf, dtype=np.float64)
    mask = np.zeros((H, W), dtype=bool)

    faces_arr = np.asarray(faces, dtype=np.int32)
    for fi in range(faces_arr.shape[0]):
        i0, i1, i2 = faces_arr[fi]

        x0, y0, z0 = sx[i0], sy[i0], sz[i0]
        x1, y1, z1 = sx[i1], sy[i1], sz[i1]
        x2, y2, z2 = sx[i2], sy[i2], sz[i2]

        # Skip triangles fully behind camera
        if z0 > -0.1 and z1 > -0.1 and z2 > -0.1:
            continue

        # Bounding box
        min_x = max(0, int(math.floor(min(x0, x1, x2))))
        max_x = min(W - 1, int(math.ceil(max(x0, x1, x2))))
        min_y = max(0, int(math.floor(min(y0, y1, y2))))
        max_y = min(H - 1, int(math.ceil(max(y0, y1, y2))))

        if min_x > max_x or min_y > max_y:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            continue
        inv_denom = 1.0 / denom

        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                w0 = (
                    (y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)
                ) * inv_denom
                w1 = (
                    (y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)
                ) * inv_denom
                w2 = 1.0 - w0 - w1

                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue

                z_interp = w0 * z0 + w1 * z1 + w2 * z2
                linear_depth = -z_interp

                if linear_depth <= 0:
                    continue
                if linear_depth >= depth_buf[py, px]:
                    continue

                depth_buf[py, px] = linear_depth
                mask[py, px] = True

    return mask


def compute_silhouette_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Intersection over Union of two binary masks.

    Parameters
    ----------
    mask_a, mask_b:
        ``[H, W]`` boolean or integer masks.

    Returns
    -------
    float
        IoU in ``[0, 1]``.  Returns ``1.0`` if both masks are empty.
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)

    intersection = np.sum(a & b)
    union = np.sum(a | b)

    if union == 0:
        return 1.0

    return float(intersection / union)


# ------------------------------------------------------------------
# Coarse-to-fine alignment
# ------------------------------------------------------------------

def align_reference_image_to_mesh(
    reference_silhouette: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    n_coarse: int = 48,
    n_refine_iters: int = 50,
) -> tuple[CameraPose, float]:
    """Align a reference silhouette to a 3D mesh via Fibonacci search.

    **Stage 1A** — generate *n_coarse* Fibonacci-sphere cameras,
    render silhouettes, rank by IoU.

    **Stage 1B** — refine Top-3 candidates with Nelder-Mead
    optimisation on silhouette IoU.

    **Gate** — reject if best ``silhouette_iou < 0.88``.

    Parameters
    ----------
    reference_silhouette:
        ``[H, W]`` binary reference mask.
    vertices:
        ``[N, 3]`` mesh vertex positions.
    faces:
        ``[F, 3]`` triangle face indices.
    n_coarse:
        Number of Fibonacci-sphere cameras for coarse search.
    n_refine_iters:
        Maximum optimiser iterations per candidate.

    Returns
    -------
    tuple[CameraPose, float]
        Best camera pose and achieved IoU.

    Raises
    ------
    QualityGateError
        If best IoU < 0.88 after refinement.
    """
    from scipy.optimize import minimize  # lazy import

    ref = np.asarray(reference_silhouette, dtype=bool)
    res = ref.shape[0]  # assume square

    # Compute mesh bounding sphere for radius scale
    centroid = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - centroid, axis=1)
    bbox_radius = float(np.max(radii))
    radius_scale = bbox_radius * 2.5

    # Stage 1A: coarse search
    coarse_cameras = generate_fibonacci_sphere_cameras(
        n_cameras=n_coarse,
        radius_scale=radius_scale,
        fov_deg=50.0,
    )

    coarse_scores: list[tuple[float, int]] = []
    for idx, cam in enumerate(coarse_cameras):
        sil = compute_silhouette(vertices, faces, cam, resolution=res)
        iou = compute_silhouette_iou(ref, sil)
        coarse_scores.append((iou, idx))

    coarse_scores.sort(key=lambda x: x[0], reverse=True)
    top_k = min(3, len(coarse_scores))
    candidates = [
        coarse_cameras[coarse_scores[i][1]] for i in range(top_k)
    ]
    candidate_ious = [coarse_scores[i][0] for i in range(top_k)]

    # Stage 1B: refine each candidate
    best_iou = -1.0
    best_pose: CameraPose = candidates[0]

    for ci, cam in enumerate(candidates):
        # Parameterise as spherical coords (theta, phi, radius)
        pos = np.array(cam.position, dtype=np.float64)
        r = float(np.linalg.norm(pos))
        theta = math.acos(np.clip(pos[1] / max(r, 1e-10), -1, 1))
        phi = math.atan2(pos[2], pos[0])

        def _neg_iou(params: np.ndarray) -> float:
            th, ph, rad = params
            x = rad * math.sin(th) * math.cos(ph)
            y = rad * math.cos(th)
            z = rad * math.sin(th) * math.sin(ph)
            trial_pose = CameraPose(
                position=[x, y, z],
                rotation=_look_at_rotation(
                    np.array([x, y, z], dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    np.array([0.0, 1.0, 0.0], dtype=np.float64),
                ).T.flatten().tolist(),
                fov_deg=cam.fov_deg,
                resolution=(res, res),
            )
            sil = compute_silhouette(
                vertices, faces, trial_pose, resolution=res,
            )
            return -compute_silhouette_iou(ref, sil)

        x0 = np.array([theta, phi, r], dtype=np.float64)
        result = minimize(
            _neg_iou,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": n_refine_iters,
                "xatol": 1e-3,
                "fatol": 1e-4,
            },
        )

        refined_iou = -float(result.fun)
        if refined_iou > best_iou:
            best_iou = refined_iou
            th, ph, rad = result.x
            bx = rad * math.sin(th) * math.cos(ph)
            by = rad * math.cos(th)
            bz = rad * math.sin(th) * math.sin(ph)
            best_pose = CameraPose(
                position=[bx, by, bz],
                rotation=_look_at_rotation(
                    np.array([bx, by, bz], dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    np.array([0.0, 1.0, 0.0], dtype=np.float64),
                ).T.flatten().tolist(),
                fov_deg=cam.fov_deg,
                resolution=(res, res),
            )

    # Quality gate
    if best_iou < 0.88:
        raise QualityGateError(
            message=(
                f"Silhouette IoU {best_iou:.4f} < 0.88 threshold. "
                "Pose alignment failed — reference image may not "
                "match the mesh."
            ),
            agent_fix_hint=(
                "Re-capture the reference image from a canonical "
                "viewpoint or verify the mesh asset_id matches."
            ),
            details={"best_iou": best_iou},
        )

    return best_pose, best_iou


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Compute a 3x3 rotation matrix for a look-at camera.

    Returns a row-major rotation where:
    - Row 0 = right
    - Row 1 = camera up
    - Row 2 = -forward (camera looks along -Z)
    """
    forward = target - eye
    f_len = np.linalg.norm(forward)
    if f_len < 1e-12:
        return np.eye(3, dtype=np.float64)
    forward = forward / f_len

    right = np.cross(forward, up)
    r_len = np.linalg.norm(right)
    if r_len < 1e-12:
        # Fallback: pick an arbitrary perpendicular
        alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(forward, alt_up)
        r_len = np.linalg.norm(right)
        if r_len < 1e-12:
            return np.eye(3, dtype=np.float64)
    right = right / r_len

    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)

    # Row-major: right, up, -forward
    rot = np.array([
        right,
        cam_up,
        -forward,
    ], dtype=np.float64)

    return rot


def _camera_pose_to_view_matrix(pose: CameraPose) -> np.ndarray:
    """Convert a CameraPose to a 4x4 view (world-to-camera) matrix.

    Parameters
    ----------
    pose:
        Camera pose with position and rotation.

    Returns
    -------
    np.ndarray
        ``[4, 4]`` view matrix.
    """
    pos = np.array(pose.position, dtype=np.float64)

    rot_vals = pose.rotation
    if len(rot_vals) == 9:
        # Column-major 3x3 -> row-major by transposing
        rot_col = np.array(rot_vals, dtype=np.float64).reshape(3, 3)
        rot = rot_col.T
    elif len(rot_vals) == 4:
        rot = _quat_to_matrix(rot_vals)
    else:
        rot = np.eye(3, dtype=np.float64)

    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = rot
    view[:3, 3] = -rot @ pos
    return view


def _view_matrix_to_camera_pose(
    view_matrix: np.ndarray,
    fov_deg: float,
) -> CameraPose:
    """Convert a 4x4 view matrix back to a CameraPose.

    Parameters
    ----------
    view_matrix:
        ``[4, 4]`` world-to-camera matrix.
    fov_deg:
        Vertical field of view in degrees.

    Returns
    -------
    CameraPose
    """
    view = np.asarray(view_matrix, dtype=np.float64)
    rot = view[:3, :3]
    t = view[:3, 3]

    # Position: -R^T * t
    pos = (-rot.T @ t).tolist()
    # Column-major rotation
    rot_col = rot.T.flatten().tolist()

    return CameraPose(
        position=pos,
        rotation=rot_col,
        fov_deg=fov_deg,
    )


def _quat_to_matrix(q: list[float]) -> np.ndarray:
    """Convert quaternion ``[x, y, z, w]`` to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _build_projection(
    fov_deg: float,
    width: int,
    height: int,
    z_near: float = 0.1,
    z_far: float = 1000.0,
) -> np.ndarray:
    """Build a 4x4 perspective projection matrix."""
    fov_rad = math.radians(fov_deg)
    aspect = width / max(height, 1)
    f = 1.0 / math.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = -(z_far + z_near) / (z_far - z_near)
    proj[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)
    proj[3, 2] = -1.0
    return proj
