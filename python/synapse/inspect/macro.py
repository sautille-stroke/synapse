"""Multi-channel residual heatmap and virtual macro camera 3D grounding.

Implements isotopic residual comparison between reference and rendered
images across RGB, depth-edge, and surface-normal channels, with
connected-component NMS for Top-K ROI extraction and 3D unproject-based
virtual macro camera planning.

Heavy dependencies (``scipy``) are lazy-imported inside functions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synapse.errors import QualityGateError
from synapse.inspect.schemas import CameraPose


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _sobel_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a 2D array.

    Args:
        image: ``[H, W]`` float array.

    Returns:
        ``[H, W]`` gradient magnitude.
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape

    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    # Horizontal Sobel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    if w > 2 and h > 2:
        gx[1:-1, 1:-1] = (
            -img[:-2, :-2] + img[:-2, 2:]
            - 2.0 * img[1:-1, :-2] + 2.0 * img[1:-1, 2:]
            - img[2:, :-2] + img[2:, 2:]
        )
        # Vertical Sobel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
        gy[1:-1, 1:-1] = (
            -img[:-2, :-2] - 2.0 * img[:-2, 1:-1] - img[:-2, 2:]
            + img[2:, :-2] + 2.0 * img[2:, 1:-1] + img[2:, 2:]
        )

    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _rgb_to_lab_approx(rgb: np.ndarray) -> np.ndarray:
    """Approximate RGB [0,1] to perceptual LAB-like space.

    Args:
        rgb: ``[H, W, 3]`` float32 in [0, 1].

    Returns:
        ``[H, W, 3]`` with (L, a, b) channels.
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    l_ch = 0.2126 * r + 0.7152 * g + 0.0722 * b
    a_ch = r - g
    b_ch = (r + g) * 0.5 - b

    return np.stack([l_ch, a_ch, b_ch], axis=-1)


def _unproject_pixel(
    u: float,
    v: float,
    z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Unproject 2D pixel ``(u, v, z)`` to 3D camera-space point.

    Args:
        u, v: Pixel coordinates.
        z: Depth value at the pixel.
        fx, fy: Focal lengths.
        cx, cy: Principal point.

    Returns:
        ``[3]`` float64 camera-space point.
    """
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


# -------------------------------------------------------------------
# compute_isotopic_residual_heatmap
# -------------------------------------------------------------------


def compute_isotopic_residual_heatmap(
    reference_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    depth_reference: np.ndarray | None = None,
    depth_rendered: np.ndarray | None = None,
    normals_reference: np.ndarray | None = None,
    normals_rendered: np.ndarray | None = None,
    top_k: int = 6,
    nms_radius: int = 32,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute multi-channel isotopic residual heatmap.

    Fuses RGB color difference, depth-edge residual, and normal
    angular error into a normalised 2D anomaly heatmap, then
    extracts Top-K ROIs via connected-component NMS.

    Args:
        reference_rgb: ``[H, W, 3]`` float32 in [0, 1].
        rendered_rgb: ``[H, W, 3]`` float32 in [0, 1].
        depth_reference: Optional ``[H, W]`` float32 depth.
        depth_rendered: Optional ``[H, W]`` float32 depth.
        normals_reference: Optional ``[H, W, 3]`` float32 normals.
        normals_rendered: Optional ``[H, W, 3]`` float32 normals.
        top_k: Maximum number of ROIs to return.
        nms_radius: Suppress ROIs whose centroids are within
            this pixel distance of a higher-scoring ROI.

    Returns:
        ``(heatmap, rois)`` where heatmap is ``[H, W]`` float32
        in [0, 1] and rois is a list of dicts with keys:
        ``roi_id``, ``center_uv``, ``bounding_radius_px``,
        ``mean_severity``, ``pixel_count``.
    """
    ref = np.asarray(reference_rgb, dtype=np.float32)
    ren = np.asarray(rendered_rgb, dtype=np.float32)
    h, w = ref.shape[:2]

    # --- Channel 1: RGB perceptual difference ---
    lab_ref = _rgb_to_lab_approx(ref)
    lab_ren = _rgb_to_lab_approx(ren)
    rgb_diff = np.sqrt(
        np.sum((lab_ref - lab_ren) ** 2, axis=-1),
    ).astype(np.float32)

    rgb_max = rgb_diff.max()
    if rgb_max > 1e-8:
        rgb_diff /= rgb_max

    # --- Channel weights ---
    w_rgb = 0.5
    w_depth = 0.3
    w_normal = 0.2

    channels_used = [("rgb", w_rgb, rgb_diff)]
    total_weight = w_rgb

    # --- Channel 2: Depth edge residual ---
    if (
        depth_reference is not None
        and depth_rendered is not None
    ):
        d_ref = np.asarray(depth_reference, dtype=np.float32)
        d_ren = np.asarray(depth_rendered, dtype=np.float32)
        edge_ref = _sobel_gradient_magnitude(d_ref)
        edge_ren = _sobel_gradient_magnitude(d_ren)
        depth_diff = np.abs(edge_ref - edge_ren)
        d_max = depth_diff.max()
        if d_max > 1e-8:
            depth_diff /= d_max
        channels_used.append(("depth", w_depth, depth_diff))
        total_weight += w_depth

    # --- Channel 3: Normal angular error ---
    if (
        normals_reference is not None
        and normals_rendered is not None
    ):
        n_ref = np.asarray(
            normals_reference, dtype=np.float32,
        )
        n_ren = np.asarray(
            normals_rendered, dtype=np.float32,
        )
        # Dot product per pixel
        dot = np.sum(n_ref * n_ren, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        angular_err = np.arccos(dot) / math.pi
        channels_used.append(
            ("normal", w_normal, angular_err),
        )
        total_weight += w_normal

    # --- Fuse channels ---
    heatmap = np.zeros((h, w), dtype=np.float32)
    for _, weight, channel in channels_used:
        heatmap += (weight / total_weight) * channel

    heatmap = np.clip(heatmap, 0.0, 1.0)

    # --- Connected-component NMS ---
    rois = _extract_rois_nms(
        heatmap, top_k=top_k, nms_radius=nms_radius,
    )

    return heatmap, rois


def _extract_rois_nms(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    top_k: int = 6,
    nms_radius: int = 32,
) -> list[dict[str, Any]]:
    """Extract Top-K ROIs from heatmap via connected-component NMS.

    Args:
        heatmap: ``[H, W]`` float32 in [0, 1].
        threshold: Binarisation threshold.
        top_k: Maximum number of ROIs.
        nms_radius: NMS suppression radius in pixels.

    Returns:
        List of ROI dicts sorted by mean_severity descending.
    """
    from scipy.ndimage import label  # lazy import

    binary = heatmap >= threshold
    labeled, n_components = label(binary)

    if n_components == 0:
        return []

    # Compute per-component statistics
    candidates: list[dict[str, Any]] = []
    for comp_id in range(1, n_components + 1):
        mask = labeled == comp_id
        pixels = np.argwhere(mask)  # [N, 2] — (row, col)
        if len(pixels) == 0:
            continue

        center_y = float(pixels[:, 0].mean())
        center_x = float(pixels[:, 1].mean())

        # Bounding radius: max distance from center
        dists = np.sqrt(
            (pixels[:, 0] - center_y) ** 2
            + (pixels[:, 1] - center_x) ** 2,
        )
        bounding_radius = float(dists.max()) if len(dists) > 0 else 0.0

        mean_sev = float(heatmap[mask].mean())
        pixel_count = int(mask.sum())

        candidates.append({
            "roi_id": comp_id,
            "center_uv": (int(round(center_x)), int(round(center_y))),
            "bounding_radius_px": bounding_radius,
            "mean_severity": mean_sev,
            "pixel_count": pixel_count,
        })

    # Sort by mean severity descending
    candidates.sort(
        key=lambda r: r["mean_severity"], reverse=True,
    )

    # NMS: suppress ROIs whose centroid is within nms_radius
    # of a higher-scoring ROI
    selected: list[dict[str, Any]] = []
    for cand in candidates:
        if len(selected) >= top_k:
            break
        cx, cy = cand["center_uv"]
        suppressed = False
        for sel in selected:
            sx, sy = sel["center_uv"]
            dist = math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
            if dist < nms_radius:
                suppressed = True
                break
        if not suppressed:
            selected.append(cand)

    # Re-index
    for idx, roi in enumerate(selected):
        roi["roi_id"] = idx

    return selected


# -------------------------------------------------------------------
# plan_macro_camera_for_roi
# -------------------------------------------------------------------


def plan_macro_camera_for_roi(
    roi: dict[str, Any],
    depth_map: np.ndarray,
    normal_map: np.ndarray,
    camera_intrinsics: dict[str, float],
    camera_extrinsics: np.ndarray,
    macro_fov_deg: float = 35.0,
    macro_resolution: int = 512,
    occlusion_depth_map: np.ndarray | None = None,
) -> CameraPose:
    """Plan a virtual macro camera pose for a detected ROI.

    Unprojects the ROI centre to 3D, computes the macro camera
    distance along the local surface normal, and builds a
    ``CameraPose`` targeting the defect.

    Args:
        roi: ROI dict from :func:`compute_isotopic_residual_heatmap`.
        depth_map: ``[H, W]`` float32 depth.
        normal_map: ``[H, W, 3]`` float32 world-space normals.
        camera_intrinsics: Dict with ``fx``, ``fy``, ``cx``, ``cy``.
        camera_extrinsics: ``[4, 4]`` world-to-camera view matrix.
        macro_fov_deg: Vertical FOV for the macro camera.
        macro_resolution: Output image resolution.
        occlusion_depth_map: Optional depth map for occlusion check.

    Returns:
        :class:`CameraPose` for the macro close-up view.

    Raises:
        QualityGateError: If no unoccluded view can be found.
    """
    from synapse.inspect.align import _look_at_rotation

    cx_px, cy_px = roi["center_uv"]
    h, w = depth_map.shape[:2]

    # Clamp to image bounds
    cx_px = max(0, min(cx_px, w - 1))
    cy_px = max(0, min(cy_px, h - 1))

    z_c = float(depth_map[cy_px, cx_px])
    if z_c <= 0:
        z_c = 1.0  # fallback

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    # Unproject to camera space
    p_cam = _unproject_pixel(
        float(cx_px), float(cy_px), z_c, fx, fy, cx, cy,
    )

    # Transform to world space
    view = np.asarray(camera_extrinsics, dtype=np.float64)
    rot = view[:3, :3]
    t = view[:3, 3]
    rot_inv = rot.T
    p_world = rot_inv @ (p_cam - t)

    # Compute local average normal
    win = 8
    y_lo = max(0, cy_px - win)
    y_hi = min(h, cy_px + win)
    x_lo = max(0, cx_px - win)
    x_hi = min(w, cx_px + win)
    local_normals = normal_map[y_lo:y_hi, x_lo:x_hi].reshape(
        -1, 3,
    )
    n_avg = local_normals.mean(axis=0).astype(np.float64)
    n_len = np.linalg.norm(n_avg)
    if n_len < 1e-8:
        n_avg = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n_avg = n_avg / n_len

    # Estimate 3D defect radius
    r_px = roi.get("bounding_radius_px", 16.0)
    r_3d = float(r_px * z_c / fx)

    # Compute macro camera distance
    half_fov = math.radians(macro_fov_deg / 2.0)
    tan_half = math.tan(half_fov)
    if tan_half < 1e-8:
        tan_half = 1e-8
    d_macro = 1.35 * r_3d / tan_half

    # Position macro camera along normal
    cam_pos = p_world + d_macro * n_avg

    # Build look-at rotation
    rot_mat = _look_at_rotation(
        cam_pos,
        p_world,
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    rot_col_major = rot_mat.T.flatten().tolist()

    return CameraPose(
        position=cam_pos.tolist(),
        rotation=rot_col_major,
        fov_deg=macro_fov_deg,
        resolution=(macro_resolution, macro_resolution),
    )
