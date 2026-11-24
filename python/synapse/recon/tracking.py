"""Dense 4D point tracking and 6-DoF camera trajectory solving.

Implements sparse corner-based tracking with pyramidal Lucas-Kanade optical
flow for real 2D feature correspondence, and Perspective-n-Point (PnP) /
fundamental matrix decomposition for consistent camera extrinsic recovery.

World-to-Camera (w2c) convention
---------------------------------
All ``transform`` fields in :class:`CameraExtrinsics` follow the
**world-to-camera (w2c)** convention:

    p_cam = R @ p_world + t          (3-vector form)
    p_cam = T @ [p_world; 1]         (4x4 homogeneous form)

Where ``T[:3,:3] = R`` (rotation) and ``T[:3,3] = t`` (translation).
The camera position in world space is ``position = -R^T @ t``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synapse.recon.types import CameraExtrinsics, CameraTrajectory

# ---------------------------------------------------------------------------
# Image processing helpers (pure numpy, no OpenCV)
# ---------------------------------------------------------------------------


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) uint8/float frame to float32 grayscale."""
    if frame.dtype == np.uint8:
        f = frame.astype(np.float32) / 255.0
    else:
        f = frame.astype(np.float32)
    gray: np.ndarray = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return gray


def _harris_corners(
    gray: np.ndarray,
    num_points: int,
    block_size: int = 3,
    k: float = 0.04,
) -> np.ndarray:
    """Detect Harris corners and return the top-N as (x, y) coordinates.

    Uses Sobel gradient approximation and windowed structure tensor.
    Returns shape ``(N, 2)`` float32 in pixel coordinates ``(x, y)``.
    """
    h, w = gray.shape

    # Sobel gradients (3x3 kernel via convolution)
    # Pad to handle borders
    padded = np.pad(gray, 1, mode="reflect")
    ix = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    iy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )

    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    # Box filter (sum over block_size x block_size window)
    half = block_size // 2
    padded_ix2 = np.pad(ix2, half, mode="reflect")
    padded_iy2 = np.pad(iy2, half, mode="reflect")
    padded_ixy = np.pad(ixy, half, mode="reflect")

    s_ix2 = np.zeros_like(ix2)
    s_iy2 = np.zeros_like(iy2)
    s_ixy = np.zeros_like(ixy)
    for dy in range(block_size):
        for dx in range(block_size):
            s_ix2 += padded_ix2[dy : dy + h, dx : dx + w]
            s_iy2 += padded_iy2[dy : dy + h, dx : dx + w]
            s_ixy += padded_ixy[dy : dy + h, dx : dx + w]

    # Harris response: det(M) - k * trace(M)^2
    det = s_ix2 * s_iy2 - s_ixy * s_ixy
    trace = s_ix2 + s_iy2
    response = det - k * trace * trace

    # Non-maximum suppression: keep margin and top-N
    margin = max(block_size, 5)
    response[:margin, :] = 0
    response[-margin:, :] = 0
    response[:, :margin] = 0
    response[:, -margin:] = 0

    # Flatten and pick top N
    flat = response.ravel()
    indices = np.argpartition(flat, -min(num_points * 4, len(flat) - 1))[-num_points * 4 :]
    indices = indices[np.argsort(flat[indices])[::-1]]

    # Non-maximum suppression with minimum distance
    min_dist_sq = (min(h, w) * 0.02) ** 2
    selected: list[tuple[float, float]] = []
    for idx in indices:
        y_c, x_c = divmod(int(idx), w)
        too_close = False
        for sx, sy in selected:
            if (x_c - sx) ** 2 + (y_c - sy) ** 2 < min_dist_sq:
                too_close = True
                break
        if not too_close:
            selected.append((float(x_c), float(y_c)))
        if len(selected) >= num_points:
            break

    if len(selected) < num_points:
        # Fill with grid points if not enough corners
        grid_dim = int(math.ceil(math.sqrt(num_points - len(selected))))
        xs = np.linspace(w * 0.1, w * 0.9, grid_dim)
        ys = np.linspace(h * 0.1, h * 0.9, grid_dim)
        for gx in xs:
            for gy in ys:
                if len(selected) >= num_points:
                    break
                selected.append((float(gx), float(gy)))

    pts = np.array(selected[:num_points], dtype=np.float32)
    return pts


def _lucas_kanade_track(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    points: np.ndarray,
    win_size: int = 15,
    num_levels: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Track points from prev to curr using pyramidal Lucas-Kanade.

    Args:
        prev_gray: Previous frame grayscale, shape (H, W).
        curr_gray: Current frame grayscale, shape (H, W).
        points: Points to track, shape (N, 2), in (x, y) format.
        win_size: Window size for local patch matching.
        num_levels: Number of pyramid levels.

    Returns:
        (new_points, status): new_points shape (N, 2), status shape (N,) bool.
    """
    h, w = prev_gray.shape
    n_pts = len(points)
    new_points = np.copy(points)
    status = np.ones(n_pts, dtype=bool)

    # Build Gaussian pyramids with [1, 2, 1]/4 separable anti-aliasing pre-blur
    def _build_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
        pyr = [img]
        for _ in range(levels - 1):
            ph, pw = pyr[-1].shape
            if ph < 8 or pw < 8:
                break
            p = np.pad(pyr[-1], 1, mode="reflect")
            bh = 0.25 * p[1:-1, :-2] + 0.5 * p[1:-1, 1:-1] + 0.25 * p[1:-1, 2:]
            p2 = np.pad(bh, 1, mode="reflect")
            blurred = 0.25 * p2[:-2, 1:-1] + 0.5 * p2[1:-1, 1:-1] + 0.25 * p2[2:, 1:-1]
            nh, nw = ph // 2, pw // 2
            down = (
                blurred[: nh * 2 : 2, : nw * 2 : 2]
                + blurred[1 : nh * 2 : 2, : nw * 2 : 2]
                + blurred[: nh * 2 : 2, 1 : nw * 2 : 2]
                + blurred[1 : nh * 2 : 2, 1 : nw * 2 : 2]
            ) * 0.25
            pyr.append(down)
        return pyr

    pyr_prev = _build_pyramid(prev_gray, num_levels)
    pyr_curr = _build_pyramid(curr_gray, num_levels)
    actual_levels = min(len(pyr_prev), len(pyr_curr))

    # Iterate from coarsest to finest level
    flow = np.zeros((n_pts, 2), dtype=np.float32)

    for level in range(actual_levels - 1, -1, -1):
        scale = 2.0**level
        p_img = pyr_prev[level]
        c_img = pyr_curr[level]
        lh, lw = p_img.shape
        # Clamp window size at coarse pyramid levels so small frames (e.g. 64x64 -> 16x16)
        # still track interior points instead of skipping the coarse level completely.
        lvl_win = min(win_size, max(5, (min(lh, lw) // 2) | 1))
        half_win = lvl_win // 2

        # Compute gradients on previous image
        padded_p = np.pad(p_img, 1, mode="reflect")
        grad_x = 0.5 * (padded_p[1:-1, 2:] - padded_p[1:-1, :-2])
        grad_y = 0.5 * (padded_p[2:, 1:-1] - padded_p[:-2, 1:-1])

        for pi in range(n_pts):
            if not status[pi]:
                continue

            # Scale point to this level
            px = points[pi, 0] / scale
            py = points[pi, 1] / scale
            fx = flow[pi, 0] / scale
            fy = flow[pi, 1] / scale

            # Window bounds in previous image
            x0 = int(round(px)) - half_win
            y0 = int(round(py)) - half_win
            x1 = x0 + lvl_win
            y1 = y0 + lvl_win

            if x0 < 0 or y0 < 0 or x1 > lw or y1 > lh:
                # Window falls outside image at this pyramid level;
                # skip this level for this point (keep it stationary).
                continue

            # Exact center of patch_p so patch_c alignment has zero subpixel bias
            px_c = float(x0 + half_win)
            py_c = float(y0 + half_win)

            # Extract patches
            patch_p = p_img[y0:y1, x0:x1]
            gx_patch = grad_x[y0:y1, x0:x1]
            gy_patch = grad_y[y0:y1, x0:x1]

            # Structure tensor elements
            sxx = np.sum(gx_patch * gx_patch)
            sxy = np.sum(gx_patch * gy_patch)
            syy = np.sum(gy_patch * gy_patch)

            det = sxx * syy - sxy * sxy
            if abs(det) < 1e-6:
                continue

            # Iterative refinement (up to 8 iterations) with sub-pixel bilinear interpolation
            dx, dy = fx, fy
            for _ in range(8):
                cx = px_c + dx
                cy = py_c + dy
                cx0 = int(np.floor(cx)) - half_win
                cy0 = int(np.floor(cy)) - half_win
                alpha = float(cx - np.floor(cx))
                beta = float(cy - np.floor(cy))

                if cx0 < 0 or cy0 < 0 or (cx0 + lvl_win + 1) > lw or (cy0 + lvl_win + 1) > lh:
                    status[pi] = False
                    break

                p00 = c_img[cy0 : cy0 + lvl_win, cx0 : cx0 + lvl_win]
                p10 = c_img[cy0 : cy0 + lvl_win, cx0 + 1 : cx0 + 1 + lvl_win]
                p01 = c_img[cy0 + 1 : cy0 + 1 + lvl_win, cx0 : cx0 + lvl_win]
                p11 = c_img[cy0 + 1 : cy0 + 1 + lvl_win, cx0 + 1 : cx0 + 1 + lvl_win]
                patch_c = (
                    (1.0 - alpha) * (1.0 - beta) * p00
                    + alpha * (1.0 - beta) * p10
                    + (1.0 - alpha) * beta * p01
                    + alpha * beta * p11
                )

                diff = patch_p - patch_c
                bx = np.sum(diff * gx_patch)
                by = np.sum(diff * gy_patch)

                inv_det = 1.0 / det
                ddx = inv_det * (syy * bx - sxy * by)
                ddy = inv_det * (-sxy * bx + sxx * by)
                dx += ddx
                dy += ddy
                if abs(ddx) + abs(ddy) < 1e-3:
                    break

            if status[pi]:
                flow[pi, 0] = dx * scale
                flow[pi, 1] = dy * scale

    new_points = points + flow

    # Clip to image bounds and mark out-of-bounds as failed
    for pi in range(n_pts):
        if not status[pi]:
            continue
        if (
            new_points[pi, 0] < 0
            or new_points[pi, 0] >= w - 1
            or new_points[pi, 1] < 0
            or new_points[pi, 1] >= h - 1
        ):
            status[pi] = False
        else:
            new_points[pi, 0] = np.clip(new_points[pi, 0], 0, w - 1)
            new_points[pi, 1] = np.clip(new_points[pi, 1], 0, h - 1)

    return new_points, status


def solve_point_tracks(
    frames: np.ndarray | list[np.ndarray],
    num_points: int = 256,
    *,
    foundary_client: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track dense surface points across a video sequence.

    When *foundary_client* is provided, dispatches CoTracker inference to the
    foundary service first; otherwise uses Harris corner seeding and pyramidal
    Lucas-Kanade optical flow.

    Args:
        frames: Sequence of video frames, shape (T, H, W, 3) or list of frames.
        num_points: Number of persistent surface points to track.
        foundary_client: Optional :class:`~synapse.bridge.FoundaryClient` instance.

    Returns:
        Tuple of (tracks_2d, visibility):
            tracks_2d: Point coordinates over time, shape (T, num_points, 2).
            visibility: Boolean visibility mask, shape (T, num_points).
    """
    frames_arr = np.asarray(frames)
    if frames_arr.ndim != 4:
        raise ValueError(f"Expected 4D array of video frames, got {frames_arr.shape}")

    num_frames, height, width, _ = frames_arr.shape

    if foundary_client is not None:
        resp = foundary_client.infer(
            "cotracker",
            inputs={"num_frames": num_frames, "height": height, "width": width},
            params={"num_points": num_points},
        )
        outputs = getattr(resp, "outputs", {})
        if isinstance(outputs, dict) and "tracks_2d" in outputs and "visibility" in outputs:
            return (
                np.asarray(outputs["tracks_2d"], dtype=np.float32),
                np.asarray(outputs["visibility"], dtype=bool),
            )

    # Convert first frame to grayscale and detect corners
    gray_0 = _to_gray(frames_arr[0])
    init_pts = _harris_corners(gray_0, num_points)

    actual_points = len(init_pts)
    tracks = np.zeros((num_frames, actual_points, 2), dtype=np.float32)
    visibility = np.ones((num_frames, actual_points), dtype=bool)
    tracks[0] = init_pts

    # Track via Lucas-Kanade across consecutive frames
    prev_gray = gray_0
    active_mask = np.ones(actual_points, dtype=bool)

    for t in range(1, num_frames):
        curr_gray = _to_gray(frames_arr[t])

        # Short-circuit: when frames are identical (zero motion),
        # skip tracking and copy previous positions with full visibility.
        if np.allclose(prev_gray, curr_gray, atol=1e-6):
            tracks[t] = tracks[t - 1].copy()
            visibility[t] = active_mask
            continue

        new_pts, status = _lucas_kanade_track(
            prev_gray,
            curr_gray,
            tracks[t - 1],
            win_size=11,
            num_levels=3,
        )

        # Update active mask — once lost, stay lost
        active_mask = active_mask & status
        tracks[t] = new_pts
        visibility[t] = active_mask

        prev_gray = curr_gray

    return tracks, visibility


def _build_intrinsics_matrix(intrinsics: dict[str, float]) -> np.ndarray:
    """Build 3x3 camera intrinsics matrix K from parameter dict."""
    K = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def _solve_fundamental_matrix_8point(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Estimate fundamental matrix using normalized 8-point algorithm.

    Args:
        pts1: Points in image 1, shape (N, 2).
        pts2: Points in image 2, shape (N, 2).

    Returns:
        3x3 fundamental matrix F such that pts2^T @ F @ pts1 = 0.
    """
    n = len(pts1)

    # Normalize points for numerical stability
    def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        avg_dist = float(np.mean(np.sqrt(np.sum(centered**2, axis=1))))
        if avg_dist < 1e-10:
            avg_dist = 1.0
        s = math.sqrt(2.0) / avg_dist
        T = np.array(
            [
                [s, 0, -s * mean[0]],
                [0, s, -s * mean[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        norm_pts = (T @ pts_h.T).T[:, :2]
        return norm_pts, T

    pts1_n, T1 = _normalize(pts1)
    pts2_n, T2 = _normalize(pts2)

    # Build the constraint matrix A (N x 9)
    A = np.zeros((n, 9), dtype=np.float64)
    for i in range(n):
        x1, y1 = pts1_n[i]
        x2, y2 = pts2_n[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0]

    # SVD to find null space
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt2

    # Denormalize
    F = T2.T @ F @ T1
    f_norm: np.ndarray = F / (np.linalg.norm(F) + 1e-12)

    return f_norm


def _solve_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    max_iters: int = 100,
    threshold: float = 2.0,
) -> np.ndarray:
    """Estimate fundamental matrix using RANSAC + normalized 8-point algorithm with inlier refit."""
    n = len(pts1)
    if n < 8:
        return np.eye(3, dtype=np.float64)

    if n == 8:
        return _solve_fundamental_matrix_8point(pts1, pts2)

    best_inliers = 0
    best_F = None
    best_mask: np.ndarray | None = None
    rng = np.random.default_rng(42)

    pts1_h = np.column_stack([pts1, np.ones(n, dtype=np.float64)])
    pts2_h = np.column_stack([pts2, np.ones(n, dtype=np.float64)])

    for _ in range(max_iters):
        sample_idx = rng.choice(n, size=8, replace=False)
        F_cand = _solve_fundamental_matrix_8point(pts1[sample_idx], pts2[sample_idx])

        # Sampson epipolar distance
        Fx1 = (F_cand @ pts1_h.T).T
        Ftx2 = (F_cand.T @ pts2_h.T).T
        num = np.sum(pts2_h * Fx1, axis=1) ** 2
        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
        sampson_dist = np.sqrt(num / np.maximum(denom, 1e-12))

        mask = sampson_dist < threshold
        inliers = int(np.sum(mask))
        if inliers > best_inliers:
            best_inliers = inliers
            best_F = F_cand
            best_mask = mask

    if best_F is None or best_inliers < 8:
        return _solve_fundamental_matrix_8point(pts1, pts2)
    if best_mask is not None and best_inliers >= 8:
        return _solve_fundamental_matrix_8point(pts1[best_mask], pts2[best_mask])
    return best_F


def _essential_to_rotation_translation(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose essential matrix and choose the correct R, t via cheirality check.

    Returns:
        (R, t) where R is 3x3 and t is (3,).
    """
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation (det = +1)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # Four possible solutions
    candidates = [
        (U @ W @ Vt, U[:, 2]),
        (U @ W @ Vt, -U[:, 2]),
        (U @ W.T @ Vt, U[:, 2]),
        (U @ W.T @ Vt, -U[:, 2]),
    ]

    best_R, best_t = candidates[0]
    best_count = -1

    for R_cand, t_cand in candidates:
        if np.linalg.det(R_cand) < 0:
            R_cand = -R_cand
            t_cand = -t_cand

        # Cheirality check: count points in front of both cameras
        count = 0
        P1 = K @ np.eye(3, 4, dtype=np.float64)
        T2 = np.hstack([R_cand, t_cand.reshape(3, 1)])
        P2 = K @ T2

        for i in range(min(len(pts1), 20)):
            # Triangulate point
            x1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
            x2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])

            A_tri = np.zeros((4, 4), dtype=np.float64)
            A_tri[0] = x1[0] * P1[2] - P1[0]
            A_tri[1] = x1[1] * P1[2] - P1[1]
            A_tri[2] = x2[0] * P2[2] - P2[0]
            A_tri[3] = x2[1] * P2[2] - P2[1]

            _, _, Vt_tri = np.linalg.svd(A_tri)
            pt4 = Vt_tri[-1]
            if abs(pt4[3]) < 1e-10:
                continue
            pt3d = pt4[:3] / pt4[3]

            # Check depth in camera 1
            if pt3d[2] > 0:
                # Check depth in camera 2
                pt_cam2 = R_cand @ pt3d + t_cand
                if pt_cam2[2] > 0:
                    count += 1

        if count > best_count:
            best_count = count
            best_R = R_cand
            best_t = t_cand

    return best_R, best_t


def _fit_homography_dlt(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray | None:
    """Estimate planar homography H (pts2 ~ H @ pts1) via normalized DLT."""
    n = len(pts1)
    if n < 4:
        return None

    def _norm(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        avg_d = float(np.mean(np.linalg.norm(centered, axis=1)))
        s = math.sqrt(2.0) / max(avg_d, 1e-10)
        T = np.array([[s, 0.0, -s * mean[0]], [0.0, s, -s * mean[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
        ph = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
        return (T @ ph.T).T[:, :2], T

    p1n, T1 = _norm(pts1)
    p2n, T2 = _norm(pts2)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x1, y1 = p1n[i]
        x2, y2 = p2n[i]
        A[2 * i] = [0.0, 0.0, 0.0, -x1, -y1, -1.0, y2 * x1, y2 * y1, y2]
        A[2 * i + 1] = [x1, y1, 1.0, 0.0, 0.0, 0.0, -x2 * x1, -x2 * y1, -x2]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    try:
        H = np.linalg.inv(T2) @ Hn @ T1
    except np.linalg.LinAlgError:
        return None
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return np.asarray(H, dtype=np.float64)


def _is_pure_rotation_parallax_gric(
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    R_rel: np.ndarray,
    K: np.ndarray,
) -> bool:
    """Resolution-invariant pure-rotation test using de-rotated ray parallax + H/F GRIC.

    Replaces the resolution-broken ``e_norm < 1e-3 * k_norm**2`` heuristic (which
    falsely zeroed translation for any video wider than ~435 px).
    """
    n = len(pts1)
    if n < 8:
        return True

    # Near-zero pixel motion -> static camera (no translation)
    med_disp = float(np.median(np.linalg.norm(pts2 - pts1, axis=1)))
    if med_disp < 1e-3:
        return True

    # 1. De-rotated bearing-ray parallax angle (invariant to focal length / resolution)
    K_inv = np.linalg.inv(K)
    p1_h = np.column_stack([pts1, np.ones(n, dtype=np.float64)])
    p2_h = np.column_stack([pts2, np.ones(n, dtype=np.float64)])
    r1 = (K_inv @ p1_h.T).T
    r2 = (K_inv @ p2_h.T).T
    r1 /= np.maximum(np.linalg.norm(r1, axis=1, keepdims=True), 1e-12)
    r2 /= np.maximum(np.linalg.norm(r2, axis=1, keepdims=True), 1e-12)
    r1_rot = (R_rel @ r1.T).T
    cos_ang = np.clip(np.sum(r1_rot * r2, axis=1), -1.0, 1.0)
    median_parallax_deg = float(np.median(np.degrees(np.arccos(cos_ang))))

    if median_parallax_deg >= 0.2:
        return False

    # 2. H vs F GRIC model selection (Torr 1998)
    H = _fit_homography_dlt(pts1, pts2)
    if H is None:
        return median_parallax_deg < 0.05

    proj2 = (H @ p1_h.T).T
    z2 = np.where(np.abs(proj2[:, 2]) > 1e-10, proj2[:, 2], 1e-10)
    err_h_sq = np.sum((proj2[:, :2] / z2[:, None] - pts2) ** 2, axis=1)

    Fx1 = (F @ p1_h.T).T
    Ftx2 = (F.T @ p2_h.T).T
    denom = np.maximum(Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2, 1e-12)
    err_f_sq = (np.sum(p2_h * Fx1, axis=1) ** 2) / denom

    sigma2 = 1.0
    lambda1 = math.log(4.0)
    lambda2 = math.log(max(4 * n, 8))
    lambda3 = 2.0
    # Homography: d=2, k=8; Fundamental: d=3, k=7
    gric_h = float(np.sum(np.minimum(err_h_sq / sigma2, lambda3 * (4 - 2)))) + lambda1 * 2 * n + lambda2 * 8
    gric_f = float(np.sum(np.minimum(err_f_sq / sigma2, lambda3 * (4 - 3)))) + lambda1 * 3 * n + lambda2 * 7
    med_err_h = float(np.sqrt(np.median(err_h_sq)))
    return bool(gric_h < gric_f and med_err_h < 1.5)


def solve_camera_trajectory(
    tracks_2d: np.ndarray,
    tracks_3d: np.ndarray | None = None,
    intrinsics: dict[str, float] | None = None,
    *,
    frame_height: int | None = None,
    frame_width: int | None = None,
    visibility: np.ndarray | None = None,
    foundary_client: Any = None,
) -> CameraTrajectory:
    """Solve 6-DoF metric camera trajectory from tracked points.

    Recovers camera rotation and translation per frame using the fundamental
    matrix decomposition between frame pairs. All output transforms follow
    the **world-to-camera (w2c)** convention (see module docstring).

    Convention: Internally all camera-space computations use the OpenCV
    convention (+Z forward, +X right, +Y down). The ``transform`` matrices
    in the output ``CameraExtrinsics`` are in this same convention. If your
    downstream consumer uses OpenGL convention (-Z forward), negate the
    Z-row of the rotation matrix.

    Args:
        tracks_2d: 2D feature trajectories, shape (T, N, 2).
        tracks_3d: Optional 3D point cloud or depth back-projection, shape (N, 3).
        intrinsics: Camera intrinsics dict (fx, fy, cx, cy). When *None*,
            derived from *frame_width* / *frame_height* if provided, else
            falls back to 1050/1050/960/540.
        frame_height: Optional frame height in pixels.
        frame_width: Optional frame width in pixels.
        visibility: Optional boolean visibility mask, shape (T, N). When
            provided, only visible points are used for each frame pair.

    Returns:
        CameraTrajectory object with per-frame CameraExtrinsics.
        All transforms use the w2c convention (OpenCV: +Z forward).
    """
    num_frames = int(tracks_2d.shape[0])

    if intrinsics is None:
        if frame_width is not None and frame_height is not None:
            # Derive a reasonable pinhole model: fx=fy=width, cx=w/2, cy=h/2.
            intrinsics = {
                "fx": float(frame_width),
                "fy": float(frame_width),
                "cx": frame_width / 2.0,
                "cy": frame_height / 2.0,
            }
        else:
            intrinsics = {"fx": 1050.0, "fy": 1050.0, "cx": 960.0, "cy": 540.0}

    K = _build_intrinsics_matrix(intrinsics)

    extrinsics_list: list[CameraExtrinsics] = []

    # Frame 0: identity (world frame = camera 0 frame)
    R_accum = np.eye(3, dtype=np.float64)
    t_accum = np.zeros(3, dtype=np.float64)

    mat4_0 = np.eye(4, dtype=np.float64)
    cam_pos_0 = np.zeros(3, dtype=np.float64)
    extrinsics_list.append(
        CameraExtrinsics(
            frame=0,
            transform=mat4_0,
            position=cam_pos_0,
            rotation_matrix=np.eye(3, dtype=np.float64),
            yaw_rad=0.0,
        )
    )

    # Track incremental relative pose between consecutive frames
    # Previous 3D triangulated points keyed by global feature track ID
    prev_3d_map: dict[int, np.ndarray] = {}

    for t in range(1, num_frames):
        pts_prev = tracks_2d[t - 1].astype(np.float64)
        pts_curr = tracks_2d[t].astype(np.float64)

        # Filter out NaN/Inf points and use visibility mask if provided
        valid = np.isfinite(pts_prev).all(axis=1) & np.isfinite(pts_curr).all(axis=1)
        if visibility is not None and visibility.shape[0] > t:
            valid = valid & visibility[t - 1] & visibility[t]
        valid_ids = np.where(valid)[0]
        pts_prev_v = pts_prev[valid_ids]
        pts_curr_v = pts_curr[valid_ids]

        if len(pts_prev_v) >= 8:
            # Fast-path static/zero-motion frame pairs to avoid degenerate 8-point SVDs
            med_motion = float(np.median(np.linalg.norm(pts_curr_v - pts_prev_v, axis=1)))
            if med_motion < 1e-4:
                R_rel = np.eye(3, dtype=np.float64)
                t_rel = np.zeros(3, dtype=np.float64)
            else:
                # Estimate fundamental matrix
                F = _solve_fundamental_matrix(pts_prev_v, pts_curr_v)

                # Essential matrix: E = K^T @ F @ K
                E = K.T @ F @ K

                # Decompose essential matrix
                R_rel, t_rel = _essential_to_rotation_translation(
                    E,
                    pts_prev_v,
                    pts_curr_v,
                    K,
                )

                # Resolution-invariant parallax + H/F GRIC pure-rotation guard
                is_pure_rotation = _is_pure_rotation_parallax_gric(
                    pts_prev_v, pts_curr_v, F, R_rel, K
                )

                if is_pure_rotation:
                    # Suppress translation for pure rotation to prevent drift
                    t_rel = np.zeros(3, dtype=np.float64)

                # Inter-step scale estimation matching shared global landmark track IDs
                if not is_pure_rotation:
                    P1_rel = K @ np.eye(3, 4, dtype=np.float64)
                    T_rel_34 = np.hstack([R_rel, t_rel.reshape(3, 1)])
                    P2_rel = K @ T_rel_34
                    curr_3d_map: dict[int, np.ndarray] = {}
                    for local_idx in range(min(len(pts_prev_v), 30)):
                        track_id = int(valid_ids[local_idx])
                        x1 = np.array([pts_prev_v[local_idx, 0], pts_prev_v[local_idx, 1], 1.0])
                        x2 = np.array([pts_curr_v[local_idx, 0], pts_curr_v[local_idx, 1], 1.0])
                        A_tri = np.zeros((4, 4), dtype=np.float64)
                        A_tri[0] = x1[0] * P1_rel[2] - P1_rel[0]
                        A_tri[1] = x1[1] * P1_rel[2] - P1_rel[1]
                        A_tri[2] = x2[0] * P2_rel[2] - P2_rel[0]
                        A_tri[3] = x2[1] * P2_rel[2] - P2_rel[1]
                        _, _, Vt_tri = np.linalg.svd(A_tri)
                        pt4 = Vt_tri[-1]
                        if abs(pt4[3]) > 1e-8:
                            p3 = pt4[:3] / pt4[3]
                            if p3[2] > 0.01:
                                curr_3d_map[track_id] = p3

                    shared_ids = sorted(set(prev_3d_map.keys()) & set(curr_3d_map.keys()))
                    if len(shared_ids) >= 4:
                        prev_arr = np.array([prev_3d_map[k] for k in shared_ids], dtype=np.float64)
                        curr_arr = np.array([curr_3d_map[k] for k in shared_ids], dtype=np.float64)
                        d_prev = np.linalg.norm(prev_arr[1:] - prev_arr[:-1], axis=1)
                        d_curr = np.linalg.norm(curr_arr[1:] - curr_arr[:-1], axis=1)
                        good = (d_prev > 1e-6) & (d_curr > 1e-6)
                        if np.any(good):
                            # Ratio prev/curr rescales unit-norm t_rel to match previous metric scale
                            median_scale = float(np.median(d_prev[good] / d_curr[good]))
                            if 0.2 < median_scale < 5.0:
                                t_rel *= median_scale
                                curr_3d_map = {k: v * median_scale for k, v in curr_3d_map.items()}
                    if curr_3d_map:
                        # Transform landmarks into current frame coordinates for step t+1
                        prev_3d_map = {
                            k: R_rel @ p + t_rel for k, p in curr_3d_map.items()
                        }
        else:
            # Not enough points — assume identity relative pose
            R_rel = np.eye(3, dtype=np.float64)
            t_rel = np.zeros(3, dtype=np.float64)

        # Accumulate w2c transform: T_wc(t) = T_rel * T_wc(t-1)
        R_accum = R_rel @ R_accum
        t_accum = R_rel @ t_accum + t_rel

        # Build 4x4 w2c matrix
        mat4 = np.eye(4, dtype=np.float64)
        mat4[:3, :3] = R_accum
        mat4[:3, 3] = t_accum

        # Camera position in world space: -R^T @ t
        cam_pos = -R_accum.T @ t_accum

        # Compute yaw from camera Z-axis direction in world
        cam_z_world = R_accum.T @ np.array([0.0, 0.0, 1.0])
        yaw = math.atan2(cam_z_world[0], cam_z_world[2])

        extrinsics_list.append(
            CameraExtrinsics(
                frame=t,
                transform=mat4,
                position=cam_pos,
                rotation_matrix=R_accum.copy(),
                yaw_rad=yaw,
            )
        )

    # Compute real 3D reprojection error in pixels
    reproj_errors: list[float] = []
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])

    for t in range(1, num_frames):
        pts_prev = tracks_2d[t - 1].astype(np.float64)
        pts_curr = tracks_2d[t].astype(np.float64)
        valid = np.isfinite(pts_prev).all(axis=1) & np.isfinite(pts_curr).all(axis=1)
        if visibility is not None and visibility.shape[0] > t:
            valid = valid & visibility[t - 1] & visibility[t]
        if np.sum(valid) >= 8:
            # Use t-1 frame extrinsics as P1 (not identity)
            T1 = extrinsics_list[t - 1].transform[:3]
            P1 = K @ T1
            T2 = extrinsics_list[t].transform[:3]
            P2 = K @ T2
            v_indices = np.where(valid)[0]
            errs: list[float] = []
            for idx in v_indices[:30]:
                x1 = np.array([pts_prev[idx, 0], pts_prev[idx, 1], 1.0])
                x2 = np.array([pts_curr[idx, 0], pts_curr[idx, 1], 1.0])
                A_tri = np.zeros((4, 4), dtype=np.float64)
                A_tri[0] = x1[0] * P1[2] - P1[0]
                A_tri[1] = x1[1] * P1[2] - P1[1]
                A_tri[2] = x2[0] * P2[2] - P2[0]
                A_tri[3] = x2[1] * P2[2] - P2[1]
                _, _, Vt_tri = np.linalg.svd(A_tri)
                pt4 = Vt_tri[-1]
                if abs(pt4[3]) > 1e-8:
                    p3 = pt4[:3] / pt4[3]
                    p_cam2 = T2[:3, :3] @ p3 + T2[:3, 3]
                    if p_cam2[2] > 0.01:
                        u_proj = fx * p_cam2[0] / p_cam2[2] + cx
                        v_proj = fy * p_cam2[1] / p_cam2[2] + cy
                        err = float(
                            math.hypot(u_proj - pts_curr[idx, 0], v_proj - pts_curr[idx, 1])
                        )
                        if err < 50.0:  # inlier error bounds
                            errs.append(err)
            if errs:
                reproj_errors.append(float(np.mean(errs)))
            else:
                disp = pts_curr[valid] - pts_prev[valid]
                reproj_errors.append(float(np.mean(np.linalg.norm(disp, axis=1))))
        elif np.any(valid):
            disp = pts_curr[valid] - pts_prev[valid]
            reproj_errors.append(float(np.mean(np.linalg.norm(disp, axis=1))))

    avg_reproj = float(np.mean(reproj_errors)) if reproj_errors else 0.0

    if foundary_client is not None:
        resp = foundary_client.infer(
            "megasam",
            inputs={"num_frames": num_frames},
            params={"intrinsics": intrinsics},
        )
        outputs = getattr(resp, "outputs", {})
        if isinstance(outputs, dict) and "metric_scale" in outputs:
            ext_scale = float(outputs["metric_scale"])
            if ext_scale > 0.0:
                for ext in extrinsics_list:
                    ext.transform[:3, 3] *= ext_scale
                    ext.position = ext.position * ext_scale
                return CameraTrajectory(
                    frame_count=num_frames,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics_list,
                    metric_scale=ext_scale,
                    ground_plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    ground_height_m=0.0,
                    reprojection_error_px=avg_reproj,
                )

    return CameraTrajectory(
        frame_count=num_frames,
        intrinsics=intrinsics,
        extrinsics=extrinsics_list,
        metric_scale=1.0,
        ground_plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ground_height_m=0.0,
        reprojection_error_px=avg_reproj,
    )


def anchor_metric_scale(
    depths: np.ndarray,
    camera_trajectory: CameraTrajectory,
    *,
    metric_depths: np.ndarray | None = None,
    known_height_m: float | None = None,
) -> tuple[CameraTrajectory, float]:
    """Anchor affine relative trajectory translation scale to physical metres.

    Computes a robust median scale ratio against reference *metric_depths*
    (or *known_height_m* height span) and rescales all world-to-camera
    translations and camera positions consistently.
    """
    d_rel = np.asarray(depths, dtype=np.float64)
    scale = 1.0

    if metric_depths is not None:
        d_met = np.asarray(metric_depths, dtype=np.float64)
        valid = np.isfinite(d_rel) & np.isfinite(d_met) & (d_rel > 1e-6) & (d_met > 1e-6)
        if np.any(valid):
            scale = float(np.median(d_met[valid] / d_rel[valid]))
    elif known_height_m is not None and known_height_m > 0.0 and camera_trajectory.extrinsics:
        ys = np.array([float(e.position[1]) for e in camera_trajectory.extrinsics], dtype=np.float64)
        span = float(np.max(ys) - np.min(ys))
        if span > 1e-6:
            scale = float(known_height_m) / span

    scale = float(np.clip(scale, 1e-4, 1e4))
    scaled_exts: list[CameraExtrinsics] = []
    for ext in camera_trajectory.extrinsics:
        t_mat = ext.transform.copy()
        t_mat[:3, 3] = t_mat[:3, 3] * scale
        pos = ext.position.copy() * scale
        scaled_exts.append(
            CameraExtrinsics(
                frame=ext.frame,
                transform=t_mat,
                position=pos,
                rotation_matrix=ext.rotation_matrix.copy(),
                yaw_rad=ext.yaw_rad,
            )
        )

    scaled_traj = CameraTrajectory(
        frame_count=camera_trajectory.frame_count,
        intrinsics=dict(camera_trajectory.intrinsics),
        extrinsics=scaled_exts,
        metric_scale=float(camera_trajectory.metric_scale) * scale,
        ground_plane_normal=camera_trajectory.ground_plane_normal.copy(),
        ground_height_m=float(camera_trajectory.ground_height_m) * scale,
        reprojection_error_px=camera_trajectory.reprojection_error_px,
    )
    return scaled_traj, scale
