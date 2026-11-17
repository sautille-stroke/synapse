"""Automated video artifact detection from FGPP-aligned video frames.

Detects three classes of visual defects by comparing generated video
frames against FGPP ground-truth signals:

1. **Temporal flicker** — motion-compensated luminance variance spikes.
2. **Silhouette drift** — video edges diverging from Cryptomatte boundaries.
3. **Ground violation** — entities sinking below the world-space ground plane.

All detection kernels are pure NumPy (vectorised, no pixel loops).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from synapse.fgpp.frame import FGPPFrame
from synapse.render_core.morphology import exact_euclidean_distance_transform

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class VideoDefectRegion:
    """A detected visual defect in a video frame range."""

    defect_type: Literal[
        "temporal_flicker", "silhouette_drift", "ground_violation"
    ]
    frame_range: tuple[int, int]  # [start, end] inclusive
    bbox_xyxy: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    entity_id: int  # Cryptomatte ID (0 = global)
    severity_score: float  # [0.0, 1.0]
    details: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "flicker": 0.15,
    "silhouette": 8.0,
    "ground_y": -0.01,
}

# Block size for local variance computation (temporal flicker).
_BLOCK_SIZE: int = 16


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB image [H, W, 3] or [H, W] to float32 luminance in [0, 1]."""
    arr = np.asarray(rgb)
    if np.issubdtype(arr.dtype, np.uint8):
        scale = 255.0
    elif np.issubdtype(arr.dtype, np.uint16):
        scale = 65535.0
    else:
        scale = 1.0
    if arr.ndim == 2:
        return np.asarray(np.clip(arr.astype(np.float32) / scale, 0.0, 1.0), dtype=np.float32)
    gray = (
        0.2126 * arr[..., 0].astype(np.float32)
        + 0.7152 * arr[..., 1].astype(np.float32)
        + 0.0722 * arr[..., 2].astype(np.float32)
    ) / scale
    return np.asarray(np.clip(gray, 0.0, 1.0), dtype=np.float32)


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel edge magnitude on a float32 [H, W] image.

    Returns a float32 [H, W] edge magnitude map.
    """
    # Horizontal kernel  [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # Vertical kernel    [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
    # Use slicing-based convolution to stay fully vectorised.
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2] + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    return np.asarray(np.sqrt(gx * gx + gy * gy), dtype=np.float32)


def _block_mean(img: np.ndarray, block: int) -> np.ndarray:
    """Compute per-block mean of a float32 [H, W] image, handling remainder boundary blocks.

    Returns a float32 array holding the mean value of each non-overlapping block.
    """
    h, w = img.shape
    if block <= 0:
        return np.asarray(img.copy(), dtype=np.float32)
    pad_h = (block - (h % block)) % block
    pad_w = (block - (w % block)) % block
    if pad_h == 0 and pad_w == 0:
        bh = h // block
        bw = w // block
        return np.asarray(img.reshape(bh, block, bw, block).mean(axis=(1, 3)), dtype=np.float32)

    pad_img = np.pad(img.astype(np.float64), ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
    valid_mask = np.pad(
        np.ones((h, w), dtype=np.float64), ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0
    )

    bh, bw = pad_img.shape
    blocks = pad_img.reshape(bh // block, block, bw // block, block)
    mask_blocks = valid_mask.reshape(bh // block, block, bw // block, block)

    sums = blocks.sum(axis=(1, 3))
    counts = mask_blocks.sum(axis=(1, 3))
    return np.asarray(np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0), dtype=np.float32)


def _simple_warp(
    frame: np.ndarray, mv: np.ndarray
) -> np.ndarray:
    """Warp *frame* by *mv* using backward nearest-neighbour sampling: dst(x, y) = src(x - mv_x, y - mv_y).

    Args:
        frame: float32 [H, W] luminance image.
        mv: float16/32 [H, W, 2] screen-space motion vectors (dx, dy).

    Returns:
        Warped float32 [H, W] image.
    """
    h, w = frame.shape
    yy, xx = np.mgrid[:h, :w]
    mv_f = np.nan_to_num(mv.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    src_x = np.clip(np.round(xx - mv_f[..., 0]).astype(np.intp), 0, w - 1)
    src_y = np.clip(np.round(yy - mv_f[..., 1]).astype(np.intp), 0, h - 1)
    return np.asarray(frame[src_y, src_x])


# ---------------------------------------------------------------------------
# Detection kernels
# ---------------------------------------------------------------------------


def _detect_temporal_flicker(
    video_gray: np.ndarray,
    fgpp_frames: list[FGPPFrame],
    threshold: float,
) -> list[VideoDefectRegion]:
    """Detect temporal flicker between consecutive frames."""
    defects: list[VideoDefectRegion] = []
    t_count = video_gray.shape[0]
    if t_count < 2:
        return defects

    for t in range(t_count - 1):
        frame_a = video_gray[t]
        frame_b = video_gray[t + 1]

        # Motion-compensate frame_a → frame_b using FGPP motion vectors.
        mv_next = fgpp_frames[t + 1].motion_vector if t + 1 < len(fgpp_frames) else None
        if mv_next is not None:
            warped = _simple_warp(frame_a, mv_next)
        else:
            warped = frame_a

        diff = np.abs(frame_b - warped)
        bmean = _block_mean(diff, _BLOCK_SIZE)

        if bmean.size == 0:
            continue

        hot_mask = bmean > threshold
        if not np.any(hot_mask):
            continue

        # Severity: proportion of blocks that are hot, weighted by max mean.
        max_mean = float(np.max(bmean))
        hot_ratio = float(hot_mask.sum()) / max(bmean.size, 1)
        severity = float(np.clip(hot_ratio * max_mean / threshold, 0.0, 1.0))

        # Bounding box of hot blocks in pixel space, clamped to frame.
        h, w = frame_b.shape
        ys, xs = np.nonzero(hot_mask)
        x1 = min(int(np.min(xs)) * _BLOCK_SIZE, w)
        y1 = min(int(np.min(ys)) * _BLOCK_SIZE, h)
        x2 = min((int(np.max(xs)) + 1) * _BLOCK_SIZE, w)
        y2 = min((int(np.max(ys)) + 1) * _BLOCK_SIZE, h)

        defects.append(VideoDefectRegion(
            defect_type="temporal_flicker",
            frame_range=(t, t + 1),
            bbox_xyxy=(x1, y1, x2, y2),
            entity_id=0,
            severity_score=severity,
            details={"max_block_mean": max_mean, "hot_ratio": hot_ratio},
        ))

    return defects


def _detect_silhouette_drift(
    video_gray: np.ndarray,
    fgpp_frames: list[FGPPFrame],
    threshold: float,
) -> list[VideoDefectRegion]:
    """Detect silhouette drift between video edges and FGPP boundaries."""
    defects: list[VideoDefectRegion] = []
    t_count = video_gray.shape[0]

    for t in range(t_count):
        if t >= len(fgpp_frames):
            break
        fgpp = fgpp_frames[t]
        if fgpp.cryptomatte_id is None:
            continue

        # Video luminance edges
        video_edge_mag = _sobel_edges(video_gray[t])
        video_edge_mask = video_edge_mag > 0.05

        # FGPP cryptomatte boundary edges
        cid = fgpp.cryptomatte_id
        h, w = cid.shape
        boundary = np.zeros((h, w), dtype=bool)
        boundary[:, :-1] |= cid[:, :-1] != cid[:, 1:]
        boundary[:-1, :] |= cid[:-1, :] != cid[1:, :]

        if not np.any(boundary):
            continue

        # Distance from each pixel to the nearest FGPP boundary edge.
        dist_from_boundary = exact_euclidean_distance_transform(boundary)

        # Where video edges are far from FGPP edges → drift.
        drift_mask = video_edge_mask & (dist_from_boundary > threshold)
        if not np.any(drift_mask):
            continue

        ys, xs = np.nonzero(drift_mask)
        mean_dist = float(dist_from_boundary[drift_mask].mean())
        severity = float(np.clip(mean_dist / (threshold * 5.0), 0.0, 1.0))

        defects.append(VideoDefectRegion(
            defect_type="silhouette_drift",
            frame_range=(t, t),
            bbox_xyxy=(
                int(np.min(xs)), int(np.min(ys)),
                int(np.max(xs)) + 1, int(np.max(ys)) + 1,
            ),
            entity_id=0,
            severity_score=severity,
            details={"mean_edge_distance": mean_dist},
        ))

    return defects


def _detect_ground_violation(
    fgpp_frames: list[FGPPFrame],
    ground_y: float,
) -> list[VideoDefectRegion]:
    """Detect entities whose world-space Y drops below ground.

    Uses OpenGL camera convention:
    - Camera looks along -Z (forward depth z > 0 corresponds to z_cam = -z)
    - +Y is camera-space up (image top y_px=0 corresponds to +Y_cam)
    - +X is camera-space right
    Pixel centers (xx + 0.5, yy + 0.5) are unprojected to camera space,
    then transformed to world coordinates via cam.world_matrix.
    """
    defects: list[VideoDefectRegion] = []

    # Collect per-entity, per-frame violations: (t, min_y, x1, y1, x2, y2)
    entity_violations: dict[int, list[tuple[int, float, int, int, int, int]]] = {}

    for t, fgpp in enumerate(fgpp_frames):
        if fgpp.cryptomatte_id is None:
            continue

        cid = fgpp.cryptomatte_id
        depth = fgpp.depth
        cam = fgpp.camera
        h, w = depth.shape

        # Unproject to world space to get Y coordinate.
        yy, xx = np.mgrid[:h, :w]
        z = depth.astype(np.float64)

        # OpenGL camera convention:
        # Camera looks down -Z: z_cam = -z
        # Top of image (yy = 0) is +Y_cam: y_cam = -((yy + 0.5) - cy) * z / fy
        # Right of image (xx = W-1) is +X_cam: x_cam = ((xx + 0.5) - cx) * z / fx
        # Derive focal length from fov_degrees if intrinsics are missing.
        if cam.fx > 0:
            fx = float(cam.fx)
        elif hasattr(cam, 'fov_degrees') and cam.fov_degrees > 0:
            fx = float(w) / (2.0 * math.tan(math.radians(cam.fov_degrees / 2.0)))
        else:
            fx = max(float(cam.fx), 1e-6)
        if cam.fy > 0:
            fy = float(cam.fy)
        elif hasattr(cam, 'fov_degrees') and cam.fov_degrees > 0:
            fy = float(h) / (2.0 * math.tan(math.radians(cam.fov_degrees / 2.0)))
        else:
            fy = max(float(cam.fy), 1e-6)
        x_cam = ((xx.astype(np.float64) + 0.5) - cam.cx) * z / fx
        y_cam = -((yy.astype(np.float64) + 0.5) - cam.cy) * z / fy
        z_cam = -z

        # Build [N, 4] homogeneous coordinates
        ones = np.ones_like(z)
        pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=-1)  # [H,W,4]
        pts_flat = pts_cam.reshape(-1, 4)

        # Camera-to-world: world_matrix is the inverse of view_matrix
        world_mat = cam.world_matrix.astype(np.float64)
        pts_world = (world_mat @ pts_flat.T).T  # [N, 4]

        # World Y coordinate (+Y = up direction)
        world_y = pts_world[:, 1].reshape(h, w)

        # Check per entity
        unique_ids = np.unique(cid)
        for eid in unique_ids:
            if eid == 0:
                continue
            mask = (cid == eid) & (world_y < ground_y) & np.isfinite(world_y)
            if not np.any(mask):
                continue
            ys, xs = np.nonzero(mask)
            min_y = float(world_y[mask].min())
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max()) + 1
            y2 = int(ys.max()) + 1
            entity_violations.setdefault(int(eid), []).append(
                (t, min_y, x1, y1, x2, y2)
            )

    # Emit defects for entities violating ground across >= 2 frames.
    for eid, violations in entity_violations.items():
        if len(violations) < 2:
            continue
        frames = [v[0] for v in violations]
        min_ys = [v[1] for v in violations]
        x1_all = [v[2] for v in violations]
        y1_all = [v[3] for v in violations]
        x2_all = [v[4] for v in violations]
        y2_all = [v[5] for v in violations]

        worst_y = min(min_ys)
        severity = float(np.clip(
            abs(worst_y - ground_y) / 1.0, 0.0, 1.0
        ))
        bbox = (min(x1_all), min(y1_all), max(x2_all), max(y2_all))
        defects.append(VideoDefectRegion(
            defect_type="ground_violation",
            frame_range=(min(frames), max(frames)),
            bbox_xyxy=bbox,
            entity_id=eid,
            severity_score=severity,
            details={"min_world_y": worst_y},
        ))

    return defects


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_video_artifacts(
    video_srgb: np.ndarray,
    fgpp_frames: list[FGPPFrame],
    thresholds: dict[str, float] | None = None,
) -> list[VideoDefectRegion]:
    """Detect visual artifacts in generated video using FGPP signals.

    Args:
        video_srgb: Video tensor of shape ``[T, H, W, 3]`` with
            ``uint8`` sRGB pixels.
        fgpp_frames: List of :class:`FGPPFrame` instances aligned
            1-to-1 with the video frames.
        thresholds: Optional overrides for detection thresholds.
            Supported keys: ``"flicker"``, ``"silhouette"``,
            ``"ground_y"``.

    Returns:
        List of :class:`VideoDefectRegion` sorted by
        ``severity_score`` in descending order.

    Raises:
        ValueError: If *video_srgb* does not have 4 dimensions or
            the last dimension is not 3.
    """
    if video_srgb.ndim != 4 or video_srgb.shape[-1] != 3:
        raise ValueError(
            f"Expected video_srgb shape [T, H, W, 3], "
            f"got {video_srgb.shape}"
        )

    th = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    # Pre-compute grayscale for the entire video.
    t_count = video_srgb.shape[0]
    video_gray = np.empty(
        (t_count, video_srgb.shape[1], video_srgb.shape[2]),
        dtype=np.float32,
    )
    for t in range(t_count):
        video_gray[t] = _to_grayscale(video_srgb[t])

    all_defects: list[VideoDefectRegion] = []
    all_defects.extend(
        _detect_temporal_flicker(video_gray, fgpp_frames, th["flicker"])
    )
    all_defects.extend(
        _detect_silhouette_drift(video_gray, fgpp_frames, th["silhouette"])
    )
    all_defects.extend(
        _detect_ground_violation(fgpp_frames, th["ground_y"])
    )

    # Sort by severity descending.
    all_defects.sort(key=lambda d: d.severity_score, reverse=True)
    return all_defects
