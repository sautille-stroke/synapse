"""Automated video artifact detection from FGPP-aligned video frames.

Detects three classes of visual defects by comparing generated video
frames against FGPP ground-truth signals:

1. **Temporal flicker** — motion-compensated luminance variance spikes.
2. **Silhouette drift** — video edges diverging from Cryptomatte boundaries.
3. **Ground violation** — entities sinking below the world-space ground plane.

All detection kernels are pure NumPy (vectorised, no pixel loops).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from synapse.fgpp.frame import FGPPFrame


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
    """Convert uint8 sRGB [H, W, 3] → float32 luminance [H, W]."""
    return (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    ) / 255.0


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
    return np.sqrt(gx * gx + gy * gy)


def _block_mean(img: np.ndarray, block: int) -> np.ndarray:
    """Compute per-block mean of a float32 [H, W] image.

    Returns a float32 array of shape ``(H // block, W // block)``
    holding the mean value of each non-overlapping block.
    """
    h, w = img.shape
    bh = (h // block) * block
    bw = (w // block) * block
    cropped = img[:bh, :bw]
    blocks = cropped.reshape(bh // block, block, bw // block, block)
    return blocks.mean(axis=(1, 3))


def _block_variance(img: np.ndarray, block: int) -> np.ndarray:
    """Compute per-block variance of a float32 [H, W] image.

    Returns a float32 array of shape ``(H // block, W // block)``
    holding the variance of each non-overlapping block.
    """
    h, w = img.shape
    bh = (h // block) * block
    bw = (w // block) * block
    cropped = img[:bh, :bw]
    blocks = cropped.reshape(bh // block, block, bw // block, block)
    return blocks.var(axis=(1, 3))


def _simple_warp(
    frame: np.ndarray, mv: np.ndarray
) -> np.ndarray:
    """Warp *frame* by *mv* using nearest-neighbour sampling.

    Args:
        frame: float32 [H, W] luminance image.
        mv: float16/32 [H, W, 2] screen-space motion vectors (dx, dy).

    Returns:
        Warped float32 [H, W] image.
    """
    h, w = frame.shape
    yy, xx = np.mgrid[:h, :w]
    mv_f = np.nan_to_num(mv.astype(np.float32), nan=0.0)
    src_x = np.clip(np.round(xx + mv_f[..., 0]).astype(np.intp), 0, w - 1)
    src_y = np.clip(np.round(yy + mv_f[..., 1]).astype(np.intp), 0, h - 1)
    return frame[src_y, src_x]


def _distance_transform_bf(mask: np.ndarray) -> np.ndarray:
    """Brute-force Euclidean distance transform (pure NumPy fallback).

    Args:
        mask: bool [H, W].  Distance is measured **from True pixels**.

    Returns:
        float32 [H, W] distance from the nearest True pixel.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask).astype(np.float32)
    except ImportError:  # pragma: no cover
        pass

    # Fallback: chamfer-style 2-pass approximation.
    h, w = mask.shape
    dist = np.full((h, w), dtype=np.float32, fill_value=float(h + w))
    dist[mask] = 0.0
    # Forward pass
    for y in range(h):
        for x in range(w):
            if y > 0:
                dist[y, x] = min(dist[y, x], dist[y - 1, x] + 1.0)
            if x > 0:
                dist[y, x] = min(dist[y, x], dist[y, x - 1] + 1.0)
    # Backward pass
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if y < h - 1:
                dist[y, x] = min(dist[y, x], dist[y + 1, x] + 1.0)
            if x < w - 1:
                dist[y, x] = min(dist[y, x], dist[y, x + 1] + 1.0)
    return dist


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
        if (
            t + 1 < len(fgpp_frames)
            and fgpp_frames[t + 1].motion_vector is not None
        ):
            warped = _simple_warp(
                frame_a, fgpp_frames[t + 1].motion_vector
            )
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
        max_mean = float(bmean.max())
        hot_ratio = float(hot_mask.sum()) / max(bmean.size, 1)
        severity = float(np.clip(hot_ratio * max_mean / threshold, 0.0, 1.0))

        # Bounding box of hot blocks in pixel space.
        ys, xs = np.nonzero(hot_mask)
        x1 = int(xs.min()) * _BLOCK_SIZE
        y1 = int(ys.min()) * _BLOCK_SIZE
        x2 = (int(xs.max()) + 1) * _BLOCK_SIZE
        y2 = (int(ys.max()) + 1) * _BLOCK_SIZE

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
        dist_from_boundary = _distance_transform_bf(boundary)

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
                int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1,
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
    """Detect entities whose world-space Y drops below ground."""
    defects: list[VideoDefectRegion] = []

    # Collect per-entity, per-frame minimum world Y.
    entity_violations: dict[int, list[tuple[int, float]]] = {}

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

        # Camera-space 3D coordinates
        x_cam = (xx.astype(np.float64) - cam.cx) * z / cam.fx
        y_cam = (yy.astype(np.float64) - cam.cy) * z / cam.fy
        z_cam = z

        # Build [N, 4] homogeneous coordinates
        ones = np.ones_like(z)
        pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=-1)  # [H,W,4]
        pts_flat = pts_cam.reshape(-1, 4)

        # Camera-to-world
        world_mat = cam.world_matrix.astype(np.float64)
        pts_world = (world_mat @ pts_flat.T).T  # [N, 4]
        world_y = pts_world[:, 1].reshape(h, w)

        # Check per entity
        unique_ids = np.unique(cid)
        for eid in unique_ids:
            if eid == 0:
                continue
            mask = cid == eid
            entity_y = world_y[mask]
            valid = np.isfinite(entity_y)
            if not np.any(valid):
                continue
            min_y = float(entity_y[valid].min())
            if min_y < ground_y:
                entity_violations.setdefault(int(eid), []).append(
                    (t, min_y)
                )

    # Emit defects for entities violating ground across ≥2 frames.
    for eid, violations in entity_violations.items():
        if len(violations) < 2:
            continue
        frames = [v[0] for v in violations]
        min_ys = [v[1] for v in violations]
        worst_y = min(min_ys)
        severity = float(np.clip(
            abs(worst_y - ground_y) / 1.0, 0.0, 1.0
        ))
        defects.append(VideoDefectRegion(
            defect_type="ground_violation",
            frame_range=(min(frames), max(frames)),
            bbox_xyxy=(0, 0, 0, 0),
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
