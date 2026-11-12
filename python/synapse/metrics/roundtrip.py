"""Roundtrip geometry and appearance metrics.

Chamfer distance, F-score, volume breathing, foot sliding, bone length
drift, and albedo colour difference.  Heavy dependencies (``scipy``) are
lazy-imported to keep the base package light.
"""

from __future__ import annotations

import numpy as np

# ── Geometry metrics ─────────────────────────────────────────────────

def _nearest_neighbor_distances(
    src: np.ndarray,
    tgt: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Compute exact Euclidean nearest-neighbour distance from each *src* point to *tgt*.

    Uses ``scipy.spatial.cKDTree`` when available, falling back to a memory-safe
    chunked pure-NumPy implementation (avoiding O(N*M*3) memory blowup).
    """
    if len(src) == 0 or len(tgt) == 0:
        return np.zeros(len(src), dtype=np.float64)

    try:
        from scipy.spatial import cKDTree  # lazy import

        tree = cKDTree(tgt)
        dists, _ = tree.query(src, k=1)
        return np.asarray(dists, dtype=np.float64)
    except ImportError:
        tgt_sq = np.sum(tgt * tgt, axis=1)  # [M]
        min_dists = np.empty(len(src), dtype=np.float64)
        for start in range(0, len(src), chunk_size):
            end = min(start + chunk_size, len(src))
            chunk = src[start:end]  # [C, 3]
            chunk_sq = np.sum(chunk * chunk, axis=1, keepdims=True)  # [C, 1]
            # Exact refinement on nearest candidates to avoid cancellation drift
            approx_d2 = np.maximum(chunk_sq + tgt_sq[np.newaxis, :] - 2.0 * (chunk @ tgt.T), 0.0)
            nn_idx = np.argmin(approx_d2, axis=1)
            diff = chunk - tgt[nn_idx]
            min_dists[start:end] = np.linalg.norm(diff, axis=1)
        return min_dists


def chamfer_mm(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """Bilateral Chamfer distance in millimetres.

    Args:
        points_a: ``[N, 3]`` point cloud in **metres**.
        points_b: ``[M, 3]`` point cloud in **metres**.

    Returns:
        Symmetric Chamfer distance in **millimetres** (scalar float).
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return 0.0

    d_a2b = _nearest_neighbor_distances(a, b)
    d_b2a = _nearest_neighbor_distances(b, a)

    chamfer_m = float(d_a2b.mean() + d_b2a.mean()) / 2.0
    return chamfer_m * 1000.0  # metres → mm


def f_score_1mm(
    points_a: np.ndarray,
    points_b: np.ndarray,
    threshold_mm: float = 1.0,
) -> float:
    """F-score at a given distance threshold (default 1 mm).

    Args:
        points_a: ``[N, 3]`` point cloud in **metres**.
        points_b: ``[M, 3]`` point cloud in **metres**.
        threshold_mm: Distance threshold in millimetres.

    Returns:
        F-score in [0, 1] (harmonic mean of precision and recall).
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return 0.0

    thresh_m = threshold_mm / 1000.0

    d_a2b = _nearest_neighbor_distances(a, b)
    d_b2a = _nearest_neighbor_distances(b, a)

    precision = float(np.mean(d_a2b <= thresh_m))
    recall = float(np.mean(d_b2a <= thresh_m))

    if precision + recall == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


# ── Volume stability ─────────────────────────────────────────────────

def _signed_volume(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> float:
    """Signed volume via directed tetrahedra summation.

    Each face (v0, v1, v2) defines a tetrahedron with the origin and
    contributes ``v0 . (v1 x v2) / 6`` to the total signed volume.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    cross = np.cross(v1, v2)
    vol = np.sum(v0 * cross) / 6.0
    return float(vol)


def volume_breathing_pct(
    meshes_per_frame: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Percentage volume deviation across frames.

    Args:
        meshes_per_frame: List of ``(vertices, faces)`` tuples per
            frame, where vertices is ``[N, 3]`` and faces is ``[F, 3]``.

    Returns:
        Percentage of max absolute volume deviation from mean.
    """
    if not meshes_per_frame:
        return 0.0

    volumes = np.array(
        [abs(_signed_volume(v, f)) for v, f in meshes_per_frame],
        dtype=np.float64,
    )
    mean_vol = volumes.mean()
    if mean_vol < 1e-15:
        return 0.0

    max_dev = np.max(np.abs(volumes - mean_vol))
    return float(max_dev / mean_vol * 100.0)


# ── Skeletal metrics ─────────────────────────────────────────────────

def foot_sliding_cm(
    foot_positions: np.ndarray,
    contact_phases: np.ndarray,
) -> float:
    """Mean foot sliding distance during ground contact (cm).

    Args:
        foot_positions: ``[T, num_feet, 3]`` world-space positions.
        contact_phases: ``[T, num_feet]`` bool array (``True`` during
            ground contact).

    Returns:
        Mean per-frame horizontal displacement during contact in
        **centimetres** (scalar float).
    """
    pos = np.asarray(foot_positions, dtype=np.float64)
    contact = np.asarray(contact_phases, dtype=bool)

    t_count = pos.shape[0]
    if t_count < 2:
        return 0.0

    # Horizontal displacement between consecutive contact frames
    delta = pos[1:] - pos[:-1]  # [T-1, num_feet, 3]
    horiz = np.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2)  # XY plane

    # Both frames must be in contact for the slide to count
    both_contact = contact[:-1] & contact[1:]  # [T-1, num_feet]

    total_slide = horiz[both_contact].sum()
    num_contact = both_contact.sum()

    if num_contact == 0:
        return 0.0

    return float(total_slide / num_contact) * 100.0  # m → cm


def bone_length_drift_pct(
    bone_lengths_per_frame: np.ndarray,
) -> float:
    """Maximum bone-length drift as a percentage of mean length.

    Args:
        bone_lengths_per_frame: ``[T, num_bones]`` bone lengths.

    Returns:
        Maximum percentage drift from mean across all bones and frames.
    """
    bl = np.asarray(bone_lengths_per_frame, dtype=np.float64)
    if bl.size == 0:
        return 0.0

    mean_per_bone = bl.mean(axis=0)  # [num_bones]
    # Guard against zero-length bones
    safe_mean = np.where(mean_per_bone < 1e-12, 1.0, mean_per_bone)
    drift = np.abs(bl - mean_per_bone) / safe_mean * 100.0
    return float(np.max(drift))


# ── Appearance metrics ───────────────────────────────────────────────

def albedo_delta_e00(
    pred_albedo: np.ndarray,
    gt_albedo: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean CIEDE2000 colour difference between albedo patches.

    Converts sRGB albedo to linear, then to CIE Lab, and computes the
    CIEDE2000 metric.

    Args:
        pred_albedo: ``[H, W, 3]`` sRGB albedo in [0, 1].
        gt_albedo: ``[H, W, 3]`` sRGB albedo in [0, 1].
        mask: Optional ``[H, W]`` bool mask (``True`` = valid).

    Returns:
        Mean deltaE00 (scalar float).
    """
    from synapse.style.aces_color import (
        ciede2000_delta_e,
        rgb_to_cielab,
        srgb_to_linear,
    )

    p = np.asarray(pred_albedo, dtype=np.float64)
    g = np.asarray(gt_albedo, dtype=np.float64)

    lab_p = rgb_to_cielab(srgb_to_linear(p))
    lab_g = rgb_to_cielab(srgb_to_linear(g))

    de = ciede2000_delta_e(lab_p, lab_g)  # [H, W]

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        de = de[m]

    if de.size == 0:
        return 0.0

    return float(np.mean(de))
