"""Kabsch / ICP 6-DoF rigid pose alignment.

Extracted from ``synapse.recon.geometry``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def refine_pose_6dof(
    vertices: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Estimate a rigid 6-DoF transformation aligning vertices to target points.

    Uses the closed-form Kabsch/Horn SVD algorithm for matching correspondences,
    and iterative closest point (ICP) when source and target sizes differ.

    Args:
        vertices: Source points, shape (N, 3).
        target_points: Target points, shape (M, 3).
        max_iterations: Maximum ICP iterations for mismatched point sets.
        tolerance: Convergence tolerance on transformation change.

    Returns:
        4x4 homogeneous transformation matrix in SE(3).
    """
    src = np.asarray(vertices, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)

    if len(src) == 0 or len(tgt) == 0:
        return np.eye(4, dtype=np.float64)

    def _solve_kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form Kabsch SVD alignment of paired points."""
        centroid_p = np.mean(p, axis=0)
        centroid_q = np.mean(q, axis=0)
        x = p - centroid_p
        y = q - centroid_q
        h = x.T @ y
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1.0
            r = vt.T @ u.T
        t = centroid_q - r @ centroid_p
        return r, t

    # Equal correspondence direct solution
    if len(src) == len(tgt):
        r, t = _solve_kabsch(src, tgt)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = r
        mat[:3, 3] = t
        return mat

    # ICP for non-matching point counts
    transformed_src = np.copy(src)
    accum_r = np.eye(3, dtype=np.float64)
    accum_t = np.zeros(3, dtype=np.float64)

    # Fast KD-Tree / BVH nearest neighbor lookup if available
    _has_cg_bvh = False
    _cg = None
    try:
        from synapse.bridge.aether_bridge import get_aether_geo

        _cg = get_aether_geo()
        if _cg is not None and hasattr(_cg, "bvh_closest_points"):
            _has_cg_bvh = True
    except AttributeError:
        _has_cg_bvh = False

    _has_scipy_tree = False
    if not _has_cg_bvh:
        try:
            from scipy.spatial import cKDTree

            _has_scipy_tree = True
        except ImportError:
            _has_scipy_tree = False

    # Cache KD-trees to avoid rebuilding the same tree multiple times per
    # iteration (forward NN + reverse NN both query the target tree).
    _tree_cache: dict[int, Any] = {}  # id(array) -> tree

    def _find_nn(
        source: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find nearest-neighbor indices and distances from source to target."""
        if _has_cg_bvh:
            assert _cg is not None
            return cast(tuple[np.ndarray, np.ndarray], _cg.bvh_closest_points(target, source))
        elif _has_scipy_tree:
            target_id = id(target)
            if target_id in _tree_cache:
                tree = _tree_cache[target_id]
            else:
                tree = cKDTree(target)
                _tree_cache[target_id] = tree
            d, idx = tree.query(source, k=1)
            return idx, d
        else:
            # Chunked distance matrix (batch size 1024) to prevent memory explosion
            nn_idx = np.empty(len(source), dtype=np.int64)
            nn_dist = np.empty(len(source), dtype=np.float64)
            batch_size = 1024
            for i in range(0, len(source), batch_size):
                chunk = source[i : i + batch_size]
                diff = chunk[:, np.newaxis, :] - target[np.newaxis, :, :]
                sq_dist = np.sum(diff**2, axis=-1)
                nn_idx[i : i + batch_size] = np.argmin(sq_dist, axis=1)
                nn_dist[i : i + batch_size] = np.sqrt(np.min(sq_dist, axis=1))
            return nn_idx, nn_dist

    for it in range(max_iterations):
        # Invalidate tree cache each iteration since source points changed
        _tree_cache.clear()

        nearest_indices, nearest_dists = _find_nn(transformed_src, tgt)

        # Robust outlier rejection: use mutual nearest-neighbor filtering.
        # A source-target pair is kept only if the reverse NN lookup agrees,
        # effectively eliminating correspondences to noise/outlier targets.
        nearest_dists = np.asarray(nearest_dists, dtype=np.float64)
        matched_tgt = tgt[nearest_indices]

        # Bidirectional check: for each matched target point, verify that
        # the source point is also its nearest neighbor.
        reverse_indices, _ = _find_nn(matched_tgt, transformed_src)
        mutual_mask = reverse_indices == np.arange(len(transformed_src))

        # Additionally apply distance-based trimming (IQR fence)
        if np.sum(mutual_mask) >= 4:
            inlier_mask = mutual_mask
        else:
            # Fallback: keep best 80% by distance
            trim_k = max(int(0.8 * len(transformed_src)), 4)
            order = np.argsort(nearest_dists)
            inlier_mask = np.zeros(len(transformed_src), dtype=bool)
            inlier_mask[order[:trim_k]] = True

        # Step transformation (using only inlier correspondences)
        step_r, step_t = _solve_kabsch(
            transformed_src[inlier_mask], matched_tgt[inlier_mask]
        )

        # Update transformed points
        transformed_src = (step_r @ transformed_src.T).T + step_t
        accum_r = step_r @ accum_r
        accum_t = step_r @ accum_t + step_t

        delta_dist = np.linalg.norm(step_t) + np.abs(np.trace(step_r) - 3.0)
        if delta_dist < tolerance:
            break

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = accum_r
    mat[:3, 3] = accum_t
    return mat
