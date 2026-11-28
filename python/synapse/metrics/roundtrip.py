"""Roundtrip geometry and appearance metrics.

Chamfer distance, F-score, volume breathing, foot sliding, bone length
drift, and albedo colour difference.  Heavy dependencies (``scipy``) are
lazy-imported to keep the base package light.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__ = [
    "chamfer_mm",
    "compute_chamfer_distance_mm",
    "sample_mesh_surface",
    "f_score_1mm",
    "f_score_5mm",
    "watertight_pct",
    "volume_breathing_pct",
    "foot_sliding_cm",
    "bone_length_drift_pct",
    "albedo_delta_e00",
    "add_s_norm",
    "reprojection_error_px",
]

# ── Geometry metrics ─────────────────────────────────────────────────


def _nearest_neighbor_distances(
    src: np.ndarray,
    tgt: np.ndarray,
) -> np.ndarray:
    """Compute exact Euclidean nearest-neighbour distance from each *src* point to *tgt*.

    Uses ``aether_geo.bvh_closest_points`` when the native extension is available,
    falling back to a pure NumPy brute-force / scipy KDTree implementation otherwise.
    """
    if len(src) == 0 or len(tgt) == 0:
        return np.zeros(len(src), dtype=np.float64)

    from synapse.bridge.aether_bridge import get_aether_geo

    geo = get_aether_geo()
    if geo is not None:
        try:
            _, dists = geo.bvh_closest_points(tgt, src)
            return np.asarray(dists, dtype=np.float64)
        except Exception:
            pass

    # Pure Python fallback: try scipy KDTree, else brute-force
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(tgt)
        dists, _ = tree.query(src, k=1)
        return np.asarray(dists, dtype=np.float64)
    except ImportError:
        pass

    # Brute-force for small point clouds (chunked to limit memory)
    s = np.asarray(src, dtype=np.float64)
    t = np.asarray(tgt, dtype=np.float64)
    chunk = 2048
    result = np.empty(len(s), dtype=np.float64)
    for i in range(0, len(s), chunk):
        end = min(i + chunk, len(s))
        diff = s[i:end, np.newaxis, :] - t[np.newaxis, :, :]  # [chunk, M, 3]
        result[i:end] = np.sqrt(np.sum(diff ** 2, axis=-1).min(axis=1))
    return result


def chamfer_mm(
    points_a: np.ndarray,
    points_b: np.ndarray,
    reduction: Literal["mean", "sum"] = "mean",
) -> float:
    """Bilateral Chamfer distance in millimetres.

    Note on Industry Normalization Conventions:
        - In PointNet, AtlasNet, and Microsoft TRELLIS benchmarks, Chamfer distance
          is often reported as the unweighted sum:
              CD_sum = mean(d_a2b) + mean(d_b2a)
          which is exactly 2x the bidirectional average.
        - In Synapse and RFC-0063, the default convention is the bidirectional average:
              CD_mean = (mean(d_a2b) + mean(d_b2a)) / 2.0
        - Pass ``reduction="sum"`` to obtain the PointNet/TRELLIS sum metric (2 * CD_mean).

    Args:
        points_a: ``[N, 3]`` point cloud in **metres**.
        points_b: ``[M, 3]`` point cloud in **metres**.
        reduction: ``"mean"`` for bidirectional average (default, RFC-0063), or
            ``"sum"`` for unweighted sum (PointNet / TRELLIS convention).

    Returns:
        Symmetric Chamfer distance in **millimetres** (scalar float).
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(
            f"Unsupported reduction: {reduction!r}. Choose 'mean' or 'sum'."
        )

    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    from synapse.bridge.aether_bridge import get_aether_geo, native_compute_chamfer

    # Try aether_geo first (full geometry module)
    geo = get_aether_geo()
    if geo is not None:
        try:
            return float(geo.compute_chamfer_distance(a, b, reduction=reduction))
        except ValueError:
            raise
        except Exception:
            pass

    # Try aether_metrics (lighter metrics-only module with Hausdorff bonus)
    native_cd = native_compute_chamfer(a, b, reduction=reduction)
    if native_cd is not None:
        return native_cd

    # Pure NumPy fallback via _nearest_neighbor_distances
    d_a2b = _nearest_neighbor_distances(a, b) * 1000.0  # m -> mm
    d_b2a = _nearest_neighbor_distances(b, a) * 1000.0

    mean_a2b = float(np.mean(d_a2b))
    mean_b2a = float(np.mean(d_b2a))

    if reduction == "sum":
        return mean_a2b + mean_b2a
    return (mean_a2b + mean_b2a) / 2.0



def _triangulate_polygons(faces: Any) -> np.ndarray:
    """Convert arbitrary polygon faces to triangle faces via fan triangulation.

    Supports:
      - ``[M, 3]`` int array (returned directly as int64)
      - ``[M, K]`` int array where K >= 4 (fan-triangulated to [M * (K - 2), 3])
      - Sequence of variable-length index sequences (e.g. list of lists)
    """
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        if faces.shape[1] == 3:
            return faces.astype(np.int64, copy=False)
        if faces.shape[1] >= 4:
            f = faces.astype(np.int64, copy=False)
            k = f.shape[1]
            tris = []
            for i in range(1, k - 1):
                tris.append(f[:, [0, i, i + 1]])
            return np.vstack(tris)

    # Sequence of polygons
    tris_list = []
    for poly in faces:
        p = [int(idx) for idx in poly]
        n = len(p)
        for i in range(1, n - 1):
            tris_list.append((p[0], p[i], p[i + 1]))
    if not tris_list:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(tris_list, dtype=np.int64)


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: Any,
    num_samples: int = 10000,
    seed: int | None = 42,
) -> np.ndarray:
    """Sample points uniformly on a triangle mesh surface weighted by face area.

    Solves vertex-density sensitivity: models with non-uniform tessellation
    (dense planar faces vs sparse curvature) receive unbiased, surface-area-proportional
    point clouds for geometry evaluation.

    Args:
        vertices: ``[N, 3]`` float array of vertex coordinates in metres.
        faces: ``[M, 3]`` (or ``[M, K]`` for polygons) face vertex indices.
        num_samples: Number of surface points to sample (default: 10000).
        seed: Optional RNG seed for reproducible sampling.

    Returns:
        ``[num_samples, 3]`` float64 array of sampled 3D points on the mesh surface.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = _triangulate_polygons(faces)
    if len(v) == 0 or len(f) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Triangle vertices: [M, 3]
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    # Face areas: Area = 0.5 * ||(v1 - v0) x (v2 - v0)||
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(np.sum(areas))

    rng = np.random.default_rng(seed)
    if total_area <= 1e-12:
        indices = rng.integers(0, len(v), size=num_samples)
        return v[indices]

    probs = areas / total_area
    face_choices = rng.choice(len(f), size=num_samples, p=probs)

    # Turk/Osada uniform triangle interior sampling via square root folding
    r1 = rng.uniform(0.0, 1.0, size=num_samples)
    r2 = rng.uniform(0.0, 1.0, size=num_samples)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    val_v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    pts_v0 = v0[face_choices]
    pts_v1 = v1[face_choices]
    pts_v2 = v2[face_choices]

    sampled = u[:, np.newaxis] * pts_v0 + val_v[:, np.newaxis] * pts_v1 + w[:, np.newaxis] * pts_v2
    return np.asarray(sampled, dtype=np.float64)


def _extract_or_sample_points(
    mesh_or_points: Any,
    faces: Any = None,
    num_samples: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Extract or sample points from mesh or point cloud."""
    if isinstance(mesh_or_points, (tuple, list)) and len(mesh_or_points) == 2:
        try:
            v_cand, f_cand = mesh_or_points
            verts = np.asarray(v_cand, dtype=np.float64)
            if verts.ndim == 2 and verts.shape[1] == 3:
                return sample_mesh_surface(verts, f_cand, num_samples=num_samples, seed=seed)
        except (ValueError, TypeError):
            pass

    if hasattr(mesh_or_points, "vertices") and hasattr(mesh_or_points, "faces"):
        verts = np.asarray(mesh_or_points.vertices, dtype=np.float64)
        mesh_faces = mesh_or_points.faces
        if len(mesh_faces) > 0:
            return sample_mesh_surface(verts, mesh_faces, num_samples=num_samples, seed=seed)
        return verts

    arr = np.asarray(mesh_or_points, dtype=np.float64)
    if faces is not None and len(faces) > 0:
        return sample_mesh_surface(arr, faces, num_samples=num_samples, seed=seed)
    return arr


def compute_chamfer_distance_mm(
    mesh_or_points_a: Any,
    mesh_or_points_b: Any,
    faces_a: np.ndarray | None = None,
    faces_b: np.ndarray | None = None,
    num_samples: int = 10000,
    reduction: Literal["mean", "sum"] = "mean",
    seed: int = 42,
    seed_b: int | None = None,
) -> float:
    """Bilateral Chamfer distance in millimetres with area-weighted surface sampling.

    Accepts point clouds, mesh objects (with ``.vertices`` and ``.faces``),
    or vertex arrays with explicit face indices. When faces are provided,
    applies area-weighted triangle uniform surface sampling (:func:`sample_mesh_surface`)
    to eliminate vertex density bias.

    Args:
        mesh_or_points_a: Point cloud [N, 3] or mesh A.
        mesh_or_points_b: Point cloud [M, 3] or mesh B.
        faces_a: Optional triangle faces for A.
        faces_b: Optional triangle faces for B.
        num_samples: Number of surface samples when sampling from faces (default: 10000).
        reduction: "mean" (default) or "sum".
        seed: RNG seed for reproducible surface sampling.
        seed_b: Optional separate RNG seed for sampling mesh B. When omitted, defaults
            to ``seed`` (or pass ``seed + 1`` for multi-seed variance testing).

    Returns:
        Symmetric Chamfer distance in millimetres.
    """
    eff_seed_b = seed if seed_b is None else seed_b
    pts_a = _extract_or_sample_points(mesh_or_points_a, faces_a, num_samples, seed)
    pts_b = _extract_or_sample_points(mesh_or_points_b, faces_b, num_samples, eff_seed_b)
    return chamfer_mm(pts_a, pts_b, reduction=reduction)


def f_score_1mm(
    points_a: Any,
    points_b: Any,
    threshold_mm: float = 1.0,
    faces_a: Any = None,
    faces_b: Any = None,
    num_samples: int = 10000,
    seed: int = 42,
    seed_b: int | None = None,
) -> float:
    """F-score at a given distance threshold (default 1 mm).

    Args:
        points_a: ``[N, 3]`` point cloud, or ``(vertices, faces)`` mesh.
        points_b: ``[M, 3]`` point cloud, or ``(vertices, faces)`` mesh.
        threshold_mm: Distance threshold in millimetres.
        faces_a: Optional face indices for mesh A.
        faces_b: Optional face indices for mesh B.
        num_samples: Sample count for surface sampling.
        seed: RNG seed for surface sampling.
        seed_b: Optional separate RNG seed for mesh B. When omitted, uses
            ``seed`` for self-comparison (identical objects) or ``seed + 1``
            for independent sampling of distinct meshes.

    Returns:
        F-score in [0, 1] (harmonic mean of precision and recall).
    """
    is_self = (
        points_a is points_b
        or (
            isinstance(points_a, (tuple, list))
            and isinstance(points_b, (tuple, list))
            and len(points_a) == 2
            and len(points_b) == 2
            and points_a[0] is points_b[0]
            and points_a[1] is points_b[1]
        )
    )
    if is_self and seed_b is None:
        eff_seed_b = seed
    else:
        eff_seed_b = (seed + 1 if seed is not None else None) if seed_b is None else seed_b

    a = _extract_or_sample_points(points_a, faces_a, num_samples=num_samples, seed=seed)
    b = _extract_or_sample_points(points_b, faces_b, num_samples=num_samples, seed=eff_seed_b)
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


def f_score_5mm(
    points_a: Any,
    points_b: Any,
    threshold_mm: float = 5.0,
    faces_a: Any = None,
    faces_b: Any = None,
    num_samples: int = 10000,
    seed: int = 42,
    seed_b: int | None = None,
) -> float:
    """F-score at 5 mm distance threshold (RFC-0063 Track 1/Track 3 standard).

    Args:
        points_a: ``[N, 3]`` point cloud, or ``(vertices, faces)`` mesh.
        points_b: ``[M, 3]`` point cloud, or ``(vertices, faces)`` mesh.
        threshold_mm: Distance threshold in millimetres (default: 5.0).
        faces_a: Optional face indices for mesh A.
        faces_b: Optional face indices for mesh B.
        num_samples: Sample count for surface sampling.
        seed: RNG seed for surface sampling.
        seed_b: Optional separate RNG seed for mesh B.

    Returns:
        F-score in [0, 1] (harmonic mean of precision and recall).
    """
    return f_score_1mm(
        points_a,
        points_b,
        threshold_mm=threshold_mm,
        faces_a=faces_a,
        faces_b=faces_b,
        num_samples=num_samples,
        seed=seed,
        seed_b=seed_b,
    )


def _is_mesh_watertight(faces_or_vertices: Any, faces: Any = None) -> bool:
    """Check whether a mesh face array represents a closed 2-manifold orientation-consistent mesh.

    Accepts:
      - ``_is_mesh_watertight(faces)``
      - ``_is_mesh_watertight(vertices, faces)``
    """
    if faces is not None:
        verts = np.asarray(faces_or_vertices, dtype=np.float64)
        if not np.all(np.isfinite(verts)):
            return False
        raw_faces = faces
    else:
        verts = None
        raw_faces = faces_or_vertices

    # Fast path: checkWatertight via native extension if uniform faces and vertices are provided
    if verts is not None and isinstance(raw_faces, np.ndarray) and raw_faces.ndim == 2:
        from synapse.bridge.aether_bridge import get_aether_geo

        geo = get_aether_geo()
        if geo is not None:
            try:
                return bool(
                    geo.check_watertight(
                        verts,
                        raw_faces.astype(np.int64, copy=False),
                        int(raw_faces.shape[1]),
                    )
                )
            except Exception:
                pass  # fall through to pure Python

    # Can be a list of polygons or a numpy array
    f: list[Any] | np.ndarray
    if isinstance(raw_faces, np.ndarray) and raw_faces.dtype != object:
        f = raw_faces.astype(np.int64, copy=False)
    else:
        f = [list(map(int, p)) for p in raw_faces]

    if len(f) == 0:
        return False

    directed_edges: dict[tuple[int, int], int] = {}
    undirected_edges: dict[tuple[int, int], int] = {}
    from collections import defaultdict

    vertex_faces: dict[int, list[int]] = defaultdict(list)

    for f_idx, poly in enumerate(f):
        n = len(poly)
        for i in range(n):
            u, w = int(poly[i]), int(poly[(i + 1) % n])
            vertex_faces[u].append(f_idx)
            directed_edges[(u, w)] = directed_edges.get((u, w), 0) + 1
            undirected_key = (min(u, w), max(u, w))
            undirected_edges[undirected_key] = undirected_edges.get(undirected_key, 0) + 1

    # Every edge must be shared by exactly 2 faces
    for count in undirected_edges.values():
        if count != 2:
            return False

    # Consistent orientation check: opposite directed edges
    for (u, w), count in directed_edges.items():
        if count != 1:
            return False
        if directed_edges.get((w, u), 0) != 1:
            return False

    # Vertex manifold check: link of each vertex must form a single closed simple cycle (prevents pinch vertices)
    for v, f_indices in vertex_faces.items():
        if len(f_indices) < 3:
            return False
        next_vertex: dict[int, int] = {}
        for fi in f_indices:
            poly = f[fi]
            n = len(poly)
            for i in range(n):
                if int(poly[i]) == v:
                    a = int(poly[(i + 1) % n])
                    b = int(poly[(i - 1) % n])
                    if a in next_vertex:
                        return False
                    next_vertex[a] = b
                    break
        if not next_vertex:
            return False
        start = next(iter(next_vertex.keys()))
        curr = start
        visited_count = 0
        while True:
            if curr not in next_vertex:
                return False
            curr = next_vertex[curr]
            visited_count += 1
            if curr == start:
                break
            if visited_count > len(next_vertex):
                return False
        if visited_count != len(next_vertex):
            return False

    return True


def watertight_pct(
    meshes: list[Any] | tuple[Any, ...],
) -> float:
    """Compute percentage of watertight meshes in a batch.

    Each mesh entry in ``meshes`` may be:
    - A tuple ``(vertices, faces)`` where vertices is ``[N, 3]`` and faces is ``[F, 3]``
    - A mesh object with ``.vertices`` and ``.faces`` attributes
    - A trimesh-like object with an ``.is_watertight`` boolean property

    Args:
        meshes: Collection of 3D triangular surface meshes.

    Returns:
        Percentage of watertight meshes in [0.0, 100.0]. Returns 0.0 if empty.
    """
    if not meshes:
        return 0.0

    watertight_count = 0
    for m in meshes:
        if hasattr(m, "is_watertight") and isinstance(m.is_watertight, bool):
            v = getattr(m, "vertices", None)
            v_finite = True
            if v is not None:
                v_arr = np.asarray(v, dtype=np.float64)
                v_finite = bool(np.all(np.isfinite(v_arr)))
            if m.is_watertight and v_finite:
                watertight_count += 1
        elif isinstance(m, (tuple, list)) and len(m) >= 2:
            v = np.asarray(m[0], dtype=np.float64)
            f = np.asarray(m[1], dtype=np.int64)
            if _is_mesh_watertight(v, f):
                watertight_count += 1
        elif hasattr(m, "faces"):
            v = getattr(m, "vertices", None)
            f = np.asarray(m.faces, dtype=np.int64)
            if v is not None:
                if _is_mesh_watertight(v, f):
                    watertight_count += 1
            elif _is_mesh_watertight(f):
                watertight_count += 1

    return float((watertight_count / len(meshes)) * 100.0)


# ── Volume stability ─────────────────────────────────────────────────


def _signed_volume(
    vertices: np.ndarray,
    faces: Any,
) -> float:
    """Signed volume via directed tetrahedra summation.

    Each face (v0, v1, v2) defines a tetrahedron with the origin and
    contributes ``v0 . (v1 x v2) / 6`` to the total signed volume.
    Automatically triangulates polygon faces via fan triangulation.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = _triangulate_polygons(faces)
    if len(v) == 0 or len(f) == 0:
        return 0.0

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
        foot_positions: ``[T, num_feet, 3]`` world-space positions (Y-up convention).
        contact_phases: ``[T, num_feet]`` bool array (``True`` during
            ground contact).

    Returns:
        Mean per-frame horizontal displacement on ground XZ plane during contact in
        **centimetres** (scalar float). Note: multiply by fps to convert to cm/s
        sliding velocity if needed.
    """
    pos = np.asarray(foot_positions, dtype=np.float64)
    contact = np.asarray(contact_phases, dtype=bool)

    t_count = pos.shape[0]
    if t_count < 2:
        return 0.0

    # Horizontal displacement on Y-up ground plane (XZ plane)
    delta = pos[1:] - pos[:-1]  # [T-1, num_feet, 3]
    horiz = np.sqrt(delta[..., 0] ** 2 + delta[..., 2] ** 2)  # XZ plane

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
    input_space: Literal["srgb", "linear"] = "srgb",
) -> float:
    """Mean CIEDE2000 colour difference between albedo patches.

    Converts albedo to CIE Lab and computes the CIEDE2000 metric.

    Args:
        pred_albedo: ``[H, W, 3]`` albedo in [0, 1].
        gt_albedo: ``[H, W, 3]`` albedo in [0, 1].
        mask: Optional ``[H, W]`` bool mask (``True`` = valid).
        input_space: Colour space of the input images (``"srgb"`` or
            ``"linear"``).  When ``"srgb"``, inputs are converted to linear
            before computing Lab.  When ``"linear"``, the sRGB-to-linear
            step is skipped.

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
    if hasattr(pred_albedo, "dtype") and pred_albedo.dtype == np.uint8:
        p /= 255.0
    if hasattr(gt_albedo, "dtype") and gt_albedo.dtype == np.uint8:
        g /= 255.0

    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: pred_albedo={p.shape} vs gt_albedo={g.shape}")

    if input_space == "srgb":
        lab_p = rgb_to_cielab(srgb_to_linear(p))
        lab_g = rgb_to_cielab(srgb_to_linear(g))
    elif input_space == "linear":
        lab_p = rgb_to_cielab(p)
        lab_g = rgb_to_cielab(g)
    else:
        raise ValueError(f"Unsupported input_space: {input_space!r}. Expected 'srgb' or 'linear'.")

    de = ciede2000_delta_e(lab_p, lab_g)  # [H, W]

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        de = de[m]

    if de.size == 0:
        return 0.0

    return float(np.mean(de))


def _estimate_object_diameter(points: np.ndarray) -> float:
    """Estimate orientation-invariant 3D object diameter for ADD-S normalization."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    if len(pts) > 512:
        idx = np.linspace(0, len(pts) - 1, 512, dtype=np.int64)
        sample = pts[idx]
    else:
        sample = pts
    diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
    max_sq = float(np.max(np.sum(diff * diff, axis=-1)))
    return float(np.sqrt(max(max_sq, 0.0)))


def add_s_norm(
    pred_points: Any,
    gt_points: Any,
    *,
    diameter: float | None = None,
    num_samples: int = 10000,
    seed: int = 42,
) -> float:
    """Symmetric Average Distance of Model Points normalised by object diameter (ADD-S).

    Follows the BOP / NOCS pose-and-geometry metric definition:
        ``ADD-S_norm = 0.5 * (mean_i min_j ||p_i - g_j|| + mean_j min_i ||g_j - p_i||) / D``
    where ``D`` is the 3D ground-truth object diameter (or explicit *diameter*).

    Args:
        pred_points: Predicted ``[N, 3]`` point cloud or mesh object.
        gt_points: Ground-truth ``[M, 3]`` point cloud or mesh object.
        diameter: Optional explicit object diameter in the same units as input points.
            When omitted, estimated from *gt_points*.
        num_samples: Surface sample count when mesh objects with faces are supplied.
        seed: Reproducible sampling seed.

    Returns:
        Dimensionless normalised symmetric point-set distance (>= 0.0; ``nan`` if empty).
    """
    p = _extract_or_sample_points(pred_points, num_samples=num_samples, seed=seed)
    g = _extract_or_sample_points(gt_points, num_samples=num_samples, seed=seed)
    if len(p) == 0 or len(g) == 0:
        return float("nan")

    eff_diam = float(diameter) if diameter is not None else _estimate_object_diameter(g)
    if eff_diam <= 1e-8:
        eff_diam = 1.0

    d_p2g = _nearest_neighbor_distances(p, g)
    d_g2p = _nearest_neighbor_distances(g, p)
    add_s = 0.5 * (float(np.mean(d_p2g)) + float(np.mean(d_g2p)))
    return float(add_s / eff_diam)


def reprojection_error_px(
    points_3d: np.ndarray,
    observed_2d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray | None = None,
    t: np.ndarray | None = None,
) -> float:
    """Mean Euclidean 2D pixel reprojection error under pinhole camera ``(K, R, t)``.

    Transforms world 3D points ``p_cam = R @ p_world + t`` (or identity if ``R``/``t``
    are omitted), projects valid front-facing points (``Z > 1e-8``) through
    intrinsics ``K``, and returns mean pixel L2 error against *observed_2d*.

    Args:
        points_3d: World coordinates, shape ``[N, 3]`` (or ``[T, N, 3]``).
        observed_2d: Observed image pixel coordinates ``(u, v)``, shape ``[N, 2]``
            (or ``[T, N, 2]``).
        K: Intrinsics matrix ``[3, 3]``.
        R: Optional rotation matrix ``[3, 3]`` (world-to-camera) or ``[4, 4]``
            extrinsic matrix. Defaults to identity.
        t: Optional translation vector ``[3]`` or ``[3, 1]``. Defaults to zero.

    Returns:
        Mean 2D Euclidean reprojection error in pixels (>= 0.0).
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    obs = np.asarray(observed_2d, dtype=np.float64)
    k_mat = np.asarray(K, dtype=np.float64)

    if pts.size == 0 or obs.size == 0:
        return 0.0

    pts_flat = pts.reshape(-1, 3)
    obs_flat = obs.reshape(-1, 2)
    if pts_flat.shape[0] != obs_flat.shape[0]:
        raise ValueError(
            f"Point count mismatch: points_3d={pts_flat.shape[0]} vs observed_2d={obs_flat.shape[0]}"
        )
    if k_mat.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics K, got shape {k_mat.shape}.")

    if R is None:
        r_mat = np.eye(3, dtype=np.float64)
        t_vec = np.zeros(3, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64).reshape(3)
    else:
        r_arr = np.asarray(R, dtype=np.float64)
        if r_arr.shape == (4, 4):
            r_mat = r_arr[:3, :3]
            t_vec = r_arr[:3, 3] if t is None else np.asarray(t, dtype=np.float64).reshape(3)
        elif r_arr.shape == (3, 3):
            r_mat = r_arr
            t_vec = np.zeros(3, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64).reshape(3)
        else:
            raise ValueError(f"Expected 3x3 or 4x4 matrix R, got shape {r_arr.shape}.")

    p_cam = (pts_flat @ r_mat.T) + t_vec[np.newaxis, :]
    z = p_cam[:, 2]
    valid = z > 1e-8
    if not np.any(valid):
        return float("inf")

    p_valid = p_cam[valid]
    proj_h = p_valid @ k_mat.T
    uv = proj_h[:, :2] / p_valid[:, 2:3]
    err = np.linalg.norm(uv - obs_flat[valid], axis=1)
    return float(np.mean(err))

