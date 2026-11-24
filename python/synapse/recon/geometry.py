"""Ground Truth Oracle for geometric algorithm correctness verification.

This module provides pure-Python (NumPy) reference implementations of mesh
geometry checks (manifold, watertight, degenerate faces, self-intersections,
connected components, etc.).

**This is NOT production code.**  Normal workflows use the C++20 native kernel
(``aether_geo`` via ``@aether//geo``) for actual geometry validation.
If the C++ extension is unavailable, the geometry gate raises ``ImportError``
rather than falling back to this module.

This Python Oracle exists solely to:

1. Serve as the algorithm correctness reference (ground truth).
2. Enable parity testing against the C++ implementation
   (see ``tests/test_geometry_oracle_parity.py``).

Additionally includes: area-weighted normal estimation, Kabsch/ICP 6-DoF rigid
alignment, and quad-dominant remeshing — utilities used by the recon pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

import numpy as np


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted smooth vertex normals.

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Triangle or quad indices, shape (F, 3) or (F, 4).

    Returns:
        Unit normal vectors per vertex, shape (V, 3).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    num_verts = len(v)
    normals = np.zeros((num_verts, 3), dtype=np.float64)

    if len(v) == 0 or len(f) == 0:
        return normals.astype(np.float32)

    if f.shape[1] == 3:
        # Triangles
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        for i in range(3):
            np.add.at(normals, f[:, i], face_normals)
    elif f.shape[1] == 4:
        # Quads: diagonal cross product
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        v3 = v[f[:, 3]]
        face_normals = 0.5 * np.cross(v2 - v0, v3 - v1)
        for i in range(4):
            np.add.at(normals, f[:, i], face_normals)
    else:
        # Fallback for polygon soup
        for poly in f:
            if len(poly) >= 3:
                fn = np.cross(v[poly[1]] - v[poly[0]], v[poly[2]] - v[poly[0]])
                for idx in poly:
                    normals[idx] += fn

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    normals = normals / lengths
    return normals.astype(np.float32)


def check_manifold(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Check whether a mesh is a valid 2-manifold surface.

    A surface mesh is 2-manifold if:
    1. Every edge is incident to at most 2 faces (no non-manifold edges).
    2. Vertices do not form non-manifold pinch points (the incident faces
       around every vertex form a single connected umbrella/fan).
    3. Vertices do not contain NaN, Inf, or empty arrays.

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Face indices, shape (F, 3) or (F, 4).

    Returns:
        True if the mesh satisfies 2-manifold criteria, False otherwise.
    """
    v = np.asarray(vertices)
    f = np.asarray(faces)

    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] < 3:
        return False

    if len(v) == 0 or len(f) == 0:
        return False

    if not np.all(np.isfinite(v)):
        return False

    if np.min(f) < 0 or np.max(f) >= len(v):
        return False

    # Check non-manifold edges & per-face vertex uniqueness
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    vertex_faces: dict[int, list[int]] = defaultdict(list)
    face_incident_edges: list[dict[int, list[tuple[int, int]]]] = [
        defaultdict(list) for _ in range(len(f))
    ]

    num_faces = len(f)
    num_verts_per_face = f.shape[1]

    for face_idx in range(num_faces):
        poly = [int(x) for x in f[face_idx]]
        if len(set(poly)) != num_verts_per_face:
            return False  # Duplicate / self-pinching vertex within polygon
        for k in range(num_verts_per_face):
            u, w = poly[k], poly[(k + 1) % num_verts_per_face]
            edge_key = (min(u, w), max(u, w))
            edge_counts[edge_key] += 1
            if edge_counts[edge_key] > 2:
                # Non-manifold edge shared by >= 3 faces
                return False
            vertex_faces[u].append(face_idx)
            face_incident_edges[face_idx][u].append(edge_key)
            face_incident_edges[face_idx][w].append(edge_key)

    # Check vertex manifoldness: face umbrella connectedness around each vertex
    for vert_idx, incident_faces in vertex_faces.items():
        if len(incident_faces) <= 1:
            continue
        # Build edge → face list map in O(d)
        edge_to_faces_local: dict[tuple[int, int], list[int]] = defaultdict(list)
        for fi in incident_faces:
            for edge_key in face_incident_edges[fi][vert_idx]:
                canonical = (min(edge_key), max(edge_key))
                edge_to_faces_local[canonical].append(fi)

        # Build adjacency from shared edges in O(d)
        adj: dict[int, set[int]] = defaultdict(set)
        for faces_sharing_edge in edge_to_faces_local.values():
            for i, f1 in enumerate(faces_sharing_edge):
                for f2 in faces_sharing_edge[i + 1 :]:
                    adj[f1].add(f2)
                    adj[f2].add(f1)

        # Count connected components among incident faces
        visited: set[int] = set()
        components = 0
        for f_seed in incident_faces:
            if f_seed not in visited:
                components += 1
                if components > 1:
                    # Non-manifold vertex (e.g. hourglass pinch point)
                    return False
                queue = [f_seed]
                visited.add(f_seed)
                while queue:
                    curr = queue.pop()
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

    return True


def check_watertight(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Check whether a mesh is strictly watertight (closed 2-manifold).

    A mesh is watertight if it is 2-manifold and has zero boundary edges
    (every edge is shared by exactly 2 faces with opposite orientations).

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Face indices, shape (F, 3).

    Returns:
        True if the mesh is closed and watertight, False otherwise.
    """
    if not check_manifold(vertices, faces):
        return False

    f = np.asarray(faces, dtype=np.int64)
    directed_edges: dict[tuple[int, int], int] = defaultdict(int)
    undirected_edges: dict[tuple[int, int], int] = defaultdict(int)

    for poly in f:
        n = len(poly)
        for i in range(n):
            u, w = int(poly[i]), int(poly[(i + 1) % n])
            directed_edges[(u, w)] += 1
            undirected_edges[(min(u, w), max(u, w))] += 1

    # Every edge must be shared by exactly 2 faces
    for edge, count in undirected_edges.items():
        if count != 2:
            return False  # Boundary edge or isolated edge

    # Consistent orientation check: opposite directed edges
    for (u, w), count in directed_edges.items():
        if count != 1:
            return False
        if directed_edges.get((w, u), 0) != 1:
            return False

    return True


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


def _unwrap_uv_atlas_with_seams(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    expand_seams: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate UV coordinates via xatlas or spherical fallback.

    When ``expand_seams=True`` and xatlas splits vertices along UV seams
    (providing ``original_vertex_map`` and ``indices``), expands the vertex
    and face buffers so every seam corner keeps its exact unaveraged UV.
    When ``expand_seams=False``, selects the primary chart UV per vertex
    instead of arithmetically averaging disconnected seam UVs (e.g. 0.01 & 0.99).
    """
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)

    if len(v) == 0 or len(f) == 0:
        return v, f, None

    # Try xatlas C++ binding first
    try:
        from synapse.bridge.aether_bridge import get_aether_geo

        _cg = get_aether_geo()
        if _cg is not None and hasattr(_cg, "unwrap_uv_atlas"):
            # For quad faces, fan-triangulate to pass to xatlas
            if f.shape[1] == 4:
                tri_faces = np.empty((len(f) * 2, 3), dtype=np.int32)
                tri_faces[0::2, 0] = f[:, 0]
                tri_faces[0::2, 1] = f[:, 1]
                tri_faces[0::2, 2] = f[:, 2]
                tri_faces[1::2, 0] = f[:, 0]
                tri_faces[1::2, 1] = f[:, 2]
                tri_faces[1::2, 2] = f[:, 3]
                raw_res = _cg.unwrap_uv_atlas(v, tri_faces)
            else:
                raw_res = _cg.unwrap_uv_atlas(v, f)

            original_vertex_map = None
            new_indices = None
            if isinstance(raw_res, dict):
                original_vertex_map = raw_res.get("original_vertex_map", None)
                new_indices = raw_res.get("indices", raw_res.get("faces", None))
                uvs = raw_res.get("uvs", raw_res.get("uv_coords", np.empty((0, 2))))
            else:
                uvs = raw_res
            uvs = np.asarray(uvs, dtype=np.float32)

            if uvs.shape[0] == len(v) and uvs.shape[1] == 2:
                return v, f, uvs
            elif (
                uvs.shape[0] > len(v)
                and uvs.shape[1] == 2
                and original_vertex_map is not None
            ):
                ovm = np.asarray(original_vertex_map, dtype=np.int64)
                if expand_seams and new_indices is not None:
                    idx_arr = np.asarray(new_indices, dtype=np.int32)
                    v_exp = v[ovm].astype(np.float32)
                    if f.shape[1] == 3 and idx_arr.size == len(f) * 3:
                        return v_exp, idx_arr.reshape(len(f), 3), uvs
                    if f.shape[1] == 4 and idx_arr.size == len(f) * 6:
                        tri_idx = idx_arr.reshape(len(f) * 2, 3)
                        quad_exp = np.empty((len(f), 4), dtype=np.int32)
                        quad_exp[:, 0] = tri_idx[0::2, 0]
                        quad_exp[:, 1] = tri_idx[0::2, 1]
                        quad_exp[:, 2] = tri_idx[0::2, 2]
                        quad_exp[:, 3] = tri_idx[1::2, 2]
                        return v_exp, quad_exp, uvs

                # Non-expanding path: preserve primary chart UV per vertex
                # (never average disconnected seam UVs like 0.01 and 0.99 to 0.5).
                primary_uvs = np.zeros((len(v), 2), dtype=np.float32)
                seen = np.zeros(len(v), dtype=bool)
                for new_idx in range(len(ovm)):
                    orig_idx = int(ovm[new_idx])
                    if 0 <= orig_idx < len(v) and not seen[orig_idx]:
                        primary_uvs[orig_idx] = uvs[new_idx]
                        seen[orig_idx] = True
                return v, f, primary_uvs
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Fallback: spherical projection unwrap
    centroid = np.mean(v, axis=0)
    centered = v - centroid
    r = np.linalg.norm(centered, axis=1, keepdims=True)
    r = np.clip(r, 1e-8, None)
    normalized = centered / r

    theta = np.arctan2(normalized[:, 0], normalized[:, 2])
    phi = np.arcsin(np.clip(normalized[:, 1], -1.0, 1.0))

    u = (theta / (2.0 * np.pi)) + 0.5
    v_coord = (phi / np.pi) + 0.5

    uvs = np.stack([u, v_coord], axis=-1).astype(np.float32)
    return v, f, uvs


def _generate_uvs(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray | None:
    """Generate UV coordinates for a mesh."""
    _, _, uvs = _unwrap_uv_atlas_with_seams(vertices, faces, expand_seams=False)
    return uvs


def _qem_decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate a triangle mesh using Quadric Error Metric (QEM).

    Iteratively collapses the edge with the lowest quadric error cost until the
    target face count is reached.  This prevents face-count explosion in
    downstream quad remeshing.

    Args:
        vertices: Vertex positions, shape (V, 3), float32/float64.
        faces: Triangle indices, shape (F, 3), int32/int64.
        target_faces: Desired number of output faces.

    Returns:
        (decimated_vertices, decimated_faces).
    """
    import heapq

    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64).copy()

    if len(f) <= target_faces or target_faces <= 0:
        return v.astype(np.float32), f.astype(np.int32)

    num_verts = len(v)
    num_faces = len(f)

    # Vectorized per-vertex symmetric 4x4 quadric accumulation (10 upper-tri floats/vert)
    va_arr = v[f[:, 0]]
    e1_arr = v[f[:, 1]] - va_arr
    e2_arr = v[f[:, 2]] - va_arr
    fn_arr = np.cross(e1_arr, e2_arr)
    fn_len = np.linalg.norm(fn_arr, axis=1)
    valid_fn = fn_len >= 1e-15
    fn_unit = np.zeros_like(fn_arr)
    fn_unit[valid_fn] = fn_arr[valid_fn] / fn_len[valid_fn, None]
    d_arr = -np.einsum("ij,ij->i", fn_unit, va_arr)
    d_arr[~valid_fn] = 0.0

    # Symmetric 4x4 outer product terms: (a, b, c, d) -> 10 unique entries
    pa, pb, pc, pd = fn_unit[:, 0], fn_unit[:, 1], fn_unit[:, 2], d_arr
    q_arr = np.zeros((num_verts, 10), dtype=np.float64)
    q_face = np.stack(
        [
            pa * pa,
            pa * pb,
            pa * pc,
            pa * pd,
            pb * pb,
            pb * pc,
            pb * pd,
            pc * pc,
            pc * pd,
            pd * pd,
        ],
        axis=1,
    )
    for col in range(3):
        np.add.at(q_arr, f[:, col], q_face)

    # Pure-Python scalar arrays avoid per-element NumPy allocation overhead in the collapse loop
    vx: list[float] = v[:, 0].tolist()
    vy: list[float] = v[:, 1].tolist()
    vz: list[float] = v[:, 2].tolist()
    f0: list[int] = f[:, 0].tolist()
    f1: list[int] = f[:, 1].tolist()
    f2: list[int] = f[:, 2].tolist()
    q_list: list[list[float]] = q_arr.tolist()

    # Build vertex -> incident faces adjacency and initial edges
    alive_faces: set[int] = set(range(num_faces))
    vertex_alias = list(range(num_verts))
    vert_to_faces: list[set[int]] = [set() for _ in range(num_verts)]
    edge_set: set[tuple[int, int]] = set()
    for fi in range(num_faces):
        a, b, c = f0[fi], f1[fi], f2[fi]
        vert_to_faces[a].add(fi)
        vert_to_faces[b].add(fi)
        vert_to_faces[c].add(fi)
        edge_set.add((a, b) if a < b else (b, a))
        edge_set.add((b, c) if b < c else (c, b))
        edge_set.add((c, a) if c < a else (a, c))

    def _find(x: int) -> int:
        while vertex_alias[x] != x:
            vertex_alias[x] = vertex_alias[vertex_alias[x]]
            x = vertex_alias[x]
        return x

    def _eval_quadric(
        q00: float,
        q01: float,
        q02: float,
        q03: float,
        q11: float,
        q12: float,
        q13: float,
        q22: float,
        q23: float,
        q33: float,
        px: float,
        py: float,
        pz: float,
    ) -> float:
        return (
            q00 * px * px
            + 2.0 * q01 * px * py
            + 2.0 * q02 * px * pz
            + 2.0 * q03 * px
            + q11 * py * py
            + 2.0 * q12 * py * pz
            + 2.0 * q13 * py
            + q22 * pz * pz
            + 2.0 * q23 * pz
            + q33
        )

    def _optimal_collapse(a: int, b: int) -> tuple[float, float, float, float]:
        """Solve Garland-Heckbert 3x3 optimal collapse position (cost, x, y, z).

        Falls back to minimum-error endpoint/midpoint when the 3x3 quadric
        upper block is ill-conditioned or the unconstrained solution strays
        beyond 1.5x the edge length from the midpoint.
        """
        qa = q_list[a]
        qb = q_list[b]
        q00 = qa[0] + qb[0]
        q01 = qa[1] + qb[1]
        q02 = qa[2] + qb[2]
        q03 = qa[3] + qb[3]
        q11 = qa[4] + qb[4]
        q12 = qa[5] + qb[5]
        q13 = qa[6] + qb[6]
        q22 = qa[7] + qb[7]
        q23 = qa[8] + qb[8]
        q33 = qa[9] + qb[9]

        ax, ay, az = vx[a], vy[a], vz[a]
        bx, by, bz = vx[b], vy[b], vz[b]
        mx = 0.5 * (ax + bx)
        my = 0.5 * (ay + by)
        mz = 0.5 * (az + bz)

        # 3x3 symmetric cofactors for A * v* = -[q03, q13, q23]^T
        c00 = q11 * q22 - q12 * q12
        c01 = q12 * q02 - q01 * q22
        c02 = q01 * q12 - q11 * q02
        det = q00 * c00 + q01 * c01 + q02 * c02
        trace = q00 + q11 + q22
        if abs(det) > 1e-10 * (trace * trace * trace + 1e-18):
            c11 = q00 * q22 - q02 * q02
            c12 = q01 * q02 - q00 * q12
            c22 = q00 * q11 - q01 * q01
            inv_det = -1.0 / det
            ox = (c00 * q03 + c01 * q13 + c02 * q23) * inv_det
            oy = (c01 * q03 + c11 * q13 + c12 * q23) * inv_det
            oz = (c02 * q03 + c12 * q13 + c22 * q23) * inv_det
            ex, ey, ez = bx - ax, by - ay, bz - az
            edge_len_sq = ex * ex + ey * ey + ez * ez
            dx, dy, dz = ox - mx, oy - my, oz - mz
            if dx * dx + dy * dy + dz * dz <= 2.25 * edge_len_sq + 1e-12:
                ocost = _eval_quadric(q00, q01, q02, q03, q11, q12, q13, q22, q23, q33, ox, oy, oz)
                return max(0.0, ocost), ox, oy, oz

        cost_m = _eval_quadric(q00, q01, q02, q03, q11, q12, q13, q22, q23, q33, mx, my, mz)
        cost_a = _eval_quadric(q00, q01, q02, q03, q11, q12, q13, q22, q23, q33, ax, ay, az)
        cost_b = _eval_quadric(q00, q01, q02, q03, q11, q12, q13, q22, q23, q33, bx, by, bz)
        if cost_a <= cost_m and cost_a <= cost_b:
            return max(0.0, cost_a), ax, ay, az
        if cost_b <= cost_m:
            return max(0.0, cost_b), bx, by, bz
        return max(0.0, cost_m), mx, my, mz

    def _edge_cost(a: int, b: int) -> float:
        return _optimal_collapse(a, b)[0]

    def _is_boundary_vertex(vr: int) -> bool:
        edge_counts: dict[tuple[int, int], int] = {}
        for fi in vert_to_faces[vr]:
            if fi not in alive_faces:
                continue
            r0, r1, r2 = _find(f0[fi]), _find(f1[fi]), _find(f2[fi])
            for u_i, w_i in ((r0, r1), (r1, r2), (r2, r0)):
                if u_i == vr or w_i == vr:
                    ek = (u_i, w_i) if u_i < w_i else (w_i, u_i)
                    edge_counts[ek] = edge_counts.get(ek, 0) + 1
        return any(cnt == 1 for cnt in edge_counts.values())

    def _link_and_normal_safe(va: int, vb: int, nx_pos: float, ny_pos: float, nz_pos: float) -> bool:
        """Dey-Edelsbrunner Link Condition + boundary pinch + normal-flip guard."""
        inc_a = vert_to_faces[va]
        inc_b = vert_to_faces[vb]
        shared_faces = inc_a & inc_b
        n_shared = len(shared_faces)
        if n_shared == 0 or n_shared > 2:
            return False

        nb_a: set[int] = set()
        for fi in inc_a:
            for r in (_find(f0[fi]), _find(f1[fi]), _find(f2[fi])):
                if r != va and r != vb:
                    nb_a.add(r)

        nb_b: set[int] = set()
        for fi in inc_b:
            for r in (_find(f0[fi]), _find(f1[fi]), _find(f2[fi])):
                if r != va and r != vb:
                    nb_b.add(r)

        # Link Condition: common neighbors must match the opposite vertices of shared faces
        common_nb = nb_a & nb_b
        if len(common_nb) != n_shared:
            return False

        # Reject collapsing an edge of a 3-vertex cap / tetrahedron (degree <= 3)
        for w_nb in common_nb:
            if len(vert_to_faces[w_nb]) <= 3:
                return False

        # Boundary pinch guard: interior edge connecting two boundary vertices would pinch
        if n_shared == 2 and _is_boundary_vertex(va) and _is_boundary_vertex(vb):
            return False

        surviving = (inc_a | inc_b) - shared_faces
        existing_triples: set[tuple[int, int, int]] = set()
        seen_triples: set[tuple[int, int, int]] = set()
        for w_nb in common_nb:
            for fi_w in vert_to_faces[w_nb]:
                if fi_w in alive_faces and fi_w not in shared_faces and fi_w not in surviving:
                    a_w, b_w, c_w = _find(f0[fi_w]), _find(f1[fi_w]), _find(f2[fi_w])
                    if a_w > b_w:
                        a_w, b_w = b_w, a_w
                    if b_w > c_w:
                        b_w, c_w = c_w, b_w
                    if a_w > b_w:
                        a_w, b_w = b_w, a_w
                    existing_triples.add((a_w, b_w, c_w))

        for fi in surviving:
            r0, r1, r2 = _find(f0[fi]), _find(f1[fi]), _find(f2[fi])
            nr0 = va if r0 == vb else r0
            nr1 = va if r1 == vb else r1
            nr2 = va if r2 == vb else r2
            if nr0 in (nr1, nr2) or nr1 == nr2:
                return False
            a_k, b_k, c_k = nr0, nr1, nr2
            if a_k > b_k:
                a_k, b_k = b_k, a_k
            if b_k > c_k:
                b_k, c_k = c_k, b_k
            if a_k > b_k:
                a_k, b_k = b_k, a_k
            key = (a_k, b_k, c_k)
            if key in seen_triples or key in existing_triples:
                return False
            seen_triples.add(key)

            # Scalar cross products for normal-flip guard (zero NumPy overhead)
            ox0, oy0, oz0 = vx[r0], vy[r0], vz[r0]
            oe1x, oe1y, oe1z = vx[r1] - ox0, vy[r1] - oy0, vz[r1] - oz0
            oe2x, oe2y, oe2z = vx[r2] - ox0, vy[r2] - oy0, vz[r2] - oz0
            onx = oe1y * oe2z - oe1z * oe2y
            ony = oe1z * oe2x - oe1x * oe2z
            onz = oe1x * oe2y - oe1y * oe2x
            l_old_sq = onx * onx + ony * ony + onz * onz

            if nr0 == va:
                px0, py0, pz0 = nx_pos, ny_pos, nz_pos
            else:
                px0, py0, pz0 = vx[nr0], vy[nr0], vz[nr0]
            if nr1 == va:
                px1, py1, pz1 = nx_pos, ny_pos, nz_pos
            else:
                px1, py1, pz1 = vx[nr1], vy[nr1], vz[nr1]
            if nr2 == va:
                px2, py2, pz2 = nx_pos, ny_pos, nz_pos
            else:
                px2, py2, pz2 = vx[nr2], vy[nr2], vz[nr2]

            ne1x, ne1y, ne1z = px1 - px0, py1 - py0, pz1 - pz0
            ne2x, ne2y, ne2z = px2 - px0, py2 - py0, pz2 - pz0
            nnx = ne1y * ne2z - ne1z * ne2y
            nny = ne1z * ne2x - ne1x * ne2z
            nnz = ne1x * ne2y - ne1y * ne2x
            l_new_sq = nnx * nnx + nny * nny + nnz * nnz
            if l_new_sq < 1e-24:
                return False
            if l_old_sq > 1e-24:
                dot = onx * nnx + ony * nny + onz * nnz
                if dot <= 0.0 or (dot * dot) <= 0.01 * l_old_sq * l_new_sq:
                    return False

        return True

    # Build initial edge heap
    heap: list[tuple[float, int, int, int]] = []
    seq_counter = 0
    for a, b in edge_set:
        heap.append((_edge_cost(a, b), seq_counter, a, b))
        seq_counter += 1
    heapq.heapify(heap)

    current_face_count = len(alive_faces)

    while current_face_count > target_faces and heap:
        _, _, va_raw, vb_raw = heapq.heappop(heap)

        va = _find(va_raw)
        vb = _find(vb_raw)
        if va == vb:
            continue

        _, nx_pos, ny_pos, nz_pos = _optimal_collapse(va, vb)
        if not _link_and_normal_safe(va, vb, nx_pos, ny_pos, nz_pos):
            continue

        # Commit collapse: merge vb into va in O(deg)
        vx[va], vy[va], vz[va] = nx_pos, ny_pos, nz_pos
        qa = q_list[va]
        qb = q_list[vb]
        for qi in range(10):
            qa[qi] += qb[qi]
        vertex_alias[vb] = va

        affected = list(vert_to_faces[va] | vert_to_faces[vb])
        neighbors: set[int] = set()
        for fi in affected:
            if fi not in alive_faces:
                continue
            r0, r1, r2 = _find(f0[fi]), _find(f1[fi]), _find(f2[fi])
            f0[fi], f1[fi], f2[fi] = r0, r1, r2
            if r0 in (r1, r2) or r1 == r2:
                alive_faces.discard(fi)
                vert_to_faces[r0].discard(fi)
                vert_to_faces[r1].discard(fi)
                vert_to_faces[r2].discard(fi)
                vert_to_faces[va].discard(fi)
                vert_to_faces[vb].discard(fi)
            else:
                vert_to_faces[va].add(fi)
                if r0 != va:
                    neighbors.add(r0)
                if r1 != va:
                    neighbors.add(r1)
                if r2 != va:
                    neighbors.add(r2)
        vert_to_faces[vb].clear()
        current_face_count = len(alive_faces)

        for nb in neighbors:
            heapq.heappush(heap, (_edge_cost(va, nb), seq_counter, va, nb))
            seq_counter += 1

    # Compact output
    if not alive_faces:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    kept_faces = np.array(
        [[_find(f0[fi]), _find(f1[fi]), _find(f2[fi])] for fi in sorted(alive_faces)],
        dtype=np.int64,
    )
    v_out = np.column_stack([vx, vy, vz])
    unique_verts, inverse = np.unique(kept_faces.ravel(), return_inverse=True)
    new_verts = v_out[unique_verts]
    new_faces = inverse.reshape(kept_faces.shape)

    return new_verts.astype(np.float32), new_faces.astype(np.int32)


def quad_remesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = 0,
    *,
    expand_uv_seams: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Remesh a triangle mesh into a quad-dominant or quad mesh.

    Pairs adjacent coplanar triangles sharing an edge based on dihedral angle
    and aspect ratio, removing interior diagonals to form quad faces.
    After remeshing, generates UV coordinates via xatlas binding
    (``synapse._c_geometry.unwrap_uv_atlas``) or projection unwrap fallback.

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Triangle indices, shape (F, 3).
        target_faces: Optional target face count (0 = maximize quads).

    Returns:
        Tuple of (new_vertices, new_faces, uvs).
        uvs is (V_new, 2) float32 or None if UV generation fails.
    """
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)

    if len(v) == 0 or len(f) == 0:
        return v, f, None

    if f.shape[1] == 4:
        v, f, uvs = _unwrap_uv_atlas_with_seams(v, f, expand_seams=expand_uv_seams)
        return v, f, uvs  # Already quad mesh

    if f.shape[1] != 3:
        return v, f, None  # Unsupported polygon type

    # QEM decimation: if target_faces is specified and the input face count
    # is significantly larger, decimate first to prevent face count explosion
    # during quad pairing + subdivision.
    if target_faces > 0 and len(f) > target_faces * 2:
        v, f = _qem_decimate(v, f, target_faces * 2)
        if len(f) == 0:
            return v, f, None

    num_faces = len(f)
    # Map each undirected edge to face indices
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_idx, tri in enumerate(f):
        for i in range(3):
            e = (min(tri[i], tri[(i + 1) % 3]), max(tri[i], tri[(i + 1) % 3]))
            edge_to_faces[e].append(face_idx)

    # Compute face normals
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    fn_lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
    fn_lens[fn_lens < 1e-12] = 1.0
    face_normals = face_normals / fn_lens

    # Find candidate pairs sharing an edge
    candidates = []
    for edge, incident in edge_to_faces.items():
        if len(incident) == 2:
            f1, f2 = incident[0], incident[1]
            n1 = face_normals[f1]
            n2 = face_normals[f2]
            dot = float(np.dot(n1, n2))
            if dot > 0.5:  # Dihedral angle within ~60 degrees
                candidates.append((dot, edge, f1, f2))

    # Sort candidates by highest planarity (largest dot product)
    candidates.sort(key=lambda x: x[0], reverse=True)

    paired_faces: set[int] = set()
    quad_faces: list[list[int]] = []

    for dot, edge, f1, f2 in candidates:
        if f1 in paired_faces or f2 in paired_faces:
            continue
        if target_faces > 0 and len(quad_faces) >= target_faces:
            break

        # Triangles f1 and f2 share 'edge' = (u, w)
        u, w = edge
        tri1 = list(f[f1])
        tri2 = list(f[f2])

        # Opposite vertices
        opp1 = [idx for idx in tri1 if idx != u and idx != w][0]
        opp2 = [idx for idx in tri2 if idx != u and idx != w][0]

        # Order quad vertices cyclically: opp1 -> u -> opp2 -> w
        # Check orientation with respect to tri1
        idx_u = tri1.index(u)
        next_after_u = tri1[(idx_u + 1) % 3]
        if next_after_u == w:
            quad = [opp1, u, opp2, w]
        else:
            quad = [opp1, w, opp2, u]

        # Convexity and orientation check
        qp = v[quad]
        n_ref = face_normals[f1]
        c0 = np.cross(qp[1] - qp[0], qp[2] - qp[1])
        c1 = np.cross(qp[2] - qp[1], qp[3] - qp[2])
        c2 = np.cross(qp[3] - qp[2], qp[0] - qp[3])
        c3 = np.cross(qp[0] - qp[3], qp[1] - qp[0])
        dots = [
            float(np.dot(c0, n_ref)),
            float(np.dot(c1, n_ref)),
            float(np.dot(c2, n_ref)),
            float(np.dot(c3, n_ref)),
        ]
        if min(dots) <= 1e-6:
            # Non-convex or butterfly self-intersecting quad — do not pair
            continue

        quad_faces.append(quad)
        paired_faces.add(f1)
        paired_faces.add(f2)

    # Remaining unpaired triangles
    unpaired_triangles: list[list[int]] = []
    for idx in range(num_faces):
        if idx not in paired_faces:
            unpaired_triangles.append(list(f[idx]))

    # If all paired or partially paired
    if quad_faces and not unpaired_triangles:
        new_faces = np.array(quad_faces, dtype=np.int32)
    elif unpaired_triangles:
        # Mixed tri/quad mesh: convert remaining triangles into quads via
        # Catmull-Clark midpoint subdivision — each triangle [a, b, c] is
        # split into 3 quads by inserting edge midpoints and a face centroid.
        # Shared edge midpoints avoid cracks and T-junctions between adjacent faces.
        extra_verts: list[np.ndarray] = []
        subdivided_quads: list[list[int]] = []
        vert_offset = len(v)
        edge_midpoint_map: dict[tuple[int, int], int] = {}

        def _get_midpoint(idx_a: int, idx_b: int) -> int:
            edge_key = (min(idx_a, idx_b), max(idx_a, idx_b))
            if edge_key not in edge_midpoint_map:
                mid_pos = (v[idx_a] + v[idx_b]) * 0.5
                mid_idx = vert_offset + len(extra_verts)
                extra_verts.append(mid_pos)
                edge_midpoint_map[edge_key] = mid_idx
            return edge_midpoint_map[edge_key]

        # Collect all edges touched by unpaired triangles so we can detect
        # T-junction exposure on adjacent quad faces.
        subdivided_edges: set[tuple[int, int]] = set()
        for tri in unpaired_triangles:
            a, b, c = tri[0], tri[1], tri[2]
            subdivided_edges.add((min(a, b), max(a, b)))
            subdivided_edges.add((min(b, c), max(b, c)))
            subdivided_edges.add((min(c, a), max(c, a)))

        for tri in unpaired_triangles:
            a, b, c = tri[0], tri[1], tri[2]
            idx_mid_ab = _get_midpoint(a, b)
            idx_mid_bc = _get_midpoint(b, c)
            idx_mid_ca = _get_midpoint(c, a)

            centroid = (v[a] + v[b] + v[c]) / 3.0
            idx_centroid = vert_offset + len(extra_verts)
            extra_verts.append(centroid)

            # Three quads per triangle
            subdivided_quads.append([a, idx_mid_ab, idx_centroid, idx_mid_ca])
            subdivided_quads.append([b, idx_mid_bc, idx_centroid, idx_mid_ab])
            subdivided_quads.append([c, idx_mid_ca, idx_centroid, idx_mid_bc])

        # Also subdivide any quad that shares an edge with a subdivided face
        # (one-step Catmull-Clark quad subdivision) to prevent T-junctions.
        # Propagate: newly subdivided quads expose new midpoint vertices on
        # their edges, so adjacent quads must also be subdivided until no
        # T-junctions remain.
        remaining_quads = list(quad_faces)
        converged = False
        while not converged:
            converged = True
            safe_quads_round: list[list[int]] = []
            for quad in remaining_quads:
                q0, q1, q2, q3 = quad[0], quad[1], quad[2], quad[3]
                edges = [
                    (min(q0, q1), max(q0, q1)),
                    (min(q1, q2), max(q1, q2)),
                    (min(q2, q3), max(q2, q3)),
                    (min(q3, q0), max(q3, q0)),
                ]
                shared = any(e in subdivided_edges for e in edges)
                if shared:
                    # Subdivide quad into 4 sub-quads via edge midpoints + face centre
                    m01 = _get_midpoint(q0, q1)
                    m12 = _get_midpoint(q1, q2)
                    m23 = _get_midpoint(q2, q3)
                    m30 = _get_midpoint(q3, q0)
                    face_centre = (v[q0] + v[q1] + v[q2] + v[q3]) * 0.25
                    fc_idx = vert_offset + len(extra_verts)
                    extra_verts.append(face_centre)
                    subdivided_quads.append([q0, m01, fc_idx, m30])
                    subdivided_quads.append([q1, m12, fc_idx, m01])
                    subdivided_quads.append([q2, m23, fc_idx, m12])
                    subdivided_quads.append([q3, m30, fc_idx, m23])
                    # Register newly split edges for propagation
                    for e in edges:
                        if e not in subdivided_edges:
                            subdivided_edges.add(e)
                            converged = False
                else:
                    safe_quads_round.append(quad)
            remaining_quads = safe_quads_round
        safe_quads = remaining_quads

        if extra_verts:
            v = np.vstack([v, np.array(extra_verts, dtype=np.float32)])

        all_quads = safe_quads + subdivided_quads
        new_faces = np.array(all_quads, dtype=np.int32)
    else:
        new_faces = f

    # Generate UV coordinates via xatlas binding or projection unwrap fallback
    v, new_faces, uvs = _unwrap_uv_atlas_with_seams(
        v, new_faces, expand_seams=expand_uv_seams
    )
    return v, new_faces, uvs


def check_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_area: float = 1e-8,
    max_aspect_ratio: float = 100.0,
) -> np.ndarray:
    """Detect degenerate (near-zero-area or poorly-shaped) triangle faces.

    A face is considered degenerate if its area falls below *min_area* or its
    aspect ratio (longest-edge² / (2 · area), a circumradius-to-inradius
    approximation) exceeds *max_aspect_ratio*.

    The implementation is fully vectorized over faces — no Python loops.

    Args:
        vertices: Vertex positions, shape ``(V, 3)``.
        faces: Triangle indices, shape ``(F, 3)``.
        min_area: Minimum triangle area threshold.
        max_aspect_ratio: Maximum permissible aspect ratio.

    Returns:
        Sorted 1-D integer array of face indices flagged as degenerate.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if (
        v.ndim != 2
        or v.shape[1] != 3
        or f.ndim != 2
        or f.shape[1] < 3
        or len(v) == 0
        or len(f) == 0
    ):
        return np.array([], dtype=np.intp)

    vpf = f.shape[1]  # vertices per face

    # For N-gon faces, check bounds and finiteness of ALL vpf vertices,
    # then fan-triangulate: for k=1..vpf-2, check triangle (v0, v[k], v[k+1]).
    # A face is degenerate if ANY sub-triangle is degenerate.
    num_verts = len(v)
    degenerate_indices: list[int] = []

    for fi in range(len(f)):
        face = f[fi]

        # Check bounds for ALL vpf vertices.
        if np.any(face < 0) or np.any(face >= num_verts):
            degenerate_indices.append(fi)
            continue

        # Check finiteness of ALL vpf vertices.
        face_verts = v[face]
        if not np.all(np.isfinite(face_verts)):
            degenerate_indices.append(fi)
            continue

        # Fan-triangulate and check each sub-triangle.
        p0 = face_verts[0]
        is_degen = False
        for k in range(1, vpf - 1):
            pk = face_verts[k]
            pk1 = face_verts[k + 1]

            e0 = pk - p0
            e1 = pk1 - p0
            e2 = pk1 - pk

            cross_vec = np.cross(e0, e1)
            area = 0.5 * float(np.linalg.norm(cross_vec))

            if not np.isfinite(area) or area < min_area:
                is_degen = True
                break

            len_sq = [
                float(np.dot(e0, e0)),
                float(np.dot(e1, e1)),
                float(np.dot(e2, e2)),
            ]
            longest_sq = max(len_sq)
            aspect = longest_sq / (2.0 * area)
            if not np.isfinite(aspect) or aspect > max_aspect_ratio:
                is_degen = True
                break

        if is_degen:
            degenerate_indices.append(fi)

    return np.array(degenerate_indices, dtype=np.intp)


def check_self_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    verts_per_face: int = 3,
) -> list[tuple[int, int]]:
    """Find all pairs of non-adjacent triangles that geometrically intersect.

    Uses spatial hashing (uniform voxel grid) as a broad-phase filter and an
    exact plane-distance + coplanar 2D SAT + Möller–Trumbore segment-piercing
    narrow phase matching ``csrc/geometry/src/mesh_checks.cc`` 100%.
    Adjacent face pairs (sharing ≥ 1 vertex) are skipped.

    This is the **Python Oracle** reference implementation — correctness over
    speed.

    Args:
        vertices: Vertex positions, shape ``(V, 3)``.
        faces: Triangle indices, shape ``(F, 3)``.

    Returns:
        Sorted list of ``(face_i, face_j)`` pairs (``face_i < face_j``)
        that geometrically intersect.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if (
        v.ndim != 2
        or v.shape[1] != 3
        or f.ndim != 2
        or f.shape[1] < 3
        or len(v) == 0
        or len(f) <= 1
    ):
        return []

    num_faces = len(f)
    num_verts = len(v)

    def _segment_pierces_triangle(
        origin: np.ndarray,
        direction: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
    ) -> bool:
        parallel_eps = 1e-12
        bary_eps = 1e-9

        e1 = v1 - v0
        e2 = v2 - v0
        h = np.cross(direction, e2)
        a = float(np.dot(e1, h))
        if abs(a) < parallel_eps:
            return False

        inv_a = 1.0 / a
        s = origin - v0
        u = inv_a * float(np.dot(s, h))
        if u < -bary_eps or u > 1.0 + bary_eps:
            return False

        q = np.cross(s, e1)
        bary_v = inv_a * float(np.dot(direction, q))
        if bary_v < -bary_eps or u + bary_v > 1.0 + bary_eps:
            return False

        t = inv_a * float(np.dot(e2, q))
        return -bary_eps <= t <= 1.0 + bary_eps

    def _tri_tri_intersect(
        t1: np.ndarray,
        t2: np.ndarray,
    ) -> bool:
        """Test whether two non-adjacent triangles intersect in 3D space."""
        area_eps = 1e-12
        plane_eps = 1e-9

        n1 = np.cross(t1[1] - t1[0], t1[2] - t1[0])
        len1 = float(np.linalg.norm(n1))
        n2 = np.cross(t2[1] - t2[0], t2[2] - t2[0])
        len2 = float(np.linalg.norm(n2))
        if len1 < area_eps or len2 < area_eps:
            return False

        unit1 = n1 / len1
        unit2 = n2 / len2

        # Signed distances of t2 vertices to t1's plane
        db = (t2 - t1[0]) @ unit1
        if float(db.min()) > plane_eps or float(db.max()) < -plane_eps:
            return False

        # Signed distances of t1 vertices to t2's plane
        da = (t1 - t2[0]) @ unit2
        if float(da.min()) > plane_eps or float(da.max()) < -plane_eps:
            return False

        # Coplanar case: 2D SAT on the 6 in-plane edge normals
        if float(np.max(np.abs(db))) <= plane_eps and float(np.max(np.abs(da))) <= plane_eps:
            edges = (
                t1[1] - t1[0],
                t1[2] - t1[1],
                t1[0] - t1[2],
                t2[1] - t2[0],
                t2[2] - t2[1],
                t2[0] - t2[2],
            )
            for edge in edges:
                axis = np.cross(unit1, edge)
                axis_len = float(np.linalg.norm(axis))
                if axis_len < area_eps:
                    continue
                unit_axis = axis / axis_len
                p1 = t1 @ unit_axis
                p2 = t2 @ unit_axis
                if (
                    float(np.max(p1)) < float(np.min(p2)) - plane_eps
                    or float(np.max(p2)) < float(np.min(p1)) - plane_eps
                ):
                    return False
            return True

        # Non-coplanar case: bidirectional segment-triangle piercing
        a0, a1, a2 = t1[0], t1[1], t1[2]
        b0, b1, b2 = t2[0], t2[1], t2[2]
        return (
            _segment_pierces_triangle(a0, a1 - a0, b0, b1, b2)
            or _segment_pierces_triangle(a1, a2 - a1, b0, b1, b2)
            or _segment_pierces_triangle(a2, a0 - a2, b0, b1, b2)
            or _segment_pierces_triangle(b0, b1 - b0, a0, a1, a2)
            or _segment_pierces_triangle(b1, b2 - b1, a0, a1, a2)
            or _segment_pierces_triangle(b2, b0 - b2, a0, a1, a2)
        )

    # ------------------------------------------------------------------
    # Validate per-face indices & build broad-phase spatial hash
    # ------------------------------------------------------------------
    vpf = f.shape[1]  # vertices per face

    # Validate ALL vpf vertex indices per face and build AABBs over all vertices.
    valid_face = np.ones(num_faces, dtype=bool)
    face_all_verts = np.empty((num_faces, vpf, 3), dtype=np.float64)
    for fi_idx in range(num_faces):
        face = f[fi_idx]
        if np.any(face < 0) or np.any(face >= num_verts):
            valid_face[fi_idx] = False
            face_all_verts[fi_idx] = 0.0
        else:
            fv = v[face]
            if not np.all(np.isfinite(fv)):
                valid_face[fi_idx] = False
                face_all_verts[fi_idx] = 0.0
            else:
                face_all_verts[fi_idx] = fv

    valid_ids = np.nonzero(valid_face)[0]
    if len(valid_ids) <= 1:
        return []

    # AABB over all vertices per face (not just first 3)
    tri_min = face_all_verts.min(axis=1)
    tri_max = face_all_verts.max(axis=1)

    global_min = tri_min[valid_ids].min(axis=0) - 1e-6
    global_max = tri_max[valid_ids].max(axis=0) + 1e-6
    extent = np.maximum(global_max - global_min, 1e-6)

    num_cells = max(min(int(round(len(valid_ids) ** (1.0 / 3.0))), 256), 4)
    cell_size = extent / num_cells

    _MAX_CELLS_PER_FACE = 512  # prevent memory explosion for large faces

    large_faces: list[int] = []
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for fi_np in valid_ids:
        fi = int(fi_np)
        lo = np.clip(
            np.floor((tri_min[fi] - global_min) / cell_size).astype(int),
            0,
            num_cells - 1,
        )
        hi = np.clip(
            np.floor((tri_max[fi] - global_min) / cell_size).astype(int),
            0,
            num_cells - 1,
        )
        total_cells = int(
            (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1)
        )
        if total_cells > _MAX_CELLS_PER_FACE:
            large_faces.append(fi)
            continue

        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    grid[(ix, iy, iz)].append(fi)

    # ------------------------------------------------------------------
    # Narrow phase
    # ------------------------------------------------------------------
    face_vertex_sets = [set(int(x) for x in f[i]) for i in range(num_faces)]
    tested: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    aabb_eps = 1e-9

    def _test_pair(fi: int, fj: int) -> None:
        # Skip adjacent faces (sharing ≥ 1 vertex)
        if face_vertex_sets[fi] & face_vertex_sets[fj]:
            return

        # Exact AABB overlap pre-filter
        if np.any(tri_min[fi] > tri_max[fj] + aabb_eps) or np.any(
            tri_max[fi] + aabb_eps < tri_min[fj]
        ):
            return

        # Skip UV / hard-normal split-vertex seam neighbors.
        # Check ALL vpf vertices per face, not just 3.
        verts_fi = face_all_verts[fi]  # (vpf, 3)
        verts_fj = face_all_verts[fj]  # (vpf, 3)
        diff_sq = np.sum(
            (verts_fi[:, None, :] - verts_fj[None, :, :]) ** 2,
            axis=2,
        )
        shared_pos_i = int(np.count_nonzero(np.any(diff_sq <= 1e-18, axis=1)))
        shared_pos_j = int(np.count_nonzero(np.any(diff_sq <= 1e-18, axis=0)))
        shared_pos = max(shared_pos_i, shared_pos_j)
        if 1 <= shared_pos <= vpf - 1:
            return

        # Fan-triangulate both faces and test ALL sub-triangle pairs.
        found = False
        p0_a = verts_fi[0]
        p0_b = verts_fj[0]
        for ka in range(1, vpf - 1):
            if found:
                break
            tri_a = np.array([p0_a, verts_fi[ka], verts_fi[ka + 1]])
            for kb in range(1, vpf - 1):
                tri_b = np.array([p0_b, verts_fj[kb], verts_fj[kb + 1]])
                if _tri_tri_intersect(tri_a, tri_b):
                    found = True
                    break
        if found:
            pair = (min(fi, fj), max(fi, fj))
            result.append(pair)

    for cell_faces in grid.values():
        for a_idx in range(len(cell_faces)):
            for b_idx in range(a_idx + 1, len(cell_faces)):
                fi = cell_faces[a_idx]
                fj = cell_faces[b_idx]
                pair = (min(fi, fj), max(fi, fj))
                if pair in tested:
                    continue
                tested.add(pair)
                _test_pair(fi, fj)

    # Test large faces against all other valid faces via vectorized AABB overlap
    if large_faces:
        for fi in large_faces:
            overlap_mask = ~(
                (tri_min[fi] > tri_max[valid_ids] + aabb_eps).any(axis=1)
                | (tri_max[fi] + aabb_eps < tri_min[valid_ids]).any(axis=1)
            )
            cand_fjs = valid_ids[overlap_mask]
            for fj_np in cand_fjs:
                fj = int(fj_np)
                if fi == fj:
                    continue
                pair = (min(fi, fj), max(fi, fj))
                if pair in tested:
                    continue
                tested.add(pair)
                _test_pair(fi, fj)

    result.sort()
    return result


def find_connected_components(
    faces: np.ndarray,
    verts_per_face: int = 3,
    num_vertices: int = 0,
) -> list[np.ndarray]:
    """Partition mesh faces into connected components via shared edges.

    Uses a Union-Find (disjoint-set) data structure keyed on face indices
    where edges provide the union relation.

    Args:
        faces: Triangle (or quad) indices, shape ``(F, K)`` or 1-D flat buffer.
        verts_per_face: Number of vertices per polygon (default 3).
        num_vertices: Total vertex count hint (default 0 = auto-infer).

    Returns:
        List of 1-D arrays, each containing the sorted face indices of one
        connected component.  Components are ordered largest-first.
    """
    f = np.asarray(faces, dtype=np.int64)

    if len(f) == 0:
        return []

    if f.ndim == 1:
        if verts_per_face < 3:
            raise ValueError(f"verts_per_face must be >= 3, got {verts_per_face}")
        if len(f) % verts_per_face != 0:
            raise ValueError(
                f"Flat faces buffer length ({len(f)}) not divisible by verts_per_face ({verts_per_face})"
            )
        f = f.reshape(-1, verts_per_face)
    elif f.ndim == 2:
        pass
    else:
        raise ValueError(f"faces must be 1-D or 2-D, got {f.ndim}-D")

    num_faces = len(f)

    # Union-Find with path compression and union by rank
    parent = list(range(num_faces))
    rank = [0] * num_faces

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Build edge → face mapping, then union faces sharing an edge
    edge_to_face: dict[tuple[int, int], int] = {}
    num_verts_per_face = f.shape[1]
    for fi in range(num_faces):
        poly = f[fi]
        for k in range(num_verts_per_face):
            u, w = int(poly[k]), int(poly[(k + 1) % num_verts_per_face])
            edge_key = (min(u, w), max(u, w))
            if edge_key in edge_to_face:
                _union(fi, edge_to_face[edge_key])
            else:
                edge_to_face[edge_key] = fi

    # Collect components
    components: dict[int, list[int]] = defaultdict(list)
    for fi in range(num_faces):
        components[_find(fi)].append(fi)

    # Sort: largest component first, tie-break on smallest face index
    result = sorted(
        (np.array(indices, dtype=np.intp) for indices in components.values()),
        key=lambda arr: (-len(arr), int(arr[0])),
    )
    return result


def prune_floating_islands(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_face_ratio: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove small disconnected mesh islands.

    Connected components whose face count is strictly less than
    ``min_face_ratio * total_faces`` are removed.  Vertex indices in
    the returned face array are compacted so that only referenced
    vertices remain.

    Args:
        vertices: Vertex positions, shape ``(V, 3)``.
        faces: Face indices, shape ``(F, K)``.
        min_face_ratio: Minimum fraction of total faces a component must
            have to be kept.

    Returns:
        Tuple of ``(cleaned_vertices, cleaned_faces, kept_face_mask)``.

        - *cleaned_vertices*: shape ``(V', 3)`` — only referenced vertices.
        - *cleaned_faces*: shape ``(F', K)`` — remapped face indices.
        - *kept_face_mask*: boolean array, shape ``(F,)`` — ``True`` for
          faces that were kept.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if len(v) == 0 or len(f) == 0:
        return (
            np.empty((0, 3), dtype=v.dtype),
            np.empty((0, f.shape[1] if len(f) else 3), dtype=f.dtype),
            np.zeros(len(f), dtype=bool),
        )

    components = find_connected_components(f)
    total_faces = len(f)
    threshold = min_face_ratio * total_faces

    kept_face_mask = np.zeros(total_faces, dtype=bool)
    for comp in components:
        if len(comp) >= threshold:
            kept_face_mask[comp] = True

    kept_faces = f[kept_face_mask]

    if len(kept_faces) == 0:
        return (
            np.empty((0, 3), dtype=v.dtype),
            np.empty((0, f.shape[1]), dtype=f.dtype),
            kept_face_mask,
        )

    # Remap vertices: compact to only those referenced by kept faces
    unique_verts, inverse = np.unique(kept_faces.ravel(), return_inverse=True)
    cleaned_vertices = v[unique_verts]
    cleaned_faces = inverse.reshape(kept_faces.shape)

    return cleaned_vertices, cleaned_faces, kept_face_mask


def validate_recon_mesh_manifold(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    backend: str = "native",
) -> bool:
    """Check whether a mesh is 2-manifold using native C++ or Python oracle.

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Triangle indices, shape (F, 3).
        backend: Execution backend — "native" (aether_geo) or "python".

    Returns:
        True if the mesh is 2-manifold.
    """
    if backend == "native":
        from synapse.bridge.aether_bridge import get_aether_geo

        _cg = get_aether_geo()
        if _cg is not None:
            v = np.ascontiguousarray(vertices, dtype=np.float64)
            f = np.ascontiguousarray(faces, dtype=np.int64)
            if v.ndim == 1 and len(v) == 0:
                v = v.reshape((0, 3))
            if f.ndim == 1 and len(f) == 0:
                f = f.reshape((0, 3))
            verts_per_face = int(f.shape[1]) if f.ndim == 2 and f.shape[1] >= 3 else 3
            res = _cg.validate_mesh_geometry(
                v, f, verts_per_face=verts_per_face, require_watertight=False
            )
            return bool(res["is_manifold"])
    return bool(check_manifold(vertices, faces))


def compute_euler_characteristic(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Compute Euler characteristic chi = V - E + F of the mesh.

    For closed 2-manifold surfaces of genus g, chi = 2 - 2g.
    For a genus-0 sphere-like watertight mesh, chi = 2.

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Triangle or quad indices, shape (F, 3) or (F, 4).

    Returns:
        Integer Euler characteristic chi.
    """
    v = np.asarray(vertices)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        return 0

    edge_set: set[tuple[int, int]] = set()
    num_verts_per_face = f.shape[1]
    for i in range(num_verts_per_face):
        next_i = (i + 1) % num_verts_per_face
        v_curr = f[:, i]
        v_next = f[:, next_i]
        u = np.minimum(v_curr, v_next)
        w = np.maximum(v_curr, v_next)
        for u_val, w_val in zip(u, w):
            if u_val != w_val:
                edge_set.add((int(u_val), int(w_val)))

    num_v = len(v)
    num_e = len(edge_set)
    num_f = len(f)
    return int(num_v - num_e + num_f)


def count_non_manifold_edges(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Count the exact number of non-manifold edges in a polygon mesh.

    An edge is non-manifold if:
    1. It is shared by more than 2 faces (branching edge).
    2. It has conflicting orientation (traversed in the same direction by >= 2 faces).

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Triangle or quad indices, shape (F, 3) or (F, 4).

    Returns:
        Count of non-manifold undirected edges (>= 0).
    """
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return 0

    edge_face_count: dict[tuple[int, int], int] = defaultdict(int)
    directed_half_edges: set[tuple[int, int]] = set()
    orientation_conflict_edges: set[tuple[int, int]] = set()

    num_verts_per_face = f.shape[1]
    for i in range(num_verts_per_face):
        next_i = (i + 1) % num_verts_per_face
        for u_raw, v_raw in zip(f[:, i], f[:, next_i]):
            u, v = int(u_raw), int(v_raw)
            if u == v:
                continue
            undirected = (min(u, v), max(u, v))
            edge_face_count[undirected] += 1
            if (u, v) in directed_half_edges:
                orientation_conflict_edges.add(undirected)
            else:
                directed_half_edges.add((u, v))

    non_manifold_count = 0
    for edge, count in edge_face_count.items():
        if count > 2 or edge in orientation_conflict_edges:
            non_manifold_count += 1

    return non_manifold_count


def compute_uv_area_distortion_ratio(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    uv_faces: np.ndarray | None = None,
) -> float:
    """Compute the UV area distortion ratio (max/min normalised face area ratio).

    Measures conformal stretching between 3D surface triangles and 2D UV charts.
    A perfectly isometric or equiareal parameterisation has ratio 1.0.

    Args:
        vertices: 3D vertex positions, shape (V, 3).
        faces: Face vertex indices, shape (F, 3).
        uvs: UV coordinates in [0, 1], shape (U, 2).
        uv_faces: Optional UV face indices, shape (F, 3). Defaults to faces.

    Returns:
        Area distortion ratio >= 1.0. Returns 1.0 if mesh has no faces or zero area.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(uvs, dtype=np.float64)
    uf = f if uv_faces is None else np.asarray(uv_faces, dtype=np.int64)

    if len(f) == 0 or len(uv) == 0 or len(uf) == 0:
        return 1.0

    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross_3d = np.cross(v1 - v0, v2 - v0)
    area_3d = 0.5 * np.linalg.norm(cross_3d, axis=-1)

    u0, u1, u2 = uv[uf[:, 0]], uv[uf[:, 1]], uv[uf[:, 2]]
    cross_2d = (u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1]) - (u1[:, 1] - u0[:, 1]) * (u2[:, 0] - u0[:, 0])
    area_uv = 0.5 * np.abs(cross_2d)

    total_3d = float(np.sum(area_3d))
    total_uv = float(np.sum(area_uv))
    if total_3d <= 1e-12 or total_uv <= 1e-12:
        return 1.0

    p_3d = area_3d / total_3d
    p_uv = area_uv / total_uv

    valid_mask = p_3d > 1e-7
    if not np.any(valid_mask):
        return 1.0

    ratios = p_uv[valid_mask] / np.maximum(p_3d[valid_mask], 1e-12)
    ratios = ratios[np.isfinite(ratios) & (ratios > 1e-12)]
    if len(ratios) == 0:
        return 1.0

    min_r = float(np.min(ratios))
    max_r = float(np.max(ratios))
    if min_r <= 1e-12:
        return 1.0
    return float(max(1.0, max_r / min_r))


def compute_seam_discontinuity_deg(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    normals: np.ndarray | None = None,
    uv_faces: np.ndarray | None = None,
) -> float:
    """Compute mean normal angular discontinuity (in degrees) across UV seams.

    UV seams divide 3D surface charts. Along seam edges, normals should remain
    continuous (small angular jump). Large jumps indicate shading tears.

    Args:
        vertices: 3D vertex positions, shape (V, 3).
        faces: Triangle indices, shape (F, 3).
        uvs: UV coordinates, shape (U, 2).
        normals: Optional vertex or face normals. If None, computes vertex normals.
        uv_faces: Optional UV face indices, shape (F, 3). Defaults to faces.

    Returns:
        Mean angular jump in degrees [0, 180]. 0.0 if no seams exist.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(uvs, dtype=np.float64)
    uf = f if uv_faces is None else np.asarray(uv_faces, dtype=np.int64)

    if len(f) == 0 or len(uv) == 0:
        return 0.0

    edge_to_uv_edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face_idx in range(len(f)):
        tri = f[face_idx]
        uv_tri = uf[face_idx]
        for i in range(3):
            next_i = (i + 1) % 3
            v_edge = (min(int(tri[i]), int(tri[next_i])), max(int(tri[i]), int(tri[next_i])))
            uv_edge = (min(int(uv_tri[i]), int(uv_tri[next_i])), max(int(uv_tri[i]), int(uv_tri[next_i])))
            edge_to_uv_edges[v_edge].append(uv_edge)

    seam_edges = [edge for edge, uv_list in edge_to_uv_edges.items() if len(set(uv_list)) > 1]
    if not seam_edges:
        return 0.0

    norms = compute_normals(v, f) if normals is None else np.asarray(normals, dtype=np.float64)
    if len(norms) == 0:
        return 0.0

    angles = []
    for u, w in seam_edges:
        if u < len(norms) and w < len(norms):
            nu = norms[u] / max(np.linalg.norm(norms[u]), 1e-12)
            nw = norms[w] / max(np.linalg.norm(norms[w]), 1e-12)
            cos_theta = np.clip(np.dot(nu, nw), -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cos_theta))))

    return float(np.mean(angles)) if angles else 0.0


def compute_texel_density_variance(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    uv_faces: np.ndarray | None = None,
    texture_resolution: int = 1024,
) -> float:
    """Compute the Coefficient of Variation (CV = sigma / mu) of texel density.

    Texel density measures the ratio of texture pixels to 3D world surface area.
    Lower variance indicates more uniform texture resolution allocation.

    Args:
        vertices: 3D vertex positions, shape (V, 3).
        faces: Face indices, shape (F, 3).
        uvs: UV coordinates, shape (U, 2).
        uv_faces: Optional UV face indices, shape (F, 3). Defaults to faces.
        texture_resolution: Target texture map resolution (e.g. 1024, 2048).

    Returns:
        Coefficient of variation >= 0.0. 0.0 indicates completely uniform texel density.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(uvs, dtype=np.float64)
    uf = f if uv_faces is None else np.asarray(uv_faces, dtype=np.int64)

    if len(f) == 0 or len(uv) == 0 or len(uf) == 0:
        return 0.0

    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area_3d = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)

    u0, u1, u2 = uv[uf[:, 0]], uv[uf[:, 1]], uv[uf[:, 2]]
    cross_2d = (u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1]) - (u1[:, 1] - u0[:, 1]) * (u2[:, 0] - u0[:, 0])
    area_uv_norm = 0.5 * np.abs(cross_2d)
    area_uv_pixels = area_uv_norm * (texture_resolution**2)

    valid = area_3d > 1e-8
    if not np.any(valid):
        return 0.0

    densities = np.sqrt(np.maximum(area_uv_pixels[valid], 0.0)) / np.sqrt(np.maximum(area_3d[valid], 1e-12))
    mu = float(np.mean(densities))
    if mu <= 1e-12:
        return 0.0
    sigma = float(np.std(densities))
    return float(sigma / mu)

