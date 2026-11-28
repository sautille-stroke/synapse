"""Differential and topological mesh quality metrics.

Pure-Python Ground Truth Oracle metrics extracted from ``synapse.recon.geometry``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from synapse.recon.geometry.topology import compute_normals
from synapse.recon.geometry.uv import _compute_face_normals


def compute_euler_characteristic(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Compute Euler characteristic chi = V - E + F of the mesh.

    For closed 2-manifold surfaces of genus g, chi = 2 - 2g.
    For a genus-0 sphere-like watertight mesh, chi = 2.

    Only vertices actually referenced by *faces* are counted in ``V`` so that
    unreferenced orphan vertices in the buffer do not corrupt ``chi``.

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Triangle or quad indices, shape (F, 3) or (F, 4).

    Returns:
        Integer Euler characteristic chi.
    """
    v = np.asarray(vertices)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0 or f.ndim != 2 or f.shape[1] < 3:
        return 0

    k = f.shape[1]
    edges_parts = [
        np.stack([f[:, i], f[:, (i + 1) % k]], axis=1)
        for i in range(k)
    ]
    all_edges = np.concatenate(edges_parts, axis=0)
    # Exclude self-loops, canonicalise undirected edges, deduplicate vectorised
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]
    if len(all_edges) > 0:
        sorted_edges = np.sort(all_edges, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)
        num_e = int(len(unique_edges))
    else:
        num_e = 0

    num_v = int(np.unique(f).size)
    num_f = int(len(f))
    return int(num_v - num_e + num_f)


def count_non_manifold_edges(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Count the exact number of non-manifold edges in a polygon mesh.

    Delegates to ``aether_geo.count_non_manifold_edges`` when exposed by the
    compiled native extension, with pure-Python Oracle fallback.

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

    from synapse.bridge.aether_bridge import get_aether_geo

    geo = get_aether_geo()
    if geo is not None and hasattr(geo, "count_non_manifold_edges"):
        try:
            v_arr = np.ascontiguousarray(vertices, dtype=np.float64)
            f_arr = np.ascontiguousarray(f, dtype=np.int64)
            return int(geo.count_non_manifold_edges(v_arr, f_arr))
        except Exception:
            pass

    edge_face_count: dict[tuple[int, int], int] = defaultdict(int)
    directed_half_edges: set[tuple[int, int]] = set()
    orientation_conflict_edges: set[tuple[int, int]] = set()

    num_verts_per_face = f.shape[1]
    for i in range(num_verts_per_face):
        next_i = (i + 1) % num_verts_per_face
        for u_raw, v_raw in zip(f[:, i], f[:, next_i], strict=True):
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


def count_non_manifold_vertices(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Count non-manifold ("bowtie" / pinched) vertices in a polygon mesh.

    A vertex is 2-manifold iff its incident faces form a single connected fan
    via edges sharing that vertex. Vertices connecting multiple disconnected
    surface umbrellas (e.g. two tetrahedra meeting at one apex vertex) are
    flagged as non-manifold vertices even if every incident edge has degree <= 2.

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Triangle or quad indices, shape (F, 3) or (F, 4).

    Returns:
        Count of non-manifold vertices (>= 0).
    """
    del vertices  # topology-only check; kept in signature for parity
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0 or f.ndim != 2 or f.shape[1] < 3:
        return 0

    k = f.shape[1]
    # Build incident face lists per vertex and edge-to-face adjacency
    vert_to_faces: dict[int, list[int]] = defaultdict(list)
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(len(f)):
        poly = [int(x) for x in f[fi]]
        seen_in_face: set[int] = set()
        for idx, vid in enumerate(poly):
            if vid not in seen_in_face:
                vert_to_faces[vid].append(fi)
                seen_in_face.add(vid)
            nxt = poly[(idx + 1) % k]
            if vid != nxt:
                edge_to_faces[(min(vid, nxt), max(vid, nxt))].append(fi)

    non_manifold_verts = 0
    for vid, inc_faces in vert_to_faces.items():
        if len(inc_faces) <= 1:
            continue
        inc_set = set(inc_faces)
        # Build local adjacency among incident faces via edges containing `vid`
        adj: dict[int, set[int]] = {fi: set() for fi in inc_faces}
        for fi in inc_faces:
            poly = [int(x) for x in f[fi]]
            for idx, v_cur in enumerate(poly):
                v_nxt = poly[(idx + 1) % k]
                if v_cur == vid or v_nxt == vid:
                    e = (min(v_cur, v_nxt), max(v_cur, v_nxt))
                    for nbr in edge_to_faces.get(e, ()):
                        if nbr != fi and nbr in inc_set:
                            adj[fi].add(nbr)
                            adj[nbr].add(fi)

        # Check single connected component via BFS
        visited: set[int] = set()
        stack = [inc_faces[0]]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nxt_f in adj[cur]:
                if nxt_f not in visited:
                    stack.append(nxt_f)
        if len(visited) != len(inc_set):
            non_manifold_verts += 1

    return non_manifold_verts


def _triangulate_faces(faces: np.ndarray) -> np.ndarray:
    """Fan-triangulate (F, K) faces to (F', 3)."""
    if faces.ndim != 2 or len(faces) == 0 or faces.shape[1] < 3:
        return np.empty((0, 3), dtype=np.int64)
    if faces.shape[1] == 3:
        return faces
    tris = [faces[:, [0, i, i + 1]] for i in range(1, faces.shape[1] - 1)]
    return np.concatenate(tris, axis=0)


def discrete_mean_curvature_variance(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> float:
    """Compute variance of discrete Laplace-Beltrami mean curvature magnitudes.

    Uses the cotangent-weighted Laplacian (``(Lv)_i = 0.5 * sum cot_ij (v_i - v_j)``)
    divided by barycentric Voronoi vertex area ``A_i`` to estimate per-vertex
    mean curvature ``|H_i| = 0.5 * ||(Lv)_i|| / A_i``. High variance across a
    mesh indicates uneven surface ripples or topological noise relative to
    smooth constant-curvature geometry (e.g. a smooth plane or sphere has ~0
    curvature variance).

    Args:
        vertices: Vertex positions, shape ``(V, 3)``.
        faces: Triangle or polygon indices, shape ``(F, K)``.

    Returns:
        Non-negative scalar variance of discrete mean curvature (0.0 if < 2 valid
        vertices).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f_raw = np.asarray(faces, dtype=np.int64)
    f = _triangulate_faces(f_raw)
    if len(v) < 3 or len(f) < 2:
        return 0.0

    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    v0, v1, v2 = v[i0], v[i1], v[i2]

    # Triangle edge vectors opposite to vertices 0, 1, 2
    e01 = v1 - v0
    e02 = v2 - v0
    e12 = v2 - v1

    tri_cross = np.cross(e01, e02)
    double_area = np.linalg.norm(tri_cross, axis=1)
    tri_area = 0.5 * double_area

    # Cotangent weights opposite to edges (1,2), (0,2), (0,1)
    # Using dot / ||cross|| with safe clamping for near-degenerate angles
    safe_da = np.where(double_area > 1e-12, double_area, 1.0)
    cot0 = np.where(double_area > 1e-12, np.sum(e01 * e02, axis=1) / safe_da, 0.0)
    cot1 = np.where(double_area > 1e-12, np.sum((-e01) * e12, axis=1) / safe_da, 0.0)
    cot2 = np.where(double_area > 1e-12, np.sum((-e02) * (-e12), axis=1) / safe_da, 0.0)
    cot0 = np.clip(cot0, -1e4, 1e4)
    cot1 = np.clip(cot1, -1e4, 1e4)
    cot2 = np.clip(cot2, -1e4, 1e4)

    laplacian = np.zeros_like(v, dtype=np.float64)
    # Opposite vertex 0 has edge (1, 2) with weight cot0
    diff12 = v1 - v2
    np.add.at(laplacian, i1, 0.5 * cot0[:, None] * diff12)
    np.add.at(laplacian, i2, -0.5 * cot0[:, None] * diff12)
    # Opposite vertex 1 has edge (2, 0) with weight cot1
    diff20 = v2 - v0
    np.add.at(laplacian, i2, 0.5 * cot1[:, None] * diff20)
    np.add.at(laplacian, i0, -0.5 * cot1[:, None] * diff20)
    # Opposite vertex 2 has edge (0, 1) with weight cot2
    diff01 = v0 - v1
    np.add.at(laplacian, i0, 0.5 * cot2[:, None] * diff01)
    np.add.at(laplacian, i1, -0.5 * cot2[:, None] * diff01)

    vert_area = np.zeros(len(v), dtype=np.float64)
    area_share = tri_area / 3.0
    np.add.at(vert_area, i0, area_share)
    np.add.at(vert_area, i1, area_share)
    np.add.at(vert_area, i2, area_share)

    ref_mask = np.zeros(len(v), dtype=bool)
    ref_mask[f.ravel()] = True
    valid_mask = ref_mask & (vert_area > 1e-12)

    # Identify interior (non-boundary) vertices when enough exist so open boundary
    # rings do not introduce spurious Laplace-Beltrami drift.
    e_all = np.concatenate(
        [
            np.sort(f[:, [0, 1]], axis=1),
            np.sort(f[:, [1, 2]], axis=1),
            np.sort(f[:, [2, 0]], axis=1),
        ],
        axis=0,
    )
    uniq_e, e_counts = np.unique(e_all, axis=0, return_counts=True)
    boundary_verts = uniq_e[e_counts == 1].ravel()
    interior_mask = valid_mask.copy()
    if len(boundary_verts) > 0:
        interior_mask[boundary_verts] = False
    eval_mask = interior_mask if int(np.sum(interior_mask)) >= 2 else valid_mask

    if int(np.sum(eval_mask)) < 2:
        return 0.0

    # Normal-projected signed mean curvature H_i = 0.5 * <(Lv)_i, n_i> / A_i.
    # Projecting onto smooth vertex normals eliminates in-plane tangential drift
    # on non-uniform triangulations while capturing hills (+) vs valleys (-).
    v_norms = np.asarray(compute_normals(v, f), dtype=np.float64)
    h_vals = (
        0.5
        * np.sum(laplacian[eval_mask] * v_norms[eval_mask], axis=1)
        / vert_area[eval_mask]
    )
    h_vals = h_vals[np.isfinite(h_vals)]
    if len(h_vals) < 2:
        return 0.0
    return float(np.var(h_vals))


def surface_roughness_var(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> float:
    """Compute surface roughness as the variance of interior edge dihedral angles (rad^2).

    A planar patch or uniform-dihedral polyhedron (e.g. regular tetrahedron/cube)
    exhibits ~0 dihedral angle variance, while noisy/bumpy reconstructed surfaces
    exhibit large variance across adjacent face normals.

    Args:
        vertices: Vertex positions, shape ``(V, 3)``.
        faces: Face indices, shape ``(F, K)``.

    Returns:
        Non-negative variance of dihedral angles in radians^2 (0.0 if < 2 interior
        edges).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) < 3 or len(f) < 2 or f.ndim != 2 or f.shape[1] < 3:
        return 0.0

    face_norms = _compute_face_normals(v, f)
    k = f.shape[1]
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(len(f)):
        poly = f[fi]
        for i in range(k):
            a, b = int(poly[i]), int(poly[(i + 1) % k])
            if a != b:
                edge_faces[(min(a, b), max(a, b))].append(fi)

    angles: list[float] = []
    for inc in edge_faces.values():
        if len(inc) == 2:
            f1, f2 = inc[0], inc[1]
            dot = float(np.clip(np.dot(face_norms[f1], face_norms[f2]), -1.0, 1.0))
            angles.append(float(np.arccos(dot)))

    if len(angles) < 2:
        return 0.0
    return float(np.var(np.asarray(angles, dtype=np.float64)))
