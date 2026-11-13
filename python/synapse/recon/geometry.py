"""Ground Truth Oracle for geometric algorithm correctness verification.

This module provides pure-Python (NumPy) reference implementations of mesh
geometry checks (manifold, watertight, degenerate faces, self-intersections,
connected components, etc.).

**This is NOT production code.**  Normal workflows use the C++20 native kernel
(``synapse._c_geometry`` via ``csrc/geometry/``) for actual geometry validation.
This Python Oracle exists solely to:

1. Serve as the algorithm correctness reference (ground truth).
2. Enable parity testing against the C++ implementation
   (see ``tests/test_geometry_oracle_parity.py``).
3. Act as a last-resort fallback when the C++ extension is unavailable
   (with a ``RuntimeWarning``).

Additionally includes: area-weighted normal estimation, Kabsch/ICP 6-DoF rigid
alignment, and quad-dominant remeshing — utilities used by the recon pipeline.
"""

from __future__ import annotations

from collections import defaultdict

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
        # Two incident faces are linked iff they share an undirected polygon
        # edge incident to vert_idx.
        adj: dict[int, set[int]] = defaultdict(set)
        for i, f1 in enumerate(incident_faces):
            edges1 = set(face_incident_edges[f1][vert_idx])
            for f2 in incident_faces[i + 1 :]:
                edges2 = set(face_incident_edges[f2][vert_idx])
                if edges1 & edges2:
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

    for _ in range(max_iterations):
        # Find nearest neighbors via vectorized euclidean distance
        # For efficiency, compute pairwise squared distances
        diff = transformed_src[:, np.newaxis, :] - tgt[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=-1)
        nearest_indices = np.argmin(sq_dist, axis=1)
        matched_tgt = tgt[nearest_indices]

        # Step transformation
        step_r, step_t = _solve_kabsch(transformed_src, matched_tgt)

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


def quad_remesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remesh a triangle mesh into a quad-dominant or quad mesh.

    Pairs adjacent coplanar triangles sharing an edge based on dihedral angle
    and aspect ratio, removing interior diagonals to form quad faces.

    Args:
        vertices: Vertex positions, shape (V, 3).
        faces: Triangle indices, shape (F, 3).
        target_faces: Optional target face count (0 = maximize quads).

    Returns:
        Tuple of (new_vertices, new_faces).
    """
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)

    if len(v) == 0 or len(f) == 0:
        return v, f

    if f.shape[1] == 4:
        return v, f  # Already quad mesh

    if f.shape[1] != 3:
        return v, f  # Unsupported polygon type

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
    elif quad_faces and unpaired_triangles:
        # Convert unpaired triangles to degenerate quads [a, b, c, c] or keep quad dominant
        # In graphics standard, a quad-dominant list converts triangles by repeating last index
        padded_tris = [[tri[0], tri[1], tri[2], tri[2]] for tri in unpaired_triangles]
        all_quads = quad_faces + padded_tris
        new_faces = np.array(all_quads, dtype=np.int32)
    else:
        new_faces = f

    return v, new_faces


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

    tri = f[:, :3]
    oob_mask = np.any((tri < 0) | (tri >= len(v)), axis=1)
    safe_tri = np.where(oob_mask[:, None], 0, tri)

    v0 = v[safe_tri[:, 0]]
    v1 = v[safe_tri[:, 1]]
    v2 = v[safe_tri[:, 2]]
    nonfinite_mask = (
        ~np.all(np.isfinite(v0), axis=1)
        | ~np.all(np.isfinite(v1), axis=1)
        | ~np.all(np.isfinite(v2), axis=1)
    )

    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)

    # Area = 0.5 * |cross(e1, e2)|
    area = 0.5 * np.linalg.norm(cross, axis=1)

    # Edge lengths squared
    e3 = v2 - v1
    len_sq_e1 = np.sum(e1 * e1, axis=1)
    len_sq_e2 = np.sum(e2 * e2, axis=1)
    len_sq_e3 = np.sum(e3 * e3, axis=1)
    longest_edge_sq = np.maximum(np.maximum(len_sq_e1, len_sq_e2), len_sq_e3)

    # Aspect ratio ≈ longest_edge² / (2 * area)
    safe_area = np.where(area > 0.0, area, 1.0)
    aspect = longest_edge_sq / (2.0 * safe_area)
    # If area is zero the aspect is infinite → always degenerate
    aspect = np.where(area > 0.0, aspect, np.inf)

    mask = (
        oob_mask
        | nonfinite_mask
        | ~np.isfinite(area)
        | (area < min_area)
        | ~np.isfinite(aspect)
        | (aspect > max_aspect_ratio)
    )
    return np.nonzero(mask)[0].astype(np.intp)


def check_self_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
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
        if (
            float(np.max(np.abs(db))) <= plane_eps
            and float(np.max(np.abs(da))) <= plane_eps
        ):
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
    tri_indices = f[:, :3]
    valid_face = ~np.any((tri_indices < 0) | (tri_indices >= num_verts), axis=1)
    safe_indices = np.where(valid_face[:, None], tri_indices, 0)
    tri_verts = v[safe_indices]  # (F, 3, 3)
    valid_face &= np.all(np.isfinite(tri_verts), axis=(1, 2))

    valid_ids = np.nonzero(valid_face)[0]
    if len(valid_ids) <= 1:
        return []

    tri_min = tri_verts.min(axis=1)
    tri_max = tri_verts.max(axis=1)

    global_min = tri_min[valid_ids].min(axis=0) - 1e-6
    global_max = tri_max[valid_ids].max(axis=0) + 1e-6
    extent = np.maximum(global_max - global_min, 1e-6)

    num_cells = max(min(int(round(len(valid_ids) ** (1.0 / 3.0))), 256), 4)
    cell_size = extent / num_cells

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

    for cell_faces in grid.values():
        for a_idx in range(len(cell_faces)):
            for b_idx in range(a_idx + 1, len(cell_faces)):
                fi = cell_faces[a_idx]
                fj = cell_faces[b_idx]
                pair = (min(fi, fj), max(fi, fj))
                if pair in tested:
                    continue
                tested.add(pair)

                # Skip adjacent faces (sharing ≥ 1 vertex)
                if face_vertex_sets[fi] & face_vertex_sets[fj]:
                    continue

                # Exact AABB overlap pre-filter
                if np.any(tri_min[fi] > tri_max[fj] + aabb_eps) or np.any(
                    tri_max[fi] + aabb_eps < tri_min[fj]
                ):
                    continue

                # Skip UV / hard-normal split-vertex seam neighbors (1 or 2
                # coincident 3D corners, while still catching 3-vertex duplicate
                # z-fighting faces).
                diff_sq = np.sum(
                    (tri_verts[fi][:, None, :] - tri_verts[fj][None, :, :]) ** 2,
                    axis=2,
                )
                shared_pos = int(np.count_nonzero(np.any(diff_sq <= 1e-18, axis=1)))
                if shared_pos in (1, 2):
                    continue

                if _tri_tri_intersect(tri_verts[fi], tri_verts[fj]):
                    result.append(pair)

    result.sort()
    return result


def find_connected_components(
    faces: np.ndarray,
) -> list[np.ndarray]:
    """Partition mesh faces into connected components via shared edges.

    Uses a Union-Find (disjoint-set) data structure keyed on face indices
    where edges provide the union relation.

    Args:
        faces: Triangle (or quad) indices, shape ``(F, K)``.

    Returns:
        List of 1-D arrays, each containing the sorted face indices of one
        connected component.  Components are ordered largest-first.
    """
    f = np.asarray(faces, dtype=np.int64)

    if len(f) == 0:
        return []

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
