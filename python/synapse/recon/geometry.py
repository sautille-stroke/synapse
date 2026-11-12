"""Geometric algorithms for Video-to-3D inverse reconstruction.

Includes 2-manifold verification, watertight validation, area-weighted normal
estimation, Kabsch/ICP 6-DoF rigid alignment, and quad-dominant remeshing.
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

    if len(v) == 0 or len(f) == 0:
        return False

    if not np.all(np.isfinite(v)):
        return False

    if np.min(f) < 0 or np.max(f) >= len(v):
        return False

    # Check non-manifold edges
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    vertex_faces: dict[int, list[int]] = defaultdict(list)

    num_faces = len(f)
    num_verts_per_face = f.shape[1]

    for face_idx in range(num_faces):
        poly = f[face_idx]
        for k in range(num_verts_per_face):
            u, w = int(poly[k]), int(poly[(k + 1) % num_verts_per_face])
            if u == w:
                return False  # Degenerate zero-length edge
            edge_key = (min(u, w), max(u, w))
            edge_counts[edge_key] += 1
            if edge_counts[edge_key] > 2:
                # Non-manifold edge shared by >= 3 faces
                return False
            vertex_faces[u].append(face_idx)

    # Check vertex manifoldness: face umbrella connectedness around each vertex
    for vert_idx, incident_faces in vertex_faces.items():
        if len(incident_faces) <= 1:
            continue
        # Build adjacency graph of incident faces sharing an edge
        adj: dict[int, set[int]] = defaultdict(set)
        for i, f1 in enumerate(incident_faces):
            p1 = set(f[f1])
            for f2 in incident_faces[i + 1:]:
                p2 = set(f[f2])
                # If they share an edge containing vert_idx
                if len(p1.intersection(p2)) >= 2:
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
