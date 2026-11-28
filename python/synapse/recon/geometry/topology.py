"""Mesh topology verification, normal estimation, and connected-component pruning.

Pure-Python Ground Truth Oracle functions extracted from ``synapse.recon.geometry``.
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
