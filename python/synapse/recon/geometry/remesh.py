"""Quad-dominant mesh remeshing utilities.

Extracted from ``synapse.recon.geometry``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from synapse.recon.geometry.decimate import _qem_decimate
from synapse.recon.geometry.uv import _unwrap_uv_atlas_with_seams


def quad_remesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = 0,
    *,
    expand_uv_seams: bool = False,
    worker_client: Any = None,
    strict_native: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Remesh a triangle mesh into a quad-dominant or quad mesh.

    When *strict_native* is True (or *worker_client* is provided), delegates to
    :class:`~synapse.bridge.aether_geo_worker_client.AetherGeoWorkerClient.quadriflow`,
    which strictly raises :class:`~synapse.errors.NotImplementedStageError`
    (exit code ``50``, ``NEG-100-08``) on upstream stub exit codes.  Otherwise
    runs the Python coplanar-pairing + UV unwrap Oracle.
    """
    if strict_native or worker_client is not None:
        from synapse.bridge.aether_geo_worker_client import AetherGeoWorkerClient

        client = worker_client or AetherGeoWorkerClient()
        res = client.quadriflow(
            vertices,
            faces,
            target_faces=target_faces or 1500,
        )
        return res.vertices, res.faces, res.uvs

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
