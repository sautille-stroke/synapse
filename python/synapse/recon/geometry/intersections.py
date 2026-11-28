"""Triangle-triangle self-intersection oracle for mesh validation.

Pure-Python Ground Truth Oracle functions extracted from ``synapse.recon.geometry``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


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
