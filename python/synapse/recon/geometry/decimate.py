"""Quadric Error Metrics (QEM) mesh decimation helper.

Extracted from ``synapse.recon.geometry``.
"""

from __future__ import annotations

import numpy as np


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
