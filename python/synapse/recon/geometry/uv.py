"""UV atlas unwrapping, seam discontinuity, and texel density metrics.

Pure-Python Ground Truth Oracle & helper functions extracted from ``synapse.recon.geometry``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


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


def _weld_vertex_positions(vertices: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Map vertices with coincident 3D coordinates (within *tol*) to canonical IDs."""
    if len(vertices) == 0:
        return np.empty((0,), dtype=np.int64)
    scale = 1.0 / max(tol, 1e-9)
    quantized = np.round(vertices * scale).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute unit face normals for triangles/polygons."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return np.asarray(fn / norms, dtype=np.float64)


def compute_seam_discontinuity_deg(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    normals: np.ndarray | None = None,
    uv_faces: np.ndarray | None = None,
) -> float:
    """Compute mean normal angular discontinuity (in degrees) **across** UV seams.

    Handles both indexed-UV meshes (explicit ``uv_faces``) and split-vertex UV
    meshes (e.g. Wavefront OBJ / glTF / trimesh where ``uv_faces is None`` and
    seam vertices are duplicated in ``vertices``).  Along each physical seam edge
    shared by two faces with diverging UV coordinates/indices, compares the
    normals across the two sides of the seam (split corner normals when
    explicitly supplied, or adjacent face normals across the crease) rather
    than endpoints along the same edge.

    Args:
        vertices: 3D vertex positions, shape (V, 3).
        faces: Face vertex indices, shape (F, 3) or (F, 4).
        uvs: UV coordinates, shape (U, 2).
        normals: Optional vertex normals (V, 3) or face normals (F, 3).
        uv_faces: Optional UV face indices, shape (F, 3) or (F, 4).

    Returns:
        Mean angular jump across seams in degrees [0, 180]. 0.0 if no seams exist.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(uvs, dtype=np.float64)
    uf = f if uv_faces is None else np.asarray(uv_faces, dtype=np.int64)

    if len(v) == 0 or len(f) == 0 or len(uv) == 0 or f.shape != uf.shape:
        return 0.0

    # When uv_faces is omitted (split-UV meshes), weld 3D positions to recover
    # physical edge connectivity across duplicated UV-seam vertices.
    if uv_faces is None:
        phys_ids = _weld_vertex_positions(v, tol=1e-5)
    else:
        phys_ids = np.arange(len(v), dtype=np.int64)

    face_norms = _compute_face_normals(v, f)
    vert_norms: np.ndarray | None = None
    explicit_face_norms: np.ndarray | None = None
    if normals is not None:
        n_arr = np.asarray(normals, dtype=np.float64)
        if len(n_arr) == len(f) and len(n_arr) != len(v):
            explicit_face_norms = n_arr
        elif len(n_arr) == len(v):
            n_len = np.linalg.norm(n_arr, axis=1, keepdims=True)
            vert_norms = n_arr / np.where(n_len < 1e-12, 1.0, n_len)

    # Quantize UV coords so seam detection distinguishes split-coordinate seams cleanly
    uv_q = np.round(uv * 1e5).astype(np.int64)

    # Map canonical physical edge (p_lo, p_hi) -> list of edge incidences:
    # (face_idx, ca_corner, cb_corner, uv_pair_key)
    k = f.shape[1]
    edge_incidences: dict[
        tuple[int, int],
        list[tuple[int, int, int, tuple[tuple[int, int, int], tuple[int, int, int]]]],
    ] = defaultdict(list)

    for face_idx in range(len(f)):
        tri = f[face_idx]
        uv_tri = uf[face_idx]
        for i in range(k):
            next_i = (i + 1) % k
            va, vb = int(tri[i]), int(tri[next_i])
            pa, pb = int(phys_ids[va]), int(phys_ids[vb])
            if pa == pb:
                continue
            ua, ub = int(uv_tri[i]), int(uv_tri[next_i])
            if pa > pb:
                pa, pb = pb, pa
                va, vb = vb, va
                ua, ub = ub, ua
            # Include both UV index and quantized UV coords for canonical edge
            uv_a_q = (int(uv_q[ua, 0]), int(uv_q[ua, 1])) if 0 <= ua < len(uv_q) else (0, 0)
            uv_b_q = (int(uv_q[ub, 0]), int(uv_q[ub, 1])) if 0 <= ub < len(uv_q) else (0, 0)
            uv_key = (
                (ua if uv_faces is not None else 0, uv_a_q[0], uv_a_q[1]),
                (ub if uv_faces is not None else 0, uv_b_q[0], uv_b_q[1]),
            )
            edge_incidences[(pa, pb)].append((face_idx, va, vb, uv_key))

    angles: list[float] = []
    fn_src = explicit_face_norms if explicit_face_norms is not None else face_norms
    for entries in edge_incidences.values():
        if len(entries) < 2:
            continue
        unique_uv_keys = {e[3] for e in entries}
        if len(unique_uv_keys) <= 1:
            continue  # Continuous UV edge, not a seam

        # Compare pairs of adjacent faces across the seam
        for i in range(len(entries)):
            f1, c1a, c1b, k1 = entries[i]
            for j in range(i + 1, len(entries)):
                f2, c2a, c2b, k2 = entries[j]
                if k1 == k2 or f1 == f2:
                    continue
                if vert_norms is not None and (c1a != c2a or c1b != c2b):
                    # Compare split corner normals across the seam at matching endpoints
                    ca_dot = float(np.clip(np.dot(vert_norms[c1a], vert_norms[c2a]), -1.0, 1.0))
                    cb_dot = float(np.clip(np.dot(vert_norms[c1b], vert_norms[c2b]), -1.0, 1.0))
                    ca_deg = float(np.degrees(np.arccos(ca_dot)))
                    cb_deg = float(np.degrees(np.arccos(cb_dot)))
                    angles.append(0.5 * (ca_deg + cb_deg))
                else:
                    # Compare adjacent face normals across the seam edge
                    n1 = fn_src[f1]
                    n2 = fn_src[f2]
                    n1_u = n1 / max(float(np.linalg.norm(n1)), 1e-12)
                    n2_u = n2 / max(float(np.linalg.norm(n2)), 1e-12)
                    dot = float(np.clip(np.dot(n1_u, n2_u), -1.0, 1.0))
                    angles.append(float(np.degrees(np.arccos(dot))))

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
