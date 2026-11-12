"""Tier 0 deterministic geometry gate for 3D mesh quality validation.

Runs topology and geometric sanity checks (manifold, watertight, degenerate
faces, self-intersections, floating islands) and produces a structured
:class:`~synapse.inspect.schemas.GeometricValidationResult`.

The gate supports two execution backends:

* **python** — Pure-Python Oracle (NumPy-based, always available).
* **native** — C++20 accelerated via ``synapse._c_geometry`` (optional).
* **auto** (default) — Tries native first, falls back to Python with a
  ``RuntimeWarning``.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from synapse.inspect.schemas import GeometricValidationResult
from synapse.recon.geometry import (
    check_degenerate_faces,
    check_manifold,
    check_self_intersections,
    check_watertight,
    find_connected_components,
)


def _floating_island_indices_from_components(
    components: list[Any],
    total_faces: int,
    min_face_ratio: float,
) -> list[int]:
    """Return sorted face indices belonging to floating islands.

    A connected component is a "floating island" if its face count is
    strictly less than ``min_face_ratio * total_faces``.
    """
    if total_faces <= 0:
        return []
    threshold = min_face_ratio * total_faces
    island_indices: list[int] = []
    for comp in components:
        if len(comp) < threshold:
            island_indices.extend(int(i) for i in comp)
    island_indices.sort()
    return island_indices


def validate_mesh_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    require_watertight: bool = False,
    min_area: float = 1e-8,
    max_aspect_ratio: float = 100.0,
    min_face_ratio: float = 0.01,
    backend: Literal["auto", "python", "native"] = "auto",
) -> GeometricValidationResult:
    """Run all Tier 0 deterministic geometry checks on a triangle mesh.

    This is the unified geometry gate that must pass before any neural or
    rendering-based evaluation is performed.

    Args:
        vertices: Vertex positions, shape ``(V, 3)``.
        faces: Triangle indices, shape ``(F, 3)``.
        require_watertight: If ``True``, the mesh must be watertight to
            pass the gate.
        min_area: Minimum triangle area for degenerate-face detection.
        max_aspect_ratio: Maximum aspect ratio for degenerate-face
            detection.
        min_face_ratio: Minimum fraction of total faces a connected
            component must have to avoid being classified as a floating
            island.
        backend: Execution backend — ``"auto"`` (try native, fallback to
            Python), ``"python"`` (force Python Oracle), or ``"native"``
            (force C++, raises :class:`ImportError` if unavailable).

    Returns:
        A :class:`GeometricValidationResult` with all gate results.

    Raises:
        ImportError: If ``backend="native"`` and the C++ extension
            ``synapse._c_geometry`` is not available.
        ValueError: If ``backend`` is not one of the accepted values.
    """
    if backend not in ("auto", "python", "native"):
        raise ValueError(
            f"Invalid backend {backend!r}; expected 'auto', 'python', or 'native'."
        )

    use_native = False
    if backend in ("auto", "native"):
        try:
            import synapse._c_geometry as _cg  # type: ignore[import-not-found]

            use_native = True
        except ImportError:
            if backend == "native":
                raise
            warnings.warn(
                "synapse._c_geometry not available; falling back to Python "
                "Oracle for geometry validation.",
                RuntimeWarning,
                stacklevel=2,
            )

    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)

    if v.ndim == 1 and len(v) == 0:
        v = v.reshape((0, 3))
    if f.ndim == 1 and len(f) == 0:
        f = f.reshape((0, 3))

    failure_reasons: list[str] = []

    if use_native:
        verts_per_face = int(f.shape[1]) if f.ndim == 2 and f.shape[1] >= 3 else 3
        is_manifold: bool = bool(
            _cg.check_manifold(v, f, verts_per_face=verts_per_face)  # type: ignore[possibly-undefined]
        )
        is_watertight: bool = bool(
            _cg.check_watertight(v, f, verts_per_face=verts_per_face)  # type: ignore[possibly-undefined]
        )
        degen_indices = _cg.check_degenerate_faces(  # type: ignore[possibly-undefined]
            v,
            f,
            verts_per_face=verts_per_face,
            min_area=min_area,
            max_aspect_ratio=max_aspect_ratio,
        )
        si_pairs: list[tuple[int, int]] = [
            (int(a), int(b))
            for a, b in _cg.check_self_intersections(  # type: ignore[possibly-undefined]
                v, f, verts_per_face=verts_per_face,
            )
        ]
        components = _cg.find_connected_components(  # type: ignore[possibly-undefined]
            f,
            verts_per_face=verts_per_face,
            num_vertices=int(len(v)),
        )
        num_faces = int(len(f)) if f.ndim == 2 else int(len(f) // verts_per_face)
    else:
        is_manifold = check_manifold(v, f)
        is_watertight = check_watertight(v, f)
        degen_indices = check_degenerate_faces(
            v, f, min_area=min_area, max_aspect_ratio=max_aspect_ratio,
        )
        si_pairs = check_self_intersections(v, f)
        components = find_connected_components(f) if f.ndim == 2 and len(f) > 0 else []
        num_faces = int(len(f)) if f.ndim == 2 else 0

    num_components = len(components)
    floating_island_indices = _floating_island_indices_from_components(
        components, num_faces, min_face_ratio,
    )

    degenerate_list = [int(i) for i in degen_indices]

    if not is_manifold:
        failure_reasons.append("Mesh is not a valid 2-manifold.")

    if require_watertight and not is_watertight:
        failure_reasons.append("Mesh is not watertight (has boundary edges).")

    if len(degenerate_list) > 0:
        failure_reasons.append(
            f"{len(degenerate_list)} degenerate face(s) detected "
            f"(area < {min_area} or aspect ratio > {max_aspect_ratio})."
        )

    if len(si_pairs) > 0:
        failure_reasons.append(
            f"{len(si_pairs)} self-intersecting face pair(s) detected."
        )

    if floating_island_indices:
        failure_reasons.append(
            f"{len(floating_island_indices)} face(s) belong to floating islands "
            f"(components with < {min_face_ratio:.1%} of total faces)."
        )

    passed = (
        is_manifold
        and (not require_watertight or is_watertight)
        and len(degenerate_list) == 0
        and len(si_pairs) == 0
        and len(floating_island_indices) == 0
    )

    return GeometricValidationResult(
        is_manifold=is_manifold,
        is_watertight=is_watertight,
        num_vertices=int(len(v)),
        num_faces=num_faces,
        degenerate_face_indices=degenerate_list,
        self_intersecting_pairs=si_pairs,
        num_connected_components=num_components,
        floating_island_face_indices=floating_island_indices,
        passed=passed,
        failure_reasons=failure_reasons,
    )
