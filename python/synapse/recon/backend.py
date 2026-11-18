# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Synapse native reconstruction backend.

Provides :class:`SynapseReconBackend`, a self-contained implementation of
:class:`~synapse.bridge.vg_bridge.VgBackend` using synapse.recon modules
(de-lighting, quad remesh, manifold check, IK, 6-DoF alignment).

.. note::

   This module was extracted from ``synapse.bridge.vg_bridge`` to break the
   ``bridge → recon`` circular dependency (P0-ARCH-01). ``bridge.vg_bridge``
   re-exports for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from synapse.render_core.backend import MockVgBackend, VgBackend


class SynapseReconBackend(VgBackend):
    """Native Python/algorithmic backend using synapse.recon.

    Provides fully working, self-contained implementations for de-lighting,
    quad remeshing, manifold verification, foot-planting IK, and 6-DoF alignment.
    Does not depend on external vg-core C++ extension.
    """

    def delight(
        self,
        albedo: np.ndarray,
        normal: np.ndarray,
        radiance: np.ndarray,
    ) -> np.ndarray:
        """De-light radiance image using synapse.recon.delight."""
        from synapse.recon.delight import delight as recon_delight

        return recon_delight(albedo, normal, radiance)

    def quad_remesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remesh to quad-dominant mesh using synapse.recon.geometry."""
        from synapse.recon.geometry import quad_remesh as recon_quad_remesh

        v, f, _uvs = recon_quad_remesh(vertices, faces, target_faces)
        return v, f

    def check_manifold(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> bool:
        """Check 2-manifold status using native C++ kernel with Oracle fallback."""
        try:
            import synapse._c_geometry as _cg

            v_arr = np.ascontiguousarray(vertices, dtype=np.float64)
            f_arr = np.ascontiguousarray(faces, dtype=np.int64)
            return bool(_cg.check_manifold(v_arr, f_arr))
        except (ImportError, AttributeError):
            from synapse.recon.geometry import check_manifold as recon_check_manifold

            return recon_check_manifold(vertices, faces)

    def check_watertight(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> bool:
        """Check watertight status using native C++ kernel with Oracle fallback."""
        try:
            import synapse._c_geometry as _cg

            v_arr = np.ascontiguousarray(vertices, dtype=np.float64)
            f_arr = np.ascontiguousarray(faces, dtype=np.int64)
            if hasattr(_cg, "check_watertight"):
                return bool(_cg.check_watertight(v_arr, f_arr))
            from synapse.recon.geometry import check_watertight as recon_check_watertight

            return recon_check_watertight(vertices, faces)
        except (ImportError, AttributeError):
            from synapse.recon.geometry import check_watertight as recon_check_watertight

            return recon_check_watertight(vertices, faces)

    def solve_zero_slip_ik(
        self,
        skeleton_data: dict[str, Any],
        foot_contacts: np.ndarray,
    ) -> dict[str, Any]:
        """Solve foot zero-slip IK using synapse.recon.kinematics."""
        from synapse.recon.kinematics import solve_zero_slip_ik as recon_solve_ik

        return recon_solve_ik(skeleton_data, foot_contacts)

    def refine_pose_6dof(
        self,
        vertices: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Align 6-DoF pose using Kabsch/ICP via synapse.recon.geometry."""
        from synapse.recon.geometry import refine_pose_6dof as recon_refine_pose

        return recon_refine_pose(vertices, target_points)


def get_recon_backend(mock: bool = False) -> VgBackend:
    """Obtain a self-contained :class:`SynapseReconBackend` instance.

    Args:
        mock: If ``True`` (or env var ``SYNAPSE_VG_MOCK=1``), return a
            :class:`MockVgBackend`.

    Returns:
        A ready-to-use backend instance backed by ``synapse.recon``.
    """
    if mock or os.environ.get("SYNAPSE_VG_MOCK", "") == "1":
        return MockVgBackend()
    return SynapseReconBackend()


# Auto-register backend with Tier 0 bridge registry
from synapse.bridge.vg_bridge import register_recon_backend  # noqa: E402

register_recon_backend(SynapseReconBackend, get_recon_backend)
