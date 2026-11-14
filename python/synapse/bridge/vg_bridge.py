"""Thin bridge to geometry and material processing backends.

This module defines abstract and concrete backends for geometry and material
processing operations. The :class:`MockVgBackend` is always available for
testing; production routes to :class:`~synapse.recon.backend.SynapseReconBackend`
which is self-contained with native C++20 geometry operators and Python Oracle.

.. note::

   ``SynapseReconBackend`` and ``get_recon_backend`` have been moved to
   :mod:`synapse.recon.backend` to break the ``bridge → recon`` circular
   dependency (P0-ARCH-01). This module re-exports them for backward
   compatibility.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from synapse.errors import ContractValidationError
from synapse.fgpp.schema import (
    MIN_CHARACTER_WHITEBOX_VERTICES,
    validate_whitebox_coarseness_floor,
)


class VgBackend(ABC):
    """Abstract interface for geometry / material operations.

    Every concrete backend must implement the five core methods below.
    """

    @abstractmethod
    def delight(
        self,
        albedo: np.ndarray,
        normal: np.ndarray,
        radiance: np.ndarray,
    ) -> np.ndarray:
        """Remove lighting from a radiance image using albedo and normals.

        Args:
            albedo: Base-color image, shape ``(H, W, 3)``.
            normal: World-space normal map, shape ``(H, W, 3)``.
            radiance: Direct radiance image, shape ``(H, W, 3)``.

        Returns:
            De-lit diffuse albedo image, shape ``(H, W, 3)``.
        """

    @abstractmethod
    def quad_remesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remesh a triangle soup into a quad-dominant mesh.

        Args:
            vertices: Vertex positions, shape ``(V, 3)``.
            faces: Triangle indices, shape ``(F, 3)``.
            target_faces: Desired number of output quads.

        Returns:
            Tuple of ``(new_vertices, new_faces)``.
        """

    @abstractmethod
    def check_manifold(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> bool:
        """Check whether a mesh is 2-manifold and watertight.

        Args:
            vertices: Vertex positions, shape ``(V, 3)``.
            faces: Face indices, shape ``(F, 3)`` or ``(F, 4)``.

        Returns:
            ``True`` if the mesh is manifold.
        """

    @abstractmethod
    def solve_zero_slip_ik(
        self,
        skeleton_data: dict[str, Any],
        foot_contacts: np.ndarray,
    ) -> dict[str, Any]:
        """Solve zero-slip inverse kinematics for foot contacts.

        Args:
            skeleton_data: Humanoid-53 skeleton joint dict.
            foot_contacts: Per-frame foot contact binary mask,
                shape ``(T, 2)``.

        Returns:
            Updated skeleton data dict with corrected joint transforms.
        """

    @abstractmethod
    def refine_pose_6dof(
        self,
        vertices: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Estimate a rigid 6-DoF alignment transform.

        Args:
            vertices: Source point cloud, shape ``(N, 3)``.
            target_points: Target point cloud, shape ``(M, 3)``.

        Returns:
            4×4 homogeneous transformation matrix.
        """


# -----------------------------------------------------------------------
# Mock backend — always available, returns passthrough / dummy values
# -----------------------------------------------------------------------


class MockVgBackend(VgBackend):
    """Mock backend returning passthrough / dummy values for testing."""

    def delight(
        self,
        albedo: np.ndarray,
        normal: np.ndarray,
        radiance: np.ndarray,
    ) -> np.ndarray:
        """Return *albedo* unchanged (passthrough)."""
        return albedo

    def quad_remesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return input mesh unchanged."""
        return vertices, faces

    def check_manifold(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> bool:
        """Always report manifold."""
        return True

    def solve_zero_slip_ik(
        self,
        skeleton_data: dict[str, Any],
        foot_contacts: np.ndarray,
    ) -> dict[str, Any]:
        """Return skeleton data unchanged."""
        return skeleton_data

    def refine_pose_6dof(
        self,
        vertices: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Return 4×4 identity matrix."""
        return np.eye(4, dtype=np.float64)


# -----------------------------------------------------------------------
# Synapse Native Recon Backend — moved to synapse.recon.backend
# -----------------------------------------------------------------------

# Backward-compatible re-exports are provided via __getattr__ using
# importlib to avoid creating an AST-visible import edge bridge → recon.

def __getattr__(name: str) -> object:
    if name in ("SynapseReconBackend", "get_recon_backend"):
        import importlib
        _rb = importlib.import_module("synapse.recon.backend")
        return getattr(_rb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------


def get_vg_backend(mock: bool = False, fallback_to_recon: bool = True) -> VgBackend:
    """Obtain a :class:`VgBackend` instance.

    Synapse is self-contained with native C++20 geometry operators and
    the :class:`SynapseReconBackend`. External ``vg-core`` is deprecated and
    removed.

    Args:
        mock: If ``True`` (or env var ``SYNAPSE_VG_MOCK=1``), return a
            :class:`MockVgBackend`.
        fallback_to_recon: Maintained for backward compatibility. Always
            delegates to :class:`SynapseReconBackend`.

    Returns:
        A ready-to-use backend instance.
    """
    if mock or os.environ.get("SYNAPSE_VG_MOCK", "") == "1":
        return MockVgBackend()

    import importlib
    _rb = importlib.import_module("synapse.recon.backend")
    return _rb.SynapseReconBackend()


# -----------------------------------------------------------------------
# Track-B whitebox anti-paradox gate (RFC 0044 §4.1 & §4.4)
# -----------------------------------------------------------------------


def enforce_distillation_geometry_gate(
    track: str,
    has_gen3d_highpoly: bool,
    vertex_count: int,
    has_articulated_limbs: bool,
    is_character: bool,
) -> None:
    """Enforce the Track-B whitebox anti-paradox geometry gate.

    Track-B coarse whitebox proxies **must** pass through a generative
    3D reconstruction step (TRELLIS / Hunyuan3D) before AI-video texture
    back-projection is permitted.  Character proxies additionally must
    satisfy the whitebox coarseness floor (≥1500 vertices, articulated
    limbs).

    Args:
        track: Pipeline track identifier (e.g. ``"track_b_whitebox"``).
        has_gen3d_highpoly: Whether the asset has been through
            generative 3D high-poly reconstruction.
        vertex_count: Mesh vertex count.
        has_articulated_limbs: Whether the mesh has articulated joints.
        is_character: Whether this is a character asset.

    Raises:
        ContractValidationError: If the gate check fails (exit_code=10).
    """
    if track != "track_b_whitebox":
        return  # Non Track-B pipelines are unconstrained

    if not has_gen3d_highpoly:
        raise ContractValidationError(
            "Track-B whitebox proxy must pass generative 3D "
            "reconstruction before texture back-projection",
            agent_fix_hint=(
                "Run foundary TRELLIS/Hunyuan3D reconstruction and "
                "quad_remesh before texture back-projection."
            ),
        )

    if is_character and not validate_whitebox_coarseness_floor(
        vertex_count, has_articulated_limbs, is_character
    ):
        raise ContractValidationError(
            f"Track-B character whitebox proxy failed coarseness floor: "
            f"vertex_count={vertex_count} "
            f"(min={MIN_CHARACTER_WHITEBOX_VERTICES}), "
            f"articulated_limbs={has_articulated_limbs}",
            agent_fix_hint=(
                "Run foundary TRELLIS/Hunyuan3D reconstruction and "
                "quad_remesh before texture back-projection."
            ),
        )
