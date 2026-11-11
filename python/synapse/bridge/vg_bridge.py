"""Thin bridge to vg-core native pybind11 extensions.

This module defines abstract and concrete backends for calling into the
``vg-core`` C++/pybind11 geometry and material processing library.  The
:class:`MockVgBackend` is always available for testing; the
:class:`NativeVgBackend` delegates to ``import vg`` and raises
:class:`~synapse.errors.BackendUnavailableError` if the native package is
not installed.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from synapse.errors import BackendUnavailableError, ContractValidationError
from synapse.fgpp.schema import (
    MIN_CHARACTER_WHITEBOX_VERTICES,
    validate_whitebox_coarseness_floor,
)


class VgBackend(ABC):
    """Abstract interface for vg-core geometry / material operations.

    Every concrete backend must implement the five core methods below.
    Heavy native dependencies (``vg``) are lazy-imported only by
    :class:`NativeVgBackend`, keeping the base class dependency-free.
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
# Native backend — delegates to the real ``vg`` pybind11 module
# -----------------------------------------------------------------------

_VG_IMPORT_HINT = (
    "Install vg-core native extension: pip install vg-core"
)


class NativeVgBackend(VgBackend):
    """Native backend delegating to ``vg`` pybind11 C++ extension.

    Each method lazy-imports ``vg`` so that the bridge module itself
    remains importable without the native package installed.
    """

    @staticmethod
    def _import_vg() -> Any:
        """Lazy-import the ``vg`` native module or raise."""
        try:
            import vg  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BackendUnavailableError(
                "vg-core native extension is not installed",
                agent_fix_hint=_VG_IMPORT_HINT,
            ) from exc
        return vg

    def delight(
        self,
        albedo: np.ndarray,
        normal: np.ndarray,
        radiance: np.ndarray,
    ) -> np.ndarray:
        """Delegate to ``vg.material.delight``."""
        vg = self._import_vg()
        return vg.material.delight(albedo, normal, radiance)  # type: ignore[no-any-return]

    def quad_remesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Delegate to ``vg.geometry.quad_remesh``."""
        vg = self._import_vg()
        return vg.geometry.quad_remesh(vertices, faces, target_faces)  # type: ignore[no-any-return]

    def check_manifold(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> bool:
        """Delegate to ``vg.geometry.check_manifold``."""
        vg = self._import_vg()
        return vg.geometry.check_manifold(vertices, faces)  # type: ignore[no-any-return]

    def solve_zero_slip_ik(
        self,
        skeleton_data: dict[str, Any],
        foot_contacts: np.ndarray,
    ) -> dict[str, Any]:
        """Delegate to ``vg.ik.solve_zero_slip``."""
        vg = self._import_vg()
        return vg.ik.solve_zero_slip(skeleton_data, foot_contacts)  # type: ignore[no-any-return]

    def refine_pose_6dof(
        self,
        vertices: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Delegate to ``vg.geometry.refine_pose_6dof``."""
        vg = self._import_vg()
        return vg.geometry.refine_pose_6dof(vertices, target_points)  # type: ignore[no-any-return]


# -----------------------------------------------------------------------
# Synapse Native Recon Backend — self-contained geometric reconstruction
# -----------------------------------------------------------------------


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

        return recon_quad_remesh(vertices, faces, target_faces)

    def check_manifold(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> bool:
        """Check 2-manifold status using synapse.recon.geometry."""
        from synapse.recon.geometry import check_manifold as recon_check_manifold

        return recon_check_manifold(vertices, faces)

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


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------


def get_vg_backend(mock: bool = False, fallback_to_recon: bool = False) -> VgBackend:
    """Obtain a :class:`VgBackend` instance.

    Args:
        mock: If ``True`` (or env var ``SYNAPSE_VG_MOCK=1``), return a
            :class:`MockVgBackend` without attempting native import.
        fallback_to_recon: If ``True`` (or env var ``SYNAPSE_USE_RECON=1``),
            fall back to :class:`SynapseReconBackend` when native ``vg`` is not installed.

    Returns:
        A ready-to-use backend instance.

    Raises:
        BackendUnavailableError: If the native ``vg`` package cannot
            be imported, *mock* is ``False``, and *fallback_to_recon* is ``False``.
    """
    if mock or os.environ.get("SYNAPSE_VG_MOCK", "") == "1":
        return MockVgBackend()

    use_recon = fallback_to_recon or os.environ.get("SYNAPSE_USE_RECON", "") == "1"

    # Probe native availability
    try:
        import vg  # type: ignore[import-untyped]  # noqa: F401
        return NativeVgBackend()
    except ImportError as exc:
        if use_recon:
            return SynapseReconBackend()
        raise BackendUnavailableError(
            "vg-core native extension is not installed",
            agent_fix_hint=_VG_IMPORT_HINT + " (or use get_recon_backend() / fallback_to_recon=True)",
        ) from exc


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
