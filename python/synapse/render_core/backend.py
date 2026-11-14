# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Common backend interfaces and mock implementations for geometry & material ops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


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
        """Remove lighting from a radiance image using albedo and normals."""

    @abstractmethod
    def quad_remesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remesh a triangle soup into a quad-dominant mesh."""

    @abstractmethod
    def check_manifold(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> bool:
        """Check whether a mesh is 2-manifold and watertight."""

    @abstractmethod
    def solve_zero_slip_ik(
        self,
        skeleton_data: dict[str, Any],
        foot_contacts: np.ndarray,
    ) -> dict[str, Any]:
        """Solve zero-slip inverse kinematics for foot contacts."""

    @abstractmethod
    def refine_pose_6dof(
        self,
        vertices: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Estimate a rigid 6-DoF alignment transform."""


class MockVgBackend(VgBackend):
    """Mock backend returning passthrough / dummy values for testing."""

    def delight(
        self,
        albedo: np.ndarray,
        normal: np.ndarray,
        radiance: np.ndarray,
    ) -> np.ndarray:
        return albedo

    def quad_remesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return vertices, faces

    def check_manifold(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> bool:
        return True

    def solve_zero_slip_ik(
        self,
        skeleton_data: dict[str, Any],
        foot_contacts: np.ndarray,
    ) -> dict[str, Any]:
        return skeleton_data

    def refine_pose_6dof(
        self,
        vertices: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        return np.eye(4, dtype=np.float64)
