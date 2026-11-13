# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Kalman-filtered bounding-box tracker and affine alignment utilities.

Provides temporally smooth ROI tracking for anatomical local refinement
(RFC 0061).  The Kalman filter is implemented in pure NumPy — no external
filtering library is required.
"""

from __future__ import annotations

import numpy as np


def _xyxy_to_cxcywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` → ``[cx, cy, w, h]``."""
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float64)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)


def _cxcywh_to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, w, h]`` → ``[x1, y1, x2, y2]``."""
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class KalmanBBoxTracker:
    """Kalman filter for smooth temporal bounding-box tracking.

    State vector (8-D):
        ``[cx, cy, w, h, dcx, dcy, dw, dh]``
        — centre-x, centre-y, width, height and their velocities.

    Measurement vector (4-D):
        ``[cx, cy, w, h]``

    All internal computation uses float64 for numerical stability; the public
    API accepts and returns float32 arrays consistent with the rest of the
    refine pipeline.

    Args:
        process_noise: Diagonal process noise variance (``Q``).
        measurement_noise: Diagonal measurement noise variance (``R``).
    """

    _DIM_STATE: int = 8
    _DIM_MEAS: int = 4

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        self._q = float(process_noise)
        self._r = float(measurement_noise)
        # Declare instance variable types for mypy (assigned in reset())
        self._x: np.ndarray
        self._P: np.ndarray
        self._F: np.ndarray
        self._H: np.ndarray
        self._Q: np.ndarray
        self._R_mat: np.ndarray
        self._initialised: bool
        self.reset()

    # -- public API ----------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Propagate state one time-step forward.

        Returns:
            Predicted ``[cx, cy, w, h]`` as float32 array of shape ``(4,)``.
        """
        # x = F @ x
        self._x = self._F @ self._x
        # P = F @ P @ F^T + Q
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x[: self._DIM_MEAS].astype(np.float32)

    def update(self, bbox_xyxy: np.ndarray) -> np.ndarray:
        """Incorporate a new measurement and return the filtered bbox.

        Args:
            bbox_xyxy: Observed bounding box ``[x1, y1, x2, y2]`` (float32).

        Returns:
            Kalman-filtered bounding box ``[x1, y1, x2, y2]`` as float32.
        """
        z = _xyxy_to_cxcywh(np.asarray(bbox_xyxy, dtype=np.float64))

        if not self._initialised:
            self._x[: self._DIM_MEAS] = z
            self._initialised = True
            return _cxcywh_to_xyxy(self._x[: self._DIM_MEAS])

        # Innovation
        y = z - self._H @ self._x
        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R_mat
        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)
        # State update
        self._x = self._x + K @ y
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self._DIM_STATE, dtype=np.float64) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R_mat @ K.T

        return _cxcywh_to_xyxy(self._x[: self._DIM_MEAS])

    def reset(self) -> None:
        """Reset the tracker to its uninitialised state."""
        self._x = np.zeros(self._DIM_STATE, dtype=np.float64)
        self._P = np.eye(self._DIM_STATE, dtype=np.float64) * 10.0

        # State-transition matrix (constant-velocity model)
        self._F = np.eye(self._DIM_STATE, dtype=np.float64)
        for i in range(self._DIM_MEAS):
            self._F[i, i + self._DIM_MEAS] = 1.0

        # Observation matrix
        self._H = np.zeros((self._DIM_MEAS, self._DIM_STATE), dtype=np.float64)
        for i in range(self._DIM_MEAS):
            self._H[i, i] = 1.0

        # Process noise
        self._Q = np.eye(self._DIM_STATE, dtype=np.float64) * self._q

        # Measurement noise
        self._R_mat = np.eye(self._DIM_MEAS, dtype=np.float64) * self._r

        self._initialised = False


def compute_affine_align(
    bbox_xyxy: np.ndarray,
    target_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward and inverse 2×3 affine matrices aligning *bbox_xyxy*
    to a ``target_size × target_size`` square.

    The forward matrix maps the bounding-box region in the original image to
    the normalised square ``[0, target_size)²``.  The inverse matrix maps
    back.  Round-trip error is guaranteed < 1 × 10⁻⁴ px (exact analytical
    inverse).

    Args:
        bbox_xyxy: Bounding box ``[x1, y1, x2, y2]`` (float32 or float64).
        target_size: Side length of the target square in pixels.

    Returns:
        ``(M_forward, M_inverse)`` — each a ``(2, 3)`` float64 array.
    """
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    x1, y1, x2, y2 = bbox

    src_w = x2 - x1
    src_h = y2 - y1

    if src_w <= 0 or src_h <= 0:
        raise ValueError(
            f"Invalid bbox with non-positive dimensions: w={src_w}, h={src_h}"
        )

    sx = float(target_size) / src_w
    sy = float(target_size) / src_h

    # Forward:  dst = diag(sx, sy) @ (src - (x1, y1))
    M_fwd = np.array(
        [
            [sx, 0.0, -sx * x1],
            [0.0, sy, -sy * y1],
        ],
        dtype=np.float64,
    )

    # Inverse:  src = diag(1/sx, 1/sy) @ dst + (x1, y1)
    M_inv = np.array(
        [
            [1.0 / sx, 0.0, x1],
            [0.0, 1.0 / sy, y1],
        ],
        dtype=np.float64,
    )

    return M_fwd, M_inv
