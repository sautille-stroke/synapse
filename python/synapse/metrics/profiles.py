"""3-Tier quality gate profiles with 16-metric threshold evaluation.

Defines three gate profiles (``track_b_video``, ``track_a_asset``,
``rsi_promotion``) with increasing strictness, plus the gate evaluation
logic that compares measured metrics against profile thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ── Public type alias ────────────────────────────────────────────────
GateProfileName = Literal["track_b_video", "track_a_asset", "rsi_promotion"]

# Direction constants
_LOWER = "lower_is_better"
_HIGHER = "higher_is_better"

# Each entry: (threshold, direction)
# direction tells the gate whether the metric is "lower_is_better" or
# "higher_is_better".  For lower_is_better the measured value must be
# <= threshold; for higher_is_better it must be >= threshold.

_METRIC_DIRECTIONS: dict[str, str] = {
    # Temporal
    "temporal_warp_error": _LOWER,
    "flicker_score": _LOWER,
    "fvd": _LOWER,
    # FGPP fidelity
    "depth_rmse_m": _LOWER,
    "scale_invariant_depth_error": _LOWER,
    "normal_angular_error_deg": _LOWER,
    "flow_endpoint_error_px": _LOWER,
    "silhouette_iou": _HIGHER,
    # Roundtrip
    "chamfer_mm": _LOWER,
    "f_score_1mm": _HIGHER,
    "volume_breathing_pct": _LOWER,
    "foot_sliding_cm": _LOWER,
    "bone_length_drift_pct": _LOWER,
    "albedo_delta_e00": _LOWER,
    # Style consistency
    "border_ring_ciede2000": _LOWER,
    "light_dir_error_deg": _LOWER,
}

# ── Profile threshold tables ────────────────────────────────────────

TRACK_B_VIDEO: dict[str, float] = {
    "temporal_warp_error": 8.0,
    "flicker_score": 0.05,
    "fvd": 300.0,
    "depth_rmse_m": 0.50,
    "scale_invariant_depth_error": 0.30,
    "normal_angular_error_deg": 25.0,
    "flow_endpoint_error_px": 4.0,
    "silhouette_iou": 0.85,
    "chamfer_mm": 10.0,
    "f_score_1mm": 0.40,
    "volume_breathing_pct": 8.0,
    "foot_sliding_cm": 5.0,
    "bone_length_drift_pct": 5.0,
    "albedo_delta_e00": 8.0,
    "border_ring_ciede2000": 6.0,
    "light_dir_error_deg": 20.0,
}

TRACK_A_ASSET: dict[str, float] = {
    "temporal_warp_error": 4.0,
    "flicker_score": 0.02,
    "fvd": 150.0,
    "depth_rmse_m": 0.10,
    "scale_invariant_depth_error": 0.10,
    "normal_angular_error_deg": 10.0,
    "flow_endpoint_error_px": 2.0,
    "silhouette_iou": 0.92,
    "chamfer_mm": 3.0,
    "f_score_1mm": 0.70,
    "volume_breathing_pct": 3.0,
    "foot_sliding_cm": 2.0,
    "bone_length_drift_pct": 2.0,
    "albedo_delta_e00": 4.0,
    "border_ring_ciede2000": 3.0,
    "light_dir_error_deg": 10.0,
}

RSI_PROMOTION: dict[str, float] = {
    "temporal_warp_error": 2.0,
    "flicker_score": 0.008,
    "fvd": 50.0,
    "depth_rmse_m": 0.05,
    "scale_invariant_depth_error": 0.05,
    "normal_angular_error_deg": 5.0,
    "flow_endpoint_error_px": 1.0,
    "silhouette_iou": 0.97,
    "chamfer_mm": 1.0,
    "f_score_1mm": 0.90,
    "volume_breathing_pct": 1.5,
    "foot_sliding_cm": 0.5,
    "bone_length_drift_pct": 0.5,
    "albedo_delta_e00": 2.0,
    "border_ring_ciede2000": 1.5,
    "light_dir_error_deg": 5.0,
}

_PROFILES: dict[GateProfileName, dict[str, float]] = {
    "track_b_video": TRACK_B_VIDEO,
    "track_a_asset": TRACK_A_ASSET,
    "rsi_promotion": RSI_PROMOTION,
}


# ── Data classes ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateViolation:
    """A single metric threshold violation."""

    metric_name: str
    measured: float
    threshold: float
    direction: str


@dataclass(frozen=True)
class GateVerdict:
    """Result of evaluating measured metrics against a gate profile."""

    passed: bool
    profile: GateProfileName
    violations: list[GateViolation] = field(default_factory=list)
    measured: dict[str, float] = field(default_factory=dict)


# ── Evaluation entry-point ───────────────────────────────────────────

def evaluate_metrics_gate(
    measured: dict[str, float],
    profile: GateProfileName,
    scenario_overrides: dict[str, float] | None = None,
) -> GateVerdict:
    """Evaluate *measured* metrics against the thresholds of *profile*.

    Args:
        measured: Mapping of metric name to measured value.
        profile: One of ``"track_b_video"``, ``"track_a_asset"``,
            ``"rsi_promotion"``.
        scenario_overrides: Optional per-metric threshold overrides that
            take precedence over the profile defaults.

    Returns:
        A :class:`GateVerdict` indicating pass/fail and listing all
        violations.
    """
    thresholds = dict(_PROFILES[profile])  # shallow copy
    if scenario_overrides:
        thresholds.update(scenario_overrides)

    violations: list[GateViolation] = []
    for name, thresh in thresholds.items():
        if name not in measured:
            continue
        value = float(measured[name])
        direction = _METRIC_DIRECTIONS.get(name, _LOWER)
        violated = False
        if direction == _LOWER and value > thresh:
            violated = True
        elif direction == _HIGHER and value < thresh:
            violated = True
        if violated:
            violations.append(
                GateViolation(
                    metric_name=name,
                    measured=value,
                    threshold=thresh,
                    direction=direction,
                )
            )

    return GateVerdict(
        passed=len(violations) == 0,
        profile=profile,
        violations=violations,
        measured=dict(measured),
    )
