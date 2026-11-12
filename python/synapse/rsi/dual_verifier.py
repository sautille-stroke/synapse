"""Dual physical + PRM verifier engine for RSI candidates.

Provides:
- ``PhysicsCheckResult``: Result of deterministic physics checks
- ``DualVerifierEngine``: Conjunctive verifier that enforces
  physical geometry checks BEFORE any neural evaluation

**Anti echo-chamber redline**: Every RSI candidate MUST pass at
least one closed-form physical geometry check. Pure neural
self-evaluation as sole judge is STRICTLY FORBIDDEN.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field

from synapse.errors import ContractValidationError

# -------------------------------------------------------------------
# Physical check thresholds
# -------------------------------------------------------------------


class PhysicsThresholds(BaseModel):
    """Deterministic physical verification thresholds.

    Attributes:
        interpenetration_volume_ratio: Max interpenetration volume.
        foot_slide_velocity_ms: Max foot sliding speed (m/s).
        silhouette_iou: Minimum silhouette IoU.
        depth_rmse_m: Maximum depth RMSE in metres.
        normal_angular_error_deg: Maximum normal error in degrees.
        manifold_watertight: Whether manifold must be watertight.
    """

    interpenetration_volume_ratio: float = Field(
        default=0.001, ge=0.0,
    )
    foot_slide_velocity_ms: float = Field(
        default=0.02, ge=0.0,
    )
    silhouette_iou: float = Field(default=0.90, ge=0.0, le=1.0)
    depth_rmse_m: float = Field(default=0.03, ge=0.0)
    normal_angular_error_deg: float = Field(
        default=12.0, ge=0.0,
    )
    manifold_watertight: bool = True


# -------------------------------------------------------------------
# Failure attribution labels
# -------------------------------------------------------------------

# Maps failure reason to LoRA category for targeted retraining
_FAILURE_TO_LORA_BUCKET: dict[str, str] = {
    "SILHOUETTE_DRIFT": "control",
    "DEPTH_INCONSISTENCY": "control",
    "NORMAL_ERROR": "control",
    "FOOT_SLIPPING": "motion",
    "INTERPENETRATION": "motion",
    "COLOR_DRIFT": "style",
    "TEXTURE_ARTIFACT": "style",
    "IDENTITY_LOSS": "identity",
    "WHITEBOX_MISMATCH": "whitebox",
    "NON_MANIFOLD": "control",
}


# -------------------------------------------------------------------
# Physics check result
# -------------------------------------------------------------------


class PhysicsCheckResult(BaseModel):
    """Result of deterministic physical geometry checks.

    Attributes:
        passed: Whether ALL physics checks passed.
        checks: Individual check results.
        failures: List of failure attribution labels.
        lora_buckets: LoRA categories to retrain.
    """

    passed: bool = True
    checks: dict[str, Any] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    lora_buckets: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------
# Verification verdict
# -------------------------------------------------------------------


class VerificationVerdict(BaseModel):
    """Combined dual-verification result.

    Attributes:
        sample_id: Candidate sample identifier.
        physics_passed: Whether physics checks passed.
        prm_score: Process Reward Model score (0-1).
        verdict: Final verdict (ACCEPT / HARD_NEGATIVE / REJECT).
        failure_attributions: Failure reason labels.
        target_lora_buckets: LoRA categories for retraining.
        physics_details: Detailed physics check results.
    """

    sample_id: str
    physics_passed: bool = True
    prm_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verdict: Literal[
        "ACCEPT", "HARD_NEGATIVE", "REJECT",
    ] = "ACCEPT"
    failure_attributions: list[str] = Field(
        default_factory=list,
    )
    target_lora_buckets: list[str] = Field(
        default_factory=list,
    )
    physics_details: dict[str, Any] = Field(
        default_factory=dict,
    )


# -------------------------------------------------------------------
# DualVerifierEngine
# -------------------------------------------------------------------


class DualVerifierEngine:
    """Conjunctive physical + PRM dual verifier.

    Enforces the anti echo-chamber redline: every candidate
    sample MUST first pass deterministic physics checks before
    any neural (PRM/VLM) evaluation is considered.

    Args:
        thresholds: Physics check thresholds.
        prm_accept_threshold: PRM score threshold for ACCEPT.
        prm_reject_threshold: PRM score below this is REJECT.
    """

    def __init__(
        self,
        thresholds: PhysicsThresholds | None = None,
        prm_accept_threshold: float = 0.7,
        prm_reject_threshold: float = 0.3,
    ) -> None:
        self._thresholds = thresholds or PhysicsThresholds()
        self._prm_accept = prm_accept_threshold
        self._prm_reject = prm_reject_threshold

    @property
    def thresholds(self) -> PhysicsThresholds:
        """Return the physics thresholds."""
        return self._thresholds

    def verify_physics(
        self,
        sample_id: str,
        physics_metrics: dict[str, float],
    ) -> PhysicsCheckResult:
        """Run deterministic physics verification.

        This check is MANDATORY and must run BEFORE any neural
        evaluation. Missing or non-finite required metrics cause an
        immediate ``ContractValidationError(10)``.

        Args:
            sample_id: Candidate sample identifier.
            physics_metrics: Dict of measured physical metrics.
                Required keys: ``silhouette_iou``,
                ``depth_rmse_m``, ``normal_angular_error_deg``.
                Optional: ``interpenetration_volume_ratio``,
                ``foot_slide_velocity_ms``,
                ``is_manifold_watertight``.

        Returns:
            Physics check result with pass/fail and attributions.

        Raises:
            ContractValidationError: If required physics metrics
                are missing or non-finite (exit code 10). This enforces
                the anti echo-chamber redline.
        """
        required_keys = [
            "silhouette_iou",
            "depth_rmse_m",
            "normal_angular_error_deg",
        ]
        missing = [
            k for k in required_keys
            if k not in physics_metrics
        ]
        if missing:
            raise ContractValidationError(
                f"Missing required physics metrics for sample "
                f"'{sample_id}': {missing}. "
                "Pure neural self-evaluation without physical "
                "geometry checks is FORBIDDEN.",
                agent_fix_hint=(
                    "Provide all required physics metrics: "
                    + ", ".join(required_keys)
                    + ". Run vg_bridge / fgpp_fidelity to "
                    "compute them."
                ),
            )

        # R1 fix: reject NaN / Inf values across all physics metrics
        for key, val in physics_metrics.items():
            if not isinstance(val, (int, float)) or not math.isfinite(
                float(val)
            ):
                raise ContractValidationError(
                    f"Non-finite physics metric '{key}'={val!r} for "
                    f"sample '{sample_id}'. NaN/Inf values are "
                    "FORBIDDEN in physical verification.",
                    agent_fix_hint=(
                        f"Check geometry/depth computation for '{key}' "
                        "to eliminate NaN/Inf."
                    ),
                )

        t = self._thresholds
        checks: dict[str, Any] = {}
        failures: list[str] = []
        buckets: list[str] = []

        # Silhouette IoU
        sil = physics_metrics["silhouette_iou"]
        checks["silhouette_iou"] = {
            "value": sil,
            "threshold": t.silhouette_iou,
            "passed": sil >= t.silhouette_iou,
        }
        if sil < t.silhouette_iou:
            failures.append("SILHOUETTE_DRIFT")

        # Depth RMSE
        depth = physics_metrics["depth_rmse_m"]
        checks["depth_rmse_m"] = {
            "value": depth,
            "threshold": t.depth_rmse_m,
            "passed": depth <= t.depth_rmse_m,
        }
        if depth > t.depth_rmse_m:
            failures.append("DEPTH_INCONSISTENCY")

        # Normal angular error
        nae = physics_metrics["normal_angular_error_deg"]
        checks["normal_angular_error_deg"] = {
            "value": nae,
            "threshold": t.normal_angular_error_deg,
            "passed": nae <= t.normal_angular_error_deg,
        }
        if nae > t.normal_angular_error_deg:
            failures.append("NORMAL_ERROR")

        # Optional: interpenetration
        interp = physics_metrics.get(
            "interpenetration_volume_ratio",
        )
        if interp is not None:
            checks["interpenetration_volume_ratio"] = {
                "value": interp,
                "threshold": t.interpenetration_volume_ratio,
                "passed": interp <= t.interpenetration_volume_ratio,
            }
            if interp > t.interpenetration_volume_ratio:
                failures.append("INTERPENETRATION")

        # Optional: foot sliding
        fs = physics_metrics.get("foot_slide_velocity_ms")
        if fs is not None:
            checks["foot_slide_velocity_ms"] = {
                "value": fs,
                "threshold": t.foot_slide_velocity_ms,
                "passed": fs <= t.foot_slide_velocity_ms,
            }
            if fs > t.foot_slide_velocity_ms:
                failures.append("FOOT_SLIPPING")

        # Optional: manifold watertight
        wt = physics_metrics.get("is_manifold_watertight")
        if wt is not None and t.manifold_watertight:
            checks["is_manifold_watertight"] = {
                "value": wt,
                "required": True,
                "passed": bool(wt),
            }
            if not wt:
                failures.append("NON_MANIFOLD")

        # Map failures to LoRA buckets
        for f in failures:
            bucket = _FAILURE_TO_LORA_BUCKET.get(f)
            if bucket and bucket not in buckets:
                buckets.append(bucket)

        return PhysicsCheckResult(
            passed=len(failures) == 0,
            checks=checks,
            failures=failures,
            lora_buckets=buckets,
        )

    def verify(
        self,
        sample_id: str,
        physics_metrics: dict[str, float],
        prm_score: float = 0.5,
    ) -> VerificationVerdict:
        """Run dual physics + PRM verification.

        Physics checks are run FIRST. If physics fails, the
        sample is classified as HARD_NEGATIVE regardless of
        PRM score (anti echo-chamber enforcement).

        Args:
            sample_id: Candidate sample identifier.
            physics_metrics: Physical verification metrics.
            prm_score: Neural PRM score in [0, 1].

        Returns:
            Combined verification verdict.
        """
        if not math.isfinite(prm_score) or not (0.0 <= prm_score <= 1.0):
            raise ContractValidationError(
                f"Invalid prm_score={prm_score!r} for sample "
                f"'{sample_id}'; must be finite in [0, 1].",
                agent_fix_hint="Provide a finite PRM score in [0, 1].",
            )

        physics = self.verify_physics(
            sample_id, physics_metrics,
        )

        if not physics.passed:
            # Physics failure -> HARD_NEGATIVE regardless of PRM
            return VerificationVerdict(
                sample_id=sample_id,
                physics_passed=False,
                prm_score=prm_score,
                verdict="HARD_NEGATIVE",
                failure_attributions=physics.failures,
                target_lora_buckets=physics.lora_buckets,
                physics_details=physics.checks,
            )

        # Physics passed: use PRM score for final verdict
        verdict: Literal["ACCEPT", "HARD_NEGATIVE", "REJECT"]
        if prm_score >= self._prm_accept:
            verdict = "ACCEPT"
        elif prm_score < self._prm_reject:
            verdict = "REJECT"
        else:
            verdict = "HARD_NEGATIVE"

        return VerificationVerdict(
            sample_id=sample_id,
            physics_passed=True,
            prm_score=prm_score,
            verdict=verdict,
            failure_attributions=[],
            target_lora_buckets=[],
            physics_details=physics.checks,
        )
