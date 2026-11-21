"""Structured error hierarchy with deterministic exit codes.

Every ``SynapseError`` carries an ``exit_code`` (per AGENTS.md §3) and an
``agent_fix_hint`` field that upstream orchestrating agents can consume
to auto-remediate failures without human intervention.

Exit code mapping::

    0  — OK
    10 — FGPP / StyleProfile contract validation failure
    20 — Missing scene, asset, or EXR file
    30 — GPU OOM or inference backend unreachable
    40 — HITL 15-min circuit breaker or quality gate not met
    50 — Stage or CLI subcommand not yet implemented
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal


class ExitCode(IntEnum):
    """Deterministic process exit codes per AGENTS.md §3."""

    OK = 0
    CONTRACT_ERROR = 10
    RESOURCE_MISSING = 20
    BACKEND_UNAVAILABLE = 30
    CIRCUIT_BREAKER = 40
    NOT_IMPLEMENTED = 50


class SynapseError(Exception):
    """Base error for all Synapse operations.

    Every SynapseError carries:

    - ``exit_code``: deterministic process exit code
    - ``agent_fix_hint``: machine-readable fix suggestion for upstream agents
    - ``details``: arbitrary metadata dict for structured JSON output
    """

    def __init__(
        self,
        message: str,
        exit_code: ExitCode = ExitCode.CONTRACT_ERROR,
        agent_fix_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.agent_fix_hint = agent_fix_hint
        self.details = details or {}

    def to_json_dict(self, trace_id: str = "") -> dict[str, Any]:
        """Serialize to JSON-safe dict for ``--json`` CLI output.

        Recursively sanitizes values: ``pathlib.Path`` → ``str``,
        NumPy scalars → Python float/int, ``NaN``/``Inf`` → ``null``.
        """

        def _sanitize(obj: Any) -> Any:
            """Recursively convert non-JSON-safe types."""
            from pathlib import PurePath

            if isinstance(obj, PurePath):
                return str(obj)
            if isinstance(obj, set):
                return sorted(str(v) for v in obj)
            if isinstance(obj, dict):
                return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            # NumPy ndarray → nested list
            if hasattr(obj, "tolist") and hasattr(obj, "shape"):
                try:
                    return obj.tolist()
                except (ValueError, AttributeError):
                    return str(obj)
            # NumPy scalar → Python scalar
            if hasattr(obj, "item"):
                try:
                    obj = obj.item()
                except (ValueError, AttributeError):
                    pass
            # NaN / Inf → None (JSON null)
            if isinstance(obj, float) and not math.isfinite(obj):
                return None
            return obj

        return {
            "status": "error",
            "exit_code": int(self.exit_code),
            "error_type": type(self).__name__,
            "message": str(self),
            "agent_fix_hint": self.agent_fix_hint,
            "trace_id": trace_id,
            "details": _sanitize(self.details),
        }


class ContractValidationError(SynapseError, ValueError):
    """FGPP schema / contract violation (exit_code=10).

    Raised when metadata.json fails schema validation, channel names
    mismatch, or other FGPP wire-format invariants are broken.
    """

    def __init__(
        self,
        message: str,
        agent_fix_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            exit_code=ExitCode.CONTRACT_ERROR,
            agent_fix_hint=agent_fix_hint,
            details=details,
        )


class AssetNotFoundError(SynapseError):
    """Missing scene, asset, or EXR file (exit_code=20).

    Raised when a required input path does not exist or SHA256
    verification fails.
    """

    def __init__(
        self,
        message: str,
        agent_fix_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            exit_code=ExitCode.RESOURCE_MISSING,
            agent_fix_hint=agent_fix_hint,
            details=details,
        )


class BackendUnavailableError(SynapseError):
    """GPU OOM or inference backend unreachable (exit_code=30).

    Raised when CUDA is unavailable, GPU memory is exhausted, or the
    downstream inference service cannot be contacted.
    """

    def __init__(
        self,
        message: str,
        agent_fix_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            exit_code=ExitCode.BACKEND_UNAVAILABLE,
            agent_fix_hint=agent_fix_hint,
            details=details,
        )


class CircuitBreakerError(SynapseError):
    """HITL 15-minute timeout or engineering circuit breaker (exit_code=40).

    Raised when a single asset or shot exceeds the 15-minute manual
    refinement budget defined in RFC 0044 §4.4.
    """

    def __init__(
        self,
        message: str,
        agent_fix_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            exit_code=ExitCode.CIRCUIT_BREAKER,
            agent_fix_hint=agent_fix_hint,
            details=details,
        )


class QualityGateError(SynapseError):
    """Metric threshold not met (exit_code=40).

    Raised when a rendered output fails quality gate checks such as
    PSNR, SSIM, LPIPS, or temporal warp error thresholds.
    """

    def __init__(
        self,
        message: str,
        agent_fix_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            exit_code=ExitCode.CIRCUIT_BREAKER,
            agent_fix_hint=agent_fix_hint,
            details=details,
        )


class NotImplementedStageError(SynapseError):
    """Stage or CLI subcommand not yet implemented (exit_code=50).

    Raised when a CLI command or bundle worker stage is only a
    scaffold and has no real execution implementation, preventing
    silent success reports.
    """

    def __init__(
        self,
        message: str,
        agent_fix_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            exit_code=ExitCode.NOT_IMPLEMENTED,
            agent_fix_hint=agent_fix_hint,
            details=details,
        )


# ---------------------------------------------------------------------------
# HITL circuit breaker (Tier-0 — moved here to break refine → bridge cycle)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_SECONDS: float = 900.0  # 15 minutes


def evaluate_hitl_circuit_breaker(
    elapsed_seconds: float,
    domain: Literal["video_shot", "asset_3d"],
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> None:
    """Evaluate the HITL 15-minute circuit breaker (RFC 0044 §4.4).

    If the elapsed time exceeds *max_seconds*, a
    :class:`CircuitBreakerError` is raised with a domain-appropriate
    fallback hint.

    Args:
        elapsed_seconds: Cumulative wall-clock time spent on manual /
            agent refinement for this single asset or shot.
        domain: ``"video_shot"`` or ``"asset_3d"`` — determines the
            fallback action and fix hint.
        max_seconds: Budget ceiling in seconds (default 900 = 15 min).

    Raises:
        ContractValidationError: If ``elapsed_seconds`` is NaN, non-finite,
            negative, or if ``domain`` is invalid (exit_code=10).
        CircuitBreakerError: If ``elapsed_seconds > max_seconds``
            (exit_code=40).
    """
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise ContractValidationError(
            f"elapsed_seconds must be a finite, non-negative number, got {elapsed_seconds!r}",
            agent_fix_hint="Ensure elapsed timer is initialized and valid before calling circuit breaker.",
        )

    if domain not in ("video_shot", "asset_3d"):
        raise ContractValidationError(
            f"Invalid HITL circuit breaker domain {domain!r}, must be 'video_shot' or 'asset_3d'",
            agent_fix_hint="Specify domain='video_shot' or domain='asset_3d'.",
        )

    if not math.isfinite(max_seconds) or max_seconds <= 0.0:
        raise ContractValidationError(
            f"max_seconds must be a finite positive number, got {max_seconds!r}",
            agent_fix_hint="Specify a positive number of seconds for max_seconds.",
        )

    if elapsed_seconds <= max_seconds:
        return

    msg = f"HITL circuit breaker: {elapsed_seconds:.1f}s > {max_seconds:.1f}s limit for {domain}"

    if domain == "video_shot":
        raise CircuitBreakerError(
            msg,
            agent_fix_hint=(
                "Fallback to FGPP re-bake from updated 3D. "
                "Reduce complexity or accept quality trade-off."
            ),
        )
    else:  # asset_3d
        raise CircuitBreakerError(
            msg,
            agent_fix_hint=(
                "Fallback to single-view decal projection or "
                "procedural material. Skip manual refinement."
            ),
        )
