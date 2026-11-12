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

from enum import IntEnum
from typing import Any


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
        """Serialize to JSON-safe dict for ``--json`` CLI output."""
        return {
            "status": "error",
            "exit_code": int(self.exit_code),
            "error_type": type(self).__name__,
            "message": str(self),
            "agent_fix_hint": self.agent_fix_hint,
            "trace_id": trace_id,
            "details": self.details,
        }


class ContractValidationError(SynapseError):
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

