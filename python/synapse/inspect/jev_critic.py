"""Visual Jev calibrated logits critic for 3D asset defect detection.

Provides:
- ``JevTransportProtocol``: Protocol for VLM logprobs endpoints
- ``MockJevTransport``: Deterministic test transport
- ``VisualJevCritic``: Temperature-scaled binary defect probe
  evaluator with severity-based action routing

Heavy dependencies are NOT required for this module (pure math).
"""

from __future__ import annotations

import datetime
import math
from typing import Literal, Protocol, runtime_checkable

from synapse.errors import (
    BackendUnavailableError,
    ContractValidationError,
)
from synapse.inspect.schemas import (
    DefectAnnotation3D,
    InspectionReport,
)

DefectCategory = Literal[
    "mesh_crumple",
    "texture_stretch",
    "seam_tearing",
    "floating_debris",
    "anatomical_distortion",
    "plastic_shading",
]

# -------------------------------------------------------------------
# Transport protocol
# -------------------------------------------------------------------


@runtime_checkable
class JevTransportProtocol(Protocol):
    """Protocol for VLM logprobs query endpoints."""

    def query_logprobs(
        self,
        image_uris: list[str],
        prompt: str,
    ) -> dict[str, float]:
        """Return top-token log-probabilities.

        Args:
            image_uris: URIs to inspection images.
            prompt: Binary Yes/No probe prompt.

        Returns:
            Mapping of token string to its log-probability,
            e.g. ``{"Yes": -0.5, "No": -1.2}``.
        """
        ...


# -------------------------------------------------------------------
# Mock transport for testing
# -------------------------------------------------------------------


_DEFECT_CATEGORIES: list[DefectCategory] = [
    "mesh_crumple",
    "texture_stretch",
    "seam_tearing",
    "floating_debris",
    "anatomical_distortion",
    "plastic_shading",
]


class MockJevTransport:
    """Deterministic test transport returning fake logprobs.

    For defect probes, returns configurable Yes/No logprobs based
    on per-category severity overrides.  Default behaviour returns
    a clean asset (No logprob >> Yes logprob).

    Args:
        defect_overrides: Mapping of category name to severity
            score in ``[0, 1]``.  Higher severity produces a
            higher ``Yes`` logprob.
        raise_on_connect: If ``True``, all queries raise
            ``ConnectionError`` to simulate network failure.
    """

    def __init__(
        self,
        defect_overrides: dict[str, float] | None = None,
        raise_on_connect: bool = False,
    ) -> None:
        self._overrides = defect_overrides or {}
        self._raise_on_connect = raise_on_connect

    def query_logprobs(
        self,
        image_uris: list[str],
        prompt: str,
    ) -> dict[str, float]:
        """Return synthetic logprobs for a defect probe.

        Raises:
            ConnectionError: If ``raise_on_connect`` is ``True``.
        """
        if self._raise_on_connect:
            raise ConnectionError(
                "Mock connection failure: endpoint unreachable"
            )

        category = self._extract_category(prompt)
        severity = self._overrides.get(category, 0.05)

        # Clamp to avoid log(0)
        severity = max(min(severity, 1.0 - 1e-10), 1e-10)

        yes_lp = math.log(severity)
        no_lp = math.log(1.0 - severity)
        return {"Yes": yes_lp, "No": no_lp}

    @staticmethod
    def _extract_category(prompt: str) -> str:
        """Parse probe prompt to identify defect category."""
        prompt_lower = prompt.lower()
        for cat in _DEFECT_CATEGORIES:
            # Normalise underscores to spaces for matching
            cat_spaced = cat.replace("_", " ")
            if cat_spaced in prompt_lower:
                return cat
            if cat in prompt_lower:
                return cat
            # Also match word stems: e.g. "crumpl" in "crumpling"
            words = cat.split("_")
            if all(w[:5] in prompt_lower for w in words):
                return cat
        return "unknown"


# -------------------------------------------------------------------
# Token normalization helpers
# -------------------------------------------------------------------

_YES_TOKENS = frozenset({"yes", "y", "true"})
_NO_TOKENS = frozenset({"no", "n", "false"})


def _extract_yes_no_logprobs(
    logprobs: dict[str, float],
    category: str,
) -> tuple[float, float]:
    """Extract normalized Yes/No log-probabilities from VLM response.

    Normalizes token keys by stripping whitespace and case-folding so
    that variants like ``" Yes"``, ``"yes"``, ``"NO"`` are matched.

    Raises:
        ContractValidationError: If either Yes or No token logprob is
            missing or non-finite (anti-silent-pass enforcement).
    """
    yes_candidates: list[float] = []
    no_candidates: list[float] = []

    for token, lp in logprobs.items():
        if not isinstance(lp, (int, float)) or not math.isfinite(lp):
            raise ContractValidationError(
                f"Non-finite logprob for token {token!r} in probe "
                f"'{category}': {lp}",
                agent_fix_hint=(
                    "Ensure VLM endpoint returns finite float logprobs "
                    "for top tokens."
                ),
            )
        norm = token.strip().casefold()
        if norm in _YES_TOKENS:
            yes_candidates.append(float(lp))
        elif norm in _NO_TOKENS:
            no_candidates.append(float(lp))

    if not yes_candidates or not no_candidates:
        raise ContractValidationError(
            f"Missing Yes/No token logprobs for probe '{category}'; "
            f"got keys {list(logprobs.keys())}. Silent fallback to "
            "clean asset is FORBIDDEN.",
            agent_fix_hint=(
                "Ensure VLM query requests top_logprobs >= 5 and binary "
                "Yes/No tokens are present in the response."
            ),
        )

    return max(yes_candidates), max(no_candidates)


# -------------------------------------------------------------------
# Visual Jev Critic
# -------------------------------------------------------------------


class VisualJevCritic:
    """Calibrated VLM-based 3D asset defect detector.

    Uses temperature-scaled logprobs from a visual language model
    to detect 6 categories of 3D asset defects.  Each defect
    probe is a binary Yes/No question; the first-token logprobs
    are converted to a calibrated probability via:

    .. math::

        P_{\\text{defect}} = \\frac{e^{z_{\\text{yes}}/T_c}}
        {e^{z_{\\text{yes}}/T_c} + e^{z_{\\text{no}}/T_c}}

    Severity is then routed to one of:
    ``accept``, ``auto_remesh``, ``auto_inpaint_texture``,
    or ``manual_hitl``.

    Args:
        transport: VLM logprobs transport backend.
        calibration_temperature: Temperature scaling factor
            for logprob calibration (default 1.5).
    """

    DEFECT_PROBES: list[DefectCategory] = list(_DEFECT_CATEGORIES)

    SEVERITY_HITL_THRESHOLD: float = 0.7
    SEVERITY_AUTO_THRESHOLD: float = 0.3
    SEVERITY_REPORT_THRESHOLD: float = 0.1

    _PROBE_PROMPTS: dict[DefectCategory, str] = {
        "mesh_crumple": (
            "Does this 3D rendered image show mesh crumpling, "
            "folding, or self-intersection artifacts? "
            "Answer Yes or No."
        ),
        "texture_stretch": (
            "Does this 3D rendered image show texture "
            "stretching, distortion, or UV mapping artifacts? "
            "Answer Yes or No."
        ),
        "seam_tearing": (
            "Does this 3D rendered image show visible seam "
            "tearing or edge discontinuity artifacts? "
            "Answer Yes or No."
        ),
        "floating_debris": (
            "Does this 3D rendered image show floating debris, "
            "disconnected mesh fragments, or stray vertices? "
            "Answer Yes or No."
        ),
        "anatomical_distortion": (
            "Does this 3D rendered image show anatomical "
            "distortion such as abnormal proportions, "
            "missing limbs, or impossible joint angles? "
            "Answer Yes or No."
        ),
        "plastic_shading": (
            "Does this 3D rendered image show unrealistic "
            "plastic-like shading or overly smooth specular "
            "highlights? Answer Yes or No."
        ),
    }

    def __init__(
        self,
        transport: JevTransportProtocol,
        calibration_temperature: float = 1.5,
    ) -> None:
        if calibration_temperature <= 0.0 or not math.isfinite(
            calibration_temperature
        ):
            raise ValueError(
                "calibration_temperature must be a positive finite float"
            )
        self._transport = transport
        self._tc = calibration_temperature

    def calibrated_probability(
        self,
        z_yes: float,
        z_no: float,
    ) -> float:
        """Compute temperature-scaled defect probability.

        Args:
            z_yes: Log-probability of ``Yes`` token.
            z_no: Log-probability of ``No`` token.

        Returns:
            Calibrated probability of defect presence in [0, 1].
        """
        # Numerically stable softmax with temperature
        a = z_yes / self._tc
        b = z_no / self._tc
        z_max = max(a, b)
        e_yes = math.exp(a - z_max)
        e_no = math.exp(b - z_max)
        return e_yes / (e_yes + e_no)

    def _route_action(
        self,
        category: str,
        severity: float,
    ) -> Literal[
        "auto_remesh",
        "auto_inpaint_texture",
        "manual_hitl",
        "accept",
    ]:
        """Route defect to remediation action by severity.

        Args:
            category: Defect taxonomy category.
            severity: Calibrated severity score in [0, 1].

        Returns:
            Suggested remediation action string.
        """
        if severity >= self.SEVERITY_HITL_THRESHOLD:
            return "manual_hitl"
        if severity < self.SEVERITY_AUTO_THRESHOLD:
            return "accept"
        # Medium severity: auto-fix based on category
        if category in ("mesh_crumple", "floating_debris"):
            return "auto_remesh"
        if category in (
            "texture_stretch",
            "seam_tearing",
            "plastic_shading",
        ):
            return "auto_inpaint_texture"
        # anatomical_distortion at medium -> HITL
        return "manual_hitl"

    def evaluate(
        self,
        asset_id: str,
        asset_name: str,
        image_uris: list[str],
    ) -> InspectionReport:
        """Run all 6 defect probes and assemble a report.

        Args:
            asset_id: Unique asset identifier.
            asset_name: Human-readable asset name.
            image_uris: URIs to rendered inspection images.

        Returns:
            :class:`InspectionReport` with defect annotations.

        Raises:
            BackendUnavailableError: If the transport connection
                fails (exit code 30).
            ContractValidationError: If logprob tokens are missing or
                non-finite (exit code 10).
        """
        try:
            defects: list[DefectAnnotation3D] = []

            for i, category in enumerate(self.DEFECT_PROBES):
                prompt = self._PROBE_PROMPTS.get(
                    category,
                    f"Is there a {category} defect? "
                    "Answer Yes or No.",
                )
                logprobs = self._transport.query_logprobs(
                    image_uris, prompt,
                )

                z_yes, z_no = _extract_yes_no_logprobs(
                    logprobs, category,
                )
                severity = self.calibrated_probability(
                    z_yes, z_no,
                )

                if severity > self.SEVERITY_REPORT_THRESHOLD:
                    action = self._route_action(
                        category, severity,
                    )
                    confidence = min(1.0, severity * 1.2)
                    defects.append(
                        DefectAnnotation3D(
                            defect_id=(
                                f"{asset_id}_defect_{i:03d}"
                            ),
                            category=category,
                            severity=round(severity, 6),
                            confidence=round(confidence, 6),
                            center_3d=[0.0, 0.0, 0.0],
                            bounding_radius_m=0.05,
                            affected_vertices_count=0,
                            suggested_action=action,
                        )
                    )

            requires_hitl = any(
                d.severity >= 0.8
                and d.category == "anatomical_distortion"
                for d in defects
            ) or any(
                d.suggested_action == "manual_hitl"
                for d in defects
            )

            max_sev = max(
                (d.severity for d in defects), default=0.0,
            )
            overall = max(0.0, 1.0 - max_sev)

            ts = datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat()

            return InspectionReport(
                asset_id=asset_id,
                asset_name=asset_name,
                inspection_timestamp=ts,
                overall_quality_score=round(overall, 6),
                defects=defects,
                requires_hitl=requires_hitl,
            )

        except ConnectionError as exc:
            raise BackendUnavailableError(
                f"Jev transport connection failed: {exc}",
                agent_fix_hint=(
                    "Check that the vLLM/OpenAI-compatible "
                    "endpoint is reachable and responding."
                ),
            ) from exc

