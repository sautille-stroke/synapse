"""Preference dataset builder with 20% golden sample enforcement.

Provides:
- ``PreferencePair``: A single (chosen, rejected) preference pair
- ``PreferenceDatasetBuilder``: Merges self-learning and human
  preference data with mandatory 20% L0 golden sample ratio

**20% golden sample redline**: Human expert L0 golden samples must
be >= 20% of the final training dataset. If this ratio is not met,
the builder auto-oversamples the golden set or refuses to build.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from synapse.errors import ContractValidationError


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------


class PreferencePair(BaseModel):
    """A single preference pair for DPO training.

    Attributes:
        pair_id: Unique pair identifier.
        prompt: Task/defect description.
        chosen_uri: URI to the preferred (accepted) state.
        rejected_uri: URI to the rejected state.
        source: Origin of this pair (self-learning or human).
        category: LoRA category for targeted training.
        is_golden: Whether this is an L0 golden sample.
    """

    pair_id: str
    prompt: str = ""
    chosen_uri: str = ""
    rejected_uri: str = ""
    source: Literal["self", "human", "golden"] = "self"
    category: str = "style"
    is_golden: bool = False


# -------------------------------------------------------------------
# PreferenceDatasetBuilder
# -------------------------------------------------------------------


class PreferenceDatasetBuilder:
    """Build preference datasets with golden sample enforcement.

    Merges self-learning pairs (from auto-fix roundtrips) with
    human expert pairs, enforcing the 20% L0 golden minimum.

    Args:
        min_golden_ratio: Minimum ratio of L0 golden samples
            in the final dataset (default 0.20 = 20%).
        auto_oversample: If True, automatically oversample
            golden pairs to meet the ratio. If False, raise
            ContractValidationError when ratio is insufficient.
    """

    MIN_GOLDEN_RATIO: float = 0.20

    def __init__(
        self,
        min_golden_ratio: float = 0.20,
        auto_oversample: bool = True,
    ) -> None:
        self._min_ratio = max(min_golden_ratio, 0.0)
        self._auto_oversample = auto_oversample
        self._self_pairs: list[PreferencePair] = []
        self._human_pairs: list[PreferencePair] = []
        self._golden_pairs: list[PreferencePair] = []

    def add_self_pairs(
        self, pairs: list[PreferencePair],
    ) -> None:
        """Add self-learning preference pairs."""
        self._self_pairs.extend(pairs)

    def add_human_pairs(
        self, pairs: list[PreferencePair],
    ) -> None:
        """Add human expert preference pairs."""
        for p in pairs:
            self._human_pairs.append(p)

    def add_golden_pairs(
        self, pairs: list[PreferencePair],
    ) -> None:
        """Add L0 golden expert baseline pairs."""
        for p in pairs:
            # Force golden flag
            golden = p.model_copy(
                update={"is_golden": True, "source": "golden"},
            )
            self._golden_pairs.append(golden)

    def build_pairs(
        self,
    ) -> list[PreferencePair]:
        """Build the final preference dataset.

        Enforces the 20% L0 golden sample minimum:
        - If auto_oversample is True and golden ratio is below
          minimum, oversamples golden pairs to meet the ratio.
        - If auto_oversample is False and ratio is insufficient,
          raises ContractValidationError(10).

        Returns:
            Final merged list of preference pairs.

        Raises:
            ContractValidationError: If golden pairs are empty
                or ratio cannot be met (exit code 10).
        """
        if not self._golden_pairs:
            raise ContractValidationError(
                "L0 golden sample set is empty. Cannot build "
                "preference dataset without human expert "
                "golden baseline samples.",
                agent_fix_hint=(
                    "Provide at least one L0 golden expert "
                    "preference pair via add_golden_pairs(). "
                    "The 20% golden sample redline requires "
                    "human expert data."
                ),
            )

        # Merge all non-golden pairs
        all_non_golden = self._self_pairs + self._human_pairs
        n_non_golden = len(all_non_golden)
        n_golden = len(self._golden_pairs)

        # Check golden ratio
        total = n_non_golden + n_golden
        if total == 0:
            raise ContractValidationError(
                "Dataset is empty (no pairs provided).",
                agent_fix_hint=(
                    "Add preference pairs via add_self_pairs(), "
                    "add_human_pairs(), or add_golden_pairs()."
                ),
            )

        current_ratio = n_golden / total

        if current_ratio >= self._min_ratio:
            # Ratio is already satisfied
            return all_non_golden + self._golden_pairs

        # Ratio insufficient
        if not self._auto_oversample:
            raise ContractValidationError(
                f"L0 golden sample ratio {current_ratio:.2%} "
                f"is below minimum {self._min_ratio:.2%}. "
                f"Have {n_golden} golden / {total} total. "
                "Auto-oversample is disabled.",
                agent_fix_hint=(
                    "Add more L0 golden expert pairs or enable "
                    "auto_oversample=True to automatically "
                    "repeat golden samples."
                ),
            )

        # Auto-oversample: compute how many golden copies needed
        # We need: n_golden_final / (n_non_golden + n_golden_final)
        #   >= min_ratio
        # => n_golden_final >= min_ratio * n_non_golden
        #    / (1 - min_ratio)
        needed = math.ceil(
            self._min_ratio * n_non_golden
            / max(1.0 - self._min_ratio, 0.01)
        )
        needed = max(needed, 1)

        # Repeat golden pairs to meet the target
        oversampled: list[PreferencePair] = []
        idx = 0
        while len(oversampled) < needed:
            src = self._golden_pairs[
                idx % len(self._golden_pairs)
            ]
            oversampled.append(src.model_copy(
                update={
                    "pair_id": (
                        f"{src.pair_id}_os{idx:04d}"
                    ),
                },
            ))
            idx += 1

        return all_non_golden + oversampled

    def export_jsonl(
        self,
        output_path: Path,
    ) -> dict[str, Any]:
        """Build and export to JSONL.

        Returns:
            Dict with ``total``, ``golden_count``,
            ``golden_ratio``, ``output_path``.
        """
        pairs = self.build_pairs()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        golden_count = sum(1 for p in pairs if p.is_golden)
        total = len(pairs)
        ratio = golden_count / max(total, 1)

        with open(output_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(
                    json.dumps(p.model_dump(), default=str)
                    + "\n"
                )

        return {
            "total": total,
            "golden_count": golden_count,
            "golden_ratio": round(ratio, 4),
            "output_path": str(output_path),
        }


# -------------------------------------------------------------------
# Stage registration
# -------------------------------------------------------------------

from synapse.pipeline.bundle_worker import (  # noqa: E402
    register_stage,
)


@register_stage("rsi_dual_verify")
def _handle_rsi_dual_verify(
    manifest: Any,
) -> dict[str, Any]:
    """Run dual verification from a job bundle manifest."""
    from synapse.rsi.dual_verifier import DualVerifierEngine

    params = dict(manifest.params)
    sample_id = str(params.get("sample_id", "unknown"))
    physics = params.get("physics_metrics", {})
    prm_score = float(params.get("prm_score", 0.5))

    engine = DualVerifierEngine()
    verdict = engine.verify(
        sample_id=sample_id,
        physics_metrics=physics,
        prm_score=prm_score,
    )

    return {
        "output_paths": {},
        "verdict": verdict.model_dump(),
    }
