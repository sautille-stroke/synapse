"""Preference dataset builder with 20% golden sample enforcement.

Provides:
- ``PreferencePair`` / ``PreferenceRecord``: Unified preference pair schema
  connecting ``synapse.inspect.flywheel_exporter`` (`DPORecord`) and
  ``synapse.rsi.pair_builder``
- ``PreferenceDatasetBuilder``: Merges self-learning and human
  preference data with mandatory 20% L0 golden sample ratio

**20% golden sample redline**: Human expert L0 golden samples must
be >= 20% of the final training dataset. If this ratio is not met,
the builder auto-oversamples the golden set (up to ``max_oversample_factor``)
or refuses to build.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from synapse.errors import ContractValidationError

# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------


class PreferencePair(BaseModel):
    """A single preference pair for DPO training (unified ``PreferenceRecord``).

    Attributes:
        pair_id: Unique pair identifier.
        prompt: Task/defect description.
        chosen_uri: URI to the preferred (accepted) state.
        rejected_uri: URI to the rejected state.
        condition_key: Shared initial state / FGPP condition URI.
        asset_id: Asset or scenario identifier.
        source_intent_id: Provenance intent ID from HITL / inspection log.
        source: Origin of this pair (self-learning, human, or golden).
        category: LoRA category for targeted training.
        is_golden: Whether this is an L0 golden sample.
    """

    pair_id: str
    prompt: str = ""
    chosen_uri: str = ""
    rejected_uri: str = ""
    condition_key: str = ""
    asset_id: str = ""
    source_intent_id: str = ""
    source: Literal["self", "human", "golden"] = "self"
    category: str = "style"
    is_golden: bool = False

    @model_validator(mode="after")
    def _validate_distinct_uris(self) -> PreferencePair:
        if (
            self.chosen_uri
            and self.rejected_uri
            and self.chosen_uri == self.rejected_uri
        ):
            raise ValueError(
                f"PreferencePair '{self.pair_id}' has identical "
                f"chosen_uri and rejected_uri ({self.chosen_uri!r})."
            )
        return self

    @classmethod
    def from_dpo_record(
        cls,
        record: Any,
        *,
        source: Literal["self", "human", "golden"] = "human",
        category: str = "style",
        is_golden: bool = False,
    ) -> PreferencePair:
        """Convert a ``DPORecord`` (or dict) from ``flywheel_exporter`` into ``PreferencePair``."""
        data = (
            record.model_dump()
            if hasattr(record, "model_dump")
            else dict(record)
        )
        pair_id = str(
            data.get("pair_id")
            or data.get("source_intent_id")
            or f"dpo_{data.get('asset_id', 'asset')}"
        )
        return cls(
            pair_id=pair_id,
            prompt=str(data.get("prompt", "")),
            chosen_uri=str(
                data.get("chosen_uri") or data.get("chosen", "")
            ),
            rejected_uri=str(
                data.get("rejected_uri") or data.get("rejected", "")
            ),
            condition_key=str(
                data.get("condition_key") or data.get("state_before", "")
            ),
            asset_id=str(data.get("asset_id", "")),
            source_intent_id=str(data.get("source_intent_id", "")),
            source="golden" if is_golden else source,
            category=str(data.get("category", category)),
            is_golden=is_golden or source == "golden",
        )


PreferenceRecord = PreferencePair


# -------------------------------------------------------------------
# PreferenceDatasetBuilder
# -------------------------------------------------------------------


class PreferenceDatasetBuilder:
    """Build preference datasets with golden sample enforcement.

    Merges self-learning pairs (from auto-fix roundtrips) with
    human expert pairs, enforcing the 20% L0 golden minimum and
    bounding oversampling to prevent extreme duplicate memorization.

    Args:
        min_golden_ratio: Minimum ratio of L0 golden samples
            in the final dataset (default 0.20 = 20%, must be in [0.20, 0.95]).
        auto_oversample: If True, automatically oversample
            golden pairs to meet the ratio. If False, raise
            ContractValidationError when ratio is insufficient.
        max_oversample_factor: Maximum times a single golden pair
            may be repeated during auto-oversampling (default 50).
    """

    MIN_GOLDEN_RATIO: float = 0.20

    def __init__(
        self,
        min_golden_ratio: float = 0.20,
        auto_oversample: bool = True,
        max_oversample_factor: int = 50,
    ) -> None:
        if (
            not math.isfinite(min_golden_ratio)
            or min_golden_ratio < self.MIN_GOLDEN_RATIO
            or min_golden_ratio > 0.95
        ):
            raise ContractValidationError(
                f"Invalid min_golden_ratio={min_golden_ratio!r}; must be in "
                f"[{self.MIN_GOLDEN_RATIO}, 0.95] to satisfy the 20% L0 golden redline.",
                agent_fix_hint="Set min_golden_ratio between 0.20 and 0.95.",
            )
        if max_oversample_factor < 1:
            raise ContractValidationError(
                f"Invalid max_oversample_factor={max_oversample_factor!r}; must be >= 1.",
                agent_fix_hint="Set max_oversample_factor >= 1 (e.g. 5 or 10).",
            )
        self._min_ratio = float(min_golden_ratio)
        self._auto_oversample = auto_oversample
        self._max_oversample_factor = int(max_oversample_factor)
        self._self_pairs: list[PreferencePair] = []
        self._human_pairs: list[PreferencePair] = []
        self._golden_pairs: list[PreferencePair] = []

    @staticmethod
    def _deduplicate(pairs: list[PreferencePair]) -> list[PreferencePair]:
        """Deduplicate pairs by (pair_id, condition_key, chosen_uri, rejected_uri)."""
        seen: set[tuple[str, str, str, str]] = set()
        out: list[PreferencePair] = []
        for p in pairs:
            key = (p.pair_id, p.condition_key, p.chosen_uri, p.rejected_uri)
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    def add_self_pairs(
        self,
        pairs: list[PreferencePair],
    ) -> None:
        """Add self-learning preference pairs."""
        self._self_pairs.extend(pairs)

    def add_human_pairs(
        self,
        pairs: list[PreferencePair],
    ) -> None:
        """Add human expert preference pairs."""
        for p in pairs:
            self._human_pairs.append(p)

    def add_golden_pairs(
        self,
        pairs: list[PreferencePair],
    ) -> None:
        """Add L0 golden expert baseline pairs."""
        for p in pairs:
            golden = p.model_copy(
                update={"is_golden": True, "source": "golden"},
            )
            self._golden_pairs.append(golden)

    def build_pairs(
        self,
    ) -> list[PreferencePair]:
        """Build the final preference dataset.

        Enforces the 20% L0 golden sample minimum:
        - Deduplicates inputs before computing ratios.
        - If auto_oversample is True and golden ratio is below
          minimum, oversamples golden pairs to meet the ratio
          (bounded by ``max_oversample_factor``).
        - If auto_oversample is False (or ``max_oversample_factor``
          is exceeded), raises ContractValidationError(10).

        Returns:
            Final merged list of preference pairs.

        Raises:
            ContractValidationError: If golden pairs are empty
                or ratio cannot be met (exit code 10).
        """
        golden_unique = self._deduplicate(self._golden_pairs)
        if not golden_unique:
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

        all_non_golden = self._deduplicate(
            self._self_pairs + self._human_pairs
        )
        n_non_golden = len(all_non_golden)
        n_golden = len(golden_unique)

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
            return all_non_golden + golden_unique

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

        needed = math.ceil(
            self._min_ratio
            * n_non_golden
            / max(1.0 - self._min_ratio, 0.01)
        )
        needed = max(needed, 1)

        required_factor = math.ceil(needed / n_golden)
        if required_factor > self._max_oversample_factor:
            raise ContractValidationError(
                f"Required golden oversample factor {required_factor}x "
                f"(need {needed} from {n_golden} unique golden pairs) "
                f"exceeds max_oversample_factor={self._max_oversample_factor}x.",
                agent_fix_hint=(
                    "Add more distinct L0 golden expert pairs or reduce "
                    "the batch of unverified self-play pairs."
                ),
            )

        oversampled: list[PreferencePair] = []
        idx = 0
        while len(oversampled) < needed:
            src = golden_unique[idx % len(golden_unique)]
            oversampled.append(
                src.model_copy(
                    update={
                        "pair_id": f"{src.pair_id}_os{idx:04d}",
                    },
                )
            )
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
                    json.dumps(p.model_dump(), default=str) + "\n"
                )

        return {
            "total": total,
            "golden_count": golden_count,
            "golden_ratio": round(ratio, 4),
            "output_path": str(output_path),
        }
