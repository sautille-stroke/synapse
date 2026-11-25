"""Four-track industry benchmark comparator for Synapse-Bench (PR-215 / RFC 0063 §5).

Aggregates quantitative metrics across the 4 evaluation tracks:
- Track 1 (``track1_3d_assets``): Generative 3D geometry & topology quality
- Track 2 (``track2_video_continuity``): Spatio-temporal video & control stability
- Track 3 (``track3_kinematics_mocap``): Articulated kinematics & volumetric fidelity
- Track 4 (``track4_inverse_rendering``): Physical material / lighting recovery
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.observability import get_probe, traceboard_stage

Direction = Literal["lower_is_better", "higher_is_better"]

TRACK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "track1_3d_assets": {
        "title": "Track 1: Generative 3D Assets & Topology",
        "metrics": {
            "chamfer_mm": ("lower_is_better", 20.0),
            "f_score_1mm": ("higher_is_better", 1.0),
            "uv_stretch": ("lower_is_better", 2.0),
            "is_watertight": ("higher_is_better", 1.0),
        },
    },
    "track2_video_continuity": {
        "title": "Track 2: Video Spatio-Temporal Continuity",
        "metrics": {
            "fvd": ("lower_is_better", 200.0),
            "temporal_warp_error": ("lower_is_better", 0.05),
            "foot_sliding_cm": ("lower_is_better", 1.0),
        },
    },
    "track3_kinematics_mocap": {
        "title": "Track 3: Kinematics & Volumetric Rigging",
        "metrics": {
            "bone_length_drift_pct": ("lower_is_better", 5.0),
            "volume_breathing_pct": ("lower_is_better", 10.0),
            "silhouette_iou": ("higher_is_better", 1.0),
        },
    },
    "track4_inverse_rendering": {
        "title": "Track 4: Physical Relighting & Inverse Rendering",
        "metrics": {
            "depth_rmse_m": ("lower_is_better", 0.1),
            "border_ring_ciede2000": ("lower_is_better", 5.0),
            "albedo_delta_e00": ("lower_is_better", 5.0),
            "light_dir_error_deg": ("lower_is_better", 45.0),
        },
    },
}


@dataclass
class TrackScore:
    """Per-track normalized score and metric summary for one adapter/model."""

    track_id: str
    score: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelComparisonRow:
    """Aggregated 4-track leaderboard row for one evaluated model/adapter."""

    model_name: str
    overall_score: float
    track_scores: dict[str, TrackScore] = field(default_factory=dict)
    raw_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "overall_score": self.overall_score,
            "track_scores": {
                k: {"score": v.score, "metrics": v.metrics}
                for k, v in self.track_scores.items()
            },
            "raw_metrics": self.raw_metrics,
        }


@dataclass
class FourTrackComparisonReport:
    """Multi-model comparison leaderboard across RFC 0063 4 benchmark tracks."""

    rows: list[ModelComparisonRow]
    track_winners: dict[str, str]
    overall_winner: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_winner": self.overall_winner,
            "track_winners": self.track_winners,
            "rows": [r.to_dict() for r in self.rows],
            "tracks": {
                tid: {
                    "title": info["title"],
                    "metrics": list(info["metrics"].keys()),
                }
                for tid, info in TRACK_DEFINITIONS.items()
            },
        }


def _extract_flat_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten metrics from either a flat dict or a ``BenchPipelineRunner`` report JSON."""
    accum: dict[str, list[float]] = {}

    def _collect(src: dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                accum.setdefault(k, []).append(float(v))
            elif isinstance(v, bool):
                accum.setdefault(k, []).append(1.0 if v else 0.0)

    if isinstance(payload.get("metrics"), dict):
        _collect(payload["metrics"])
    if isinstance(payload.get("reports"), list):
        for rep in payload["reports"]:
            if isinstance(rep, dict) and isinstance(rep.get("metrics"), dict):
                _collect(rep["metrics"])
    if not accum:
        _collect(payload)

    return {k: sum(vals) / len(vals) for k, vals in accum.items() if vals}


def _normalize_metric(value: float, direction: Direction, scale: float) -> float:
    """Map raw metric onto ``[0.0, 1.0]`` (higher always better)."""
    if direction == "higher_is_better":
        return float(max(0.0, min(1.0, value / max(scale, 1e-8))))
    return float(max(0.0, min(1.0, 1.0 - (value / max(scale, 1e-8)))))


def aggregate_four_track_scores(
    run_payloads: dict[str, dict[str, Any]],
) -> FourTrackComparisonReport:
    """Compare >= 2 benchmark runs across all 4 RFC 0063 industry tracks."""
    if len(run_payloads) < 2:
        raise ContractValidationError(
            f"4-track comparison requires at least 2 model runs, got {len(run_payloads)}.",
            agent_fix_hint="Provide >= 2 benchmark run files or dicts for comparison.",
        )

    rows: list[ModelComparisonRow] = []
    for model_name, payload in run_payloads.items():
        flat = _extract_flat_metrics(payload)
        track_scores: dict[str, TrackScore] = {}
        active_track_vals: list[float] = []

        for tid, info in TRACK_DEFINITIONS.items():
            t_metrics: dict[str, float] = {}
            norm_vals: list[float] = []
            for m_name, (direction, scale) in info["metrics"].items():
                if m_name in flat:
                    val = flat[m_name]
                    t_metrics[m_name] = round(val, 6)
                    norm_vals.append(_normalize_metric(val, direction, scale))
            score = round(sum(norm_vals) / len(norm_vals), 6) if norm_vals else 0.0
            track_scores[tid] = TrackScore(track_id=tid, score=score, metrics=t_metrics)
            if norm_vals:
                active_track_vals.append(score)

        overall = (
            round(sum(active_track_vals) / len(active_track_vals), 6)
            if active_track_vals
            else 0.0
        )
        rows.append(
            ModelComparisonRow(
                model_name=model_name,
                overall_score=overall,
                track_scores=track_scores,
                raw_metrics=flat,
            )
        )

    rows.sort(key=lambda r: r.overall_score, reverse=True)
    overall_winner = rows[0].model_name

    track_winners: dict[str, str] = {}
    for tid in TRACK_DEFINITIONS:
        best_row = max(
            rows,
            key=lambda r: r.track_scores[tid].score if tid in r.track_scores else -1.0,
        )
        track_winners[tid] = best_row.model_name

    return FourTrackComparisonReport(
        rows=rows,
        track_winners=track_winners,
        overall_winner=overall_winner,
    )


def compare_runs(
    run_paths: list[Path],
    labels: list[str] | None = None,
) -> FourTrackComparisonReport:
    """Load >= 2 JSON benchmark reports from disk and build a 4-track leaderboard."""
    if len(run_paths) < 2:
        raise ContractValidationError(
            f"compare_runs requires at least 2 run JSON paths, got {len(run_paths)}.",
            agent_fix_hint="Pass --inputs <run_a.json> <run_b.json> [...].",
        )

    with traceboard_stage(
        "bench/compare",
        num_runs=str(len(run_paths)),
    ):
        payloads: dict[str, dict[str, Any]] = {}
        for idx, p in enumerate(run_paths):
            path = Path(p)
            if not path.is_file():
                raise AssetNotFoundError(
                    f"Benchmark run file not found: {path}",
                    agent_fix_hint="Verify all paths passed to synapse bench compare --inputs.",
                )
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ContractValidationError(
                    f"Invalid JSON in benchmark run file {path}: {exc}",
                    agent_fix_hint="Ensure input files are valid Synapse-Bench JSON reports.",
                ) from exc
            if not isinstance(data, dict):
                raise ContractValidationError(
                    f"Expected JSON object in {path}, got {type(data).__name__}.",
                    agent_fix_hint="Provide valid benchmark report dicts.",
                )
            label = (
                labels[idx]
                if labels is not None and idx < len(labels) and labels[idx]
                else str(data.get("adapter_id") or path.stem)
            )
            if label in payloads:
                label = f"{label}_{idx}"
            payloads[label] = data

        report = aggregate_four_track_scores(payloads)

        # Log leaderboard table to Traceboard
        probe = get_probe()
        if probe is not None and probe.active:
            table_rows: list[dict[str, Any]] = []
            for row in report.rows:
                table_row: dict[str, Any] = {
                    "model": row.model_name,
                    "overall_score": row.overall_score,
                }
                for tid, ts in row.track_scores.items():
                    table_row[tid] = ts.score
                table_rows.append(table_row)
            probe.log_table("bench/compare/leaderboard", table_rows)

    return report
