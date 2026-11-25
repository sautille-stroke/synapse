"""Convert Synapse-Bench reports to Traceboard `.tbevt` event streams.

Provides a dual-track migration path: the existing HTML report generator
(``synapse.bench.report``) remains unchanged, while this module adds a
parallel Traceboard rendering path that writes per-scenario scalar metrics,
pillar scores, and timing data as structured ``.tbevt`` events.

Usage::

    from synapse.bench.traceboard_export import export_reports_to_traceboard
    from synapse.pipeline.bench_runner import ScenarioRunReport

    reports: list[ScenarioRunReport] = [...]
    export_reports_to_traceboard(reports, log_dir="./tblog", run_name="bench_v1")
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_reports_to_traceboard(
    reports: list[Any],
    *,
    log_dir: str | Path,
    run_name: str = "synapse-bench",
    engine: str = "synapse-bench",
    close_on_finish: bool = True,
) -> bool:
    """Export benchmark reports to Traceboard `.tbevt` files.

    Creates a standalone Traceboard run with one stage per scenario,
    logging all metrics, timing data, and pillar scores as scalars.
    This function is **additive** — it does not modify or replace the
    existing HTML report pipeline.

    Parameters
    ----------
    reports:
        List of ``ScenarioRunReport`` instances from a bench run.
    log_dir:
        Directory where ``.tbevt`` files will be written.
    run_name:
        Name for the Traceboard run.
    engine:
        Engine identifier recorded in the run metadata.
    close_on_finish:
        Whether to end the run and flush after export.

    Returns
    -------
    bool
        ``True`` if export succeeded, ``False`` if ``aether-traceboard``
        is not installed or an error occurred.
    """
    try:
        from aether_traceboard import FileCollector, Probe  # type: ignore[import-untyped]
    except ImportError:
        logger.info(
            "aether-traceboard not installed; skipping Traceboard export. "
            "Install with: pip install aether-traceboard"
        )
        return False

    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        probe = Probe()
        collector = FileCollector(log_path)
        probe.add_collector(collector)
        probe.begin_run(
            run_name,
            engine=engine,
            num_scenarios=str(len(reports)),
        )
        probe.activate()

        # Per-scenario metrics and timings
        n_pass = 0
        n_fail = 0
        for report in reports:
            passed = (
                report.error is None
                and (report.gate_verdict is None or report.gate_verdict.passed)
            )
            if passed:
                n_pass += 1
            else:
                n_fail += 1

            prefix = f"scenario/{report.scenario_name}"

            # Status
            probe.log_scalar(f"{prefix}/passed", 1.0 if passed else 0.0)
            probe.log_scalar(f"{prefix}/num_frames", float(report.num_frames))

            # Metrics
            if report.metrics:
                for key, value in sorted(report.metrics.items()):
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        probe.log_scalar(f"{prefix}/metrics/{key}", float(value))

            # Timings
            if report.timings:
                for key, value in sorted(report.timings.items()):
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        probe.log_scalar(f"{prefix}/timing/{key}", float(value))

        # Aggregate summary
        probe.log_scalar("summary/total_scenarios", float(len(reports)))
        probe.log_scalar("summary/passed", float(n_pass))
        probe.log_scalar("summary/failed", float(n_fail))
        probe.log_scalar(
            "summary/pass_rate",
            float(n_pass) / max(len(reports), 1),
        )

        # Pillar scores
        pillar_scores = _compute_pillar_scores_from_reports(reports)
        for pillar, score in pillar_scores.items():
            probe.log_scalar(f"pillar/{pillar}", score)

        if close_on_finish:
            probe.end_run()

        logger.info(
            "Traceboard export complete: %d scenarios → %s",
            len(reports),
            log_path,
        )
        return True

    except Exception:
        logger.warning("Traceboard export failed", exc_info=True)
        return False


def _compute_pillar_scores_from_reports(
    reports: list[Any],
) -> dict[str, float]:
    """Derive 4-pillar aggregate scores from scenario reports.

    Mirrors the logic in ``synapse.bench.report._compute_pillar_scores``
    to ensure parity between the HTML and Traceboard outputs.

    Pillars:
    - temporal: Based on flicker_score and temporal_warp_error
    - fgpp_fidelity: Based on depth_rmse_m, normal_angular_error_deg,
      silhouette_iou
    - roundtrip: Based on chamfer_mm, psnr_db, ssim
    - style: Based on style_color_de, style_light_error_deg
    """
    accumulators: dict[str, list[float]] = {
        "temporal": [],
        "fgpp_fidelity": [],
        "roundtrip": [],
        "style": [],
    }

    for r in reports:
        m = r.metrics
        if not m:
            continue

        if "flicker_score" in m:
            accumulators["temporal"].append(
                max(0.0, 1.0 - min(m["flicker_score"], 1.0))
            )
        if "temporal_warp_error" in m:
            accumulators["temporal"].append(
                max(0.0, 1.0 - min(m["temporal_warp_error"], 1.0))
            )

        if "silhouette_iou" in m:
            accumulators["fgpp_fidelity"].append(min(m["silhouette_iou"], 1.0))
        if "depth_rmse_m" in m:
            accumulators["fgpp_fidelity"].append(
                max(0.0, 1.0 - min(m["depth_rmse_m"], 1.0))
            )
        if "normal_angular_error_deg" in m and not math.isnan(
            m["normal_angular_error_deg"]
        ):
            accumulators["fgpp_fidelity"].append(
                max(0.0, 1.0 - min(m["normal_angular_error_deg"] / 90.0, 1.0))
            )

        if "psnr_db" in m:
            accumulators["roundtrip"].append(
                min(1.0, max(0.0, m["psnr_db"] / 40.0))
            )
        if "ssim" in m:
            accumulators["roundtrip"].append(
                min(1.0, max(0.0, m["ssim"]))
            )
        if "chamfer_mm" in m and not math.isnan(m["chamfer_mm"]):
            accumulators["roundtrip"].append(
                max(0.0, 1.0 - min(m["chamfer_mm"] / 50.0, 1.0))
            )

        if "style_color_de" in m:
            accumulators["style"].append(
                max(0.0, 1.0 - min(m["style_color_de"] / 20.0, 1.0))
            )
        if "style_light_error_deg" in m:
            accumulators["style"].append(
                max(0.0, 1.0 - min(m["style_light_error_deg"] / 45.0, 1.0))
            )

    scores: dict[str, float] = {}
    for pillar, vals in accumulators.items():
        scores[pillar] = sum(vals) / len(vals) if vals else 0.0

    return scores
