"""Convert Synapse-Bench reports to Traceboard ``.tbevt`` event streams.

Provides a dual-track migration path alongside the HTML report generator
(``synapse.bench.report``) using identical pillar scoring (`compute_pillar_scores`)
and unified ``bench/scenario/{name}/*``, ``bench/summary/*``, ``bench/pillar/*``
tag namespaces.
"""

from __future__ import annotations

import contextlib
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from synapse.bench.report import compute_pillar_scores

if TYPE_CHECKING:
    from synapse.observability import ProbeProtocol

logger = logging.getLogger(__name__)

# Backward-compatible alias (compute_pillar_scores in synapse.bench.report is SSOT)
_compute_pillar_scores_from_reports = compute_pillar_scores


def export_reports_to_traceboard(
    reports: list[Any],
    *,
    log_dir: str | Path,
    run_name: str = "synapse-bench",
    engine: str = "synapse-bench",
    probe: ProbeProtocol | None = None,
) -> bool:
    """Export benchmark reports to Traceboard ``.tbevt`` events.

    Creates a standalone Traceboard run with one stage per scenario when *probe*
    is *None*, or writes into an existing caller-managed *probe* without
    closing it.

    Returns
    -------
    bool
        ``True`` if export succeeded, ``False`` if ``aether-traceboard``
        is unavailable or an error occurred.
    """
    owns_probe = probe is None
    active_probe: Any = probe

    if owns_probe:
        try:
            from aether_traceboard import FileCollector, Probe
        except ImportError:
            logger.info(
                "aether-traceboard not installed; skipping Traceboard export. "
                "Install with: pip install "
                '"aether-traceboard @ git+https://github.com/dervgen/aether.git'
                '@v0.7.2#subdirectory=traceboard/python"'
            )
            return False
    else:
        if active_probe is None or not active_probe.active:
            return False

    collector: Any = None
    run_opened = False
    had_error = False

    try:
        if owns_probe:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            active_probe = Probe()
            collector = FileCollector(log_path)
            active_probe.add_collector(collector)
            active_probe.activate()
            active_probe.begin_run(
                run_name,
                engine=engine,
                num_scenarios=str(len(reports)),
            )
            run_opened = True

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

            prefix = f"bench/scenario/{report.scenario_name}"
            with active_probe.begin_stage(
                f"bench/{report.scenario_name}",
                adapter=str(getattr(report, "adapter_id", "unknown")),
            ) as stage:
                active_probe.log_scalar(f"{prefix}/passed", 1.0 if passed else 0.0)
                active_probe.log_scalar(f"{prefix}/num_frames", float(report.num_frames))

                if report.metrics:
                    for key, value in sorted(report.metrics.items()):
                        if (
                            not isinstance(value, bool)
                            and isinstance(value, (int, float))
                            and math.isfinite(float(value))
                        ):
                            active_probe.log_scalar(f"{prefix}/metrics/{key}", float(value))

                if report.timings:
                    for key, value in sorted(report.timings.items()):
                        if (
                            not isinstance(value, bool)
                            and isinstance(value, (int, float))
                            and math.isfinite(float(value))
                        ):
                            active_probe.log_scalar(f"{prefix}/timing/{key}", float(value))

                if report.error is not None:
                    active_probe.log_text(f"{prefix}/error", str(report.error))
                    stage.set_fail(str(report.error))
                elif report.gate_verdict is not None and not report.gate_verdict.passed:
                    stage.set_fail("gate_failed")
                else:
                    stage.set_pass()

        active_probe.log_scalar("bench/summary/total_scenarios", float(len(reports)))
        active_probe.log_scalar("bench/summary/passed", float(n_pass))
        active_probe.log_scalar("bench/summary/failed", float(n_fail))
        active_probe.log_scalar(
            "bench/summary/pass_rate",
            float(n_pass) / max(len(reports), 1),
        )

        pillar_scores = compute_pillar_scores(reports)
        for pillar, score in sorted(pillar_scores.items()):
            active_probe.log_scalar(f"bench/pillar/{pillar}", float(score))

        logger.info(
            "Traceboard export complete: %d scenarios",
            len(reports),
        )
        return True

    except Exception:
        had_error = True
        logger.warning("Traceboard export failed", exc_info=True)
        return False
    finally:
        if owns_probe:
            if run_opened and active_probe is not None:
                with contextlib.suppress(Exception):
                    active_probe.end_run("failed" if had_error else "finished")
            elif collector is not None:
                with contextlib.suppress(Exception):
                    collector.on_run_close()
