"""synapse.pipeline — 5-stage hybrid cinema pipeline orchestration."""

from __future__ import annotations

from synapse.pipeline.bench_runner import (
    BenchPipelineRunner,
    ScenarioRunReport,
    evaluate_bundle,
)
from synapse.pipeline.bundle_worker import (
    BundleManifest,
    HardwareBudget,
    JobBundleWorker,
    ResultManifest,
    register_stage,
)

__all__ = [
    "BenchPipelineRunner",
    "BundleManifest",
    "HardwareBudget",
    "JobBundleWorker",
    "ResultManifest",
    "ScenarioRunReport",
    "evaluate_bundle",
    "register_stage",
]
