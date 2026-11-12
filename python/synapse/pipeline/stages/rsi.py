"""RSI pipeline stage handlers (rsi_dual_verify)."""

from __future__ import annotations

from typing import Any

from synapse.pipeline.bundle_worker import (
    BundleManifest,
    register_stage,
)
from synapse.rsi.dual_verifier import DualVerifierEngine


@register_stage("rsi_dual_verify")
def handle_rsi_dual_verify(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run dual verification from a job bundle manifest."""
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
