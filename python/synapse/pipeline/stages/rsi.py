"""RSI pipeline stage handlers (rsi_dual_verify)."""

from __future__ import annotations

import json
from pathlib import Path
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

    output_paths: dict[str, str] = {}
    verdict_target = manifest.outputs.get("verdict")
    if verdict_target:
        p_ver = Path(verdict_target)
        p_ver.parent.mkdir(parents=True, exist_ok=True)
        p_ver.write_text(json.dumps(verdict.model_dump(), indent=2), encoding="utf-8")
        output_paths["verdict"] = str(p_ver)

    return {
        "output_paths": output_paths,
        "verdict": verdict.model_dump(),
    }
