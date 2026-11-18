# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Pipeline stage adapter for anatomical local refinement (RFC 0061).

Registers ``anatomical_refine`` stage with the
:func:`~synapse.pipeline.bundle_worker.register_stage` decorator
so it is discoverable by the ``JobBundleWorker``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from synapse.pipeline.bundle_worker import BundleManifest, register_stage


@register_stage("anatomical_refine")
def handle_anatomical_refine(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Execute anatomical local refinement stage.

    Expects:
        - ``manifest.inputs["video_frames"]``: Path to ``(T,H,W,3)`` ``.npy``
          float32 ACEScg frames.
        - ``manifest.params["regions"]``: List of region names
          (``"HAND"``, ``"FACE"``, etc.).
        - ``manifest.params["fusion_method"]``: Fusion strategy name.
        - ``manifest.params["max_elapsed_seconds"]``: HITL circuit breaker
          budget (default 900).

    Returns:
        Dict with ``output_paths`` and ``diagnostics``.
    """
    from synapse.refine.pipeline import (
        _METHOD_MAP,
        LocalRefinementPipeline,
    )
    from synapse.refine.types import FusionMethod

    params = dict(manifest.params)

    from synapse.bridge.hitl import PersistentHitlLedger
    from synapse.errors import evaluate_hitl_circuit_breaker

    max_elapsed_seconds = float(params.get("max_elapsed_seconds", 900.0))
    elapsed_seconds = float(params.get("elapsed_seconds", 0.0))
    ledger_path = params.get("hitl_ledger_path")
    ledger = PersistentHitlLedger(ledger_path)
    group_id = str(params.get("intent_group_id") or manifest.job_id or "default")
    accumulated = ledger.get_accumulated_elapsed(group_id)
    total_elapsed = accumulated + elapsed_seconds

    # Unconditionally evaluate circuit breaker
    evaluate_hitl_circuit_breaker(
        total_elapsed,
        domain="video_shot",
        max_seconds=max_elapsed_seconds,
    )

    # Load input frames — no synthetic fallback
    video_path = manifest.inputs.get("video_frames", "")
    if not video_path or not Path(video_path).is_file():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Video frames file not found: {video_path!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['video_frames'] pointing to "
                "a valid .npy file of shape (T, H, W, 3) float32 ACEScg."
            ),
        )
    frames = np.load(video_path).astype(np.float32)

    # Parse regions
    region_names: list[str] = params.get("regions", ["HAND", "FACE"])
    if isinstance(region_names, str):
        region_names = [region_names]

    # Parse fusion method
    fusion_name = str(params.get("fusion_method", "LAPLACIAN_FREQUENCY"))
    fusion_method = _METHOD_MAP.get(
        fusion_name,
        FusionMethod.LAPLACIAN_FREQUENCY,
    )

    # Load ROIs — no synthetic fallback
    rois_path = manifest.inputs.get("rois", "")
    if not rois_path or not Path(rois_path).is_file():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"ROIs file not found: {rois_path!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['rois'] pointing to a valid "
                "JSON file with per-frame ROI annotations."
            ),
        )

    p_rois = Path(rois_path)
    if p_rois.suffix.lower() not in (".json", ".npy"):
        from synapse.errors import ContractValidationError

        raise ContractValidationError(
            f"Invalid ROI metadata file extension: {p_rois.suffix!r}. Expected .json or .npy",
            agent_fix_hint="Provide a valid .json or .npy file for manifest.inputs['rois'].",
        )

    from synapse.refine.roi_tracker import parse_rois_json

    meta_content = json.loads(p_rois.read_text(encoding="utf-8"))
    _height, _width = frames.shape[1], frames.shape[2]
    rois_per_frame = parse_rois_json(
        meta_content,
        num_frames=frames.shape[0],
        height=_height,
        width=_width,
        frames=frames,
    )

    from synapse.refine.gates import GateThresholds

    gate_thresholds = GateThresholds(max_elapsed_seconds=max_elapsed_seconds)
    pipeline = LocalRefinementPipeline(gate_thresholds=gate_thresholds)
    result = pipeline.run(
        frames_acescg=frames,
        rois_per_frame=rois_per_frame,
        fusion_method=fusion_method,
        elapsed_seconds_override=elapsed_seconds if elapsed_seconds > 0 else None,
    )

    output_paths: dict[str, str] = {}

    # Save refined frames
    refined_path = manifest.outputs.get(
        "refined_frames",
        str(Path("output") / "refined_frames.npy"),
    )
    p_refined = Path(refined_path)
    p_refined.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p_refined), result.frames_refined)
    output_paths["refined_frames"] = str(p_refined)

    # Save diagnostics
    diag_path = manifest.outputs.get(
        "diagnostics",
        str(Path("output") / "refine_diagnostics.json"),
    )
    p_diag = Path(diag_path)
    p_diag.parent.mkdir(parents=True, exist_ok=True)
    with open(p_diag, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    output_paths["diagnostics"] = str(p_diag)

    # Record action to PersistentHitlLedger to track cumulative budget
    from synapse.bridge.hitl import CommandIRRecord

    action_record = CommandIRRecord(
        intent_group_id=group_id,
        intent="anatomical_refine_stage",
        elapsed_seconds=elapsed_seconds if elapsed_seconds > 0 else float(result.elapsed_seconds),
    )
    ledger.record_action(action_record, max_seconds=max_elapsed_seconds)

    return {
        "output_paths": output_paths,
        "diagnostics": result.to_dict(),
    }
