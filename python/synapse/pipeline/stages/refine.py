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
        _REGION_MAP,
        LocalRefinementPipeline,
    )
    from synapse.refine.types import FusionMethod, RefinementROI

    params = dict(manifest.params)

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
        fusion_name, FusionMethod.LAPLACIAN_FREQUENCY,
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

    rois_raw = json.loads(Path(rois_path).read_text(encoding="utf-8"))
    height, width = frames.shape[1], frames.shape[2]

    rois_per_frame: list[list[RefinementROI]] = []
    for t, frame_rois_raw in enumerate(rois_raw):
        frame_rois: list[RefinementROI] = []
        for roi_dict in frame_rois_raw:
            region = _REGION_MAP.get(roi_dict["region"])
            if region is None:
                continue
            bbox = np.array(roi_dict["bbox_xyxy"], dtype=np.float32)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frames[t, y1:y2, x1:x2, :].copy()
            roi = RefinementROI(
                region=region,
                bbox_xyxy=bbox,
                crop_acescg=crop,
                confidence=float(roi_dict.get("confidence", 0.9)),
            )
            frame_rois.append(roi)
        rois_per_frame.append(frame_rois)

    pipeline = LocalRefinementPipeline()
    result = pipeline.run(
        frames_acescg=frames,
        rois_per_frame=rois_per_frame,
        fusion_method=fusion_method,
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

    return {
        "output_paths": output_paths,
        "diagnostics": result.to_dict(),
    }
