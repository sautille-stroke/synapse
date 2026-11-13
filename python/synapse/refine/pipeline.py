# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""LocalRefinementPipeline — end-to-end anatomical refinement orchestrator.

Assembles the full refinement pipeline from RFC 0061:

  1. Detect anatomical ROIs (hand, face, etc.)
  2. For each ROI: tile → multi-model generate → fuse → gate
  3. Stitch all refined patches back into the source video
  4. Export hard negatives for RSI flywheel ingestion

The pipeline accepts a ``refine_session.v1.0`` wire-format payload
from upstream ``forgekit-anatomical-refine`` sessions and emits
structured JSON results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.refine.fusion import MultiModelRefineFusion
from synapse.refine.gates import (
    GateThresholds,
    evaluate_refinement_gates,
    export_hard_negatives,
)
from synapse.refine.stitcher import ScalpelStitcher, StitchConfig, StitchReport
from synapse.refine.types import (
    AnatomicalRegion,
    FusionMethod,
    RefinementResult,
    RefinementROI,
)


# ---------------------------------------------------------------------------
# Session wire format
# ---------------------------------------------------------------------------


@dataclass
class RefineSessionPayload:
    """Wire-format payload for ``synapse.refine_session.v1.0``.

    Attributes:
        schema_version: Must be ``"synapse.refine_session.v1.0"``.
        session_id: Unique session identifier.
        regions: List of anatomical regions to refine.
        fusion_method: Fusion strategy to use.
        max_elapsed_seconds: HITL circuit breaker budget.
        command_ir_records: CommandIR entries from upstream forgekit.
    """

    schema_version: str = "synapse.refine_session.v1.0"
    session_id: str = ""
    regions: list[str] = field(default_factory=lambda: ["HAND", "FACE"])
    fusion_method: str = "LAPLACIAN_FREQUENCY"
    max_elapsed_seconds: float = 900.0
    command_ir_records: list[dict[str, Any]] = field(default_factory=list)


def build_refine_session_payload(
    session_id: str,
    regions: list[str] | None = None,
    fusion_method: str = "LAPLACIAN_FREQUENCY",
    max_elapsed_seconds: float = 900.0,
) -> dict[str, Any]:
    """Build a ``synapse.refine_session.v1.0`` payload dict.

    Args:
        session_id: Unique session identifier.
        regions: List of anatomical region names.
        fusion_method: Fusion strategy name.
        max_elapsed_seconds: HITL circuit breaker budget.

    Returns:
        JSON-serialisable payload dict.
    """
    payload = RefineSessionPayload(
        session_id=session_id,
        regions=regions or ["HAND", "FACE"],
        fusion_method=fusion_method,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    return {
        "schema_version": payload.schema_version,
        "session_id": payload.session_id,
        "regions": payload.regions,
        "fusion_method": payload.fusion_method,
        "max_elapsed_seconds": payload.max_elapsed_seconds,
        "command_ir_records": payload.command_ir_records,
    }


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class RefinePipelineResult:
    """Result of the local refinement pipeline.

    Attributes:
        frames_refined: Refined video frames ``(T, H, W, 3)`` float32 ACEScg.
        stitch_report: Stitching quality diagnostics.
        num_rois_detected: Total ROIs detected across all frames.
        num_patches_fused: Total patches that went through fusion.
        hard_negatives_count: Number of hard negatives exported.
        elapsed_seconds: Total pipeline wall-clock time.
    """

    frames_refined: np.ndarray
    stitch_report: StitchReport
    num_rois_detected: int = 0
    num_patches_fused: int = 0
    hard_negatives_count: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "num_frames": int(self.frames_refined.shape[0]),
            "num_rois_detected": self.num_rois_detected,
            "num_patches_fused": self.num_patches_fused,
            "hard_negatives_count": self.hard_negatives_count,
            "elapsed_seconds": self.elapsed_seconds,
            "mean_border_psnr_db": self.stitch_report.mean_border_psnr_db,
            "mean_border_ciede2000": self.stitch_report.mean_border_ciede2000,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_REGION_MAP: dict[str, AnatomicalRegion] = {
    "HAND": AnatomicalRegion.HAND,
    "FACE": AnatomicalRegion.FACE,
    "FOOT": AnatomicalRegion.FOOT,
    "HAIR": AnatomicalRegion.HAIR,
    "FULL_BODY": AnatomicalRegion.FULL_BODY,
}

_METHOD_MAP: dict[str, FusionMethod] = {
    "LAPLACIAN_FREQUENCY": FusionMethod.LAPLACIAN_FREQUENCY,
    "CASCADE": FusionMethod.CASCADE,
    "LATENT_TIMESTEP_SWITCH": FusionMethod.LATENT_TIMESTEP_SWITCH,
    "UNCERTAINTY_WEIGHTED": FusionMethod.UNCERTAINTY_WEIGHTED,
    "POISSON_GRADIENT": FusionMethod.POISSON_GRADIENT,
}


class LocalRefinementPipeline:
    """End-to-end local anatomical refinement pipeline.

    Args:
        stitch_config: Configuration for the scalpel stitcher.
        gate_thresholds: Quality gate thresholds.
    """

    def __init__(
        self,
        stitch_config: StitchConfig | None = None,
        gate_thresholds: GateThresholds | None = None,
    ) -> None:
        self.stitch_config = stitch_config or StitchConfig()
        self.gate_thresholds = gate_thresholds or GateThresholds()
        self.stitcher = ScalpelStitcher(self.stitch_config)
        self.fusion = MultiModelRefineFusion()

    def run(
        self,
        frames_acescg: np.ndarray,
        rois_per_frame: list[list[RefinementROI]],
        candidates_per_roi: (
            list[list[list[np.ndarray]]] | None
        ) = None,
        fusion_method: FusionMethod = FusionMethod.LAPLACIAN_FREQUENCY,
        hard_negatives_path: str | None = None,
    ) -> RefinePipelineResult:
        """Run the full refinement pipeline.

        Args:
            frames_acescg: ``(T, H, W, 3)`` float32 ACEScg source video.
            rois_per_frame: Per-frame list of detected ROIs.
            candidates_per_roi: Per-ROI list of candidate refinement
                patches.  If ``None``, the ROI crop is used as-is
                (single-candidate passthrough).
            fusion_method: Fusion strategy to use.
            hard_negatives_path: Optional path for hard negatives JSONL.

        Returns:
            :class:`RefinePipelineResult` with refined frames and
            diagnostics.
        """
        t_start = time.monotonic()
        t_count = frames_acescg.shape[0]

        total_rois = 0
        total_fused = 0
        hard_negatives: list[dict[str, Any]] = []

        # Build per-frame refinement results for the stitcher
        refinements: list[list[RefinementResult]] = []

        for t in range(t_count):
            frame_refs: list[RefinementResult] = []
            rois = rois_per_frame[t] if t < len(rois_per_frame) else []

            for roi_idx, roi in enumerate(rois):
                total_rois += 1

                # Get candidates (or use ROI crop as single candidate)
                if (
                    candidates_per_roi is not None
                    and t < len(candidates_per_roi)
                    and roi_idx < len(candidates_per_roi[t])
                ):
                    cands = candidates_per_roi[t][roi_idx]
                else:
                    cands = [roi.crop_acescg]

                # Fuse candidates
                fused = self.fusion.fuse(
                    cands, method=fusion_method,
                )
                total_fused += 1

                ref = RefinementResult(
                    refined_acescg=fused,
                    region=roi.region,
                    fusion_method=fusion_method,
                    quality_score=roi.confidence,
                    frame_index=t,
                    bbox_xyxy=roi.bbox_xyxy.copy(),
                )
                frame_refs.append(ref)

            refinements.append(frame_refs)

        # Stitch
        result_frames, stitch_report = self.stitcher.stitch_sequence(
            frames_acescg, refinements,
        )

        elapsed = time.monotonic() - t_start

        # Gate evaluation (aggregate metrics)
        try:
            evaluate_refinement_gates(
                border_psnr_db=stitch_report.mean_border_psnr_db,
                border_ciede2000=stitch_report.mean_border_ciede2000,
                elapsed_seconds=elapsed,
                thresholds=self.gate_thresholds,
            )
        except Exception:
            # Record hard negatives but don't fail the pipeline
            for t in range(t_count):
                for ref in refinements[t]:
                    hard_negatives.append({
                        "frame_index": t,
                        "region": ref.region.name,
                        "failure_reasons": ["gate_failed"],
                        "metrics": {
                            "border_psnr_db": stitch_report.mean_border_psnr_db,
                            "border_ciede2000": stitch_report.mean_border_ciede2000,
                        },
                    })

        # Export hard negatives
        hn_count = 0
        if hard_negatives and hard_negatives_path:
            export_hard_negatives(hard_negatives, hard_negatives_path)
            hn_count = len(hard_negatives)

        return RefinePipelineResult(
            frames_refined=result_frames,
            stitch_report=stitch_report,
            num_rois_detected=total_rois,
            num_patches_fused=total_fused,
            hard_negatives_count=hn_count,
            elapsed_seconds=elapsed,
        )
