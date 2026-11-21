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

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.errors import CircuitBreakerError, QualityGateError
from synapse.refine.face import FaceRefiner
from synapse.refine.fusion import MultiModelRefineFusion
from synapse.refine.gates import (
    GateThresholds,
    GateVerdict,
    evaluate_refinement_gates,
    export_hard_negatives,
)
from synapse.refine.hand import HandRefiner
from synapse.refine.roi_tracker import KalmanBBoxTracker
from synapse.refine.stitcher import ScalpelStitcher, StitchConfig, StitchReport
from synapse.refine.tiled_executor import TiledBypassExecutor
from synapse.refine.types import (
    AnatomicalRegion,
    FusionMethod,
    RefinementResult,
    RefinementROI,
)

logger = logging.getLogger(__name__)

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
    bypassed: bool = False
    bypass_reasons: list[str] = field(default_factory=list)

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
            "bypassed": self.bypassed,
            "bypass_reasons": self.bypass_reasons,
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


def _pad_to_canonical(patch: np.ndarray, target_size: int) -> np.ndarray:
    """Pad/crop a patch to a canonical square size for consistent LPIPS.

    Kalman-smoothed bounding boxes can differ by 1–2 px between frames,
    causing shape mismatches in LPIPS computation.  This helper pads
    (or centre-crops) to a canonical ``(target_size, target_size, C)``
    to ensure consistent comparison.
    """
    h, w = patch.shape[:2]
    c = patch.shape[2] if patch.ndim == 3 else 1
    out = np.zeros((target_size, target_size, c), dtype=patch.dtype)

    # Centre-crop or centre-pad
    src_y0 = max(0, (h - target_size) // 2)
    src_x0 = max(0, (w - target_size) // 2)
    dst_y0 = max(0, (target_size - h) // 2)
    dst_x0 = max(0, (target_size - w) // 2)

    copy_h = min(h, target_size)
    copy_w = min(w, target_size)
    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = (
        patch[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
    )
    return out


class LocalRefinementPipeline:
    """End-to-end local anatomical refinement pipeline.

    Args:
        stitch_config: Configuration for the scalpel stitcher.
        gate_thresholds: Quality gate thresholds.
        strict: When ``True``, raise :class:`QualityGateError` on
            gate failure instead of falling back to the original frame.
    """

    def __init__(
        self,
        stitch_config: StitchConfig | None = None,
        gate_thresholds: GateThresholds | None = None,
        strict: bool = False,
    ) -> None:
        self.stitch_config = stitch_config or StitchConfig()
        self.gate_thresholds = gate_thresholds or GateThresholds()
        self.strict = strict
        self.stitcher = ScalpelStitcher(self.stitch_config)
        self.fusion = MultiModelRefineFusion()

        # --- Wire up anatomical refiners (PR-45 fix #1) ---
        self.hand_refiner = HandRefiner(strict=self.strict)
        self.face_refiner = FaceRefiner(strict=self.strict)
        # PR-81: Independent Kalman tracker per track_id instead of
        # a single shared tracker that cross-contaminates state.
        self._bbox_trackers: dict[str, KalmanBBoxTracker] = {}
        self.tiled_executor = TiledBypassExecutor()
        # Canonical LPIPS patch size for consistent comparison
        self._lpips_patch_size: int = 64

    def run(
        self,
        frames_acescg: np.ndarray,
        rois_per_frame: list[list[RefinementROI]],
        candidates_per_roi: (list[list[list[np.ndarray]]] | None) = None,
        fusion_method: FusionMethod = FusionMethod.LAPLACIAN_FREQUENCY,
        hard_negatives_path: str | None = None,
        elapsed_seconds_override: float | None = None,
        motion_vectors: np.ndarray | None = None,
        model_ensemble: Any | None = None,
    ) -> RefinePipelineResult:
        """Run the full refinement pipeline.

        Args:
            frames_acescg: ``(T, H, W, 3)`` float32 ACEScg source video.
            rois_per_frame: Per-frame list of detected ROIs.
            candidates_per_roi: Per-ROI list of candidate refinement
                patches.  If ``None``, dispatches to region-specific
                refiners (``HandRefiner`` / ``FaceRefiner``) instead of
                using the ROI crop as-is.
            fusion_method: Fusion strategy to use.
            hard_negatives_path: Optional path for hard negatives JSONL.
            elapsed_seconds_override: Optional override for elapsed time
                (used for testing circuit breaker propagation).
            motion_vectors: Optional motion vectors ``(T, H, W, 2)`` float32.
            model_ensemble: Optional multi-model candidate generator.

        Returns:
            :class:`RefinePipelineResult` with refined frames and
            diagnostics.

        Raises:
            CircuitBreakerError: If elapsed time exceeds the 900 s HITL
                circuit breaker budget (exit_code=40).
            QualityGateError: If ``strict=True`` and quality gates fail.
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
            used_track_ids: set[str] = set()

            # Predict existing trackers for frame t
            for tid, trk in self._bbox_trackers.items():
                trk.predict()

            for roi_idx, roi in enumerate(rois):
                total_rois += 1
                smoothed_bbox: np.ndarray | None = None
                is_bypassed = False

                # --- PR-45 fix #2: Region-based dispatch when no
                # explicit candidates are provided ---
                if (
                    candidates_per_roi is not None
                    and t < len(candidates_per_roi)
                    and roi_idx < len(candidates_per_roi[t])
                ):
                    cands = candidates_per_roi[t][roi_idx]
                else:
                    # IoU-based multi-target tracking across frames
                    best_track_id: str | None = None
                    best_iou = 0.3
                    from synapse.refine.roi_tracker import _cxcywh_to_xyxy, compute_bbox_iou

                    for tid, trk in self._bbox_trackers.items():
                        if tid.startswith(roi.region.name) and tid not in used_track_ids:
                            pred_bbox = _cxcywh_to_xyxy(trk._x[:4])
                            iou = compute_bbox_iou(roi.bbox_xyxy, pred_bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_track_id = tid

                    if best_track_id is not None:
                        track_id = best_track_id
                        tracker = self._bbox_trackers[track_id]
                    else:
                        candidate_id = f"{roi.region.name}_{roi_idx}"
                        if candidate_id not in self._bbox_trackers:
                            track_id = candidate_id
                        else:
                            track_id = f"{candidate_id}_{len(self._bbox_trackers)}"
                        tracker = KalmanBBoxTracker()
                        self._bbox_trackers[track_id] = tracker

                    used_track_ids.add(track_id)
                    smoothed_bbox = tracker.update(roi.bbox_xyxy)

                    # Re-crop frames_acescg[t] based on smoothed_bbox
                    _h, _w = frames_acescg.shape[1], frames_acescg.shape[2]
                    sx1 = max(0, min(_w - 1, int(round(float(smoothed_bbox[0])))))
                    sy1 = max(0, min(_h - 1, int(round(float(smoothed_bbox[1])))))
                    sx2 = max(sx1 + 1, min(_w, int(round(float(smoothed_bbox[2])))))
                    sy2 = max(sy1 + 1, min(_h, int(round(float(smoothed_bbox[3])))))
                    smoothed_crop = frames_acescg[t, sy1:sy2, sx1:sx2, :].copy()

                    roi_smoothed = RefinementROI(
                        bbox_xyxy=smoothed_bbox,
                        crop_acescg=smoothed_crop,
                        region=roi.region,
                        confidence=roi.confidence,
                        anatomical_quality_score=roi.anatomical_quality_score,
                        frame_index=roi.frame_index,
                    )

                    # Use TiledBypassExecutor to skip healthy tiles
                    defective_rois, _bypass_count = self.tiled_executor.extract_and_filter(
                        frames_acescg,
                        [roi_smoothed],
                    )
                    if not defective_rois:
                        # All tiles healthy — bypass refinement
                        is_bypassed = True
                        cands = [smoothed_crop]
                    else:
                        crop_h, crop_w = roi_smoothed.crop_acescg.shape[:2]
                        ts = self.tiled_executor.tile_size
                        if roi.region == AnatomicalRegion.HAND:
                            if crop_h > ts or crop_w > ts:

                                def _tile_hand(
                                    tile_img: np.ndarray,
                                    curr_roi: RefinementROI = roi_smoothed,
                                ) -> np.ndarray:
                                    t_roi = RefinementROI(
                                        bbox_xyxy=curr_roi.bbox_xyxy,
                                        crop_acescg=tile_img,
                                        region=curr_roi.region,
                                        confidence=curr_roi.confidence,
                                        frame_index=curr_roi.frame_index,
                                    )
                                    return self.hand_refiner.refine(t_roi).refined_acescg

                                refined_tile = self.tiled_executor.tile_and_execute(
                                    roi_smoothed.crop_acescg, _tile_hand
                                )
                                cands = [refined_tile]
                            else:
                                ref_result = self.hand_refiner.refine(roi_smoothed)
                                cands = [ref_result.refined_acescg]
                        elif roi.region == AnatomicalRegion.FACE:
                            if crop_h > ts or crop_w > ts:

                                def _tile_face(
                                    tile_img: np.ndarray,
                                    curr_roi: RefinementROI = roi_smoothed,
                                ) -> np.ndarray:
                                    t_roi = RefinementROI(
                                        bbox_xyxy=curr_roi.bbox_xyxy,
                                        crop_acescg=tile_img,
                                        region=curr_roi.region,
                                        confidence=curr_roi.confidence,
                                        frame_index=curr_roi.frame_index,
                                    )
                                    return self.face_refiner.refine(t_roi).refined_acescg

                                refined_tile = self.tiled_executor.tile_and_execute(
                                    roi_smoothed.crop_acescg, _tile_face
                                )
                                cands = [refined_tile]
                            else:
                                ref_result = self.face_refiner.refine(roi_smoothed)
                                cands = [ref_result.refined_acescg]
                        else:
                            cands = [smoothed_crop]

                        # Multi-candidate ensemble support
                        active_ensemble = model_ensemble or getattr(self, "model_ensemble", None)
                        if active_ensemble is not None:
                            ens_cands = active_ensemble.generate_candidates(roi_smoothed)
                            if len(ens_cands) > 1:
                                cands = ens_cands

                # Fuse candidates
                fused = self.fusion.fuse(
                    cands,
                    method=fusion_method,
                )
                total_fused += 1

                ref = RefinementResult(
                    refined_acescg=fused,
                    region=roi.region,
                    fusion_method=fusion_method,
                    quality_score=roi.confidence,
                    frame_index=t,
                    bbox_xyxy=(
                        smoothed_bbox.copy() if smoothed_bbox is not None else roi.bbox_xyxy.copy()
                    ),
                    smoothed_bbox=(smoothed_bbox.copy() if smoothed_bbox is not None else None),
                    bypassed=is_bypassed,
                )
                frame_refs.append(ref)

            refinements.append(frame_refs)

        # Stitch with motion vectors
        result_frames, stitch_report = self.stitcher.stitch_sequence(
            frames_acescg,
            refinements,
            motion_vectors=motion_vectors,
        )

        elapsed = (
            elapsed_seconds_override
            if elapsed_seconds_override is not None
            else time.monotonic() - t_start
        )

        # Extract real original and refined patches for gate evaluation.
        # Pad/resize to consistent canonical size to prevent LPIPS failures
        # from 1px Kalman box-size jitter.
        first_orig_patch: np.ndarray | None = None
        first_ref_patch: np.ndarray | None = None
        for t in range(t_count):
            if refinements[t] and rois_per_frame[t]:
                orig = rois_per_frame[t][0].crop_acescg
                ref_patch = refinements[t][0].refined_acescg
                # Pad both patches to the same canonical size
                first_orig_patch = _pad_to_canonical(orig, self._lpips_patch_size)
                first_ref_patch = _pad_to_canonical(ref_patch, self._lpips_patch_size)
                break

        # Compute real temporal warp error across frames
        temporal_warp_err = 0.0
        if t_count > 1:
            diffs = []
            for t_idx in range(1, t_count):
                if motion_vectors is not None and t_idx < motion_vectors.shape[0]:
                    from synapse.refine.stitcher import _backward_warp_bilinear

                    warped_prev = _backward_warp_bilinear(
                        result_frames[t_idx - 1],
                        motion_vectors[t_idx],
                        fallback=result_frames[t_idx],
                    )
                    diffs.append(
                        float(
                            np.mean(
                                np.abs(
                                    result_frames[t_idx].astype(np.float64)
                                    - warped_prev.astype(np.float64)
                                )
                            )
                        )
                    )
                else:
                    diffs.append(
                        float(
                            np.mean(
                                np.abs(
                                    result_frames[t_idx].astype(np.float64)
                                    - result_frames[t_idx - 1].astype(np.float64)
                                )
                            )
                        )
                    )
            temporal_warp_err = float(np.mean(diffs))

        # --- PR-45 fix #3 & #4: Consume GateVerdict and propagate
        # CircuitBreakerError ---
        try:
            verdict: GateVerdict = evaluate_refinement_gates(
                border_psnr_db=stitch_report.mean_border_psnr_db,
                border_ciede2000=stitch_report.mean_border_ciede2000,
                temporal_warp_error=temporal_warp_err,
                elapsed_seconds=elapsed,
                thresholds=self.gate_thresholds,
                original_patch=first_orig_patch,
                refined_patch=first_ref_patch,
            )
        except CircuitBreakerError:
            raise
        except Exception:
            # PR-165: When strict=True, errors must NOT be swallowed
            if self.strict:
                raise
            verdict = GateVerdict(passed=False, failure_reasons=["gate_evaluation_error"])

        # PR-45 fix #3: Act on the verdict
        if not verdict.passed and stitch_report.num_patches_applied > 0:
            for t in range(t_count):
                for ref in refinements[t]:
                    hard_negatives.append(
                        {
                            "frame_index": t,
                            "region": ref.region.name,
                            "failure_reasons": verdict.failure_reasons,
                            "metrics": {
                                "border_psnr_db": stitch_report.mean_border_psnr_db,
                                "border_ciede2000": stitch_report.mean_border_ciede2000,
                            },
                        }
                    )
            if self.strict:
                raise QualityGateError(
                    f"Refinement quality gates failed: {', '.join(verdict.failure_reasons)}",
                    agent_fix_hint=(
                        "Retry with adjusted thresholds or fall back to original frames."
                    ),
                    details={"verdict": verdict.metrics},
                )
            else:
                # Non-strict: keep refined result but flag as bypassed
                # (AUD-P0-04: do NOT silently copy original as output)
                logger.warning(
                    "Quality gates failed (%s); keeping refined output "
                    "with bypassed=True in non-strict mode.",
                    ", ".join(verdict.failure_reasons),
                )
                gate_bypassed = True
                gate_bypass_reasons = list(verdict.failure_reasons)
        else:
            gate_bypassed = False
            gate_bypass_reasons = []

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
            bypassed=gate_bypassed,
            bypass_reasons=gate_bypass_reasons,
        )
