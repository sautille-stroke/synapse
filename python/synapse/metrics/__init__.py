"""synapse.metrics — Cross-modal temporal, fidelity, image quality, and style metrics.

4-pillar, 16-metric core evaluation suite plus standard image quality metrics
(PSNR, SSIM, CIEDE2000, LPIPS) and 3-tier quality gate profiles.

Pillars:
    temporal          Temporal consistency (warp error, flicker, FVD)
    fgpp_fidelity     FGPP channel fidelity (depth, normals, flow, IoU)
    image_quality     Frame/multi-view perceptual quality (PSNR, SSIM, CIEDE2000, LPIPS)
    roundtrip         Roundtrip geometry & appearance (Chamfer, F-score,
                      volume, skeleton, albedo)
    style_consistency Style alignment (border ring, light, tone, histogram)
    profiles          3-tier gate profiles and evaluation logic
"""

from __future__ import annotations

from synapse.metrics.fgpp_fidelity import (
    depth_rmse_m,
    flow_endpoint_error_px,
    normal_angular_error_deg,
    scale_invariant_depth_error,
    silhouette_iou,
)
from synapse.metrics.image_quality import (
    ciede2000,
    lpips,
    psnr_db,
    ssim,
)
from synapse.metrics.profiles import (
    METRIC_ALIASES,
    RSI_PROMOTION,
    RSI_PROMOTION_TARGET,
    TRACK_A_ASSET,
    TRACK_B_VIDEO,
    GateProfileName,
    GateVerdict,
    GateViolation,
    evaluate_gate,
    evaluate_metrics_gate,
    get_profile_thresholds,
)
from synapse.metrics.roundtrip import (
    albedo_delta_e00,
    bone_length_drift_pct,
    chamfer_mm,
    f_score_1mm,
    foot_sliding_cm,
    volume_breathing_pct,
)
from synapse.metrics.style_consistency import (
    border_ring_ciede2000,
    histogram_intersection,
    light_dir_error_deg,
    tone_curve_rmse,
)
from synapse.metrics.temporal import (
    flicker_score,
    fvd_feature_distance,
    temporal_warp_error,
)

__all__ = [
    # profiles
    "GateProfileName",
    "GateVerdict",
    "GateViolation",
    "METRIC_ALIASES",
    "TRACK_B_VIDEO",
    "TRACK_A_ASSET",
    "RSI_PROMOTION",
    "RSI_PROMOTION_TARGET",
    "evaluate_gate",
    "evaluate_metrics_gate",
    "get_profile_thresholds",
    # temporal
    "temporal_warp_error",
    "flicker_score",
    "fvd_feature_distance",
    # fgpp_fidelity
    "depth_rmse_m",
    "scale_invariant_depth_error",
    "normal_angular_error_deg",
    "flow_endpoint_error_px",
    "silhouette_iou",
    # image_quality
    "psnr_db",
    "ssim",
    "ciede2000",
    "lpips",
    # roundtrip
    "chamfer_mm",
    "f_score_1mm",
    "volume_breathing_pct",
    "foot_sliding_cm",
    "bone_length_drift_pct",
    "albedo_delta_e00",
    # style_consistency
    "border_ring_ciede2000",
    "histogram_intersection",
    "light_dir_error_deg",
    "tone_curve_rmse",
]
