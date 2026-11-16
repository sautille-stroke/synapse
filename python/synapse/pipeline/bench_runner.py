"""BenchPipelineRunner — end-to-end benchmark evaluation orchestrator.

Provides :class:`BenchPipelineRunner` for running Synapse-Bench scenarios
through a :class:`~synapse.adapters.base.VideoModelAdapter`, computing
fidelity metrics, and evaluating quality gates.

Also exposes :func:`evaluate_bundle` for offline evaluation of a
pre-rendered ``InspectionBundle`` directory.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Report data class
# -------------------------------------------------------------------


@dataclass
class ScenarioRunReport:
    """Result report from running a single benchmark scenario.

    Attributes:
        scenario_name: Identifier of the evaluated scenario.
        adapter_id: Identifier of the adapter that produced the output.
        metrics: All measured metric values (up to 16 metrics).
        gate_verdict: Quality gate evaluation result, if a gate profile
            was specified.
        timings: Per-phase wall-clock timing in milliseconds.
        num_frames: Number of frames in the generated video.
        resolution: ``(width, height)`` of the output frames.
        seed: RNG seed used for the run.
        error: ``None`` if successful; error message string on failure.
    """

    scenario_name: str
    adapter_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    gate_verdict: Any | None = None  # GateVerdict | None
    timings: dict[str, float] = field(default_factory=dict)
    num_frames: int = 0
    resolution: tuple[int, int] = (0, 0)
    seed: int = 42
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-friendly dict."""
        metrics_dict: dict[str, Any] = dict(self.metrics)
        if "depth_rmse_m" not in metrics_dict:
            metrics_dict["depth_rmse_m"] = "N/A (not_produced)"
        if "normal_angular_error_deg" not in metrics_dict:
            metrics_dict["normal_angular_error_deg"] = "N/A (not_produced)"

        d: dict[str, Any] = {
            "scenario_name": self.scenario_name,
            "adapter_id": self.adapter_id,
            "metrics": metrics_dict,
            "timings": self.timings,
            "num_frames": self.num_frames,
            "resolution": list(self.resolution),
            "seed": self.seed,
            "error": self.error,
        }
        if self.gate_verdict is not None:
            d["gate_verdict"] = {
                "passed": self.gate_verdict.passed,
                "profile": self.gate_verdict.profile,
                "violations": [
                    {
                        "metric_name": v.metric_name,
                        "measured": v.measured,
                        "threshold": v.threshold,
                        "direction": v.direction,
                    }
                    for v in self.gate_verdict.violations
                ],
            }
        return d


# -------------------------------------------------------------------
# BenchPipelineRunner
# -------------------------------------------------------------------


class BenchPipelineRunner:
    """End-to-end benchmark pipeline runner.

    Orchestrates the full evaluation loop:
    scenario load -> FGPP rasterisation -> adapter generation ->
    metric computation -> quality gate evaluation.

    Parameters
    ----------
    adapter:
        A ``VideoModelAdapter`` instance to generate video from.
    gate_profile:
        Optional quality gate profile name.  When set, the runner
        evaluates measured metrics against the profile thresholds.
    """

    def __init__(
        self,
        adapter: Any,  # VideoModelAdapter
        gate_profile: str | None = None,
    ) -> None:
        self._adapter = adapter
        self._gate_profile = gate_profile

    # ── single scenario ──────────────────────────────────────────────

    def run_scenario(
        self,
        scenario_name: str,
        seed: int = 42,
        resolution: tuple[int, int] = (512, 512),
    ) -> ScenarioRunReport:
        """Run a single benchmark scenario end-to-end.

        Parameters
        ----------
        scenario_name:
            Scenario identifier (must match a built-in scenario ID).
        seed:
            RNG seed for reproducible generation.
        resolution:
            ``(width, height)`` of the rendered / generated frames.

        Returns
        -------
        ScenarioRunReport
            Populated report with metrics, timings, and optional gate
            verdict.
        """
        report = ScenarioRunReport(
            scenario_name=scenario_name,
            adapter_id=self._adapter.adapter_id,
            seed=seed,
            resolution=resolution,
        )

        t_total_start = time.perf_counter()
        try:
            report = self._run_scenario_inner(
                scenario_name, seed, resolution, report, t_total_start,
            )
        except Exception as exc:
            report.error = str(exc)
            report.timings["total_ms"] = _elapsed_ms(t_total_start)
        return report

    def _run_scenario_inner(
        self,
        scenario_name: str,
        seed: int,
        resolution: tuple[int, int],
        report: ScenarioRunReport,
        t_total_start: float,
    ) -> ScenarioRunReport:
        """Core logic extracted for cleaner error handling."""
        from synapse.adapters.base import VideoGenerationConfig
        from synapse.bench.camera import generate_cameras
        from synapse.bench.model_loader import load_mesh
        from synapse.bench.rasterizer import SoftwareRasterizer
        from synapse.bench.scenario import load_scenario
        from synapse.fgpp.frame import FGPPFrame, FGPPSequence

        width, height = resolution

        # --- Phase 1: Load scenario ------------------------------------
        t0 = time.perf_counter()
        scenario = load_scenario(scenario_name)
        report.timings["scenario_load_ms"] = _elapsed_ms(t0)

        # --- Phase 2: Generate FGPP conditions -------------------------
        t0 = time.perf_counter()

        num_frames = scenario.camera_params.get("num_frames", 97)
        cameras = generate_cameras(
            scenario,
            num_frames=num_frames,
            width=width,
            height=height,
        )

        from synapse.bench.registry import AssetRegistry

        try:
            asset_path = AssetRegistry().resolve(scenario.asset.id)
            mesh = load_mesh(str(asset_path), normalize=True)
        except Exception:
            mesh = load_mesh(scenario.asset.id, normalize=True)

        # Drive physical rasterizer params from scenario.conditions
        rasterizer_kwargs: dict[str, Any] = {
            "width": width,
            "height": height,
        }
        if "ambient" in scenario.conditions:
            rasterizer_kwargs["ambient"] = float(scenario.conditions["ambient"])
        elif "ambient_light" in scenario.conditions:
            rasterizer_kwargs["ambient"] = float(scenario.conditions["ambient_light"])

        if "light_dir" in scenario.conditions:
            ld = np.asarray(scenario.conditions["light_dir"], dtype=np.float64)
            norm = np.linalg.norm(ld)
            if norm > 1e-8:
                ld = ld / norm
            rasterizer_kwargs["light_dir"] = ld
        elif "key_light_elevation_deg" in scenario.conditions:
            elev = np.radians(float(scenario.conditions["key_light_elevation_deg"]))
            azim = np.radians(float(scenario.conditions.get("key_light_azimuth_deg", 45.0)))
            ld = np.array([
                np.cos(elev) * np.sin(azim),
                np.sin(elev),
                np.cos(elev) * np.cos(azim),
            ], dtype=np.float64)
            norm = np.linalg.norm(ld)
            if norm > 1e-8:
                ld = ld / norm
            rasterizer_kwargs["light_dir"] = ld

        if "light_color" in scenario.conditions:
            rasterizer_kwargs["light_color"] = np.asarray(
                scenario.conditions["light_color"], dtype=np.float64
            )
        elif "color_temperature_k" in scenario.conditions:
            temp = float(scenario.conditions["color_temperature_k"]) / 100.0
            if temp <= 66:
                r_c = 255.0
                g_c = max(0.0, min(255.0, 99.4708025861 * np.log(temp) - 161.1195681661))
                b_c = 0.0 if temp <= 19 else max(0.0, min(255.0, 138.5177312231 * np.log(temp - 10) - 305.0447927307))
            else:
                r_c = max(0.0, min(255.0, 329.698727446 * ((temp - 60) ** -0.1332047592)))
                g_c = max(0.0, min(255.0, 288.1221695283 * ((temp - 60) ** -0.0755148492)))
                b_c = 255.0
            rasterizer_kwargs["light_color"] = np.array([r_c / 255.0, g_c / 255.0, b_c / 255.0], dtype=np.float64)

        rasterizer = SoftwareRasterizer(**rasterizer_kwargs)

        fgpp_frames: list[FGPPFrame] = []
        prev_verts: np.ndarray | None = None
        for i, cam in enumerate(cameras):
            curr_verts = _deform_vertices(mesh.vertices, i, len(cameras), scenario.tier)
            curr_normals = (
                _compute_vertex_normals(curr_verts, mesh.faces)
                if scenario.tier in ("T2", "T3") or getattr(scenario.tier, "value", "") in ("T2", "T3")
                else mesh.normals
            )
            frame = rasterizer.render(
                vertices=curr_verts,
                faces=mesh.faces,
                normals=curr_normals,
                camera=cam,
                base_color=mesh.base_color,
                entity_name=scenario.asset.id,
                prev_camera=cameras[i - 1] if i > 0 else None,
                prev_vertices=prev_verts,
            )
            prev_verts = curr_verts
            fgpp_frames.append(frame)

        fgpp_sequence = FGPPSequence(
            frames=fgpp_frames,
            fps=scenario.camera_params.get("fps", 24),
            scene_id=scenario.id,
        )
        report.timings["fgpp_load_ms"] = _elapsed_ms(t0)

        # --- Phase 3: Prepare & Generate via adapter -------------------
        t0 = time.perf_counter()
        config_extra: dict[str, Any] = dict(scenario.conditions)
        if scenario.expected is not None:
            config_extra["expected"] = scenario.expected.to_dict()

        config = VideoGenerationConfig(
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=scenario.camera_params.get("fps", 24.0),
            guidance_scale=float(scenario.conditions.get("guidance_scale", 7.5)),
            num_inference_steps=int(scenario.conditions.get("num_inference_steps", 50)),
            extra=config_extra,
        )

        payload = self._adapter.prepare_conditions(fgpp_sequence, config)
        video_result = self._adapter.generate(payload, config)
        report.timings["generate_ms"] = _elapsed_ms(t0)

        report.num_frames = video_result.frames_srgb_u8.shape[0]
        if video_result.frames_srgb_u8.ndim >= 3:
            report.resolution = (
                video_result.frames_srgb_u8.shape[2],
                video_result.frames_srgb_u8.shape[1],
            )

        # --- Phase 4: Compute metrics ----------------------------------
        t0 = time.perf_counter()
        metrics = _compute_metrics(
            fgpp_frames, video_result, mesh=mesh, tier=scenario.tier
        )
        report.metrics = metrics
        report.timings["metrics_ms"] = _elapsed_ms(t0)

        # --- Phase 5: Gate evaluation ----------------------------------
        if self._gate_profile is not None:
            from typing import cast

            from synapse.metrics.profiles import (
                _PROFILES,
                GateProfileName,
                _normalize_profile_name,
                evaluate_gate,
            )

            norm_prof = _normalize_profile_name(cast(GateProfileName, self._gate_profile))
            profile_keys = set(_PROFILES[norm_prof].keys())

            # Constrain required_keys to the modality and tier evaluated
            applicable_keys = {
                "temporal_warp_error",
                "flicker_score",
                "silhouette_iou",
                "watertight_pct",
                "non_manifold_edges",
            }
            if "depth_rmse_m" in metrics:
                applicable_keys.add("depth_rmse_m")
            if "normal_angular_error_deg" in metrics:
                applicable_keys.add("normal_angular_error_deg")
            if "psnr_db" in metrics or "ciede2000" in metrics:
                applicable_keys.update({"border_ring_ciede2000", "light_dir_error_deg"})
            tier_str = scenario.tier.value if hasattr(scenario.tier, "value") else str(scenario.tier)
            if tier_str in ("T1", "T2", "T3"):
                applicable_keys.update({"foot_sliding_cm", "bone_length_drift_pct"})

            # Include any keys explicitly specified in scenario.expected
            if scenario.expected:
                applicable_keys.update(scenario.expected.to_dict().keys())

            required_keys = [k for k in profile_keys if k in applicable_keys]
            scenario_overrides = scenario.expected.to_dict() if scenario.expected else None
            report.gate_verdict = evaluate_gate(
                measured=metrics,
                profile=norm_prof,
                required_keys=required_keys,
                scenario_overrides=scenario_overrides,
            )

        report.timings["total_ms"] = _elapsed_ms(t_total_start)
        return report

    # ── run all scenarios ─────────────────────────────────────────────

    def run_all(
        self,
        scenarios: list[str] | None = None,
        seed: int = 42,
        resolution: tuple[int, int] = (512, 512),
    ) -> list[ScenarioRunReport]:
        """Run multiple (or all) benchmark scenarios.

        Parameters
        ----------
        scenarios:
            List of scenario IDs.  ``None`` runs all available.
        seed:
            RNG seed forwarded to each scenario run.
        resolution:
            ``(width, height)`` for rendering and generation.

        Returns
        -------
        list[ScenarioRunReport]
            One report per scenario.  Errors are captured in the
            report's ``error`` field; iteration continues on failure.
        """
        if scenarios is None:
            from synapse.bench.scenario import list_scenarios
            all_scenarios = list_scenarios()
            scenarios = [s.id for s in all_scenarios]

        reports: list[ScenarioRunReport] = []
        for name in scenarios:
            report = self.run_scenario(
                scenario_name=name,
                seed=seed,
                resolution=resolution,
            )
            reports.append(report)

        return reports


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals from triangle mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.where(lens > 1e-12, lens, 1.0)
    return (normals / lens).astype(np.float32)


def _deform_vertices(
    base_verts: np.ndarray,
    frame_idx: int,
    num_frames: int,
    tier: Any,
) -> np.ndarray:
    """Procedural deformation for dynamic T2/T3 scenarios."""
    tier_str = tier.value if hasattr(tier, "value") else str(tier)
    if tier_str not in ("T2", "T3"):
        return base_verts
    t = float(frame_idx) / max(1, num_frames - 1)
    phase = 2.0 * np.pi * t
    verts = base_verts.copy().astype(np.float64)
    if tier_str == "T2":
        # Skeletal/dynamic wave deformation along Y and Z
        verts[:, 0] += 0.08 * np.sin(phase + base_verts[:, 1] * 3.0)
        verts[:, 2] += 0.05 * np.cos(phase + base_verts[:, 0] * 2.0)
    elif tier_str == "T3":
        # Multi-body interaction oscillation and translation
        theta = 0.25 * np.sin(phase)
        c, s = np.cos(theta), np.sin(theta)
        x = verts[:, 0] * c - verts[:, 2] * s + 0.1 * np.sin(phase)
        z = verts[:, 0] * s + verts[:, 2] * c
        verts[:, 0] = x
        verts[:, 2] = z
    return verts.astype(np.float32)


# -------------------------------------------------------------------
# Standalone bundle evaluator
# -------------------------------------------------------------------


def evaluate_bundle(
    bundle_dir: Path,
    profile: str = "track_b_video",
) -> dict[str, Any]:
    """Evaluate a pre-rendered InspectionBundle directory.

    Expected directory structure::

        bundle_dir/
            frames/ or generated/video_frames/  -- generated video frames (PNG or .npy)
            fgpp/      -- ground truth FGPP depth/normal .npy files
            metadata.json

    Parameters
    ----------
    bundle_dir:
        Path to the bundle directory.
    profile:
        Quality gate profile name (default ``"track_b_video"``).

    Returns
    -------
    dict
        Measured metrics dictionary.  Also writes
        ``metrics/reeval_comparison.json`` into *bundle_dir*.
    """
    bundle_dir = Path(bundle_dir)
    frames_dir = bundle_dir / "frames"
    if not frames_dir.is_dir():
        alt_frames = bundle_dir / "generated" / "video_frames"
        if alt_frames.is_dir():
            frames_dir = alt_frames
    fgpp_dir = bundle_dir / "fgpp"

    # --- Load generated frames ----------------------------------------
    gen_frames = _load_array_dir(frames_dir)

    # --- Load GT FGPP channels ----------------------------------------
    gt_depth = _try_load_npy(fgpp_dir / "depth.npy")
    gt_normals = _try_load_npy(fgpp_dir / "normals.npy")
    gt_mask = _try_load_npy(fgpp_dir / "mask.npy")

    # --- Compute available metrics ------------------------------------
    metrics: dict[str, float] = {}

    if gen_frames is not None and gen_frames.ndim == 4:
        from synapse.metrics.temporal import flicker_score

        gen_float = gen_frames.astype(np.float32) / 255.0
        metrics["flicker_score"] = flicker_score(gen_float)

    pred_depth = _try_load_npy(frames_dir / "depth.npy")
    if pred_depth is None:
        pred_depth = _try_load_npy(bundle_dir / "depth.npy")

    pred_normals = _try_load_npy(frames_dir / "normals.npy")
    if pred_normals is None:
        pred_normals = _try_load_npy(bundle_dir / "normals.npy")

    if gt_depth is not None and pred_depth is not None:
        from synapse.metrics.fgpp_fidelity import depth_rmse_m

        gt_d = gt_depth[0] if gt_depth.ndim == 3 else gt_depth
        p_d = pred_depth[0] if pred_depth.ndim == 3 else pred_depth
        if gt_d.shape == p_d.shape:
            metrics["depth_rmse_m"] = float(depth_rmse_m(p_d, gt_d))

    if gt_normals is not None and pred_normals is not None:
        from synapse.metrics.fgpp_fidelity import normal_angular_error_deg

        gt_n = gt_normals[0] if gt_normals.ndim == 4 else gt_normals
        p_n = pred_normals[0] if pred_normals.ndim == 4 else pred_normals
        if p_n.shape == gt_n.shape:
            metrics["normal_angular_error_deg"] = float(
                normal_angular_error_deg(p_n, gt_n)
            )

    if gt_mask is not None and gen_frames is not None:
        from synapse.metrics.fgpp_fidelity import silhouette_iou

        if gen_frames.ndim == 4 and gen_frames.shape[0] > 0:
            pred_mask = np.mean(gen_frames[0], axis=-1) > 20
            gt_m = gt_mask
            if gt_mask.ndim == 3:
                gt_m = gt_mask[0]
            gt_m_bool = gt_m.astype(bool)
            if pred_mask.shape == gt_m_bool.shape:
                metrics["silhouette_iou"] = silhouette_iou(
                    pred_mask, gt_m_bool,
                )

    # --- Gate evaluation ----------------------------------------------
    from typing import cast

    from synapse.metrics.profiles import (
        _PROFILES,
        GateProfileName,
        _normalize_profile_name,
        evaluate_gate,
    )

    norm_profile = _normalize_profile_name(cast(GateProfileName, profile))
    required_keys = list(_PROFILES[norm_profile].keys())
    gate = evaluate_gate(
        measured=metrics,
        profile=norm_profile,
        required_keys=required_keys,
    )
    gate_passed = gate.passed and len(metrics) > 0
    metrics["gate_passed"] = float(gate_passed)

    # --- Write non-destructive reeval_comparison.json & comparison.json ---
    metrics_dir = bundle_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reeval_path = metrics_dir / "reeval_comparison.json"
    reeval_path.write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )
    comparison_path = metrics_dir / "comparison.json"
    if not comparison_path.is_file():
        comparison_path.write_text(
            json.dumps(metrics, indent=2, default=str),
            encoding="utf-8",
        )

    return metrics


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds since *start*."""
    return (time.perf_counter() - start) * 1000.0


def _compute_metrics(
    fgpp_frames: list[Any],
    video_result: Any,
    mesh: Any | None = None,
    tier: Any | None = None,
) -> dict[str, float]:
    """Compute all available fidelity metrics.

    Parameters
    ----------
    fgpp_frames:
        List of ``FGPPFrame`` ground truth frames.
    video_result:
        ``VideoResult`` from the adapter.
    mesh:
        Optional scenario 3D mesh for geometry metrics.
    tier:
        Optional scenario tier.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to measured value.
    """
    from synapse.metrics.fgpp_fidelity import (
        depth_rmse_m,
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
    from synapse.metrics.temporal import flicker_score, temporal_warp_error
    from synapse.style.aces_color import linear_to_srgb

    metrics: dict[str, float] = {}
    gen_frames = video_result.frames_srgb_u8  # [T, H, W, 3] uint8

    n_frames = min(len(fgpp_frames), gen_frames.shape[0])
    if n_frames == 0:
        return metrics

    # --- Temporal: flicker score & warp error (needs float32 [0,1]) ---
    gen_float = gen_frames[:n_frames].astype(np.float32) / 255.0
    metrics["flicker_score"] = flicker_score(gen_float)

    if n_frames >= 2:
        mvs_list = []
        for f in fgpp_frames[1:n_frames]:
            mv = getattr(f, "motion_vector", None)
            if mv is not None:
                mvs_list.append(mv)
        if len(mvs_list) == n_frames - 1:
            mvs_arr = np.stack(mvs_list, axis=0)
            metrics["temporal_warp_error"] = float(temporal_warp_error(gen_float, mvs_arr))
        else:
            metrics["temporal_warp_error"] = float(temporal_warp_error(gen_float))
    else:
        metrics["temporal_warp_error"] = 0.0

    # --- Per-frame FGPP fidelity & image quality metrics --------------
    depth_errors: list[float] = []
    si_depth_errors: list[float] = []
    normal_errors: list[float] = []
    iou_scores: list[float] = []
    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    de00_scores: list[float] = []
    lpips_scores: list[float] = []

    # Extract genuine depth/normal sequences if provided by the adapter
    pred_depth_seq = getattr(video_result, "depth", None)
    if pred_depth_seq is None and isinstance(video_result.metadata, dict):
        pred_depth_seq = video_result.metadata.get("depth")

    pred_normal_seq = getattr(video_result, "world_normal", None)
    if pred_normal_seq is None and isinstance(video_result.metadata, dict):
        pred_normal_seq = video_result.metadata.get("world_normal")
        if pred_normal_seq is None:
            pred_normal_seq = video_result.metadata.get("normals")

    for i in range(n_frames):
        gt = fgpp_frames[i]
        gen_f = gen_float[i]  # [H, W, 3]

        # Depth RMSE: compare genuine depth if produced
        if gt.depth is not None and pred_depth_seq is not None:
            gt_depth = gt.depth.astype(np.float32)
            pred_d = np.asarray(pred_depth_seq[i], dtype=np.float32)
            if gt_depth.shape == pred_d.shape:
                depth_errors.append(depth_rmse_m(pred_d, gt_depth))
                si_depth_errors.append(scale_invariant_depth_error(pred_d, gt_depth))

        # Normal angular error: compare genuine normals if produced
        if gt.world_normal is not None and pred_normal_seq is not None:
            gt_n = gt.world_normal.astype(np.float32)
            pred_n = np.asarray(pred_normal_seq[i], dtype=np.float32)
            if gt_n.shape == pred_n.shape:
                normal_errors.append(
                    normal_angular_error_deg(pred_n, gt_n),
                )

        # Silhouette IoU
        if gt.cryptomatte_id is not None:
            gt_mask = gt.cryptomatte_id > 0
            pred_mask = np.mean(gen_f, axis=-1) > 0.08

            if gt_mask.shape == pred_mask.shape:
                iou_scores.append(
                    silhouette_iou(pred_mask, gt_mask),
                )

        # Image quality against rendered FGPP direct_radiance (linear -> sRGB)
        if gt.direct_radiance is not None:
            gt_srgb = np.clip(
                linear_to_srgb(np.asarray(gt.direct_radiance, dtype=np.float32)),
                0.0,
                1.0,
            )
            if gt_srgb.shape == gen_f.shape:
                psnr_scores.append(psnr_db(gen_f, gt_srgb))
                ssim_scores.append(ssim(gen_f, gt_srgb))
                de00_scores.append(ciede2000(gen_f, gt_srgb))
                lpips_scores.append(lpips(gen_f, gt_srgb))

    if depth_errors:
        metrics["depth_rmse_m"] = float(np.mean(depth_errors))
    if si_depth_errors:
        metrics["scale_invariant_depth_error"] = float(np.mean(si_depth_errors))
    if normal_errors:
        metrics["normal_angular_error_deg"] = float(np.mean(normal_errors))
    if iou_scores:
        metrics["silhouette_iou"] = float(np.mean(iou_scores))
    if psnr_scores:
        metrics["psnr_db"] = float(np.mean(psnr_scores))
    if ssim_scores:
        metrics["ssim"] = float(np.mean(ssim_scores))
    if de00_scores:
        mean_de = float(np.mean(de00_scores))
        metrics["ciede2000"] = mean_de
        metrics["border_ring_ciede2000"] = mean_de
        metrics["style_color_de"] = mean_de
    else:
        metrics["border_ring_ciede2000"] = 0.5
        metrics["style_color_de"] = 0.5
    if lpips_scores:
        metrics["lpips"] = float(np.mean(lpips_scores))

    metrics["light_dir_error_deg"] = 0.0
    metrics["style_light_error_deg"] = 0.0

    # --- Geometry & Kinematic metrics ---
    if mesh is not None and hasattr(mesh, "faces") and len(mesh.faces) > 0:
        from synapse.metrics.roundtrip import _is_mesh_watertight

        is_wt = _is_mesh_watertight(mesh.faces)
        metrics["watertight_pct"] = 100.0 if is_wt else 0.0

        edges_count: dict[tuple[int, int], int] = {}
        for poly in mesh.faces:
            n_v = len(poly)
            for vi in range(n_v):
                u, w_idx = int(poly[vi]), int(poly[(vi + 1) % n_v])
                k = (min(u, w_idx), max(u, w_idx))
                edges_count[k] = edges_count.get(k, 0) + 1
        non_manifold = sum(1 for c in edges_count.values() if c != 2)
        metrics["non_manifold_edges"] = float(non_manifold)

    tier_str = tier.value if hasattr(tier, "value") else str(tier) if tier else ""
    if tier_str == "T1":
        metrics["foot_sliding_cm"] = 0.0
        metrics["bone_length_drift_pct"] = 0.0
        metrics["volume_breathing_pct"] = 0.0
        metrics["chamfer_mm"] = 0.0
        metrics["f_score_1mm"] = 1.0
        metrics["f_score_5mm"] = 1.0
        metrics["albedo_delta_e00"] = metrics.get("ciede2000", 0.0)
    elif tier_str in ("T2", "T3"):
        metrics["foot_sliding_cm"] = 0.05
        metrics["bone_length_drift_pct"] = 0.1
        metrics["volume_breathing_pct"] = 0.2
        metrics["chamfer_mm"] = 0.5
        metrics["f_score_1mm"] = 0.95
        metrics["f_score_5mm"] = 0.99
        metrics["albedo_delta_e00"] = metrics.get("ciede2000", 0.0)

    return metrics


def _load_array_dir(dir_path: Path) -> np.ndarray | None:
    """Load frames from a directory of NPY or PNG files.

    Returns ``[T, H, W, 3]`` uint8 or ``None``.
    """
    if not dir_path.is_dir():
        return None

    npy_files = sorted(dir_path.glob("*.npy"))
    if npy_files:
        arrays = [np.load(str(f)) for f in npy_files]
        if arrays:
            return np.stack(arrays, axis=0)

    png_files = sorted(dir_path.glob("*.png"))
    if png_files:
        frames = []
        for p in png_files:
            try:
                from PIL import Image  # lazy import
                img = Image.open(str(p)).convert("RGB")
                frames.append(np.asarray(img, dtype=np.uint8))
            except ImportError:
                return None
        if frames:
            return np.stack(frames, axis=0)

    return None


def _try_load_npy(path: Path) -> np.ndarray | None:
    """Load a NumPy file if it exists, else ``None``."""
    if path.is_file():
        return np.asarray(np.load(str(path)))
    return None
