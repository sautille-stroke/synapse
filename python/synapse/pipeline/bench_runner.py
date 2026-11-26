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

from synapse.observability import get_probe, traceboard_stage

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
    external_dataset_meta: dict[str, Any] = field(default_factory=dict)

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
        if self.external_dataset_meta:
            d["external_dataset_meta"] = dict(self.external_dataset_meta)
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
        rasterizer_cache: dict[str, Any] | None = None,
    ) -> None:
        self._adapter = adapter
        self._gate_profile = gate_profile
        self._rasterizer_cache = rasterizer_cache

    # ── single scenario ──────────────────────────────────────────────

    def run_scenario(
        self,
        scenario_name: str,
        seed: int = 42,
        resolution: tuple[int, int] = (512, 512),
        num_frames: int | None = None,
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
        num_frames:
            Optional frame count override. When omitted, uses scenario default.

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
        with traceboard_stage(
            f"bench/{scenario_name}",
            adapter=self._adapter.adapter_id,
            seed=str(seed),
            resolution=f"{resolution[0]}x{resolution[1]}",
        ) as stage:
            try:
                report = self._run_scenario_inner(
                    scenario_name,
                    seed,
                    resolution,
                    report,
                    t_total_start,
                    num_frames=num_frames,
                )
            except Exception as exc:
                report.error = str(exc)
                report.timings["total_ms"] = _elapsed_ms(t_total_start)

            # Log metrics and timings to Traceboard (finite values only)
            import math

            from synapse.observability import safe_log_scalar, safe_log_text

            probe = get_probe()
            if probe is not None and probe.active:
                prefix = f"bench/scenario/{scenario_name}"
                passed = report.error is None and (
                    report.gate_verdict is None or report.gate_verdict.passed
                )
                safe_log_scalar(f"{prefix}/passed", 1.0 if passed else 0.0)
                for k, v in report.metrics.items():
                    if math.isfinite(float(v)):
                        safe_log_scalar(f"{prefix}/metrics/{k}", float(v))
                for k, v in report.timings.items():
                    if math.isfinite(float(v)):
                        safe_log_scalar(f"{prefix}/timing/{k}", float(v))
                if report.num_frames > 0:
                    safe_log_scalar(
                        f"{prefix}/num_frames",
                        float(report.num_frames),
                    )
                if report.error is not None:
                    safe_log_text(f"{prefix}/error", report.error)

            # Set stage verdict
            if stage is not None:
                if report.error is not None:
                    stage.set_fail(report.error)
                elif (
                    report.gate_verdict is not None
                    and not report.gate_verdict.passed
                ):
                    stage.set_fail("gate_failed")
                else:
                    stage.set_pass()

        return report

    def _run_scenario_inner(
        self,
        scenario_name: str,
        seed: int,
        resolution: tuple[int, int],
        report: ScenarioRunReport,
        t_total_start: float,
        num_frames: int | None = None,
    ) -> ScenarioRunReport:
        """Core logic extracted for cleaner error handling."""
        from synapse.adapters.base import VideoGenerationConfig
        from synapse.bench.model_loader import load_mesh
        from synapse.bench.scenario import load_scenario
        from synapse.fgpp.frame import FGPPFrame, FGPPSequence
        from synapse.render_core.camera import generate_cameras
        from synapse.render_core.rasterizer import SoftwareRasterizer

        width, height = resolution

        # --- Phase 1: Load scenario ------------------------------------
        t0 = time.perf_counter()
        scenario = load_scenario(scenario_name)
        report.timings["scenario_load_ms"] = _elapsed_ms(t0)

        # --- Phase 2: Generate FGPP conditions -------------------------
        t0 = time.perf_counter()

        actual_num_frames = (
            num_frames if num_frames is not None else scenario.camera_params.get("num_frames", 97)
        )
        cameras = generate_cameras(
            scenario,
            num_frames=actual_num_frames,
            width=width,
            height=height,
        )

        from synapse.bench.registry import AssetRegistry, filter_stage0_quality

        registry = AssetRegistry()

        def _load_asset_mesh(asset_obj: Any) -> Any:
            if not isinstance(asset_obj, str):
                ap, meta = registry.resolve_with_metadata(asset_obj)
                report.external_dataset_meta = meta
                loaded = load_mesh(str(ap), normalize=True)
                ok_s0, reasons = filter_stage0_quality(loaded.vertices, loaded.faces)
                report.external_dataset_meta["stage0_passed"] = ok_s0
                if reasons:
                    report.external_dataset_meta["stage0_reasons"] = reasons
                return loaded
            try:
                ap = registry.resolve(asset_obj)
                return load_mesh(str(ap), normalize=True)
            except Exception:
                return load_mesh(asset_obj, normalize=True)

        mesh = _load_asset_mesh(scenario.asset)
        mesh, entity_slices = _build_scene_mesh(scenario, mesh, _load_asset_mesh)

        # Drive physical rasterizer params from scenario.conditions (T1/T4 lighting)
        rasterizer_kwargs = _build_rasterizer_kwargs(scenario.conditions, width, height)
        multi_light_count = int(scenario.conditions.get("light_directions", 0) or 0)

        def _get_rasterizer(kwargs: dict[str, Any]) -> Any:
            ambient_val = kwargs.get("ambient", 0.1)
            light_dir_val = tuple(kwargs.get("light_dir", (0.0, 1.0, 0.0)))
            light_col_val = tuple(kwargs.get("light_color", (1.0, 1.0, 1.0)))
            ckey = f"{width}x{height}_a{ambient_val}_ld{light_dir_val}_lc{light_col_val}"
            if self._rasterizer_cache is not None and ckey in self._rasterizer_cache:
                return self._rasterizer_cache[ckey]
            rast = SoftwareRasterizer(**kwargs)
            if self._rasterizer_cache is not None:
                self._rasterizer_cache[ckey] = rast
            return rast

        base_rasterizer = _get_rasterizer(rasterizer_kwargs)
        fps = float(scenario.camera_params.get("fps", 24.0))
        has_motion = bool(scenario.conditions.get("animation")) or (
            scenario.tier in ("T2", "T3")
            or getattr(scenario.tier, "value", "") in ("T2", "T3")
        )

        fgpp_frames: list[FGPPFrame] = []
        prev_verts: np.ndarray | None = None
        for i, cam in enumerate(cameras):
            curr_verts = _deform_vertices(
                mesh.vertices,
                i,
                len(cameras),
                scenario.tier,
                conditions=scenario.conditions,
                fps=fps,
                entity_slices=entity_slices,
            )
            curr_normals = (
                _compute_vertex_normals(curr_verts, mesh.faces)
                if has_motion
                else mesh.normals
            )
            if multi_light_count > 1:
                az = 2.0 * np.pi * (i % multi_light_count) / float(multi_light_count)
                frame_kwargs = dict(rasterizer_kwargs)
                frame_kwargs["light_dir"] = np.array(
                    [np.cos(0.6) * np.sin(az), np.sin(0.6), np.cos(0.6) * np.cos(az)],
                    dtype=np.float64,
                )
                frame_rasterizer = _get_rasterizer(frame_kwargs)
            else:
                frame_rasterizer = base_rasterizer

            frame = frame_rasterizer.render(
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
            num_frames=actual_num_frames,
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
        metrics = _compute_metrics(fgpp_frames, video_result, mesh=mesh, tier=scenario.tier)
        report.metrics = metrics
        report.timings["metrics_ms"] = _elapsed_ms(t0)

        # --- Phase 5: Gate evaluation ----------------------------------
        if self._gate_profile is not None:
            from typing import cast

            from synapse.metrics.profiles import (
                _PROFILES,
                METRIC_ALIASES,
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
            }
            if "depth_rmse_m" in metrics:
                applicable_keys.add("depth_rmse_m")
            if "normal_angular_error_deg" in metrics:
                applicable_keys.add("normal_angular_error_deg")
            if "border_ring_ciede2000" in metrics:
                applicable_keys.add("border_ring_ciede2000")
            if "light_dir_error_deg" in metrics:
                applicable_keys.add("light_dir_error_deg")
            if "psnr_db" in metrics:
                applicable_keys.add("psnr_db")
            if "ssim" in metrics:
                applicable_keys.add("ssim")
            if "ciede2000" in metrics:
                applicable_keys.add("ciede2000")
            if "lpips" in metrics:
                applicable_keys.add("lpips")
            if "watertight_pct" in metrics:
                applicable_keys.update({"watertight_pct", "non_manifold_edges"})

            tier_str = (
                scenario.tier.value if hasattr(scenario.tier, "value") else str(scenario.tier)
            )
            if tier_str in ("T1", "T2", "T3") and "foot_sliding_cm" in metrics:
                applicable_keys.update({"foot_sliding_cm", "bone_length_drift_pct"})

            # Include any keys explicitly specified in scenario.expected,
            # resolving aliases so e.g. "style_color_de" also adds
            # "border_ring_ciede2000" and vice versa.
            if scenario.expected:
                for exp_key in scenario.expected.to_dict().keys():
                    applicable_keys.add(exp_key)
                    # Forward alias: alias -> canonical
                    if exp_key in METRIC_ALIASES:
                        applicable_keys.add(METRIC_ALIASES[exp_key])
                    # Reverse alias: canonical -> alias
                    for alias, canonical in METRIC_ALIASES.items():
                        if canonical == exp_key:
                            applicable_keys.add(alias)

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
        num_frames: int | None = None,
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
        num_frames:
            Optional frame count override for each scenario run.

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
        with traceboard_stage(
            "bench/run_all",
            count=str(len(scenarios)),
        ) as all_stage:
            for name in scenarios:
                report = self.run_scenario(
                    scenario_name=name,
                    seed=seed,
                    resolution=resolution,
                    num_frames=num_frames,
                )
                reports.append(report)
            if all_stage is not None:
                from synapse.bench.report import compute_pillar_scores
                from synapse.observability import safe_log_scalar

                n_pass = sum(
                    1
                    for r in reports
                    if r.error is None and (r.gate_verdict is None or r.gate_verdict.passed)
                )
                n_fail = len(reports) - n_pass
                safe_log_scalar("bench/summary/total_scenarios", float(len(reports)))
                safe_log_scalar("bench/summary/passed", float(n_pass))
                safe_log_scalar("bench/summary/failed", float(n_fail))
                safe_log_scalar(
                    "bench/summary/pass_rate", float(n_pass) / max(len(reports), 1)
                )
                for pillar, score in sorted(compute_pillar_scores(reports).items()):
                    safe_log_scalar(f"bench/pillar/{pillar}", float(score))
                if n_fail > 0:
                    all_stage.set_fail("one_or_more_scenarios_failed")
                else:
                    all_stage.set_pass()

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


def _build_rasterizer_kwargs(
    conditions: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Translate T1/T4 scenario lighting conditions into SoftwareRasterizer kwargs."""
    kwargs: dict[str, Any] = {"width": width, "height": height}
    style = str(conditions.get("style", "")).lower()

    if "ambient" in conditions:
        kwargs["ambient"] = float(conditions["ambient"])
    elif "ambient_light" in conditions:
        kwargs["ambient"] = float(conditions["ambient_light"])
    elif "ambient_intensity" in conditions:
        kwargs["ambient"] = float(conditions["ambient_intensity"])
    elif style == "backlit_silhouette":
        kwargs["ambient"] = float(conditions.get("fill_intensity", 0.1))
    elif style == "high_contrast_split":
        ratio = max(float(conditions.get("key_fill_ratio", 8.0)), 1.0)
        kwargs["ambient"] = 1.0 / ratio
    elif style == "three_point":
        kwargs["ambient"] = 0.25

    if "light_dir" in conditions:
        ld = np.asarray(conditions["light_dir"], dtype=np.float64)
        norm = float(np.linalg.norm(ld))
        if norm > 1e-8:
            kwargs["light_dir"] = ld / norm
    elif style == "backlit_silhouette":
        ld = np.array([0.0, 0.2, -1.0], dtype=np.float64)
        kwargs["light_dir"] = ld / float(np.linalg.norm(ld))
    elif (
        "key_light_elevation_deg" in conditions
        or "key_light_azimuth_deg" in conditions
        or "key_azimuth_deg" in conditions
    ):
        elev = np.radians(float(conditions.get("key_light_elevation_deg", 35.0)))
        azim = np.radians(
            float(
                conditions.get(
                    "key_light_azimuth_deg",
                    conditions.get("key_azimuth_deg", 45.0),
                )
            )
        )
        ld = np.array(
            [
                np.cos(elev) * np.sin(azim),
                np.sin(elev),
                np.cos(elev) * np.cos(azim),
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(ld))
        if norm > 1e-8:
            kwargs["light_dir"] = ld / norm

    if "light_color" in conditions:
        kwargs["light_color"] = np.asarray(conditions["light_color"], dtype=np.float64)
    elif "color_temperature_k" in conditions:
        temp = float(conditions["color_temperature_k"]) / 100.0
        if temp <= 66:
            r_c = 255.0
            g_c = max(0.0, min(255.0, 99.4708025861 * np.log(temp) - 161.1195681661))
            b_c = (
                0.0
                if temp <= 19
                else max(0.0, min(255.0, 138.5177312231 * np.log(temp - 10) - 305.0447927307))
            )
        else:
            r_c = max(0.0, min(255.0, 329.698727446 * ((temp - 60) ** -0.1332047592)))
            g_c = max(0.0, min(255.0, 288.1221695283 * ((temp - 60) ** -0.0755148492)))
            b_c = 255.0
        kwargs["light_color"] = np.array(
            [r_c / 255.0, g_c / 255.0, b_c / 255.0], dtype=np.float64
        )
    elif style == "ambient_only":
        direct = float(conditions.get("direct_intensity", 0.0))
        kwargs["light_color"] = np.full(3, max(0.0, direct), dtype=np.float64)
    elif style == "backlit_silhouette":
        intensity = min(float(conditions.get("backlight_intensity", 5.0)), 5.0)
        kwargs["light_color"] = np.full(3, intensity, dtype=np.float64)

    return kwargs


def _build_scene_mesh(
    scenario: Any,
    base_mesh: Any,
    load_asset_fn: Any,
) -> tuple[Any, list[tuple[int, int]]]:
    """Compose multi-object scenes for T3/T5 scenarios with distinct spatial layouts.

    Returns ``(combined_mesh, entity_slices)`` where ``entity_slices`` lists
    ``(start_v, end_v)`` vertex index ranges for each object in the scene.
    """
    from synapse.bench.model_loader import MeshData

    cond = scenario.conditions
    if cond.get("scene_type") != "multi_object":
        return base_mesh, [(0, len(base_mesh.vertices))]

    placements: list[tuple[Any, np.ndarray, float]] = []
    layout = str(cond.get("layout", "")).lower()

    if "occluder" in cond and "occludee" in cond:
        occludee_m = load_asset_fn(str(cond["occludee"]))
        occluder_m = load_asset_fn(str(cond["occluder"]))
        placements.append((occludee_m, np.array([0.0, 0.0, -0.45], dtype=np.float32), 0.7))
        placements.append((occluder_m, np.array([0.0, 0.0, 0.45], dtype=np.float32), 0.7))
    elif "prop" in cond:
        prop_m = load_asset_fn(str(cond["prop"]))
        placements.append((base_mesh, np.array([0.0, 0.0, 0.0], dtype=np.float32), 1.0))
        placements.append((prop_m, np.array([0.55, 0.1, 0.25], dtype=np.float32), 0.35))
    elif layout == "ring":
        count = max(2, int(cond.get("object_count", 8)))
        radius = 0.85
        for k in range(count):
            ang = 2.0 * np.pi * k / float(count)
            pos = np.array([radius * np.cos(ang), 0.0, radius * np.sin(ang)], dtype=np.float32)
            placements.append((base_mesh, pos, 0.28))
    elif layout == "vertical_stack":
        obj_names = list(cond.get("objects", [scenario.asset.id]))
        for idx, name in enumerate(obj_names):
            m = load_asset_fn(str(name))
            y_off = (idx - (len(obj_names) - 1) / 2.0) * 0.65
            placements.append((m, np.array([0.0, y_off, 0.0], dtype=np.float32), 0.5))
    elif layout == "scattered" or "objects" in cond:
        obj_names = list(cond.get("objects", [scenario.asset.id]))
        for idx, name in enumerate(obj_names):
            m = load_asset_fn(str(name))
            ang = 2.0 * np.pi * idx / max(len(obj_names), 1)
            pos = np.array([0.65 * np.cos(ang), 0.0, 0.65 * np.sin(ang)], dtype=np.float32)
            placements.append((m, pos, 0.55))
    elif int(cond.get("object_count", 1)) >= 2:
        count = int(cond["object_count"])
        for idx in range(count):
            x_off = -0.6 + 1.2 * idx / max(count - 1, 1)
            z_off = -0.25 if idx % 2 == 0 else 0.25
            placements.append(
                (base_mesh, np.array([x_off, 0.0, z_off], dtype=np.float32), 0.55)
            )
    else:
        return base_mesh, [(0, len(base_mesh.vertices))]

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    slices: list[tuple[int, int]] = []
    v_offset = 0
    for m, offset, scale in placements:
        v = m.vertices.astype(np.float32) * float(scale) + offset[None, :]
        f = m.faces.astype(np.int32) + v_offset
        n = m.normals.astype(np.float32)
        slices.append((v_offset, v_offset + len(v)))
        all_v.append(v)
        all_f.append(f)
        all_n.append(n)
        v_offset += len(v)

    combined = MeshData(
        name=str(getattr(scenario, "id", "multi_object_scene")),
        vertices=np.concatenate(all_v, axis=0),
        faces=np.concatenate(all_f, axis=0),
        normals=np.concatenate(all_n, axis=0),
        base_color=base_mesh.base_color,
    )
    return combined, slices


def _deform_vertices(
    base_verts: np.ndarray,
    frame_idx: int,
    num_frames: int,
    tier: Any,
    conditions: dict[str, Any] | None = None,
    fps: float = 24.0,
    entity_slices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Procedural animation/deformation driven by scenario.conditions and tier."""
    cond = conditions or {}
    anim = str(cond.get("animation", "")).lower()
    tier_str = tier.value if hasattr(tier, "value") else str(tier)

    if not anim and tier_str not in ("T2", "T3"):
        return base_verts

    t = float(frame_idx) / max(1, num_frames - 1)
    t_sec = float(frame_idx) / max(fps, 1.0)
    phase = 2.0 * np.pi * t
    verts = base_verts.copy().astype(np.float64)

    if anim == "fall":
        g = float(cond.get("gravity_m_s2", 9.81))
        h0 = float(cond.get("initial_height_m", 2.0))
        y_drop = min(0.5 * g * (t_sec**2), h0)
        verts[:, 1] += (h0 - y_drop) * 0.4
    elif anim == "rotate_y":
        speed = float(cond.get("rotation_speed_deg_per_s", 90.0))
        theta = np.radians(speed * t_sec)
        c, s = np.cos(theta), np.sin(theta)
        x = verts[:, 0] * c - verts[:, 2] * s
        z = verts[:, 0] * s + verts[:, 2] * c
        verts[:, 0] = x
        verts[:, 2] = z
    elif anim == "pendulum":
        amp = np.radians(float(cond.get("amplitude_deg", 45.0)))
        period = max(float(cond.get("period_s", 2.0)), 1e-3)
        theta = amp * np.sin(2.0 * np.pi * t_sec / period)
        pivot_y = float(np.max(verts[:, 1])) + 0.5
        c, s = np.cos(theta), np.sin(theta)
        dx = verts[:, 0]
        dy = verts[:, 1] - pivot_y
        verts[:, 0] = dx * c - dy * s
        verts[:, 1] = dx * s + dy * c + pivot_y
    elif anim == "walk":
        cycles = float(cond.get("walk_cycles", 2.0))
        w_phase = 2.0 * np.pi * cycles * t
        verts[:, 1] += 0.04 * abs(np.sin(w_phase))
        verts[:, 2] += 0.12 * np.sin(w_phase) * np.clip(-base_verts[:, 1], 0.0, 1.0)
        verts[:, 0] += 0.04 * np.sin(w_phase + base_verts[:, 1] * 2.5)
    elif anim in ("idle", "idle_hold"):
        verts[:, 1] += 0.02 * np.sin(phase)
        verts[:, 0] *= 1.0 + 0.015 * np.sin(phase)
    elif anim == "cross_paths" and entity_slices and len(entity_slices) >= 2:
        s0, e0 = entity_slices[0]
        s1, e1 = entity_slices[1]
        shift = 1.2 * (t - 0.5)
        verts[s0:e0, 0] += shift
        verts[s1:e1, 0] -= shift
    elif tier_str == "T2":
        verts[:, 0] += 0.08 * np.sin(phase + base_verts[:, 1] * 3.0)
        verts[:, 2] += 0.05 * np.cos(phase + base_verts[:, 0] * 2.0)
    elif tier_str == "T3":
        theta = 0.25 * np.sin(phase)
        c, s = np.cos(theta), np.sin(theta)
        x = verts[:, 0] * c - verts[:, 2] * s + 0.1 * np.sin(phase)
        z = verts[:, 0] * s + verts[:, 2] * c
        verts[:, 0] = x
        verts[:, 2] = z

    out_verts: np.ndarray = np.asarray(verts, dtype=np.float32)
    return out_verts


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
    # Try loading full sequence from EXR files first, then fall back to .npy
    gt_depth_seq = _try_load_exr_sequence(fgpp_dir, "depth")
    gt_normals_seq = _try_load_exr_sequence(fgpp_dir, "normals")
    gt_mask_seq = _try_load_exr_sequence(fgpp_dir, "mask")

    # Fall back to single .npy files
    if gt_depth_seq is None:
        gt_depth_single = _try_load_npy(fgpp_dir / "depth.npy")
        gt_depth_seq = gt_depth_single[np.newaxis] if gt_depth_single is not None and gt_depth_single.ndim == 2 else gt_depth_single
    if gt_normals_seq is None:
        gt_normals_single = _try_load_npy(fgpp_dir / "normals.npy")
        gt_normals_seq = gt_normals_single[np.newaxis] if gt_normals_single is not None and gt_normals_single.ndim == 3 else gt_normals_single
    if gt_mask_seq is None:
        gt_mask_single = _try_load_npy(fgpp_dir / "mask.npy")
        gt_mask_seq = gt_mask_single[np.newaxis] if gt_mask_single is not None and gt_mask_single.ndim == 2 else gt_mask_single

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

    # Process all frames for depth and normal metrics (not just first)
    if gt_depth_seq is not None and pred_depth is not None:
        from synapse.metrics.fgpp_fidelity import depth_rmse_m

        n_depth = min(
            gt_depth_seq.shape[0] if gt_depth_seq.ndim == 3 else 1,
            pred_depth.shape[0] if pred_depth.ndim == 3 else 1,
        )
        depth_errors: list[float] = []
        for fi in range(n_depth):
            gt_d = gt_depth_seq[fi] if gt_depth_seq.ndim == 3 else gt_depth_seq
            p_d = pred_depth[fi] if pred_depth.ndim == 3 else pred_depth
            if gt_d.shape == p_d.shape:
                depth_errors.append(float(depth_rmse_m(p_d, gt_d)))
        if depth_errors:
            metrics["depth_rmse_m"] = float(np.mean(depth_errors))

    if gt_normals_seq is not None and pred_normals is not None:
        from synapse.metrics.fgpp_fidelity import normal_angular_error_deg

        n_norm = min(
            gt_normals_seq.shape[0] if gt_normals_seq.ndim == 4 else 1,
            pred_normals.shape[0] if pred_normals.ndim == 4 else 1,
        )
        normal_errors: list[float] = []
        for fi in range(n_norm):
            gt_n = gt_normals_seq[fi] if gt_normals_seq.ndim == 4 else gt_normals_seq
            p_n = pred_normals[fi] if pred_normals.ndim == 4 else pred_normals
            if p_n.shape == gt_n.shape:
                normal_errors.append(float(normal_angular_error_deg(p_n, gt_n)))
        if normal_errors:
            metrics["normal_angular_error_deg"] = float(np.mean(normal_errors))

    if gt_mask_seq is not None and gen_frames is not None:
        from synapse.metrics.fgpp_fidelity import silhouette_iou

        n_mask = min(
            gt_mask_seq.shape[0] if gt_mask_seq.ndim == 3 else 1,
            gen_frames.shape[0] if gen_frames.ndim == 4 else 0,
        )
        iou_scores: list[float] = []
        for fi in range(n_mask):
            gt_m = gt_mask_seq[fi] if gt_mask_seq.ndim == 3 else gt_mask_seq
            gen_f = gen_frames[fi]
            pred_mask = np.mean(gen_f, axis=-1) > 20
            gt_m_bool = gt_m.astype(bool)
            if pred_mask.shape == gt_m_bool.shape:
                iou_scores.append(silhouette_iou(pred_mask, gt_m_bool))
        if iou_scores:
            metrics["silhouette_iou"] = float(np.mean(iou_scores))

    # --- Gate evaluation ----------------------------------------------
    from typing import cast

    from synapse.metrics.profiles import (
        _PROFILES,
        METRIC_ALIASES,
        GateProfileName,
        _normalize_profile_name,
        evaluate_gate,
    )

    norm_profile = _normalize_profile_name(cast(GateProfileName, profile))

    # Build required_keys from profile, resolving aliases so measured
    # metrics under alias names (e.g. "style_color_de") are matched.
    resolved_metrics = set(metrics.keys())
    for alias, canonical in METRIC_ALIASES.items():
        if alias in resolved_metrics:
            resolved_metrics.add(canonical)
        if canonical in resolved_metrics:
            resolved_metrics.add(alias)

    required_keys = [k for k in _PROFILES[norm_profile].keys() if k in resolved_metrics]
    gate = evaluate_gate(
        measured=metrics,
        profile=norm_profile,
        required_keys=required_keys,
    )
    # Fail-closed: gate passes only when at least some metrics were measured
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
            fallback_mvs = np.zeros(
                (n_frames - 1, gen_float.shape[1], gen_float.shape[2], 2),
                dtype=np.float32,
            )
            metrics["temporal_warp_error"] = float(temporal_warp_error(gen_float, fallback_mvs))
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

    border_de_scores: list[float] = []

    for i in range(n_frames):
        gt = fgpp_frames[i]
        gen_f = gen_float[i]  # [H, W, 3]

        # Depth RMSE: compare genuine depth if produced
        if gt.depth is not None and pred_depth_seq is not None:
            gt_depth = gt.depth.astype(np.float32)
            pred_d = np.asarray(pred_depth_seq[i], dtype=np.float32)
            if gt_depth.shape == pred_d.shape:
                depth_mask = gt_depth > 0
                if gt.cryptomatte_id is not None:
                    depth_mask = depth_mask & (gt.cryptomatte_id > 0)
                depth_errors.append(depth_rmse_m(pred_d, gt_depth, valid_mask=depth_mask))
                si_depth_errors.append(scale_invariant_depth_error(pred_d, gt_depth))

        # Normal angular error: compare genuine normals if produced
        if gt.world_normal is not None and pred_normal_seq is not None:
            gt_n = gt.world_normal.astype(np.float32)
            pred_n = np.asarray(pred_normal_seq[i], dtype=np.float32)
            if gt_n.shape == pred_n.shape:
                norm_mask = np.linalg.norm(gt_n, axis=-1) > 1e-4
                if gt.cryptomatte_id is not None:
                    norm_mask = norm_mask & (gt.cryptomatte_id > 0)
                normal_errors.append(
                    normal_angular_error_deg(pred_n, gt_n, valid_mask=norm_mask),
                )

        # Silhouette IoU
        if gt.cryptomatte_id is not None:
            gt_mask = gt.cryptomatte_id > 0
            pred_mask = np.mean(gen_f, axis=-1) > 0.08

            if gt_mask.shape == pred_mask.shape:
                iou_scores.append(
                    silhouette_iou(pred_mask, gt_mask),
                )

        # Image quality against rendered FGPP direct_radiance (already linear sRGB from rasterizer)
        if gt.direct_radiance is not None:
            rad = np.asarray(gt.direct_radiance, dtype=np.float32)
            gt_srgb = np.clip(
                linear_to_srgb(rad),
                0.0,
                1.0,
            )
            if gt_srgb.shape == gen_f.shape:
                psnr_scores.append(psnr_db(gen_f, gt_srgb))
                ssim_scores.append(ssim(gen_f, gt_srgb))
                de00_scores.append(ciede2000(gen_f, gt_srgb))
                lpips_scores.append(lpips(gen_f, gt_srgb))
                if gt.cryptomatte_id is not None:
                    from synapse.metrics.style_consistency import border_ring_ciede2000

                    m_fg = gt.cryptomatte_id > 0
                    if m_fg.shape == gen_f.shape[:2]:
                        border_de_scores.append(
                            border_ring_ciede2000(
                                gen_f,
                                gt_srgb,
                                mask=m_fg,
                                ring_width=16,
                                input_space="srgb",
                            )
                        )

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
        metrics["ciede2000"] = float(np.mean(de00_scores))
    if border_de_scores:
        metrics["border_ring_ciede2000"] = float(np.mean(border_de_scores))
        metrics["style_color_de"] = metrics["border_ring_ciede2000"]
    elif de00_scores:
        mean_de = float(np.mean(de00_scores))
        metrics["border_ring_ciede2000"] = mean_de
        metrics["style_color_de"] = mean_de
    if lpips_scores:
        metrics["lpips"] = float(np.mean(lpips_scores))

    # --- Style / light direction estimation (real computation) ---
    try:
        from synapse.metrics.style_consistency import light_dir_error_deg

        # Derive GT light direction from the rasterizer's configured light_dir
        # or from the FGPP intrinsic extraction on the full sequence.
        gt_ld: np.ndarray | None = None

        # First try: use the light_dir baked in the rasterizer from scenario conditions
        if n_frames > 0 and fgpp_frames[0].direct_radiance is not None:
            # Try FGPP intrinsic extraction with proper (frames_srgb, fgpp_seq) signature
            try:
                from synapse.fgpp.frame import FGPPSequence
                from synapse.style import extract_style_fgpp_intrinsic

                fgpp_seq_for_style = FGPPSequence(frames=fgpp_frames[:n_frames], fps=24)
                # Build GT sRGB frames from direct radiance (already linear sRGB)
                gt_srgb_frames_for_style = []
                for f_gt in fgpp_frames[:min(n_frames, 8)]:
                    if f_gt.direct_radiance is not None:
                        rad = np.asarray(f_gt.direct_radiance, dtype=np.float32)
                        srgb = np.clip(linear_to_srgb(rad), 0.0, 1.0)
                        gt_srgb_frames_for_style.append((srgb * 255.0).astype(np.uint8))
                if gt_srgb_frames_for_style:
                    gt_srgb_arr = np.stack(gt_srgb_frames_for_style, axis=0)
                    gt_style = extract_style_fgpp_intrinsic(gt_srgb_arr, fgpp_seq_for_style)
                    gt_ld = np.asarray(gt_style.key_light_dir_world, dtype=np.float64)
            except Exception:
                pass

        if gt_ld is not None:
            # Extract predicted light direction from generated video frames;
            # fail-closed with NaN on extraction error (never copy gt_ld -> fake 0.0 deg).
            pred_ld: np.ndarray | None = None
            try:
                from synapse.style import extract_style_optical

                # extract_style_optical expects uint8 [T, H, W, 3]
                gen_u8 = (np.clip(gen_float, 0.0, 1.0) * 255.0).astype(np.uint8)
                pred_style = extract_style_optical(gen_u8)
                pred_ld = np.asarray(pred_style.key_light_dir_world, dtype=np.float64)
            except Exception:
                pred_ld = None
            if pred_ld is not None:
                metrics["light_dir_error_deg"] = float(light_dir_error_deg(pred_ld, gt_ld))
            else:
                metrics["light_dir_error_deg"] = float("nan")
    except Exception as exc:
        logger.debug("light_dir_error_deg computation failed: %s", exc)

    try:
        from synapse.style import extract_style_optical

        style_desc = extract_style_optical(gen_float)
        if hasattr(style_desc, "color_temp_kelvin"):
            metrics["style_color_temp"] = float(style_desc.color_temp_kelvin)
    except (ImportError, Exception) as exc:
        logger.debug("extract_style_optical unavailable or failed: %s", exc)

    # --- Geometry & Kinematic metrics (real computation only on output mesh) ---
    out_mesh = getattr(video_result, "reconstructed_mesh", None) or getattr(
        video_result, "mesh", None
    )
    if out_mesh is not None and hasattr(out_mesh, "faces") and len(out_mesh.faces) > 0:
        from synapse.metrics.roundtrip import _is_mesh_watertight

        is_wt = _is_mesh_watertight(out_mesh.faces)
        metrics["watertight_pct"] = 100.0 if is_wt else 0.0

        edges_count: dict[tuple[int, int], int] = {}
        for poly in out_mesh.faces:
            n_v = len(poly)
            for vi in range(n_v):
                u, w_idx = int(poly[vi]), int(poly[(vi + 1) % n_v])
                k = (min(u, w_idx), max(u, w_idx))
                edges_count[k] = edges_count.get(k, 0) + 1
        non_manifold = sum(1 for c in edges_count.values() if c != 2)
        metrics["non_manifold_edges"] = float(non_manifold)

    # NOTE: Geometry and kinematic metrics (chamfer_mm, f_score_*,
    # foot_sliding_cm, etc.) are ONLY populated when a real ReconPipeline
    # roundtrip or kinematic analysis has been performed upstream.
    # Hardcoded fake constants (0.0 / 0.5 / 1.0) have been removed to
    # prevent false-pass gate evaluation.  The applicable_keys set in the
    # gate profile must be narrowed to the actually-measured subset when
    # only video-stage metrics are available.

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


def _try_load_exr_sequence(fgpp_dir: Path, channel_prefix: str) -> np.ndarray | None:
    """Try loading a channel sequence from fgpp directory.

    Looks for numbered .npy files matching ``<channel_prefix>_*.npy``
    or ``<channel_prefix>/frame_*.npy`` patterns and stacks them into
    an array. Returns ``None`` if no matching files found.
    """
    if not fgpp_dir.is_dir():
        return None

    # Pattern 1: channel_prefix_0000.npy, channel_prefix_0001.npy, ...
    npy_files = sorted(fgpp_dir.glob(f"{channel_prefix}_*.npy"))
    if not npy_files:
        # Pattern 2: channel_prefix/frame_0000.npy, ...
        sub_dir = fgpp_dir / channel_prefix
        if sub_dir.is_dir():
            npy_files = sorted(sub_dir.glob("*.npy"))

    if not npy_files:
        return None

    frames: list[np.ndarray] = []
    for npy_path in npy_files:
        try:
            data = np.load(str(npy_path))
            frames.append(np.asarray(data))
        except Exception:
            continue

    if frames:
        return np.stack(frames, axis=0)
    return None
