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
from dataclasses import asdict, dataclass, field
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
        d: dict[str, Any] = {
            "scenario_name": self.scenario_name,
            "adapter_id": self.adapter_id,
            "metrics": self.metrics,
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

        mesh = load_mesh(scenario.asset.id, normalize=True)
        rasterizer = SoftwareRasterizer(width=width, height=height)

        fgpp_frames: list[FGPPFrame] = []
        for i, cam in enumerate(cameras):
            frame = rasterizer.render(
                vertices=mesh.vertices,
                faces=mesh.faces,
                normals=mesh.normals,
                camera=cam,
                base_color=mesh.base_color,
                entity_name=scenario.asset.id,
                prev_camera=cameras[i - 1] if i > 0 else None,
            )
            fgpp_frames.append(frame)

        fgpp_sequence = FGPPSequence(
            frames=fgpp_frames,
            fps=scenario.camera_params.get("fps", 24),
            scene_id=scenario.id,
        )
        report.timings["fgpp_load_ms"] = _elapsed_ms(t0)

        # --- Phase 3: Prepare & Generate via adapter -------------------
        t0 = time.perf_counter()
        config = VideoGenerationConfig(
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=scenario.camera_params.get("fps", 24.0),
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
        metrics = _compute_metrics(fgpp_frames, video_result)
        report.metrics = metrics
        report.timings["metrics_ms"] = _elapsed_ms(t0)

        # --- Phase 5: Gate evaluation ----------------------------------
        if self._gate_profile is not None:
            from synapse.metrics.profiles import evaluate_metrics_gate

            report.gate_verdict = evaluate_metrics_gate(
                metrics, self._gate_profile,
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
            frames/    -- generated video frames (PNG or .npy)
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
        ``metrics/comparison.json`` into *bundle_dir*.
    """
    bundle_dir = Path(bundle_dir)
    frames_dir = bundle_dir / "frames"
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

    if gt_depth is not None and gen_frames is not None:
        from synapse.metrics.fgpp_fidelity import depth_rmse_m

        # Use average luminance as proxy depth from generated frames
        if gen_frames.ndim == 4:
            gen_lum = np.mean(
                gen_frames.astype(np.float32) / 255.0, axis=-1,
            )
            # Use only the first frame for depth comparison
            if gen_lum.shape[0] > 0 and gt_depth.ndim >= 2:
                gt_d = gt_depth
                if gt_depth.ndim == 3:
                    gt_d = gt_depth[0]
                pred_d = gen_lum[0]
                if gt_d.shape == pred_d.shape:
                    metrics["depth_rmse_m"] = depth_rmse_m(
                        pred_d, gt_d,
                    )

    if gt_normals is not None and gen_frames is not None:
        from synapse.metrics.fgpp_fidelity import normal_angular_error_deg

        # Use normalized RGB as proxy normals
        if gen_frames.ndim == 4 and gen_frames.shape[0] > 0:
            pred_n = (
                gen_frames[0].astype(np.float32) / 127.5 - 1.0
            )
            gt_n = gt_normals
            if gt_normals.ndim == 4:
                gt_n = gt_normals[0]
            if pred_n.shape == gt_n.shape:
                metrics["normal_angular_error_deg"] = (
                    normal_angular_error_deg(pred_n, gt_n)
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
    from synapse.metrics.profiles import evaluate_metrics_gate

    gate = evaluate_metrics_gate(metrics, profile)
    metrics["gate_passed"] = float(gate.passed)

    # --- Write comparison.json ----------------------------------------
    metrics_dir = bundle_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = metrics_dir / "comparison.json"
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
) -> dict[str, float]:
    """Compute all available fidelity metrics.

    Parameters
    ----------
    fgpp_frames:
        List of ``FGPPFrame`` ground truth frames.
    video_result:
        ``VideoResult`` from the adapter.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to measured value.
    """
    from synapse.metrics.fgpp_fidelity import (
        depth_rmse_m,
        normal_angular_error_deg,
        silhouette_iou,
    )
    from synapse.metrics.temporal import flicker_score

    metrics: dict[str, float] = {}
    gen_frames = video_result.frames_srgb_u8  # [T, H, W, 3] uint8

    n_frames = min(len(fgpp_frames), gen_frames.shape[0])
    if n_frames == 0:
        return metrics

    # --- Temporal: flicker score (needs float32 [0,1]) ----------------
    gen_float = gen_frames[:n_frames].astype(np.float32) / 255.0
    metrics["flicker_score"] = flicker_score(gen_float)

    # --- Per-frame FGPP fidelity metrics (average over frames) --------
    depth_errors: list[float] = []
    normal_errors: list[float] = []
    iou_scores: list[float] = []

    for i in range(n_frames):
        gt = fgpp_frames[i]
        gen_f = gen_float[i]  # [H, W, 3]

        # Depth RMSE: compare GT depth with luminance proxy from gen
        if gt.depth is not None:
            gt_depth = gt.depth.astype(np.float32)
            pred_lum = np.mean(gen_f, axis=-1)  # luminance proxy

            # Resize if shapes differ
            if gt_depth.shape == pred_lum.shape:
                depth_errors.append(depth_rmse_m(pred_lum, gt_depth))

        # Normal angular error
        if gt.world_normal is not None:
            gt_n = gt.world_normal.astype(np.float32)
            # Use gen RGB remapped to [-1,1] as proxy normals
            pred_n = gen_f * 2.0 - 1.0

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

    if depth_errors:
        metrics["depth_rmse_m"] = float(np.mean(depth_errors))
    if normal_errors:
        metrics["normal_angular_error_deg"] = float(
            np.mean(normal_errors),
        )
    if iou_scores:
        metrics["silhouette_iou"] = float(np.mean(iou_scores))

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
        return np.load(str(path))
    return None
