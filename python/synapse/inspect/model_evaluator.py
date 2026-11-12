"""Batch 3D model evaluation orchestrator (topology + multi-view + HTML/JSON).

Shared core library backing both ``scripts/evaluate_3d_models.py`` and the
``synapse inspect evaluate`` CLI sub-command.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from synapse.errors import (
    AssetNotFoundError,
    ContractValidationError,
)
from synapse.inspect.eval_report import (
    EvalReportData,
    ModelEvalResult,
    TopologyResult,
    generate_eval_html_report,
)
from synapse.inspect.geometry_gate import validate_mesh_geometry

# -------------------------------------------------------------------
# Gate profiles (aligned with synapse.metrics.profiles)
# -------------------------------------------------------------------

GATE_PROFILES: dict[str, dict[str, tuple[float, str]]] = {
    "TRACK_A_ASSET": {
        "is_manifold": (1.0, "eq"),
        "is_watertight": (1.0, "eq"),
        "num_degenerate_faces": (0.0, "eq"),
        "self_intersecting_pairs": (0.0, "eq"),
        "chamfer_mm": (5.0, "le"),
    },
    "TRACK_B_VIDEO": {
        "is_manifold": (1.0, "eq"),
        "chamfer_mm": (8.0, "le"),
    },
    "RSI_PROMOTION": {
        "is_manifold": (1.0, "eq"),
        "is_watertight": (1.0, "eq"),
        "num_degenerate_faces": (0.0, "eq"),
        "self_intersecting_pairs": (0.0, "eq"),
        "chamfer_mm": (4.0, "le"),
    },
    "RSI_PROMOTION_TARGET": {
        "is_manifold": (1.0, "eq"),
        "is_watertight": (1.0, "eq"),
        "num_degenerate_faces": (0.0, "eq"),
        "self_intersecting_pairs": (0.0, "eq"),
        "chamfer_mm": (1.0, "le"),
    },
}

MODEL_EXTENSIONS: frozenset[str] = frozenset(
    {".obj", ".glb", ".ply", ".gltf", ".fbx", ".stl"}
)

_VIEW_LABELS_8 = [
    "Front",
    "Front-Right",
    "Right",
    "Back-Right",
    "Back",
    "Back-Left",
    "Left",
    "Front-Left",
]

_VIEW_LABELS_6 = [
    "Front",
    "Front-Right",
    "Right",
    "Back",
    "Back-Left",
    "Left",
]


def normalize_eval_profile(profile_name: str) -> str:
    """Resolve case-insensitive profile name or raise ContractValidationError (exit 10)."""
    key = profile_name.strip().upper()
    if key not in GATE_PROFILES:
        valid = ", ".join(sorted(GATE_PROFILES.keys()))
        raise ContractValidationError(
            f"Unknown 3D evaluation gate profile: {profile_name!r}. "
            f"Valid profiles: {valid}.",
            agent_fix_hint=(
                f"Pass a valid '--profile' / '--gate-profile' value ({valid})."
            ),
        )
    return key


def encode_png_rgb(image: np.ndarray) -> bytes:
    """Encode an ``[H, W, 3]`` uint8 array as PNG bytes using stdlib ``zlib``."""
    h, w = image.shape[:2]

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        raw = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + raw
            + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    rows = [b"\x00" + image[y].tobytes() for y in range(h)]
    compressed = zlib.compress(b"".join(rows))

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return png


def count_boundary_edges(faces: np.ndarray) -> int:
    """Count undirected edges incident to strictly one polygon face."""
    edge_count: Counter[tuple[int, int]] = Counter()
    for face in faces:
        n = len(face)
        for i in range(n):
            u, w = int(face[i]), int(face[(i + 1) % n])
            edge_count[(min(u, w), max(u, w))] += 1
    return sum(1 for c in edge_count.values() if c == 1)


def subsample_point_cloud(
    points: np.ndarray,
    max_points: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Deterministically subsample a point cloud to at most *max_points*."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), max_points, replace=False)
    return pts[idx]


def evaluate_profile_gates(
    metrics: dict[str, float],
    profile_name: str,
) -> dict[str, bool]:
    """Evaluate gate thresholds from *profile_name* against *metrics* (fail-closed)."""
    canonical = normalize_eval_profile(profile_name)
    profile = GATE_PROFILES[canonical]
    results: dict[str, bool] = {}
    for metric_name, (threshold, op) in profile.items():
        value = metrics.get(metric_name)
        if value is None:
            continue
        if not np.isfinite(value):
            results[metric_name] = False
        elif op == "eq":
            results[metric_name] = abs(value - threshold) < 1e-9
        elif op == "lt":
            results[metric_name] = value < threshold
        elif op == "le":
            results[metric_name] = value <= threshold
        elif op == "gt":
            results[metric_name] = value > threshold
        elif op == "ge":
            results[metric_name] = value >= threshold
        else:
            raise ValueError(f"Unsupported gate operator {op!r}.")
    return results


def render_orbit_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    resolution: int,
    num_views: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Render orbit multi-view RGB images, base64 PNGs, and view labels."""
    from synapse.bench.camera import CameraTrajectoryType, generate_cameras
    from synapse.bench.rasterizer import SoftwareRasterizer

    cameras = generate_cameras(
        CameraTrajectoryType.ORBIT,
        num_frames=num_views,
        radius=2.5,
        elevation_deg=15.0,
        width=resolution,
        height=resolution,
    )
    rasterizer = SoftwareRasterizer(width=resolution, height=resolution)

    if num_views == 8:
        labels = list(_VIEW_LABELS_8)
    elif num_views == 6:
        labels = list(_VIEW_LABELS_6)
    else:
        labels = [f"View {i + 1}" for i in range(num_views)]

    rgb_frames: list[np.ndarray] = []
    screenshots: list[str] = []
    for cam in cameras:
        frame = rasterizer.render(
            vertices=vertices.astype(np.float64),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float64),
            camera=cam,
            entity_name="eval_model",
        )
        rgb_f = np.clip(
            np.asarray(frame.direct_radiance, dtype=np.float32), 0.0, 1.0
        )
        rgb_u8 = (rgb_f * 255.0).astype(np.uint8)
        rgb_frames.append(rgb_u8)
        png_bytes = encode_png_rgb(rgb_u8)
        screenshots.append(base64.b64encode(png_bytes).decode("ascii"))

    stacked = (
        np.stack(rgb_frames, axis=0)
        if rgb_frames
        else np.empty((0, resolution, resolution, 3), dtype=np.uint8)
    )
    return stacked, screenshots, labels[:num_views]


def evaluate_single_model(
    model_path: Path,
    profile_name: str,
    resolution: int = 256,
    num_views: int = 8,
    reference_mesh: Any | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> ModelEvalResult:
    """Evaluate a single 3D model file against *profile_name*."""
    from synapse.bench.model_loader import load_mesh
    from synapse.metrics.image_quality import ciede2000, lpips, psnr_db, ssim
    from synapse.metrics.roundtrip import chamfer_mm, f_score_1mm

    canonical_profile = normalize_eval_profile(profile_name)

    if log_fn:
        log_fn(f"  Loading {model_path.name}...")

    try:
        mesh = load_mesh(str(model_path), normalize=True)
    except Exception as exc:
        return ModelEvalResult(
            model_name=model_path.name,
            model_path=str(model_path),
            topology=TopologyResult(
                is_manifold=False,
                is_watertight=False,
                num_vertices=0,
                num_faces=0,
                num_connected_components=0,
                has_degenerate_faces=False,
                num_boundary_edges=0,
            ),
            metrics={},
            gate_results={"load_mesh": False},
            screenshots_base64=[],
            screenshot_labels=[],
            overall_passed=False,
            error=str(exc),
        )

    verts = mesh.vertices
    faces_arr = mesh.faces
    norms = mesh.normals

    if log_fn:
        log_fn("  Running Tier-0 geometry gate...")
    require_wt = canonical_profile != "TRACK_B_VIDEO"
    geo_res = validate_mesh_geometry(
        verts,
        faces_arr,
        require_watertight=require_wt,
    )
    n_degenerate = len(geo_res.degenerate_face_indices)
    n_self_intersections = len(geo_res.self_intersecting_pairs)
    n_boundary = count_boundary_edges(faces_arr)

    topology = TopologyResult(
        is_manifold=geo_res.is_manifold,
        is_watertight=geo_res.is_watertight,
        num_vertices=mesh.num_vertices,
        num_faces=mesh.num_faces,
        num_connected_components=geo_res.num_connected_components,
        has_degenerate_faces=n_degenerate > 0,
        num_boundary_edges=n_boundary,
    )

    metrics: dict[str, float] = {
        "is_manifold": 1.0 if geo_res.is_manifold else 0.0,
        "is_watertight": 1.0 if geo_res.is_watertight else 0.0,
        "num_degenerate_faces": float(n_degenerate),
        "self_intersecting_pairs": float(n_self_intersections),
        "vertex_count": float(mesh.num_vertices),
        "face_count": float(mesh.num_faces),
        "connected_components": float(geo_res.num_connected_components),
        "boundary_edges": float(n_boundary),
    }

    if log_fn:
        log_fn(
            f"  Rendering {num_views} views at {resolution}x{resolution}..."
        )
    try:
        pred_rgb, screenshots, labels = render_orbit_views(
            verts, faces_arr, norms, resolution, num_views,
        )
    except Exception:
        pred_rgb = np.empty((0, resolution, resolution, 3), dtype=np.uint8)
        screenshots, labels = [], []

    if reference_mesh is not None:
        if log_fn:
            log_fn("  Computing reference Chamfer / F-score / multi-view metrics...")
        src_pts = subsample_point_cloud(verts, max_points=10000, seed=42)
        tgt_pts = subsample_point_cloud(
            reference_mesh.vertices, max_points=10000, seed=42
        )
        metrics["chamfer_mm"] = chamfer_mm(src_pts, tgt_pts)
        metrics["f_score_1mm"] = f_score_1mm(src_pts, tgt_pts)

        if len(pred_rgb) > 0:
            try:
                ref_rgb, _, _ = render_orbit_views(
                    reference_mesh.vertices,
                    reference_mesh.faces,
                    reference_mesh.normals,
                    resolution,
                    num_views,
                )
                if ref_rgb.shape == pred_rgb.shape:
                    metrics["psnr_db"] = psnr_db(pred_rgb, ref_rgb)
                    metrics["ssim"] = ssim(pred_rgb, ref_rgb)
                    metrics["ciede2000"] = ciede2000(pred_rgb, ref_rgb)
                    metrics["lpips"] = lpips(pred_rgb, ref_rgb)
            except Exception:
                pass

    gate_results = evaluate_profile_gates(metrics, canonical_profile)
    overall_passed = bool(gate_results) and all(gate_results.values())

    return ModelEvalResult(
        model_name=model_path.name,
        model_path=str(model_path),
        topology=topology,
        metrics=metrics,
        gate_results=gate_results,
        screenshots_base64=screenshots,
        screenshot_labels=labels,
        overall_passed=overall_passed,
    )


def evaluate_3d_models_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    reference_dir: Path | None = None,
    profile_name: str = "TRACK_A_ASSET",
    resolution: int = 256,
    num_views: int = 8,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate all 3D models in *input_dir* and generate HTML + JSON reports."""
    canonical_profile = normalize_eval_profile(profile_name)

    if resolution <= 0:
        raise ContractValidationError(
            f"Invalid --resolution {resolution}; must be a positive integer.",
            agent_fix_hint="Pass a positive integer such as '--resolution 256'.",
        )
    if num_views <= 0:
        raise ContractValidationError(
            f"Invalid --num-views {num_views}; must be a positive integer.",
            agent_fix_hint="Pass a positive integer such as '--num-views 8'.",
        )

    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise AssetNotFoundError(
            f"Input directory does not exist: {input_dir}",
            agent_fix_hint="Provide a valid --input-dir directory path.",
        )

    model_files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS
    )
    if not model_files:
        raise AssetNotFoundError(
            f"No model files found in {input_dir}",
            agent_fix_hint="Ensure --input-dir contains .obj/.glb/.ply files.",
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_meshes: dict[str, Any] = {}
    if reference_dir is not None:
        ref_dir = Path(reference_dir)
        if ref_dir.exists() and ref_dir.is_dir():
            from synapse.bench.model_loader import load_mesh

            for ref_path in sorted(ref_dir.iterdir()):
                if ref_path.is_file() and ref_path.suffix.lower() in MODEL_EXTENSIONS:
                    try:
                        reference_meshes[ref_path.stem] = load_mesh(
                            str(ref_path), normalize=True,
                        )
                    except Exception:
                        pass

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    results: list[ModelEvalResult] = []

    if log_fn:
        log_fn(
            f"Evaluating {len(model_files)} models "
            f"(profile: {canonical_profile})..."
        )

    for i, model_path in enumerate(model_files, 1):
        if log_fn:
            log_fn(f"[{i}/{len(model_files)}] {model_path.name}")
        ref_mesh = reference_meshes.get(model_path.stem)
        res = evaluate_single_model(
            model_path=model_path,
            profile_name=canonical_profile,
            resolution=resolution,
            num_views=num_views,
            reference_mesh=ref_mesh,
            log_fn=log_fn,
        )
        results.append(res)

    report_data = EvalReportData(
        timestamp=timestamp,
        profile_name=canonical_profile,
        models=results,
        reference_dir=str(reference_dir) if reference_dir is not None else None,
    )
    report_path = output_dir / "eval_report.html"
    generate_eval_html_report(report_data, report_path)
    if log_fn:
        log_fn(f"HTML report: {report_path}")

    json_path = output_dir / "eval_results.json"
    n_pass = sum(1 for r in results if r.overall_passed)
    n_fail = len(results) - n_pass
    exit_code = 0 if n_fail == 0 else 40

    json_data: dict[str, Any] = {
        "status": "success" if n_fail == 0 else "gate_failed",
        "exit_code": exit_code,
        "timestamp": timestamp,
        "profile": canonical_profile,
        "summary": {
            "total": len(results),
            "passed": n_pass,
            "failed": n_fail,
        },
        "models": [
            {
                "name": r.model_name,
                "path": r.model_path,
                "passed": r.overall_passed,
                "topology": {
                    "is_manifold": r.topology.is_manifold,
                    "is_watertight": r.topology.is_watertight,
                    "num_vertices": r.topology.num_vertices,
                    "num_faces": r.topology.num_faces,
                    "num_connected_components": (
                        r.topology.num_connected_components
                    ),
                    "has_degenerate_faces": r.topology.has_degenerate_faces,
                    "num_boundary_edges": r.topology.num_boundary_edges,
                },
                "metrics": r.metrics,
                "gate_results": r.gate_results,
                "error": r.error,
            }
            for r in results
        ],
        "report_path": str(report_path),
        "results_json_path": str(json_path),
        "agent_fix_hint": (
            "Review failed models in eval_report.html and resolve topology or "
            "Chamfer quality gate violations."
            if n_fail > 0
            else ""
        ),
    }

    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if log_fn:
        log_fn(f"JSON results: {json_path}")
        log_fn(
            f"Done: {n_pass} passed, {n_fail} failed (exit code {exit_code})"
        )

    return json_data
