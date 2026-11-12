#!/usr/bin/env python3
"""Evaluate 3D models: topology checks, multi-view screenshots, metrics.

Produces a self-contained HTML report with per-model metrics tables,
pass/fail gate badges, multi-view screenshot galleries, and SVG radar
charts.  Also writes a machine-readable JSON results file.

Usage::

    python scripts/evaluate_3d_models.py \\
        --input-dir ./models/ \\
        --output-dir ./eval_output/ \\
        [--reference-dir ./reference_models/] \\
        [--profile TRACK_A_ASSET] \\
        [--resolution 256] \\
        [--num-views 8] \\
        [--json]

Exit codes (per AGENTS.md §3)::

    0  — Success (all models evaluated, report generated)
    10 — Validation error (bad arguments)
    20 — Input directory or files missing
    40 — Quality gate failures detected (report still generated)
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# -------------------------------------------------------------------
# Gate profiles
# -------------------------------------------------------------------

GATE_PROFILES: dict[str, dict[str, tuple[float, str]]] = {
    "TRACK_A_ASSET": {
        "is_manifold": (1.0, "eq"),
        "is_watertight": (1.0, "eq"),
        "num_degenerate_faces": (0.0, "eq"),
        "chamfer_mm": (5.0, "lt"),
    },
    "TRACK_B_VIDEO": {
        "is_manifold": (1.0, "eq"),
        "chamfer_mm": (10.0, "lt"),
    },
    "RSI_PROMOTION": {
        "is_manifold": (1.0, "eq"),
        "is_watertight": (1.0, "eq"),
        "num_degenerate_faces": (0.0, "eq"),
        "chamfer_mm": (3.0, "lt"),
    },
}

_VIEW_LABELS_8 = [
    "Front", "Front-Right", "Right", "Back-Right",
    "Back", "Back-Left", "Left", "Front-Left",
]

_VIEW_LABELS_6 = [
    "Front", "Front-Right", "Right",
    "Back", "Back-Left", "Left",
]


# -------------------------------------------------------------------
# Pure Python PNG encoder (no PIL / opencv dependency)
# -------------------------------------------------------------------


def _encode_png_rgb(image: np.ndarray) -> bytes:
    """Encode an ``[H, W, 3]`` uint8 array as PNG bytes.

    Minimal PNG encoder using ``zlib`` for DEFLATE compression.
    No external image library required.
    """
    h, w = image.shape[:2]

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        raw = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + raw
            + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)

    raw_data = b""
    for y in range(h):
        raw_data += b"\x00" + image[y].tobytes()
    compressed = zlib.compress(raw_data)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return png


# -------------------------------------------------------------------
# Geometry analysis helpers
# -------------------------------------------------------------------


def _count_connected_components(faces: np.ndarray) -> int:
    """Count connected components via union-find on vertex adjacency."""
    if len(faces) == 0:
        return 0
    max_v = int(faces.max()) + 1
    parent = list(range(max_v))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for face in faces:
        for i in range(len(face) - 1):
            union(int(face[i]), int(face[i + 1]))

    used_verts = set(int(v) for v in faces.flatten())
    roots = {find(v) for v in used_verts}
    return len(roots)


def _count_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold: float = 1e-10,
) -> int:
    """Count faces with near-zero area."""
    if len(faces) == 0:
        return 0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5
    return int(np.sum(areas < threshold))


def _count_boundary_edges(faces: np.ndarray) -> int:
    """Count edges shared by only one face (boundary edges)."""
    edge_count: Counter[tuple[int, int]] = Counter()
    for face in faces:
        n = len(face)
        for i in range(n):
            u, w = int(face[i]), int(face[(i + 1) % n])
            edge_count[(min(u, w), max(u, w))] += 1
    return sum(1 for c in edge_count.values() if c == 1)


def _chamfer_distance_mm(
    src: np.ndarray,
    tgt: np.ndarray,
) -> float:
    """Compute symmetric Chamfer distance in mm.

    Assumes vertex positions are in metre-scale.  Sub-samples to
    10 000 points for performance.
    """
    rng = np.random.default_rng(42)
    if len(src) > 10000:
        idx = rng.choice(len(src), 10000, replace=False)
        src = src[idx]
    if len(tgt) > 10000:
        idx = rng.choice(len(tgt), 10000, replace=False)
        tgt = tgt[idx]

    # src -> tgt nearest
    diff_st = src[:, None, :] - tgt[None, :, :]
    d_st = np.min(np.sum(diff_st ** 2, axis=-1), axis=1)

    # tgt -> src nearest
    diff_ts = tgt[:, None, :] - src[None, :, :]
    d_ts = np.min(np.sum(diff_ts ** 2, axis=-1), axis=1)

    chamfer = (float(np.mean(d_st)) + float(np.mean(d_ts))) / 2.0
    return float(np.sqrt(chamfer) * 1000.0)


# -------------------------------------------------------------------
# Gate evaluation
# -------------------------------------------------------------------


def _evaluate_gates(
    metrics: dict[str, float],
    profile_name: str,
) -> dict[str, bool]:
    """Evaluate gate thresholds from a profile against metric values."""
    profile = GATE_PROFILES.get(profile_name, {})
    results: dict[str, bool] = {}
    for metric_name, (threshold, op) in profile.items():
        value = metrics.get(metric_name)
        if value is None:
            continue
        if op == "eq":
            results[metric_name] = abs(value - threshold) < 1e-9
        elif op == "lt":
            results[metric_name] = value < threshold
        elif op == "gt":
            results[metric_name] = value > threshold
        else:
            results[metric_name] = True
    return results


# -------------------------------------------------------------------
# Multi-view rendering
# -------------------------------------------------------------------


def _render_screenshots(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    resolution: int,
    num_views: int,
) -> tuple[list[str], list[str]]:
    """Render multi-view screenshots and return base64 PNGs + labels.

    Returns
    -------
    tuple[list[str], list[str]]
        (base64_pngs, view_labels)
    """
    from synapse.bench.camera import CameraTrajectoryType, generate_cameras
    from synapse.bench.rasterizer import SoftwareRasterizer
    from synapse.fgpp.frame import CameraParams

    cameras = generate_cameras(
        CameraTrajectoryType.ORBIT,
        num_frames=num_views,
        radius=2.5,
        elevation_deg=15.0,
        width=resolution,
        height=resolution,
    )

    rasterizer = SoftwareRasterizer(
        width=resolution,
        height=resolution,
    )

    if num_views == 8:
        labels = list(_VIEW_LABELS_8)
    elif num_views == 6:
        labels = list(_VIEW_LABELS_6)
    else:
        labels = [f"View {i + 1}" for i in range(num_views)]

    screenshots: list[str] = []
    for cam in cameras:
        frame = rasterizer.render(
            vertices=vertices.astype(np.float64),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float64),
            camera=cam,
            entity_name="eval_model",
        )
        # Extract RGB from direct_radiance (float16 -> uint8)
        rgb = np.asarray(frame.direct_radiance, dtype=np.float32)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        png_bytes = _encode_png_rgb(rgb)
        screenshots.append(base64.b64encode(png_bytes).decode("ascii"))

    return screenshots, labels[:num_views]


# -------------------------------------------------------------------
# Main evaluation logic
# -------------------------------------------------------------------


def _evaluate_single_model(
    model_path: Path,
    profile_name: str,
    resolution: int,
    num_views: int,
    reference_mesh: Any | None = None,
    log_fn: Any = None,
) -> Any:
    """Evaluate a single model file.

    Returns a ``ModelEvalResult`` instance.
    """
    from synapse.bench.model_loader import load_mesh
    from synapse.inspect.eval_report import (
        ModelEvalResult,
        TopologyResult,
    )
    from synapse.recon.geometry import check_manifold, check_watertight

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
            gate_results={},
            screenshots_base64=[],
            screenshot_labels=[],
            overall_passed=False,
            error=str(exc),
        )

    verts = mesh.vertices
    faces_arr = mesh.faces
    norms = mesh.normals

    # Topology checks
    if log_fn:
        log_fn(f"  Topology analysis...")
    is_manifold = check_manifold(verts, faces_arr)
    is_watertight = check_watertight(verts, faces_arr)
    n_components = _count_connected_components(faces_arr)
    n_degenerate = _count_degenerate_faces(verts, faces_arr)
    n_boundary = _count_boundary_edges(faces_arr)

    topology = TopologyResult(
        is_manifold=is_manifold,
        is_watertight=is_watertight,
        num_vertices=mesh.num_vertices,
        num_faces=mesh.num_faces,
        num_connected_components=n_components,
        has_degenerate_faces=n_degenerate > 0,
        num_boundary_edges=n_boundary,
    )

    # Compute metrics
    metrics: dict[str, float] = {
        "is_manifold": 1.0 if is_manifold else 0.0,
        "is_watertight": 1.0 if is_watertight else 0.0,
        "num_degenerate_faces": float(n_degenerate),
        "vertex_count": float(mesh.num_vertices),
        "face_count": float(mesh.num_faces),
        "connected_components": float(n_components),
        "boundary_edges": float(n_boundary),
    }

    # Chamfer distance if reference available
    if reference_mesh is not None:
        if log_fn:
            log_fn(f"  Computing Chamfer distance...")
        chamfer = _chamfer_distance_mm(verts, reference_mesh.vertices)
        metrics["chamfer_mm"] = chamfer

    # Gate evaluation
    gate_results = _evaluate_gates(metrics, profile_name)
    overall_passed = all(gate_results.values()) if gate_results else True

    # Multi-view screenshots
    if log_fn:
        log_fn(f"  Rendering {num_views} views at {resolution}x{resolution}...")
    try:
        screenshots, labels = _render_screenshots(
            verts, faces_arr, norms, resolution, num_views,
        )
    except Exception:
        screenshots, labels = [], []

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


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------


def main() -> int:
    """CLI entry point for 3D model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate 3D models: topology, metrics, screenshots.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing .obj/.glb/.ply model files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for report and screenshots.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Optional directory with reference meshes for comparison.",
    )
    parser.add_argument(
        "--profile",
        default="TRACK_A_ASSET",
        help="Gate profile name (default: TRACK_A_ASSET).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Screenshot render resolution (default: 256).",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=8,
        help="Number of multi-view screenshots (default: 8).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output pure JSON to stdout.",
    )

    args = parser.parse_args()

    # Validate
    def log_fn(msg: str) -> None:
        if not args.json_output:
            print(msg, file=sys.stderr if args.json_output else sys.stdout)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        err = {
            "status": "error",
            "exit_code": 20,
            "error_type": "AssetNotFoundError",
            "message": f"Input directory does not exist: {input_dir}",
            "agent_fix_hint": "Provide a valid --input-dir path.",
        }
        if args.json_output:
            print(json.dumps(err, indent=2))
        else:
            print(f"ERROR: {err['message']}", file=sys.stderr)
        return 20

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover model files
    model_extensions = {".obj", ".glb", ".ply", ".gltf", ".fbx", ".stl"}
    model_files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in model_extensions
    )

    if not model_files:
        err = {
            "status": "error",
            "exit_code": 20,
            "error_type": "AssetNotFoundError",
            "message": f"No model files found in {input_dir}",
            "agent_fix_hint": (
                "Ensure --input-dir contains .obj/.glb/.ply files."
            ),
        }
        if args.json_output:
            print(json.dumps(err, indent=2))
        else:
            print(f"ERROR: {err['message']}", file=sys.stderr)
        return 20

    # Load reference meshes if provided
    reference_meshes: dict[str, Any] = {}
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        if ref_dir.exists():
            from synapse.bench.model_loader import load_mesh

            for ref_path in ref_dir.iterdir():
                if ref_path.suffix.lower() in model_extensions:
                    try:
                        reference_meshes[ref_path.stem] = load_mesh(
                            str(ref_path), normalize=True,
                        )
                    except Exception:
                        pass

    # Evaluate each model
    from synapse.inspect.eval_report import (
        EvalReportData,
        generate_eval_html_report,
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    results: list[Any] = []

    log_fn(
        f"Evaluating {len(model_files)} models "
        f"(profile: {args.profile})..."
    )

    for i, model_path in enumerate(model_files, 1):
        log_fn(f"[{i}/{len(model_files)}] {model_path.name}")
        ref_mesh = reference_meshes.get(model_path.stem)
        result = _evaluate_single_model(
            model_path=model_path,
            profile_name=args.profile,
            resolution=args.resolution,
            num_views=args.num_views,
            reference_mesh=ref_mesh,
            log_fn=log_fn if not args.json_output else None,
        )
        results.append(result)

    # Generate HTML report
    report_data = EvalReportData(
        timestamp=timestamp,
        profile_name=args.profile,
        models=results,
        reference_dir=args.reference_dir,
    )
    report_path = output_dir / "eval_report.html"
    generate_eval_html_report(report_data, report_path)
    log_fn(f"HTML report: {report_path}")

    # Write JSON results
    json_path = output_dir / "eval_results.json"
    n_pass = sum(1 for r in results if r.overall_passed)
    n_fail = len(results) - n_pass

    json_data: dict[str, Any] = {
        "status": "success",
        "exit_code": 0 if n_fail == 0 else 40,
        "timestamp": timestamp,
        "profile": args.profile,
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
                    "has_degenerate_faces": (
                        r.topology.has_degenerate_faces
                    ),
                    "num_boundary_edges": r.topology.num_boundary_edges,
                },
                "metrics": r.metrics,
                "gate_results": r.gate_results,
                "error": r.error,
            }
            for r in results
        ],
        "report_path": str(report_path),
        "agent_fix_hint": (
            "Review failed models and improve mesh quality."
            if n_fail > 0
            else ""
        ),
    }

    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_fn(f"JSON results: {json_path}")

    if args.json_output:
        print(json.dumps(json_data, indent=2, ensure_ascii=False))

    exit_code = 0 if n_fail == 0 else 40
    log_fn(
        f"Done: {n_pass} passed, {n_fail} failed "
        f"(exit code {exit_code})"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
