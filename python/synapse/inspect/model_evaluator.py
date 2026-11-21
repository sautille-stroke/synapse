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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

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
# Gate profiles (dynamically derived from synapse.metrics.profiles)
# -------------------------------------------------------------------
from synapse.metrics.profiles import (
    _LOWER,
    _METRIC_DIRECTIONS,
    RSI_PROMOTION,
    RSI_PROMOTION_TARGET,
    TRACK_A_ASSET,
    TRACK_B_VIDEO,
)
from synapse.metrics.roundtrip import (
    compute_chamfer_distance_mm,
    sample_mesh_surface,
)
from synapse.render_core.mesh import MeshData, load_obj


def _build_gate_profiles() -> dict[str, dict[str, tuple[float, str]]]:
    """Dynamically derive 3D evaluation gate profiles from canonical profile SSOT."""
    raw_profiles = {
        "TRACK_A_ASSET": TRACK_A_ASSET,
        "TRACK_B_VIDEO": TRACK_B_VIDEO,
        "RSI_PROMOTION": RSI_PROMOTION,
        "RSI_PROMOTION_TARGET": RSI_PROMOTION_TARGET,
    }
    profiles: dict[str, dict[str, tuple[float, str]]] = {}
    for name, p_dict in raw_profiles.items():
        gate: dict[str, tuple[float, str]] = {
            "is_manifold": (1.0, "eq"),
        }
        if name != "TRACK_B_VIDEO":
            gate["is_watertight"] = (1.0, "eq")
            gate["num_degenerate_faces"] = (0.0, "eq")
            gate["self_intersecting_pairs"] = (0.0, "eq")
        # Full 16-item profiles.py SSOT gate alignment
        for metric_name, thresh in p_dict.items():
            dir_str = _METRIC_DIRECTIONS.get(metric_name, _LOWER)
            op = "le" if dir_str == _LOWER else "ge"
            gate[metric_name] = (float(thresh), op)
        profiles[name] = gate
    return profiles


GATE_PROFILES: dict[str, dict[str, tuple[float, str]]] = _build_gate_profiles()

MODEL_EXTENSIONS: frozenset[str] = frozenset({".obj", ".glb", ".ply", ".gltf", ".fbx", ".stl"})

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
            f"Unknown 3D evaluation gate profile: {profile_name!r}. Valid profiles: {valid}.",
            agent_fix_hint=(f"Pass a valid '--profile' / '--gate-profile' value ({valid})."),
        )
    return key


def encode_png_rgb(image: np.ndarray) -> bytes:
    """Encode an ``[H, W, 3]`` uint8 array as PNG bytes using stdlib ``zlib``."""
    h, w = image.shape[:2]

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        raw = chunk_type + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

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
    required_keys: set[str] | list[str] | tuple[str, ...] | None = None,
    require_reference: bool = True,
) -> dict[str, bool]:
    """Evaluate gate thresholds from *profile_name* against *metrics* (fail-closed)."""
    canonical = normalize_eval_profile(profile_name)
    profile = GATE_PROFILES[canonical]
    results: dict[str, bool] = {}

    base_required = {"is_manifold"}
    if require_reference:
        base_required.add("chamfer_mm")
    if canonical != "TRACK_B_VIDEO":
        base_required.update({"is_watertight", "num_degenerate_faces", "self_intersecting_pairs"})
    if required_keys is not None:
        base_required.update(required_keys)

    for metric_name, (threshold, op) in profile.items():
        value = metrics.get(metric_name)
        if value is None:
            # Fail-closed: missing metric in base_required cannot pass gate
            if metric_name in base_required:
                if metric_name == "chamfer_mm":
                    results["reference_required_not_evaluated"] = False
                results[metric_name] = False
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
    from synapse.render_core.camera import CameraTrajectoryType, generate_cameras
    from synapse.render_core.rasterizer import SoftwareRasterizer
    from synapse.style.aces_color import linear_to_srgb

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
        rad_f = np.asarray(frame.direct_radiance, dtype=np.float32)
        rgb_srgb = np.clip(linear_to_srgb(rad_f), 0.0, 1.0)
        rgb_u8 = (rgb_srgb * 255.0).astype(np.uint8)
        rgb_frames.append(rgb_u8)
        png_bytes = encode_png_rgb(rgb_u8)
        screenshots.append(base64.b64encode(png_bytes).decode("ascii"))

    stacked = (
        np.stack(rgb_frames, axis=0)
        if rgb_frames
        else np.empty((0, resolution, resolution, 3), dtype=np.uint8)
    )
    return stacked, screenshots, labels[:num_views]


def _collect_model_files(directory: Path) -> list[Path]:
    """Collect all supported 3D model files recursively in directory."""
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS
    )


# Tier-0/2 Mesh loader registry avoiding upward Tier 3 imports
_DEFAULT_MESH_LOADER: Callable[[str], Any] | None = None


def register_default_mesh_loader(loader: Callable[[str], Any]) -> None:
    """Register custom or procedural mesh loader from higher tiers (e.g. bench)."""
    global _DEFAULT_MESH_LOADER
    _DEFAULT_MESH_LOADER = loader


def _default_load_mesh(p: str, *, normalize: bool = False) -> Any:
    """Load mesh without upward Tier 3 dependencies.

    Args:
        p: Path to mesh file.
        normalize: If True, normalize mesh to unit bounding sphere.
            Default False to preserve physical world (metric) scale.
    """
    if _DEFAULT_MESH_LOADER is not None:
        return _DEFAULT_MESH_LOADER(p)
    path = Path(p)
    if path.suffix.lower() == ".obj" and path.exists():
        mesh = load_obj(path)
        return mesh.normalize() if normalize else mesh
    try:
        import trimesh

        tm: Any = trimesh.load(p, force="mesh")
        mesh = MeshData(
            name=path.stem,
            vertices=np.asarray(tm.vertices, dtype=np.float32),
            faces=np.asarray(tm.faces, dtype=np.int32),
            normals=np.asarray(tm.vertex_normals, dtype=np.float32),
        )
        return mesh.normalize() if normalize else mesh
    except Exception as exc:
        raise ValueError(f"Cannot load mesh '{p}': {exc}") from exc


def evaluate_single_model(
    model_path: Path,
    profile_name: str,
    resolution: int = 256,
    num_views: int = 8,
    reference_mesh: Any | None = None,
    model_name: str | None = None,
    log_fn: Callable[[str], None] | None = None,
    geometry_backend: Literal["native", "python"] = "native",
    mesh_loader_fn: Callable[[str], Any] | None = None,
    require_reference: bool = True,
) -> ModelEvalResult:
    """Evaluate a single 3D model file against *profile_name*.

    Args:
        mesh_loader_fn: Optional callable ``(path_str) -> MeshData``.
            If not provided, uses registered default or native OBJ loader.
    """
    if mesh_loader_fn is None:
        mesh_loader_fn = _default_load_mesh

    from synapse.metrics.image_quality import ciede2000, lpips, psnr_db, ssim
    from synapse.metrics.roundtrip import (
        f_score_1mm,
    )

    canonical_profile = normalize_eval_profile(profile_name)
    effective_name = model_name if model_name is not None else model_path.name

    if log_fn:
        log_fn(f"  Loading {effective_name}...")

    try:
        mesh = mesh_loader_fn(str(model_path))
    except Exception as exc:
        return ModelEvalResult(
            model_name=effective_name,
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
        backend=geometry_backend,
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

    floating_island_faces = 0
    if geo_res.num_connected_components > 1:
        if (
            hasattr(geo_res, "floating_island_face_indices")
            and geo_res.floating_island_face_indices
        ):
            floating_island_faces = len(geo_res.floating_island_face_indices)
        else:
            floating_island_faces = max(
                0, int(mesh.num_faces - (mesh.num_faces // geo_res.num_connected_components))
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
        "floating_island_faces": float(floating_island_faces),
    }

    if log_fn:
        log_fn(f"  Rendering {num_views} views at {resolution}x{resolution}...")
    try:
        v_center = verts.mean(axis=0)
        v_centered = verts - v_center
        max_r = float(np.max(np.linalg.norm(v_centered, axis=-1)))
        v_render = v_centered / max_r if max_r > 1e-8 else v_centered
        pred_rgb, screenshots, labels = render_orbit_views(
            v_render,
            faces_arr,
            norms,
            resolution,
            num_views,
        )
    except Exception as exc:
        if log_fn:
            log_fn(f"  Warning: failed to render orbit views for {effective_name}: {exc}")
        pred_rgb = np.empty((0, resolution, resolution, 3), dtype=np.uint8)
        screenshots, labels = [], []

    if reference_mesh is not None:
        if log_fn:
            log_fn("  Computing reference Chamfer / F-score / multi-view metrics...")
        # Area-weighted triangle surface sampling for vertex-density independence
        if hasattr(mesh, "faces") and len(mesh.faces) > 0:
            src_pts = sample_mesh_surface(verts, faces_arr, num_samples=10000, seed=42)
        else:
            src_pts = subsample_point_cloud(verts, max_points=10000, seed=42)

        if hasattr(reference_mesh, "faces") and len(reference_mesh.faces) > 0:
            tgt_pts = sample_mesh_surface(
                reference_mesh.vertices, reference_mesh.faces, num_samples=10000, seed=42
            )
        else:
            tgt_pts = subsample_point_cloud(reference_mesh.vertices, max_points=10000, seed=42)
        # Compute in world scale (not normalized screenshot space)
        metrics["chamfer_mm"] = compute_chamfer_distance_mm(src_pts, tgt_pts)
        metrics["f_score_1mm"] = f_score_1mm(src_pts, tgt_pts)

        if len(pred_rgb) > 0:
            try:
                ref_v_center = reference_mesh.vertices.mean(axis=0)
                ref_v_centered = reference_mesh.vertices - ref_v_center
                ref_max_r = float(np.max(np.linalg.norm(ref_v_centered, axis=-1)))
                ref_v_render = (
                    ref_v_centered / ref_max_r if ref_max_r > 1e-8 else ref_v_centered
                )
                ref_rgb, _, _ = render_orbit_views(
                    ref_v_render,
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
            except Exception as exc:
                if log_fn:
                    log_fn(
                        f"  Warning: failed to compute multi-view comparison for {effective_name}: {exc}"
                    )
    else:
        # No-reference mode: restrict applicable keys to self-topology metrics only.
        # Do NOT fail-closed on missing chamfer_mm when no reference was provided.
        if log_fn:
            log_fn("  No reference mesh provided; restricting to self-topology metrics only.")

    req_ref = require_reference if reference_mesh is None else True
    gate_results = evaluate_profile_gates(
        metrics,
        canonical_profile,
        require_reference=req_ref,
    )
    overall_passed = bool(gate_results) and all(gate_results.values())

    return ModelEvalResult(
        model_name=effective_name,
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
    geometry_backend: Literal["native", "python"] = "native",
    require_reference: bool | None = None,
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

    model_files = _collect_model_files(input_dir)
    if not model_files:
        raise AssetNotFoundError(
            f"No model files found in {input_dir}",
            agent_fix_hint="Ensure --input-dir contains .obj/.glb/.ply files.",
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-load mesh loader for reference meshes (Tier 2→3 bypass)
    _ref_loader_fn: Callable[[str], Any] | None = None

    reference_meshes: dict[str, Any] = {}
    if reference_dir is not None:
        ref_dir = Path(reference_dir)
        if not ref_dir.exists() or not ref_dir.is_dir():
            raise AssetNotFoundError(
                f"Reference directory does not exist: {reference_dir}",
                agent_fix_hint="Provide a valid --reference-dir directory path, or omit it for no-reference mode.",
            )
        if ref_dir.exists() and ref_dir.is_dir():
            _ref_loader_fn = _default_load_mesh

            for ref_path in _collect_model_files(ref_dir):
                try:
                    loaded = _ref_loader_fn(str(ref_path))
                    rel_key = ref_path.relative_to(ref_dir).with_suffix("").as_posix()
                    reference_meshes[rel_key] = loaded
                    if ref_path.stem not in reference_meshes:
                        reference_meshes[ref_path.stem] = loaded
                except Exception as exc:
                    if log_fn:
                        log_fn(f"  Warning: failed to load reference mesh {ref_path.name}: {exc}")

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    results: list[ModelEvalResult] = []

    if log_fn:
        log_fn(f"Evaluating {len(model_files)} models (profile: {canonical_profile})...")

    req_ref = (reference_dir is not None) if require_reference is None else require_reference
    for i, model_path in enumerate(model_files, 1):
        rel_posix = model_path.relative_to(input_dir).as_posix()
        rel_stem = model_path.relative_to(input_dir).with_suffix("").as_posix()
        if log_fn:
            log_fn(f"[{i}/{len(model_files)}] {rel_posix}")
        ref_mesh = (
            reference_meshes.get(rel_posix)
            or reference_meshes.get(rel_stem)
            or reference_meshes.get(model_path.name)
            or reference_meshes.get(model_path.stem)
        )
        if reference_dir is not None and ref_mesh is None and log_fn:
            log_fn(
                f"  Warning: no matching reference mesh found for {rel_posix} in {reference_dir}"
            )
        res = evaluate_single_model(
            model_path=model_path,
            profile_name=canonical_profile,
            resolution=resolution,
            num_views=num_views,
            reference_mesh=ref_mesh,
            model_name=rel_stem,
            log_fn=log_fn,
            geometry_backend=geometry_backend,
            require_reference=req_ref,
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
                    "num_connected_components": (r.topology.num_connected_components),
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
        log_fn(f"Done: {n_pass} passed, {n_fail} failed (exit code {exit_code})")

    return json_data


class ModelEvaluator:
    """Batch 3D model evaluation orchestrator."""

    def __init__(
        self,
        profile_name: str = "TRACK_A_ASSET",
        resolution: int = 256,
        num_views: int = 8,
        geometry_backend: Literal["native", "python"] = "native",
    ) -> None:
        self.profile_name = profile_name
        self.resolution = resolution
        self.num_views = num_views
        self.geometry_backend = geometry_backend

    def evaluate_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        reference_dir: Path | None = None,
        log_fn: Callable[[str], None] | None = None,
        require_reference: bool | None = None,
    ) -> dict[str, Any]:
        """Evaluate all models in directory."""
        return evaluate_3d_models_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            reference_dir=reference_dir,
            profile_name=self.profile_name,
            resolution=self.resolution,
            num_views=self.num_views,
            log_fn=log_fn,
            geometry_backend=self.geometry_backend,
            require_reference=require_reference,
        )

    def evaluate_model(
        self,
        model_path: Path,
        reference_mesh: Any | None = None,
        model_name: str | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> ModelEvalResult:
        """Evaluate a single 3D model."""
        return evaluate_single_model(
            model_path=model_path,
            profile_name=self.profile_name,
            resolution=self.resolution,
            num_views=self.num_views,
            reference_mesh=reference_mesh,
            model_name=model_name,
            log_fn=log_fn,
            geometry_backend=self.geometry_backend,
        )
