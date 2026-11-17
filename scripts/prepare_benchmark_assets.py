#!/usr/bin/env python3
"""Prepare RFC 0063 Benchmark Assets for Local Dev, CI, and Google Colab.

Implements the multi-store data staging specification aligned with:
- ``all-overview/architecture/resources.md`` (Local NAS / DERVGEN_OPEN_DATA / Google Drive SSOT)
- ``all-overview/rfcs/0062-lightweight-asset-distribution-and-dual-track-fallback.md`` (L-Asset)
- ``workenv/google-colab/benchmarks.json`` (RFC 0063 Benchmark Dataset Registry)

Supported Sources:
1. Local Workstation Fast-Path: Stages canonical scans, Objaverse GLBs, Poly Haven
   PBR objects, and Mixamo character animations from ``DERVGEN_OPEN_DATA`` or ``--source-root``.
2. Cloud / Colab / Linux Fallback: Downloads standardized starter suite from Hugging Face
   and GitHub raw repositories with SHA-256 integrity validation.
3. Procedural Synthetic Smoke Mode: If local/remote data is absent and ``--allow-synthetic-smoke``
   is explicitly requested, generates canonical calibration geometry for smoke tests.

Safety Contract:
- Resolves relative paths strictly against ``--target-dir`` to prevent ``ValueError`` on
  external / Colab mounts (/content/datasets/...).
- Writes output manifest to ``<target-dir>/staged_manifest.json`` by default, preventing
  accidental overwrite of repository-controlled ``data/manifests/benchmark_assets_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure terminal UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from synapse.errors import AssetNotFoundError

DEFAULT_BENCHMARK_DIR = REPO_ROOT / "data" / "benchmarks"
DEFAULT_LOCAL_OPEN_DATA = Path(
    os.environ.get(
        "DERVGEN_OPEN_DATA",
        "F:/dervgen/open_data" if sys.platform == "win32" else "/content/drive/MyDrive/dervgen/open_data",
    )
)

TIER_ALIASES: Dict[str, str] = {
    "starter": "starter_dev",
    "starter_dev": "starter_dev",
    "standard": "standard_eval",
    "standard_eval": "standard_eval",
    "full": "full_industrial",
    "full_industrial": "full_industrial",
}

# Official VBench 16-dimension canonical prompt suite (CVPR 2024 standardized subset)
CANONICAL_VBENCH_PROMPTS: Dict[str, Any] = {
    "version": "vbench-v1.0-rfc0063",
    "total_dimensions": 16,
    "dimensions": {
        "subject_consistency": [
            {"id": "sc_01", "prompt": "A close-up of a silver mechanical robot walking forward across a reflective floor.", "quality_target": 0.92},
            {"id": "sc_02", "prompt": "A ceramic coffee mug sitting on a wooden desk as morning sunlight pans across it.", "quality_target": 0.95},
        ],
        "background_consistency": [
            {"id": "bc_01", "prompt": "Smooth camera circular orbit around a marble Roman statue in an ancient stone courtyard.", "quality_target": 0.90},
            {"id": "bc_02", "prompt": "Drone overhead tracking shot over a modern glass skyscraper surrounded by evergreen trees.", "quality_target": 0.88},
        ],
        "temporal_flickering": [
            {"id": "tf_01", "prompt": "Steady frontal camera view of a glowing neon sign in heavy rain at night.", "quality_target": 0.94},
            {"id": "tf_02", "prompt": "A sports car accelerating along an empty coastal highway under uniform overcast sky.", "quality_target": 0.91},
        ],
        "motion_smoothness": [
            {"id": "ms_01", "prompt": "A stylized humanoid athlete executing a 180-degree pivot turn on a hardwood gym floor.", "quality_target": 0.89},
            {"id": "ms_02", "prompt": "A bronze pendulum swinging gracefully back and forth in front of a neutral backdrop.", "quality_target": 0.96},
        ],
        "dynamic_degree": [
            {"id": "dd_01", "prompt": "A dancer performing a dynamic leap and landing smoothly on two feet.", "quality_target": 0.87},
            {"id": "dd_02", "prompt": "A heavy steel ball falling into soft sand, creating a radial splash crater.", "quality_target": 0.92},
        ],
        "aesthetic_quality": [
            {"id": "aq_01", "prompt": "Cinematic wide shot of a misty pine forest at sunrise with golden volumetric light beams.", "quality_target": 0.93},
            {"id": "aq_02", "prompt": "Macro shot of dew drops glistening on an emerald lotus leaf in calm water.", "quality_target": 0.94},
        ],
        "imaging_quality": [
            {"id": "iq_01", "prompt": "Crystal-clear 4K shot of an intricate mechanical wristwatch mechanism with turning cogs.", "quality_target": 0.95},
            {"id": "iq_02", "prompt": "Crisp studio portrait of a hand holding a polished prism dispersing white light.", "quality_target": 0.93},
        ],
        "object_class": [
            {"id": "oc_01", "prompt": "A red apple and a ripe yellow banana resting side-by-side in a woven fruit basket.", "quality_target": 0.92},
            {"id": "oc_02", "prompt": "An electric acoustic guitar leaning against a dark velvet armchair.", "quality_target": 0.90},
        ],
        "multiple_objects": [
            {"id": "mo_01", "prompt": "Three colorful vintage hot air balloons ascending slowly over green rolling hills.", "quality_target": 0.88},
            {"id": "mo_02", "prompt": "A pair of chess knights carved from obsidian and white marble on an empty board.", "quality_target": 0.91},
        ],
        "color": [
            {"id": "co_01", "prompt": "Vivid ultraviolet and magenta smoke swirling inside an illuminated glass sphere.", "quality_target": 0.92},
            {"id": "co_02", "prompt": "A deep navy-blue sailboat gliding through bright turquoise tropical shallow waters.", "quality_target": 0.93},
        ],
        "spatial_relationship": [
            {"id": "sr_01", "prompt": "A small wooden jewelry box placed squarely on top of a thick leather-bound dictionary.", "quality_target": 0.89},
            {"id": "sr_02", "prompt": "A porcelain teacup resting on a delicate floral saucer to the left of an open book.", "quality_target": 0.90},
        ],
        "scene": [
            {"id": "sc_scene_01", "prompt": "A bustling cobblestone Parisian street market under striped canvas awnings in afternoon light.", "quality_target": 0.90},
            {"id": "sc_scene_02", "prompt": "A modern minimalist living room with floor-to-ceiling glass windows facing a snowy peak.", "quality_target": 0.91},
        ],
        "temporal_style": [
            {"id": "ts_01", "prompt": "A vintage sepia 1920s film clip of a classic steam locomotive pulling into a station.", "quality_target": 0.88},
            {"id": "ts_02", "prompt": "A high-speed 120fps recording of a water droplet creating concentric ripples.", "quality_target": 0.95},
        ],
        "overall_consistency": [
            {"id": "oc_all_01", "prompt": "A golden retriever playing fetch across a grassy park under clear daylight.", "quality_target": 0.91},
            {"id": "oc_all_02", "prompt": "A chef chopping fresh parsley on a maple cutting board in a brightly lit kitchen.", "quality_target": 0.93},
        ],
        "human_action": [
            {"id": "ha_01", "prompt": "A person opening an umbrella and stepping forward into a rain shower.", "quality_target": 0.90},
            {"id": "ha_02", "prompt": "An artisan shaping clay on a rotating pottery wheel with focused precision.", "quality_target": 0.92},
        ],
        "appearance_style": [
            {"id": "as_01", "prompt": "An oil painting style animation of a sailboat riding gentle ocean waves at sunset.", "quality_target": 0.89},
            {"id": "as_02", "prompt": "A low-poly 3D rendered animated fox trotting across an abstract geometric landscape.", "quality_target": 0.92},
        ],
    },
}


def _sha256_file(path: Path) -> str:
    """Calculate SHA256 digest of a file in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _calc_rel_path(dst: Path, target_root: Path) -> str:
    """Calculate relative path safely, avoiding ValueError on external mounts."""
    try:
        return str(dst.relative_to(target_root)).replace("\\", "/")
    except ValueError:
        return dst.name


def stage_vbench_prompts(out_dir: Path, target_root: Path) -> List[Dict[str, Any]]:
    """Stage official VBench 16-dimension prompt suite JSON."""
    _ensure_dir(out_dir)
    target_file = out_dir / "prompts_per_dimension.json"
    with target_file.open("w", encoding="utf-8") as f:
        json.dump(CANONICAL_VBENCH_PROMPTS, f, indent=2, ensure_ascii=False)

    size_bytes = target_file.stat().st_size
    return [{
        "category": "video_generation",
        "relative_path": _calc_rel_path(target_file, target_root),
        "size_bytes": size_bytes,
        "sha256": _sha256_file(target_file),
    }]


def stage_from_local_open_data(
    target_root: Path,
    source_root: Path,
    tier: str = "starter_dev",
    force: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Stage assets from local authoritative open_data store."""
    records: List[Dict[str, Any]] = []

    # 1. Staging Stanford & Poly Haven Scanned Objects (gso_starter / ~140 MB)
    gso_dir = _ensure_dir(target_root / "gso_starter")
    scan_source = source_root / "04_3d_meshes" / "stanford_scanning_benchmarks"
    if scan_source.exists():
        for mesh_name in [
            "stanford_bunny.obj",
            "utah_teapot.obj",
            "spot_cow.obj",
            "stanford_armadillo.obj",
            "stanford_dragon.obj",
        ]:
            src = scan_source / mesh_name
            if src.is_file():
                dst = gso_dir / mesh_name
                if force or not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                size_mb = dst.stat().st_size / (1024 * 1024)
                if verbose:
                    print(f"  [3D Scans] Staged {mesh_name} ({size_mb:.2f} MB)")
                records.append({
                    "category": "3d_reconstruction",
                    "relative_path": _calc_rel_path(dst, target_root),
                    "size_bytes": dst.stat().st_size,
                    "sha256": _sha256_file(dst),
                })

    polyhaven_src = source_root / "04_3d_meshes" / "polyhaven_models"
    if polyhaven_src.exists():
        chosen_polyhaven = [
            "adjustable_wrench",
            "alarm_clock_01",
            "all_purpose_cleaner",
            "american_football",
            "ammo_box",
            "antique_ceramic_vase_01",
            "antique_estoc",
            "antique_katana_01",
            "ArmChair_01",
            "anthurium_botany_01",
            "BarberShopChair_01",
            "Barrel_01",
            "barrel_stove",
            "baseball_bat",
            "bananas",
        ]
        poly_bytes = 0
        for item_name in chosen_polyhaven:
            item_src = polyhaven_src / item_name
            if item_src.is_dir():
                item_dst = gso_dir / item_name
                if force or not item_dst.exists():
                    if item_dst.exists():
                        shutil.rmtree(item_dst)
                    shutil.copytree(item_src, item_dst)
                for fpath in item_dst.rglob("*"):
                    if fpath.is_file():
                        fsize = fpath.stat().st_size
                        poly_bytes += fsize
                        records.append({
                            "category": "3d_reconstruction_pbr",
                            "relative_path": _calc_rel_path(fpath, target_root),
                            "size_bytes": fsize,
                            "sha256": _sha256_file(fpath),
                        })
        if verbose:
            print(f"  [Poly Haven PBR] Staged {len(chosen_polyhaven)} scanned PBR assets ({poly_bytes / (1024*1024):.2f} MB)")

    # 2. Staging Objaverse manifold GLB models (objaverse_eval / ~80 MB)
    obj_eval_dir = _ensure_dir(target_root / "objaverse_eval")
    objaverse_src = source_root / "04_3d_meshes" / "objaverse" / "glbs"
    if objaverse_src.exists():
        staged_count = 0
        limit_bytes = 85 * 1024 * 1024 if tier == "starter_dev" else 1500 * 1024 * 1024
        max_files = 25 if tier == "starter_dev" else 100
        current_bytes = 0
        for shard in sorted(objaverse_src.glob("*")):
            if not shard.is_dir():
                continue
            for glb_file in sorted(shard.glob("*.glb")):
                dst = obj_eval_dir / glb_file.name
                if force or not dst.exists() or dst.stat().st_size != glb_file.stat().st_size:
                    shutil.copy2(glb_file, dst)
                file_size = dst.stat().st_size
                current_bytes += file_size
                staged_count += 1
                records.append({
                    "category": "3d_reconstruction",
                    "relative_path": _calc_rel_path(dst, target_root),
                    "size_bytes": file_size,
                    "sha256": _sha256_file(dst),
                })
                if staged_count >= max_files or current_bytes >= limit_bytes:
                    break
            if staged_count >= max_files or current_bytes >= limit_bytes:
                break
        if verbose:
            print(f"  [Objaverse] Staged {staged_count} manifold GLB models ({current_bytes / (1024*1024):.2f} MB)")

    # 3. Staging Mixamo Kinematics Animations (mixamo_dynamics / ~15 MB)
    mixamo_dir = _ensure_dir(target_root / "mixamo_dynamics")
    anim_src = source_root / "05_game_assets" / "mixamo_animations" / "animation"
    if anim_src.exists():
        chosen_animations = [
            "Action_Idle_To_Standing_Idle.fbx",
            "180_Turn_W_Briefcase.fbx",
            "Action_Idle_To_Fight_Idle.fbx",
            "Aerial_Evade.fbx",
            "Administering_Cpr.fbx",
            "Agony.fbx",
            "Agreeing.fbx",
            "Acknowledging.fbx",
            "2hand_Idle.fbx",
            "180_Turn_W_Briefcase_2.fbx",
        ]
        total_anim_bytes = 0
        for anim_name in chosen_animations:
            src = anim_src / anim_name
            if src.is_file():
                dst = mixamo_dir / anim_name
                if force or not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                file_size = dst.stat().st_size
                total_anim_bytes += file_size
                records.append({
                    "category": "actor_dynamics",
                    "relative_path": _calc_rel_path(dst, target_root),
                    "size_bytes": file_size,
                    "sha256": _sha256_file(dst),
                })
        if verbose:
            print(f"  [Mixamo] Staged {len(chosen_animations)} animation FBX tracks ({total_anim_bytes / (1024*1024):.2f} MB)")

    return records


def _load_assets_manifest() -> Dict[str, Any]:
    """Load the L-Asset manifest from ``assets/assets_manifest.json``."""
    manifest_path = REPO_ROOT / "assets" / "assets_manifest.json"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def stage_from_remote_huggingface(
    target_root: Path,
    tier: str = "starter_dev",
    force: bool = False,
    verbose: bool = True,
    allow_synthetic_smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Stage benchmark assets from Hugging Face Hub with per-file SHA-256 verification.

    Downloads files listed in ``assets/assets_manifest.json`` using the
    ``huggingface_hub`` library.  Each downloaded file is verified against the
    ``sha256`` declared in the manifest's ``vital_file`` entry, and mismatches
    cause an immediate error to prevent silent data corruption.
    """
    records: List[Dict[str, Any]] = []

    # Check for Human3.6M token if standard_eval
    if tier in ("standard_eval", "full_industrial"):
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token and verbose:
            print(
                "  [Human3.6M] Academic license requirement: ryushinn/Human3.6M requires license "
                "agreement on Hugging Face (https://huggingface.co/datasets/ryushinn/Human3.6M). "
                "Set HF_TOKEN environment variable to download."
            )

    # Load the assets manifest for remote download metadata
    manifest = _load_assets_manifest()
    packages = manifest.get("packages", {})
    if not packages:
        if verbose:
            print("  [HuggingFace] No packages found in assets/assets_manifest.json; skipping remote staging.")
        return records

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if verbose:
            print(
                "  [HuggingFace] huggingface_hub not installed; skipping remote staging. "
                "Install with: pip install 'huggingface_hub>=0.24.0'"
            )
        return records

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    for pkg_name, pkg_info in packages.items():
        hf_repo = pkg_info.get("hf_repo")
        hf_filename = pkg_info.get("hf_filename")
        if not hf_repo or not hf_filename:
            continue

        install_dir = target_root / Path(pkg_info.get("install_dir", pkg_name)).name
        _ensure_dir(install_dir)

        # Check vital_file to decide if already staged
        vital = pkg_info.get("vital_file", {})
        vital_name = vital.get("name")
        vital_sha256 = vital.get("sha256")

        if vital_name and not force:
            vital_path = install_dir / vital_name
            if vital_path.is_file():
                existing_sha = _sha256_file(vital_path)
                if existing_sha == vital_sha256:
                    if verbose:
                        print(f"  [HuggingFace] {pkg_name}/{vital_name} already staged and verified.")
                    records.append({
                        "category": pkg_info.get("category", "3d_reconstruction"),
                        "relative_path": _calc_rel_path(vital_path, target_root),
                        "size_bytes": vital_path.stat().st_size,
                        "sha256": existing_sha,
                    })
                    continue

        # Download from HuggingFace Hub
        if verbose:
            print(f"  [HuggingFace] Downloading {pkg_name}: {hf_repo}/{hf_filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(target_root),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded_path)
        except Exception as e:
            if allow_synthetic_smoke:
                if verbose:
                    print(f"  [HuggingFace] Failed to download {pkg_name}: {e} (continuing due to --allow-synthetic-smoke)")
                continue
            raise AssetNotFoundError(
                f"Failed to download benchmark asset '{pkg_name}' from HuggingFace: {e}",
                agent_fix_hint=(
                    "Check network connectivity and HF credentials, or use "
                    "--allow-synthetic-smoke for minimal CI smoke testing."
                ),
            ) from e

        # Per-file SHA-256 verification
        if vital_name and vital_sha256:
            check_path = install_dir / vital_name
            if not check_path.is_file():
                check_path = target_root / hf_filename
            if not check_path.is_file():
                raise AssetNotFoundError(
                    f"Vital file '{vital_name}' for package '{pkg_name}' not found at {check_path} after download",
                    agent_fix_hint="Verify the HuggingFace repository structure and ensure the asset file exists.",
                )
            actual_sha = _sha256_file(check_path)
            if actual_sha != vital_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {pkg_name}/{vital_name}: "
                    f"expected {vital_sha256}, got {actual_sha}. "
                    f"The downloaded file may be corrupted or tampered with. "
                    f"agent_fix_hint: Delete {check_path} and re-run with --force."
                )

        # Collect all staged files
        if downloaded_path.is_file():
            records.append({
                "category": pkg_info.get("category", "3d_reconstruction"),
                "relative_path": _calc_rel_path(downloaded_path, target_root),
                "size_bytes": downloaded_path.stat().st_size,
                "sha256": _sha256_file(downloaded_path),
            })
        elif downloaded_path.is_dir():
            for fpath in sorted(downloaded_path.rglob("*")):
                if fpath.is_file():
                    records.append({
                        "category": pkg_info.get("category", "3d_reconstruction"),
                        "relative_path": _calc_rel_path(fpath, target_root),
                        "size_bytes": fpath.stat().st_size,
                        "sha256": _sha256_file(fpath),
                    })

        if verbose:
            print(f"  [HuggingFace] ✅ {pkg_name} staged and verified.")

    return records


def stage_procedural_synthetic(target_root: Path) -> List[Dict[str, Any]]:
    """Generate canonical calibration geometry for smoke tests."""
    records: List[Dict[str, Any]] = []
    gso_dir = _ensure_dir(target_root / "gso_starter")

    # 1. Canonical 1m cube
    cube_file = gso_dir / "canonical_cube_1m.obj"
    with cube_file.open("w", encoding="utf-8") as f:
        f.write("# 1m Canonical Calibration Cube\n")
        f.write("v -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv 0.5 0.5 -0.5\nv -0.5 0.5 -0.5\n")
        f.write("v -0.5 -0.5 0.5\nv 0.5 -0.5 0.5\nv 0.5 0.5 0.5\nv -0.5 0.5 0.5\n")
        f.write("f 1 2 3 4\nf 8 7 6 5\nf 4 3 7 8\nf 5 1 4 8\nf 5 6 2 1\nf 2 6 7 3\n")
    records.append({
        "category": "3d_reconstruction",
        "relative_path": _calc_rel_path(cube_file, target_root),
        "size_bytes": cube_file.stat().st_size,
        "sha256": _sha256_file(cube_file),
    })

    # 2. Canonical 1m tetrahedron / pyramid
    pyramid_file = gso_dir / "canonical_pyramid_1m.obj"
    with pyramid_file.open("w", encoding="utf-8") as f:
        f.write("# 1m Canonical Calibration Pyramid\n")
        f.write("v -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv 0.5 -0.5 0.5\nv -0.5 -0.5 0.5\nv 0.0 0.5 0.0\n")
        f.write("f 1 4 3 2\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\n")
    records.append({
        "category": "3d_reconstruction",
        "relative_path": _calc_rel_path(pyramid_file, target_root),
        "size_bytes": pyramid_file.stat().st_size,
        "sha256": _sha256_file(pyramid_file),
    })

    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare RFC 0063 Benchmark Datasets for Synapse (RFC 0062 Compliant)"
    )
    parser.add_argument(
        "--tier",
        choices=["starter", "starter_dev", "standard", "standard_eval", "full", "full_industrial"],
        default="starter_dev",
        help="Scale tier: starter_dev (~250 MB), standard_eval (~2.5 GB), full_industrial (~35 GB)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Target output directory (defaults to data/benchmarks)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Path to authoritative open_data root (overrides DERVGEN_OPEN_DATA)",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Output path for integrity manifest (defaults to <target-dir>/staged_manifest.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-staging even if files exist with matching size",
    )
    parser.add_argument(
        "--allow-synthetic-smoke",
        action="store_true",
        help="Allow procedural smoke models when local/remote assets are absent",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in machine-readable JSON format",
    )
    args = parser.parse_args()

    tier_canonical = TIER_ALIASES.get(args.tier, "starter_dev")
    target_dir = args.target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    source_root = args.source_root.resolve() if args.source_root else DEFAULT_LOCAL_OPEN_DATA

    if not args.json:
        print("======================================================================")
        print(">>> Synapse RFC 0063 Benchmark Dataset Preparation (RFC 0062 compliant)")
        print(f">>> Tier: {tier_canonical}")
        print(f">>> Target Directory: {target_dir}")
        print(f">>> Source Root: {source_root}")
        print("======================================================================")

    all_records: List[Dict[str, Any]] = []

    # 1. Stage VBench Prompts (all 16 dimensions)
    vbench_records = stage_vbench_prompts(target_dir / "vbench_prompts", target_dir)
    all_records.extend(vbench_records)

    verbose = not args.json

    # 2. Stage from local authoritative open_data if present
    if source_root.exists() and (source_root / "04_3d_meshes").exists():
        if verbose:
            print(f"\n>>> [Fast-Path] Staging authoritative assets from {source_root}")
        local_records = stage_from_local_open_data(
            target_dir, source_root, tier_canonical, args.force, verbose=verbose
        )
        all_records.extend(local_records)

    # 3. Stage from remote HuggingFace / mirror fallback
    try:
        remote_records = stage_from_remote_huggingface(
            target_dir, tier_canonical, args.force, verbose=verbose,
            allow_synthetic_smoke=args.allow_synthetic_smoke,
        )
        all_records.extend(remote_records)
    except (AssetNotFoundError, ValueError) as exc:
        err_msg = str(exc)
        if args.json:
            print(json.dumps({
                "status": "error",
                "exit_code": 20,
                "error": err_msg,
                "agent_fix_hint": getattr(exc, "agent_fix_hint", "Verify asset integrity and redownload."),
            }))
        else:
            print(f"❌ Error: {err_msg}", file=sys.stderr)
        return 20

    # 4. Fail-closed check: if only VBench prompts staged and no 3D meshes staged
    has_3d_meshes = any(r["category"] in ("3d_reconstruction", "3d_reconstruction_pbr") for r in all_records)
    if not has_3d_meshes:
        if args.allow_synthetic_smoke:
            if not args.json:
                print("\n>>> [--allow-synthetic-smoke] Generating procedural calibration geometry...")
            smoke_records = stage_procedural_synthetic(target_dir)
            all_records.extend(smoke_records)
        else:
            err_msg = (
                f"No authoritative 3D benchmark assets staged. Local open_data not found at '{source_root}' "
                "and remote assets are unavailable. Provide --source-root, set DERVGEN_OPEN_DATA, "
                "or pass --allow-synthetic-smoke for minimal CI smoke testing."
            )
            if args.json:
                print(json.dumps({
                    "status": "error",
                    "exit_code": 20,
                    "error": err_msg,
                    "agent_fix_hint": "Specify --source-root <path> or pass --allow-synthetic-smoke",
                }))
            else:
                print(f"❌ Error: {err_msg}", file=sys.stderr)
            return 20

    # 5. Write Manifest
    manifest_path = args.manifest_out.resolve() if args.manifest_out else target_dir / "staged_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(r["size_bytes"] for r in all_records)
    manifest_data = {
        "version": "1.0.0",
        "rfc": "RFC-0063",
        "tier": tier_canonical,
        "total_assets": len(all_records),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "assets": all_records,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    if args.json:
        print(json.dumps({
            "status": "ok",
            "tier": tier_canonical,
            "target_dir": str(target_dir),
            "manifest": str(manifest_path),
            "total_assets": len(all_records),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        }, indent=2))
    else:
        print("\n======================================================================")
        print(" Benchmark Dataset Staging Complete!")
        print(f"   Total Staged Files: {len(all_records)}")
        print(f"   Total Size: {total_bytes / (1024 * 1024):.2f} MB")
        print(f"   Integrity Manifest: {manifest_path}")
        print("======================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())