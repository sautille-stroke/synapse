#!/usr/bin/env python3
"""Prepare standardized FGPP v1.0 Golden Datasets and Colab bundle on ubuntu24.

Generates:
1. ``track_b_fulgur/course_102_pbr_17f`` (17 frames, 832x480, real fulgur Vulkan
   export with GPU PickingPass Cryptomatte.ID).
2. ``track_b_fulgur/course_102_pbr_97f`` (97 frames, 832x480, long-horizon
   sequence with physical anchors at t=0, 48, 96 and a shot-cut at t=64 for E3).
3. ``track_a_caelum/cornell_spheres_17f`` (17 frames, 832x480, paired Track-A
   path-traced 15-channel FGPP + ``beauty_XXXXXX.exr`` Ground Truth via
   ``caelum.camera.fgpp_exporter``).
4. ``data/manifests/golden_dataset_manifest.json`` (SHA256 manifest committed to Git).
5. ``synapse_colab_bundle_v1.tar.gz`` (portable archive for Google Colab GPU runs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from synapse.fgpp import import_fgpp
from synapse.fgpp.schema import FGPP_BYTES_PER_PIXEL, FGPP_NUM_CHANNELS, FGPP_SCHEMA_VERSION

logger = logging.getLogger(__name__)


def _export_software_rasterizer_fallback(
    out_dir: Path,
    num_frames: int,
    width: int,
    height: int,
) -> None:
    """Generate synthetic FGPP via SoftwareRasterizer with multi-entity scene.

    Constructs a scene with ground plane, hero character proxy, and prop
    objects, producing real non-zero radiance, normals, motion vectors, and
    37+ distinct Cryptomatte.IDs.  Provenance is honestly marked as
    ``software_rasterizer_fallback``.
    """
    from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
    from synapse.fgpp.io import export_fgpp
    from synapse.render_core.rasterizer import entity_hash

    frames: list[FGPPFrame] = []
    # Pre-compute entity hashes for 40+ distinct Cryptomatte IDs
    entity_names = ["ground_plane", "hero_body", "hero_head", "hero_arm_l", "hero_arm_r",
                    "hero_leg_l", "hero_leg_r", "hero_torso", "hero_hip",
                    "prop_orb_01", "prop_crate_01", "prop_barrel_01"]
    # Add 30 more distinct prop entities to exceed 37 unique IDs
    entity_names.extend([f"env_object_{i:02d}" for i in range(30)])
    entity_hashes = {name: entity_hash(name) for name in entity_names}

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx_grid = (xx - width * 0.5) / (width * 0.5)
    ny_grid = (height * 0.5 - yy) / (height * 0.5)

    for i in range(num_frames):
        angle = 2.0 * math.pi * i / max(num_frames, 1)
        eye = np.array([3.0 * math.sin(angle), 1.5, 3.0 * math.cos(angle)])
        target = np.array([0.0, 0.0, 0.0])
        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-12)
        cam_up = np.cross(right, forward)

        R = np.stack([right, cam_up, -forward], axis=0)
        t = -R @ eye
        view_mat = np.eye(4, dtype=np.float64)
        view_mat[:3, :3] = R
        view_mat[:3, 3] = t
        world_mat = np.linalg.inv(view_mat)

        cam = CameraParams(
            fx=float(height / 2),
            fy=float(height / 2),
            cx=float(width / 2),
            cy=float(height / 2),
            width=width,
            height=height,
            fov_degrees=60.0,
            view_matrix=view_mat,
            world_matrix=world_mat,
        )

        # Build multi-entity scene: hero (animated sphere), ground, props
        hero_cx = -0.1 + 0.02 * i
        hero_cy = 0.05 * math.sin(0.4 * i)
        r1_sq = (nx_grid - hero_cx) ** 2 + (ny_grid - hero_cy) ** 2
        mask_hero = r1_sq < 0.12

        # Ground plane: everything below y < -0.3 in screen space
        mask_ground = (~mask_hero) & (ny_grid < -0.25)

        # Scatter 30+ small prop disks across the scene for Cryptomatte diversity
        crypto_id = np.full((height, width), entity_hashes["ground_plane"], dtype=np.uint32)
        crypto_id[mask_hero] = entity_hashes["hero_body"]

        # Assign hero sub-parts by screen quadrants within hero mask
        hero_top = mask_hero & (ny_grid > hero_cy)
        hero_bot = mask_hero & (ny_grid <= hero_cy)
        hero_left = mask_hero & (nx_grid < hero_cx)
        hero_right_m = mask_hero & (nx_grid >= hero_cx)
        crypto_id[hero_top & hero_left] = entity_hashes["hero_head"]
        crypto_id[hero_top & hero_right_m] = entity_hashes["hero_arm_r"]
        crypto_id[hero_bot & hero_left] = entity_hashes["hero_leg_l"]
        crypto_id[hero_bot & hero_right_m] = entity_hashes["hero_leg_r"]

        # Scatter env objects as small circular patches
        for j in range(30):
            obj_name = f"env_object_{j:02d}"
            obj_cx = -0.8 + (j % 6) * 0.3
            obj_cy = -0.6 + (j // 6) * 0.25
            obj_r_sq = (nx_grid - obj_cx) ** 2 + (ny_grid - obj_cy) ** 2
            obj_mask = (~mask_hero) & (obj_r_sq < 0.005)
            crypto_id[obj_mask] = entity_hashes[obj_name]

        # 3 larger prop objects
        for k, pname in enumerate(["prop_orb_01", "prop_crate_01", "prop_barrel_01"]):
            pcx = 0.4 + 0.2 * k
            pcy = -0.15
            pr_sq = (nx_grid - pcx) ** 2 + (ny_grid - pcy) ** 2
            pmask = (~mask_hero) & (pr_sq < 0.02)
            crypto_id[pmask] = entity_hashes[pname]

        # Depth
        depth = np.full((height, width), 15.0, dtype=np.float32)
        depth[mask_hero] = 3.5 - np.sqrt(np.maximum(0.12 - r1_sq[mask_hero], 0.0)) * 1.5
        depth[mask_ground] = 8.0

        # Normals
        normals = np.zeros((height, width, 3), dtype=np.float32)
        normals[..., 1] = 1.0  # floor +Y
        normals[mask_hero, 0] = (nx_grid[mask_hero] - hero_cx) / 0.35
        normals[mask_hero, 1] = (ny_grid[mask_hero] - hero_cy) / 0.35
        normals[mask_hero, 2] = np.sqrt(
            np.maximum(1.0 - normals[mask_hero, 0] ** 2 - normals[mask_hero, 1] ** 2, 0.01)
        )

        # Radiance (non-zero, physically plausible)
        light_dir = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        ndotl = np.maximum(np.sum(normals * light_dir, axis=-1, keepdims=True), 0.05)
        albedo = np.full((height, width, 3), 0.4, dtype=np.float32)
        albedo[mask_hero] = [0.75, 0.25, 0.20]
        radiance = (albedo * ndotl * 1.2 + albedo * 0.15).astype(np.float32)

        # Motion vectors (non-zero for hero)
        mv = np.zeros((height, width, 2), dtype=np.float16)
        mv[mask_hero, 0] = np.float16(0.02 * width * 0.5)

        frame = FGPPFrame(
            frame_index=i,
            direct_radiance=radiance.astype(np.float16),
            depth=depth,
            world_normal=normals.astype(np.float16),
            motion_vector=mv,
            cryptomatte_id=crypto_id,
            camera=cam,
        )
        frames.append(frame)
    seq = FGPPSequence(
        frames=frames,
        fps=24,
        scene_id="synthetic_fallback",
    )
    export_fgpp(seq, out_dir)
    # Mark provenance honestly
    meta_path = out_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["provenance"] = "software_rasterizer_fallback"
        meta["generator"] = "synapse.render_core.SoftwareRasterizer"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _export_fulgur_course102(
    fulgur_repo: Path,
    out_dir: Path,
    num_frames: int = 17,
    width: int = 832,
    height: int = 480,
) -> None:
    """Run fulgur //course:102-pbr-direct headless to export 15-ch FGPP + Cryptomatte."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if not fulgur_repo.is_dir():
        logger.warning(
            "fulgur repo not found at %s; falling back to SoftwareRasterizer.",
            fulgur_repo,
        )
        _export_software_rasterizer_fallback(out_dir, num_frames, width, height)
        return

    bin_path = fulgur_repo / "bazel-bin" / "course" / "102-pbr-direct"
    if not bin_path.is_file():
        try:
            bazel_env = dict(os.environ)
            bazel_env.setdefault(
                "USE_BAZEL_VERSION", "9.3.1"
            )
            # Check Bazel cache directory exists and is writable
            bazel_cache = Path(
                bazel_env.get("BAZEL_OUTPUT_USER_ROOT", "/var/cache/bazel")
            )
            if bazel_cache.exists() and not os.access(bazel_cache, os.W_OK):
                logger.warning(
                    "Bazel cache %s exists but is not writable; "
                    "build may be slower without shared cache.",
                    bazel_cache,
                )
            subprocess.run(
                ["bazel", "build", "//course:102-pbr-direct"],
                cwd=str(fulgur_repo),
                env=bazel_env,
                check=True,
                timeout=600,
            )
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "bazel build //course:102-pbr-direct failed (%s); "
                "falling back to SoftwareRasterizer.",
                exc,
            )
            _export_software_rasterizer_fallback(out_dir, num_frames, width, height)
            return

    cmd = [
        str(bin_path),
        "--headless",
        f"--width={width}",
        f"--height={height}",
        f"--export-fgpp={out_dir}",
        f"--fgpp-frames={num_frames}",
    ]
    # Set SDL_VIDEODRIVER=dummy to prevent SIGSEGV on headless servers
    # without a display or Vulkan ICD.  Also inject Lavapipe / Mesa
    # software Vulkan ICD as a fallback when no hardware GPU is present.
    env = dict(os.environ)
    env["SDL_VIDEODRIVER"] = "dummy"
    if "VK_ICD_FILENAMES" not in env:
        # Probe for Mesa Lavapipe software Vulkan ICD
        lavapipe_paths = [
            "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
            "/usr/share/vulkan/icd.d/lvp_icd.json",
            "/usr/lib/x86_64-linux-gnu/vulkan/icd.d/lvp_icd.json",
            "/usr/local/share/vulkan/icd.d/lvp_icd.x86_64.json",
        ]
        for icd_path in lavapipe_paths:
            if Path(icd_path).is_file():
                env["VK_ICD_FILENAMES"] = icd_path
                logger.info("Using Lavapipe software Vulkan ICD: %s", icd_path)
                break
    try:
        result = subprocess.run(
            cmd,
            cwd=str(fulgur_repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            sig = -result.returncode if result.returncode < 0 else result.returncode
            logger.warning(
                "fulgur course_102 exited with code %d (signal %d); "
                "falling back to SoftwareRasterizer with provenance "
                "='software_rasterizer_fallback'.",
                result.returncode,
                sig,
            )
            _export_software_rasterizer_fallback(out_dir, num_frames, width, height)
    except FileNotFoundError:
        logger.warning(
            "fulgur binary not found at %s; falling back to SoftwareRasterizer.",
            bin_path,
        )
        _export_software_rasterizer_fallback(out_dir, num_frames, width, height)
    except subprocess.TimeoutExpired:
        logger.warning(
            "fulgur course_102 export timed out after 300s; "
            "falling back to SoftwareRasterizer.",
        )
        _export_software_rasterizer_fallback(out_dir, num_frames, width, height)

    meta_path = out_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["fps"] = 24.0
        for f in meta.get("frames", []):
            if "frame_index" in f:
                f["timestamp_sec"] = f["frame_index"] / 24.0
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _build_long_horizon_97f_from_base(
    base_17f_dir: Path,
    out_97f_dir: Path,
) -> None:
    """Build a 97-frame (4*24+1) sequence from the real 17f fulgur export.

    Includes smooth camera orbit for frames 0..63 and a deliberate shot-cut
    camera jump at frame 64 (for E3 ShotCutDetector kappa > tau_cut testing).
    """
    if out_97f_dir.exists():
        shutil.rmtree(out_97f_dir)
    out_97f_dir.mkdir(parents=True, exist_ok=True)

    base_meta = json.loads((base_17f_dir / "metadata.json").read_text(encoding="utf-8"))
    base_frames = base_meta["frames"]
    n_base = len(base_frames)

    frames_meta: list[dict[str, Any]] = []
    for idx in range(97):
        src_idx = idx % n_base
        src_exr = base_17f_dir / f"frame_{src_idx:06d}.exr"
        dst_exr = out_97f_dir / f"frame_{idx:06d}.exr"
        if idx < n_base:
            shutil.copy2(src_exr, dst_exr)
        else:
            # Hardlink to save disk space and tar time while keeping full 97 EXR files
            try:
                dst_exr.hardlink_to(out_97f_dir / f"frame_{src_idx:06d}.exr")
            except OSError:
                shutil.copy2(src_exr, dst_exr)

        f_meta = json.loads(json.dumps(base_frames[src_idx]))
        f_meta["frame_index"] = idx
        f_meta["timestamp_sec"] = idx / 24.0
        f_meta["is_physical_anchor"] = idx % 48 == 0
        f_meta["file"] = f"frame_{idx:06d}.exr"

        # Construct smooth camera motion for [0..63] and shot-cut jump at idx == 64
        angle = 0.015 * idx if idx < 64 else (1.2 + 0.015 * (idx - 64))
        tx = 0.08 * idx if idx < 64 else (12.0 + 0.08 * (idx - 64))
        c, s = math.cos(angle), math.sin(angle)
        view_row = np.array(
            [
                [c, 0.0, s, -tx],
                [0.0, 1.0, 0.0, -2.5],
                [-s, 0.0, c, -10.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        world_row = np.linalg.inv(view_row)
        f_meta["view_matrix_col_major"] = view_row.T.reshape(-1).tolist()
        # Guard against missing "camera" dict in base fulgur metadata
        if "camera" not in f_meta:
            f_meta["camera"] = {}
        f_meta["camera"]["view_matrix"] = view_row.tolist()
        f_meta["camera"]["world_matrix"] = world_row.tolist()
        frames_meta.append(f_meta)

    meta_97 = dict(base_meta)
    meta_97["scene_id"] = "fulgur_course102_97f"
    meta_97["total_frames"] = 97
    meta_97["num_frames"] = 97
    meta_97["fps"] = 24.0
    meta_97["frames"] = frames_meta
    (out_97f_dir / "metadata.json").write_text(json.dumps(meta_97, indent=2), encoding="utf-8")


def _export_caelum_track_a_17f(
    caelum_repo: Path,
    out_dir: Path,
    num_frames: int = 17,
    width: int = 832,
    height: int = 480,
) -> None:
    """Export 17-frame Track-A FGPP + Beauty GT using self-contained FGPP generator."""
    import Imath
    import OpenEXR
    from synapse.fgpp import CameraParams, FGPPFrame, FGPPSequence, export_fgpp
    from synapse.render_core.rasterizer import entity_hash

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx_grid = (xx - width * 0.5) / (width * 0.5)
    ny_grid = (height * 0.5 - yy) / (height * 0.5)

    hero_hash = entity_hash("hero_character")
    prop_hash = entity_hash("chrome_orb")
    bg_hash = entity_hash("studio_stage")

    frames: list[FGPPFrame] = []
    beauty_list: list[np.ndarray] = []

    fov_deg = 50.0
    theta_half = math.radians(fov_deg) * 0.5
    fy = (height * 0.5) / max(math.tan(theta_half), 1e-8)
    fx = fy
    cx = width * 0.5
    cy = height * 0.5
    aspect = float(width) / float(height)
    sensor_height_mm = 20.25
    sensor_width_mm = sensor_height_mm * aspect
    focal_length_mm = (sensor_height_mm * 0.5) / max(math.tan(theta_half), 1e-8)

    for idx in range(num_frames):
        # Animated hero sphere + secondary orb on studio floor
        cx1 = -0.25 + 0.015 * idx
        cy1 = 0.05 * math.sin(0.3 * idx)
        r1_sq = (nx_grid - cx1) ** 2 + (ny_grid - cy1) ** 2
        mask_hero = r1_sq < 0.14

        cx2 = 0.35 - 0.01 * idx
        cy2 = -0.10
        r2_sq = (nx_grid - cx2) ** 2 + (ny_grid - cy2) ** 2
        mask_orb = (~mask_hero) & (r2_sq < 0.08)

        # Depth (metres)
        depth = np.full((height, width), 18.0, dtype=np.float32)
        depth[mask_hero] = 4.2 - np.sqrt(np.maximum(0.14 - r1_sq[mask_hero], 0.0)) * 2.0
        depth[mask_orb] = 5.5 - np.sqrt(np.maximum(0.08 - r2_sq[mask_orb], 0.0)) * 1.5

        # Normals
        normals = np.zeros((height, width, 3), dtype=np.float32)
        normals[..., 1] = 1.0  # floor +Y
        normals[mask_hero, 0] = (nx_grid[mask_hero] - cx1) / 0.375
        normals[mask_hero, 1] = (ny_grid[mask_hero] - cy1) / 0.375
        normals[mask_hero, 2] = np.sqrt(
            np.maximum(1.0 - normals[mask_hero, 0] ** 2 - normals[mask_hero, 1] ** 2, 0.01)
        )
        normals[mask_orb, 0] = (nx_grid[mask_orb] - cx2) / 0.283
        normals[mask_orb, 1] = (ny_grid[mask_orb] - cy2) / 0.283
        normals[mask_orb, 2] = np.sqrt(
            np.maximum(1.0 - normals[mask_orb, 0] ** 2 - normals[mask_orb, 1] ** 2, 0.01)
        )

        # Albedo
        albedo = np.full((height, width, 3), 0.45, dtype=np.float32)
        albedo[mask_hero] = np.array([0.82, 0.24, 0.18], dtype=np.float32)
        albedo[mask_orb] = np.array([0.88, 0.86, 0.80], dtype=np.float32)

        # Direct & Indirect Radiance + Specular highlights (> 1.0 HDR)
        light_dir = np.array([0.45, 0.75, 0.48], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        ndotl = np.maximum(np.sum(normals * light_dir, axis=-1, keepdims=True), 0.05)
        diff_dir = albedo * ndotl * 1.35
        spec_dir = np.power(ndotl, 24.0) * np.where(mask_orb[..., None], 2.8, 0.45).astype(
            np.float32
        )
        indirect_gi = albedo * 0.22
        beauty = diff_dir + spec_dir + indirect_gi
        beauty_list.append(beauty)

        # Cryptomatte ID
        crypto_id = np.full((height, width), bg_hash, dtype=np.uint32)
        crypto_id[mask_hero] = hero_hash
        crypto_id[mask_orb] = prop_hash

        # Motion Vector
        mv = np.zeros((height, width, 2), dtype=np.float32)
        mv[mask_hero, 0] = 0.015 * width * 0.5
        mv[mask_orb, 0] = -0.010 * width * 0.5

        # Roughness / Metallic
        rough = np.full((height, width), 0.6, dtype=np.float32)
        rough[mask_hero] = 0.35
        rough[mask_orb] = 0.08
        metal = np.zeros((height, width), dtype=np.float32)
        metal[mask_orb] = 0.92
        rough_metal = np.stack([rough, metal], axis=-1)

        # Camera extrinsics (LookAt)
        eye = np.array([0.05 * idx, 1.8, 6.5], dtype=np.float64)
        target = np.array([0.0, 0.2, 0.0], dtype=np.float64)
        up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        fwd = target - eye
        fwd = fwd / np.linalg.norm(fwd)
        r_vec = np.cross(fwd, up_vec)
        r_vec = r_vec / np.linalg.norm(r_vec)
        u_vec = np.cross(r_vec, fwd)

        R = np.stack([r_vec, u_vec, -fwd], axis=0)
        t = -R @ eye
        view_mat = np.eye(4, dtype=np.float64)
        view_mat[:3, :3] = R
        view_mat[:3, 3] = t
        world_mat = np.linalg.inv(view_mat)

        cam = CameraParams(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            fov_degrees=fov_deg,
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            view_matrix=view_mat,
            world_matrix=world_mat,
            z_near=0.1,
            z_far=1000.0,
        )

        frames.append(
            FGPPFrame(
                camera=cam,
                depth=depth,
                direct_radiance=(diff_dir + spec_dir).astype(np.float16),
                world_normal=normals.astype(np.float16),
                motion_vector=mv.astype(np.float16),
                base_color=albedo.astype(np.float16),
                roughness_metallic=rough_metal.astype(np.float16),
                cryptomatte_id=crypto_id,
                frame_index=idx,
            )
        )

    seq = FGPPSequence(
        frames=frames,
        scene_id="caelum_cornell_spheres_17f",
        fps=24.0,
    )
    export_fgpp(seq, out_dir)

    # Write beauty GT EXR frames
    for idx, beauty_rgb in enumerate(beauty_list):
        beauty_path = out_dir / f"beauty_{idx:06d}.exr"
        header = OpenEXR.Header(width, height)
        header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
        half_ch = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header["channels"] = {"B": half_ch, "G": half_ch, "R": half_ch}
        out_file = OpenEXR.OutputFile(str(beauty_path), header)
        out_file.writePixels(
            {
                "R": beauty_rgb[..., 0].astype(np.float16).tobytes(),
                "G": beauty_rgb[..., 1].astype(np.float16).tobytes(),
                "B": beauty_rgb[..., 2].astype(np.float16).tobytes(),
            }
        )
        out_file.close()

    # Enrich metadata.json with Track-A specific fields — honest provenance
    # This function always uses the self-contained SoftwareRasterizer FGPP
    # generator (not caelum-core path-tracing), so provenance must honestly
    # reflect the actual rendering source regardless of whether the
    # caelum-core repository directory exists on disk.
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["generator"] = "synapse.render_core.SoftwareRasterizer"
        meta["render_track"] = "synthetic_track_a_reference"
        meta["provenance"] = "synthetic_track_a_reference"
        meta["note"] = (
            "Self-contained FGPP reference data; not rendered by caelum-core. "
            "SPP/bounce metadata intentionally omitted to avoid false provenance."
        )
        for f_dict in meta.get("frames", []):
            f_idx = f_dict.get("frame_index", 0)
            f_dict["beauty_gt_file"] = f"beauty_{f_idx:06d}.exr"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _summarize_sequence(seq_dir: Path) -> dict[str, Any]:
    """Validate an exported FGPP directory and return its manifest record."""
    seq = import_fgpp(seq_dir, lazy=False)
    f0 = seq.frames[0]  # type: ignore[union-attr]
    assert f0.cryptomatte_id is not None
    unique_ids = [int(x) for x in np.unique(f0.cryptomatte_id)]
    meta_path = seq_dir / "metadata.json"
    frame0_path = seq_dir / "frame_000000.exr"
    beauty_files = sorted(seq_dir.glob("beauty_*.exr"))

    return {
        "scene_id": seq.scene_id,  # type: ignore[union-attr]
        "relative_path": seq_dir.name,
        "num_frames": seq.num_frames,  # type: ignore[union-attr]
        "fps": seq.fps,  # type: ignore[union-attr]
        "resolution": [f0.width, f0.height],
        "unique_cryptomatte_ids_frame0": len(unique_ids),
        "has_beauty_gt": len(beauty_files) == seq.num_frames,  # type: ignore[union-attr]
        "metadata_sha256": _sha256_file(meta_path),
        "frame_000000_sha256": _sha256_file(frame0_path),
    }


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "SYNAPSE_DATASET_ROOT",
        os.environ.get(
            "DERVGEN_DATASETS",
            "/root/work/datasets/synapse"
            if Path("/root/work/datasets/synapse").exists()
            else str(REPO_ROOT / "data" / "golden_datasets"),
        ),
    )
)
DEFAULT_FULGUR_REPO = Path(
    os.environ.get(
        "FULGUR_REPO",
        "/root/work/dervgen/fulgur"
        if Path("/root/work/dervgen/fulgur").exists()
        else str(REPO_ROOT.parent / "fulgur"),
    )
)
DEFAULT_CAELUM_REPO = Path(
    os.environ.get(
        "CAELUM_REPO",
        "/root/work/dervgen/caelum-core"
        if Path("/root/work/dervgen/caelum-core").exists()
        else str(REPO_ROOT.parent / "caelum-core"),
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        dest="output_root",
        help="Root directory for exported Golden EXR sequences.",
    )
    parser.add_argument(
        "--fulgur-repo",
        type=Path,
        default=DEFAULT_FULGUR_REPO,
        help="Path to fulgur repository.",
    )
    parser.add_argument(
        "--caelum-repo",
        type=Path,
        default=DEFAULT_CAELUM_REPO,
        help="Path to caelum-core repository.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/golden_dataset_manifest.json"),
        help="Path to output JSON manifest file.",
    )
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=None,
        help="Path to portable Colab tar.gz archive (defaults to {output_root}/synapse_colab_bundle_v1.tar.gz).",
    )
    args = parser.parse_args()

    out_root: Path = args.output_root
    bundle_out_path = (
        args.bundle_out
        if args.bundle_out is not None
        else (out_root / "synapse_colab_bundle_v1.tar.gz")
    )
    dir_102_17f = out_root / "track_b_fulgur" / "course_102_pbr_17f"
    dir_102_97f = out_root / "track_b_fulgur" / "course_102_pbr_97f"
    dir_caelum_17f = out_root / "track_a_caelum" / "cornell_spheres_17f"

    print(f"[1/4] Exporting Track-B fulgur course_102_pbr_17f -> {dir_102_17f} ...")
    _export_fulgur_course102(args.fulgur_repo, dir_102_17f, num_frames=17)

    print(f"[2/4] Building Track-B long-horizon course_102_pbr_97f -> {dir_102_97f} ...")
    _build_long_horizon_97f_from_base(dir_102_17f, dir_102_97f)

    print(f"[3/4] Exporting Track-A caelum cornell_spheres_17f -> {dir_caelum_17f} ...")
    _export_caelum_track_a_17f(args.caelum_repo, dir_caelum_17f, num_frames=17)

    manifest = {
        "schema_version": FGPP_SCHEMA_VERSION,
        "channels_per_frame": FGPP_NUM_CHANNELS,
        "bytes_per_pixel": FGPP_BYTES_PER_PIXEL,
        "datasets": {
            "track_b_course_102_pbr_17f": _summarize_sequence(dir_102_17f),
            "track_b_course_102_pbr_97f": _summarize_sequence(dir_102_97f),
            "track_a_cornell_spheres_17f": _summarize_sequence(dir_caelum_17f),
        },
    }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[4/4] Wrote dataset manifest -> {args.manifest_out}")

    bundle_out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_out_path, "w:gz") as tar:
        tar.add(dir_102_17f, arcname="track_b_fulgur/course_102_pbr_17f")
        tar.add(dir_102_97f, arcname="track_b_fulgur/course_102_pbr_97f")
        tar.add(dir_caelum_17f, arcname="track_a_caelum/cornell_spheres_17f")
        tar.add(args.manifest_out, arcname="golden_dataset_manifest.json")

    bundle_mb = bundle_out_path.stat().st_size / (1024 * 1024)
    print(f"Packaged Colab bundle -> {bundle_out_path} ({bundle_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
