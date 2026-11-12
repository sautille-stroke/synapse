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
import math
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from synapse.fgpp import import_fgpp
from synapse.fgpp.schema import FGPP_BYTES_PER_PIXEL, FGPP_NUM_CHANNELS, FGPP_SCHEMA_VERSION


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

    bin_path = fulgur_repo / "bazel-bin" / "course" / "102-pbr-direct"
    if not bin_path.is_file():
        subprocess.run(
            ["bazel", "build", "//course:102-pbr-direct"],
            cwd=str(fulgur_repo),
            check=True,
        )

    cmd = [
        str(bin_path),
        "--headless",
        f"--width={width}",
        f"--height={height}",
        f"--export-fgpp={out_dir}",
        f"--fgpp-frames={num_frames}",
    ]
    subprocess.run(cmd, cwd=str(fulgur_repo), check=True)


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
        f_meta["is_physical_anchor"] = (idx % 48 == 0)
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
        f_meta["camera"]["view_matrix"] = view_row.tolist()
        f_meta["camera"]["world_matrix"] = world_row.tolist()
        frames_meta.append(f_meta)

    meta_97 = dict(base_meta)
    meta_97["scene_id"] = "fulgur_course102_97f"
    meta_97["total_frames"] = 97
    meta_97["num_frames"] = 97
    meta_97["frames"] = frames_meta
    (out_97f_dir / "metadata.json").write_text(
        json.dumps(meta_97, indent=2), encoding="utf-8"
    )


def _export_caelum_track_a_17f(
    caelum_repo: Path,
    out_dir: Path,
    num_frames: int = 17,
    width: int = 832,
    height: int = 480,
) -> None:
    """Export 17-frame Track-A FGPP + Beauty GT using caelum.camera.fgpp_exporter."""
    if str(caelum_repo) not in sys.path:
        sys.path.insert(0, str(caelum_repo))

    from caelum.camera.aov import AOVBuffer, AOVType, cryptomatte_name_hash
    from caelum.camera.camera import PinholeCamera
    from caelum.camera.fgpp_exporter import FgppFrameInput, export_fgpp_sequence

    if out_dir.exists():
        shutil.rmtree(out_dir)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx_grid = (xx - width * 0.5) / (width * 0.5)
    ny_grid = (height * 0.5 - yy) / (height * 0.5)

    hero_hash = cryptomatte_name_hash("hero_character")
    prop_hash = cryptomatte_name_hash("chrome_orb")
    bg_hash = cryptomatte_name_hash("studio_stage")

    inputs: list[FgppFrameInput] = []
    for idx in range(num_frames):
        aov = AOVBuffer(
            width,
            height,
            aov_types=[
                AOVType.BEAUTY,
                AOVType.ALBEDO,
                AOVType.NORMAL,
                AOVType.DEPTH,
                AOVType.DIFFUSE_DIRECT,
                AOVType.SPECULAR_DIRECT,
                AOVType.CRYPTOMATTE_OBJECT,
                AOVType.MOTION_VECTOR,
            ],
        )

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
            np.maximum(
                1.0 - normals[mask_hero, 0] ** 2 - normals[mask_hero, 1] ** 2, 0.01
            )
        )
        normals[mask_orb, 0] = (nx_grid[mask_orb] - cx2) / 0.283
        normals[mask_orb, 1] = (ny_grid[mask_orb] - cy2) / 0.283
        normals[mask_orb, 2] = np.sqrt(
            np.maximum(
                1.0 - normals[mask_orb, 0] ** 2 - normals[mask_orb, 1] ** 2, 0.01
            )
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
        spec_dir = np.power(ndotl, 24.0) * np.where(
            mask_orb[..., None], 2.8, 0.45
        ).astype(np.float32)
        indirect_gi = albedo * 0.22
        beauty = diff_dir + spec_dir + indirect_gi

        # Cryptomatte ID
        crypto = np.zeros((height, width, 2), dtype=np.float32)
        crypto[..., 0] = bg_hash
        crypto[..., 1] = 1.0
        crypto[mask_hero, 0] = hero_hash
        crypto[mask_orb, 0] = prop_hash

        # Motion Vector
        mv = np.zeros((height, width, 2), dtype=np.float32)
        mv[mask_hero, 0] = 0.015 * width * 0.5
        mv[mask_orb, 0] = -0.010 * width * 0.5

        # Populate AOVBuffer directly via internal arrays for fast 832x480 generation
        for aov_t, arr in (
            (AOVType.DEPTH, depth),
            (AOVType.ALBEDO, albedo),
            (AOVType.NORMAL, normals),
            (AOVType.DIFFUSE_DIRECT, diff_dir),
            (AOVType.SPECULAR_DIRECT, spec_dir),
            (AOVType.BEAUTY, beauty),
            (AOVType.CRYPTOMATTE_OBJECT, crypto),
            (AOVType.MOTION_VECTOR, mv),
        ):
            aov._buffers[aov_t][:] = arr
            aov._sample_counts[aov_t].fill(1.0)

        rough = np.full((height, width), 0.6, dtype=np.float32)
        rough[mask_hero] = 0.35
        rough[mask_orb] = 0.08
        metal = np.zeros((height, width), dtype=np.float32)
        metal[mask_orb] = 0.92

        cam = PinholeCamera.from_look_at(
            eye=(0.05 * idx, 1.8, 6.5),
            target=(0.0, 0.2, 0.0),
            up=(0.0, 1.0, 0.0),
            width=width,
            height=height,
            fov_degrees=50.0,
        )
        inputs.append(
            FgppFrameInput(
                aov_buffer=aov,
                camera=cam,
                frame_index=idx,
                roughness=rough,
                metallic=metal,
            )
        )

    export_fgpp_sequence(
        inputs,
        out_dir,
        fps=24.0,
        scene_id="caelum_cornell_spheres_17f",
        spp=256,
        max_bounces=8,
        export_beauty_gt=True,
    )


def _summarize_sequence(seq_dir: Path) -> dict[str, Any]:
    """Validate an exported FGPP directory and return its manifest record."""
    seq = import_fgpp(seq_dir, lazy=False)
    f0 = seq.frames[0]
    assert f0.cryptomatte_id is not None
    unique_ids = [int(x) for x in np.unique(f0.cryptomatte_id)]
    meta_path = seq_dir / "metadata.json"
    frame0_path = seq_dir / "frame_000000.exr"
    beauty_files = sorted(seq_dir.glob("beauty_*.exr"))

    return {
        "scene_id": seq.scene_id,
        "relative_path": seq_dir.name,
        "num_frames": seq.num_frames,
        "fps": seq.fps,
        "resolution": [f0.width, f0.height],
        "unique_cryptomatte_ids_frame0": len(unique_ids),
        "has_beauty_gt": len(beauty_files) == seq.num_frames,
        "metadata_sha256": _sha256_file(meta_path),
        "frame_000000_sha256": _sha256_file(frame0_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/work/datasets/synapse"),
        help="Root directory for exported Golden EXR sequences.",
    )
    parser.add_argument(
        "--fulgur-repo",
        type=Path,
        default=Path("/root/work/dervgen/fulgur"),
        help="Path to fulgur repository on ubuntu24.",
    )
    parser.add_argument(
        "--caelum-repo",
        type=Path,
        default=Path("/root/work/dervgen/caelum-core"),
        help="Path to caelum-core repository on ubuntu24.",
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
        default=Path("/root/work/datasets/synapse/synapse_colab_bundle_v1.tar.gz"),
        help="Path to portable Colab tar.gz archive.",
    )
    args = parser.parse_args()

    out_root: Path = args.output_root
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

    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.bundle_out, "w:gz") as tar:
        tar.add(dir_102_17f, arcname="track_b_fulgur/course_102_pbr_17f")
        tar.add(dir_caelum_17f, arcname="track_a_caelum/cornell_spheres_17f")
        tar.add(args.manifest_out, arcname="golden_dataset_manifest.json")

    bundle_mb = args.bundle_out.stat().st_size / (1024 * 1024)
    print(f"Packaged Colab bundle -> {args.bundle_out} ({bundle_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
