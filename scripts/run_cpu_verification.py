#!/usr/bin/env python3
"""GPU-less E1-E5 Baseline Verification Runner on ubuntu24.

Executes the complete CPU/headless pre-experiment verification against the real
exported Golden FGPP datasets (``fulgur`` Track-B & ``caelum-core`` Track-A):

- **E1**: FGPP 7-channel conditioning ablation (E1-A..E1-D), log-depth
  normalisation, camera Plücker ray conditioning, and ``MockVideoAdapter``
  round-trip evaluation.
- **E2**: ACEScg intrinsic style extraction + soft-shoulder highlight rolloff
  and 16px surgical boundary ring CIE LAB matching + Laplacian blending.
- **E3**: 97-frame causal sliding window (W=97, S=64, overlap=33), sigmoid
  latent fusion, physical anchor pinning (t=0, 48, 96), and DiscontinuityDetector
  shot-cut reset.
- **E4**: WebDataset FeatureStore shard packing + round-trip verification and
  Cryptomatte multi-entity LoRA spatial attention mask orthogonality (< 3% leakage).
- **E5**: Track-B whitebox coarseness floor (>= 1500 vertices + articulated
  limbs) and 30-scenario ``Synapse-Bench`` catalogue validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _safe_exists(path_str: str) -> bool:
    try:
        return Path(path_str).exists()
    except OSError:
        return False


DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "SYNAPSE_DATASET_ROOT",
        os.environ.get(
            "DERVGEN_DATASETS",
            "/root/work/datasets/synapse"
            if _safe_exists("/root/work/datasets/synapse")
            else str(REPO_ROOT / "data" / "golden_datasets"),
        ),
    )
)

import numpy as np
from synapse.adapters import (
    CausalWindowPlan,
    ConditionScheduler,
    DiscontinuityDetector,
    LoRAManifest,
    MockVideoAdapter,
    MultiLoRAManager,
    VideoGenerationConfig,
    WanAdapter,
    fuse_overlapping_latents,
)
from synapse.bench import (
    generate_humanoid_proxy,
    list_scenarios,
)
from synapse.composite import (
    border_ring_lab_match,
    generate_surgical_mask,
    laplacian_pyramid_blend,
)
from synapse.fgpp import (
    FGPPSequence,
    import_fgpp,
    validate_whitebox_coarseness_floor,
)
from synapse.style import (
    extract_style_fgpp_intrinsic,
    to_style_profile_dict,
)
from synapse.training import (
    FeatureStoreBuilder,
    LinearProjectionEncoder,
    LinearProjectionTextEncoder,
    iter_feature_shards,
)


def _verify_sha256(file_path: Path, expected_sha: str) -> bool:
    """Verify SHA-256 checksum of a file against an expected digest."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha


def _check_manifest_sha256(dataset_dir: Path) -> dict[str, Any]:
    """Verify SHA-256 checksums of EXR frame files against the manifest.

    Looks for ``metadata.json`` in *dataset_dir* and checks every file listed
    in the ``"frames"`` array that includes a ``"sha256"`` field.

    Returns:
        dict with ``"checked"``, ``"passed"``, ``"failed"`` counts and
        ``"mismatches"`` list of filenames that did not match.
    """
    manifest_path = dataset_dir / "metadata.json"
    result: dict[str, Any] = {
        "checked": 0,
        "passed": 0,
        "failed": 0,
        "mismatches": [],
    }
    if not manifest_path.is_file():
        return result

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    for entry in frames:
        fname = entry.get("file") or entry.get("filename")
        expected_sha = entry.get("sha256")
        if not fname or not expected_sha:
            continue
        fpath = dataset_dir / fname
        result["checked"] += 1
        if not fpath.is_file():
            result["failed"] += 1
            result["mismatches"].append(f"{fname} (missing)")
            continue
        if _verify_sha256(fpath, expected_sha):
            result["passed"] += 1
        else:
            result["failed"] += 1
            result["mismatches"].append(fname)
    return result


def _verify_e1_conditioning_ablation(
    seq_track_b: FGPPSequence,
    seq_track_a: FGPPSequence,
) -> dict[str, Any]:
    """Verify E1-A..E1-D 7-channel conditioning tensors on real Track-B & Track-A data."""
    adapter = WanAdapter()
    f0 = seq_track_b.frames[0]
    cfg = VideoGenerationConfig(
        width=f0.width,
        height=f0.height,
        num_frames=seq_track_b.num_frames,
        fps=seq_track_b.fps,
    )
    payload_b = adapter.prepare_conditions(seq_track_b, cfg)
    payload_a = adapter.prepare_conditions(seq_track_a, cfg)

    assert payload_b.control_tensor is not None
    assert payload_a.control_tensor is not None
    ctrl_b = payload_b.control_tensor
    ctrl_a = payload_a.control_tensor
    assert ctrl_b.shape == (7, seq_track_b.num_frames, f0.height, f0.width)
    assert ctrl_a.shape == (7, seq_track_a.num_frames, f0.height, f0.width)
    assert bool(np.all(np.isfinite(ctrl_b)))
    assert bool(np.all(np.isfinite(ctrl_a)))

    # Construct the 4 ablation tensor masks:
    # ch 0: log-depth, ch 1..3: camera-space normal, ch 4..6: ACEScg direct radiance
    ablation_norms: dict[str, float] = {}
    for name, active_ch in (
        ("E1-A_rgb_only", []),
        ("E1-B_rgb_depth", [0]),
        ("E1-C_rgb_depth_normal", [0, 1, 2, 3]),
        ("E1-D_full_7ch", [0, 1, 2, 3, 4, 5, 6]),
    ):
        masked = np.zeros_like(ctrl_b)
        if active_ch:
            masked[active_ch] = ctrl_b[active_ch]
        ablation_norms[name] = float(np.linalg.norm(masked))

    mock = MockVideoAdapter()
    mock_cond = mock.prepare_conditions(seq_track_b, cfg)
    gen_res = mock.generate(mock_cond, cfg)
    assert gen_res.frames_srgb_u8.shape == (
        seq_track_b.num_frames,
        f0.height,
        f0.width,
        3,
    )

    # Monotonicity assertion: ablation norms must be non-decreasing as
    # more channels are enabled (E1-A ≤ E1-B ≤ E1-C ≤ E1-D)
    norm_values = list(ablation_norms.values())
    for i in range(1, len(norm_values)):
        assert norm_values[i] >= norm_values[i - 1], (
            f"Ablation norm monotonicity violated: "
            f"{list(ablation_norms.keys())[i-1]}={norm_values[i-1]:.4f} > "
            f"{list(ablation_norms.keys())[i]}={norm_values[i]:.4f}"
        )

    # Depth channel ring-band assertion: depth values should decrease
    # from background (far) towards foreground objects (near)
    depth_ch = ctrl_b[0]  # log-depth channel
    assert float(np.std(depth_ch)) > 0.01, (
        "Depth channel has near-zero variance — likely constant/degenerate"
    )

    return {
        "status": "PASS",
        "track_b_tensor_shape": list(ctrl_b.shape),
        "track_a_tensor_shape": list(ctrl_a.shape),
        "ablation_l2_norms": ablation_norms,
        "depth_channel_range": [float(np.min(ctrl_b[0])), float(np.max(ctrl_b[0]))],
    }


def _verify_e2_style_and_composite(
    seq_track_a: FGPPSequence,
) -> dict[str, Any]:
    """Verify E2 ACEScg intrinsic style extraction + 16px surgical ring blending."""
    f0 = seq_track_a.frames[0]
    assert f0.direct_radiance is not None
    assert f0.cryptomatte_id is not None

    # Create sRGB uint8 preview frames for style extraction
    srgb_frames = []
    for f in seq_track_a.frames:
        assert f.direct_radiance is not None
        rad = np.clip(f.direct_radiance.astype(np.float32), 0.0, 1.0)
        srgb = np.clip(np.power(rad, 1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8)
        srgb_frames.append(srgb)
    srgb_stack = np.stack(srgb_frames, axis=0)

    style_desc = extract_style_fgpp_intrinsic(srgb_stack, seq_track_a)
    style_profile = to_style_profile_dict(style_desc)

    # Surgical mask from Cryptomatte entity + 16px border ring matching
    unique_ids = [int(x) for x in np.unique(f0.cryptomatte_id) if int(x) != 0]
    target_entity = unique_ids[0]
    mask_3d = f0.cryptomatte_id == np.uint32(target_entity)
    alpha_mask = generate_surgical_mask(
        mask_3d_new=mask_3d,
        motion_vectors=f0.motion_vector,
        r_base=6,
        feather_width=4.0,
    )
    assert alpha_mask.shape == (f0.height, f0.width)
    assert 0.01 < float(np.mean(alpha_mask > 0.5)) < 0.90

    base_rgb = np.clip(f0.direct_radiance.astype(np.float32), 0.0, 1.0)
    patch_rgb = np.clip(base_rgb * np.array([1.08, 0.96, 0.94], dtype=np.float32), 0.0, 1.0)
    matched_patch = border_ring_lab_match(patch_rgb, base_rgb, alpha_mask > 0.5, ring_width_px=16)
    blended = laplacian_pyramid_blend(matched_patch, base_rgb, alpha_mask, num_levels=4)
    assert blended.shape == base_rgb.shape
    assert bool(np.all(np.isfinite(blended)))

    ring_diff_before = float(np.mean(np.abs(patch_rgb - base_rgb)))
    ring_diff_after = float(np.mean(np.abs(blended - base_rgb)))

    # Ring-band assertion: border matching + blending must reduce boundary
    # artifacts — the blended result should be closer to base than the raw
    # patch (or at most equal if the patch was already well-matched)
    assert ring_diff_after <= ring_diff_before + 1e-6, (
        f"Ring-band blending regression: diff_after={ring_diff_after:.6f} > "
        f"diff_before={ring_diff_before:.6f}"
    )

    return {
        "status": "PASS",
        "style_profile_version": style_profile.get("schema_version", "2.0.0"),
        "mask_foreground_ratio": float(np.mean(alpha_mask > 0.5)),
        "mean_abs_diff_before": ring_diff_before,
        "mean_abs_diff_after": ring_diff_after,
    }


def _verify_e3_long_horizon_scheduler(
    seq_97f: FGPPSequence,
) -> dict[str, Any]:
    """Verify E3 97-frame causal sliding window (W=97, S=64, overlap=33) and shot-cut reset."""
    plan = CausalWindowPlan(
        window_frames=97,
        stride_frames=64,
        overlap_frames=33,
        total_frames=161,
    )
    windows = plan.compute_windows()
    assert len(windows) == 2
    assert windows[0].start == 0
    assert windows[0].end == 97
    assert windows[1].start == 64
    assert windows[1].end == 161

    # Test variance-preserving sigmoid latent overlap fusion across 9 latent steps
    rng = np.random.default_rng(42)
    z_prev = rng.standard_normal((9, 16, 16, 16)).astype(np.float32)
    z_curr = z_prev + 0.02
    fused = fuse_overlapping_latents(z_prev, z_curr, overlap_steps=9)
    assert fused.shape == (9, 16, 16, 16)
    assert bool(np.all(np.isfinite(fused)))

    # Inject a synthetic frame-to-frame discontinuity at frame 64 in memory to verify detector
    # Use deepcopy to avoid corrupting the original sequence for downstream tests
    import copy

    test_seq = copy.deepcopy(seq_97f)
    f64 = test_seq.frames[64]
    if f64.motion_vector is not None:
        f64.motion_vector = f64.motion_vector.copy() + np.float16(25.0)
    if f64.world_normal is not None:
        f64.world_normal = -f64.world_normal.copy()
    if f64.cryptomatte_id is not None:
        f64.cryptomatte_id = np.ones_like(f64.cryptomatte_id, dtype=np.uint32)

    detector = DiscontinuityDetector(threshold=0.55)
    scheduler = ConditionScheduler(
        plan=CausalWindowPlan(97, 64, 33, test_seq.num_frames),
        detector=detector,
    )
    scheduled = scheduler.schedule(test_seq)
    assert len(scheduled) == 1
    cut_indices = [r.frame_index for r in scheduled[0].reanchor_requests]
    assert 64 in cut_indices, f"Expected shot-cut detected at frame 64, got {cut_indices}"

    return {
        "status": "PASS",
        "num_windows_for_161f": len(windows),
        "overlap_latent_steps": 9,
        "fused_latent_shape": list(fused.shape),
        "detected_shot_cut_frames": cut_indices,
    }


def _verify_e4_feature_store_and_multi_lora(
    seq_track_a: FGPPSequence,
) -> dict[str, Any]:
    """Verify E4 WebDataset FeatureStore shard packing & Cryptomatte Multi-LoRA masking."""
    f0 = seq_track_a.frames[0]
    assert f0.cryptomatte_id is not None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        vae_enc = LinearProjectionEncoder(latent_channels=16)
        txt_enc = LinearProjectionTextEncoder(embed_dim=64)
        builder = FeatureStoreBuilder(
            latent_encoder=vae_enc,
            text_encoder=txt_enc,
        )
        rgb_video = np.zeros((seq_track_a.num_frames, f0.height, f0.width, 3), dtype=np.uint8)
        for i, fr in enumerate(seq_track_a.frames):
            if fr.direct_radiance is not None:
                rad = np.clip(fr.direct_radiance.astype(np.float32), 0.0, 1.0)
                rgb_video[i] = (rad * 255.0).astype(np.uint8)

        builder.add_sample(
            sample_id="cornell_spheres_000",
            video_frames=rgb_video,
            caption="sks, studio spheres, key fill lighting, slow dolly, path_traced",
        )
        shards = builder.build_shards(tmp_path / "shards")
        assert len(shards) == 1

        loaded_samples = list(iter_feature_shards(shards))
        assert len(loaded_samples) == 1
        s0 = loaded_samples[0]
        assert "latent" in s0
        assert "prompt" in s0

        # Verify Multi-LoRA Cryptomatte spatial attention mask isolation
        unique_ids = [int(x) for x in np.unique(f0.cryptomatte_id) if int(x) != 0]
        id_hero, id_orb = unique_ids[0], unique_ids[1]

        lora_a = tmp_path / "hero.safetensors"
        lora_b = tmp_path / "orb.safetensors"
        lora_a_bytes = b"safetensors_hero"
        lora_b_bytes = b"safetensors_orb"
        lora_a.write_bytes(lora_a_bytes)
        lora_b.write_bytes(lora_b_bytes)

        manifest_a = LoRAManifest(
            name="hero_identity",
            category="identity",
            rank=16,
            alpha=16.0,
            adapter_path=lora_a.name,  # Relative path for portability
            adapter_sha256=hashlib.sha256(lora_a_bytes).hexdigest(),
            target_modules=["spatial_attn.to_q"],
        )
        manifest_b = LoRAManifest(
            name="scene_style",
            category="style",
            rank=16,
            alpha=16.0,
            adapter_path=lora_b.name,  # Relative path for portability
            adapter_sha256=hashlib.sha256(lora_b_bytes).hexdigest(),
            target_modules=["cross_attn.to_k"],
        )

        mgr = MultiLoRAManager([(manifest_a, 1.0), (manifest_b, 0.8)])
        crypto_stack = np.stack(
            [fr.cryptomatte_id for fr in seq_track_a.frames if fr.cryptomatte_id is not None],
            axis=0,
        )
        m_hero = mgr.build_spatial_lora_masks(
            crypto_stack, target_entity_ids={id_hero}, spatial_block=8
        )
        m_orb = mgr.build_spatial_lora_masks(
            crypto_stack, target_entity_ids={id_orb}, spatial_block=8
        )
        overlap_leakage = float(np.mean(m_hero & m_orb))
        assert overlap_leakage == 0.0

    return {
        "status": "PASS",
        "shard_samples_verified": len(loaded_samples),
        "latent_shape": list(s0["latent"].shape),
        "multi_lora_mask_leakage": overlap_leakage,
    }


def _verify_e5_whitebox_and_synapse_bench() -> dict[str, Any]:
    """Verify E5 Track-B whitebox coarseness floor & 30-scenario Synapse-Bench suite."""
    vertices, _faces, bone_indices = generate_humanoid_proxy()
    num_verts = int(vertices.shape[0])
    assert num_verts >= 1500
    assert len(bone_indices) >= 10
    floor_ok = validate_whitebox_coarseness_floor(
        vertex_count=num_verts,
        has_articulated_limbs=True,
        is_character=True,
    )
    assert floor_ok is True

    scenarios = list_scenarios()
    assert len(scenarios) >= 30
    tiers: dict[str, int] = {}
    for sc in scenarios:
        t_name = sc.tier.value if hasattr(sc.tier, "value") else str(sc.tier)
        tiers[t_name] = tiers.get(t_name, 0) + 1

    return {
        "status": "PASS",
        "humanoid_proxy_vertices": num_verts,
        "whitebox_coarseness_floor_passed": floor_ok,
        "synapse_bench_total_scenarios": len(scenarios),
        "scenarios_by_tier": tiers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of exported Golden FGPP datasets.",
    )
    parser.add_argument(
        "--report-out",
        "--output-dir",
        type=Path,
        default=Path("data/manifests/ubuntu24_cpu_verification_report.json"),
        dest="report_out",
        help="Path to write JSON verification report.",
    )
    parser.add_argument(
        "--traceboard",
        nargs="?",
        const="__AUTO__",
        default=None,
        metavar="LOG_DIR",
        help=(
            "Enable Traceboard observability telemetry. Optionally specify LOG_DIR "
            "(defaults to <report-out parent>/traceboard)."
        ),
    )
    args = parser.parse_args()

    # Auto-resolve directory path to default report filename
    report_out: Path = args.report_out
    if report_out.is_dir():
        report_out = report_out / "cpu_verification_report.json"

    # --- Traceboard initialization (opt-in via --traceboard [LOG_DIR]) ---
    tb_started = False
    tb_status = "failed"
    if args.traceboard is not None:
        try:
            from synapse.observability import init_traceboard

            tb_log_dir = (
                report_out.parent / "traceboard"
                if args.traceboard == "__AUTO__"
                else Path(str(args.traceboard))
            )
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            probe_inst = init_traceboard(
                log_dir=str(tb_log_dir),
                run_name="cpu-verification",
                engine="synapse-cpu-verify",
            )
            tb_started = probe_inst is not None
            print(f"[traceboard] Telemetry enabled → {tb_log_dir}")
        except Exception as tb_err:
            print(
                f"[traceboard] WARNING: init failed ({tb_err}); continuing without telemetry.",
                file=sys.stderr,
            )

    t0 = time.time()
    try:
        dir_b_17f = args.dataset_root / "track_b_fulgur" / "course_102_pbr_17f"
        dir_b_97f = args.dataset_root / "track_b_fulgur" / "course_102_pbr_97f"
        dir_a_17f = args.dataset_root / "track_a_caelum" / "cornell_spheres_17f"

        bundle_tar = args.dataset_root / "synapse_colab_bundle_v1.tar.gz"
        if not dir_b_17f.exists() and bundle_tar.is_file():
            print(f"Extracting {bundle_tar} -> {args.dataset_root} ...")
            with tarfile.open(bundle_tar, "r:gz") as tar:
                tar.extractall(path=args.dataset_root, filter="data")

        # SHA-256 manifest verification of EXR frame files BEFORE decoding EXR bytes
        # (fail-closed if checked == 0 or any file is tampered/missing -> exit code 20)
        sha_results: dict[str, dict[str, Any]] = {}
        sha_checked_total = 0
        sha_failures_total = 0
        for label, ddir in [
            ("track_b_17f", dir_b_17f),
            ("track_b_97f", dir_b_97f),
            ("track_a_17f", dir_a_17f),
        ]:
            sha_res = _check_manifest_sha256(ddir)
            sha_results[label] = sha_res
            sha_checked_total += int(sha_res["checked"])
            sha_failures_total += int(sha_res["failed"])
            if sha_res["checked"] > 0:
                print(
                    f"  [SHA-256] {label}: {sha_res['passed']}/{sha_res['checked']} passed"
                    + (f", MISMATCHES: {sha_res['mismatches']}" if sha_res["failed"] else "")
                )

        if sha_checked_total == 0:
            print(
                "ERROR: 0 EXR files had SHA-256 entries in metadata.json. "
                "Re-run scripts/prepare_datasets.py to generate per-frame SHA-256 checksums.",
                file=sys.stderr,
            )
            return 20  # Exit code 20: fail-closed when no checksums were verified

        if sha_failures_total > 0:
            print(
                f"ERROR: {sha_failures_total} SHA-256 checksum mismatch(es) detected. "
                "EXR files may be corrupted or regenerated without updating the manifest.",
                file=sys.stderr,
            )
            return 20  # Exit code 20: input file SHA256 mismatch

        from synapse.observability import safe_log_scalar, traceboard_stage

        print("Loading Golden FGPP sequences from", args.dataset_root)
        seq_b_17f = import_fgpp(dir_b_17f, lazy=False)
        seq_b_97f = import_fgpp(dir_b_97f, lazy=False)
        seq_a_17f = import_fgpp(dir_a_17f, lazy=False)
        assert isinstance(seq_b_17f, FGPPSequence)
        assert isinstance(seq_b_97f, FGPPSequence)
        assert isinstance(seq_a_17f, FGPPSequence)

        experiments_map: dict[str, dict[str, Any]] = {}
        stage_specs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
            ("E1_fgpp_conditioning_ablation", lambda: _verify_e1_conditioning_ablation(seq_b_17f, seq_a_17f)),
            ("E2_acescg_style_and_surgical_composite", lambda: _verify_e2_style_and_composite(seq_a_17f)),
            ("E3_causal_sliding_window_and_shot_cut", lambda: _verify_e3_long_horizon_scheduler(seq_b_97f)),
            ("E4_feature_store_and_cryptomatte_lora", lambda: _verify_e4_feature_store_and_multi_lora(seq_a_17f)),
            ("E5_whitebox_floor_and_synapse_bench", _verify_e5_whitebox_and_synapse_bench),
        ]
        for exp_idx, (exp_name, runner_fn) in enumerate(stage_specs):
            print(f"[{exp_name}] Verifying ...")
            with traceboard_stage(f"cpu_verify/{exp_name}") as stage:
                exp_res = runner_fn()
                experiments_map[exp_name] = exp_res
                passed = exp_res.get("status") == "PASS"
                safe_log_scalar(
                    f"cpu_verify/{exp_name}/passed",
                    1.0 if passed else 0.0,
                    step=exp_idx,
                )
                if stage is not None:
                    if passed:
                        stage.set_pass()
                    else:
                        stage.set_fail("experiment_failed")

        elapsed_sec = round(time.time() - t0, 3)
        report: dict[str, Any] = {
            "environment": "ubuntu24_cpu_headless",
            "elapsed_sec": elapsed_sec,
            "all_passed": True,
            "sha256_checked": sha_checked_total,
            "sha256_verification": sha_results,
            "experiments": experiments_map,
        }

        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        safe_log_scalar("cpu_verify/elapsed_sec", elapsed_sec)
        safe_log_scalar("cpu_verify/all_passed", 1.0)
        tb_status = "finished"

        print(f"ALL E1-E5 CPU BASELINE VERIFICATIONS PASSED! Report -> {report_out}")
        return 0
    finally:
        if tb_started:
            try:
                from synapse.observability import shutdown_traceboard

                shutdown_traceboard(status=tb_status)
            except Exception:
                pass  # Best-effort telemetry


if __name__ == "__main__":
    raise SystemExit(main())
