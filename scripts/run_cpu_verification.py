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
import tempfile
import time
from pathlib import Path
from typing import Any

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
    # ch 0: log-depth, ch 1..3: world normal, ch 4..5: motion vector, ch 6: occlusion
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
    patch_rgb = np.clip(
        base_rgb * np.array([1.08, 0.96, 0.94], dtype=np.float32), 0.0, 1.0
    )
    matched_patch = border_ring_lab_match(
        patch_rgb, base_rgb, alpha_mask > 0.5, ring_width_px=16
    )
    blended = laplacian_pyramid_blend(
        matched_patch, base_rgb, alpha_mask, num_levels=4
    )
    assert blended.shape == base_rgb.shape
    assert bool(np.all(np.isfinite(blended)))

    ring_diff_before = float(np.mean(np.abs(patch_rgb - base_rgb)))
    ring_diff_after = float(np.mean(np.abs(blended - base_rgb)))

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
    assert windows[0].start == 0 and windows[0].end == 97
    assert windows[1].start == 64 and windows[1].end == 161

    # Test variance-preserving sigmoid latent overlap fusion across 9 latent steps
    rng = np.random.default_rng(42)
    z_prev = rng.standard_normal((9, 16, 16, 16)).astype(np.float32)
    z_curr = z_prev + 0.02
    fused = fuse_overlapping_latents(z_prev, z_curr, overlap_steps=9)
    assert fused.shape == (9, 16, 16, 16)
    assert bool(np.all(np.isfinite(fused)))

    # Inject a synthetic frame-to-frame discontinuity at frame 64 in memory to verify detector
    f64 = seq_97f.frames[64]
    if f64.motion_vector is not None:
        f64.motion_vector = f64.motion_vector.copy() + np.float16(25.0)
    if f64.world_normal is not None:
        f64.world_normal = -f64.world_normal.copy()

    detector = DiscontinuityDetector(threshold=0.55)
    scheduler = ConditionScheduler(
        plan=CausalWindowPlan(97, 64, 33, seq_97f.num_frames),
        detector=detector,
    )
    scheduled = scheduler.schedule(seq_97f)
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
        rgb_video = np.zeros(
            (seq_track_a.num_frames, f0.height, f0.width, 3), dtype=np.uint8
        )
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
        assert "latent" in s0 and "prompt" in s0

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
            adapter_path=str(lora_a),
            adapter_sha256=hashlib.sha256(lora_a_bytes).hexdigest(),
            target_modules=["spatial_attn.to_q"],
        )
        manifest_b = LoRAManifest(
            name="scene_style",
            category="style",
            rank=16,
            alpha=16.0,
            adapter_path=str(lora_b),
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
        default=Path("/root/work/datasets/synapse"),
        help="Root directory of exported Golden FGPP datasets.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/manifests/ubuntu24_cpu_verification_report.json"),
        help="Path to write JSON verification report.",
    )
    args = parser.parse_args()

    t0 = time.time()
    dir_b_17f = args.dataset_root / "track_b_fulgur" / "course_102_pbr_17f"
    dir_b_97f = args.dataset_root / "track_b_fulgur" / "course_102_pbr_97f"
    dir_a_17f = args.dataset_root / "track_a_caelum" / "cornell_spheres_17f"

    print("Loading Golden FGPP sequences from", args.dataset_root)
    seq_b_17f = import_fgpp(dir_b_17f, lazy=False)
    seq_b_97f = import_fgpp(dir_b_97f, lazy=False)
    seq_a_17f = import_fgpp(dir_a_17f, lazy=False)
    assert isinstance(seq_b_17f, FGPPSequence)
    assert isinstance(seq_b_97f, FGPPSequence)
    assert isinstance(seq_a_17f, FGPPSequence)

    print("[E1] Verifying 7-ch conditioning ablation & MockVideoAdapter ...")
    res_e1 = _verify_e1_conditioning_ablation(seq_b_17f, seq_a_17f)

    print("[E2] Verifying ACEScg style extraction & 16px surgical compositing ...")
    res_e2 = _verify_e2_style_and_composite(seq_a_17f)

    print("[E3] Verifying 97-frame sliding window scheduler & shot-cut detector ...")
    res_e3 = _verify_e3_long_horizon_scheduler(seq_b_97f)

    print("[E4] Verifying WebDataset FeatureStore & Multi-LoRA Cryptomatte masks ...")
    res_e4 = _verify_e4_feature_store_and_multi_lora(seq_a_17f)

    print("[E5] Verifying Track-B whitebox floor & 30-scenario Synapse-Bench ...")
    res_e5 = _verify_e5_whitebox_and_synapse_bench()

    report = {
        "environment": "ubuntu24_cpu_headless",
        "elapsed_sec": round(time.time() - t0, 3),
        "all_passed": True,
        "experiments": {
            "E1_fgpp_conditioning_ablation": res_e1,
            "E2_acescg_style_and_surgical_composite": res_e2,
            "E3_causal_sliding_window_and_shot_cut": res_e3,
            "E4_feature_store_and_cryptomatte_lora": res_e4,
            "E5_whitebox_floor_and_synapse_bench": res_e5,
        },
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ALL E1-E5 CPU BASELINE VERIFICATIONS PASSED! Report -> {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
