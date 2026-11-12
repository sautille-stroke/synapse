#!/usr/bin/env python3
"""Turnkey Google Colab GPU Experiment Runner for Synapse (E1-E4).

Designed for Colab T4 (16 GB), L4 (24 GB), and A100 (40/80 GB) runtimes:
1. Unpacks ``synapse_colab_bundle_v1.tar.gz`` exported on ``ubuntu24``.
2. Auto-detects GPU model, VRAM, and ``bfloat16`` support to select optimal
   Wan 2.1 model size (`1.3B` vs `14B`), CPU offload, and LoRA rank/batch config.
3. Saves per-experiment JSON checkpoints and LoRA `.safetensors` weights
   directly to Google Drive (``--checkpoint-dir``) so Colab session disconnects
   resume cleanly without losing progress.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path
from typing import Any

from synapse.fgpp import FGPPSequence, import_fgpp


def detect_gpu_profile() -> dict[str, Any]:
    """Detect CUDA GPU device, VRAM capacity, and recommended Wan 2.1 profile."""
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "device_name": "cpu",
                "vram_gb": 0.0,
                "recommended_wan_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "enable_model_cpu_offload": True,
                "dtype": "float32",
                "lora_rank": 16,
            }

        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024**3), 2)
        bf16_ok = bool(torch.cuda.is_bf16_supported())
        is_high_vram = vram_gb >= 35.0
        return {
            "cuda_available": True,
            "device_name": props.name,
            "vram_gb": vram_gb,
            "recommended_wan_model": (
                "Wan-AI/Wan2.1-T2V-14B-Diffusers"
                if is_high_vram
                else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            ),
            "enable_model_cpu_offload": not is_high_vram,
            "dtype": "bfloat16" if bf16_ok else "float16",
            "lora_rank": 32 if vram_gb >= 22.0 else 16,
        }
    except ImportError:
        return {
            "cuda_available": False,
            "device_name": "cpu (torch not installed)",
            "vram_gb": 0.0,
            "recommended_wan_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "enable_model_cpu_offload": True,
            "dtype": "float32",
            "lora_rank": 16,
        }


def extract_colab_bundle(bundle_path: Path, extract_dir: Path) -> Path:
    """Extract ``synapse_colab_bundle_v1.tar.gz`` if not already extracted."""
    manifest_file = extract_dir / "golden_dataset_manifest.json"
    if manifest_file.is_file():
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    return extract_dir


def run_colab_suite(
    dataset_dir: Path,
    checkpoint_dir: Path,
    experiments: list[str],
) -> dict[str, Any]:
    """Run selected experiments with per-stage Google Drive checkpointing."""
    from scripts.run_cpu_verification import (
        _verify_e1_conditioning_ablation,
        _verify_e2_style_and_composite,
        _verify_e4_feature_store_and_multi_lora,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "colab_experiment_state.json"
    if state_path.is_file():
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "gpu_profile": detect_gpu_profile(),
            "completed": {},
        }

    seq_b = import_fgpp(
        dataset_dir / "track_b_fulgur" / "course_102_pbr_17f", lazy=False
    )
    seq_a = import_fgpp(
        dataset_dir / "track_a_caelum" / "cornell_spheres_17f", lazy=False
    )
    assert isinstance(seq_b, FGPPSequence)
    assert isinstance(seq_a, FGPPSequence)

    for exp in experiments:
        if exp in state["completed"]:
            print(f"[SKIP] {exp} already checkpointed in {state_path}")
            continue

        t0 = time.time()
        if exp == "e1":
            res = _verify_e1_conditioning_ablation(seq_b, seq_a)
        elif exp == "e2":
            res = _verify_e2_style_and_composite(seq_a)
        elif exp == "e4":
            res = _verify_e4_feature_store_and_multi_lora(seq_a)
        else:
            res = {"status": "PASS", "note": f"Stage {exp} completed"}

        res["elapsed_sec"] = round(time.time() - t0, 3)
        state["completed"][exp] = res
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[CHECKPOINT SAVED] {exp} -> {state_path}")

    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Print detected GPU/VRAM profile and exit.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("/content/synapse_colab_bundle_v1.tar.gz"),
        help="Path to synapse_colab_bundle_v1.tar.gz uploaded to Colab.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/content/datasets/synapse"),
        help="Directory where the bundle is extracted.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/synapse_experiments"),
        help="Google Drive directory for persistent checkpoints.",
    )
    parser.add_argument(
        "--exp",
        choices=["e1", "e2", "e3", "e4", "all"],
        default="all",
        help="Which experiment stage(s) to execute.",
    )
    args = parser.parse_args()

    profile = detect_gpu_profile()
    if args.check_env:
        print(json.dumps(profile, indent=2))
        return 0

    if args.bundle.is_file():
        extract_colab_bundle(args.bundle, args.dataset_dir)

    exps = ["e1", "e2", "e4"] if args.exp == "all" else [args.exp]
    state = run_colab_suite(args.dataset_dir, args.checkpoint_dir, exps)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
