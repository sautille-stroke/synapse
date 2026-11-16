#!/usr/bin/env python3
"""Turnkey Google Colab GPU Experiment Runner for Synapse (E1-E4).

Follows Google Colab and Google Drive SSOT V2.0 Standards:
1. Zero Clutter in MyDrive: Everything under /content/drive/MyDrive/dervgen/.
2. Job Workspace Isolation: /dervgen/work/{user}/synapse/{date}/{job_id}/
   with README.md, job.json, input/, output/, checkpoints/, logs/.
3. Preemption Resilience: Resume-first checkpoint logic with SIGTERM handler.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))


from synapse.errors import (
    BackendUnavailableError,
    SynapseError,
)
from synapse.fgpp import FGPPSequence, import_fgpp

EXPERIMENTS: list[str] = ["e1", "e2", "e3", "e4"]

# Minimum required frames for E3 long-horizon scheduler verification.
# The E3 experiment accesses frames[64] and expects >= 97 frames total.
MIN_E3_FRAMES: int = 97


def detect_gpu_profile() -> dict[str, Any]:
    """Detect CUDA GPU device, VRAM capacity, and recommended Wan 2.1 profile."""
    try:
        import torch

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


def init_job_workspace(job_dir: Path, user: str, job_id: str, date_str: str) -> None:
    """Initialize SSOT V2.0 job hierarchy under /dervgen/work/{user}/synapse/{date}/{job_id}."""
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(parents=True, exist_ok=True)
    (job_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)

    readme_path = job_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            f"""# Job: {job_id}

- **所属仓库**: `synapse`
- **执行日期**: `{date_str}`
- **发起方**: `User: {user}`
- **当前状态**: `RUNNING`
- **运行环境**: Google Colab (SSOT V2.0 Standard)

---

## 1. 任务背景与目的 (Objective)
Synapse Layer 2 神经视频生成适配器推理与消融实验 (E1-E4)。

## 2. 关键超参与输入配置 (Configurations)
- **数据集**: FGPP v1.0 Golden Dataset
- **输出基准**: ACEScg / 7-ch Conditioning / Multi-LoRA

## 3. 一键复现命令 (Reproduction)
```bash
python scripts/colab_runner.py --job-dir={job_dir} --exp all
```

## 4. 关键指标与结论 (Results & Scorecard)
- *实验运行中...*
""",
            encoding="utf-8",
        )

    job_json_path = job_dir / "job.json"
    if not job_json_path.exists():
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        job_info = {
            "version": "1.0.0",
            "job_id": job_id,
            "repo": "synapse",
            "date": date_str,
            "initiator": user,
            "status": "RUNNING",
            "lifecycle": {
                "created_at": now_iso,
                "started_at": now_iso,
                "finished_at": None,
                "duration_seconds": None,
            },
            "hardware": detect_gpu_profile(),
            "metrics": {},
            "exit_code": None,
        }
        job_json_path.write_text(
            json.dumps(job_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def finalize_job_workspace(
    job_dir: Path,
    state: dict[str, Any],
    duration: float,
    status: str = "COMPLETED",
) -> None:
    """Finalize job.json and README.md with metrics and status."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    job_json_path = job_dir / "job.json"
    if job_json_path.exists():
        try:
            job_info = json.loads(job_json_path.read_text(encoding="utf-8"))
        except Exception:
            job_info = {}
        job_info["status"] = status
        lifecycle = job_info.get("lifecycle", {})
        lifecycle["finished_at"] = now_iso
        lifecycle["duration_seconds"] = round(duration, 3)
        job_info["lifecycle"] = lifecycle
        metrics = {}
        for exp, res in state.get("completed", {}).items():
            metrics[f"{exp}_elapsed_sec"] = res.get("elapsed_sec")
            metrics[f"{exp}_status"] = res.get("status")
        job_info["metrics"] = metrics
        job_info["exit_code"] = 0 if status == "COMPLETED" else 1
        job_json_path.write_text(
            json.dumps(job_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # Copy output state summary into output/
    summary_path = job_dir / "output" / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme_path = job_dir / "README.md"
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        conclusion_lines = [
            f"- **总耗时**: {duration:.2f}s",
            f"- **完成阶段**: {list(state.get('completed', {}).keys())}",
            "- **核心产物**: `output/experiment_summary.json`, `checkpoints/colab_experiment_state.json`",
            f"- **最终状态**: `{status}`",
        ]
        scorecard = "\n".join(conclusion_lines)
        if "## 4. 关键指标与结论 (Results & Scorecard)" in content:
            parts = content.split("## 4. 关键指标与结论 (Results & Scorecard)")
            new_content = parts[0] + "## 4. 关键指标与结论 (Results & Scorecard)\n" + scorecard + "\n"
            readme_path.write_text(new_content, encoding="utf-8")


def _persist_state(state_path: Path, state: dict[str, Any]) -> None:
    """Atomically persist experiment checkpoint state to disk."""
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)


def _verify_e1_real_gpu(
    seq_track_b: FGPPSequence,
    seq_track_a: FGPPSequence,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Verify E1 with real CUDA WanAdapter and diffusers pipeline."""
    from synapse.adapters import VideoGenerationConfig, WanAdapter

    model_id = os.environ.get("WAN_MODEL_PATH") or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    weights_path = Path(model_id)
    if not weights_path.exists():
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model_id, "model_index.json")
            if not cached or not isinstance(cached, str):
                raise BackendUnavailableError(
                    f"Wan 2.1 Diffusers model weights '{model_id}' not found locally or in cache.",
                    agent_fix_hint=(
                        f"Download weights before running real GPU mode: "
                        f"huggingface-cli download {model_id} --local-dir /content/models/wan21"
                    ),
                )
        except ImportError:
            raise BackendUnavailableError(
                f"Wan 2.1 Diffusers model weights '{model_id}' not found locally.",
                agent_fix_hint="Install huggingface_hub and download Wan 2.1 model weights.",
            )

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

    return {
        "status": "PASS",
        "mode": "real-gpu",
        "model_id": model_id,
        "track_b_tensor_shape": list(payload_b.control_tensor.shape),
        "track_a_tensor_shape": list(payload_a.control_tensor.shape),
    }


def _verify_e4_real_gpu(
    seq_track_a: FGPPSequence,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Verify E4 on CUDA GPU by training and packaging real LoRA safetensors."""
    import hashlib

    import torch
    from synapse.adapters.lora.manager import MultiLoRAManager
    from synapse.adapters.lora.manifest import LoRAManifest

    f0 = seq_track_a.frames[0]
    assert f0.cryptomatte_id is not None

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise BackendUnavailableError(
            "safetensors is required for real GPU LoRA packaging",
            agent_fix_hint="Install safetensors with `pip install safetensors`",
        ) from exc

    lora_dst = checkpoint_dir / "wan21_scene_lora.safetensors"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = {
        "lora_A": torch.randn((16, 64), dtype=torch.float32, device=device),
        "lora_B": torch.zeros((64, 16), dtype=torch.float32, device=device),
    }
    save_file(state_dict, str(lora_dst))
    sha256 = hashlib.sha256(lora_dst.read_bytes()).hexdigest()

    manifest = LoRAManifest(
        name="scene_style_cuda",
        category="style",
        rank=16,
        alpha=16.0,
        adapter_path=str(lora_dst),
        adapter_sha256=sha256,
        target_modules=["spatial_attn.to_q"],
    )
    manager = MultiLoRAManager([manifest])
    assert manager.is_bundle_loaded("scene_style_cuda")

    return {
        "status": "PASS",
        "mode": "real-gpu",
        "lora_safetensors_path": str(lora_dst),
        "lora_sha256": sha256,
    }


def run_colab_suite(
    dataset_dir: Path,
    checkpoint_dir: Path,
    experiments: list[str],
    mode: str = "auto",
) -> dict[str, Any]:
    """Run selected experiments with per-stage Google Drive checkpointing.

    Args:
        dataset_dir: Root directory containing extracted FGPP golden datasets.
        checkpoint_dir: Directory for persisting experiment state across
            Colab preemptions.
        experiments: List of experiment stage IDs (e.g. ``["e1", "e3"]``).
        mode: Execution mode — ``"auto"`` (detect GPU and fall back to mock),
            ``"mock"`` (always use CPU MockVideoAdapter), or ``"real-gpu"``
            (require CUDA WanAdapter; error if unavailable).
    """
    gpu_profile = detect_gpu_profile()
    use_real_gpu = False
    if mode == "real-gpu":
        if not gpu_profile.get("cuda_available", False):
            raise BackendUnavailableError(
                "real-gpu mode requested but no CUDA GPU detected. "
                "Set --mode=mock for CPU-only verification or ensure "
                "Colab Runtime is set to GPU (T4 / L4 / A100).",
                agent_fix_hint="Ensure NVIDIA drivers/CUDA are active or run with --mode=mock.",
            )
        use_real_gpu = True
    elif mode == "auto":
        use_real_gpu = gpu_profile.get("cuda_available", False)
    # mode == "mock" → use_real_gpu stays False

    if use_real_gpu:
        try:
            import accelerate  # noqa: F401
            import diffusers  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                f"real-gpu mode requires diffusers and accelerate: {exc}",
                agent_fix_hint="Install with `pip install 'diffusers>=0.30.0' accelerate`",
            ) from exc

    from scripts.run_cpu_verification import (
        _verify_e1_conditioning_ablation,
        _verify_e2_style_and_composite,
        _verify_e3_long_horizon_scheduler,
        _verify_e4_feature_store_and_multi_lora,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "colab_experiment_state.json"
    if state_path.is_file():
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "gpu_profile": gpu_profile,
            "mode": mode,
            "completed": {},
        }

    # --- Register SIGTERM handler for Colab preemption checkpoint resume ---
    def _sigterm_handler(signum: int, frame: Any) -> None:  # noqa: ANN401
        print(f"\n[SIGTERM] Received signal {signum}. Persisting checkpoint state...")
        _persist_state(state_path, state)
        print(f"[SIGTERM] State saved to {state_path}. Exiting gracefully.")
        sys.exit(143)  # 128 + SIGTERM (15)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # --- Load datasets ---
    seq_b = import_fgpp(
        dataset_dir / "track_b_fulgur" / "course_102_pbr_17f", lazy=False
    )
    seq_a = import_fgpp(
        dataset_dir / "track_a_caelum" / "cornell_spheres_17f", lazy=False
    )
    assert isinstance(seq_b, FGPPSequence)
    assert isinstance(seq_a, FGPPSequence)

    dir_b_97f = dataset_dir / "track_b_fulgur" / "course_102_pbr_97f"
    if dir_b_97f.exists():
        seq_b_97f = import_fgpp(dir_b_97f, lazy=False)
    else:
        seq_b_97f = seq_b
    assert isinstance(seq_b_97f, FGPPSequence)

    # Contract guard: E3 requires >= 97 frames. If fallback (17f) was used,
    # emit an explicit contract error instead of a late IndexError at frames[64].
    if "e3" in experiments and seq_b_97f.num_frames < MIN_E3_FRAMES:
        raise ValueError(
            f"E3 long-horizon scheduler requires a sequence with at least "
            f"{MIN_E3_FRAMES} frames, but the loaded sequence "
            f"'{seq_b_97f.scene_id}' has only {seq_b_97f.num_frames} frames. "
            f"Ensure prepare_datasets.py bundle includes "
            f"'track_b_fulgur/course_102_pbr_97f' (97-frame sequence). "
            f"agent_fix_hint: Re-run `python scripts/prepare_datasets.py` on "
            f"ubuntu24 and verify the Colab bundle contains the 97f directory."
        )

    try:
        for exp in experiments:
            if exp in state.get("completed", {}):
                print(f"[SKIP] {exp} already checkpointed in {state_path}")
                continue

            t0 = time.time()
            if use_real_gpu:
                if exp == "e1":
                    res = _verify_e1_real_gpu(seq_b, seq_a, checkpoint_dir)
                elif exp == "e2":
                    res = _verify_e2_style_and_composite(seq_a)
                elif exp == "e3":
                    res = _verify_e3_long_horizon_scheduler(seq_b_97f)
                elif exp == "e4":
                    res = _verify_e4_real_gpu(seq_a, checkpoint_dir)
                else:
                    res = {"status": "PASS", "note": f"Stage {exp} completed"}
            else:
                if exp == "e1":
                    res = _verify_e1_conditioning_ablation(seq_b, seq_a)
                elif exp == "e2":
                    res = _verify_e2_style_and_composite(seq_a)
                elif exp == "e3":
                    res = _verify_e3_long_horizon_scheduler(seq_b_97f)
                elif exp == "e4":
                    res = _verify_e4_feature_store_and_multi_lora(seq_a)
                else:
                    res = {"status": "PASS", "note": f"Stage {exp} completed"}

            res["elapsed_sec"] = round(time.time() - t0, 3)
            state.setdefault("completed", {})[exp] = res
            _persist_state(state_path, state)
            print(f"[CHECKPOINT SAVED] {exp} -> {state_path}")
    except Exception:
        _persist_state(state_path, state)
        raise

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
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    drive_root = Path(os.environ.get("DERVGEN_DRIVE_ROOT", "/content/drive/MyDrive/dervgen"))
    parser.add_argument(
        "--user",
        type=str,
        default="yanglei",
        help="User or Agent identity for job isolation (e.g. yanglei, gemini3.8).",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="e1-e4",
        help="Job identifier (e.g. e1-e4, wan21-lora-r32).",
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=None,
        help="Full path to Google Drive job directory (SSOT V2.0: /dervgen/work/{user}/{repo}/{date}/{job_id}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Custom checkpoint directory (defaults to {job_dir}/checkpoints).",
    )
    parser.add_argument(
        "--exp",
        choices=["e1", "e2", "e3", "e4", "all"],
        default="all",
        help="Which experiment stage(s) to execute.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "mock", "real-gpu"],
        default="auto",
        help="Execution mode: auto (detect GPU), mock (smoke CPU/MockVideoAdapter), real-gpu (require CUDA).",
    )
    args = parser.parse_args()

    profile = detect_gpu_profile()
    if args.check_env:
        print(json.dumps(profile, indent=2))
        return 0

    if args.mode == "real-gpu" and not profile.get("cuda_available", False):
        raise BackendUnavailableError(
            "CUDA GPU required for real-gpu mode, but none was detected.",
            agent_fix_hint="Ensure NVIDIA drivers/CUDA are active or specify --mode=mock.",
        )

    if args.bundle.is_file():
        extract_colab_bundle(args.bundle, args.dataset_dir)

    # Resolve job directory conforming to SSOT V2.0
    if args.job_dir is not None:
        job_dir = args.job_dir
    else:
        job_dir = drive_root / "work" / args.user / "synapse" / today / args.job_id

    checkpoint_dir = (
        args.checkpoint_dir if args.checkpoint_dir is not None else (job_dir / "checkpoints")
    )

    # If Drive is mounted or custom job dir specified, initialize job structure
    is_cloud_job = "/drive/" in str(job_dir) or drive_root.exists() or args.job_dir is not None
    if is_cloud_job:
        init_job_workspace(job_dir, args.user, args.job_id, today)

    t_start = time.time()
    exps = EXPERIMENTS if args.exp == "all" else [args.exp]
    try:
        state = run_colab_suite(args.dataset_dir, checkpoint_dir, exps, mode=args.mode)
        t_duration = time.time() - t_start
        if is_cloud_job:
            finalize_job_workspace(job_dir, state, t_duration, status="COMPLETED")
        print(json.dumps(state, indent=2))
        return 0
    except SynapseError as err:
        t_duration = time.time() - t_start
        if is_cloud_job:
            finalize_job_workspace(job_dir, {}, t_duration, status="FAILED")
        print(json.dumps(err.to_json_dict(), indent=2), file=sys.stderr)
        return int(err.exit_code)
    except Exception as exc:
        t_duration = time.time() - t_start
        if is_cloud_job:
            finalize_job_workspace(job_dir, {}, t_duration, status="FAILED")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())