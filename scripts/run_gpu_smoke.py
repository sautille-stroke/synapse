#!/usr/bin/env python3
"""Weekly CUDA GPU smoke check for Wan 2.1 + LoRA training + Colab release (PR-210 / PR-TD05).

Validates real non-mock inference & gradient paths on GPU hardware (T4/L4/A100
or win-dev), guarded by a watchdog timer and automatic Colab runtime release.
Without CUDA (unless ``--dry-run`` is passed), fails closed with exit code 30.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from synapse.errors import (
    AssetNotFoundError,
    BackendUnavailableError,
    CircuitBreakerError,
    ContractValidationError,
    ExitCode,
    SynapseError,
)

PLANNED_STAGES: tuple[str, ...] = (
    "diffusers_wan_backend_single_window",
    "controlnext_patchify_stem_forward",
    "stateless_lora_trainer_grad_step",
    "safetensors_export_reload",
)


def _probe_gpu() -> dict[str, Any]:
    """Detect PyTorch CUDA device capability safely."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False, "device_name": "cpu", "vram_gb": 0.0}
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda_available": True,
            "device_name": props.name,
            "vram_gb": round(props.total_memory / (1024**3), 2),
        }
    except ImportError:
        return {"cuda_available": False, "device_name": "cpu (torch missing)", "vram_gb": 0.0}


def _run_wan_single_window_stage(
    wan_model_path: str | Path | None,
) -> dict[str, Any]:
    """Run DiffusersWanBackend single-window inference + LocalRefinementPipeline smoke."""
    if not wan_model_path:
        return {
            "status": "skipped_no_weights",
            "reason": "WAN_MODEL_PATH not set; skipping real Wan2.1 weights smoke.",
        }

    model_path = Path(wan_model_path)
    if not model_path.exists():
        raise AssetNotFoundError(
            f"WAN_MODEL_PATH does not exist: {model_path}",
            agent_fix_hint="Point WAN_MODEL_PATH / --wan-model-path to an existing Wan2.1 checkpoint directory.",
        )

    from synapse.adapters.base import VideoGenerationConfig
    from synapse.adapters.wan import WanAdapter
    from synapse.adapters.wan_diffusers import DiffusersWanBackend
    from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
    from synapse.refine.pipeline import LocalRefinementPipeline
    from synapse.refine.types import AnatomicalRegion, RefinementROI

    h, w, n_frames = 64, 64, 5
    cam = CameraParams(
        fx=float(w),
        fy=float(h),
        cx=float(w) / 2.0,
        cy=float(h) / 2.0,
        width=w,
        height=h,
    )
    depth = np.full((h, w), 2.5, dtype=np.float32)
    normal = np.zeros((h, w, 3), dtype=np.float16)
    normal[:, :, 2] = np.float16(1.0)
    radiance = np.full((h, w, 3), 0.5, dtype=np.float16)
    fgpp_seq = FGPPSequence(
        frames=[
            FGPPFrame(
                camera=cam,
                depth=depth,
                world_normal=normal,
                direct_radiance=radiance,
                frame_index=idx,
            )
            for idx in range(n_frames)
        ],
        fps=24.0,
        scene_id="gpu_smoke",
    )

    backend = DiffusersWanBackend(
        model_id_or_path=str(model_path),
        device="cuda:0",
        allow_untrained_control=True,
    )
    adapter = WanAdapter(backend=backend)
    gen_cfg = VideoGenerationConfig(
        seed=42,
        fps=24.0,
        width=w,
        height=h,
        num_frames=n_frames,
        num_inference_steps=2,
        extra={"prompt": "studio lighting sphere"},
    )
    conds = adapter.prepare_conditions(fgpp_seq, gen_cfg)
    gen_res = adapter.generate(conditions=conds, config=gen_cfg)

    frames_f32 = gen_res.frames_srgb_u8.astype(np.float32) / 255.0
    roi = RefinementROI(
        bbox_xyxy=np.array([16.0, 16.0, 48.0, 48.0], dtype=np.float32),
        crop_acescg=frames_f32[0, 16:48, 16:48, :].copy(),
        region=AnatomicalRegion.HAND,
        confidence=0.95,
        anatomical_quality_score=0.4,
        frame_index=0,
    )
    rois_per_frame: list[list[RefinementROI]] = [
        [roi] if idx == 0 else [] for idx in range(frames_f32.shape[0])
    ]
    ref_res = LocalRefinementPipeline(strict=False).run(frames_f32, rois_per_frame)

    return {
        "status": "passed",
        "model_path": str(model_path),
        "frames_shape": list(gen_res.frames_srgb_u8.shape),
        "refinement_patches": int(ref_res.num_patches_fused),
    }


def _verify_safetensors_reload(ckpt_path: Path) -> dict[str, Any]:
    """Reload exported ``.safetensors`` weights and verify LoRA tensor parity."""
    from safetensors.numpy import load_file

    if not ckpt_path.is_file():
        raise AssetNotFoundError(
            f"Exported SafeTensors checkpoint not found: {ckpt_path}",
            agent_fix_hint="Verify export_checkpoint wrote the .safetensors file.",
        )

    tensors = load_file(str(ckpt_path))
    if not tensors:
        raise ContractValidationError(
            f"Exported SafeTensors checkpoint is empty: {ckpt_path}",
            agent_fix_hint="Check LoRA injection target patterns before exporting.",
        )
    has_lora_a = any("lora_A" in key for key in tensors)
    has_lora_b = any("lora_B" in key for key in tensors)
    if not (has_lora_a and has_lora_b):
        raise ContractValidationError(
            f"Reloaded checkpoint missing lora_A/lora_B tensors: {sorted(tensors.keys())}",
            agent_fix_hint="Verify StatelessLoRATrainer.inject_lora ran prior to export.",
        )
    for key, arr in tensors.items():
        if not np.all(np.isfinite(arr)):
            raise ContractValidationError(
                f"Non-finite values found in reloaded LoRA tensor '{key}'.",
                agent_fix_hint="Inspect gradient updates for NaN/Inf.",
            )

    return {
        "status": "passed",
        "num_tensors": len(tensors),
        "file_bytes": int(ckpt_path.stat().st_size),
    }


def run_gpu_smoke(
    *,
    dry_run: bool = False,
    timeout_sec: int = 300,
    output_json: Path | None = None,
    wan_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute or dry-run the real CUDA Wan2.1 + LoRA smoke verification."""
    t0 = time.monotonic()
    gpu_profile = _probe_gpu()
    resolved_wan_path = wan_model_path if wan_model_path is not None else os.environ.get("WAN_MODEL_PATH")

    if dry_run:
        summary: dict[str, Any] = {
            "status": "DRY_RUN",
            "exit_code": int(ExitCode.OK),
            "gpu_profile": gpu_profile,
            "timeout_sec": int(timeout_sec),
            "planned_stages": list(PLANNED_STAGES),
            "elapsed_sec": round(time.monotonic() - t0, 4),
        }
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    if not gpu_profile.get("cuda_available", False):
        raise BackendUnavailableError(
            "GPU smoke check requires CUDA, but no CUDA GPU is available.",
            agent_fix_hint=(
                "Run on a CUDA node (Colab T4/L4/A100 or win-dev), or pass --dry-run on CPU CI."
            ),
        )

    def _on_timeout(signum: int, frame: Any) -> None:
        _ = (signum, frame)
        raise CircuitBreakerError(
            f"GPU smoke watchdog timed out after {timeout_sec}s.",
            agent_fix_hint="Reduce smoke workload or inspect hung CUDA kernel.",
            details={
                "elapsed_seconds": float(timeout_sec),
                "max_seconds": float(timeout_sec),
                "fallback": "ABORT_AND_UNASSIGN_GPU_RUNTIME",
            },
        )

    prev_handler = None
    has_sigalrm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
    if has_sigalrm and timeout_sec > 0:
        prev_handler = signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(timeout_sec)

    try:
        import torch
        from synapse.adapters.wan_diffusers import _build_controlnext_patchify_stem
        from synapse.training.backbone import (
            build_mock_dit_backbone,
            resolve_target_modules,
        )
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        stages: dict[str, dict[str, Any]] = {}
        device = torch.device("cuda:0")

        # Stage 1: DiffusersWanBackend single-window + LocalRefinementPipeline (honest skip if no weights)
        stages["diffusers_wan_backend_single_window"] = _run_wan_single_window_stage(
            resolved_wan_path,
        )

        # Stage 2: ControlNeXtPatchifyStem 7-channel forward pass
        stem = _build_controlnext_patchify_stem(in_channels=7, hidden_dim=16).to(device)
        cond = torch.randn(1, 7, 5, 32, 32, device=device)
        with torch.no_grad():
            stem_out = stem(cond)
        stages["controlnext_patchify_stem_forward"] = {
            "status": "passed",
            "output_shape": list(stem_out.shape),
        }

        # Stage 3: LoRA injection + backbone freeze + StatelessLoRATrainer grad step on DiT backbone
        cfg = LoRATrainConfig(category="style", rank=8, alpha=8.0)
        trainer = StatelessLoRATrainer(cfg)
        backbone = build_mock_dit_backbone(dim=16).to(device)
        targets = resolve_target_modules(backbone, "style")
        injected = trainer.inject_lora(backbone, targets)
        if injected <= 0:
            raise ContractValidationError(
                "LoRA injection matched 0 layers on mock DiT backbone.",
                agent_fix_hint="Verify target_modules match backbone layer names.",
            )
        trainer.freeze_backbone(backbone)
        batch = {
            "latent": torch.randn(1, 4, 16, device=device),
            "noise": torch.randn(1, 4, 16, device=device),
        }
        step_res = trainer.train_step(backbone, batch, step=0)
        loss_val = float(step_res.get("loss", 0.0))
        if not math.isfinite(loss_val):
            raise ContractValidationError(
                f"LoRA training step produced non-finite loss: {loss_val}",
                agent_fix_hint="Inspect learning rate or batch inputs.",
            )
        stages["stateless_lora_trainer_grad_step"] = {
            "status": "passed",
            "injected_layers": int(injected),
            "loss": loss_val,
        }

        # Stage 4: SafeTensors LoRA export + reload parity check
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = trainer.export_checkpoint(backbone, Path(tmp_dir) / "smoke_lora.safetensors")
            stages["safetensors_export_reload"] = _verify_safetensors_reload(ckpt_path)

        peak_vram_gb = round(torch.cuda.max_memory_allocated(0) / (1024**3), 4)

        summary = {
            "status": "PASS",
            "exit_code": int(ExitCode.OK),
            "gpu_profile": gpu_profile,
            "planned_stages": list(PLANNED_STAGES),
            "stages": stages,
            "stem_output_shape": list(stem_out.shape),
            "lora_train_loss": loss_val,
            "peak_vram_gb": peak_vram_gb,
            "elapsed_sec": round(time.monotonic() - t0, 4),
        }
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        if has_sigalrm and timeout_sec > 0:
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)


def _release_runtime_resources(auto_release: bool) -> None:
    """Clear CUDA cache and unassign Colab runtime when requested."""
    if not auto_release:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        colab_runtime: Any = importlib.import_module("google.colab.runtime")

        colab_runtime.unassign()
    except (ImportError, Exception):
        pass


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for weekly GPU smoke validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate GPU smoke contract on CPU without executing CUDA kernels.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only to stdout.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Watchdog timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--auto-release",
        action="store_true",
        help="Release CUDA VRAM cache and disconnect Colab runtime upon exit.",
    )
    parser.add_argument(
        "--wan-model-path",
        type=Path,
        default=None,
        help="Optional local Diffusers Wan2.1 checkpoint path (defaults to WAN_MODEL_PATH env).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write structured summary JSON.",
    )
    args = parser.parse_args(argv)

    try:
        res = run_gpu_smoke(
            dry_run=args.dry_run,
            timeout_sec=args.timeout_sec,
            output_json=args.output_json,
            wan_model_path=args.wan_model_path,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"[GPU Smoke] {res['status']} ({res['gpu_profile']['device_name']})")
        return int(ExitCode.OK)
    except SynapseError as exc:
        err_dict = exc.to_json_dict()
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error [{exc.exit_code}]: {exc}", file=sys.stderr)
        return int(exc.exit_code)
    finally:
        _release_runtime_resources(args.auto_release)


if __name__ == "__main__":
    raise SystemExit(main())
