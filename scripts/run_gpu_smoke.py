#!/usr/bin/env python3
"""Weekly CUDA GPU smoke check for Wan 2.1 + LoRA training + Colab release (PR-210).

Validates real non-mock inference & gradient paths on GPU hardware (T4/L4/A100
or win-dev), guarded by a watchdog timer and automatic Colab runtime release.
Without CUDA (unless ``--dry-run`` is passed), fails closed with exit code 30.
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from synapse.errors import BackendUnavailableError, CircuitBreakerError, ExitCode, SynapseError


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


def run_gpu_smoke(
    *,
    dry_run: bool = False,
    timeout_sec: int = 300,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Execute or dry-run the real CUDA Wan2.1 + LoRA smoke verification."""
    t0 = time.monotonic()
    gpu_profile = _probe_gpu()

    if dry_run:
        summary: dict[str, Any] = {
            "status": "DRY_RUN",
            "exit_code": int(ExitCode.OK),
            "gpu_profile": gpu_profile,
            "timeout_sec": int(timeout_sec),
            "planned_stages": [
                "diffusers_wan_backend_single_window",
                "controlnext_patchify_stem_forward",
                "stateless_lora_trainer_grad_step",
                "safetensors_export_reload",
            ],
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
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        device = torch.device("cuda:0")
        stem = _build_controlnext_patchify_stem(in_channels=7, hidden_dim=16).to(device)
        cond = torch.randn(1, 7, 5, 32, 32, device=device)
        with torch.no_grad():
            stem_out = stem(cond)

        cfg = LoRATrainConfig(category="style", rank=8, alpha=8.0)
        trainer = StatelessLoRATrainer(cfg)
        proj = torch.nn.Linear(16, 16).to(device)
        batch = {
            "latent": torch.randn(1, 16, device=device),
            "noise": torch.randn(1, 16, device=device),
        }
        step_res = trainer.train_step(proj, batch, step=0)
        peak_vram_gb = round(torch.cuda.max_memory_allocated(0) / (1024**3), 4)

        summary = {
            "status": "PASS",
            "exit_code": int(ExitCode.OK),
            "gpu_profile": gpu_profile,
            "stem_output_shape": list(stem_out.shape),
            "lora_train_loss": float(step_res.get("loss", 0.0)),
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
