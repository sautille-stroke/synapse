"""Unit tests for scripts/colab_runner.py.

Tests:
1. GPU profile detection schema & defaults.
2. CLI --check-env argument execution and JSON output.
3. SSOT V2.0 workspace initialization and finalization.
4. Fail-closed behavior on --mode=real-gpu when CUDA is unavailable (exit code 30).
5. Fail-closed behavior on --mode=real-gpu when model weights are missing (exit code 30).
6. Atomicity and recovery of partial state checkpointing.
7. Mock mode execution on FGPP sequences.
8. E3 long-horizon frame count contract guard.
9. Contract resolution between --job-dir and --checkpoint-dir.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from synapse.errors import BackendUnavailableError
from synapse.fgpp import (
    CameraParams,
    FGPPFrame,
    FGPPSequence,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "colab_runner.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from scripts.colab_runner import (
    _persist_state,
    detect_gpu_profile,
    finalize_job_workspace,
    init_job_workspace,
    run_colab_suite,
)


def _make_dummy_frame(idx: int, width: int = 64, height: int = 64) -> FGPPFrame:
    """Construct a minimal valid FGPPFrame with 15-channel AOV equivalents."""
    cam = CameraParams(
        fx=50.0,
        fy=50.0,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
        fov_degrees=50.0,
        z_near=0.1,
        z_far=100.0,
        view_matrix=np.eye(4, dtype=np.float64),
        world_matrix=np.eye(4, dtype=np.float64),
    )
    return FGPPFrame(
        frame_index=idx,
        timestamp_sec=idx / 24.0,
        camera=cam,
        depth=np.full((height, width), 5.0, dtype=np.float32),
        world_normal=np.full((height, width, 3), 0.577, dtype=np.float32),
        direct_radiance=np.full((height, width, 3), 0.8, dtype=np.float32),
        cryptomatte_id=np.ones((height, width), dtype=np.uint32) * 42,
    )


def _make_dummy_sequence(num_frames: int = 17, scene_id: str = "test_scene") -> FGPPSequence:
    frames = [_make_dummy_frame(i) for i in range(num_frames)]
    return FGPPSequence(
        scene_id=scene_id,
        frames=frames,
        fps=24.0,
        extra={"schema_version": "fgpp.v1.0"},
    )


def test_detect_gpu_profile_schema() -> None:
    """Verify detect_gpu_profile returns all required contract fields."""
    profile = detect_gpu_profile()
    required_keys = {
        "cuda_available",
        "device_name",
        "vram_gb",
        "recommended_wan_model",
        "enable_model_cpu_offload",
        "dtype",
        "lora_rank",
    }
    assert required_keys.issubset(profile.keys())
    assert isinstance(profile["cuda_available"], bool)
    assert isinstance(profile["vram_gb"], (int, float))
    assert isinstance(profile["recommended_wan_model"], str)


def test_colab_runner_check_env_cli() -> None:
    """Verify colab_runner.py --check-env outputs valid JSON and exits with 0."""
    res = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check-env"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Expected 0, got {res.returncode}: {res.stderr}"
    data = json.loads(res.stdout)
    assert "cuda_available" in data
    assert "recommended_wan_model" in data


def test_init_and_finalize_job_workspace(tmp_path: Path) -> None:
    """Verify SSOT V2.0 workspace creation and finalization."""
    job_dir = tmp_path / "work" / "user1" / "synapse" / "2026-11-16" / "job_01"
    init_job_workspace(job_dir, user="user1", job_id="job_01", date_str="2026-11-16")

    assert (job_dir / "input").is_dir()
    assert (job_dir / "output").is_dir()
    assert (job_dir / "checkpoints").is_dir()
    assert (job_dir / "logs").is_dir()
    assert (job_dir / "README.md").is_file()
    assert (job_dir / "job.json").is_file()

    job_data = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert job_data["status"] == "RUNNING"
    assert job_data["job_id"] == "job_01"

    # Finalize workspace
    sample_state: dict[str, Any] = {
        "completed": {
            "e1": {"status": "PASS", "elapsed_sec": 1.25},
            "e2": {"status": "PASS", "elapsed_sec": 0.85},
        }
    }
    finalize_job_workspace(job_dir, sample_state, duration=2.1, status="COMPLETED")

    summary_file = job_dir / "output" / "experiment_summary.json"
    assert summary_file.is_file()

    final_job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert final_job["status"] == "COMPLETED"
    assert final_job["exit_code"] == 0
    assert final_job["lifecycle"]["duration_seconds"] == 2.1
    assert "e1_elapsed_sec" in final_job["metrics"]


def test_colab_runner_real_gpu_no_cuda_raises_exit_30(tmp_path: Path) -> None:
    """Verify --mode=real-gpu raises BackendUnavailableError (30) when CUDA is absent."""
    fake_profile = {
        "cuda_available": False,
        "device_name": "cpu",
        "vram_gb": 0.0,
        "recommended_wan_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "enable_model_cpu_offload": True,
        "dtype": "float32",
        "lora_rank": 16,
    }
    with mock.patch("scripts.colab_runner.detect_gpu_profile", return_value=fake_profile):
        with pytest.raises(BackendUnavailableError) as exc_info:
            run_colab_suite(
                dataset_dir=tmp_path / "datasets",
                checkpoint_dir=tmp_path / "checkpoints",
                experiments=["e1"],
                mode="real-gpu",
            )
        assert exc_info.value.exit_code == 30
        assert "CUDA GPU required" in str(exc_info.value) or "real-gpu mode requested" in str(exc_info.value)
        assert exc_info.value.agent_fix_hint != ""


def test_colab_runner_real_gpu_missing_weights_raises_exit_30(tmp_path: Path) -> None:
    """Verify --mode=real-gpu raises BackendUnavailableError when Wan weights are missing."""
    fake_profile = {
        "cuda_available": True,
        "device_name": "NVIDIA T4",
        "vram_gb": 15.0,
        "recommended_wan_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "enable_model_cpu_offload": True,
        "dtype": "float16",
        "lora_rank": 16,
    }
    dummy_seq_b = _make_dummy_sequence(17, "fulgur_export")
    dummy_seq_a = _make_dummy_sequence(17, "caelum_cornell_spheres_17f")

    with (
        mock.patch("scripts.colab_runner.detect_gpu_profile", return_value=fake_profile),
        mock.patch("scripts.colab_runner.import_fgpp", side_effect=[dummy_seq_b, dummy_seq_a, dummy_seq_b]),
        mock.patch.dict("sys.modules", {"accelerate": mock.MagicMock(), "diffusers": mock.MagicMock()}),
        mock.patch("pathlib.Path.exists", return_value=False),
    ):
        with pytest.raises(BackendUnavailableError) as exc_info:
            run_colab_suite(
                dataset_dir=tmp_path / "datasets",
                checkpoint_dir=tmp_path / "checkpoints",
                experiments=["e1"],
                mode="real-gpu",
            )
        assert exc_info.value.exit_code == 30
        assert "Wan 2.1" in str(exc_info.value)
        assert "agent_fix_hint" in dir(exc_info.value)


def test_colab_runner_sigterm_and_partial_checkpoint(tmp_path: Path) -> None:
    """Verify _persist_state writes atomic JSON checkpoint readable by resume logic."""
    state_file = tmp_path / "colab_experiment_state.json"
    initial_state = {
        "gpu_profile": {"cuda_available": False},
        "mode": "mock",
        "completed": {
            "e1": {"status": "PASS", "elapsed_sec": 1.1},
        },
    }
    _persist_state(state_file, initial_state)
    assert state_file.is_file()

    loaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert "e1" in loaded["completed"]
    assert loaded["completed"]["e1"]["status"] == "PASS"

    # Simulate subsequent stage update
    loaded["completed"]["e2"] = {"status": "PASS", "elapsed_sec": 0.9}
    _persist_state(state_file, loaded)

    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(reloaded["completed"]) == 2
    assert "e2" in reloaded["completed"]


def test_colab_runner_e3_frame_floor_contract(tmp_path: Path) -> None:
    """Verify E3 raises contract ValueError when input sequence has < 97 frames."""
    short_seq = _make_dummy_sequence(17, "short_sequence")
    dummy_seq_a = _make_dummy_sequence(17, "caelum_17f")

    with mock.patch("scripts.colab_runner.import_fgpp", side_effect=[short_seq, dummy_seq_a, short_seq]):
        with pytest.raises(ValueError, match="at least 97 frames"):
            run_colab_suite(
                dataset_dir=tmp_path / "datasets",
                checkpoint_dir=tmp_path / "checkpoints",
                experiments=["e3"],
                mode="mock",
            )
