# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for PR-32: CLI synapse recon invert-video + pipeline stage."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from synapse.cli import main_synapse
from synapse.pipeline.bundle_worker import (
    BundleManifest,
    JobBundleWorker,
    get_registered_stages,
)

# ---------------------------------------------------------------------------
# Tests: CLI synapse recon invert-video
# ---------------------------------------------------------------------------


def test_recon_invert_video_cli_json() -> None:
    """synapse recon invert-video --json without --video returns AssetNotFoundError(20)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code = main_synapse([
            "--json", "recon", "invert-video",
            "--output", tmpdir,
        ])
    # M13 strict contracts: missing video input → exit_code 20
    assert exit_code == 20


def test_recon_invert_video_cli_with_video() -> None:
    """synapse recon invert-video works with actual video .npy file.

    Synthetic frames cannot produce real angular separation, so we patch
    the minimum threshold to 0.0° for this unit test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small video array with varying content
        rng = np.random.default_rng(42)
        video = rng.integers(100, 200, (4, 64, 64, 3), dtype=np.uint8)
        video_path = Path(tmpdir) / "test_video.npy"
        np.save(str(video_path), video)

        with patch("synapse.recon.multiview._MIN_ANGULAR_SPAN_DEG", 0.0):
            exit_code = main_synapse([
                "--json", "recon", "invert-video",
                "--video", str(video_path),
                "--output", str(Path(tmpdir) / "output"),
                "--target-faces", "500",
            ])
    assert exit_code == 0


def test_recon_missing_subcommand_json() -> None:
    """synapse recon (no subcommand) --json returns error."""
    exit_code = main_synapse(["--json", "recon"])
    assert exit_code == 10


# ---------------------------------------------------------------------------
# Tests: pipeline stage registration
# ---------------------------------------------------------------------------


def test_video_to_3d_invert_stage_registered() -> None:
    """video_to_3d_invert stage must be in the registry."""
    stages = get_registered_stages()
    assert "video_to_3d_invert" in stages


def test_video_to_3d_invert_stage_execution() -> None:
    """video_to_3d_invert stage executes with valid video_frames input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create valid video frames as .npy file
        rng = np.random.default_rng(42)
        video = rng.integers(100, 200, (4, 64, 64, 3), dtype=np.uint8)
        video_path = Path(tmpdir) / "frames.npy"
        np.save(str(video_path), video)

        manifest = BundleManifest(
            job_id="test-pr32",
            stage="video_to_3d_invert",
            inputs={
                "video_frames": str(video_path),
            },
            params={
                "width": 64,
                "height": 64,
                "num_frames": 4,
                "target_faces": 500,
            },
            outputs={
                "mesh": str(Path(tmpdir) / "mesh.npy"),
                "trajectory": str(Path(tmpdir) / "traj.json"),
            },
        )
        worker = JobBundleWorker()
        with patch("synapse.recon.multiview._MIN_ANGULAR_SPAN_DEG", 0.0):
            result = worker.execute(manifest)

    assert result.exit_code == 0
    assert result.status in ("success", "fallback")


# ---------------------------------------------------------------------------
# Tests: PR-60 — no-input error assertions (AUD-P0-01)
# ---------------------------------------------------------------------------


def test_recon_invert_video_no_video_exit_code_and_hint() -> None:
    """synapse recon invert-video --json without --video returns exit 20 with agent_fix_hint."""
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        exit_code = main_synapse([
            "--json", "recon", "invert-video",
        ])
    assert exit_code == 20
    output = json.loads(captured.getvalue().strip())
    assert output["status"] == "error"
    assert "agent_fix_hint" in output
    assert output["agent_fix_hint"] != ""


def test_recon_closed_loop_no_video_returns_error() -> None:
    """synapse recon closed-loop --json without --video returns non-zero with agent_fix_hint."""
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        exit_code = main_synapse([
            "--json", "recon", "closed-loop",
        ])
    assert exit_code != 0
    output = json.loads(captured.getvalue().strip())
    assert output["status"] == "error"
    assert "agent_fix_hint" in output
    assert output["agent_fix_hint"] != ""


def test_recon_closed_loop_nonexistent_video_returns_error() -> None:
    """synapse recon closed-loop --json with bad --video path returns exit 20."""
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        exit_code = main_synapse([
            "--json", "recon", "closed-loop",
            "--video", "/nonexistent/video.npy",
        ])
    assert exit_code == 20
    output = json.loads(captured.getvalue().strip())
    assert output["status"] == "error"
    assert output["error_type"] == "AssetNotFoundError"
