# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for PR-32: CLI synapse recon invert-video + pipeline stage."""

from __future__ import annotations

import io
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from synapse.cli import main_synapse
from synapse.pipeline.bundle_worker import (
    BundleManifest,
    JobBundleWorker,
    get_registered_stages,
)


def _make_parallax_video(num_frames: int = 4, h: int = 64, w: int = 64) -> np.ndarray:
    """Generate multi-view video frames of a 3D object with genuine angular parallax."""
    rng = np.random.default_rng(42)
    N = 400
    pts = rng.uniform(-0.35, 0.35, (N, 3))
    colors = rng.integers(150, 255, (N, 3), dtype=np.uint8)
    frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0
    for i in range(num_frames):
        theta = math.radians(-20.0 + i * (40.0 / (num_frames - 1)))
        cam_pos = np.array([2.0 * math.sin(theta), 0.0, 2.0 * math.cos(theta)])
        z_axis = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        t = -R @ cam_pos
        img = np.full((h, w, 3), 20, dtype=np.uint8)
        pts_cam = (R @ pts.T).T + t
        for p, c in zip(pts_cam, colors, strict=False):
            if p[2] > 0.1:
                u = int(cx + fx * p[0] / p[2])
                v = int(cy + fy * p[1] / p[2])
                if 1 <= u < w - 2 and 1 <= v < h - 2:
                    img[v - 1 : v + 2, u - 1 : u + 2] = c
        frames[i] = img
    return frames


# ---------------------------------------------------------------------------
# Tests: CLI synapse recon invert-video
# ---------------------------------------------------------------------------


def test_recon_invert_video_cli_json() -> None:
    """synapse recon invert-video --json without --video returns AssetNotFoundError(20)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code = main_synapse(
            [
                "--json",
                "recon",
                "invert-video",
                "--output",
                tmpdir,
            ]
        )
    # M13 strict contracts: missing video input → exit_code 20
    assert exit_code == 20


@pytest.mark.slow
def test_recon_invert_video_cli_with_video() -> None:
    """synapse recon invert-video works with actual video .npy file with real parallax."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = _make_parallax_video(4, 64, 64)
        video_path = Path(tmpdir) / "test_video.npy"
        np.save(str(video_path), video)

        exit_code = main_synapse(
            [
                "--json",
                "recon",
                "invert-video",
                "--video",
                str(video_path),
                "--output",
                str(Path(tmpdir) / "output"),
                "--target-faces",
                "500",
            ]
        )
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


@pytest.mark.slow
def test_video_to_3d_invert_stage_execution() -> None:
    """video_to_3d_invert stage executes with valid video_frames input with real parallax."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = _make_parallax_video(4, 64, 64)
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
        result = worker.execute(manifest)

    assert result.exit_code == 0
    assert result.status == "success"


# ---------------------------------------------------------------------------
# Tests: PR-60 — no-input error assertions (AUD-P0-01)
# ---------------------------------------------------------------------------


def test_recon_invert_video_no_video_exit_code_and_hint() -> None:
    """synapse recon invert-video --json without --video returns exit 20 with agent_fix_hint."""
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        exit_code = main_synapse(
            [
                "--json",
                "recon",
                "invert-video",
            ]
        )
    assert exit_code == 20
    output = json.loads(captured.getvalue().strip())
    assert output["status"] == "error"
    assert "agent_fix_hint" in output
    assert output["agent_fix_hint"] != ""


def test_recon_closed_loop_no_video_returns_error() -> None:
    """synapse recon closed-loop --json without --video returns non-zero with agent_fix_hint."""
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        exit_code = main_synapse(
            [
                "--json",
                "recon",
                "closed-loop",
            ]
        )
    assert exit_code != 0
    output = json.loads(captured.getvalue().strip())
    assert output["status"] == "error"
    assert "agent_fix_hint" in output
    assert output["agent_fix_hint"] != ""


def test_recon_closed_loop_nonexistent_video_returns_error() -> None:
    """synapse recon closed-loop --json with bad --video path returns exit 20."""
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        exit_code = main_synapse(
            [
                "--json",
                "recon",
                "closed-loop",
                "--video",
                "/nonexistent/video.npy",
            ]
        )
    assert exit_code == 20
    output = json.loads(captured.getvalue().strip())
    assert output["status"] == "error"
    assert output["error_type"] == "AssetNotFoundError"
