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


def _make_parallax_video(num_frames: int = 8, h: int = 96, w: int = 96) -> np.ndarray:
    """Generate multi-view video frames of an opaque textured 3D sphere with genuine parallax."""
    frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
    fx = fy = float(w) * 0.8
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = (xx - cx) / fx
    dy = (yy - cy) / fy
    dz = np.ones_like(dx)
    dn = np.sqrt(dx * dx + dy * dy + dz * dz)
    rd_cam = np.stack([dx / dn, dy / dn, dz / dn], axis=-1)

    for i in range(num_frames):
        theta = math.radians(-25.0 + i * (50.0 / max(num_frames - 1, 1)))
        cam_pos = np.array(
            [2.2 * math.sin(theta), 0.25 * math.sin(2.0 * theta), 2.2 * math.cos(theta)]
        )
        z_axis = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=0)

        # Ray-sphere intersection for opaque textured sphere (radius = 0.85)
        rd = np.einsum("ji,hwj->hwi", R, rd_cam)
        ro = cam_pos
        b = np.sum(rd * ro, axis=-1)
        c = float(np.dot(ro, ro)) - 0.85**2
        disc = b * b - c
        hit = disc > 0.0
        t_hit = np.where(hit, -b - np.sqrt(np.maximum(disc, 0.0)), 1e9)
        p = ro + rd * t_hit[..., None]

        tex_r = (
            128.0
            + 60.0 * np.sin(14.0 * p[..., 0]) * np.cos(14.0 * p[..., 1])
            + 50.0 * np.sin(23.0 * p[..., 2] + 5.0 * p[..., 0])
        )
        tex_g = (
            128.0
            + 60.0 * np.cos(16.0 * p[..., 1]) * np.sin(15.0 * p[..., 2])
            + 50.0 * np.cos(21.0 * p[..., 0] - 7.0 * p[..., 1])
        )
        tex_b = (
            128.0
            + 60.0 * np.sin(18.0 * p[..., 2]) * np.cos(13.0 * p[..., 0])
            + 50.0 * np.sin(19.0 * p[..., 1] + 9.0 * p[..., 2])
        )
        img = np.full((h, w, 3), 20, dtype=np.uint8)
        img[hit, 0] = np.clip(tex_r[hit], 30.0, 250.0).astype(np.uint8)
        img[hit, 1] = np.clip(tex_g[hit], 30.0, 250.0).astype(np.uint8)
        img[hit, 2] = np.clip(tex_b[hit], 30.0, 250.0).astype(np.uint8)
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
        video = _make_parallax_video()
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
        video = _make_parallax_video()
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
