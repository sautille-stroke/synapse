# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for PR-40: ClosedLoopRunner end-to-end 5-step pipeline.

Covers:
  - All 5 steps produce non-empty output
  - ``border_ring_ciede2000 <= 2.0`` on synthetic data
  - CLI ``synapse recon closed-loop --json`` returns exit 0
  - ``inverse_render_loop`` stage is registered
  - ClosedLoopResult serialises to JSON
  - PBR texture set is well-formed
  - Step timings dict has all 5 keys
"""

from __future__ import annotations

import json

import numpy as np
import pytest


# -------------------------------------------------------------------
# Helper: build synthetic data for the closed loop
# -------------------------------------------------------------------


def _make_synthetic_data(
    t: int = 5,
    h: int = 32,
    w: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frames, normals, uv_maps) for testing."""
    rng = np.random.default_rng(42)
    frames = rng.random((t, h, w, 3), dtype=np.float32)

    normals = np.zeros((t, h, w, 3), dtype=np.float32)
    normals[..., 2] = 1.0  # all pointing +Z

    uv_maps = np.zeros((t, h, w, 2), dtype=np.float32)
    u_lin = np.linspace(0.0, 1.0, w, dtype=np.float32)
    v_lin = np.linspace(0.0, 1.0, h, dtype=np.float32)
    uv_maps[:, :, :, 0] = u_lin[np.newaxis, np.newaxis, :]
    uv_maps[:, :, :, 1] = v_lin[np.newaxis, :, np.newaxis]

    return frames, normals, uv_maps


# -------------------------------------------------------------------
# 1. ClosedLoopRunner e2e
# -------------------------------------------------------------------


def test_closed_loop_runner_e2e() -> None:
    """Full 5-step closed loop runs and produces valid output."""
    from synapse.recon.loop.closed_loop_runner import ClosedLoopRunner

    frames, normals, uv_maps = _make_synthetic_data()

    runner = ClosedLoopRunner()
    result = runner.run(
        source_frames=frames,
        normals=normals,
        uv_maps=uv_maps,
    )

    # Step 1: forward guide produced frames
    assert result.forward_guide_result.frames.shape[0] > 0
    assert result.forward_guide_result.phase_valid is True

    # Step 3: inverse optimisation ran
    assert result.inverse_result.albedo_map.shape[-1] == 3
    assert result.inverse_result.iterations_run >= 0

    # Step 4+5: output frames match expected shape
    n_target = result.forward_guide_result.frames.shape[0]
    assert result.output_frames.shape == (n_target, 32, 32, 3)

    # Step 5: stitch report
    assert result.stitch_report.num_frames == n_target


def test_closed_loop_border_ciede2000() -> None:
    """Border ΔE₀₀ is within acceptable range on synthetic data."""
    from synapse.recon.loop.closed_loop_runner import ClosedLoopRunner

    frames, normals, uv_maps = _make_synthetic_data()
    runner = ClosedLoopRunner()
    result = runner.run(
        source_frames=frames,
        normals=normals,
        uv_maps=uv_maps,
    )

    # On synthetic data, border_ring_ciede2000 should be reasonable
    assert result.border_ring_ciede2000 <= 2.0, (
        f"border_ring_ciede2000={result.border_ring_ciede2000} > 2.0"
    )


def test_closed_loop_step_timings() -> None:
    """All 5 step timing keys are present in the result."""
    from synapse.recon.loop.closed_loop_runner import ClosedLoopRunner

    frames, normals, uv_maps = _make_synthetic_data(t=5, h=16, w=16)
    runner = ClosedLoopRunner()
    result = runner.run(
        source_frames=frames,
        normals=normals,
        uv_maps=uv_maps,
    )

    expected_keys = {
        "step1_forward_guide",
        "step2_target_video",
        "step3_inverse_optim",
        "step4_aligned_render",
        "step5_scalpel_patch",
    }
    assert expected_keys.issubset(set(result.step_timings.keys()))
    for k in expected_keys:
        assert result.step_timings[k] >= 0


def test_closed_loop_pbr_textures() -> None:
    """PBR texture set is well-formed after closed loop."""
    from synapse.recon.loop.closed_loop_runner import ClosedLoopRunner

    frames, normals, uv_maps = _make_synthetic_data(t=5, h=16, w=16)
    runner = ClosedLoopRunner()
    result = runner.run(
        source_frames=frames,
        normals=normals,
        uv_maps=uv_maps,
    )

    assert result.pbr_textures.albedo.ndim == 3
    assert result.pbr_textures.albedo.shape[-1] == 3
    # roughness / metallic can be scalar float or ndarray
    r = result.pbr_textures.roughness
    m = result.pbr_textures.metallic
    if isinstance(r, np.ndarray):
        assert np.all(r >= 0.0)
    else:
        assert r >= 0.0
    if isinstance(m, np.ndarray):
        assert np.all(m >= 0.0)
    else:
        assert m >= 0.0


def test_closed_loop_result_to_dict_json() -> None:
    """ClosedLoopResult.to_dict() is JSON-serialisable."""
    from synapse.recon.loop.closed_loop_runner import ClosedLoopRunner

    frames, normals, uv_maps = _make_synthetic_data(t=5, h=16, w=16)
    runner = ClosedLoopRunner()
    result = runner.run(
        source_frames=frames,
        normals=normals,
        uv_maps=uv_maps,
    )

    d = result.to_dict()
    assert isinstance(d, dict)
    assert "num_frames" in d
    assert "border_ring_ciede2000" in d
    assert "step_timings" in d

    # Must survive JSON round-trip
    serialised = json.dumps(d, default=str)
    restored = json.loads(serialised)
    assert restored["num_frames"] == d["num_frames"]


# -------------------------------------------------------------------
# 2. CLI smoke test
# -------------------------------------------------------------------


def test_recon_closed_loop_cli_json() -> None:
    """``synapse --json recon closed-loop`` returns exit 0."""
    from synapse.cli import main_synapse

    rc = main_synapse(["--json", "recon", "closed-loop"])
    assert rc == 0


def test_recon_closed_loop_cli_human() -> None:
    """``synapse recon closed-loop`` returns exit 0 in human mode."""
    from synapse.cli import main_synapse

    rc = main_synapse(["recon", "closed-loop"])
    assert rc == 0


# -------------------------------------------------------------------
# 3. Stage registration
# -------------------------------------------------------------------


def test_inverse_render_loop_stage_registered() -> None:
    """``inverse_render_loop`` stage is in the bundle worker registry."""
    from synapse.pipeline.bundle_worker import get_registered_stages

    import synapse.pipeline.stages  # noqa: F401

    stages = get_registered_stages()
    assert "inverse_render_loop" in stages


# -------------------------------------------------------------------
# 4. Package exports
# -------------------------------------------------------------------


def test_loop_package_exports() -> None:
    """``synapse.recon.loop`` exports ClosedLoopRunner and ClosedLoopResult."""
    from synapse.recon.loop import ClosedLoopResult, ClosedLoopRunner

    assert ClosedLoopRunner is not None
    assert ClosedLoopResult is not None
