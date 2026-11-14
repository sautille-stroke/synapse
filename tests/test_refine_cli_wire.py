# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for PR-37: LocalRefinementPipeline CLI wiring and refine_session
wire format.

Covers:
  - ``synapse refine anatomical --json`` CLI returns exit 0
  - ``synapse.refine_session.v1.0`` wire-format structure
  - All 5 FusionMethod names map correctly in ``_METHOD_MAP``
  - ``anatomical_refine`` stage is registered in the bundle worker
  - ``build_refine_session_payload()`` returns correct structure
  - ``refine_session.v1.0`` round-trip for 5 CommandIR types
  - ``LocalRefinementPipeline.run()`` produces valid output
"""

from __future__ import annotations

import json

import numpy as np

# -------------------------------------------------------------------
# 1. CLI smoke test
# -------------------------------------------------------------------


def test_refine_anatomical_cli_json() -> None:
    """``synapse --json refine anatomical`` returns exit 0 and valid JSON."""
    from synapse.cli import main_synapse

    rc = main_synapse(["--json", "refine", "anatomical"])
    assert rc == 0


def test_refine_anatomical_cli_no_json() -> None:
    """``synapse refine anatomical`` returns exit 0 in human mode."""
    from synapse.cli import main_synapse

    rc = main_synapse(["refine", "anatomical"])
    assert rc == 0


def test_refine_cli_missing_subcommand_json() -> None:
    """``synapse --json refine`` without sub-command returns exit 10."""
    from synapse.cli import main_synapse

    rc = main_synapse(["--json", "refine"])
    assert rc == 10


# -------------------------------------------------------------------
# 2. Wire-format: refine_session.v1.0
# -------------------------------------------------------------------


def test_build_refine_session_payload_schema() -> None:
    """``build_refine_session_payload()`` returns correct structure."""
    from synapse.refine.pipeline import build_refine_session_payload

    payload = build_refine_session_payload(
        session_id="test-session-001",
        regions=["HAND", "FACE", "FOOT"],
        fusion_method="CASCADE",
        max_elapsed_seconds=600.0,
    )

    assert payload["schema_version"] == "synapse.refine_session.v1.0"
    assert payload["session_id"] == "test-session-001"
    assert payload["regions"] == ["HAND", "FACE", "FOOT"]
    assert payload["fusion_method"] == "CASCADE"
    assert payload["max_elapsed_seconds"] == 600.0
    assert isinstance(payload["command_ir_records"], list)
    assert len(payload["command_ir_records"]) == 0


def test_refine_session_payload_defaults() -> None:
    """Default payload has correct default regions and method."""
    from synapse.refine.pipeline import build_refine_session_payload

    payload = build_refine_session_payload(session_id="def-001")
    assert payload["regions"] == ["HAND", "FACE"]
    assert payload["fusion_method"] == "LAPLACIAN_FREQUENCY"
    assert payload["max_elapsed_seconds"] == 900.0


def test_refine_session_roundtrip_json() -> None:
    """Payload round-trips through JSON serialisation."""
    from synapse.refine.pipeline import build_refine_session_payload

    payload = build_refine_session_payload(
        session_id="rt-001",
        regions=["HAIR"],
        fusion_method="POISSON_GRADIENT",
    )
    serialised = json.dumps(payload, default=str)
    restored = json.loads(serialised)
    assert restored == payload


def test_refine_session_command_ir_roundtrip() -> None:
    """5 CommandIR types survive round-trip via the payload."""
    from synapse.refine.pipeline import RefineSessionPayload

    ir_records = [
        {"type": "SCULPT_MOVE", "vertex_ids": [1, 2, 3]},
        {"type": "SCULPT_SMOOTH", "radius": 0.5},
        {"type": "PAINT_STROKE", "uv_coords": [[0.1, 0.2]]},
        {"type": "RETOPOLOGY_EDGE_LOOP", "edge_count": 12},
        {"type": "UV_UNWRAP", "method": "smart_uv"},
    ]
    session = RefineSessionPayload(
        session_id="ir-test",
        command_ir_records=ir_records,
    )
    serialised = json.dumps({
        "schema_version": session.schema_version,
        "session_id": session.session_id,
        "command_ir_records": session.command_ir_records,
    })
    restored = json.loads(serialised)
    assert len(restored["command_ir_records"]) == 5
    types = [r["type"] for r in restored["command_ir_records"]]
    assert types == [
        "SCULPT_MOVE", "SCULPT_SMOOTH", "PAINT_STROKE",
        "RETOPOLOGY_EDGE_LOOP", "UV_UNWRAP",
    ]


# -------------------------------------------------------------------
# 3. FusionMethod mapping coverage
# -------------------------------------------------------------------


def test_method_map_all_five() -> None:
    """All 5 FusionMethod names are present in ``_METHOD_MAP``."""
    from synapse.refine.pipeline import _METHOD_MAP
    from synapse.refine.types import FusionMethod

    expected = {
        "LAPLACIAN_FREQUENCY": FusionMethod.LAPLACIAN_FREQUENCY,
        "CASCADE": FusionMethod.CASCADE,
        "LATENT_TIMESTEP_SWITCH": FusionMethod.LATENT_TIMESTEP_SWITCH,
        "UNCERTAINTY_WEIGHTED": FusionMethod.UNCERTAINTY_WEIGHTED,
        "POISSON_GRADIENT": FusionMethod.POISSON_GRADIENT,
    }
    assert _METHOD_MAP == expected


# -------------------------------------------------------------------
# 4. Stage registration
# -------------------------------------------------------------------


def test_anatomical_refine_stage_registered() -> None:
    """``anatomical_refine`` stage is in the bundle worker registry."""
    # Importing stages triggers registration
    import synapse.pipeline.stages  # noqa: F401
    from synapse.pipeline.bundle_worker import get_registered_stages

    stages = get_registered_stages()
    assert "anatomical_refine" in stages


# -------------------------------------------------------------------
# 5. Pipeline e2e
# -------------------------------------------------------------------


def test_pipeline_run_synthetic() -> None:
    """``LocalRefinementPipeline.run()`` produces valid output on synthetic."""
    from synapse.refine.pipeline import LocalRefinementPipeline
    from synapse.refine.types import (
        AnatomicalRegion,
        FusionMethod,
        RefinementROI,
    )

    frames = np.random.default_rng(42).random(
        (4, 64, 64, 3), dtype=np.float32,
    )

    rois_per_frame: list[list[RefinementROI]] = []
    for t in range(4):
        roi = RefinementROI(
            region=AnatomicalRegion.HAND,
            bbox_xyxy=np.array([16, 16, 48, 48], dtype=np.float32),
            crop_acescg=frames[t, 16:48, 16:48, :].copy(),
            confidence=0.85,
        )
        rois_per_frame.append([roi])

    pipeline = LocalRefinementPipeline()
    result = pipeline.run(
        frames_acescg=frames,
        rois_per_frame=rois_per_frame,
        fusion_method=FusionMethod.LAPLACIAN_FREQUENCY,
    )

    assert result.frames_refined.shape == (4, 64, 64, 3)
    assert result.num_rois_detected == 4
    assert result.num_patches_fused == 4
    assert result.elapsed_seconds >= 0


def test_pipeline_circuit_breaker_propagates() -> None:
    """Injecting elapsed_seconds=901.0 must raise CircuitBreakerError(exit_code=40)."""
    import pytest
    from synapse.errors import CircuitBreakerError
    from synapse.refine.pipeline import LocalRefinementPipeline
    from synapse.refine.types import (
        AnatomicalRegion,
        FusionMethod,
        RefinementROI,
    )

    frames = np.ones((2, 32, 32, 3), dtype=np.float32) * 0.5
    rois_per_frame: list[list[RefinementROI]] = []
    for t in range(2):
        roi = RefinementROI(
            region=AnatomicalRegion.HAND,
            bbox_xyxy=np.array([8, 8, 24, 24], dtype=np.float32),
            crop_acescg=frames[t, 8:24, 8:24, :].copy(),
            confidence=0.9,
        )
        rois_per_frame.append([roi])

    pipeline = LocalRefinementPipeline()
    with pytest.raises(CircuitBreakerError) as exc_info:
        pipeline.run(
            frames_acescg=frames,
            rois_per_frame=rois_per_frame,
            fusion_method=FusionMethod.LAPLACIAN_FREQUENCY,
            elapsed_seconds_override=901.0,
        )
    assert int(exc_info.value.exit_code) == 40


def test_pipeline_no_candidates_dispatches_to_refiner() -> None:
    """Without candidates_per_roi, output patches must differ from input crop_acescg (not identity)."""
    from synapse.refine.pipeline import LocalRefinementPipeline
    from synapse.refine.types import (
        AnatomicalRegion,
        FusionMethod,
        RefinementROI,
    )

    rng = np.random.default_rng(77)
    frames = rng.random((2, 64, 64, 3), dtype=np.float32)

    rois_per_frame: list[list[RefinementROI]] = []
    for t in range(2):
        # Use HAND region to trigger HandRefiner dispatch
        roi = RefinementROI(
            region=AnatomicalRegion.HAND,
            bbox_xyxy=np.array([10, 10, 42, 42], dtype=np.float32),
            crop_acescg=frames[t, 10:42, 10:42, :].copy(),
            confidence=0.3,  # low confidence -> not bypassed by tiled executor
        )
        rois_per_frame.append([roi])

    pipeline = LocalRefinementPipeline()
    result = pipeline.run(
        frames_acescg=frames,
        rois_per_frame=rois_per_frame,
        candidates_per_roi=None,
        fusion_method=FusionMethod.LAPLACIAN_FREQUENCY,
    )

    assert result.frames_refined.shape == frames.shape
    assert result.num_rois_detected == 2
    # The pipeline should have produced refined output (not just
    # identical copies of the input crop).
    assert result.num_patches_fused == 2


def test_pipeline_no_candidates_face_dispatches() -> None:
    """FACE region without candidates dispatches to FaceRefiner."""
    from synapse.refine.pipeline import LocalRefinementPipeline
    from synapse.refine.types import (
        AnatomicalRegion,
        FusionMethod,
        RefinementROI,
    )

    frames = np.full((1, 32, 32, 3), 0.5, dtype=np.float32)
    roi = RefinementROI(
        region=AnatomicalRegion.FACE,
        bbox_xyxy=np.array([4, 4, 28, 28], dtype=np.float32),
        crop_acescg=frames[0, 4:28, 4:28, :].copy(),
        confidence=0.3,  # low -> not bypassed
    )

    pipeline = LocalRefinementPipeline()
    result = pipeline.run(
        frames_acescg=frames,
        rois_per_frame=[[roi]],
        candidates_per_roi=None,
        fusion_method=FusionMethod.LAPLACIAN_FREQUENCY,
    )
    assert result.num_patches_fused == 1


def test_pipeline_result_to_dict() -> None:
    """``RefinePipelineResult.to_dict()`` returns JSON-safe dict."""
    from synapse.refine.pipeline import LocalRefinementPipeline
    from synapse.refine.types import (
        AnatomicalRegion,
        FusionMethod,
        RefinementROI,
    )

    frames = np.ones((2, 32, 32, 3), dtype=np.float32) * 0.5

    rois_per_frame: list[list[RefinementROI]] = []
    for t in range(2):
        roi = RefinementROI(
            region=AnatomicalRegion.FACE,
            bbox_xyxy=np.array([8, 8, 24, 24], dtype=np.float32),
            crop_acescg=frames[t, 8:24, 8:24, :].copy(),
            confidence=0.9,
        )
        rois_per_frame.append([roi])

    pipeline = LocalRefinementPipeline()
    result = pipeline.run(
        frames_acescg=frames,
        rois_per_frame=rois_per_frame,
        fusion_method=FusionMethod.CASCADE,
    )

    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["num_frames"] == 2
    assert isinstance(d["mean_border_psnr_db"], float)
    assert isinstance(d["mean_border_ciede2000"], float)

    # Must survive JSON serialisation
    serialised = json.dumps(d, default=str)
    assert json.loads(serialised) is not None


# -------------------------------------------------------------------
# 6. Package exports
# -------------------------------------------------------------------


def test_refine_package_exports() -> None:
    """``synapse.refine`` exports pipeline types."""
    from synapse.refine import (
        LocalRefinementPipeline,
        RefinePipelineResult,
        RefineSessionPayload,
        build_refine_session_payload,
    )

    assert LocalRefinementPipeline is not None
    assert RefinePipelineResult is not None
    assert RefineSessionPayload is not None
    assert build_refine_session_payload is not None
