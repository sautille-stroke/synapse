"""Tests for CLI deterministic exit codes, EXR deep validation, and JSON envelopes.

Covers AUD-P1-06:
- Exit code 40 on QualityGateError for 'synapse-bench run --profile' and 'synapse rsi gate-promote --strict'
- Exit code 10 on ContractValidationError for invalid EXR wire formats and channel types
- Exit code 20 on AssetNotFoundError when '--check-exr' finds no EXR files
- Exit code 0 and multi-axis support for 'synapse rsi gate-promote'
- Zero non-JSON stdout pollution for 'synapse --json generate' and 'synapse-data-gen --json'
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from synapse.cli import main_bench, main_synapse
from synapse.cli._generate import main_data_gen
from synapse.errors import ExitCode
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.fgpp.io import export_fgpp


def _capture_synapse(args: list[str]) -> tuple[int, str, str]:
    """Run main_synapse capturing stdout/stderr."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = buf_out = StringIO()
    sys.stderr = buf_err = StringIO()
    try:
        code = main_synapse(args)
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else 0
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return code, buf_out.getvalue(), buf_err.getvalue()


def _capture_bench(args: list[str]) -> tuple[int, str, str]:
    """Run main_bench capturing stdout/stderr."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = buf_out = StringIO()
    sys.stderr = buf_err = StringIO()
    try:
        code = main_bench(args)
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else 0
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return code, buf_out.getvalue(), buf_err.getvalue()


def _capture_data_gen(args: list[str]) -> tuple[int, str, str]:
    """Run main_data_gen capturing stdout/stderr."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = buf_out = StringIO()
    sys.stderr = buf_err = StringIO()
    try:
        code = main_data_gen(args)
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else 0
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return code, buf_out.getvalue(), buf_err.getvalue()


def _make_minimal_fgpp_dir(tmp: Path, num_frames: int = 1) -> Path:
    """Create a minimal FGPP directory with metadata.json only."""
    out_dir = tmp / "fgpp_meta_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        {
            "frame_index": i,
            "timestamp_sec": i / 24.0,
            "is_physical_anchor": i == 0,
            "channels": ["depth"],
        }
        for i in range(num_frames)
    ]
    meta: dict[str, Any] = {
        "schema_version": "fgpp.v1.0",
        "scene_id": "cli_test_scene",
        "fps": 24.0,
        "num_frames": num_frames,
        "total_frames": num_frames,
        "resolution": [64, 64],
        "frames": frames,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


class TestBenchCliExitCodes:
    """Test deterministic exit codes in synapse-bench CLI."""

    def test_bench_run_gate_failure_exit_40(self) -> None:
        """Running with strict gate profile should exit 40 if thresholds fail."""
        code, out, err = _capture_bench([
            "--json", "run",
            "--scenario", "t1_orbit_97f_humanoid",
            "--profile", "rsi_promotion_target",
            "--adapter", "mock",
        ])
        assert code == ExitCode.CIRCUIT_BREAKER  # 40
        payload = json.loads(out)
        assert payload["status"] == "error"
        assert payload["exit_code"] == 40
        assert payload["error_type"] == "QualityGateError"
        assert "Quality gate 'rsi_promotion_target' failed" in payload["message"]

    def test_bench_evaluate_strict_failure_exit_40(self, tmp_path: Path) -> None:
        """Evaluating bundle with --strict should exit 40 when gate fails."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "metadata.json").write_text(
            json.dumps({"schema_version": "bundle.v1.0", "sources": []}),
            encoding="utf-8",
        )
        code, out, err = _capture_bench([
            "--json", "evaluate",
            "--bundle", str(bundle_dir),
            "--profile", "track_b_video",
            "--strict",
        ])
        assert code == ExitCode.CIRCUIT_BREAKER  # 40
        payload = json.loads(out)
        assert payload["status"] == "error"
        assert payload["exit_code"] == 40
        assert payload["error_type"] == "QualityGateError"


class TestFgppCliExrValidation:
    """Test FGPP CLI deep EXR validation."""

    def test_fgpp_validate_check_exr_missing_files_exit_20(self, tmp_path: Path) -> None:
        """Specifying --check-exr when no EXR files exist should exit 20."""
        fgpp_dir = _make_minimal_fgpp_dir(tmp_path)
        code, out, err = _capture_synapse([
            "--json", "fgpp", "validate", str(fgpp_dir), "--check-exr",
        ])
        assert code == ExitCode.RESOURCE_MISSING  # 20
        payload = json.loads(out)
        assert payload["status"] == "error"
        assert payload["exit_code"] == 20
        assert payload["error_type"] == "AssetNotFoundError"

    def test_fgpp_validate_with_valid_15_channels_exr(self, tmp_path: Path) -> None:
        """Exporting a genuine 15-channel sequence should validate with exit 0."""
        try:
            import OpenEXR  # noqa: F401
        except ImportError:
            pytest.skip("OpenEXR is required for deep EXR validation test")

        h, w = 32, 32
        cam = CameraParams(
            fx=32.0,
            fy=32.0,
            cx=16.0,
            cy=16.0,
            width=w,
            height=h,
            z_near=0.1,
            z_far=100.0,
        )
        frame = FGPPFrame(
            camera=cam,
            depth=np.ones((h, w), dtype=np.float32),
            direct_radiance=np.full((h, w, 3), 0.5, dtype=np.float16),
            world_normal=np.full((h, w, 3), 0.0, dtype=np.float16),
            motion_vector=np.zeros((h, w, 2), dtype=np.float16),
            base_color=np.full((h, w, 3), 0.8, dtype=np.float16),
            roughness_metallic=np.zeros((h, w, 2), dtype=np.float16),
            cryptomatte_id=np.ones((h, w), dtype=np.uint32),
            frame_index=0,
        )
        seq = FGPPSequence(frames=[frame], fps=24.0, scene_id="cli_exr_test")
        out_dir = tmp_path / "valid_fgpp_export"
        export_fgpp(seq, out_dir)

        code, out, err = _capture_synapse([
            "--json", "fgpp", "validate", str(out_dir), "--check-exr",
        ])
        assert code == 0
        payload = json.loads(out)
        assert payload["status"] == "ok"
        assert payload["valid"] is True
        assert payload["exr_validated"] is True

    def test_fgpp_validate_corrupted_exr_missing_channels_exit_10(self, tmp_path: Path) -> None:
        """An EXR missing channels should trigger ContractValidationError (exit 10)."""
        try:
            import Imath
            import OpenEXR
        except ImportError:
            pytest.skip("OpenEXR is required for deep EXR validation test")

        out_dir = _make_minimal_fgpp_dir(tmp_path)
        exr_path = out_dir / "frame_000000.exr"

        # Create an EXR with only RGB channels (missing depth, normal, etc.)
        h, w = 16, 16
        header = OpenEXR.Header(w, h)
        header["channels"] = {
            "R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
        }
        pixels = np.zeros((h, w), dtype=np.float16).tobytes()
        exr = OpenEXR.OutputFile(str(exr_path), header)
        exr.writePixels({"R": pixels, "G": pixels, "B": pixels})
        exr.close()

        code, out, err = _capture_synapse([
            "--json", "fgpp", "validate", str(out_dir), "--check-exr",
        ])
        assert code == ExitCode.CONTRACT_ERROR  # 10
        payload = json.loads(out)
        assert payload["status"] == "error"
        assert payload["exit_code"] == 10
        assert payload["error_type"] == "ContractValidationError"
        assert "missing required FGPP channels" in payload["message"]


class TestRsiCliPromotionGate:
    """Test RSI CLI multi-axis and strict promotion gate exit codes."""

    def test_rsi_gate_promote_multi_axis_success(self) -> None:
        """All axes improving should promote and exit 0."""
        code, out, err = _capture_synapse([
            "--json", "rsi", "gate-promote",
            "--baseline-overall", "0.7",
            "--candidate-overall", "0.85",
            "--baseline-geom", "0.65",
            "--candidate-geom", "0.80",
            "--baseline-temporal", "0.70",
            "--candidate-temporal", "0.88",
        ])
        assert code == 0
        payload = json.loads(out)
        assert payload["status"] == "ok"
        assert payload["promoted"] is True
        assert payload["baseline"]["geom_score"] == 0.65
        assert payload["candidate"]["geom_score"] == 0.80

    def test_rsi_gate_promote_strict_regression_exit_40(self) -> None:
        """Regressed candidate with --strict should exit 40."""
        code, out, err = _capture_synapse([
            "--json", "rsi", "gate-promote",
            "--baseline-overall", "0.85",
            "--candidate-overall", "0.70",
            "--strict",
        ])
        assert code == ExitCode.CIRCUIT_BREAKER  # 40
        payload = json.loads(out)
        assert payload["status"] == "error"
        assert payload["exit_code"] == 40
        assert payload["error_type"] == "QualityGateError"
        assert "promotion gate failed" in payload["message"].lower()


class TestGenerateCliJsonEnvelopes:
    """Test generate and data-gen CLI JSON envelopes and zero stdout pollution."""

    def test_synapse_generate_json_mode(self, tmp_path: Path) -> None:
        """'synapse --json generate' should emit 100% parseable JSON to stdout."""
        out_dir = tmp_path / "gen_output"
        code, out, err = _capture_synapse([
            "--json", "generate",
            "--model", "cube",
            "--frames", "2",
            "--res", "64",
            "--out", str(out_dir),
        ])
        assert code == 0
        # Entire stdout must parse as single JSON object
        clean_out = out.strip()
        assert clean_out.startswith("{") and clean_out.endswith("}")
        payload = json.loads(clean_out)
        assert payload["status"] == "ok"
        assert payload["exit_code"] == 0
        assert payload["frames_count"] == 2
        assert payload["model"] == "cube"

    def test_standalone_data_gen_json_flag(self, tmp_path: Path) -> None:
        """'synapse-data-gen --json' should emit 100% parseable JSON to stdout."""
        out_dir = tmp_path / "data_gen_out"
        code, out, err = _capture_data_gen([
            "--json",
            "--model", "cube",
            "--frames", "2",
            "--res", "64",
            "--out", str(out_dir),
        ])
        assert code == 0
        clean_out = out.strip()
        assert clean_out.startswith("{") and clean_out.endswith("}")
        payload = json.loads(clean_out)
        assert payload["status"] == "ok"
        assert payload["exit_code"] == 0
        assert payload["frames_count"] == 2
