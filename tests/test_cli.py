"""Tests for CLI entry points, exit codes, JSON output, and FGPP commands.

Covers:
- Deterministic exit codes per AGENTS.md §3
- ``--json`` mode: clean JSON output with no noise on stdout
- FGPP validate / inspect sub-commands
- Bench list / export-fgpp sub-commands
- Error payloads include ``agent_fix_hint``
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from synapse.errors import (
    AssetNotFoundError,
    BackendUnavailableError,
    CircuitBreakerError,
    ContractValidationError,
    ExitCode,
    QualityGateError,
    SynapseError,
)
from synapse.telemetry import (
    FallbackLevel,
    MemoryForecaster,
    MemoryForecast,
    TraceContext,
)


# ===================================================================
# Helpers
# ===================================================================


def _capture_main_synapse(
    args: list[str],
) -> tuple[int, str, str]:
    """Run main_synapse capturing stdout/stderr, return (code, out, err)."""
    from synapse.cli import main_synapse

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


def _capture_main_bench(
    args: list[str],
) -> tuple[int, str, str]:
    """Run main_bench capturing stdout/stderr, return (code, out, err)."""
    from synapse.cli import main_bench

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


def _make_fgpp_dir(
    tmp: Path,
    schema_version: str = "fgpp.v1.0",
    num_frames: int = 2,
) -> Path:
    """Create a minimal valid FGPP directory with metadata.json."""
    fgpp_dir = tmp / "fgpp_out"
    fgpp_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(num_frames):
        frames.append({
            "frame_index": i,
            "timestamp_sec": i / 24.0,
            "is_physical_anchor": i % 48 == 0,
            "channels": ["depth"],
        })

    metadata: dict[str, Any] = {
        "schema_version": schema_version,
        "scene_id": "test_scene",
        "fps": 24.0,
        "num_frames": num_frames,
        "total_frames": num_frames,
        "resolution": [512, 512],
        "z_near_meters": 0.1,
        "z_far_meters": 100.0,
        "anchor_interval_frames": 48,
        "frames": frames,
    }

    (fgpp_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    return fgpp_dir


# ===================================================================
# Test: Exit Codes
# ===================================================================


class TestExitCodes:
    """Verify deterministic exit codes from CLI entry points."""

    def test_synapse_no_command_exits_zero(self) -> None:
        code, out, err = _capture_main_synapse([])
        assert code == 0

    def test_bench_no_command_exits_zero(self) -> None:
        code, out, err = _capture_main_bench([])
        assert code == 0

    def test_bench_list_exits_zero(self) -> None:
        code, out, err = _capture_main_bench(["list"])
        assert code == 0

    def test_fgpp_validate_missing_dir_exits_20(self) -> None:
        code, out, err = _capture_main_synapse(
            ["fgpp", "validate", "/nonexistent/path/xyz"]
        )
        assert code == ExitCode.RESOURCE_MISSING

    def test_fgpp_validate_json_mode(self) -> None:
        code, out, err = _capture_main_synapse(
            ["--json", "fgpp", "validate", "/nonexistent/path/xyz"]
        )
        assert code == ExitCode.RESOURCE_MISSING
        data = json.loads(out)
        assert data["exit_code"] == ExitCode.RESOURCE_MISSING
        assert "trace_id" in data

    def test_bench_list_json_mode(self) -> None:
        code, out, err = _capture_main_bench(["--json", "list"])
        assert code == 0
        data = json.loads(out)
        assert data["status"] == "ok"
        assert "scenarios" in data


# ===================================================================
# Test: JSON Output Quality
# ===================================================================


class TestJsonOutput:
    """Verify --json output is clean, parseable, and well-structured."""

    def test_json_no_noise_on_success(self) -> None:
        """Successful --json output must be a single JSON object."""
        code, out, err = _capture_main_bench(["--json", "list"])
        assert code == 0
        # Must parse as JSON — exactly one object on stdout
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert isinstance(data, dict)

    def test_json_error_has_fix_hint(self) -> None:
        """Error JSON must include agent_fix_hint field."""
        code, out, err = _capture_main_synapse(
            ["--json", "fgpp", "validate", "/no/such/dir"]
        )
        assert code != 0
        data = json.loads(out)
        assert "agent_fix_hint" in data
        assert len(data["agent_fix_hint"]) > 0

    def test_json_success_has_trace_id(self) -> None:
        code, out, err = _capture_main_bench(["--json", "list"])
        data = json.loads(out)
        assert "trace_id" in data
        assert len(data["trace_id"]) == 32  # 16 bytes hex

    def test_json_no_command_is_valid(self) -> None:
        code, out, err = _capture_main_synapse(["--json"])
        assert code == 0
        data = json.loads(out)
        assert data["status"] == "ok"


# ===================================================================
# Test: FGPP Commands
# ===================================================================


class TestFgppCommands:
    """Test FGPP validate and inspect sub-commands."""

    def test_fgpp_validate_valid_dir(self, tmp_path: Path) -> None:
        fgpp_dir = _make_fgpp_dir(tmp_path)
        code, out, err = _capture_main_synapse(
            ["--json", "fgpp", "validate", str(fgpp_dir)]
        )
        assert code == 0
        data = json.loads(out)
        assert data["valid"] is True
        assert data["num_frames"] == 2

    def test_fgpp_validate_schema_mismatch(self, tmp_path: Path) -> None:
        fgpp_dir = _make_fgpp_dir(tmp_path, schema_version="fgpp.v999")
        code, out, err = _capture_main_synapse(
            ["--json", "fgpp", "validate", str(fgpp_dir)]
        )
        assert code == ExitCode.CONTRACT_ERROR
        data = json.loads(out)
        assert data["error_type"] == "ContractValidationError"

    def test_fgpp_validate_frame_count_mismatch(
        self, tmp_path: Path,
    ) -> None:
        fgpp_dir = _make_fgpp_dir(tmp_path, num_frames=3)
        # Tamper with the header to create a mismatch
        meta_path = fgpp_dir / "metadata.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        raw["num_frames"] = 99
        raw["total_frames"] = 99
        meta_path.write_text(
            json.dumps(raw, indent=2), encoding="utf-8",
        )
        code, out, err = _capture_main_synapse(
            ["--json", "fgpp", "validate", str(fgpp_dir)]
        )
        assert code == ExitCode.CONTRACT_ERROR

    def test_fgpp_inspect_valid_dir(self, tmp_path: Path) -> None:
        fgpp_dir = _make_fgpp_dir(tmp_path, num_frames=5)
        code, out, err = _capture_main_synapse(
            ["--json", "fgpp", "inspect", str(fgpp_dir)]
        )
        assert code == 0
        data = json.loads(out)
        assert data["num_frames"] == 5
        assert data["scene_id"] == "test_scene"

    def test_fgpp_inspect_missing_dir(self) -> None:
        code, out, err = _capture_main_synapse(
            ["--json", "fgpp", "inspect", "/no/such/fgpp"]
        )
        assert code == ExitCode.RESOURCE_MISSING


# ===================================================================
# Test: Error Hierarchy
# ===================================================================


class TestErrorHierarchy:
    """Verify error classes, exit codes, and JSON serialization."""

    def test_synapse_error_base(self) -> None:
        err = SynapseError("test", ExitCode.CONTRACT_ERROR, "fix it")
        assert err.exit_code == ExitCode.CONTRACT_ERROR
        d = err.to_json_dict("abc123")
        assert d["status"] == "error"
        assert d["trace_id"] == "abc123"
        assert d["agent_fix_hint"] == "fix it"

    def test_contract_validation_error(self) -> None:
        err = ContractValidationError("bad schema")
        assert err.exit_code == ExitCode.CONTRACT_ERROR

    def test_asset_not_found_error(self) -> None:
        err = AssetNotFoundError("missing file")
        assert err.exit_code == ExitCode.RESOURCE_MISSING

    def test_backend_unavailable_error(self) -> None:
        err = BackendUnavailableError("GPU OOM")
        assert err.exit_code == ExitCode.BACKEND_UNAVAILABLE

    def test_circuit_breaker_error(self) -> None:
        err = CircuitBreakerError("timeout")
        assert err.exit_code == ExitCode.CIRCUIT_BREAKER

    def test_quality_gate_error(self) -> None:
        err = QualityGateError("PSNR too low")
        assert err.exit_code == ExitCode.CIRCUIT_BREAKER

    def test_all_errors_are_synapse_error(self) -> None:
        for cls in (
            ContractValidationError,
            AssetNotFoundError,
            BackendUnavailableError,
            CircuitBreakerError,
            QualityGateError,
        ):
            assert issubclass(cls, SynapseError)


# ===================================================================
# Test: Telemetry — TraceContext
# ===================================================================


class TestTraceContext:
    """Test W3C trace context parsing and generation."""

    def test_from_traceparent_header(self) -> None:
        env = {
            "traceparent": "00-aabbccdd11223344aabbccdd11223344-1122334455667788-01",
            "tracestate": "vendor=hello",
        }
        ctx = TraceContext.from_env_or_dict(env)
        assert ctx.trace_id == "aabbccdd11223344aabbccdd11223344"
        assert ctx.span_id == "1122334455667788"
        assert ctx.trace_flags == "01"
        assert ctx.tracestate == "vendor=hello"

    def test_generates_random_when_absent(self) -> None:
        ctx = TraceContext.from_env_or_dict({})
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16

    def test_to_traceparent_roundtrip(self) -> None:
        ctx = TraceContext(
            trace_id="a" * 32, span_id="b" * 16, trace_flags="01",
        )
        tp = ctx.to_traceparent()
        assert tp == f"00-{'a' * 32}-{'b' * 16}-01"

    def test_to_headers(self) -> None:
        ctx = TraceContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            trace_flags="01",
            tracestate="k=v",
        )
        headers = ctx.to_headers()
        assert "traceparent" in headers
        assert headers["tracestate"] == "k=v"

    def test_headers_no_tracestate_when_empty(self) -> None:
        ctx = TraceContext(
            trace_id="a" * 32, span_id="b" * 16,
        )
        ctx.tracestate = ""
        headers = ctx.to_headers()
        assert "tracestate" not in headers


# ===================================================================
# Test: Telemetry — MemoryForecaster
# ===================================================================


class TestMemoryForecaster:
    """Test GPU memory estimation and fallback level selection."""

    def test_small_model_low_res(self) -> None:
        forecast = MemoryForecaster.estimate_dit_peak_gb(
            num_frames=5, height=256, width=256,
            model_variant="wan2.1_1.3b",
        )
        assert isinstance(forecast, MemoryForecast)
        assert forecast.peak_gb > 0
        assert forecast.fallback_level == FallbackLevel.LEVEL_0_FULL
        assert "weights_gb" in forecast.breakdown

    def test_14b_model_high_res_many_frames(self) -> None:
        forecast = MemoryForecaster.estimate_dit_peak_gb(
            num_frames=97, height=1024, width=1024,
            model_variant="wan2.1_14b",
        )
        assert forecast.peak_gb > 28.0  # At least weights
        assert forecast.fallback_level.value >= 0

    def test_reject_oom_on_extreme_input(self) -> None:
        forecast = MemoryForecaster.estimate_dit_peak_gb(
            num_frames=500, height=4096, width=4096,
            model_variant="wan2.1_14b",
        )
        assert forecast.fallback_level == FallbackLevel.REJECT_OOM

    def test_unknown_model_uses_defaults(self) -> None:
        forecast = MemoryForecaster.estimate_dit_peak_gb(
            num_frames=5, height=256, width=256,
            model_variant="unknown_model_xyz",
        )
        # Should not crash — uses default weight size
        assert forecast.peak_gb > 0

    def test_breakdown_has_expected_keys(self) -> None:
        forecast = MemoryForecaster.estimate_dit_peak_gb(
            num_frames=5, height=512, width=512,
        )
        assert "weights_gb" in forecast.breakdown
        assert "activations_gb" in forecast.breakdown
        assert "kv_cache_gb" in forecast.breakdown
        assert "overhead_factor" in forecast.breakdown


# ===================================================================
# Test: Bench Commands
# ===================================================================


class TestBenchCommands:
    """Test synapse-bench CLI sub-commands."""

    def test_bench_list_text_mode(self) -> None:
        code, out, err = _capture_main_bench(["list"])
        assert code == 0

    def test_bench_list_json_returns_array(self) -> None:
        code, out, err = _capture_main_bench(["--json", "list"])
        assert code == 0
        data = json.loads(out)
        assert isinstance(data["scenarios"], list)
        assert "count" in data
