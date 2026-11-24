"""Tests for M51 (PR-210 ~ PR-212): weekly GPU smoke, wire_guard --require-style, Ruff SSOT."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from synapse.errors import ExitCode

from scripts.run_gpu_smoke import main as gpu_smoke_main

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tool(name: str) -> Any:
    path = _REPO_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_STYLE_FIELDS = (
    "schema_version",
    "working_colorspace",
    "color_temp_kelvin",
    "tint",
    "exposure_ev",
    "contrast",
    "saturation",
    "key_light_dir_world",
    "key_light_rgb",
    "ambient_ratio",
    "grain_sigma",
    "vignette_strength",
    "tone_curve_lut_1d",
)


def test_pr210_gpu_smoke_dry_run_and_missing_cuda(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """PR-210: run_gpu_smoke --dry-run emits valid JSON; missing CUDA exits 30."""
    out_json = tmp_path / "smoke_dry.json"
    with mock.patch(
        "scripts.run_gpu_smoke._probe_gpu",
        return_value={"cuda_available": False, "device_name": "cpu", "vram_gb": 0.0},
    ):
        rc_dry = gpu_smoke_main(
            ["--dry-run", "--json", "--output-json", str(out_json)]
        )
        assert rc_dry == int(ExitCode.OK)
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "DRY_RUN"
        assert len(payload["planned_stages"]) >= 3
        assert out_json.is_file()

        rc_fail = gpu_smoke_main(["--json"])
        assert rc_fail == int(ExitCode.BACKEND_UNAVAILABLE)
        err_payload = json.loads(capsys.readouterr().out)
        assert err_payload["exit_code"] == 30
        assert "agent_fix_hint" in err_payload


def test_pr211_wire_guard_require_style(tmp_path: Path) -> None:
    """PR-211: wire_guard --require-style fails when absent/tampered and passes on parity."""
    wire_guard = _load_tool("wire_guard")
    missing_hdr = tmp_path / "nonexistent_style_types.h"
    rc_missing = wire_guard.main(
        ["--check", "--require-style", "--aether-style-header", str(missing_hdr)]
    )
    assert rc_missing == 1

    valid_hdr = tmp_path / "style_types.h"
    fields_cpp = "\n".join(f"  float {name};" for name in _STYLE_FIELDS)
    valid_hdr.write_text(f"struct StyleDescriptor {{\n{fields_cpp}\n}};\n", encoding="utf-8")
    rc_valid = wire_guard.main(
        ["--check", "--require-style", "--aether-style-header", str(valid_hdr)]
    )
    assert rc_valid == 0

    tampered_hdr = tmp_path / "tampered_style_types.h"
    tampered_fields = "\n".join(
        f"  float {name};" for name in _STYLE_FIELDS if name != "exposure_ev"
    )
    tampered_hdr.write_text(
        f"struct StyleDescriptor {{\n{tampered_fields}\n}};\n", encoding="utf-8"
    )
    rc_tampered = wire_guard.main(
        ["--check", "--require-style", "--aether-style-header", str(tampered_hdr)]
    )
    assert rc_tampered == 1


def test_pr212_sync_ruff_ssot(tmp_path: Path) -> None:
    """PR-212: Ruff SSOT parity check handles absent SSOT, aligned SSOT, and drift."""
    sync_ruff_ssot = _load_tool("sync_ruff_ssot")
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"

    absent_ssot = tmp_path / "absent_ruff.toml"
    rc_skip = sync_ruff_ssot.main(
        ["--check", "--pyproject", str(pyproject), "--ssot-path", str(absent_ssot)]
    )
    assert rc_skip == 0

    rc_req = sync_ruff_ssot.main(
        [
            "--check",
            "--require-ssot",
            "--pyproject",
            str(pyproject),
            "--ssot-path",
            str(absent_ssot),
        ]
    )
    assert rc_req == 1

    aligned_ssot = tmp_path / "aligned_ruff.toml"
    aligned_ssot.write_text(
        'target-version = "py312"\nline-length = 100\n\n[lint]\nselect = ["E4", "F", "I"]\n',
        encoding="utf-8",
    )
    rc_aligned = sync_ruff_ssot.main(
        ["--check", "--pyproject", str(pyproject), "--ssot-path", str(aligned_ssot)]
    )
    assert rc_aligned == 0

    drifted_ssot = tmp_path / "drifted_ruff.toml"
    drifted_ssot.write_text(
        'target-version = "py312"\nline-length = 120\n\n[lint]\nselect = ["E4", "Z999"]\n',
        encoding="utf-8",
    )
    rc_drifted = sync_ruff_ssot.main(
        ["--check", "--pyproject", str(pyproject), "--ssot-path", str(drifted_ssot)]
    )
    assert rc_drifted == 1
