"""Tests for M51 (PR-210 ~ PR-212): weekly GPU smoke, wire_guard --require-style, Ruff SSOT."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from synapse.errors import ExitCode

from scripts.run_gpu_smoke import PLANNED_STAGES
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


def test_td05_gpu_smoke_stages_align_and_safetensors_reload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-TD05: run_gpu_smoke stages align 1:1 with planned_stages and verify SafeTensors reload."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    monkeypatch.delenv("WAN_MODEL_PATH", raising=False)

    out_json = tmp_path / "smoke_cuda.json"
    with (
        mock.patch(
            "scripts.run_gpu_smoke._probe_gpu",
            return_value={"cuda_available": True, "device_name": "MockCUDA-T4", "vram_gb": 16.0},
        ),
        mock.patch("scripts.run_gpu_smoke._select_torch_device", return_value=torch.device("cpu")),
        mock.patch("torch.cuda.max_memory_allocated", return_value=2 * 1024**3),
    ):
        rc = gpu_smoke_main(["--json", "--output-json", str(out_json)])
    assert rc == int(ExitCode.OK)
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert tuple(payload["planned_stages"]) == PLANNED_STAGES
    assert set(payload["stages"].keys()) == set(PLANNED_STAGES)
    assert payload["stages"]["diffusers_wan_backend_single_window"]["status"] == "skipped_no_weights"
    assert payload["stages"]["controlnext_patchify_stem_forward"]["status"] == "passed"
    assert payload["stages"]["stateless_lora_trainer_grad_step"]["status"] == "passed"
    assert payload["stages"]["stateless_lora_trainer_grad_step"]["injected_layers"] > 0
    assert payload["stages"]["safetensors_export_reload"]["status"] == "passed"
    assert payload["stages"]["safetensors_export_reload"]["num_tensors"] >= 2

    # Explicit nonexistent WAN_MODEL_PATH must fail closed with exit code 20
    missing_model = tmp_path / "missing_wan_dir"
    with mock.patch(
        "scripts.run_gpu_smoke._probe_gpu",
        return_value={"cuda_available": True, "device_name": "MockCUDA-T4", "vram_gb": 16.0},
    ):
        rc_missing = gpu_smoke_main(["--json", "--wan-model-path", str(missing_model)])
    assert rc_missing == int(ExitCode.RESOURCE_MISSING)
    missing_payload = json.loads(capsys.readouterr().out)
    assert missing_payload["exit_code"] == int(ExitCode.RESOURCE_MISSING)


def test_td05_gpu_smoke_with_wan_weights_runs_diffusers_and_refine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-TD05: valid WAN_MODEL_PATH runs DiffusersWanBackend window + LocalRefinementPipeline."""
    import numpy as np

    model_dir = tmp_path / "wan_ckpt"
    model_dir.mkdir()
    monkeypatch.setenv("WAN_MODEL_PATH", str(model_dir))

    class _FakeWanBackend:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def submit(self, _payload: Any, config: Any) -> np.ndarray:
            rng = np.random.default_rng(7)
            return np.asarray(
                rng.integers(
                    20, 220, size=(config.num_frames, config.height, config.width, 3), dtype=np.uint8
                ),
                dtype=np.uint8,
            )

    from scripts.run_gpu_smoke import _run_wan_single_window_stage

    with mock.patch("synapse.adapters.wan_diffusers.DiffusersWanBackend", _FakeWanBackend):
        stage_res = _run_wan_single_window_stage(model_dir)
    assert stage_res["status"] == "passed"
    assert stage_res["frames_shape"] == [5, 64, 64, 3]
    assert stage_res["refinement_patches"] >= 1


def test_td05_four_track_baseline_artifacts_and_generator(tmp_path: Path) -> None:
    """PR-TD05: checked-in 40-scenario baseline & generator work cleanly with compare_runs."""
    from synapse.bench.compare import compare_runs

    from scripts.run_4track_baseline import generate_four_track_baseline

    baseline_dir = _REPO_ROOT / "data" / "baselines" / "synapse_bench_4track"
    clean_json = baseline_dir / "mock_clean.json"
    perturbed_json = baseline_dir / "mock_perturbed_baseline.json"
    leaderboard_json = baseline_dir / "leaderboard.json"
    leaderboard_html = baseline_dir / "leaderboard.html"

    for p in (clean_json, perturbed_json, leaderboard_json, leaderboard_html):
        assert p.is_file(), f"Missing versioned baseline artifact: {p}"

    clean_data = json.loads(clean_json.read_text(encoding="utf-8"))
    assert clean_data["scenario_count"] == 40
    assert len(clean_data["reports"]) == 40

    rep = compare_runs([clean_json, perturbed_json])
    assert rep.overall_winner == "mock_clean"

    gen_dir = tmp_path / "baseline_smoke"
    gen_res = generate_four_track_baseline(output_dir=gen_dir, seed=42, resolution=32, num_frames=4, limit=2)
    assert gen_res["status"] == "ok"
    assert gen_res["scenario_count"] == 2
    assert gen_res["overall_winner"] == "mock_clean"
    assert Path(gen_res["leaderboard_html"]).is_file()

