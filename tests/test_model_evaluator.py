"""Tests for synapse.inspect.model_evaluator — 3D model evaluation and fail-closed gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from synapse.bench.model_loader import load_mesh
from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.inspect.model_evaluator import (
    GATE_PROFILES,
    ModelEvaluator,
    _collect_model_files,
    evaluate_3d_models_directory,
    evaluate_profile_gates,
    evaluate_single_model,
    normalize_eval_profile,
)
from synapse.metrics.profiles import (
    RSI_PROMOTION,
    RSI_PROMOTION_TARGET,
    TRACK_A_ASSET,
    TRACK_B_VIDEO,
)

try:
    import synapse._c_geometry  # noqa: F401

    _BACKEND = "native"
except ImportError:
    _BACKEND = "python"


def _create_test_obj(path: Path) -> Path:
    """Create a minimal valid OBJ file on disk."""
    obj_content = (
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "v 0.0 0.0 1.0\n"
        "f 1 3 2\n"
        "f 1 2 4\n"
        "f 2 3 4\n"
        "f 3 1 4\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj_content, encoding="utf-8")
    return path


class TestModelEvaluatorGateProfiles:
    """Tests for dynamic SSOT derivation of GATE_PROFILES."""

    def test_dynamic_derivation_from_profiles(self) -> None:
        """Verify GATE_PROFILES dynamically matches synapse.metrics.profiles thresholds."""
        assert GATE_PROFILES["TRACK_A_ASSET"]["chamfer_mm"] == (
            float(TRACK_A_ASSET["chamfer_mm"]),
            "le",
        )
        assert GATE_PROFILES["TRACK_B_VIDEO"]["chamfer_mm"] == (
            float(TRACK_B_VIDEO["chamfer_mm"]),
            "le",
        )
        assert GATE_PROFILES["RSI_PROMOTION"]["chamfer_mm"] == (
            float(RSI_PROMOTION["chamfer_mm"]),
            "le",
        )
        assert GATE_PROFILES["RSI_PROMOTION_TARGET"]["chamfer_mm"] == (
            float(RSI_PROMOTION_TARGET["chamfer_mm"]),
            "le",
        )

        # Track B does not require watertightness, others do
        assert "is_watertight" not in GATE_PROFILES["TRACK_B_VIDEO"]
        assert "is_watertight" in GATE_PROFILES["TRACK_A_ASSET"]
        assert "is_watertight" in GATE_PROFILES["RSI_PROMOTION"]
        assert "is_watertight" in GATE_PROFILES["RSI_PROMOTION_TARGET"]

    def test_all_16_canonical_metrics_aligned(self) -> None:
        """Verify full 16-item profiles.py metrics are present in GATE_PROFILES."""
        for name, p_dict in [
            ("TRACK_A_ASSET", TRACK_A_ASSET),
            ("TRACK_B_VIDEO", TRACK_B_VIDEO),
            ("RSI_PROMOTION", RSI_PROMOTION),
            ("RSI_PROMOTION_TARGET", RSI_PROMOTION_TARGET),
        ]:
            gate = GATE_PROFILES[name]
            assert len(p_dict) >= 16
            for metric_key, expected_thresh in p_dict.items():
                assert metric_key in gate, f"{metric_key} missing from {name}"
                thresh, op = gate[metric_key]
                assert thresh == float(expected_thresh)

    def test_profile_normalization(self) -> None:
        assert normalize_eval_profile("track_a_asset") == "TRACK_A_ASSET"
        assert normalize_eval_profile("RSI_PROMOTION") == "RSI_PROMOTION"
        with pytest.raises(ContractValidationError):
            normalize_eval_profile("nonexistent_profile")


class TestFailClosedGateEvaluation:
    """Tests for evaluate_profile_gates fail-closed behavior."""

    def test_missing_metric_fails_closed(self) -> None:
        """When chamfer_mm is missing, gate must record violation and not pass."""
        metrics: dict[str, float] = {
            "is_manifold": 1.0,
            "is_watertight": 1.0,
            "num_degenerate_faces": 0.0,
            "self_intersecting_pairs": 0.0,
        }
        res = evaluate_profile_gates(metrics, "TRACK_A_ASSET")
        assert res["chamfer_mm"] is False
        assert res["reference_required_not_evaluated"] is False
        assert not all(res.values())

    def test_nan_or_inf_metric_fails(self) -> None:
        metrics: dict[str, float] = {
            "is_manifold": 1.0,
            "is_watertight": 1.0,
            "num_degenerate_faces": 0.0,
            "self_intersecting_pairs": 0.0,
            "chamfer_mm": float("nan"),
        }
        res = evaluate_profile_gates(metrics, "TRACK_A_ASSET")
        assert res["chamfer_mm"] is False

    def test_exceeding_threshold_fails(self) -> None:
        metrics: dict[str, float] = {
            "is_manifold": 1.0,
            "is_watertight": 1.0,
            "num_degenerate_faces": 0.0,
            "self_intersecting_pairs": 0.0,
            "chamfer_mm": 10.0,  # exceeds 5.0 mm threshold
        }
        res = evaluate_profile_gates(metrics, "TRACK_A_ASSET")
        assert res["chamfer_mm"] is False


class TestSingleModelEvaluation:
    """Tests for evaluate_single_model with/without reference meshes."""

    def test_without_reference_fails_closed(self, tmp_path: Path) -> None:
        """Single model evaluated without reference mesh must not pass full gate."""
        obj_file = _create_test_obj(tmp_path / "cube.obj")
        res = evaluate_single_model(
            model_path=obj_file,
            profile_name="TRACK_A_ASSET",
            resolution=64,
            num_views=6,
            reference_mesh=None,
            geometry_backend=_BACKEND,  # type: ignore[arg-type]
        )
        assert res.overall_passed is False
        assert res.gate_results["reference_required_not_evaluated"] is False
        assert res.gate_results["chamfer_mm"] is False

    def test_with_reference_passes(self, tmp_path: Path) -> None:
        """Single model evaluated with matching reference mesh passes gate."""
        obj_file = _create_test_obj(tmp_path / "cube.obj")
        ref_mesh = load_mesh(str(obj_file), normalize=True)
        res = evaluate_single_model(
            model_path=obj_file,
            profile_name="TRACK_A_ASSET",
            resolution=64,
            num_views=6,
            reference_mesh=ref_mesh,
            geometry_backend=_BACKEND,  # type: ignore[arg-type]
        )
        assert res.metrics["chamfer_mm"] <= 1.0
        assert res.gate_results["chamfer_mm"] is True
        assert res.overall_passed is True


@pytest.mark.slow
class TestModelDirectoryEvaluation:
    """Tests for evaluate_3d_models_directory, including recursive rglob discovery."""

    def test_recursive_discovery(self, tmp_path: Path) -> None:
        """Models in nested subdirectories must be discovered via rglob."""
        in_dir = tmp_path / "inputs"
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"

        _create_test_obj(in_dir / "asset_a" / "meshes" / "model.obj")
        _create_test_obj(ref_dir / "asset_a" / "meshes" / "model.obj")

        res = evaluate_3d_models_directory(
            input_dir=in_dir,
            output_dir=out_dir,
            reference_dir=ref_dir,
            profile_name="TRACK_A_ASSET",
            resolution=64,
            num_views=6,
            geometry_backend=_BACKEND,  # type: ignore[arg-type]
        )
        assert res["summary"]["total"] == 1
        assert res["summary"]["passed"] == 1
        assert (out_dir / "eval_report.html").is_file()
        assert (out_dir / "eval_results.json").is_file()

    def test_invalid_arguments_raise(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            evaluate_3d_models_directory(
                input_dir=tmp_path / "does_not_exist",
                output_dir=tmp_path / "out",
            )

        with pytest.raises(ContractValidationError):
            evaluate_3d_models_directory(
                input_dir=tmp_path,
                output_dir=tmp_path / "out",
                resolution=-1,
            )

    def test_nested_assets_no_key_collision(self, tmp_path: Path) -> None:
        """Verify nested assets (e.g. asset_a/meshes/model.obj and asset_b/meshes/model.obj)

        coexist without key collision and produce two independent evaluation records.
        """
        in_dir = tmp_path / "inputs_nested"
        out_dir = tmp_path / "outputs_nested"
        ref_dir = tmp_path / "refs_nested"

        _create_test_obj(in_dir / "asset_a" / "meshes" / "model.obj")
        _create_test_obj(ref_dir / "asset_a" / "meshes" / "model.obj")
        _create_test_obj(in_dir / "asset_b" / "meshes" / "model.obj")
        _create_test_obj(ref_dir / "asset_b" / "meshes" / "model.obj")

        evaluator = ModelEvaluator(
            profile_name="TRACK_A_ASSET",
            resolution=64,
            num_views=6,
            geometry_backend=_BACKEND,  # type: ignore[arg-type]
        )
        res = evaluator.evaluate_directory(
            input_dir=in_dir,
            output_dir=out_dir,
            reference_dir=ref_dir,
        )

        assert res["summary"]["total"] == 2
        assert res["summary"]["passed"] == 2
        model_names = [m["name"] for m in res["models"]]
        assert len(model_names) == 2
        assert "asset_a/meshes/model" in model_names
        assert "asset_b/meshes/model" in model_names
        # Ensure files were collected by helper
        collected = _collect_model_files(in_dir)
        assert len(collected) == 2
