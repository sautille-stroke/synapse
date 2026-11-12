"""Tests for threshold SSOT synchronization with DESIGN_REVIEW_AND_ALIGNMENT.md."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from synapse.inspect.flywheel_exporter import RoundtripGate
from synapse.metrics.profiles import (
    RSI_PROMOTION,
    RSI_PROMOTION_TARGET,
    TRACK_A_ASSET,
    TRACK_B_VIDEO,
    evaluate_gate,
)
from synapse.rsi.dual_verifier import DualVerifierEngine, PhysicsThresholds
from synapse.training.packager import LoRAGateThresholds


def _parse_doc_threshold_table() -> dict[str, dict[str, float]]:
    """Parse §3.4 threshold matrix from docs/DESIGN_REVIEW_AND_ALIGNMENT.md."""
    repo_root = Path(__file__).resolve().parent.parent
    doc_path = repo_root / "docs" / "DESIGN_REVIEW_AND_ALIGNMENT.md"
    text = doc_path.read_text(encoding="utf-8")

    # Find lines in §3.4 table starting with | `metric_name` |
    row_pattern = re.compile(
        r"^\|\s*`([a-z0-9_]+)`\s*\|[^|]+\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|",
        re.MULTILINE,
    )
    num_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)")

    parsed: dict[str, dict[str, float]] = {
        "track_b_video": {},
        "track_a_asset": {},
        "rsi_promotion": {},
    }
    for match in row_pattern.finditer(text):
        metric, col_b, col_a, col_r = (
            match.group(1),
            match.group(2),
            match.group(3),
            match.group(4),
        )
        m_b = num_pattern.search(col_b)
        m_a = num_pattern.search(col_a)
        m_r = num_pattern.search(col_r)
        if m_b and m_a and m_r:
            parsed["track_b_video"][metric] = float(m_b.group(1))
            parsed["track_a_asset"][metric] = float(m_a.group(1))
            parsed["rsi_promotion"][metric] = float(m_r.group(1))
    return parsed


class TestProfilesDocSync:
    """Verify synapse.metrics.profiles matches DESIGN_REVIEW_AND_ALIGNMENT.md §3.4."""

    def test_doc_table_matches_profiles(self) -> None:
        doc_profiles = _parse_doc_threshold_table()
        # Core roundtrip / geometry / style metrics governed by §3.4
        governed_metrics = [
            "flicker_score",
            "depth_rmse_m",
            "normal_angular_error_deg",
            "silhouette_iou",
            "chamfer_mm",
            "volume_breathing_pct",
            "foot_sliding_cm",
            "albedo_delta_e00",
            "border_ring_ciede2000",
        ]
        for metric in governed_metrics:
            assert TRACK_B_VIDEO[metric] == pytest.approx(
                doc_profiles["track_b_video"][metric]
            ), f"track_b_video.{metric} mismatch"
            assert TRACK_A_ASSET[metric] == pytest.approx(
                doc_profiles["track_a_asset"][metric]
            ), f"track_a_asset.{metric} mismatch"
            assert RSI_PROMOTION[metric] == pytest.approx(
                doc_profiles["rsi_promotion"][metric]
            ), f"rsi_promotion.{metric} mismatch"

    def test_rsi_promotion_target_stricter_or_equal(self) -> None:
        for key, val in RSI_PROMOTION.items():
            assert key in RSI_PROMOTION_TARGET
            if key in ("silhouette_iou", "f_score_1mm"):
                assert RSI_PROMOTION_TARGET[key] >= val
            else:
                assert RSI_PROMOTION_TARGET[key] <= val

    def test_domain_thresholds_derive_from_profiles(self) -> None:
        lora_t = LoRAGateThresholds.from_profile("rsi_promotion")
        assert lora_t.chamfer_mm == RSI_PROMOTION["chamfer_mm"]
        assert (
            lora_t.normal_angular_error_deg
            == RSI_PROMOTION["normal_angular_error_deg"]
        )
        assert (
            lora_t.border_ring_ciede2000
            == RSI_PROMOTION["border_ring_ciede2000"]
        )

        phys_t = PhysicsThresholds.from_profile("rsi_promotion")
        assert phys_t.foot_sliding_cm == RSI_PROMOTION["foot_sliding_cm"]
        assert phys_t.silhouette_iou == RSI_PROMOTION["silhouette_iou"]
        assert phys_t.depth_rmse_m == RSI_PROMOTION["depth_rmse_m"]
        assert (
            phys_t.normal_angular_error_deg
            == RSI_PROMOTION["normal_angular_error_deg"]
        )

        rt_gate = RoundtripGate.from_profile("track_a_asset")
        assert (
            rt_gate.volume_breathing_pct
            == TRACK_A_ASSET["volume_breathing_pct"]
        )

    def test_dual_verifier_checks_foot_sliding_cm(self) -> None:
        engine = DualVerifierEngine()
        res = engine.verify_physics(
            "s1",
            {
                "silhouette_iou": 0.95,
                "depth_rmse_m": 0.01,
                "normal_angular_error_deg": 5.0,
                "foot_sliding_cm": 0.5,  # > 0.2 cm threshold
            },
        )
        assert res.passed is False
        assert "FOOT_SLIPPING" in res.failures
        assert "motion" in res.lora_buckets

    def test_evaluate_gate_enforces_required_keys_and_nan(self) -> None:
        v_missing = evaluate_gate(
            {"chamfer_mm": 2.0},
            profile="rsi_promotion",
            required_keys=["chamfer_mm", "depth_rmse_m"],
        )
        assert v_missing.passed is False
        assert any(v.metric_name == "depth_rmse_m" for v in v_missing.violations)

        v_nan = evaluate_gate(
            {"chamfer_mm": float("nan")},
            profile="rsi_promotion",
            required_keys=["chamfer_mm"],
        )
        assert v_nan.passed is False
