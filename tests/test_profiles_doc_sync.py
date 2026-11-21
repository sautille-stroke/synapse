"""Tests for threshold SSOT synchronization with DESIGN_REVIEW_AND_ALIGNMENT.md and ARCHITECTURE.md."""

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
    """Parse §3.4 threshold matrix from docs/ARCHITECTURE.md (or DESIGN_REVIEW_AND_ALIGNMENT.md)."""
    repo_root = Path(__file__).resolve().parent.parent
    doc_path = repo_root / "docs" / "ARCHITECTURE.md"
    if not doc_path.is_file():
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
    """Verify synapse.metrics.profiles matches ARCHITECTURE.md §3.4."""

    def test_doc_table_matches_profiles(self) -> None:
        doc_profiles = _parse_doc_threshold_table()
        # Core roundtrip / geometry / style metrics governed by §3.4
        governed_metrics = [
            "temporal_warp_error",
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
            assert TRACK_B_VIDEO[metric] == pytest.approx(doc_profiles["track_b_video"][metric]), (
                f"track_b_video.{metric} mismatch"
            )
            assert TRACK_A_ASSET[metric] == pytest.approx(doc_profiles["track_a_asset"][metric]), (
                f"track_a_asset.{metric} mismatch"
            )
            assert RSI_PROMOTION[metric] == pytest.approx(doc_profiles["rsi_promotion"][metric]), (
                f"rsi_promotion.{metric} mismatch"
            )

    def test_rsi_promotion_target_stricter_or_equal(self) -> None:
        from synapse.metrics.profiles import _METRIC_DIRECTIONS

        for key, val in RSI_PROMOTION.items():
            assert key in RSI_PROMOTION_TARGET
            direction = _METRIC_DIRECTIONS.get(key, "lower_is_better")
            if direction == "higher_is_better":
                assert RSI_PROMOTION_TARGET[key] >= val
            else:
                assert RSI_PROMOTION_TARGET[key] <= val

    def test_domain_thresholds_derive_from_profiles(self) -> None:
        lora_t = LoRAGateThresholds.from_profile("rsi_promotion")
        assert lora_t.chamfer_mm == RSI_PROMOTION["chamfer_mm"]
        assert lora_t.normal_angular_error_deg == RSI_PROMOTION["normal_angular_error_deg"]
        assert lora_t.border_ring_ciede2000 == RSI_PROMOTION["border_ring_ciede2000"]

        phys_t = PhysicsThresholds.from_profile("rsi_promotion")
        assert phys_t.foot_sliding_cm == RSI_PROMOTION["foot_sliding_cm"]
        assert phys_t.silhouette_iou == RSI_PROMOTION["silhouette_iou"]
        assert phys_t.depth_rmse_m == RSI_PROMOTION["depth_rmse_m"]
        assert phys_t.normal_angular_error_deg == RSI_PROMOTION["normal_angular_error_deg"]

        rt_gate = RoundtripGate.from_profile("track_a_asset")
        assert rt_gate.volume_breathing_pct == TRACK_A_ASSET["volume_breathing_pct"]

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

    def test_architecture_doc_table_matches_profiles(self) -> None:
        """Verify docs/ARCHITECTURE.md §3.4 table matches profiles.py."""
        repo_root = Path(__file__).resolve().parent.parent
        doc_path = repo_root / "docs" / "ARCHITECTURE.md"
        assert doc_path.is_file(), "docs/ARCHITECTURE.md must exist as canonical SSOT"
        text = doc_path.read_text(encoding="utf-8")
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
            metric = match.group(1)
            m_b = num_pattern.search(match.group(2))
            m_a = num_pattern.search(match.group(3))
            m_r = num_pattern.search(match.group(4))
            if m_b and m_a and m_r:
                parsed["track_b_video"][metric] = float(m_b.group(1))
                parsed["track_a_asset"][metric] = float(m_a.group(1))
                parsed["rsi_promotion"][metric] = float(m_r.group(1))

        governed_metrics = [
            "temporal_warp_error",
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
            assert TRACK_B_VIDEO[metric] == pytest.approx(parsed["track_b_video"][metric])
            assert TRACK_A_ASSET[metric] == pytest.approx(parsed["track_a_asset"][metric])
            assert RSI_PROMOTION[metric] == pytest.approx(parsed["rsi_promotion"][metric])

    def test_cli_subcommands_documented_in_architecture(self) -> None:
        """Verify all registered CLI domain subcommands are present in docs/ARCHITECTURE.md."""
        repo_root = Path(__file__).resolve().parent.parent
        arch_text = (repo_root / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

        expected_subcommands = [
            "fgpp",
            "worker",
            "lora",
            "inspect",
            "rsi",
            "recon",
            "refine",
            "generate",
            "view",
        ]
        for cmd in expected_subcommands:
            assert f"`synapse` | `{cmd}`" in arch_text or f"synapse {cmd}" in arch_text, (
                f"CLI subcommand {cmd!r} is missing from docs/ARCHITECTURE.md §3.2"
            )

    def test_import_dag_tier_map_matches_architecture_tree(self) -> None:
        """Verify all packages in tools/import_dag_guard.py TIER_MAP appear in ARCHITECTURE.md tree."""
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tools.import_dag_guard import TIER_MAP

        arch_text = (repo_root / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        for pkg in TIER_MAP:
            assert (
                f"── {pkg}/" in arch_text
                or f"── {pkg}.py" in arch_text
                or f"`{pkg}`" in arch_text
                or f"synapse.{pkg}" in arch_text
            ), f"TIER_MAP package {pkg!r} not referenced in docs/ARCHITECTURE.md"

    def test_style_descriptor_fields_documented(self) -> None:
        """Verify StyleDescriptor fields are reflected in docs."""
        repo_root = Path(__file__).resolve().parent.parent
        arch_text = (repo_root / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        wire_text = (repo_root / "docs" / "WIRE_CONTRACTS.md").read_text(encoding="utf-8")
        combined = arch_text + "\n" + wire_text

        core_fields = ["working_colorspace", "color_temp_kelvin", "exposure_ev"]
        for field in core_fields:
            assert field in combined, (
                f"StyleDescriptor field {field!r} missing from documentation"
            )

    def test_wire_contracts_15ch_table_matches_fgpp_schema(self) -> None:
        """Verify WIRE_CONTRACTS.md §1.1 channel table aligns with FGPP_CHANNEL_SPECS."""
        from synapse.fgpp.schema import get_channel_names

        repo_root = Path(__file__).resolve().parent.parent
        wire_path = repo_root / "docs" / "WIRE_CONTRACTS.md"
        assert wire_path.is_file(), "docs/WIRE_CONTRACTS.md must exist"
        wire_text = wire_path.read_text(encoding="utf-8")

        # Parse channel names from the markdown table rows: | `ChannelName` | ...
        channel_pattern = re.compile(r"^\|\s*`([A-Za-z]+\.[A-Z])`\s*\|", re.MULTILINE)
        doc_channels = [m.group(1) for m in channel_pattern.finditer(wire_text)]

        code_channels = get_channel_names()
        assert len(doc_channels) == 15, (
            f"WIRE_CONTRACTS.md should list 15 channels, found {len(doc_channels)}: {doc_channels}"
        )
        assert doc_channels == code_channels, (
            f"Channel name mismatch:\n  Doc:  {doc_channels}\n  Code: {code_channels}"
        )

        # Verify 34 B/px is stated in doc
        assert "34 B/px" in wire_text or "34 字节" in wire_text or "34B/px" in wire_text, (
            "WIRE_CONTRACTS.md must state 34 B/px total"
        )

    def test_wire_contracts_shm_header_offsets(self) -> None:
        """Verify WIRE_CONTRACTS.md §4.1 SHM header offsets match test constants."""
        repo_root = Path(__file__).resolve().parent.parent
        wire_path = repo_root / "docs" / "WIRE_CONTRACTS.md"
        assert wire_path.is_file()
        wire_text = wire_path.read_text(encoding="utf-8")

        # §4.1 declares canonical offsets; verify key ones are present
        expected_offsets = {
            "0x00": "seqlock_sequence",
            "0x04": "width",
            "0x08": "height",
        }
        for offset, field_substr in expected_offsets.items():
            assert offset in wire_text, (
                f"SHM offset {offset} for {field_substr} missing from WIRE_CONTRACTS.md §4.1"
            )
            # Verify the offset line actually references the field
            offset_line_pattern = re.compile(
                rf"Offset\s+{re.escape(offset)}.*{re.escape(field_substr)}",
                re.IGNORECASE,
            )
            assert offset_line_pattern.search(wire_text), (
                f"SHM offset {offset} line does not reference '{field_substr}' in WIRE_CONTRACTS.md"
            )

    def test_readme_status_markers_clean_and_released(self) -> None:
        """Verify README.md contains no unmerged '待 PR-' markers and shows Release v4.0.0."""
        repo_root = Path(__file__).resolve().parent.parent
        readme_text = (repo_root / "README.md").read_text(encoding="utf-8")

        assert "Release v4.0.0" in readme_text, "README.md must announce Release v4.0.0"
        assert "待 PR-" not in readme_text, (
            "README.md status table still contains unmerged '待 PR-' placeholder markers"
        )
        assert "🟡" not in readme_text, (
            "README.md status table still contains yellow in-progress 🟡 markers"
        )
