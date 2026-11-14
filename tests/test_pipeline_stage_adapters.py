"""Tests for pipeline stage adapter layer and domain module decoupling."""

from __future__ import annotations

import ast
from pathlib import Path

from synapse.pipeline.bundle_worker import get_registered_stages


class TestPipelineStageAdapters:
    """Verify domain modules do not import synapse.pipeline and all stages register."""

    def test_domain_modules_do_not_import_pipeline(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        domain_dirs = [
            repo_root / "python" / "synapse" / "training",
            repo_root / "python" / "synapse" / "rsi",
            repo_root / "python" / "synapse" / "inspect",
        ]
        violations: list[str] = []
        for d in domain_dirs:
            for py_file in sorted(d.glob("*.py")):
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("synapse.pipeline"):
                                violations.append(f"{py_file}: import {alias.name}")
                    elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("synapse.pipeline"):
                        violations.append(f"{py_file}: from {node.module} import ...")
        assert violations == [], f"Domain modules must not import synapse.pipeline: {violations}"

    def test_no_e402_bottom_imports_in_domain_modules(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        domain_dirs = [
            repo_root / "python" / "synapse" / "training",
            repo_root / "python" / "synapse" / "rsi",
            repo_root / "python" / "synapse" / "inspect",
        ]
        for d in domain_dirs:
            for py_file in sorted(d.glob("*.py")):
                source = py_file.read_text(encoding="utf-8")
                assert "E402" not in source, f"Found noqa: E402 in {py_file}"

    def test_all_expected_stages_registered(self) -> None:
        stages = set(get_registered_stages())
        expected = {
            "fgpp_validate",
            "neural_video_generate",
            "style_extract_and_map",
            "surgical_composite",
            "inverse_reconstruct",
            "kinematic_distill",
            "lora_preencode",
            "lora_train",
            "lora_package",
            "rsi_dual_verify",
            "asset_inspect_3d",
            "inspect_refine",
            "inspect_export_rl",
            "video_to_3d_invert",
            "inverse_render_loop",
            "anatomical_refine",
        }
        assert expected.issubset(stages)

    def test_no_monkey_patch_min_angular_span(self) -> None:
        """_MIN_ANGULAR_SPAN_DEG must not be referenced outside recon/multiview.py."""
        repo_root = Path(__file__).resolve().parent.parent
        pipeline_dir = repo_root / "python" / "synapse" / "pipeline"
        violations: list[str] = []
        for py_file in sorted(pipeline_dir.rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            if "_MIN_ANGULAR_SPAN_DEG" in source:
                violations.append(str(py_file))
        assert violations == [], (
            f"_MIN_ANGULAR_SPAN_DEG monkey-patch found in: {violations}"
        )
