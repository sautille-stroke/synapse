"""Tests for 3D model evaluation report and evaluation script."""

from __future__ import annotations

import base64
import json
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from synapse.bench.procedural import ProceduralMeshType, generate_procedural_mesh
from synapse.inspect.eval_report import (
    EvalReportData,
    ModelEvalResult,
    TopologyResult,
    _normalize_metric,
    generate_eval_html_report,
)
from synapse.recon.geometry import check_manifold, check_watertight


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_tiny_png() -> str:
    """Create a base64-encoded 4x4 red PNG for testing."""
    w, h = 4, 4
    raw = b""
    for _ in range(h):
        raw += b"\x00" + b"\xff\x00\x00" * w
    compressed = zlib.compress(raw)

    def chunk(ct: bytes, data: bytes) -> bytes:
        r = ct + data
        return (
            struct.pack(">I", len(data))
            + r
            + struct.pack(">I", zlib.crc32(r) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def _make_topology(
    *,
    manifold: bool = True,
    watertight: bool = True,
    verts: int = 100,
    faces: int = 200,
) -> TopologyResult:
    return TopologyResult(
        is_manifold=manifold,
        is_watertight=watertight,
        num_vertices=verts,
        num_faces=faces,
        num_connected_components=1,
        has_degenerate_faces=False,
        num_boundary_edges=0,
    )


def _make_model_result(
    name: str = "test_model.obj",
    passed: bool = True,
    n_screenshots: int = 2,
) -> ModelEvalResult:
    tiny = _make_tiny_png()
    return ModelEvalResult(
        model_name=name,
        model_path=f"/tmp/{name}",
        topology=_make_topology(manifold=passed, watertight=passed),
        metrics={
            "is_manifold": 1.0 if passed else 0.0,
            "is_watertight": 1.0 if passed else 0.0,
            "chamfer_mm": 2.0 if passed else 15.0,
        },
        gate_results={
            "is_manifold": passed,
            "is_watertight": passed,
            "chamfer_mm": passed,
        },
        screenshots_base64=[tiny] * n_screenshots,
        screenshot_labels=[f"View {i}" for i in range(n_screenshots)],
        overall_passed=passed,
    )


# -------------------------------------------------------------------
# Unit tests: data model
# -------------------------------------------------------------------


class TestDataModel:
    def test_topology_result_creation(self) -> None:
        t = TopologyResult(
            is_manifold=True,
            is_watertight=False,
            num_vertices=1000,
            num_faces=2000,
            num_connected_components=2,
            has_degenerate_faces=True,
            num_boundary_edges=10,
        )
        assert t.is_manifold is True
        assert t.is_watertight is False
        assert t.num_vertices == 1000
        assert t.num_faces == 2000
        assert t.num_connected_components == 2
        assert t.has_degenerate_faces is True
        assert t.num_boundary_edges == 10

    def test_model_eval_result_creation(self) -> None:
        m = _make_model_result("cube.obj", passed=True)
        assert m.overall_passed is True
        assert m.model_name == "cube.obj"
        assert "is_manifold" in m.metrics
        assert len(m.screenshots_base64) == 2

    def test_eval_report_data_creation(self) -> None:
        data = EvalReportData(
            timestamp="2024-01-01 00:00:00 UTC",
            profile_name="TRACK_A_ASSET",
            models=[_make_model_result()],
        )
        assert len(data.models) == 1
        assert data.profile_name == "TRACK_A_ASSET"


# -------------------------------------------------------------------
# Unit tests: HTML report generation
# -------------------------------------------------------------------


class TestHTMLReport:
    def test_generate_basic_report(self, tmp_path: Path) -> None:
        data = EvalReportData(
            timestamp="2024-01-01 00:00:00 UTC",
            profile_name="TRACK_A_ASSET",
            models=[
                _make_model_result("good.obj", passed=True),
                _make_model_result("bad.obj", passed=False),
            ],
        )
        out = tmp_path / "report.html"
        result = generate_eval_html_report(data, out)

        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert len(content) > 100

        # Check key content
        assert "Synapse 3D Model Evaluation Report" in content
        assert "good.obj" in content
        assert "bad.obj" in content
        assert "PASS" in content
        assert "FAIL" in content
        assert "<svg" in content  # Radar chart
        assert "data:image/png;base64," in content  # Screenshots

    def test_generate_empty_report(self, tmp_path: Path) -> None:
        data = EvalReportData(
            timestamp="2024-01-01 00:00:00 UTC",
            profile_name="TRACK_B_VIDEO",
            models=[],
        )
        out = tmp_path / "empty.html"
        generate_eval_html_report(data, out)

        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Synapse 3D Model Evaluation Report" in content
        assert "0 models" in content

    def test_report_contains_metrics_table(self, tmp_path: Path) -> None:
        model = _make_model_result("metrics_test.obj")
        data = EvalReportData(
            timestamp="2024-01-01 00:00:00 UTC",
            profile_name="TRACK_A_ASSET",
            models=[model],
        )
        out = tmp_path / "metrics.html"
        generate_eval_html_report(data, out)
        content = out.read_text(encoding="utf-8")

        for metric_name in model.metrics:
            assert metric_name in content

    def test_report_screenshot_gallery(self, tmp_path: Path) -> None:
        model = _make_model_result("gallery_test.obj", n_screenshots=4)
        model.screenshot_labels = [
            "Front", "Right", "Back", "Left",
        ]
        data = EvalReportData(
            timestamp="2024-01-01 00:00:00 UTC",
            profile_name="TRACK_A_ASSET",
            models=[model],
        )
        out = tmp_path / "gallery.html"
        generate_eval_html_report(data, out)
        content = out.read_text(encoding="utf-8")

        for label in model.screenshot_labels:
            assert label in content

    def test_report_creates_parent_dirs(self, tmp_path: Path) -> None:
        data = EvalReportData(
            timestamp="2024-01-01 00:00:00 UTC",
            profile_name="TRACK_A_ASSET",
            models=[],
        )
        out = tmp_path / "deep" / "nested" / "report.html"
        generate_eval_html_report(data, out)
        assert out.exists()


# -------------------------------------------------------------------
# Unit tests: topology checks with procedural meshes
# -------------------------------------------------------------------


class TestTopologyChecks:
    def test_cube_is_manifold(self) -> None:
        cube = generate_procedural_mesh(ProceduralMeshType.CUBE)
        assert check_manifold(cube.vertices, cube.faces) is True

    def test_cube_not_watertight_due_to_split_vertices(self) -> None:
        # Procedural cube duplicates vertices per face for sharp normals,
        # so edges are not shared — check_watertight returns False.
        cube = generate_procedural_mesh(ProceduralMeshType.CUBE)
        assert check_watertight(cube.vertices, cube.faces) is False

    def test_sphere_is_manifold(self) -> None:
        sphere = generate_procedural_mesh(
            ProceduralMeshType.SPHERE, segments=16, rings=8,
        )
        assert check_manifold(sphere.vertices, sphere.faces) is True

    def test_sphere_not_watertight_due_to_seam(self) -> None:
        # UV-sphere has a seam and poles with split vertices,
        # so it is not geometrically watertight.
        sphere = generate_procedural_mesh(
            ProceduralMeshType.SPHERE, segments=16, rings=8,
        )
        assert check_watertight(sphere.vertices, sphere.faces) is False

    def test_plane_not_watertight(self) -> None:
        plane = generate_procedural_mesh(ProceduralMeshType.PLANE)
        # A plane is open (not closed), so it should NOT be watertight
        assert check_watertight(plane.vertices, plane.faces) is False


# -------------------------------------------------------------------
# Unit tests: metric normalisation
# -------------------------------------------------------------------


class TestNormalizeMetric:
    def test_chamfer_zero(self) -> None:
        assert _normalize_metric("chamfer_mm", 0.0) == pytest.approx(1.0)

    def test_chamfer_ten(self) -> None:
        assert _normalize_metric("chamfer_mm", 10.0) == pytest.approx(0.0)

    def test_chamfer_five(self) -> None:
        assert _normalize_metric("chamfer_mm", 5.0) == pytest.approx(0.5)

    def test_manifold_true(self) -> None:
        assert _normalize_metric("is_manifold", 1.0) == pytest.approx(1.0)

    def test_manifold_false(self) -> None:
        assert _normalize_metric("is_manifold", 0.0) == pytest.approx(0.0)

    def test_silhouette_iou(self) -> None:
        assert _normalize_metric("silhouette_iou", 0.95) == pytest.approx(
            0.95
        )

    def test_volume_breathing_zero(self) -> None:
        assert _normalize_metric("volume_breathing", 0.0) == pytest.approx(
            1.0
        )

    def test_volume_breathing_half(self) -> None:
        assert _normalize_metric("volume_breathing", 0.25) == pytest.approx(
            0.5
        )


# -------------------------------------------------------------------
# Integration tests: CLI script
# -------------------------------------------------------------------


class TestEvalScript:
    @pytest.fixture()
    def script_path(self) -> Path:
        return Path(__file__).parent.parent / "scripts" / "evaluate_3d_models.py"

    @pytest.fixture()
    def model_dir(self, tmp_path: Path) -> Path:
        """Create a temp dir with a procedural cube saved as .obj."""
        d = tmp_path / "models"
        d.mkdir()
        cube = generate_procedural_mesh(ProceduralMeshType.CUBE)
        cube.save_obj(d / "test_cube.obj")
        return d

    def test_smoke_run(
        self,
        script_path: Path,
        model_dir: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--input-dir", str(model_dir),
                "--output-dir", str(output_dir),
                "--resolution", "64",
                "--num-views", "4",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode in (0, 40)

        # Parse JSON output
        data = json.loads(result.stdout)
        assert "status" in data
        assert "models" in data
        assert "report_path" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "test_cube.obj"

        # Check report file exists
        report_path = Path(data["report_path"])
        assert report_path.exists()
        html_content = report_path.read_text(encoding="utf-8")
        assert "Synapse 3D Model Evaluation Report" in html_content

    def test_missing_input_dir(
        self,
        script_path: Path,
        tmp_path: Path,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--input-dir", str(tmp_path / "nonexistent"),
                "--output-dir", str(tmp_path / "output"),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 20

    def test_empty_input_dir(
        self,
        script_path: Path,
        tmp_path: Path,
    ) -> None:
        empty = tmp_path / "empty_models"
        empty.mkdir()
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--input-dir", str(empty),
                "--output-dir", str(tmp_path / "output"),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 20
