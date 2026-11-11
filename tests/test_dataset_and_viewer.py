"""Unit and integration tests for synapse.models, generators, and viewer."""

import json
from pathlib import Path
import urllib.request

import numpy as np
import pytest

from synapse.fgpp.frame import CameraParams
from synapse.generators.dataset import DatasetConfig, DatasetGenerator
from synapse.generators.rasterizer import SoftwareRasterizer
from synapse.generators.visualizer import (
    colorize_depth,
    colorize_flow,
    colorize_mask,
    colorize_normal,
    make_contact_sheet,
)
from synapse.models import list_builtin_models, load_model
from synapse.viewer.server import ViewerServer


class TestModels:
    """Test 3D model asset loading, normalization, and properties."""

    def test_list_builtin_models(self):
        models = list_builtin_models()
        assert len(models) >= 5
        assert "bunny" in models or "sphere" in models

    def test_load_procedural_model(self):
        model = load_model("sphere", normalize=True, target_radius=1.0)
        assert model.num_vertices > 0
        assert model.num_faces > 0
        assert model.normals is not None
        center, radius = model.bounding_sphere()
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-3)
        assert abs(radius - 1.0) < 1e-3

    def test_load_damaged_helmet_or_fallback(self):
        models = list_builtin_models()
        target = "damaged_helmet" if "damaged_helmet" in models else "bunny"
        model = load_model(target, normalize=True, target_radius=1.0)
        assert model.num_vertices > 100
        assert model.num_faces > 100
        center, radius = model.bounding_sphere()
        assert abs(radius - 1.0) < 1e-3

    def test_model_export_obj(self, tmp_path):
        model = load_model("cube", normalize=True)
        out_path = tmp_path / "cube.obj"
        exported = model.export_obj(out_path)
        assert exported.exists()
        content = exported.read_text(encoding="utf-8")
        assert "v " in content
        assert "f " in content


class TestRasterizer:
    """Test software multi-channel rasterizer."""

    def test_rasterizer_channels(self):
        model = load_model("cube", normalize=True, target_radius=1.0)
        from synapse.bench.camera import generate_cameras, CameraTrajectoryType

        cameras = generate_cameras(CameraTrajectoryType.ORBIT, num_frames=2, radius=2.5, width=128, height=128)
        rasterizer = SoftwareRasterizer(width=128, height=128)

        res = rasterizer.render(model, cameras[1], cameras[0])

        assert res.rgb.shape == (128, 128, 3)
        assert res.rgb.dtype == np.uint8
        assert res.depth.shape == (128, 128)
        assert res.depth.dtype == np.float32
        assert res.world_normal.shape == (128, 128, 3)
        assert res.world_normal.dtype == np.float32
        assert res.motion_vector.shape == (128, 128, 2)
        assert res.motion_vector.dtype == np.float32
        assert res.mask.shape == (128, 128)
        assert res.mask.dtype == np.uint8

        # Foreground check
        fg = res.mask > 0
        assert np.any(fg)
        assert res.depth[fg].min() > 0.5
        assert res.depth[fg].max() < 10.0

        # Normal unit length check
        norms = np.linalg.norm(res.world_normal[fg], axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-2)


class TestVisualizer:
    """Test visualization colormapping functions."""

    def test_visualizers(self):
        depth = np.full((64, 64), 2.0, dtype=np.float32)
        mask = np.ones((64, 64), dtype=np.uint8)
        normal = np.zeros((64, 64, 3), dtype=np.float32)
        normal[:, :, 2] = 1.0  # (0, 0, 1)
        flow = np.ones((64, 64, 2), dtype=np.float32)

        c_depth = colorize_depth(depth, mask)
        c_normal = colorize_normal(normal, mask)
        c_flow = colorize_flow(flow, mask)
        c_mask = colorize_mask(mask)

        assert c_depth.shape == (64, 64, 3)
        assert c_normal.shape == (64, 64, 3)
        assert c_flow.shape == (64, 64, 3)
        assert c_mask.shape == (64, 64, 3)

        sheet = make_contact_sheet(c_depth, c_depth, c_normal, c_flow, c_mask, "Test")
        assert sheet.shape == (64, 64 * 5, 3)


class TestDatasetGenerator:
    """Test end-to-end dataset generation pipeline."""

    def test_generate_dataset(self, tmp_path):
        out_dir = tmp_path / "test_dataset"
        cfg = DatasetConfig(
            model_name="cube",
            trajectory_type="orbit",
            num_frames=6,
            width=128,
            height=128,
            fps=12.0,
            radius=2.2,
            val_ratio=0.33,
            output_dir=out_dir,
        )

        gen = DatasetGenerator(cfg)
        dataset_path = gen.generate()

        assert dataset_path.exists()
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "contact_sheet.png").exists()
        assert (out_dir / "video_all.mp4").exists()
        assert (out_dir / "video_train.mp4").exists()
        assert (out_dir / "video_val.mp4").exists()

        # Check manifest contents
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["frames"]) == 6
        assert manifest["splits"]["val_count"] >= 1
        assert manifest["splits"]["train_count"] >= 1
        assert (out_dir / "fgpp" / "frame_0000.npz").exists()
        assert (out_dir / "frames" / "frame_0000.png").exists()


class TestViewerServer:
    """Test embedded HTTP viewer server."""

    def test_viewer_http_response(self, tmp_path):
        # Create minimal dataset dir with manifest.json
        manifest_data = {
            "dataset_id": "test_ds",
            "frames": [],
            "splits": {"train_count": 0, "val_count": 0},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        server = ViewerServer(dataset_dir=tmp_path, port=9876)
        url = server.start(open_browser=False)

        try:
            # Test index.html endpoint
            with urllib.request.urlopen(f"{url}/") as response:
                assert response.status == 200
                html = response.read().decode("utf-8")
                assert "SYNAPSE" in html
                assert "trajectory-canvas" in html

            # Test manifest.json endpoint
            with urllib.request.urlopen(f"{url}/manifest.json") as response:
                assert response.status == 200
                data = json.loads(response.read().decode("utf-8"))
                assert data["dataset_id"] == "test_ds"
        finally:
            server.stop()
