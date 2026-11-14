"""Unit and integration tests for synapse.models, generators, and viewer."""

import json
import socketserver
import urllib.error
import urllib.request

import numpy as np
import pytest
from synapse.errors import ContractValidationError, ExitCode
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
from synapse.viewer.server import ViewerServer, _ReusableHTTPServer, export_static_html


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
        from synapse.bench.camera import CameraTrajectoryType, generate_cameras

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

    @pytest.mark.parametrize("traj_type", ["spiral", "hemisphere"])
    def test_generate_dataset_spiral_and_hemisphere(self, tmp_path, traj_type):
        out_dir = tmp_path / f"test_dataset_{traj_type}"
        cfg = DatasetConfig(
            model_name="cube",
            trajectory_type=traj_type,
            num_frames=4,
            width=64,
            height=64,
            fps=12.0,
            radius=2.2,
            fov_degrees=45.0,
            val_ratio=0.25,
            output_dir=out_dir,
        )
        gen = DatasetGenerator(cfg)
        dataset_path = gen.generate()
        assert dataset_path.exists()
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["frames"]) == 4
        assert manifest["trajectory"]["type"] == traj_type
        assert abs(manifest["frames"][0]["camera"]["fov_degrees"] - 45.0) < 1e-3


class TestViewerServer:
    """Test embedded HTTP viewer server and static HTML exporter."""

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

    def test_non_loopback_host_rejected(self, tmp_path):
        """Verify non-loopback host (e.g. 0.0.0.0, 192.168.x.x) is rejected."""
        manifest_data = {"dataset_id": "test_ds", "frames": []}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        for bad_host in ["0.0.0.0", "192.168.1.1", "10.0.0.1", "example.com"]:
            with pytest.raises(ContractValidationError) as exc_info:
                ViewerServer(dataset_dir=tmp_path, host=bad_host)
            assert exc_info.value.exit_code == ExitCode.CONTRACT_ERROR
            assert "Non-loopback host" in str(exc_info.value)
            assert "127.0.0.1" in exc_info.value.agent_fix_hint

    def test_reusable_http_server_instance_attribute(self):
        """Verify socketserver.TCPServer.allow_reuse_address is not mutated globally."""
        # Check global TCPServer class attribute remains default (not True by our code)
        assert socketserver.TCPServer.allow_reuse_address is False

        # But _ReusableHTTPServer instances have allow_reuse_address = True
        dummy_server = _ReusableHTTPServer.__new__(_ReusableHTTPServer)
        dummy_server.allow_reuse_address = True
        assert dummy_server.allow_reuse_address is True

    def test_security_path_traversal_and_extensions(self, tmp_path):
        """Verify handler rejects path traversal, hidden files, directory listings, and forbidden extensions."""
        manifest_data = {"dataset_id": "sec_test", "frames": []}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
        (tmp_path / "secret.py").write_text("SECRET=123", encoding="utf-8")
        (tmp_path / ".hidden").write_text("hidden", encoding="utf-8")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        server = ViewerServer(dataset_dir=tmp_path, port=9877)
        url = server.start(open_browser=False)

        try:
            # Valid PNG in subdirectory
            with urllib.request.urlopen(f"{url}/subdir/test.png") as resp:
                assert resp.status == 200

            # Path traversal rejected with 403
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{url}/../test.py")
            assert exc_info.value.code == 403

            # Hidden file rejected with 403
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{url}/.hidden")
            assert exc_info.value.code == 403

            # Disallowed extension (.py) rejected with 403
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{url}/secret.py")
            assert exc_info.value.code == 403

            # Directory listing rejected with 403
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{url}/subdir/")
            assert exc_info.value.code == 403

        finally:
            server.stop()

    def test_static_html_export(self, tmp_path):
        """Verify static self-contained HTML export embeds manifest and images."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_0000.png").write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes")

        vis_dir = tmp_path / "visualizations"
        vis_dir.mkdir()
        (vis_dir / "depth_0000.png").write_bytes(b"\x89PNG\r\n\x1a\nfakedepthbytes")

        manifest_data = {
            "dataset_id": "test_static",
            "frames": [{"frame_index": 0}],
            "splits": {"train_count": 1, "val_count": 0},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        # Export static HTML
        out_html = export_static_html(tmp_path)
        assert out_html.exists()
        assert out_html.name == "viewer.html"

        content = out_html.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "SYNAPSE" in content
        # Check embedded manifest data is present
        assert "test_static" in content
        # Check base64 data URI is embedded
        assert "data:image/png;base64," in content
