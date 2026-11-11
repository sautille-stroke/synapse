"""Tests for synapse.bench — Synapse-Bench evaluation dataset."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from synapse.bench import (
    CROSS_CONDITIONS,
    RENDERING_CONDITION_FACTORS,
    SINGLE_FACTOR_CONDITIONS,
    CameraTrajectoryType,
    ConditionLevel,
    Conditions,
    ExpectedMetrics,
    Scenario,
    ScenarioTier,
    generate_cameras,
    list_scenarios,
    load_scenario,
)
from synapse.bench.procedural import (
    ProceduralMesh,
    ProceduralMeshType,
    generate_procedural_mesh,
)
from synapse.fgpp import CameraParams, FGPPFrame, FGPPSequence


# =========================================================================
# Scenario
# =========================================================================


class TestScenario:
    """Test Scenario data model and serialisation."""

    def test_scenario_create(self):
        s = Scenario(
            id="test_cube",
            tier=ScenarioTier.T1_RIGID,
            description="Test cube scenario",
        )
        assert s.id == "test_cube"
        assert s.tier == ScenarioTier.T1_RIGID
        assert s.width == 1024

    def test_scenario_roundtrip_json(self, tmp_path):
        s = Scenario(
            id="test_sphere",
            tier=ScenarioTier.T1_RIGID,
            description="Sphere test",
            tags=["test", "sphere"],
            expected=ExpectedMetrics(chamfer_mm=3.0, psnr_db=28.0),
        )

        path = s.save(tmp_path / "test_sphere.json")
        assert path.is_file()

        loaded = Scenario.load(path)
        assert loaded.id == s.id
        assert loaded.tier == s.tier
        assert loaded.expected.chamfer_mm == 3.0
        assert loaded.expected.psnr_db == 28.0
        assert loaded.tags == ["test", "sphere"]

    def test_scenario_to_dict(self):
        s = Scenario(
            id="test",
            tier=ScenarioTier.T2_DYNAMIC,
            conditions={"F01_viewpoint": "L2"},
        )
        d = s.to_dict()
        assert d["tier"] == "T2"
        assert d["conditions"]["F01_viewpoint"] == "L2"

    def test_list_builtin_scenarios(self):
        """Verify built-in scenario JSONs are discoverable."""
        scenarios = list_scenarios()
        assert len(scenarios) >= 5, f"Expected >= 5 built-in scenarios, got {len(scenarios)}"

        ids = [s.id for s in scenarios]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_list_scenarios_filter_tier(self):
        t1 = list_scenarios(tier=ScenarioTier.T1_RIGID)
        assert all(s.tier == ScenarioTier.T1_RIGID for s in t1)

    def test_load_scenario_by_id(self):
        s = load_scenario("t1_rigid_cube_orbit")
        assert s.id == "t1_rigid_cube_orbit"
        assert s.tier == ScenarioTier.T1_RIGID
        assert s.asset.source.value == "procedural"
        assert s.asset.id == "cube"

    def test_load_scenario_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_scenario("nonexistent_scenario_xyz")


# =========================================================================
# Conditions
# =========================================================================


class TestConditions:
    """Test rendering condition factor system."""

    def test_factor_catalogue_complete(self):
        assert len(RENDERING_CONDITION_FACTORS) == 11
        for fid, fdef in RENDERING_CONDITION_FACTORS.items():
            assert "name" in fdef
            assert "levels" in fdef
            levels = fdef["levels"]
            assert set(levels.keys()) == {"L1", "L2", "L3", "L4"}, (
                f"Factor {fid} missing levels: {set(levels.keys())}"
            )

    def test_single_factor_conditions_count(self):
        assert len(SINGLE_FACTOR_CONDITIONS) == 33

    def test_cross_conditions_exist(self):
        assert len(CROSS_CONDITIONS) >= 6

    def test_conditions_roundtrip(self):
        c = Conditions(
            F01_viewpoint=ConditionLevel.L2,
            F03_lighting=ConditionLevel.L3,
        )
        d = c.to_dict()
        assert d == {"F01_viewpoint": "L2", "F03_lighting": "L3"}

        c2 = Conditions.from_dict(d)
        assert c2.F01_viewpoint == ConditionLevel.L2
        assert c2.F03_lighting == ConditionLevel.L3
        assert c2.F02_background is None

    def test_conditions_resolve(self):
        c = Conditions(F01_viewpoint=ConditionLevel.L3)
        resolved = c.resolve()
        assert "F01_viewpoint" in resolved
        assert resolved["F01_viewpoint"]["level"] == "L3"
        assert resolved["F01_viewpoint"]["label"] == "sparse_8view"
        assert resolved["F03_lighting"]["level"] == "L1"


# =========================================================================
# Camera
# =========================================================================


class TestCamera:
    """Test camera trajectory generators."""

    def test_orbit_basic(self):
        cameras = generate_cameras("orbit", num_frames=10, radius=3.0)
        assert len(cameras) == 10
        assert all(isinstance(c, CameraParams) for c in cameras)

    def test_orbit_positions_form_circle(self):
        cameras = generate_cameras(
            "orbit", num_frames=36, radius=5.0,
            elevation_deg=0.0, azimuth_range_deg=360.0,
        )
        for cam in cameras:
            pos = cam.position()
            dist = np.linalg.norm(pos)
            assert abs(dist - 5.0) < 0.01, f"Expected radius 5.0, got {dist}"

    def test_dolly_distance_changes(self):
        cameras = generate_cameras(
            "dolly", num_frames=10,
            start_distance=5.0, end_distance=2.0,
        )
        d_first = np.linalg.norm(cameras[0].position())
        d_last = np.linalg.norm(cameras[-1].position())
        assert d_first > d_last, "Dolly should move closer"

    def test_static_no_drift(self):
        cameras = generate_cameras(
            "static", num_frames=10, drift_amplitude=0.0,
        )
        pos_first = cameras[0].position()
        for cam in cameras[1:]:
            np.testing.assert_allclose(cam.position(), pos_first, atol=1e-10)

    def test_tracking_moves_laterally(self):
        cameras = generate_cameras(
            "tracking", num_frames=10,
            start=[-3, 1.5, 0], end=[3, 1.5, 0],
        )
        assert cameras[0].position()[0] < cameras[-1].position()[0]

    def test_arc_elevation_changes(self):
        cameras = generate_cameras(
            "arc", num_frames=10,
            elevation_start_deg=0.0, elevation_end_deg=60.0,
        )
        y_first = cameras[0].position()[1]
        y_last = cameras[-1].position()[1]
        assert y_last > y_first, "Arc should increase elevation"

    def test_pan_generates_frames(self):
        cameras = generate_cameras("pan", num_frames=10)
        assert len(cameras) == 10

    def test_from_scenario(self):
        scenario = load_scenario("t1_rigid_cube_orbit")
        cameras = generate_cameras(scenario)
        assert len(cameras) == scenario.camera_params.get("num_frames", 120)

    def test_view_matrix_invertible(self):
        cameras = generate_cameras("orbit", num_frames=5, radius=3.0)
        for cam in cameras:
            product = cam.view_matrix @ cam.world_matrix
            np.testing.assert_allclose(product, np.eye(4), atol=1e-10)


# =========================================================================
# Procedural Geometry
# =========================================================================


class TestProceduralMesh:
    """Test procedural mesh generators."""

    @pytest.mark.parametrize("mesh_type", [
        "cube", "sphere", "cylinder", "torus", "plane",
    ])
    def test_generate_mesh(self, mesh_type):
        mesh = generate_procedural_mesh(mesh_type)
        assert isinstance(mesh, ProceduralMesh)
        assert mesh.num_vertices > 0
        assert mesh.num_faces > 0
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert mesh.normals.shape == mesh.vertices.shape
        assert mesh.mesh_type == mesh_type

    def test_cube_face_count(self):
        mesh = generate_procedural_mesh("cube")
        assert mesh.num_faces == 12

    def test_sphere_normals_unit(self):
        mesh = generate_procedural_mesh("sphere", radius=1.0, segments=16, rings=8)
        norms = np.linalg.norm(mesh.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_torus_topology(self):
        mesh = generate_procedural_mesh("torus", major_segments=8, minor_segments=4)
        assert mesh.num_faces > 0
        assert mesh.num_vertices > 0

    def test_mesh_has_uvs(self):
        for mesh_type in ["cube", "sphere", "cylinder", "torus", "plane"]:
            mesh = generate_procedural_mesh(mesh_type)
            assert mesh.uvs is not None, f"{mesh_type} should have UVs"
            assert mesh.uvs.shape == (mesh.num_vertices, 2)

    def test_save_obj(self, tmp_path):
        mesh = generate_procedural_mesh("cube")
        path = mesh.save_obj(tmp_path / "cube.obj")
        assert path.is_file()
        content = path.read_text()
        assert "v " in content
        assert "f " in content
        assert "vn " in content

    def test_bounding_box(self):
        mesh = generate_procedural_mesh("cube", size=2.0)
        bb_min, bb_max = mesh.bounding_box()
        np.testing.assert_allclose(bb_min, [-1, -1, -1], atol=1e-10)
        np.testing.assert_allclose(bb_max, [1, 1, 1], atol=1e-10)

    def test_invalid_mesh_type(self):
        with pytest.raises(ValueError):
            generate_procedural_mesh("nonexistent_type")


# =========================================================================
# FGPP
# =========================================================================


class TestFGPP:
    """Test FGPP data classes."""

    def test_fgpp_frame_create(self):
        cam = CameraParams(
            fx=500, fy=500, cx=512, cy=512,
            width=1024, height=1024,
        )
        depth = np.ones((1024, 1024), dtype=np.float32) * 5.0
        frame = FGPPFrame(camera=cam, depth=depth, frame_index=0)

        assert frame.height == 1024
        assert frame.width == 1024
        assert "depth" in frame.available_channels()
        assert len(frame.available_channels()) == 1

    def test_fgpp_frame_with_all_channels(self):
        cam = CameraParams(
            fx=500, fy=500, cx=256, cy=256,
            width=512, height=512,
        )
        frame = FGPPFrame(
            camera=cam,
            depth=np.ones((512, 512), dtype=np.float32),
            direct_radiance=np.zeros((512, 512, 3), dtype=np.float16),
            world_normal=np.zeros((512, 512, 3), dtype=np.float16),
            motion_vector=np.zeros((512, 512, 2), dtype=np.float16),
            base_color=np.zeros((512, 512, 3), dtype=np.float16),
            roughness_metallic=np.zeros((512, 512, 2), dtype=np.float16),
            cryptomatte_id=np.zeros((512, 512), dtype=np.uint32),
        )
        channels = frame.available_channels()
        assert len(channels) == 7

    def test_fgpp_sequence_metadata(self, tmp_path):
        cam = CameraParams(
            fx=500, fy=500, cx=256, cy=256,
            width=512, height=512,
        )
        frames = [
            FGPPFrame(
                camera=cam,
                depth=np.ones((512, 512), dtype=np.float32) * (i + 1),
                frame_index=i,
            )
            for i in range(5)
        ]
        seq = FGPPSequence(frames=frames, fps=24.0, scene_id="test")

        assert seq.num_frames == 5
        assert abs(seq.duration_seconds - 5 / 24.0) < 1e-6

        meta_path = seq.save_metadata(tmp_path / "fgpp")
        assert meta_path.is_file()

        meta = json.loads(meta_path.read_text())
        assert meta["fps"] == 24.0
        assert meta["num_frames"] == 5
        assert len(meta["frames"]) == 5

    def test_camera_params_roundtrip(self):
        cam = CameraParams(
            fx=500, fy=500, cx=256, cy=256,
            width=512, height=512,
            fov_degrees=60.0,
        )
        d = cam.to_dict()
        cam2 = CameraParams.from_dict(d)
        assert cam2.fx == cam.fx
        assert cam2.width == cam.width
        np.testing.assert_allclose(cam2.view_matrix, cam.view_matrix)
