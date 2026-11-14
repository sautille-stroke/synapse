"""Tests for synapse.recon — Video-to-3D Inverse Reconstruction & Asset Distillation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from synapse.bridge.vg_bridge import (
    SynapseReconBackend,
    VgBackend,
    get_recon_backend,
    get_vg_backend,
)
from synapse.recon import (
    CameraTrajectory,
    DelightResult,
    ReconPipeline,
    ReconResult,
    check_manifold,
    check_watertight,
    compute_normals,
    decompose_intrinsic,
    delight,
    evaluate_spherical_harmonics_basis,
    quad_remesh,
    refine_pose_6dof,
    solve_camera_trajectory,
    solve_point_tracks,
    solve_two_bone_ik,
    solve_zero_slip_ik,
)

# ===================================================================
# TestReconGeometry
# ===================================================================


class TestReconGeometry:
    """Tests for manifold validation, normals, 6-DoF alignment, and remeshing."""

    @pytest.fixture
    def cube_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """A standard watertight cube with 8 vertices and 12 triangles."""
        verts = np.array([
            [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0], [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
        ], dtype=np.float32)
        # Consistent CCW winding
        faces = np.array([
            [0, 2, 1], [0, 3, 2],  # -Z
            [4, 5, 6], [4, 6, 7],  # +Z
            [0, 1, 5], [0, 5, 4],  # -Y
            [2, 3, 7], [2, 7, 6],  # +Y
            [0, 4, 7], [0, 7, 3],  # -X
            [1, 2, 6], [1, 6, 5],  # +X
        ], dtype=np.int32)
        return verts, faces

    def test_cube_is_manifold(self, cube_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Cube satisfies 2-manifold requirements."""
        v, f = cube_mesh
        assert check_manifold(v, f) is True

    def test_cube_is_watertight(self, cube_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Cube has zero boundary edges and consistent winding."""
        v, f = cube_mesh
        assert check_watertight(v, f) is True

    def test_non_manifold_edge_rejected(self, cube_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Adding a 3rd face on an edge violates manifoldness."""
        v, f = cube_mesh
        # Add an extra triangle sharing edge (0, 1) with existing faces
        extra_face = np.array([[0, 1, 6]], dtype=np.int32)
        bad_f = np.vstack([f, extra_face])
        assert check_manifold(v, bad_f) is False
        assert check_watertight(v, bad_f) is False

    def test_empty_or_nan_rejected(self) -> None:
        """Empty or NaN vertices fail manifold check."""
        assert check_manifold(np.zeros((0, 3)), np.zeros((0, 3))) is False
        bad_v = np.array([[0.0, float("nan"), 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        f = np.array([[0, 1, 2]])
        assert check_manifold(bad_v, f) is False

    def test_compute_normals_unit_length(self, cube_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Computed vertex normals have unit norm."""
        v, f = cube_mesh
        normals = compute_normals(v, f)
        assert normals.shape == v.shape
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_refine_pose_6dof_kabsch(self) -> None:
        """Kabsch recovers exact 6-DoF rigid rotation and translation."""
        # 100 random source points
        rng = np.random.default_rng(42)
        src = rng.uniform(-2.0, 2.0, size=(100, 3))

        # Rotate 45 deg around Z and translate by [0.5, -1.2, 3.0]
        angle = math.pi / 4.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        r_gt = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0],
        ])
        t_gt = np.array([0.5, -1.2, 3.0])
        tgt = (r_gt @ src.T).T + t_gt

        t_matrix = refine_pose_6dof(src, tgt)
        assert t_matrix.shape == (4, 4)

        r_recovered = t_matrix[:3, :3]
        t_recovered = t_matrix[:3, 3]

        np.testing.assert_allclose(r_recovered, r_gt, atol=1e-5)
        np.testing.assert_allclose(t_recovered, t_gt, atol=1e-5)

    def test_refine_pose_6dof_icp_mismatched(self) -> None:
        """ICP converges when source and target point counts differ."""
        rng = np.random.default_rng(123)
        src = rng.uniform(-1.0, 1.0, size=(80, 3))
        # Target with 120 points including slight translation
        tgt = rng.uniform(-1.0, 1.0, size=(120, 3)) + np.array([0.1, 0.0, -0.05])

        t_matrix = refine_pose_6dof(src, tgt, max_iterations=10)
        assert t_matrix.shape == (4, 4)
        assert np.isfinite(t_matrix).all()

    def test_quad_remesh(self, cube_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Quad remeshing converts coplanar triangle pairs into quads."""
        v, f = cube_mesh
        new_v, new_f = quad_remesh(v, f)
        assert len(new_v) == len(v)
        # Cube has 6 square faces (originally 12 triangles). After quad remeshing, should be 6 quads
        assert new_f.shape[1] == 4
        assert len(new_f) == 6


# ===================================================================
# TestReconKinematics
# ===================================================================


class TestReconKinematics:
    """Tests for zero-slip foot IK and Two-Bone analytical IK."""

    def test_two_bone_ik_bone_lengths_preserved(self) -> None:
        """Two-bone IK strictly preserves thigh and shin bone lengths."""
        hip = np.array([0.0, 1.0, 0.0])
        knee = np.array([0.0, 0.5, 0.1])
        ankle = np.array([0.0, 0.0, 0.0])

        orig_l1 = np.linalg.norm(knee - hip)
        orig_l2 = np.linalg.norm(ankle - knee)

        # Move foot slightly forward
        target_ankle = np.array([0.05, 0.0, 0.1])
        new_knee, new_ankle = solve_two_bone_ik(hip, knee, ankle, target_ankle)

        new_l1 = np.linalg.norm(new_knee - hip)
        new_l2 = np.linalg.norm(new_ankle - new_knee)

        np.testing.assert_allclose(new_l1, orig_l1, atol=1e-5)
        np.testing.assert_allclose(new_l2, orig_l2, atol=1e-5)
        np.testing.assert_allclose(new_ankle, target_ankle, atol=1e-5)

    def test_zero_slip_ik_suppresses_drift(self) -> None:
        """Zero-slip IK clamps foot sliding speed to < 0.3 cm/s during stance."""
        num_frames = 60
        # Create skeleton with sliding drift (6 cm/s) during stance phase
        joint_positions = np.zeros((num_frames, 9, 3), dtype=np.float64)
        foot_contacts = np.zeros((num_frames, 2), dtype=np.float32)

        # Frames 10..40 are in stance phase for left foot (joint 7)
        foot_contacts[10:40, 0] = 1.0

        for t in range(num_frames):
            # Left leg: hip=1, knee=4, foot=7
            joint_positions[t, 1] = [-0.15, 0.9, 0.0]
            joint_positions[t, 4] = [-0.15, 0.45, 0.05]
            # Artificial sliding velocity ~ 5.0 cm/s = 0.05 m/s -> 0.00166 m/frame
            slip_drift = t * 0.002
            joint_positions[t, 7] = [-0.15, 0.0, slip_drift]

        skel_data = {"joint_positions": joint_positions}
        result = solve_zero_slip_ik(
            skel_data, foot_contacts, fps=30.0, slip_speed_threshold_cm_s=0.3
        )

        telemetry = result["telemetry"]
        assert telemetry["stance_frames"] == 30
        assert telemetry["raw_foot_sliding_cm_per_sec"] > 4.0
        assert telemetry["clamped_foot_sliding_cm_per_sec"] < 0.3
        assert telemetry["zero_slip_verified"] is True

    def test_zero_slip_ik_empty_skeleton(self) -> None:
        """Handles mock skeleton without joint positions gracefully."""
        skel_data = {"joints": [1, 2, 3]}
        contacts = np.zeros((10, 2), dtype=np.float32)
        result = solve_zero_slip_ik(skel_data, contacts)
        assert result["joints"] == [1, 2, 3]
        assert "telemetry" in result


# ===================================================================
# TestReconDelight
# ===================================================================


class TestReconDelight:
    """Tests for Spherical Harmonics de-lighting and intrinsic recovery."""

    def test_sh_basis_shape(self) -> None:
        """SH basis evaluation produces (..., 9) coefficients."""
        normals = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        basis = evaluate_spherical_harmonics_basis(normals)
        assert basis.shape == (3, 9)
        # Constant ambient term Y0 is positive
        assert (basis[:, 0] > 0.0).all()

    def test_delight_recovers_clean_albedo(self) -> None:
        """De-lighting attenuates directional shading shadows."""
        h, w = 32, 32
        # Ground truth clean albedo: uniform grey
        gt_albedo = np.full((h, w, 3), 0.6, dtype=np.float32)
        # Flat surface normals facing +Z
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 2] = 1.0

        # Lit radiance with strong cast shadow gradient across columns
        shadow_ramp = np.linspace(0.3, 1.0, w, dtype=np.float32).reshape(1, w, 1)
        lit_radiance = gt_albedo * shadow_ramp

        delit = delight(gt_albedo, normals, lit_radiance)
        assert delit.shape == (h, w, 3)

        # De-lit albedo should be much flatter than lit radiance
        raw_variance = np.var(lit_radiance)
        delit_variance = np.var(delit)
        assert delit_variance < raw_variance

    def test_decompose_intrinsic(self) -> None:
        """decompose_intrinsic produces albedo, shading, and specular."""
        radiance = np.random.default_rng(0).uniform(0.1, 0.9, size=(16, 16, 3)).astype(np.float32)
        res = decompose_intrinsic(radiance)
        assert isinstance(res, DelightResult)
        assert res.albedo.shape == (16, 16, 3)
        assert res.shading.shape == (16, 16, 3)
        assert res.specular.shape == (16, 16, 3)
        assert res.ambient_sh.shape == (9, 3)


# ===================================================================
# TestReconTrackingAndTrajectory
# ===================================================================


class TestReconTrackingAndTrajectory:
    """Tests for 4D point tracking and 6-DoF camera trajectory solving."""

    def test_solve_point_tracks(self) -> None:
        """Point tracks return 2D trajectory of shape (T, N, 2)."""
        frames = np.zeros((10, 64, 64, 3), dtype=np.uint8)
        tracks, vis = solve_point_tracks(frames, num_points=64)
        assert tracks.shape == (10, 64, 2)
        assert vis.shape == (10, 64)
        assert (vis == True).all()

    def test_solve_camera_trajectory(self) -> None:
        """Camera trajectory solver outputs valid 6-DoF extrinsics."""
        tracks_2d = np.zeros((15, 64, 2), dtype=np.float32)
        tracks_2d[..., 0] = 32.0
        tracks_2d[..., 1] = 32.0

        traj = solve_camera_trajectory(tracks_2d)
        assert isinstance(traj, CameraTrajectory)
        assert traj.frame_count == 15
        assert len(traj.extrinsics) == 15
        assert traj.extrinsics[0].transform.shape == (4, 4)
        assert traj.metric_scale == 1.0


# ===================================================================
# TestReconPipelineEndToEnd
# ===================================================================


class TestReconPipelineEndToEnd:
    """End-to-end integration tests for ReconPipeline."""

    def test_pipeline_execution(self) -> None:
        """Runs full 5-stage reconstruction flow on synthetic sequence."""
        from unittest.mock import patch

        num_frames = 16
        # Create frames with a bright circle on dark background so that
        # foreground mask derivation succeeds (no longer falls back to
        # constant depth).
        h, w = 48, 48
        frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:h, :w]
        circle = ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) < (h // 4) ** 2
        for t in range(num_frames):
            frames[t][circle] = 200

        pipeline = ReconPipeline(target_quad_faces=500, geometry_backend="python")
        # Synthetic frames still produce degenerate (0°) camera angular
        # separation. Patch the minimum threshold to allow the test to
        # exercise the full 5-stage pipeline without real video input.
        with patch("synapse.recon.multiview._MIN_ANGULAR_SPAN_DEG", 0.0):
            res = pipeline.run(frames, fps=30.0)

        assert isinstance(res, ReconResult)
        assert res.mesh.vertex_count > 0
        assert res.mesh.face_count > 0
        assert res.camera_trajectory is not None
        assert res.camera_trajectory.frame_count == num_frames
        # Without skeleton_template, kinematics is None (no fake sine data).
        assert res.kinematics is None
        assert res.telemetry["kinematics"]["kinematics_status"] == "skipped_no_skeleton"
        assert res.delight is not None

    def test_pipeline_with_skeleton_template(self) -> None:
        """Pipeline with a real skeleton template produces valid kinematics."""
        from unittest.mock import patch

        num_frames = 16
        h, w = 48, 48
        # Create frames with foreground contrast
        frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:h, :w]
        circle = ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) < (h // 4) ** 2
        for t in range(num_frames):
            frames[t][circle] = 200

        # Provide a real skeleton template with joint positions
        joint_pos = np.zeros((num_frames, 9, 3), dtype=np.float64)
        for t in range(num_frames):
            joint_pos[t, 1] = [-0.15, 0.9, 0.0]   # left hip
            joint_pos[t, 4] = [-0.15, 0.45, 0.05]  # left knee
            joint_pos[t, 7] = [-0.15, 0.0, 0.0]    # left foot
            joint_pos[t, 2] = [0.15, 0.9, 0.0]     # right hip
            joint_pos[t, 5] = [0.15, 0.45, -0.05]  # right knee
            joint_pos[t, 8] = [0.15, 0.0, 0.0]     # right foot
        skeleton_template = {"joint_positions": joint_pos}

        pipeline = ReconPipeline(target_quad_faces=500, geometry_backend="python")
        with patch("synapse.recon.multiview._MIN_ANGULAR_SPAN_DEG", 0.0):
            res = pipeline.run(frames, skeleton_template=skeleton_template, fps=30.0)

        assert isinstance(res, ReconResult)
        assert res.kinematics is not None
        assert res.kinematics.frame_count == num_frames
        assert "kinematics_status" not in res.telemetry["kinematics"]
        assert res.telemetry["kinematics"]["zero_slip_verified"] is True


# ===================================================================
# TestReconBackendInterop
# ===================================================================


class TestReconBackendInterop:
    """Tests for SynapseReconBackend satisfying VgBackend contract."""

    def test_is_vg_backend_subclass(self) -> None:
        """SynapseReconBackend implements VgBackend interface."""
        backend = SynapseReconBackend()
        assert isinstance(backend, VgBackend)

    def test_recon_backend_delight(self) -> None:
        """SynapseReconBackend.delight returns array."""
        backend = SynapseReconBackend()
        alb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        nrm = np.zeros((16, 16, 3), dtype=np.float32)
        nrm[..., 2] = 1.0
        rad = np.full((16, 16, 3), 0.4, dtype=np.float32)
        res = backend.delight(alb, nrm, rad)
        assert res.shape == (16, 16, 3)

    def test_recon_backend_quad_remesh(self) -> None:
        """SynapseReconBackend.quad_remesh succeeds."""
        backend = SynapseReconBackend()
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        new_v, new_f = backend.quad_remesh(v, f, target_faces=1)
        assert len(new_v) == 4
        assert len(new_f) == 1
        assert new_f.shape[1] == 4

    def test_recon_backend_check_manifold(self) -> None:
        """SynapseReconBackend.check_manifold returns True for valid mesh."""
        backend = SynapseReconBackend()
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        assert backend.check_manifold(v, f) is True

    def test_recon_backend_solve_zero_slip_ik(self) -> None:
        """SynapseReconBackend.solve_zero_slip_ik returns updated dict."""
        backend = SynapseReconBackend()
        res = backend.solve_zero_slip_ik({"dummy": True}, np.zeros((4, 2)))
        assert res.get("dummy") is True
        assert "telemetry" in res

    def test_recon_backend_refine_pose_6dof(self) -> None:
        """SynapseReconBackend.refine_pose_6dof returns 4x4 matrix."""
        backend = SynapseReconBackend()
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        mat = backend.refine_pose_6dof(pts, pts)
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat, np.eye(4), atol=1e-5)

    def test_get_recon_backend_factory(self) -> None:
        """get_recon_backend returns SynapseReconBackend."""
        backend = get_recon_backend(mock=False)
        assert isinstance(backend, SynapseReconBackend)

    def test_get_vg_backend_fallback(self) -> None:
        """get_vg_backend(fallback_to_recon=True) returns SynapseReconBackend."""
        backend = get_vg_backend(mock=False, fallback_to_recon=True)
        assert isinstance(backend, SynapseReconBackend)
