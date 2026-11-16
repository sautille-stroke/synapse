"""Tests for synapse.recon — Video-to-3D Inverse Reconstruction & Asset Distillation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from synapse.bridge.vg_bridge import (
    VgBackend,
    get_vg_backend,
)
from synapse.recon import (
    CameraTrajectory,
    DelightResult,
    ReconPipeline,
    ReconResult,
    SynapseReconBackend,
    check_manifold,
    check_watertight,
    compute_normals,
    decompose_intrinsic,
    delight,
    delight_multiview_albedo,
    evaluate_spherical_harmonics_basis,
    get_recon_backend,
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

    def test_delight_multiview_albedo_callable(self) -> None:
        """delight_multiview_albedo callable contract."""
        assert callable(delight_multiview_albedo)

    def test_decompose_intrinsic_uint8_input(self) -> None:
        """PR-61: uint8 input [0..255] is properly normalized and not saturated.

        For uniform input, albedo ≈ 1.0 is valid (I/E where E ≈ I).
        Key check: albedo mean is reasonable (not 0 or 255), and delta_e >= 0.
        """
        radiance_u8 = np.full((16, 16, 3), 128, dtype=np.uint8)
        res = decompose_intrinsic(radiance_u8)
        assert res.albedo.mean() <= 1.0
        assert res.albedo.mean() > 0.0
        assert res.delta_e_residual >= 0.0

    def test_decompose_intrinsic_sphere_directional_light(self) -> None:
        """PR-79: Synthetic directional light sphere test.

        With lighting contrast > 5x, after decompose_intrinsic the albedo
        variance between lit/shadow sides converges significantly (shading
        removal ratio >= 60%).
        """
        h, w = 64, 64

        # Ground truth albedo: uniform mid-grey
        gt_albedo_val = 0.5
        gt_albedo = np.full((h, w, 3), gt_albedo_val, dtype=np.float64)

        # Create sphere normals: hemisphere facing camera (+Z)
        yy, xx = np.mgrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        radius = min(h, w) / 2.0 - 2.0
        nx = (xx - cx) / radius
        ny = (yy - cy) / radius
        r2 = nx**2 + ny**2
        sphere_mask = r2 < 1.0

        nz = np.zeros_like(nx)
        nz[sphere_mask] = np.sqrt(1.0 - r2[sphere_mask])
        normals = np.stack([nx, ny, nz], axis=-1)
        # Normalize
        lens = np.linalg.norm(normals, axis=-1, keepdims=True)
        lens[lens < 1e-8] = 1.0
        normals = normals / lens
        # Outside sphere: flat +Z
        normals[~sphere_mask] = [0.0, 0.0, 1.0]

        # Strong directional light from upper-right: L = [0.7, 0.5, 0.5]
        light_dir = np.array([0.7, 0.5, 0.5], dtype=np.float64)
        light_dir /= np.linalg.norm(light_dir)
        light_intensity = 3.0
        ambient = 0.1

        n_dot_l = np.sum(normals * light_dir[np.newaxis, np.newaxis, :], axis=-1)
        n_dot_l = np.clip(n_dot_l, 0.0, 1.0)
        shading = ambient + light_intensity * n_dot_l  # range: [0.1, 3.1]

        # Radiance = albedo * shading (Lambertian)
        radiance = gt_albedo * shading[..., np.newaxis]

        # Verify contrast > 5x
        lit_region = shading[sphere_mask] > 2.0
        shadow_region = shading[sphere_mask] < 0.5
        if np.sum(lit_region) > 0 and np.sum(shadow_region) > 0:
            contrast = np.mean(shading[sphere_mask][lit_region]) / np.mean(
                shading[sphere_mask][shadow_region]
            )
            assert contrast > 5.0, f"Lighting contrast {contrast:.1f} < 5x"

        # Run decompose_intrinsic with real normals
        res = decompose_intrinsic(radiance.astype(np.float32), normal=normals.astype(np.float32))

        assert isinstance(res, DelightResult)
        assert res.albedo.shape == (h, w, 3)
        assert res.ambient_sh.shape == (9, 3)

        # Measure shading removal: albedo should be more uniform than radiance
        # on the sphere pixels
        sphere_radiance = radiance[sphere_mask]  # (M, 3)
        sphere_albedo = res.albedo[sphere_mask].astype(np.float64)  # (M, 3)

        # Variance of luminance across sphere
        rad_lum = 0.2126 * sphere_radiance[:, 0] + 0.7152 * sphere_radiance[:, 1] + 0.0722 * sphere_radiance[:, 2]
        alb_lum = 0.2126 * sphere_albedo[:, 0] + 0.7152 * sphere_albedo[:, 1] + 0.0722 * sphere_albedo[:, 2]

        rad_var = float(np.var(rad_lum))
        alb_var = float(np.var(alb_lum))

        # Shading removal ratio: at least 60% of shading variance removed
        shading_removal_ratio = 1.0 - alb_var / max(rad_var, 1e-10)
        assert shading_removal_ratio >= 0.60, (
            f"Shading removal ratio {shading_removal_ratio:.2%} < 60%. "
            f"rad_var={rad_var:.6f}, alb_var={alb_var:.6f}"
        )

    def test_decompose_intrinsic_hdr_preserves_values(self) -> None:
        """PR-79: Linear HDR values > 1.0 must NOT be erroneously scaled to near-zero.

        A scene with linear HDR mean > 1.5 must not produce albedo scaled
        down to ~0.006 (the old /255.0 bug for float inputs).
        """
        h, w = 32, 32
        # HDR radiance with mean ~2.5 (common for linear EXR data)
        rng = np.random.default_rng(99)
        radiance_hdr = rng.uniform(1.5, 3.5, size=(h, w, 3)).astype(np.float32)

        # Normals: hemisphere
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 2] = 1.0

        res = decompose_intrinsic(radiance_hdr, normal=normals)

        assert isinstance(res, DelightResult)
        mean_albedo = float(res.albedo.mean())

        # The old bug would produce albedo ~0.006 (radiance / 255.0 / shading).
        # Correct behavior: albedo should be in a reasonable range, NOT near zero.
        assert mean_albedo > 0.5, (
            f"HDR albedo mean {mean_albedo:.4f} is far too low — likely still "
            f"dividing float input by 255.0"
        )
        # Also verify albedo is not absurdly large
        assert mean_albedo < 50.0, f"HDR albedo mean {mean_albedo:.4f} is unreasonably large"

    def test_delight_multiview_uses_cameras_and_mesh(self) -> None:
        """PR-79: delight_multiview_albedo must use cameras and mesh for projection."""
        from synapse.recon.types import CameraExtrinsics, MeshGeometry

        h, w = 32, 32
        n_views = 3
        rng = np.random.default_rng(123)

        # Create a simple mesh (flat plane)
        verts = np.array([
            [-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        norms = np.array([
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
        ], dtype=np.float32)
        mesh = MeshGeometry(vertices=verts, faces=faces, normals=norms)

        # Create camera extrinsics for 3 views
        cameras = []
        for vi in range(n_views):
            angle = vi * 0.3
            R = np.eye(3, dtype=np.float64)
            R[0, 0] = np.cos(angle)
            R[0, 2] = np.sin(angle)
            R[2, 0] = -np.sin(angle)
            R[2, 2] = np.cos(angle)
            t = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            cam = CameraExtrinsics(
                frame=vi,
                transform=T,
                position=np.array([0.0, 0.0, 0.0]),
                rotation_matrix=R,
                yaw_rad=float(angle),
            )
            cameras.append(cam)

        # Frames with varying illumination
        frames = rng.uniform(0.2, 0.8, size=(n_views, h, w, 3)).astype(np.float32)

        res = delight_multiview_albedo(frames, cameras, mesh)

        assert isinstance(res, DelightResult)
        assert res.albedo.shape == (h, w, 3)
        assert res.shading.shape == (h, w, 3)
        assert res.specular.shape == (h, w, 3)
        assert res.ambient_sh.shape == (9, 3)
        assert res.delta_e_residual >= 0.0
        # Albedo should be finite and non-negative
        assert np.all(np.isfinite(res.albedo))
        assert float(res.albedo.min()) >= 0.0

    def test_delight_multiview_hdr_not_scaled(self) -> None:
        """PR-79: delight_multiview_albedo must not scale float HDR by /255."""
        from synapse.recon.types import CameraExtrinsics, MeshGeometry

        h, w = 16, 16
        # HDR frames with values > 1.5
        frames_hdr = np.full((2, h, w, 3), 2.5, dtype=np.float32)

        verts = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = MeshGeometry(vertices=verts, faces=faces)

        cameras = []
        for vi in range(2):
            T = np.eye(4, dtype=np.float64)
            cam = CameraExtrinsics(
                frame=vi,
                transform=T,
                position=np.array([0.0, 0.0, 0.0]),
                rotation_matrix=np.eye(3),
                yaw_rad=0.0,
            )
            cameras.append(cam)

        res = delight_multiview_albedo(frames_hdr, cameras, mesh)

        # Albedo should NOT be near zero (the old bug: values > 1.5 → /255.0)
        assert float(res.albedo.mean()) > 0.5, (
            f"Multiview HDR albedo {res.albedo.mean():.4f} is too low — "
            f"likely still dividing float HDR by 255.0"
        )


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
        from synapse.recon.types import MeshGeometry

        num_frames = 16
        h, w = 48, 48
        frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:h, :w]
        for t in range(num_frames):
            cx = w // 2 + t
            cy = h // 2
            circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < (h // 4) ** 2
            frames[t][circle] = 200

        # Provide initial_mesh to skip MultiViewInitializer, which requires
        # real multi-view parallax that synthetic frames cannot provide.
        cube_v = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float32)
        cube_f = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)
        initial_mesh = MeshGeometry(vertices=cube_v, faces=cube_f)

        pipeline = ReconPipeline(target_quad_faces=500, geometry_backend="python")
        res = pipeline.run(frames, initial_mesh=initial_mesh, fps=30.0)

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
        from synapse.recon.types import MeshGeometry

        num_frames = 16
        h, w = 48, 48
        frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:h, :w]
        for t in range(num_frames):
            cx = w // 2 + t
            cy = h // 2
            circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < (h // 4) ** 2
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

        # Provide initial_mesh to skip MultiViewInitializer
        cube_v = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float32)
        cube_f = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)
        initial_mesh = MeshGeometry(vertices=cube_v, faces=cube_f)

        pipeline = ReconPipeline(target_quad_faces=500, geometry_backend="python")
        res = pipeline.run(
            frames, initial_mesh=initial_mesh,
            skeleton_template=skeleton_template, fps=30.0,
        )

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


# ===================================================================
# PR-80: stages/recon persisted file can be read back with non-empty faces (test-c)
# ===================================================================


class TestStagesReconPersistence:
    """PR-80: stages/recon persisted file can be read back with non-empty faces."""

    def test_video_to_3d_invert_persists_full_mesh(self, tmp_path: Any) -> None:
        """handle_video_to_3d_invert saves vertices, faces (and uvs) to
        output .npz, and the persisted file can be read back with
        non-empty faces."""
        from unittest.mock import patch

        from synapse.pipeline.bundle_worker import BundleManifest
        from synapse.recon.types import MeshGeometry

        # Create synthetic video frames with visible foreground
        num_frames = 8
        h, w = 48, 48
        frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:h, :w]
        for t in range(num_frames):
            cx = w // 2 + t
            cy = h // 2
            circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < (h // 4) ** 2
            frames[t][circle] = 200

        # Save frames to temp
        video_path = str(tmp_path / "frames.npy")
        np.save(video_path, frames)

        mesh_out = str(tmp_path / "output" / "mesh.npy")
        traj_out = str(tmp_path / "output" / "trajectory.json")

        manifest = BundleManifest(
            job_id="test-pr80",
            stage="video_to_3d_invert",
            inputs={"video_frames": video_path},
            outputs={"mesh": mesh_out, "trajectory": traj_out},
            params={"target_faces": 500},
        )

        from synapse.pipeline.stages.recon import handle_video_to_3d_invert

        # Mock MultiViewInitializer.reconstruct to return a known cube mesh
        # since synthetic frames cannot produce real multi-view parallax.
        cube_v = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float32)
        cube_f = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)
        mock_mesh = MeshGeometry(vertices=cube_v, faces=cube_f)

        with patch(
            "synapse.recon.multiview.MultiViewInitializer.reconstruct",
            return_value=mock_mesh,
        ):
            result = handle_video_to_3d_invert(manifest)

        assert "output_paths" in result
        mesh_path = result["output_paths"].get("mesh", "")
        assert mesh_path, "No mesh output path returned"

        # Read back and verify faces are non-empty
        loaded = np.load(mesh_path)
        assert "vertices" in loaded, "Missing 'vertices' key in saved mesh"
        assert "faces" in loaded, "Missing 'faces' key in saved mesh"
        assert loaded["vertices"].shape[0] > 0, "Vertices array is empty"
        assert loaded["faces"].shape[0] > 0, "Faces array is empty"
        assert loaded["faces"].shape[1] >= 3, "Faces must have >= 3 vertex indices"


class TestTrackingRealOpticalFlow:
    """PR-80: verify tracking uses real optical flow, not R/G mean-diff."""

    def test_tracking_with_moving_object(self) -> None:
        """Points track a moving bright patch correctly, not uniformly."""
        num_frames = 5
        h, w = 64, 64

        # Create frames with a bright square moving rightward
        frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
        for t in range(num_frames):
            x_offset = 10 + t * 8  # moves right
            y_start, y_end = 20, 40
            x_start = max(0, x_offset)
            x_end = min(w, x_offset + 15)
            frames[t, y_start:y_end, x_start:x_end] = 200

        tracks, vis = solve_point_tracks(frames, num_points=32)
        assert tracks.shape == (num_frames, 32, 2)
        assert vis.shape == (num_frames, 32)

        # Not all points should move by the same amount (unlike mean-diff)
        if vis[1:].any():
            frame_displacements = tracks[1:] - tracks[:-1]
            # At least some tracked points should show non-zero displacement
            max_disp = np.max(np.abs(frame_displacements[vis[1:]]))
            assert max_disp > 0.0, "No tracked points moved at all"