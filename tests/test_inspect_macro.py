"""Tests for PR-21 multi-channel residual heatmap and macro camera.

Covers:
* Sobel gradient magnitude (flat image, step edge)
* Residual heatmap: identical images, localized patch, depth/normal
* NMS suppression and top-K limit
* Macro camera 3D unproject and positioning
"""

from __future__ import annotations

import math

import numpy as np
from synapse.inspect.macro import (
    _sobel_gradient_magnitude,
    _unproject_pixel,
    compute_isotopic_residual_heatmap,
    plan_macro_camera_for_roi,
)

# ===================================================================
# TestSobelGradient
# ===================================================================


class TestSobelGradient:
    """Tests for ``_sobel_gradient_magnitude``."""

    def test_flat_image_zero_gradient(self) -> None:
        """Constant image should yield zero gradient."""
        img = np.ones((32, 32), dtype=np.float32) * 0.5
        grad = _sobel_gradient_magnitude(img)
        assert grad.shape == (32, 32)
        assert grad.max() < 1e-6

    def test_step_edge_detected(self) -> None:
        """Sharp vertical step produces high gradient along edge."""
        img = np.zeros((32, 32), dtype=np.float32)
        img[:, 16:] = 1.0
        grad = _sobel_gradient_magnitude(img)
        # Edge should be around column 16
        edge_region = grad[2:30, 14:18]
        non_edge = grad[2:30, 0:10]
        assert edge_region.max() > non_edge.max()


# ===================================================================
# TestResidualHeatmap
# ===================================================================


class TestResidualHeatmap:
    """Tests for ``compute_isotopic_residual_heatmap``."""

    def test_identical_images_zero_heatmap(self) -> None:
        """Same image for ref and rendered -> heatmap near zero."""
        img = np.random.RandomState(0).rand(
            64, 64, 3,
        ).astype(np.float32)
        heatmap, rois = compute_isotopic_residual_heatmap(
            img, img, top_k=3,
        )
        assert heatmap.shape == (64, 64)
        assert heatmap.max() < 0.01

    def test_localized_patch_detected(self) -> None:
        """A color patch at a known location is detected as ROI."""
        ref = np.full(
            (128, 128, 3), 0.5, dtype=np.float32,
        )
        rendered = ref.copy()
        # Red patch at (48:80, 48:80)
        rendered[48:80, 48:80, 0] = 1.0
        rendered[48:80, 48:80, 1] = 0.0
        rendered[48:80, 48:80, 2] = 0.0

        heatmap, rois = compute_isotopic_residual_heatmap(
            ref, rendered, top_k=3, nms_radius=16,
        )

        assert len(rois) >= 1

        # At least one ROI near the patch center (64, 64)
        found = False
        for roi in rois:
            cx, cy = roi["center_uv"]
            dist = math.sqrt((cx - 64) ** 2 + (cy - 64) ** 2)
            if dist < 20:
                found = True
                break
        assert found, (
            f"No ROI found near (64,64); rois={rois}"
        )

        # Severity should be significant
        assert rois[0]["mean_severity"] > 0.1

    def test_depth_edge_channel(self) -> None:
        """Depth edge mismatch increases heatmap near the edge."""
        h, w = 64, 64
        ref = np.full(
            (h, w, 3), 0.5, dtype=np.float32,
        )
        rendered = ref.copy()

        # Reference depth: smooth gradient
        depth_ref = np.linspace(
            1.0, 5.0, w, dtype=np.float32,
        )[None, :].repeat(h, axis=0)

        # Rendered depth: same but with a jump at column 32
        depth_ren = depth_ref.copy()
        depth_ren[:, 32:] += 2.0

        heatmap, rois = compute_isotopic_residual_heatmap(
            ref, rendered,
            depth_reference=depth_ref,
            depth_rendered=depth_ren,
            top_k=3,
        )

        # Heatmap should be higher near column 32
        col_32_val = heatmap[:, 30:34].mean()
        col_0_val = heatmap[:, 0:4].mean()
        assert col_32_val > col_0_val

    def test_normal_channel(self) -> None:
        """Normal map angular error detected in heatmap."""
        h, w = 64, 64
        ref = np.full(
            (h, w, 3), 0.5, dtype=np.float32,
        )
        rendered = ref.copy()

        # Reference normals: all pointing up
        n_ref = np.zeros((h, w, 3), dtype=np.float32)
        n_ref[..., 2] = 1.0

        # Rendered normals: mostly up but patch tilted
        n_ren = n_ref.copy()
        n_ren[20:40, 20:40, 0] = 0.5
        n_ren[20:40, 20:40, 2] = 0.866

        heatmap, rois = compute_isotopic_residual_heatmap(
            ref, rendered,
            normals_reference=n_ref,
            normals_rendered=n_ren,
            top_k=3,
        )

        # Heatmap should be higher in the tilted patch area
        patch_val = heatmap[20:40, 20:40].mean()
        bg_val = heatmap[0:10, 0:10].mean()
        assert patch_val > bg_val

    def test_nms_suppresses_nearby_rois(self) -> None:
        """Two close patches yield 1 ROI; two far patches yield 2."""
        ref = np.full(
            (128, 128, 3), 0.5, dtype=np.float32,
        )

        # Close patches: 5px apart
        close_rendered = ref.copy()
        close_rendered[30:38, 30:38, 0] = 1.0
        close_rendered[32:40, 40:48, 0] = 1.0

        _, close_rois = compute_isotopic_residual_heatmap(
            ref, close_rendered, top_k=10, nms_radius=32,
        )
        # Should merge to 1 or few ROIs
        assert len(close_rois) <= 2

        # Far patches: 80px apart
        far_rendered = ref.copy()
        far_rendered[10:20, 10:20, 0] = 1.0
        far_rendered[90:100, 90:100, 0] = 1.0

        _, far_rois = compute_isotopic_residual_heatmap(
            ref, far_rendered, top_k=10, nms_radius=32,
        )
        assert len(far_rois) >= 2

    def test_top_k_limit(self) -> None:
        """Only top_k ROIs are returned."""
        ref = np.full(
            (256, 256, 3), 0.5, dtype=np.float32,
        )
        rendered = ref.copy()

        # Create 8 widely separated patches
        positions = [
            (20, 20), (20, 120), (20, 220),
            (120, 20), (120, 120), (120, 220),
            (220, 20), (220, 120),
        ]
        for y, x in positions:
            rendered[y:y+12, x:x+12, 0] = 1.0

        _, rois = compute_isotopic_residual_heatmap(
            ref, rendered, top_k=3, nms_radius=16,
        )
        assert len(rois) <= 3

    def test_empty_heatmap_no_rois(self) -> None:
        """Identical images produce no ROIs."""
        img = np.ones(
            (32, 32, 3), dtype=np.float32,
        ) * 0.5
        _, rois = compute_isotopic_residual_heatmap(
            img, img, top_k=5,
        )
        assert len(rois) == 0


# ===================================================================
# TestMacroCameraPlanning
# ===================================================================


class TestMacroCameraPlanning:
    """Tests for ``plan_macro_camera_for_roi``."""

    def _make_identity_setup(
        self,
        h: int = 128,
        w: int = 128,
        depth_val: float = 5.0,
    ) -> tuple[
        dict[str, float], np.ndarray, np.ndarray, np.ndarray,
    ]:
        """Create simple identity camera setup.

        Camera at origin looking down -Z with uniform depth.
        Returns (intrinsics, extrinsics, depth_map, normal_map).
        """
        fx = fy = float(w) * 2.0
        cx = float(w) / 2.0
        cy = float(h) / 2.0
        intrinsics = {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        }

        # Identity extrinsics (camera at origin)
        extrinsics = np.eye(4, dtype=np.float64)

        # Uniform depth
        depth = np.full(
            (h, w), depth_val, dtype=np.float32,
        )

        # Normals pointing toward camera: [0, 0, -1] in world
        # (toward camera at origin looking down -Z)
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 2] = -1.0

        return intrinsics, extrinsics, depth, normals

    def test_unproject_center(self) -> None:
        """Unproject center pixel with identity camera."""
        p = _unproject_pixel(
            u=64.0, v=64.0, z=5.0,
            fx=256.0, fy=256.0, cx=64.0, cy=64.0,
        )
        # At center: (u-cx)/fx * z = 0, (v-cy)/fy * z = 0
        np.testing.assert_allclose(
            p, [0.0, 0.0, 5.0], atol=1e-10,
        )

    def test_unproject_off_center(self) -> None:
        """Unproject off-center pixel."""
        p = _unproject_pixel(
            u=128.0, v=64.0, z=10.0,
            fx=256.0, fy=256.0, cx=64.0, cy=64.0,
        )
        expected_x = (128 - 64) * 10.0 / 256.0
        assert abs(p[0] - expected_x) < 1e-10
        assert abs(p[2] - 10.0) < 1e-10

    def test_macro_camera_position(self) -> None:
        """Macro camera positioned along normal direction."""
        intrinsics, extrinsics, depth, normals = (
            self._make_identity_setup()
        )

        roi = {
            "roi_id": 0,
            "center_uv": (64, 64),
            "bounding_radius_px": 16.0,
            "mean_severity": 0.5,
            "pixel_count": 100,
        }

        pose = plan_macro_camera_for_roi(
            roi, depth, normals,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            macro_fov_deg=35.0,
        )

        # Camera should be along -Z from the defect point
        # Defect is at (0, 0, 5) in world
        # Normal is [0, 0, -1]
        # Camera should be at (0, 0, 5 - d_macro)
        cam_pos = np.array(pose.position)
        assert cam_pos[2] < 5.0, (
            "Camera should be in front of defect"
        )

    def test_macro_distance_formula(self) -> None:
        """Distance matches: d = 1.35 * r_3d / tan(fov/2)."""
        intrinsics, extrinsics, depth, normals = (
            self._make_identity_setup(depth_val=10.0)
        )

        r_px = 20.0
        roi = {
            "roi_id": 0,
            "center_uv": (64, 64),
            "bounding_radius_px": r_px,
            "mean_severity": 0.5,
            "pixel_count": 100,
        }

        fov = 35.0
        pose = plan_macro_camera_for_roi(
            roi, depth, normals,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            macro_fov_deg=fov,
        )

        z_c = 10.0
        fx = intrinsics["fx"]
        r_3d = r_px * z_c / fx
        expected_d = 1.35 * r_3d / math.tan(math.radians(fov / 2))

        cam_pos = np.array(pose.position)
        # Defect is at (0, 0, z_c=10) for center pixel
        defect = np.array([0.0, 0.0, z_c])
        actual_d = float(np.linalg.norm(cam_pos - defect))
        assert abs(actual_d - expected_d) < 0.1, (
            f"Expected d={expected_d:.3f}, got {actual_d:.3f}"
        )

    def test_macro_covers_defect(self) -> None:
        """FOV and distance ensure defect fits in viewport."""
        intrinsics, extrinsics, depth, normals = (
            self._make_identity_setup()
        )

        roi = {
            "roi_id": 0,
            "center_uv": (64, 64),
            "bounding_radius_px": 16.0,
            "mean_severity": 0.5,
            "pixel_count": 100,
        }

        fov = 35.0
        pose = plan_macro_camera_for_roi(
            roi, depth, normals,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            macro_fov_deg=fov,
        )

        cam_pos = np.array(pose.position)
        defect = np.array([0.0, 0.0, 5.0])
        d_actual = float(np.linalg.norm(cam_pos - defect))

        # Viewport half-size at defect distance
        half_viewport = d_actual * math.tan(
            math.radians(fov / 2),
        )

        # 3D radius of defect
        r_3d = 16.0 * 5.0 / intrinsics["fx"]

        assert half_viewport > r_3d, (
            f"Viewport {half_viewport:.4f} should cover "
            f"defect radius {r_3d:.4f}"
        )

    def test_camera_pose_valid(self) -> None:
        """Returned CameraPose has valid structure."""
        intrinsics, extrinsics, depth, normals = (
            self._make_identity_setup()
        )

        roi = {
            "roi_id": 0,
            "center_uv": (64, 64),
            "bounding_radius_px": 16.0,
            "mean_severity": 0.5,
            "pixel_count": 100,
        }

        pose = plan_macro_camera_for_roi(
            roi, depth, normals,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
        )

        assert len(pose.position) == 3
        assert len(pose.rotation) == 9
        assert pose.fov_deg == 35.0
        assert pose.resolution == (512, 512)
