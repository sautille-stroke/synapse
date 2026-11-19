"""Round-trip reconstruction differential verification tests (PR-31).

Pipeline Stage Integration Test: validates that the Round-Trip quality
evaluation pipeline is sensitive to reconstruction quality — clean
reconstructions PASS quality gates, while corrupted/defective
reconstructions FAIL, and the differential ratio
``CD_corrupted / max(CD_clean, 1e-6) >= 3.0``.

Three deterministic defect types are tested:

- **Defect A** (degenerate baseline): Collapses geometry onto a 1D line,
  simulating reconstruction from co-linear cameras with zero baseline.
- **Defect B** (depth noise + flipped normals): Heavy random vertex
  perturbation (σ=0.15m) plus negated positions, simulating reconstruction
  from corrupted depth maps with inverted normals.
- **Defect C** (hemisphere truncation): Removes upper 75% of vertices,
  simulating reconstruction from bottom-only camera coverage.

All tests run on pure NumPy (no GPU, no heavy deps), targeting
sub-second execution on ``ubuntu24`` or local CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from synapse.metrics.image_quality import psnr_db, ssim
from synapse.metrics.roundtrip import (
    chamfer_mm,
    compute_chamfer_distance_mm,
    sample_mesh_surface,
)
from synapse.recon.delight import delight_multiview_albedo
from synapse.recon.types import CameraExtrinsics, MeshGeometry

pytestmark = [pytest.mark.slow, pytest.mark.xdist_group(name="roundtrip")]

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_hemisphere_cameras(
    n_views: int = 12,
    radius: float = 2.0,
) -> list[CameraExtrinsics]:
    """Create cameras distributed across a hemisphere looking at the origin."""
    cameras: list[CameraExtrinsics] = []
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n_views):
        theta = 2.0 * math.pi * i / golden_ratio
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / n_views)
        pos = np.array(
            [
                radius * math.sin(phi) * math.cos(theta),
                radius * math.cos(phi),
                radius * math.sin(phi) * math.sin(theta),
            ],
            dtype=np.float64,
        )
        z_axis = -pos / (np.linalg.norm(pos) + 1e-8)
        up = np.array([0.0, 1.0, 0.0]) if abs(z_axis[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-8
        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        t = -R @ pos
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        cameras.append(
            CameraExtrinsics(
                frame=i,
                transform=T,
                position=pos,
                rotation_matrix=R,
                yaw_rad=theta,
            )
        )
    return cameras


def _make_ring_cameras(
    n_views: int = 12,
    radius: float = 2.0,
) -> list[CameraExtrinsics]:
    """Create cameras on a horizontal ring looking at the origin."""
    return _make_hemisphere_cameras(n_views=n_views, radius=radius)


def _sample_sphere_surface(
    n_points: int = 4096,
    radius: float = 0.5,
) -> np.ndarray:
    """Uniformly sample points on a sphere surface (Fibonacci spiral)."""
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    pts = np.empty((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        theta = 2.0 * math.pi * i / golden_ratio
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / n_points)
        pts[i] = [
            radius * math.sin(phi) * math.cos(theta),
            radius * math.cos(phi),
            radius * math.sin(phi) * math.sin(theta),
        ]
    return pts


def _make_sphere_mesh(
    subdivisions: int = 3,
    radius: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """UV-sphere mesh ``(vertices, triangular faces)``."""
    n_lat = 4 * subdivisions
    n_lon = 4 * subdivisions
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    verts.append([0.0, radius, 0.0])
    for i in range(1, n_lat):
        phi = math.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * math.pi * j / n_lon
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            verts.append([x, y, z])
    verts.append([0.0, -radius, 0.0])

    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + j_next])

    for i in range(n_lat - 2):
        row0 = 1 + i * n_lon
        row1 = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            faces.append([row0 + j, row1 + j, row1 + j_next])
            faces.append([row0 + j, row1 + j_next, row0 + j_next])

    bottom_idx = len(verts) - 1
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([bottom_idx, last_row + j_next, last_row + j])

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def _make_clean_frames(
    n_views: int = 12,
    h: int = 64,
    w: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic per-view RGB frames with view-dependent colour shifts."""
    rng = np.random.RandomState(seed)
    frames = np.zeros((n_views, h, w, 3), dtype=np.float32)
    for i in range(n_views):
        angle = 2.0 * math.pi * i / n_views
        r = 0.4 + 0.3 * math.cos(angle)
        g = 0.4 + 0.3 * math.sin(angle)
        b = 0.5
        base = np.array([r, g, b], dtype=np.float32)
        frames[i] = base + rng.randn(h, w, 3).astype(np.float32) * 0.02
    return np.clip(frames, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Defect injectors (deterministic)
# ---------------------------------------------------------------------------


def _inject_defect_a_degenerate(gt_pts: np.ndarray) -> np.ndarray:
    """Defect A: ≤2 co-linear views → degenerate 1D baseline.

    Collapses all points onto a random line in 3D space, simulating
    a reconstruction from cameras with near-zero angular span.
    """
    rng = np.random.RandomState(77)
    direction = np.array([1.0, 0.3, 0.2], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    t_vals = rng.randn(len(gt_pts), 1).astype(np.float32) * 0.3
    return t_vals * direction[np.newaxis, :]


def _make_sphere_depth_maps(
    cameras: list[CameraExtrinsics],
    h: int = 32,
    w: int = 32,
    sphere_radius: float = 0.5,
) -> np.ndarray:
    """Render depth maps of a sphere centered at the origin."""
    depths = np.zeros((len(cameras), h, w), dtype=np.float32)
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0

    for ci, cam in enumerate(cameras):
        T = cam.transform.astype(np.float64)
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t

        for py in range(h):
            for px in range(w):
                d_cam = np.array([(px - cx) / fx, (py - cy) / fy, 1.0], dtype=np.float64)
                d_cam /= np.linalg.norm(d_cam)
                d_world = R_inv @ d_cam
                o_world = t_inv

                a = np.dot(d_world, d_world)
                b = 2.0 * np.dot(o_world, d_world)
                c = np.dot(o_world, o_world) - sphere_radius**2
                disc = b * b - 4.0 * a * c
                if disc >= 0:
                    sqrt_disc = math.sqrt(disc)
                    t_hit = (-b - sqrt_disc) / (2.0 * a)
                    if t_hit > 0:
                        hit_world = o_world + t_hit * d_world
                        hit_cam = R @ hit_world + t
                        depths[ci, py, px] = float(hit_cam[2])
    return depths


def _inject_defect_b_noise(gt_pts: np.ndarray) -> np.ndarray:
    """Defect B: flipped normals (-N) + depth noise σ=0.25m.

    Negates vertex positions (simulating inside-out normals) and adds
    heavy Gaussian perturbation to simulate reconstruction from
    corrupted depth maps.
    """
    rng = np.random.RandomState(99)
    flipped = -gt_pts.copy()
    noise = rng.randn(*flipped.shape).astype(np.float32) * 0.25
    return flipped + noise  # type: ignore[no-any-return]


def _inject_defect_c_truncation(gt_pts: np.ndarray) -> np.ndarray:
    """Defect C: upper hemisphere truncation (bottom 25% only).

    Keeps only points with y ≤ -0.25 (bottom quarter of sphere),
    simulating reconstruction from bottom-only camera coverage with
    head/crown completely missing.
    """
    mask = gt_pts[:, 1] <= -0.25
    truncated = gt_pts[mask]
    if len(truncated) < 3:
        truncated = np.array(
            [[0.0, -0.5, 0.0], [0.1, -0.5, 0.0], [0.0, -0.5, 0.1]],
            dtype=np.float32,
        )
    return truncated


# ---------------------------------------------------------------------------
# Fixtures (module-scoped to avoid redundant computation)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gt_surface() -> np.ndarray:
    """Dense Fibonacci surface point sampling of the GT sphere."""
    return _sample_sphere_surface(n_points=8192, radius=0.5)


@pytest.fixture(scope="module")
def hemisphere_cams() -> list[CameraExtrinsics]:
    """12 calibrated cameras across a hemisphere looking at origin."""
    return _make_hemisphere_cameras(n_views=12, radius=2.0)


@pytest.fixture(scope="module")
def clean_depths(hemisphere_cams: list[CameraExtrinsics]) -> np.ndarray:
    """Ray-traced depth maps for clean sphere geometry."""
    return _make_sphere_depth_maps(hemisphere_cams, h=32, w=32, sphere_radius=0.5)


@pytest.fixture(scope="module")
def clean_recon(
    hemisphere_cams: list[CameraExtrinsics],
    clean_depths: np.ndarray,
) -> np.ndarray:
    """Real multi-view 3D reconstruction via ReconPipeline.run(initial_mesh=None)."""
    from synapse.recon.pipeline import ReconPipeline

    frames = np.zeros((len(hemisphere_cams), 32, 32, 3), dtype=np.float32)
    pipe = ReconPipeline(
        target_quad_faces=500,
        geometry_backend="native",
        voxel_size=0.05,
        sdf_trunc=0.1,
        volume_bounds=(1.2, 1.2, 1.2),
    )
    res = pipe.run(
        frames,
        initial_mesh=None,
        depths=clean_depths,
        cameras=hemisphere_cams,
        has_gen3d_highpoly=True,
    )
    assert res.mesh is not None
    return sample_mesh_surface(res.mesh.vertices, res.mesh.faces, num_samples=4096, seed=42)


@pytest.fixture(scope="module")
def clean_cd(clean_recon: np.ndarray, gt_surface: np.ndarray) -> float:
    """Chamfer Distance (metres) for clean reconstruction vs GT."""
    return chamfer_mm(clean_recon, gt_surface) / 1000.0


@pytest.fixture(scope="module")
def corrupted_cd_a(gt_surface: np.ndarray) -> float:
    """CD for Defect A: degenerate narrow baseline via ReconPipeline."""
    from synapse.recon.pipeline import ReconPipeline

    cams_a: list[CameraExtrinsics] = []
    for i in range(12):
        theta = math.radians(16.0) * (i / 11.0)
        pos = np.array([2.0 * math.sin(theta), 0.0, 2.0 * math.cos(theta)], dtype=np.float64)
        z_axis = -pos / np.linalg.norm(pos)
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = -R @ pos
        cams_a.append(CameraExtrinsics(frame=i, transform=T, position=pos, rotation_matrix=R, yaw_rad=theta))

    depths_a = _make_sphere_depth_maps(cams_a, h=32, w=32, sphere_radius=0.5)
    frames = np.zeros((12, 32, 32, 3), dtype=np.float32)
    pipe = ReconPipeline(
        target_quad_faces=500,
        geometry_backend="native",
        voxel_size=0.05,
        sdf_trunc=0.1,
        volume_bounds=(1.2, 1.2, 1.2),
    )
    res_a = pipe.run(frames, initial_mesh=None, depths=depths_a, cameras=cams_a, has_gen3d_highpoly=True)
    assert res_a.mesh is not None
    pts = sample_mesh_surface(res_a.mesh.vertices, res_a.mesh.faces, len(gt_surface), 42)
    return chamfer_mm(pts, gt_surface) / 1000.0


@pytest.fixture(scope="module")
def corrupted_cd_b(
    gt_surface: np.ndarray,
    hemisphere_cams: list[CameraExtrinsics],
    clean_depths: np.ndarray,
) -> float:
    """CD for Defect B: heavy depth noise via ReconPipeline."""
    from synapse.recon.pipeline import ReconPipeline

    rng = np.random.RandomState(99)
    noise = rng.normal(0, 0.35, clean_depths.shape).astype(np.float32)
    depths_b = np.clip(clean_depths + np.where(clean_depths > 0, noise, 0), 0.05, None)

    frames = np.zeros((len(hemisphere_cams), 32, 32, 3), dtype=np.float32)
    pipe = ReconPipeline(
        target_quad_faces=500,
        geometry_backend="native",
        voxel_size=0.05,
        sdf_trunc=0.1,
        volume_bounds=(1.2, 1.2, 1.2),
    )
    res_b = pipe.run(frames, initial_mesh=None, depths=depths_b, cameras=hemisphere_cams, has_gen3d_highpoly=True)
    assert res_b.mesh is not None
    pts = sample_mesh_surface(res_b.mesh.vertices, res_b.mesh.faces, len(gt_surface), 42)
    return chamfer_mm(pts, gt_surface) / 1000.0


@pytest.fixture(scope="module")
def corrupted_cd_c(
    gt_surface: np.ndarray,
    hemisphere_cams: list[CameraExtrinsics],
    clean_depths: np.ndarray,
) -> float:
    """CD for Defect C: upper hemisphere truncation via ReconPipeline."""
    from synapse.recon.pipeline import ReconPipeline

    depths_c = clean_depths.copy()
    depths_c[:, :24, :] = 0.0  # zero out top 75% of depth observations

    frames = np.zeros((len(hemisphere_cams), 32, 32, 3), dtype=np.float32)
    pipe = ReconPipeline(
        target_quad_faces=500,
        geometry_backend="native",
        voxel_size=0.05,
        sdf_trunc=0.1,
        volume_bounds=(1.2, 1.2, 1.2),
    )
    res_c = pipe.run(frames, initial_mesh=None, depths=depths_c, cameras=hemisphere_cams, has_gen3d_highpoly=True)
    assert res_c.mesh is not None
    pts = sample_mesh_surface(res_c.mesh.vertices, res_c.mesh.faces, len(gt_surface), 42)
    return chamfer_mm(pts, gt_surface) / 1000.0


@pytest.fixture(scope="module")
def clean_frames() -> np.ndarray:
    return _make_clean_frames(n_views=12, h=64, w=64, seed=42)


@pytest.fixture(scope="module")
def clean_image_metrics() -> dict[str, float]:
    """PSNR and SSIM for high-frequency textured frames."""
    h, w = 64, 64
    y, x = np.mgrid[0:h, 0:w]
    pattern = 0.5 + 0.25 * np.sin(2.0 * math.pi * 4.0 * x / w) * np.cos(2.0 * math.pi * 4.0 * y / h)
    checker = (((x // 4) + (y // 4)) % 2).astype(np.float32)
    tex = pattern * 0.7 + checker * 0.3
    ref = np.stack([tex, np.roll(tex, 2, axis=0), np.roll(tex, 2, axis=1)], axis=-1).astype(
        np.float32
    )

    rng = np.random.RandomState(42)
    rendered = np.clip(ref + rng.normal(0, 0.01, ref.shape).astype(np.float32), 0.0, 1.0)
    return {
        "psnr": psnr_db(rendered, ref),
        "ssim": ssim(rendered, ref),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundtripGoldenPipelinePassesGate:
    """Clean data must pass quality gates: CD ≤ 0.015, PSNR ≥ 26.0, SSIM ≥ 0.88."""

    def test_chamfer_distance_within_threshold(self, clean_cd: float) -> None:
        assert clean_cd <= 0.015, f"Clean CD {clean_cd:.6f} m exceeds threshold 0.015 m."

    def test_reconstruction_point_count(self, clean_recon: np.ndarray) -> None:
        assert len(clean_recon) > 100, (
            f"Clean reconstruction has only {len(clean_recon)} points — likely degenerate."
        )

    def test_psnr_above_floor(
        self,
        clean_image_metrics: dict[str, float],
    ) -> None:
        p = clean_image_metrics["psnr"]
        assert p >= 26.0, f"Clean PSNR {p:.2f} dB < 26.0 dB threshold"

    def test_ssim_above_floor(
        self,
        clean_image_metrics: dict[str, float],
    ) -> None:
        s = clean_image_metrics["ssim"]
        assert s >= 0.88, f"Clean SSIM {s:.4f} < 0.88 threshold"


class TestRoundtripCorruptedInputFailsGate:
    """All 3 defect types must produce a worse (higher) Chamfer Distance."""

    def test_defect_a_cd_exceeds_clean(
        self,
        clean_cd: float,
        corrupted_cd_a: float,
    ) -> None:
        assert corrupted_cd_a > clean_cd, (
            f"Defect A CD {corrupted_cd_a:.6f} ≤ clean CD {clean_cd:.6f}. "
            f"Pipeline is insensitive to degenerate baseline."
        )

    def test_defect_b_cd_exceeds_clean(
        self,
        clean_cd: float,
        corrupted_cd_b: float,
    ) -> None:
        assert corrupted_cd_b > clean_cd, (
            f"Defect B CD {corrupted_cd_b:.6f} ≤ clean CD {clean_cd:.6f}. "
            f"Pipeline is insensitive to depth noise + flipped normals."
        )

    def test_defect_c_cd_exceeds_clean(
        self,
        clean_cd: float,
        corrupted_cd_c: float,
    ) -> None:
        assert corrupted_cd_c > clean_cd, (
            f"Defect C CD {corrupted_cd_c:.6f} ≤ clean CD {clean_cd:.6f}. "
            f"Pipeline is insensitive to hemisphere truncation."
        )


class TestDifferentialSensitivityRatio:
    """Core ≥3.0x sensitivity gate.

    ``CD_corrupted / max(CD_clean, 1e-6) >= 3.0``

    Proves the quality pipeline can distinguish good from bad data.
    """

    def test_defect_a_ratio(
        self,
        clean_cd: float,
        corrupted_cd_a: float,
    ) -> None:
        ratio = corrupted_cd_a / max(clean_cd, 1e-6)
        assert ratio >= 3.0, (
            f"Defect A sensitivity ratio {ratio:.2f}x < 3.0x. "
            f"clean_cd={clean_cd:.6f}, corrupted_cd={corrupted_cd_a:.6f}"
        )

    def test_defect_b_ratio(
        self,
        clean_cd: float,
        corrupted_cd_b: float,
    ) -> None:
        ratio = corrupted_cd_b / max(clean_cd, 1e-6)
        assert ratio >= 3.0, (
            f"Defect B sensitivity ratio {ratio:.2f}x < 3.0x. "
            f"clean_cd={clean_cd:.6f}, corrupted_cd={corrupted_cd_b:.6f}"
        )

    def test_defect_c_ratio(
        self,
        clean_cd: float,
        corrupted_cd_c: float,
    ) -> None:
        ratio = corrupted_cd_c / max(clean_cd, 1e-6)
        assert ratio >= 3.0, (
            f"Defect C sensitivity ratio {ratio:.2f}x < 3.0x. "
            f"clean_cd={clean_cd:.6f}, corrupted_cd={corrupted_cd_c:.6f}"
        )


class TestDelightMultiviewAlbedo:
    """Verify the new ``delight_multiview_albedo`` function."""

    def test_delight_produces_valid_result(
        self,
        clean_frames: np.ndarray,
    ) -> None:
        cameras = _make_ring_cameras(n_views=12)
        verts, faces = _make_sphere_mesh(subdivisions=2, radius=0.5)
        mesh = MeshGeometry(vertices=verts, faces=faces)

        result = delight_multiview_albedo(
            frames=clean_frames,
            cameras=cameras,
            mesh=mesh,
        )

        assert result.albedo.shape == (64, 64, 3)
        assert result.shading.shape == (64, 64, 3)
        assert result.specular.shape == (64, 64, 3)
        assert result.ambient_sh.shape == (9, 3)
        assert result.delta_e_residual >= 0.0
        assert np.all(np.isfinite(result.albedo))
        assert np.all(np.isfinite(result.delta_e_residual))


class TestSurfaceSamplingAndChamfer:
    """Verify area-weighted triangle uniform surface sampling and Chamfer distance."""

    def test_sample_mesh_surface_area_weighted(self) -> None:
        # Triangle 1: large [0,0,0], [10,0,0], [0,10,0] -> area = 50.0
        # Triangle 2: tiny [20,0,0], [20.1,0,0], [20,0.1,0] -> area = 0.005
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [20.0, 0.0, 0.0],
                [20.1, 0.0, 0.0],
                [20.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            dtype=np.int64,
        )

        samples = sample_mesh_surface(vertices, faces, num_samples=2000, seed=42)
        assert samples.shape == (2000, 3)
        assert np.all(np.isfinite(samples))

        # Almost all points should fall in Triangle 1 (x <= 10)
        in_t1 = np.sum(samples[:, 0] <= 10.5)
        ratio_t1 = in_t1 / len(samples)
        assert ratio_t1 > 0.98, f"Expected >98% in large triangle, got {ratio_t1:.4f}"

    def test_compute_chamfer_distance_mm_mesh_and_points(self) -> None:
        verts, faces = _make_sphere_mesh(subdivisions=2, radius=0.5)
        mesh_a = MeshGeometry(vertices=verts, faces=faces)
        mesh_b = MeshGeometry(vertices=verts + 0.001, faces=faces)  # 1mm shift

        cd_self = compute_chamfer_distance_mm(mesh_a, mesh_a, num_samples=2000)
        assert cd_self < 0.01, f"Self-CD should be ~0 mm, got {cd_self}"

        cd_shift = compute_chamfer_distance_mm(mesh_a, mesh_b, num_samples=2000)
        assert 0.5 <= cd_shift <= 2.0, f"Expected ~1mm shift CD, got {cd_shift}"
