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
from synapse.metrics.roundtrip import chamfer_mm
from synapse.recon.delight import delight_multiview_albedo
from synapse.recon.types import CameraExtrinsics, MeshGeometry


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_ring_cameras(
    n_views: int = 12,
    radius: float = 2.0,
) -> list[CameraExtrinsics]:
    """Create cameras on a horizontal ring looking at the origin."""
    cameras: list[CameraExtrinsics] = []
    for i in range(n_views):
        theta = 2.0 * math.pi * i / n_views
        pos = np.array(
            [radius * math.sin(theta), 0.0, radius * math.cos(theta)],
            dtype=np.float64,
        )
        z_axis = -pos / (np.linalg.norm(pos) + 1e-8)
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_len = np.linalg.norm(x_axis)
        if x_len < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis /= x_len
        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        t = -R @ pos
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        cameras.append(
            CameraExtrinsics(
                frame=i, transform=T, position=pos,
                rotation_matrix=R, yaw_rad=theta,
            )
        )
    return cameras


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


def _inject_defect_b_noise(gt_pts: np.ndarray) -> np.ndarray:
    """Defect B: flipped normals (-N) + depth noise σ=0.15m.

    Negates vertex positions (simulating inside-out normals) and adds
    heavy Gaussian perturbation to simulate reconstruction from
    corrupted depth maps.
    """
    rng = np.random.RandomState(99)
    flipped = -gt_pts.copy()
    noise = rng.randn(*flipped.shape).astype(np.float32) * 0.15
    return flipped + noise


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
    return _sample_sphere_surface(n_points=4096, radius=0.5)


@pytest.fixture(scope="module")
def clean_recon(gt_surface: np.ndarray) -> np.ndarray:
    """Simulate a 'good reconstruction': GT surface + tiny noise (σ=2mm).

    This represents a clean, accurate reconstruction that closely
    matches the ground-truth sphere geometry.
    """
    rng = np.random.RandomState(42)
    return gt_surface + rng.randn(*gt_surface.shape).astype(np.float32) * 0.002


@pytest.fixture(scope="module")
def clean_cd(clean_recon: np.ndarray, gt_surface: np.ndarray) -> float:
    """Chamfer Distance (metres) for clean reconstruction vs GT."""
    return chamfer_mm(clean_recon, gt_surface) / 1000.0


@pytest.fixture(scope="module")
def corrupted_cd_a(gt_surface: np.ndarray) -> float:
    """CD for Defect A: degenerate 1D baseline."""
    degen = _inject_defect_a_degenerate(gt_surface)
    return chamfer_mm(degen, gt_surface) / 1000.0


@pytest.fixture(scope="module")
def corrupted_cd_b(gt_surface: np.ndarray) -> float:
    """CD for Defect B: flipped normals + heavy noise."""
    noisy = _inject_defect_b_noise(gt_surface)
    return chamfer_mm(noisy, gt_surface) / 1000.0


@pytest.fixture(scope="module")
def corrupted_cd_c(gt_surface: np.ndarray) -> float:
    """CD for Defect C: upper hemisphere truncation."""
    trunc = _inject_defect_c_truncation(gt_surface)
    return chamfer_mm(trunc, gt_surface) / 1000.0


@pytest.fixture(scope="module")
def clean_frames() -> np.ndarray:
    return _make_clean_frames(n_views=12, h=64, w=64, seed=42)


@pytest.fixture(scope="module")
def clean_image_metrics(clean_frames: np.ndarray) -> dict[str, float]:
    """PSNR and SSIM for clean frames vs their mean."""
    mean_frame = np.mean(clean_frames, axis=0)
    rendered = np.full_like(mean_frame, np.mean(mean_frame))
    return {
        "psnr": psnr_db(rendered, mean_frame),
        "ssim": ssim(rendered, mean_frame),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundtripGoldenPipelinePassesGate:
    """Clean data must pass quality gates: CD ≤ 0.008, PSNR ≥ 26.0, SSIM ≥ 0.88."""

    def test_chamfer_distance_within_threshold(self, clean_cd: float) -> None:
        assert clean_cd <= 0.008, (
            f"Clean CD {clean_cd:.6f} m exceeds threshold 0.008 m."
        )

    def test_reconstruction_point_count(self, clean_recon: np.ndarray) -> None:
        assert len(clean_recon) > 100, (
            f"Clean reconstruction has only {len(clean_recon)} points — "
            f"likely degenerate."
        )

    def test_psnr_above_floor(
        self, clean_image_metrics: dict[str, float],
    ) -> None:
        p = clean_image_metrics["psnr"]
        assert p >= 26.0, f"Clean PSNR {p:.2f} dB < 26.0 dB threshold"

    def test_ssim_above_floor(
        self, clean_image_metrics: dict[str, float],
    ) -> None:
        s = clean_image_metrics["ssim"]
        assert s >= 0.88, f"Clean SSIM {s:.4f} < 0.88 threshold"


class TestRoundtripCorruptedInputFailsGate:
    """All 3 defect types must produce a worse (higher) Chamfer Distance."""

    def test_defect_a_cd_exceeds_clean(
        self, clean_cd: float, corrupted_cd_a: float,
    ) -> None:
        assert corrupted_cd_a > clean_cd, (
            f"Defect A CD {corrupted_cd_a:.6f} ≤ clean CD {clean_cd:.6f}. "
            f"Pipeline is insensitive to degenerate baseline."
        )

    def test_defect_b_cd_exceeds_clean(
        self, clean_cd: float, corrupted_cd_b: float,
    ) -> None:
        assert corrupted_cd_b > clean_cd, (
            f"Defect B CD {corrupted_cd_b:.6f} ≤ clean CD {clean_cd:.6f}. "
            f"Pipeline is insensitive to depth noise + flipped normals."
        )

    def test_defect_c_cd_exceeds_clean(
        self, clean_cd: float, corrupted_cd_c: float,
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
        self, clean_cd: float, corrupted_cd_a: float,
    ) -> None:
        ratio = corrupted_cd_a / max(clean_cd, 1e-6)
        assert ratio >= 3.0, (
            f"Defect A sensitivity ratio {ratio:.2f}x < 3.0x. "
            f"clean_cd={clean_cd:.6f}, corrupted_cd={corrupted_cd_a:.6f}"
        )

    def test_defect_b_ratio(
        self, clean_cd: float, corrupted_cd_b: float,
    ) -> None:
        ratio = corrupted_cd_b / max(clean_cd, 1e-6)
        assert ratio >= 3.0, (
            f"Defect B sensitivity ratio {ratio:.2f}x < 3.0x. "
            f"clean_cd={clean_cd:.6f}, corrupted_cd={corrupted_cd_b:.6f}"
        )

    def test_defect_c_ratio(
        self, clean_cd: float, corrupted_cd_c: float,
    ) -> None:
        ratio = corrupted_cd_c / max(clean_cd, 1e-6)
        assert ratio >= 3.0, (
            f"Defect C sensitivity ratio {ratio:.2f}x < 3.0x. "
            f"clean_cd={clean_cd:.6f}, corrupted_cd={corrupted_cd_c:.6f}"
        )


class TestDelightMultiviewAlbedo:
    """Verify the new ``delight_multiview_albedo`` function."""

    def test_delight_produces_valid_result(
        self, clean_frames: np.ndarray,
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
