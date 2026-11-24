"""Tests for M45-M46: Video I/O, Foundary 12-service async client, multiview, and .fmesh.bin."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from synapse.bridge.aether_geo_worker_client import (
    FMESH_MAGIC,
    AetherGeoWorkerClient,
    read_fmesh_bin,
    write_fmesh_bin,
)
from synapse.bridge.foundary_client import (
    FOUNDARY_SERVICE_NAMES,
    FoundaryClient,
    FoundaryRequest,
    MockFoundaryTransport,
)
from synapse.errors import (
    AssetNotFoundError,
    BackendUnavailableError,
    ContractValidationError,
    NotImplementedStageError,
)
from synapse.recon.geometry import quad_remesh
from synapse.recon.io import export_recon_fgpp_sequence, load_video_frames
from synapse.recon.multiview import (
    Hunyuan3DReconstructionAdapter,
    TrellisReconstructionAdapter,
    select_adaptive_keyframes,
)
from synapse.recon.pipeline import ReconPipeline
from synapse.recon.tracking import (
    anchor_metric_scale,
    solve_camera_trajectory,
    solve_point_tracks,
)
from synapse.recon.types import CameraExtrinsics, CameraTrajectory, MeshGeometry


def test_load_video_frames_npy_npz_dir_and_fail_closed(tmp_path: Path) -> None:
    """PR-193: load_video_frames handles .npy, .npz, image/npy dir, and fails closed."""
    arr = np.full((6, 32, 32, 3), 128, dtype=np.uint8)
    npy_path = tmp_path / "clip.npy"
    np.save(npy_path, arr)
    loaded, meta = load_video_frames(npy_path, stride=2, max_frames=2, target_size=(16, 16))
    assert loaded.shape == (2, 16, 16, 3)
    assert meta.codec == "npy"
    assert meta.width == 16 and meta.height == 16

    npz_path = tmp_path / "clip.npz"
    np.savez(npz_path, frames=arr)
    loaded_z, meta_z = load_video_frames(npz_path)
    assert loaded_z.shape == (6, 32, 32, 3)
    assert meta_z.codec == "npz"

    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    for i in range(4):
        np.save(seq_dir / f"f_{i:03d}.npy", arr[i])
    loaded_d, meta_d = load_video_frames(seq_dir)
    assert loaded_d.shape == (4, 32, 32, 3)
    assert meta_d.codec == "image_sequence"

    with pytest.raises(AssetNotFoundError) as e20:
        load_video_frames(tmp_path / "missing.mp4")
    assert e20.value.exit_code == 20

    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("not video", encoding="utf-8")
    with pytest.raises(ContractValidationError) as e10:
        load_video_frames(bad_file)
    assert e10.value.exit_code == 10


def test_export_recon_fgpp_sequence_roundtrip(tmp_path: Path) -> None:
    """PR-193: export_recon_fgpp_sequence writes 15ch EXRs + metadata.json when OpenEXR exists."""
    pytest.importorskip("OpenEXR")
    tet_v = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
        dtype=np.float32,
    )
    tet_f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    mesh = MeshGeometry(vertices=tet_v, faces=tet_f)
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    exts = [
        CameraExtrinsics(
            frame=i,
            transform=np.eye(4, dtype=np.float64),
            position=np.array([0.0, 0.0, 2.5], dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
            yaw_rad=0.0,
        )
        for i in range(2)
    ]
    traj = CameraTrajectory(frame_count=2, intrinsics={"fx": 32.0, "fy": 32.0, "cx": 16.0, "cy": 16.0}, extrinsics=exts)
    out_dir = tmp_path / "fgpp_out"
    paths = export_recon_fgpp_sequence(frames, traj, mesh, out_dir, fps=24.0)
    assert len(paths) == 2
    assert (out_dir / "metadata.json").is_file()


def test_foundary_client_10_services_async_and_tracking_scale() -> None:
    """PR-194: FoundaryClient supports all services + async polling, plus anchor_metric_scale."""
    for expected in (
        "cotracker",
        "megasam",
        "depth_anything",
        "trellis",
        "hunyuan3d",
        "sam2",
        "delight",
        "wan",
        "flux",
        "sdxl",
    ):
        assert expected in FOUNDARY_SERVICE_NAMES

    mock_transport = MockFoundaryTransport()
    client = FoundaryClient(mock_transport)
    resp = client.infer("cotracker", inputs={"num_frames": 4}, async_mode=True)
    assert resp.status == "ok"
    assert resp.service == "cotracker"
    job_st = mock_transport.submit_job(FoundaryRequest(service="delight", job_id="j1"))
    assert job_st.state == "succeeded"

    # Unreachable HTTP endpoint fails closed with BackendUnavailableError (exit 30)
    bad_client = FoundaryClient()
    with pytest.raises(BackendUnavailableError) as e30:
        bad_client.infer("trellis")
    assert e30.value.exit_code == 30

    # Tracking Foundary hook + affine metric scale anchoring
    class _TrackStubClient:
        def infer(self, service: str, **kwargs: Any) -> Any:
            if service == "cotracker":
                return type(
                    "R",
                    (),
                    {
                        "outputs": {
                            "tracks_2d": np.zeros((4, 8, 2), dtype=np.float32),
                            "visibility": np.ones((4, 8), dtype=bool),
                        }
                    },
                )()
            return type("R", (), {"outputs": {"metric_scale": 2.5}})()

    stub = _TrackStubClient()
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    tracks, vis = solve_point_tracks(frames, num_points=8, foundary_client=stub)
    assert tracks.shape == (4, 8, 2) and vis.all()
    traj = solve_camera_trajectory(tracks, frame_height=32, frame_width=32, visibility=vis, foundary_client=stub)
    assert traj.metric_scale == pytest.approx(2.5)

    rel_depths = np.full((2, 8, 8), 2.0, dtype=np.float32)
    gt_depths = np.full((2, 8, 8), 6.0, dtype=np.float32)
    scaled_traj, ratio = anchor_metric_scale(rel_depths, traj, metric_depths=gt_depths)
    assert ratio == pytest.approx(3.0)
    assert scaled_traj.metric_scale == pytest.approx(7.5)


def test_adaptive_keyframes_gen3d_adapters_and_redline3() -> None:
    """PR-195: select_adaptive_keyframes, TRELLIS/Hunyuan3D adapters, and Redline #3 floor."""
    rng = np.random.default_rng(42)
    frames = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    for i in range(10):
        # Sharp frames on even indices, blurry flat on odd
        frames[i] = (
            rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            if i % 2 == 0
            else np.full((32, 32, 3), 100, dtype=np.uint8)
        )
    kf_idx = select_adaptive_keyframes(frames, target_views=4)
    assert len(kf_idx) == 4
    assert kf_idx == sorted(kf_idx)

    trellis_mesh = TrellisReconstructionAdapter().reconstruct(frames[kf_idx])
    hunyuan_mesh = Hunyuan3DReconstructionAdapter().reconstruct(frames[kf_idx])
    assert trellis_mesh.vertex_count >= 1500
    assert hunyuan_mesh.vertex_count >= 1500

    # Redline #3: coarse mesh rejected fail-closed with exit 10
    coarse_mesh = MeshGeometry(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
    )
    pipe = ReconPipeline(geometry_backend="python")
    with pytest.raises(ContractValidationError) as e10:
        pipe.run(
            frames[:4],
            initial_mesh=coarse_mesh,
            require_character_floor=True,
            has_gen3d_highpoly=True,
        )
    assert e10.value.exit_code == 10


def test_fmesh_bin_wire_codec_and_quadriflow_neg_100_08(tmp_path: Path) -> None:
    """PR-196: .fmesh.bin 64B header + 16B aligned streams + NEG-100-08 exit 50 on stub."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    norms = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (4, 1))
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

    bin_path = tmp_path / "mesh.fmesh.bin"
    write_fmesh_bin(bin_path, verts, faces, normals=norms, uvs=uvs)
    raw = bin_path.read_bytes()
    magic = struct.unpack_from("<I", raw, 0)[0]
    assert magic == FMESH_MAGIC

    decoded = read_fmesh_bin(bin_path)
    np.testing.assert_allclose(decoded.vertices, verts)
    np.testing.assert_array_equal(decoded.faces, faces)
    assert decoded.normals is not None and decoded.uvs is not None
    np.testing.assert_allclose(decoded.normals, norms)
    np.testing.assert_allclose(decoded.uvs, uvs)

    # Simulate aether_geo_worker stub returning exit code 1 for quadriflow (NEG-100-08)
    stub_bin = tmp_path / "mock_aether_geo_worker.py"
    stub_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if 'quadriflow' in sys.argv:\n"
        "    sys.stderr.write('quadriflow not implemented (contract stub)\\n')\n"
        "    sys.exit(1)\n"
        "src = sys.argv[sys.argv.index('--input') + 1]\n"
        "dst = sys.argv[sys.argv.index('--output') + 1]\n"
        "with open(src, 'rb') as fi, open(dst, 'wb') as fo:\n"
        "    fo.write(fi.read())\n",
        encoding="utf-8",
    )
    stub_bin.chmod(0o755)
    client = AetherGeoWorkerClient(worker_bin=stub_bin)

    remediated = client.remediate(verts, faces, normals=norms, uvs=uvs)
    assert remediated.vertices.shape == verts.shape

    with pytest.raises(NotImplementedStageError) as e50:
        client.quadriflow(verts, faces, target_faces=1500)
    assert e50.value.exit_code == 50

    with pytest.raises(NotImplementedStageError) as e50_remesh:
        quad_remesh(verts, faces, target_faces=1500, worker_client=client, strict_native=True)
    assert e50_remesh.value.exit_code == 50
