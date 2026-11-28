"""Cross-pipeline wiring tests for Horizon 14 (PR-220, PR-221, PR-223, PR-226, PR-227)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synapse.adapters.lora.manager import MultiLoRAManager
from synapse.adapters.lora.manifest import LoRAManifest
from synapse.bridge.diffusion_renderer import (
    FoundaryDiffusionRendererBackend,
    MockDiffusionRendererBackend,
)
from synapse.bridge.sam2_transport import MockSAM2Backend
from synapse.cli import main_synapse
from synapse.composite.mask import generate_surgical_mask, generate_surgical_mask_sequence
from synapse.errors import BackendUnavailableError
from synapse.recon.geometry.topology import check_manifold, check_watertight
from synapse.recon.pipeline import ReconPipeline
from synapse.recon.trellis2 import TopologyPolicy
from synapse.recon.types import MeshGeometry
from synapse.training.feature_store import (
    FeatureStoreBuilder,
    LinearProjectionEncoder,
    LinearProjectionTextEncoder,
    iter_feature_shards,
)


def _closed_tetrahedron() -> MeshGeometry:
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    # Consistently oriented closed tetrahedron
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return MeshGeometry(vertices=verts, faces=faces)


def _open_quad_mesh() -> MeshGeometry:
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return MeshGeometry(vertices=verts, faces=faces)


# ---------------------------------------------------------------------------
# PR-226: Surgical mask SAM2 VOS + scene-cut reset
# ---------------------------------------------------------------------------


class TestPR226SurgicalMaskVOS:
    """Upgraded M_surgical(t) formula and sequence tracker reset across cuts."""

    def test_generate_surgical_mask_unions_sam2_vos(self) -> None:
        h, w = 64, 64
        m3d = np.zeros((h, w), dtype=bool)
        m_old = np.zeros((h, w), dtype=bool)
        m_vos = np.zeros((h, w), dtype=bool)
        m3d[5:12, 5:12] = True
        m_old[25:32, 25:32] = True
        m_vos[48:56, 48:56] = True

        alpha = generate_surgical_mask(
            m3d,
            mask_sam2_old=m_old,
            mask_sam2_vos=m_vos,
            r_base=0,
            feather_width=0.0,
        )
        assert np.all(alpha[5:12, 5:12] == 1.0)
        assert np.all(alpha[25:32, 25:32] == 1.0)
        assert np.all(alpha[48:56, 48:56] == 1.0)

    def test_sequence_restarts_sam2_on_scene_cuts(self) -> None:
        t_len, h, w = 8, 32, 32
        frames = np.full((t_len, h, w, 3), 120, dtype=np.uint8)
        m3d = np.zeros((t_len, h, w), dtype=bool)
        m3d[:, 4:10, 4:10] = True
        crypto = np.zeros((t_len, h, w), dtype=np.uint32)
        crypto[:, 18:24, 18:24] = 3

        alphas, seg_count = generate_surgical_mask_sequence(
            frames,
            m3d,
            cryptomatte_ids=crypto,
            sam2_transport=MockSAM2Backend(),
            cut_frames=[4],
            dilation_px=2,
            r_base=0,
            feather_width=0.0,
        )
        assert seg_count == 2
        assert alphas.shape == (t_len, h, w)
        assert np.all(alphas[:, 4:10, 4:10] == 1.0)
        assert np.all(alphas[:, 18:24, 18:24] == 1.0)


# ---------------------------------------------------------------------------
# PR-227: MultiLoRAManager SAM2 VOS mask gating without Cryptomatte
# ---------------------------------------------------------------------------


class TestPR227MultiLoRASAM2Gating:
    """MultiLoRAManager supports SAM2 masks directly without Cryptomatte."""

    def test_build_spatial_lora_masks_from_sam2_array_and_result(self) -> None:
        mgr = MultiLoRAManager(lora_bundles=[], test_synthetic=True)
        raw = np.zeros((2, 16, 16), dtype=bool)
        raw[:, :8, :8] = True
        ds = mgr.build_spatial_lora_masks_from_sam2(raw, spatial_block=8)
        assert ds.shape == (2, 2, 2)
        assert bool(ds[0, 0, 0]) is True
        assert bool(ds[0, 1, 1]) is False

        sam2_res = MockSAM2Backend().segment_video(np.zeros((2, 64, 64, 3), dtype=np.uint8))
        ds_res = mgr.build_spatial_lora_masks_from_sam2(sam2_res, spatial_block=8)
        assert ds_res.shape == (2, 8, 8)
        assert np.any(ds_res)

    def test_manage_spatial_hooks_works_with_sam2_only(self) -> None:
        torch = pytest.importorskip("torch")

        m_id = LoRAManifest(
            name="id-sam2",
            category="identity",
            rank=4,
            alpha=4.0,
            adapter_path="id.safetensors",
            adapter_sha256="a" * 64,
            target_modules=["proj"],
        )
        m_st = LoRAManifest(
            name="style-sam2",
            category="style",
            rank=4,
            alpha=4.0,
            adapter_path="st.safetensors",
            adapter_sha256="b" * 64,
            target_modules=["cross_attn.to_k"],
        )

        class _TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(8, 8)
                self.cross_attn = torch.nn.ModuleDict({"to_k": torch.nn.Linear(8, 8)})

        model = _TinyModel()
        mgr = MultiLoRAManager([(m_id, 1.0), (m_st, 1.0)], test_synthetic=True)
        sam2_mask = np.zeros((1, 8, 8), dtype=bool)
        sam2_mask[:, :4, :4] = True
        # Flatten tokens matching latent grid: here block=8 -> 1x1x1 tokens
        x = torch.ones((1, 1, 8))
        with mgr.manage_spatial_hooks(model, sam2_masks=sam2_mask):
            out = model.proj(x)
            assert out.shape == (1, 1, 8)


# ---------------------------------------------------------------------------
# PR-223: TopologyPolicy parameter on check_manifold & check_watertight
# ---------------------------------------------------------------------------


class TestPR223GeometryTopologyPolicy:
    """check_manifold and check_watertight accept topology_policy cleanly."""

    def test_strict_vs_open_surface_allowed_vs_permissive(self) -> None:
        closed = _closed_tetrahedron()
        assert check_manifold(closed.vertices, closed.faces) is True
        assert check_watertight(closed.vertices, closed.faces) is True

        open_m = _open_quad_mesh()
        assert check_manifold(open_m.vertices, open_m.faces, topology_policy="strict") is True
        assert check_watertight(open_m.vertices, open_m.faces, topology_policy="strict") is False
        assert (
            check_watertight(
                open_m.vertices,
                open_m.faces,
                topology_policy=TopologyPolicy.OPEN_SURFACE_ALLOWED,
            )
            is True
        )
        assert check_manifold(open_m.vertices, open_m.faces, topology_policy="permissive") is True


# ---------------------------------------------------------------------------
# PR-220: ReconPipeline DiffusionRenderer & Dual-Track comparison
# ---------------------------------------------------------------------------


class TestPR220ReconPipelineDiffusionRenderer:
    """ReconPipeline supports diffusion_renderer and dual_track backends."""

    def test_diffusion_renderer_and_dual_track(self) -> None:
        rng = np.random.RandomState(220)
        frames = rng.randint(20, 220, (4, 16, 16, 3), dtype=np.uint8)
        prior_mesh = _closed_tetrahedron()

        pipe_dr = ReconPipeline(
            target_quad_faces=4,
            geometry_backend="python",
            inverse_backend="diffusion_renderer",
            diffusion_renderer_transport=MockDiffusionRendererBackend(),
        )
        res_dr = pipe_dr.run(frames, initial_mesh=prior_mesh, track="track_a")
        assert res_dr.telemetry["inverse_backend"] == "diffusion_renderer"
        assert res_dr.telemetry["dual_track_comparison"]["mode"] == "diffusion_renderer"

        pipe_dual = ReconPipeline(
            target_quad_faces=4,
            geometry_backend="python",
            inverse_backend="dual_track",
            diffusion_renderer_transport=MockDiffusionRendererBackend(),
        )
        res_dual = pipe_dual.run(frames, initial_mesh=prior_mesh, track="track_a")
        assert res_dual.telemetry["inverse_backend"] == "dual_track"
        cmp_meta = res_dual.telemetry["dual_track_comparison"]
        assert "albedo_psnr" in cmp_meta
        assert "normal_cosine_sim" in cmp_meta
        assert "depth_rmse" in cmp_meta

    def test_unconnected_diffusion_renderer_fails_closed(self) -> None:
        frames = np.full((3, 16, 16, 3), 100, dtype=np.uint8)
        pipe = ReconPipeline(
            target_quad_faces=4,
            geometry_backend="python",
            inverse_backend="diffusion_renderer",
            diffusion_renderer_transport=FoundaryDiffusionRendererBackend(),
        )
        with pytest.raises(BackendUnavailableError) as exc:
            pipe.run(frames, initial_mesh=_closed_tetrahedron(), track="track_a")
        assert exc.value.exit_code == 30

    def test_cli_recon_invert_video_help_exposes_inverse_backend(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as se:
            main_synapse(["recon", "invert-video", "--help"])
        assert se.value.code == 0
        out = capsys.readouterr().out
        assert "--inverse-backend" in out


# ---------------------------------------------------------------------------
# PR-221: FeatureStoreBuilder multi-light relighting augmentation
# ---------------------------------------------------------------------------


class TestPR221FeatureStoreRelightAugmentation:
    """FeatureStoreBuilder generates >=8 distinct relit variants via DiffusionRenderer."""

    def test_generates_eight_lighting_variants_passing_shards(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "shards"
        frames = np.full((4, 16, 16, 3), 140, dtype=np.uint8)
        envs = [
            np.full((4, 8, 3), 0.25 + 0.1 * i, dtype=np.float32)
            for i in range(8)
        ]

        builder = FeatureStoreBuilder(LinearProjectionEncoder(), LinearProjectionTextEncoder())
        accepted = builder.add_relight_augmented_samples(
            sample_id="hero_prop",
            video_frames=frames,
            caption="sks anodized robot, studio lighting",
            environment_maps=envs,
            renderer=MockDiffusionRendererBackend(),
            lighting_labels=[f"hdri_{i}" for i in range(8)],
        )
        assert accepted == 8
        shard_paths = builder.build_shards(out_dir)
        samples = list(iter_feature_shards(shard_paths))
        assert len(samples) == 8
        latents = [s["latent"] for s in samples]
        assert not np.allclose(latents[0], latents[-1])
