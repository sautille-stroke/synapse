"""Tests for PR-17 LoRA FeatureStore, encoders, and captions.

Covers:
* Structured caption formatting and validation
* LinearProjectionEncoder output shape and determinism
* LinearProjectionTextEncoder output shape and determinism
* FeatureStoreBuilder shard writing and reading
* lora_preencode stage registration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.training.feature_store import (
    FeatureStoreBuilder,
    LinearProjectionEncoder,
    LinearProjectionTextEncoder,
    format_structured_caption,
    iter_feature_shards,
)

# ===================================================================
# TestStructuredCaption
# ===================================================================


class TestStructuredCaption:
    """Validation tests for ``format_structured_caption``."""

    def test_valid_caption_format(self) -> None:
        """All 5 fields concatenated with ', '."""
        caption = format_structured_caption(
            trigger_token="sks",
            subject_desc="a red car",
            lighting_env="sunset",
            camera_motion="orbit",
            art_style_tag="photorealistic",
        )
        assert "sks" in caption
        assert "a red car" in caption
        assert "sunset" in caption
        assert "orbit" in caption
        assert "photorealistic" in caption
        parts = caption.split(", ")
        assert len(parts) == 5

    def test_empty_trigger_raises(self) -> None:
        """Empty trigger_token raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="trigger"):
            format_structured_caption(
                trigger_token="",
                subject_desc="car",
                lighting_env="day",
                camera_motion="static",
                art_style_tag="anime",
            )

    def test_empty_subject_raises(self) -> None:
        """Empty subject_desc raises ContractValidationError."""
        with pytest.raises(
            ContractValidationError, match="subject",
        ):
            format_structured_caption(
                trigger_token="sks",
                subject_desc="",
                lighting_env="day",
                camera_motion="static",
                art_style_tag="anime",
            )

    def test_empty_lighting_raises(self) -> None:
        """Empty lighting_env raises ContractValidationError."""
        with pytest.raises(
            ContractValidationError, match="lighting",
        ):
            format_structured_caption(
                trigger_token="sks",
                subject_desc="car",
                lighting_env="",
                camera_motion="static",
                art_style_tag="anime",
            )

    def test_empty_camera_raises(self) -> None:
        """Empty camera_motion raises ContractValidationError."""
        with pytest.raises(
            ContractValidationError, match="camera",
        ):
            format_structured_caption(
                trigger_token="sks",
                subject_desc="car",
                lighting_env="day",
                camera_motion="",
                art_style_tag="anime",
            )

    def test_empty_style_raises(self) -> None:
        """Empty art_style_tag raises ContractValidationError."""
        with pytest.raises(
            ContractValidationError, match="art_style",
        ):
            format_structured_caption(
                trigger_token="sks",
                subject_desc="car",
                lighting_env="day",
                camera_motion="static",
                art_style_tag="",
            )

    def test_special_chars_preserved(self) -> None:
        """Special characters are preserved in the caption."""
        caption = format_structured_caption(
            trigger_token="sks_v2",
            subject_desc='a "red" car & truck',
            lighting_env="golden-hour (warm)",
            camera_motion="dolly-zoom",
            art_style_tag="neo-impressionist",
        )
        assert '"red"' in caption
        assert "&" in caption
        assert "(warm)" in caption


# ===================================================================
# TestLinearProjectionEncoder
# ===================================================================


class TestLinearProjectionEncoder:
    """Tests for ``LinearProjectionEncoder``."""

    def test_output_shape(self) -> None:
        """[T=17, H=64, W=64, 3] -> [16, 5, 8, 8]."""
        enc = LinearProjectionEncoder()
        frames = np.random.RandomState(0).randint(
            0, 256, (17, 64, 64, 3), dtype=np.uint8,
        )
        latent = enc.encode(frames)
        # tau = (17-1)//4 + 1 = 5
        assert latent.shape == (16, 5, 8, 8)
        assert latent.dtype == np.float32

    def test_deterministic(self) -> None:
        """Same input produces identical output."""
        enc = LinearProjectionEncoder()
        frames = np.random.RandomState(1).randint(
            0, 256, (9, 32, 32, 3), dtype=np.uint8,
        )
        a = enc.encode(frames)
        b = enc.encode(frames)
        np.testing.assert_array_equal(a, b)

    def test_different_input_different_output(self) -> None:
        """Different frames produce different latents."""
        enc = LinearProjectionEncoder()
        f1 = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        f2 = np.full((5, 16, 16, 3), 200, dtype=np.uint8)
        a = enc.encode(f1)
        b = enc.encode(f2)
        assert not np.allclose(a, b)

    def test_temporal_compression(self) -> None:
        """Verify tau = (T-1)//4 + 1 for various T."""
        enc = LinearProjectionEncoder()
        for t_in, tau_exp in [(5, 2), (9, 3), (17, 5), (33, 9)]:
            frames = np.zeros(
                (t_in, 16, 16, 3), dtype=np.uint8,
            )
            latent = enc.encode(frames)
            assert latent.shape[1] == tau_exp, (
                f"T={t_in}: expected tau={tau_exp}, "
                f"got {latent.shape[1]}"
            )


# ===================================================================
# TestLinearProjectionTextEncoder
# ===================================================================


class TestLinearProjectionTextEncoder:
    """Tests for ``LinearProjectionTextEncoder``."""

    def test_output_shape(self) -> None:
        """Returns [512] float32."""
        enc = LinearProjectionTextEncoder()
        emb = enc.encode("hello world")
        assert emb.shape == (512,)
        assert emb.dtype == np.float32

    def test_deterministic(self) -> None:
        """Same text produces identical embedding."""
        enc = LinearProjectionTextEncoder()
        a = enc.encode("test caption")
        b = enc.encode("test caption")
        np.testing.assert_array_equal(a, b)

    def test_different_text_different_output(self) -> None:
        """Different text produces different embeddings."""
        enc = LinearProjectionTextEncoder()
        a = enc.encode("cat")
        b = enc.encode("dog")
        assert not np.allclose(a, b)

    def test_custom_dim(self) -> None:
        """Custom embed_dim produces correct shape."""
        enc = LinearProjectionTextEncoder(embed_dim=256)
        emb = enc.encode("test")
        assert emb.shape == (256,)


# ===================================================================
# TestFeatureStoreBuilder
# ===================================================================


class TestFeatureStoreBuilder:
    """Tests for ``FeatureStoreBuilder`` shard writing/reading."""

    def test_build_single_shard(self, tmp_path: object) -> None:
        """3 samples -> 1 shard, all readable back."""
        tmp = tmp_path
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        rng = np.random.RandomState(42)
        for i in range(3):
            frames = rng.randint(
                0, 256, (17, 64, 64, 3), dtype=np.uint8,
            )
            builder.add_sample(
                sample_id=f"s{i:03d}",
                video_frames=frames,
                caption=f"sample {i}",
                source_sha256=f"{'a' * 64}",
            )

        shard_paths = builder.build_shards(tmp / "shards")  # type: ignore[operator]
        assert len(shard_paths) == 1
        assert shard_paths[0].exists()

        # Read back
        samples = list(iter_feature_shards(shard_paths))
        assert len(samples) == 3

        for s in samples:
            assert "sample_id" in s
            assert s["latent"].shape == (16, 5, 8, 8)
            assert s["prompt"].shape == (512,)
            assert s["meta"]["source_sha256"] == "a" * 64

    def test_build_multiple_shards(
        self, tmp_path: object,
    ) -> None:
        """Small max_shard_bytes forces multiple shards."""
        tmp = tmp_path
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(
            enc, txt, max_shard_bytes=1024,
        )

        rng = np.random.RandomState(0)
        for i in range(3):
            frames = rng.randint(
                0, 256, (17, 32, 32, 3), dtype=np.uint8,
            )
            builder.add_sample(
                sample_id=f"multi_{i}",
                video_frames=frames,
                caption=f"caption {i}",
            )

        shard_paths = builder.build_shards(tmp / "shards")  # type: ignore[operator]
        assert len(shard_paths) > 1, (
            "Expected multiple shards with small size limit"
        )

        # All samples recoverable
        samples = list(iter_feature_shards(shard_paths))
        assert len(samples) == 3
        ids = {s["sample_id"] for s in samples}
        assert ids == {"multi_0", "multi_1", "multi_2"}

    def test_fgpp_cond_roundtrip(
        self, tmp_path: object,
    ) -> None:
        """FGPP condition tensor round-trips through shards."""
        tmp = tmp_path
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        cond = np.random.RandomState(99).randn(
            4, 2, 2, 2,
        ).astype(np.float32)

        builder.add_sample(
            sample_id="cond_test",
            video_frames=frames,
            caption="cond sample",
            fgpp_cond=cond,
        )

        shards = builder.build_shards(tmp / "shards")  # type: ignore[operator]
        samples = list(iter_feature_shards(shards))
        assert len(samples) == 1
        assert samples[0]["fgpp_cond"] is not None
        np.testing.assert_allclose(
            samples[0]["fgpp_cond"], cond, atol=1e-6,
        )

    def test_no_fgpp_cond_is_none(
        self, tmp_path: object,
    ) -> None:
        """Samples without fgpp_cond yield None."""
        tmp = tmp_path
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        builder.add_sample(
            sample_id="no_cond",
            video_frames=frames,
            caption="no cond",
        )

        shards = builder.build_shards(tmp / "shards")  # type: ignore[operator]
        samples = list(iter_feature_shards(shards))
        assert samples[0]["fgpp_cond"] is None

    def test_sha256_in_meta(
        self, tmp_path: object,
    ) -> None:
        """source_sha256 appears in meta."""
        tmp = tmp_path
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        sha = "b" * 64
        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        builder.add_sample(
            sample_id="sha_test",
            video_frames=frames,
            caption="sha sample",
            source_sha256=sha,
        )

        shards = builder.build_shards(tmp / "shards")  # type: ignore[operator]
        samples = list(iter_feature_shards(shards))
        assert samples[0]["meta"]["source_sha256"] == sha


# ===================================================================
# TestLoraPreencodeStage
# ===================================================================


class TestLoraPreencodeStage:
    """Tests for ``lora_preencode`` stage registration."""

    def test_stage_registered(self) -> None:
        """lora_preencode is in the registered stages list."""
        from synapse.pipeline.bundle_worker import (
            get_registered_stages,
        )

        stages = get_registered_stages()
        assert "lora_preencode" in stages


# ===================================================================
# TestFeatureStoreQualityGates  (PR-77)
# ===================================================================


class TestFeatureStoreQualityGates:
    """PR-77: Validate quality gates with distinct FGPP / Wan layouts."""

    @staticmethod
    def _make_valid_fgpp(t: int = 17, h: int = 32, w: int = 32) -> np.ndarray:
        """Build a valid ``[T, H, W, 15]`` raw FGPP tensor.

        Channel layout per ``fgpp_types.h``:
        - 0: CameraLinearDepth.Z  — varies 1.0..10.0
        - 1-3: DirectRadiance.RGB — moderate values < 65504
        - 4-6: WorldNormal.XYZ    — unit-length [0,0,1]
        - 7-13: Padding / reserved — zero
        - 14: Cryptomatte.ID      — non-zero foreground
        """
        fgpp = np.zeros((t, h, w, 15), dtype=np.float32)
        # Gate 2: depth span > 0.1
        fgpp[..., 0] = np.linspace(1.0, 10.0, h * w).reshape(1, h, w)
        # Gate 3: radiance < 65504
        fgpp[..., 1:4] = 500.0
        # Gate 4: unit-length normals
        fgpp[..., 4] = 0.0
        fgpp[..., 5] = 0.0
        fgpp[..., 6] = 1.0
        # Gate 1: foreground coverage > 5%
        fgpp[..., 14] = 1.0  # 100% foreground
        return fgpp

    @staticmethod
    def _make_valid_camera_matrices(t: int = 17) -> np.ndarray:
        """Build a valid ``[T, 4, 4]`` SO(3) camera matrix."""
        cam = np.zeros((t, 4, 4), dtype=np.float32)
        for i in range(t):
            cam[i] = np.eye(4, dtype=np.float32)
        return cam

    def test_raw_fgpp_with_camera_matrices_passes_all_gates(
        self, tmp_path: object,
    ) -> None:
        """15-channel FGPP + valid SO(3) camera_matrices passes all 5 gates."""
        tmp = tmp_path
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.random.RandomState(77).randint(
            0, 256, (17, 32, 32, 3), dtype=np.uint8,
        )
        fgpp_cond = self._make_valid_fgpp(17, 32, 32)
        cam_mats = self._make_valid_camera_matrices(17)

        builder.add_sample(
            sample_id="fgpp_valid",
            video_frames=frames,
            caption="valid fgpp sample",
            fgpp_cond=fgpp_cond,
            camera_matrices=cam_mats,
            source_sha256="c" * 64,
        )

        assert builder.rejected_count == 0

        shards = builder.build_shards(tmp / "shards")  # type: ignore[operator]
        assert len(shards) >= 1
        assert shards[0].stat().st_size > 0

        samples = list(iter_feature_shards(shards))
        assert len(samples) == 1
        assert samples[0]["latent"].shape == (16, 5, 4, 4)

    def test_wan_7ch_control_passes_gates(
        self, tmp_path: object,
    ) -> None:
        """[7, T, H, W] Wan control tensor with valid depth/normals passes gates."""
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.random.RandomState(88).randint(
            0, 256, (17, 32, 32, 3), dtype=np.uint8,
        )
        # [7, T, H, W] — valid Wan control layout
        wan_cond = np.zeros((7, 17, 32, 32), dtype=np.float32)
        # ch0=depth with sufficient span
        wan_cond[0] = np.linspace(1.0, 10.0, 32 * 32).reshape(1, 32, 32)
        # ch1-3=unit normals [0, 0, 1]
        wan_cond[1] = 0.0
        wan_cond[2] = 0.0
        wan_cond[3] = 1.0

        builder.add_sample(
            sample_id="wan_valid",
            video_frames=frames,
            caption="wan control sample",
            fgpp_cond=wan_cond,
        )

        # Should NOT be rejected
        assert builder.rejected_count == 0

    def test_bad_rotation_det_rejects(self) -> None:
        """Camera matrix with non-SO(3) rotation is rejected by Gate 5."""
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        fgpp = TestFeatureStoreQualityGates._make_valid_fgpp(5, 16, 16)
        cam = np.zeros((5, 4, 4), dtype=np.float32)
        # Set a non-SO(3) rotation (det ≠ 1)
        cam[0] = np.diag([2.0, 1.0, 1.0, 1.0])

        builder.add_sample(
            sample_id="bad_rot",
            video_frames=frames,
            caption="bad rotation",
            fgpp_cond=fgpp,
            camera_matrices=cam,
        )

        assert builder.rejected_count == 1

    def test_low_foreground_coverage_rejects(self) -> None:
        """FGPP with < 5% foreground coverage is rejected by Gate 1."""
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        fgpp = TestFeatureStoreQualityGates._make_valid_fgpp(5, 16, 16)
        # Zero out Cryptomatte → 0% foreground
        fgpp[..., 14] = 0.0

        builder.add_sample(
            sample_id="no_fg",
            video_frames=frames,
            caption="no foreground",
            fgpp_cond=fgpp,
        )

        assert builder.rejected_count == 1

    def test_sample_id_path_traversal_rejected(self) -> None:
        """PR-99: sample_id with path traversal or invalid characters is rejected."""
        from synapse.errors import ContractValidationError

        builder = FeatureStoreBuilder(
            LinearProjectionEncoder(),
            LinearProjectionTextEncoder(),
        )
        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)

        with pytest.raises(ContractValidationError, match="path traversal"):
            builder.add_sample(
                sample_id="../traversal",
                video_frames=frames,
                caption="bad sample",
            )

        with pytest.raises(ContractValidationError, match="path traversal"):
            builder.add_sample(
                sample_id="sub/dir/id",
                video_frames=frames,
                caption="bad sample",
            )

    def test_sample_id_dotted_name_roundtrip(self, tmp_path: object) -> None:
        """PR-99: sample_id with dots like 'wan2.1_shot01' roundtrips without truncation."""
        tmp = tmp_path
        builder = FeatureStoreBuilder(
            LinearProjectionEncoder(),
            LinearProjectionTextEncoder(),
        )
        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        builder.add_sample(
            sample_id="wan2.1_shot01",
            video_frames=frames,
            caption="dotted id test",
        )
        shards = builder.build_shards(tmp / "shards")  # type: ignore[operator]
        samples = list(iter_feature_shards(shards))
        assert len(samples) == 1
        assert samples[0]["sample_id"] == "wan2.1_shot01"

    def test_missing_shard_raises_asset_not_found(self, tmp_path: object) -> None:
        """PR-99: iter_feature_shards raises AssetNotFoundError for missing shards."""
        with pytest.raises(AssetNotFoundError):
            list(iter_feature_shards([Path("/nonexistent_shard_path_404.tar")]))

    def test_zero_samples_build_shards_returns_empty(self, tmp_path: Path) -> None:
        """PR-99: build_shards with zero samples returns empty list without creating files."""
        builder = FeatureStoreBuilder(
            LinearProjectionEncoder(),
            LinearProjectionTextEncoder(),
        )
        shards = builder.build_shards(tmp_path / "empty_shards")
        assert shards == []
        assert list((tmp_path / "empty_shards").glob("*.tar")) == []



# ===================================================================
# PR-117: Wan 7ch validation gates, streaming iter, sample_id validation
# ===================================================================


class TestPR117FeatureStoreEnhancements:
    """PR-117: Wan 7ch quality gates, streaming iter, oversized member filtering."""

    @staticmethod
    def _make_valid_wan_7ch(t: int = 17, h: int = 32, w: int = 32) -> np.ndarray:
        """Build a valid [7, T, H, W] Wan control tensor.

        - ch0=depth with span > 0.1
        - ch1-3=normal with unit-length vectors
        """
        wan = np.zeros((7, t, h, w), dtype=np.float32)
        # Depth with sufficient span
        wan[0] = np.linspace(1.0, 10.0, h * w).reshape(1, h, w)
        # Unit normals pointing up: [0, 0, 1]
        wan[1] = 0.0  # nx
        wan[2] = 0.0  # ny
        wan[3] = 1.0  # nz
        return wan

    def test_wan_7ch_valid_passes_new_gates(self, tmp_path: object) -> None:
        """PR-117: Valid Wan 7ch tensor passes Gate 2w and Gate 4w."""
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.random.RandomState(117).randint(
            0, 256, (17, 32, 32, 3), dtype=np.uint8,
        )
        wan_cond = self._make_valid_wan_7ch()

        builder.add_sample(
            sample_id="wan_117",
            video_frames=frames,
            caption="wan 7ch valid",
            fgpp_cond=wan_cond,
        )

        assert builder.rejected_count == 0

    def test_wan_7ch_flat_depth_rejected(self) -> None:
        """PR-117: Gate 2w rejects Wan 7ch with flat depth (span < 0.1)."""
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        wan = self._make_valid_wan_7ch(5, 16, 16)
        wan[0] = 5.0  # flat depth → span = 0

        builder.add_sample(
            sample_id="flat_depth",
            video_frames=frames,
            caption="flat depth",
            fgpp_cond=wan,
        )

        assert builder.rejected_count == 1

    def test_wan_7ch_non_unit_normal_rejected(self) -> None:
        """PR-117: Gate 4w rejects Wan 7ch with non-unit normals."""
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        wan = self._make_valid_wan_7ch(5, 16, 16)
        # Set normals to non-unit length [2, 0, 0]
        wan[1] = 2.0
        wan[2] = 0.0
        wan[3] = 0.0

        builder.add_sample(
            sample_id="bad_normal",
            video_frames=frames,
            caption="bad normal",
            fgpp_cond=wan,
        )

        assert builder.rejected_count == 1

    def test_streaming_iter_validates_sample_id(self, tmp_path: object) -> None:
        """PR-117: iter_feature_shards filters out invalid sample_ids in streaming mode."""
        import io
        import tarfile

        tmp = tmp_path
        shard_path = Path(str(tmp)) / "test.tar"  # type: ignore[arg-type]

        # Create a tarball with one valid and one invalid sample_id
        with tarfile.open(shard_path, "w") as tf:
            # Valid sample
            for suffix, data in [
                ("latent.npy", np.zeros((16, 5, 4, 4), dtype=np.float32)),
                ("prompt.npy", np.zeros((512,), dtype=np.float32)),
            ]:
                buf = io.BytesIO()
                np.save(buf, data)
                raw = buf.getvalue()
                info = tarfile.TarInfo(name=f"valid_sample_01.{suffix}")
                info.size = len(raw)
                tf.addfile(info, io.BytesIO(raw))

            # Invalid sample_id with special chars
            for suffix, data in [
                ("latent.npy", np.zeros((16, 5, 4, 4), dtype=np.float32)),
                ("prompt.npy", np.zeros((512,), dtype=np.float32)),
            ]:
                buf = io.BytesIO()
                np.save(buf, data)
                raw = buf.getvalue()
                info = tarfile.TarInfo(name=f"bad!id@#.{suffix}")
                info.size = len(raw)
                tf.addfile(info, io.BytesIO(raw))

        samples = list(iter_feature_shards([shard_path]))
        # Only the valid sample should be yielded
        assert len(samples) == 1
        assert samples[0]["sample_id"] == "valid_sample_01"

    def test_streaming_iter_flushes_at_boundaries(self, tmp_path: object) -> None:
        """PR-117: streaming iter yields samples at sample_id boundaries."""
        import io
        import tarfile

        tmp = tmp_path
        shard_path = Path(str(tmp)) / "multi.tar"  # type: ignore[arg-type]

        with tarfile.open(shard_path, "w") as tf:
            for sid in ["alpha", "beta", "gamma"]:
                for suffix, data in [
                    ("latent.npy", np.zeros((16, 5, 4, 4), dtype=np.float32)),
                    ("prompt.npy", np.zeros((512,), dtype=np.float32)),
                ]:
                    buf = io.BytesIO()
                    np.save(buf, data)
                    raw = buf.getvalue()
                    info = tarfile.TarInfo(name=f"{sid}.{suffix}")
                    info.size = len(raw)
                    tf.addfile(info, io.BytesIO(raw))

        samples = list(iter_feature_shards([shard_path]))
        assert len(samples) == 3
        ids = {s["sample_id"] for s in samples}
        assert ids == {"alpha", "beta", "gamma"}
