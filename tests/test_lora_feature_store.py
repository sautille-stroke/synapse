"""Tests for PR-17 LoRA FeatureStore, encoders, and captions.

Covers:
* Structured caption formatting and validation
* LinearProjectionEncoder output shape and determinism
* LinearProjectionTextEncoder output shape and determinism
* FeatureStoreBuilder shard writing and reading
* lora_preencode stage registration
"""

from __future__ import annotations

import numpy as np
import pytest

from synapse.errors import ContractValidationError
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
        tmp = tmp_path  # type: ignore[assignment]
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

        shard_paths = builder.build_shards(tmp / "shards")
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
        tmp = tmp_path  # type: ignore[assignment]
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

        shard_paths = builder.build_shards(tmp / "shards")
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
        tmp = tmp_path  # type: ignore[assignment]
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

        shards = builder.build_shards(tmp / "shards")
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
        tmp = tmp_path  # type: ignore[assignment]
        enc = LinearProjectionEncoder()
        txt = LinearProjectionTextEncoder()
        builder = FeatureStoreBuilder(enc, txt)

        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        builder.add_sample(
            sample_id="no_cond",
            video_frames=frames,
            caption="no cond",
        )

        shards = builder.build_shards(tmp / "shards")
        samples = list(iter_feature_shards(shards))
        assert samples[0]["fgpp_cond"] is None

    def test_sha256_in_meta(
        self, tmp_path: object,
    ) -> None:
        """source_sha256 appears in meta."""
        tmp = tmp_path  # type: ignore[assignment]
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

        shards = builder.build_shards(tmp / "shards")
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
