"""Unit tests for M58 VRAM-aware LoRA auto-tuning and Wan frame bucketing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from synapse.cli import main_synapse
from synapse.training import (
    FrameBucketKey,
    FrameBucketSample,
    FrameBucketSampler,
    LoRATrainConfig,
    auto_tune_lora_config,
    snap_to_frame_bucket,
)


class TestFrameBucketing:
    """Tests for snap_to_frame_bucket and FrameBucketSampler."""

    def test_snap_to_frame_bucket_wan_constraints(self) -> None:
        b1 = snap_to_frame_bucket(483, 851, 50)
        assert b1.height % 8 == 0 and b1.height == 480
        assert b1.width % 8 == 0 and b1.width == 848
        assert (b1.num_frames - 1) % 4 == 0 and b1.num_frames == 49

        b_exact = snap_to_frame_bucket(720, 1280, 97)
        assert b_exact == FrameBucketKey(height=720, width=1280, num_frames=97)

        b_over = snap_to_frame_bucket(512, 512, 160, max_frames=81)
        assert b_over.num_frames == 81

        b_min = snap_to_frame_bucket(4, 4, 2)
        assert b_min == FrameBucketKey(height=8, width=8, num_frames=1)

        with pytest.raises(ValueError):
            snap_to_frame_bucket(64, 64, 17, spatial_multiple=0)

    def test_sampler_groups_homogeneous_batches_and_interleaves(self) -> None:
        samples = [
            FrameBucketSample("s0", 480, 848, 49),
            FrameBucketSample("s1", 484, 850, 50),
            {"sample_id": "s2", "height": 720, "width": 1280, "num_frames": 81},
            {"sample_id": "s3", "height": 722, "width": 1284, "num_frames": 82},
            (512, 512, 16),
        ]
        sampler_a = FrameBucketSampler(samples, batch_size=2, seed=7, shuffle=True)
        sampler_b = FrameBucketSampler(samples, batch_size=2, seed=7, shuffle=True)
        assert len(sampler_a) == len(sampler_b) == 3
        assert [b.sample_ids for b in sampler_a] == [b.sample_ids for b in sampler_b]

        counts = sampler_a.bucket_counts()
        assert counts[FrameBucketKey(480, 848, 49)] == 2
        assert counts[FrameBucketKey(720, 1280, 81)] == 2
        assert counts[FrameBucketKey(512, 512, 13)] == 1

        # Epoch change updates order deterministically without breaking bucket purity
        sampler_a.set_epoch(3)
        for batch in sampler_a:
            assert batch.bucket.height % 8 == 0
            assert (batch.bucket.num_frames - 1) % 4 == 0
            assert 1 <= len(batch.indices) <= 2

        sampler_drop = FrameBucketSampler(samples, batch_size=2, drop_last=True)
        assert len(sampler_drop) == 2

    def test_sampler_oversized_batch_yields_single_partial_batch(self) -> None:
        """batch_size > bucket population should yield one partial batch per bucket."""
        samples = [
            FrameBucketSample("only0", 480, 848, 49),
            FrameBucketSample("only1", 720, 1280, 97),
        ]
        sampler = FrameBucketSampler(samples, batch_size=64, drop_last=False)
        batches = list(sampler)
        assert len(batches) == 2  # one per bucket
        for b in batches:
            assert len(b.sample_ids) == 1

    def test_sampler_oversized_batch_drop_last_yields_nothing(self) -> None:
        """batch_size > every bucket → drop_last=True should yield zero batches."""
        samples = [FrameBucketSample("x", 480, 848, 49)]
        sampler = FrameBucketSampler(samples, batch_size=4, drop_last=True)
        assert len(sampler) == 0
        assert list(sampler) == []

    def test_sampler_empty_input(self) -> None:
        """Empty sample list should produce zero batches without errors."""
        sampler = FrameBucketSampler([], batch_size=2)
        assert len(sampler) == 0


class TestAutoTuneLoRAConfig:
    """Tests for VRAM-aware LoRA auto-tuning & CLI exposure."""

    def test_auto_tune_tiers_enable_qlora_at_or_below_48gb(self) -> None:
        low = auto_tune_lora_config("identity", vram_gb=16.0, dataset_size=50)
        assert low.use_qlora is True
        assert low.quantization_bits == 4
        assert low.rank in (32, 64) and low.rank == 32
        assert low.gradient_checkpointing is True
        assert low.max_frames_bucket == 49
        assert 1 <= low.warmup_steps < low.max_steps

        mid = auto_tune_lora_config("style", vram_gb=40.0, dataset_size=120)
        assert mid.use_qlora is True
        assert mid.rank == 64
        assert mid.lr_spatial > mid.lr_temporal

        high = auto_tune_lora_config("motion", vram_gb=80.0, dataset_size=200)
        assert high.use_qlora is False
        assert high.quantization_bits == 16
        assert high.rank == 64
        assert high.max_frames_bucket == 97
        assert high.lr_temporal > high.lr_spatial

        cfg = high.to_lora_train_config(mask_aware=True, mask_background_weight=0.05)
        assert isinstance(cfg, LoRATrainConfig)
        assert cfg.rank == 64
        assert cfg.warmup_steps == high.warmup_steps

    def test_auto_tune_no_gpu_raises_without_explicit_vram(self) -> None:
        """When no CUDA device and no --vram-gb, auto_tune must raise ValueError."""
        from synapse.training.bucketing import detect_cuda_vram_gb

        detected = detect_cuda_vram_gb()
        if detected == 0.0:
            # CPU-only host: should raise because detect returns 0
            with pytest.raises(ValueError, match="vram_gb must be positive"):
                auto_tune_lora_config("identity", dataset_size=10)

    def test_cli_lora_train_help_and_autotune_flags(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as se:
            main_synapse(["lora", "train", "--help"])
        assert se.value.code == 0
        out = capsys.readouterr().out
        for flag in ("--auto-tune", "--vram-gb", "--mask-aware", "--use-qlora"):
            assert flag in out

        # When torch is available, verify full CLI --auto-tune --mock-backbone execution
        pytest.importorskip("torch")
        cfg_json = json.dumps({"category": "identity", "max_steps": 1})
        out_dir = tmp_path / "weights"
        rc = main_synapse(
            [
                "lora",
                "train",
                "--config",
                cfg_json,
                "--output",
                str(out_dir),
                "--mock-backbone",
                "--auto-tune",
                "--vram-gb",
                "24",
                "--mask-aware",
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"

    def test_cli_triple_flag_combo_argparse_no_torch(self) -> None:
        """--auto-tune + --mask-aware + --use-qlora must not conflict at argparse level."""
        with pytest.raises(SystemExit) as se:
            main_synapse([
                "lora", "train",
                "--auto-tune", "--mask-aware", "--use-qlora",
                "--vram-gb", "24",
                "--config", '{"category":"style","max_steps":1}',
                "--output", "/tmp/nonexistent",
                "--mock-backbone",
                "--help",
            ])
        # --help triggers SystemExit(0); if flags conflicted, argparse would exit(2)
        assert se.value.code == 0
