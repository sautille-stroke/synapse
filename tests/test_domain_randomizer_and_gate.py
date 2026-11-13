"""Tests for DomainRandomizer and FeatureStoreBuilder ingestion gates."""

from __future__ import annotations

import numpy as np
from synapse.bench.conditions import RENDERING_CONDITION_FACTORS
from synapse.bench.domain_randomizer import (
    FACTOR_KEYS,
    DomainRandomizationConfig,
    DomainRandomizer,
)
from synapse.training.feature_store import (
    FeatureStoreBuilder,
    LinearProjectionEncoder,
    LinearProjectionTextEncoder,
)

# ── DomainRandomizer tests ────────────────────────────────────────


class TestDomainRandomizer:
    """Tests for the 11-factor domain randomizer."""

    def test_generate_at_least_300_variants(self) -> None:
        """Default config must produce ≥300 variants."""
        dr = DomainRandomizer()
        variants = dr.generate_variants()
        assert len(variants) >= 300

    def test_all_factor_keys_present(self) -> None:
        """Every variant must contain all 11 F01–F11 factor keys."""
        dr = DomainRandomizer(DomainRandomizationConfig(num_variants=10, seed=99))
        variants = dr.generate_variants()
        expected = set(RENDERING_CONDITION_FACTORS.keys())
        for v in variants:
            assert set(v["factors"].keys()) == expected, (
                f"Missing factors in {v['scene_id']}"
            )

    def test_frame_count_is_4k_plus_1(self) -> None:
        """Every variant's frame_count must satisfy (T - 1) % 4 == 0."""
        dr = DomainRandomizer(DomainRandomizationConfig(num_variants=50, seed=7))
        for v in dr.generate_variants():
            t = v["frame_count"]
            assert (t - 1) % 4 == 0, f"T={t} does not satisfy 4k+1"
            assert t >= 5

    def test_deterministic_with_seed(self) -> None:
        """Two runs with the same seed must produce identical variants."""
        cfg = DomainRandomizationConfig(num_variants=20, seed=123)
        v1 = DomainRandomizer(cfg).generate_variants()
        v2 = DomainRandomizer(cfg).generate_variants()
        assert v1 == v2

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different variants."""
        v1 = DomainRandomizer(
            DomainRandomizationConfig(num_variants=5, seed=1),
        ).generate_variants()
        v2 = DomainRandomizer(
            DomainRandomizationConfig(num_variants=5, seed=2),
        ).generate_variants()
        # At least one variant must differ.
        assert v1 != v2

    def test_factor_keys_match_conditions_module(self) -> None:
        """FACTOR_KEYS must match conditions.RENDERING_CONDITION_FACTORS."""
        assert FACTOR_KEYS == sorted(RENDERING_CONDITION_FACTORS.keys())

    def test_custom_base_scenarios(self) -> None:
        """Custom base scenarios are reflected in variants."""
        cfg = DomainRandomizationConfig(num_variants=3, seed=0)
        bases = [{"name": "my_custom_scene"}]
        variants = DomainRandomizer(cfg).generate_variants(base_scenarios=bases)
        for v in variants:
            assert v["base_scene"] == "my_custom_scene"


# ── FeatureStoreBuilder ingestion gate tests ───────────────────────


def _make_builder() -> FeatureStoreBuilder:
    return FeatureStoreBuilder(
        latent_encoder=LinearProjectionEncoder(),
        text_encoder=LinearProjectionTextEncoder(),
    )


def _good_fgpp(t: int = 5, h: int = 16, w: int = 16) -> np.ndarray:
    """Create a clean FGPP tensor that passes all 5 gates.

    Uses 16 channels: depth(1) + radiance(3) + normal(3) + rotation(9).
    Gate 1 checks Cryptomatte at the last channel; here we set channel
    index 15 as foreground mask and gate code checks index 14 only when
    shape[-1] >= 15, so we allocate 16 channels to also cover gate 5
    which reads indices 7-15 (9 floats for 3×3 rotation).
    """
    n_ch = 16
    fgpp = np.zeros((t, h, w, n_ch), dtype=np.float32)

    # Depth channel (index 0) — valid range with span > 0.1 m
    fgpp[..., 0] = np.linspace(1.0, 5.0, h * w).reshape(1, h, w)

    # DirectRadiance (indices 1-3) — within half-float range
    fgpp[..., 1:4] = 10.0

    # WorldNormal (indices 4-6) — unit-length normals
    fgpp[..., 4] = 0.0
    fgpp[..., 5] = 0.0
    fgpp[..., 6] = 1.0

    # Camera rotation (indices 7-15) — identity 3×3 → det = 1
    eye = np.eye(3, dtype=np.float32).flatten()
    fgpp[..., 7:16] = eye

    # Cryptomatte (index 14 in the 15-channel spec is covered by
    # rotation col 8 above; for gate-1 we rely on index 14 being > 0
    # which the identity matrix sets R[2,1] = 0 and R[2,2] = 1 —
    # index 14 = eye[7] = 0.  Override it explicitly for foreground.)
    fgpp[..., 14] = 1.0

    return fgpp


def _good_frames(t: int = 5, h: int = 16, w: int = 16) -> np.ndarray:
    return np.random.RandomState(42).randint(
        0, 256, (t, h, w, 3), dtype=np.uint8,
    )


class TestIngestionGates:
    """Tests for FeatureStoreBuilder._validate_ingestion_quality."""

    def test_ingestion_gate_accepts_good_sample(self) -> None:
        """A clean sample must be accepted."""
        builder = _make_builder()
        passed, reason = builder._validate_ingestion_quality(
            {
                "video_frames": _good_frames(),
                "fgpp_cond": _good_fgpp(),
            }
        )
        assert passed, f"Unexpected rejection: {reason}"

    def test_ingestion_gate_rejects_low_foreground(self) -> None:
        """Sample with <5% foreground coverage must be rejected."""
        fgpp = _good_fgpp()
        # Zero out Cryptomatte channel → 0% foreground.
        fgpp[..., 14] = 0.0
        builder = _make_builder()
        passed, reason = builder._validate_ingestion_quality(
            {"video_frames": _good_frames(), "fgpp_cond": fgpp},
        )
        assert not passed
        assert "Gate 1" in reason

    def test_ingestion_gate_rejects_bad_depth(self) -> None:
        """Sample with depth span < 0.1 m must be rejected."""
        fgpp = _good_fgpp()
        fgpp[..., 0] = 2.0  # constant depth → span = 0
        builder = _make_builder()
        passed, reason = builder._validate_ingestion_quality(
            {"video_frames": _good_frames(), "fgpp_cond": fgpp},
        )
        assert not passed
        assert "Gate 2" in reason

    def test_ingestion_gate_rejects_overflow_radiance(self) -> None:
        """Sample with 95th-percentile radiance > 65504 must be rejected."""
        fgpp = _good_fgpp()
        fgpp[..., 1:4] = 70000.0  # above half-float max
        builder = _make_builder()
        passed, reason = builder._validate_ingestion_quality(
            {"video_frames": _good_frames(), "fgpp_cond": fgpp},
        )
        assert not passed
        assert "Gate 3" in reason

    def test_ingestion_gate_rejects_bad_normals(self) -> None:
        """Sample with <95% unit-length normals must be rejected."""
        fgpp = _good_fgpp()
        # Set normals to non-unit-length everywhere.
        fgpp[..., 4] = 5.0
        fgpp[..., 5] = 5.0
        fgpp[..., 6] = 5.0
        builder = _make_builder()
        passed, reason = builder._validate_ingestion_quality(
            {"video_frames": _good_frames(), "fgpp_cond": fgpp},
        )
        assert not passed
        assert "Gate 4" in reason

    def test_ingestion_gate_rejects_bad_rotation(self) -> None:
        """Sample with det(R) ≠ 1 must be rejected."""
        fgpp = _good_fgpp()
        # Set rotation to a matrix with det != 1.
        bad_rot = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ], dtype=np.float32).flatten()
        fgpp[..., 7:16] = bad_rot
        # Restore Cryptomatte at index 14 (overlaps rotation indices)
        # so that Gate 1 does not fire before Gate 5.
        fgpp[..., 14] = 1.0
        builder = _make_builder()
        passed, reason = builder._validate_ingestion_quality(
            {"video_frames": _good_frames(), "fgpp_cond": fgpp},
        )
        assert not passed
        assert "Gate 5" in reason

    def test_add_sample_increments_rejected_counter(self) -> None:
        """Rejected samples must increment rejected_count and not be buffered."""
        builder = _make_builder()
        fgpp = _good_fgpp()
        fgpp[..., 14] = 0.0  # will fail Gate 1

        builder.add_sample(
            "bad_sample",
            _good_frames(),
            "a caption",
            fgpp_cond=fgpp,
        )
        assert builder.rejected_count == 1
        assert len(builder._samples) == 0

    def test_add_sample_accepts_without_fgpp(self) -> None:
        """Samples without FGPP cond should pass (gates are no-ops)."""
        builder = _make_builder()
        builder.add_sample(
            "no_fgpp",
            _good_frames(),
            "a caption",
            fgpp_cond=None,
        )
        assert builder.rejected_count == 0
        assert len(builder._samples) == 1
