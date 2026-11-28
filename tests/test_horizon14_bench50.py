"""Verification tests for Horizon 14 Synapse-Bench 50-scenario expansion."""

from __future__ import annotations

from synapse.bench.scenario import list_scenarios


class TestHorizon14Bench50Expansion:
    """Validate 50 built-in scenarios covering all Horizon 14 modalities."""

    def test_fifty_scenarios_registered_and_horizon14_tagged(self) -> None:
        scenarios = list_scenarios()
        assert len(scenarios) == 50, f"Expected 50 scenarios, got {len(scenarios)}"
        by_id = {s.id: s for s in scenarios}

        expected_h14_ids = {
            "t2_vace_r2v_identity_orbit",
            "t3_vace_v2v_occluded_stylize",
            "t3_vace_mv2v_inpaint_prop",
            "t4_diffrenderer_relight_studio",
            "t4_diffrenderer_dual_track_split",
            "t1_trellis2_open_surface_leaf",
            "t2_trellis2_pbr_translucent_veil",
            "t3_sam2_vos_zero_cryptomatte",
            "t5_sam2_vos_cut_reanchor_97f",
            "t5_spatia_corridor_200f_memory",
        }
        assert expected_h14_ids.issubset(by_id.keys())
        for sid in expected_h14_ids:
            assert "horizon14" in by_id[sid].tags
