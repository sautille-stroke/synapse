"""Tests for PR-22: VisualJevCritic + MockJevTransport.

Covers:
- MockJevTransport logprob generation and connection errors
- VisualJevCritic calibrated probability math
- Temperature scaling effects
- Action routing by severity and category
- Full evaluate pipeline with InspectionReport assembly
- BackendUnavailableError on connection failure
- asset_inspect_3d stage registration
"""

from __future__ import annotations

import math

import pytest

from synapse.inspect.jev_critic import (
    MockJevTransport,
    VisualJevCritic,
)


# -------------------------------------------------------------------
# MockJevTransport tests
# -------------------------------------------------------------------


class TestMockJevTransport:
    """Tests for MockJevTransport."""

    def test_clean_default(self) -> None:
        """Default transport returns high No logprob."""
        transport = MockJevTransport()
        lp = transport.query_logprobs(
            ["img.png"],
            "Does this show mesh crumple? Answer Yes or No.",
        )
        assert "Yes" in lp
        assert "No" in lp
        # Default severity 0.05 → No should dominate
        assert lp["No"] > lp["Yes"]

    def test_defective_override(self) -> None:
        """With high severity override, Yes logprob dominates."""
        transport = MockJevTransport(
            defect_overrides={"mesh_crumple": 0.9},
        )
        lp = transport.query_logprobs(
            ["img.png"],
            "Does this show mesh crumple? Answer Yes or No.",
        )
        assert lp["Yes"] > lp["No"]

    def test_connection_error(self) -> None:
        """raise_on_connect=True raises ConnectionError."""
        transport = MockJevTransport(raise_on_connect=True)
        with pytest.raises(ConnectionError):
            transport.query_logprobs(["img.png"], "test")

    def test_category_extraction_with_spaces(self) -> None:
        """Category parsing handles underscore-to-space."""
        transport = MockJevTransport(
            defect_overrides={"anatomical_distortion": 0.8},
        )
        lp = transport.query_logprobs(
            ["img.png"],
            "Does this show anatomical distortion? "
            "Answer Yes or No.",
        )
        # Should match and return high Yes
        assert lp["Yes"] > lp["No"]

    def test_unknown_category_default(self) -> None:
        """Unrecognised prompt uses default severity 0.05."""
        transport = MockJevTransport()
        lp = transport.query_logprobs(
            ["img.png"],
            "Is there an unknown defect? Yes or No.",
        )
        assert lp["No"] > lp["Yes"]


# -------------------------------------------------------------------
# Calibrated probability tests
# -------------------------------------------------------------------


class TestCalibratedProbability:
    """Tests for temperature-scaled probability computation."""

    def test_equal_logits(self) -> None:
        """Equal logits produce probability 0.5."""
        critic = VisualJevCritic(
            MockJevTransport(), calibration_temperature=1.0,
        )
        p = critic.calibrated_probability(-1.0, -1.0)
        assert p == pytest.approx(0.5, abs=1e-10)

    def test_high_yes(self) -> None:
        """Much higher yes logit → probability close to 1."""
        critic = VisualJevCritic(
            MockJevTransport(), calibration_temperature=1.0,
        )
        p = critic.calibrated_probability(0.0, -10.0)
        assert p > 0.99

    def test_high_no(self) -> None:
        """Much higher no logit → probability close to 0."""
        critic = VisualJevCritic(
            MockJevTransport(), calibration_temperature=1.0,
        )
        p = critic.calibrated_probability(-10.0, 0.0)
        assert p < 0.01

    def test_temperature_scaling_flattens(self) -> None:
        """Higher temperature moves probability toward 0.5."""
        transport = MockJevTransport()
        critic_cold = VisualJevCritic(
            transport, calibration_temperature=0.5,
        )
        critic_hot = VisualJevCritic(
            transport, calibration_temperature=3.0,
        )

        z_yes, z_no = -0.5, -1.5
        p_cold = critic_cold.calibrated_probability(
            z_yes, z_no,
        )
        p_hot = critic_hot.calibrated_probability(
            z_yes, z_no,
        )

        # Cold temperature → more extreme (further from 0.5)
        # Hot temperature → closer to 0.5
        assert abs(p_cold - 0.5) > abs(p_hot - 0.5)

    def test_symmetry(self) -> None:
        """Swapping yes/no logits gives 1 - p."""
        critic = VisualJevCritic(
            MockJevTransport(), calibration_temperature=1.5,
        )
        p1 = critic.calibrated_probability(-0.3, -1.7)
        p2 = critic.calibrated_probability(-1.7, -0.3)
        assert p1 + p2 == pytest.approx(1.0, abs=1e-10)


# -------------------------------------------------------------------
# Action routing tests
# -------------------------------------------------------------------


class TestRouteAction:
    """Tests for severity-based action routing."""

    def _critic(self) -> VisualJevCritic:
        return VisualJevCritic(MockJevTransport())

    def test_accept_low_severity(self) -> None:
        """Low severity → accept."""
        c = self._critic()
        assert c._route_action("mesh_crumple", 0.1) == "accept"

    def test_auto_remesh_medium_crumple(self) -> None:
        """Medium mesh_crumple → auto_remesh."""
        c = self._critic()
        assert c._route_action(
            "mesh_crumple", 0.5,
        ) == "auto_remesh"

    def test_auto_remesh_floating_debris(self) -> None:
        """Medium floating_debris → auto_remesh."""
        c = self._critic()
        assert c._route_action(
            "floating_debris", 0.4,
        ) == "auto_remesh"

    def test_auto_inpaint_texture_stretch(self) -> None:
        """Medium texture_stretch → auto_inpaint_texture."""
        c = self._critic()
        assert c._route_action(
            "texture_stretch", 0.5,
        ) == "auto_inpaint_texture"

    def test_auto_inpaint_seam_tearing(self) -> None:
        """Medium seam_tearing → auto_inpaint_texture."""
        c = self._critic()
        assert c._route_action(
            "seam_tearing", 0.5,
        ) == "auto_inpaint_texture"

    def test_hitl_high_severity(self) -> None:
        """High severity of any type → manual_hitl."""
        c = self._critic()
        assert c._route_action(
            "mesh_crumple", 0.8,
        ) == "manual_hitl"

    def test_hitl_medium_anatomical(self) -> None:
        """Medium anatomical_distortion → manual_hitl."""
        c = self._critic()
        assert c._route_action(
            "anatomical_distortion", 0.5,
        ) == "manual_hitl"


# -------------------------------------------------------------------
# Full evaluation pipeline tests
# -------------------------------------------------------------------


class TestVisualJevCriticEvaluate:
    """Tests for VisualJevCritic.evaluate()."""

    def test_clean_asset(self) -> None:
        """Clean asset gets high quality score, no HITL."""
        transport = MockJevTransport()
        critic = VisualJevCritic(transport)
        report = critic.evaluate(
            "asset_001", "Clean Model", ["img.png"],
        )

        # Default severity ~0.05 calibrates to ~0.12 (< 0.3)
        # All defects should have "accept" action
        assert report.overall_quality_score > 0.8
        assert not report.requires_hitl
        assert report.schema_version == "2.0.0"
        assert report.asset_id == "asset_001"
        # All suggested actions should be "accept"
        for d in report.defects:
            assert d.suggested_action == "accept"

    def test_with_defects(self) -> None:
        """Multiple defects are detected and reported."""
        transport = MockJevTransport(
            defect_overrides={
                "mesh_crumple": 0.9,
                "texture_stretch": 0.85,
            },
        )
        critic = VisualJevCritic(transport)
        report = critic.evaluate(
            "asset_002", "Defective Model", ["img.png"],
        )

        categories = [d.category for d in report.defects]
        assert "mesh_crumple" in categories
        assert "texture_stretch" in categories
        # Both should have significant severity
        named_defects = {
            d.category: d for d in report.defects
        }
        assert named_defects["mesh_crumple"].severity > 0.5
        assert named_defects["texture_stretch"].severity > 0.5

    def test_anatomical_distortion_requires_hitl(self) -> None:
        """High anatomical distortion triggers HITL review."""
        transport = MockJevTransport(
            defect_overrides={
                "anatomical_distortion": 0.95,
            },
        )
        critic = VisualJevCritic(transport)
        report = critic.evaluate(
            "asset_003", "Distorted", ["img.png"],
        )

        assert report.requires_hitl is True
        anat = [
            d for d in report.defects
            if d.category == "anatomical_distortion"
        ]
        assert len(anat) == 1
        assert anat[0].severity > 0.8
        assert anat[0].suggested_action == "manual_hitl"

    def test_connection_failure_exit_30(self) -> None:
        """Transport failure raises BackendUnavailableError."""
        from synapse.errors import BackendUnavailableError

        transport = MockJevTransport(raise_on_connect=True)
        critic = VisualJevCritic(transport)

        with pytest.raises(BackendUnavailableError) as exc_info:
            critic.evaluate(
                "asset_err", "Fail", ["img.png"],
            )

        assert int(exc_info.value.exit_code) == 30

    def test_report_schema_version(self) -> None:
        """Report schema version should be 2.0.0."""
        transport = MockJevTransport()
        critic = VisualJevCritic(transport)
        report = critic.evaluate("a", "b", [])
        assert report.schema_version == "2.0.0"

    def test_defect_ids_unique(self) -> None:
        """All defect IDs should be unique."""
        transport = MockJevTransport(
            defect_overrides={
                "mesh_crumple": 0.5,
                "texture_stretch": 0.5,
                "seam_tearing": 0.5,
            },
        )
        critic = VisualJevCritic(transport)
        report = critic.evaluate("a", "b", ["img.png"])
        ids = [d.defect_id for d in report.defects]
        assert len(ids) == len(set(ids))

    def test_missing_logprob_tokens_raises_contract_error(
        self,
    ) -> None:
        """Missing Yes/No token logprobs raises ContractValidationError(10)."""
        from synapse.errors import ContractValidationError

        class EmptyTransport:
            def query_logprobs(
                self, image_uris: list[str], prompt: str,
            ) -> dict[str, float]:
                return {"Maybe": -0.5}

        critic = VisualJevCritic(EmptyTransport())
        with pytest.raises(ContractValidationError) as exc_info:
            critic.evaluate("a", "b", ["img.png"])
        assert int(exc_info.value.exit_code) == 10

    def test_whitespace_case_normalized_tokens(self) -> None:
        """Tokens like ' yes ' and 'NO' are normalized properly."""

        class WhitespaceTransport:
            def query_logprobs(
                self, image_uris: list[str], prompt: str,
            ) -> dict[str, float]:
                return {" yes ": -3.0, " NO\n": -0.05}

        critic = VisualJevCritic(WhitespaceTransport())
        report = critic.evaluate("a", "b", ["img.png"])
        assert report.overall_quality_score > 0.8


# -------------------------------------------------------------------
# Stage registration test
# -------------------------------------------------------------------


class TestStageRegistration:
    """Test asset_inspect_3d stage is registered."""

    def test_asset_inspect_3d_registered(self) -> None:
        """'asset_inspect_3d' in registered stages."""
        import synapse.inspect.jev_critic  # noqa: F401
        from synapse.pipeline.bundle_worker import (
            get_registered_stages,
        )

        stages = get_registered_stages()
        assert "asset_inspect_3d" in stages

    def test_vlm_endpoint_missing_transport_raises_30(
        self,
    ) -> None:
        """vlm_endpoint_url without vlm_transport raises BackendUnavailableError(30)."""
        from synapse.errors import BackendUnavailableError
        from synapse.inspect.jev_critic import (
            _handle_asset_inspect_3d,
        )
        from synapse.pipeline.bundle_worker import (
            BundleManifest,
        )

        manifest = BundleManifest(
            job_id="job_jev_01",
            stage="asset_inspect_3d",
            params={
                "asset_id": "a01",
                "vlm_endpoint_url": "http://127.0.0.1:8000/v1",
            },
        )
        with pytest.raises(BackendUnavailableError) as exc_info:
            _handle_asset_inspect_3d(manifest)
        assert int(exc_info.value.exit_code) == 30
