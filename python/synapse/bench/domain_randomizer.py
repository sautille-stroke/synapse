"""11-factor parametric domain randomizer for training data diversity.

Generates ≥300 scene variant JSON descriptors by randomizing the same
11 rendering condition factors defined in ``conditions.py``:

    F01_viewpoint, F02_background, F03_lighting, F04_material,
    F05_geometry, F06_animation, F07_occlusion, F08_scale,
    F09_dof, F10_motion_blur, F11_postprocess

Each variant encodes randomised parameters suitable for driving an
offline renderer or CG scene generator, and a ``frame_count`` T that
satisfies the 3-D causal VAE constraint ``T = 4k + 1``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from synapse.bench.conditions import RENDERING_CONDITION_FACTORS

logger = logging.getLogger(__name__)

# Frame counts satisfying T = 4k + 1, capped to reasonable clip lengths.
_4K_PLUS_1: list[int] = [t for t in range(5, 98) if (t - 1) % 4 == 0]

# Canonical list of all 11 factor keys — sourced from conditions.py.
FACTOR_KEYS: list[str] = sorted(RENDERING_CONDITION_FACTORS.keys())

_DEFAULT_BASE_SCENES: list[dict[str, str]] = [
    {"name": "turntable_object", "category": "product"},
    {"name": "walk_cycle_character", "category": "character"},
    {"name": "interior_room", "category": "archviz"},
    {"name": "outdoor_terrain", "category": "environment"},
    {"name": "vehicle_reveal", "category": "automotive"},
]

# ── per-factor randomisation helpers ───────────────────────────────


def _rand_f01(rng: np.random.Generator) -> dict:
    """F01_viewpoint — camera placement."""
    return {
        "azimuth_deg": float(rng.uniform(0.0, 360.0)),
        "elevation_deg": float(rng.uniform(-30.0, 60.0)),
        "distance_m": float(rng.uniform(1.5, 8.0)),
    }


def _rand_f02(rng: np.random.Generator) -> dict:
    """F02_background — backdrop type."""
    bg_type = rng.choice(
        ["neutral_gray", "gradient", "hdri_indoor", "hdri_outdoor", "cluttered"],
    )
    return {
        "background_type": str(bg_type),
        "background_intensity": float(rng.uniform(0.1, 1.0)),
    }


def _rand_f03(rng: np.random.Generator) -> dict:
    """F03_lighting — HDR illumination."""
    num_lights = int(rng.integers(1, 4))  # 1-3
    return {
        "hdr_intensity": float(rng.uniform(0.2, 5.0)),
        "color_temperature_K": int(rng.integers(2700, 10001)),
        "num_point_lights": num_lights,
        "key_azimuth_deg": float(rng.uniform(0.0, 360.0)),
        "key_elevation_deg": float(rng.uniform(10.0, 80.0)),
    }


def _rand_f04(rng: np.random.Generator) -> dict:
    """F04_material — PBR material properties."""
    return {
        "roughness": float(rng.uniform(0.05, 0.95)),
        "metallic": float(rng.uniform(0.0, 1.0)),
        "ior": float(rng.uniform(1.0, 2.5)),
    }


def _rand_f05(rng: np.random.Generator) -> dict:
    """F05_geometry — mesh density & deformation."""
    density = rng.choice(["low", "mid", "high"])
    return {
        "mesh_density": str(density),
        "deformation_amplitude": float(rng.uniform(0.0, 0.1)),
    }


def _rand_f06(rng: np.random.Generator) -> dict:
    """F06_animation — motion clip parameters."""
    anim_type = rng.choice(["static", "rigid", "skeletal", "physics"])
    return {
        "animation_type": str(anim_type),
        "speed_multiplier": float(rng.uniform(0.5, 2.0)),
        "clip_offset_ratio": float(rng.uniform(0.0, 1.0)),
    }


def _rand_f07(rng: np.random.Generator) -> dict:
    """F07_occlusion — inter-object occlusion."""
    return {
        "occlusion_ratio": float(rng.uniform(0.0, 0.5)),
        "num_occluders": int(rng.integers(0, 4)),
    }


def _rand_f08(rng: np.random.Generator) -> dict:
    """F08_scale — object scale & camera distance."""
    return {
        "uniform_scale": float(rng.uniform(0.5, 2.0)),
        "camera_distance_multiplier": float(rng.uniform(0.5, 5.0)),
    }


def _rand_f09(rng: np.random.Generator) -> dict:
    """F09_dof — depth of field / aperture."""
    return {
        "aperture_fstop": float(rng.choice([0.0, 1.4, 2.8, 4.0, 8.0, 16.0])),
        "focus_distance_m": float(rng.uniform(0.5, 10.0)),
    }


def _rand_f10(rng: np.random.Generator) -> dict:
    """F10_motion_blur — shutter angle."""
    return {
        "shutter_angle_deg": float(rng.choice([0, 90, 180, 270, 360])),
        "motion_blur_samples": int(rng.integers(1, 17)),
    }


def _rand_f11(rng: np.random.Generator) -> dict:
    """F11_postprocess — tone-mapping & grading."""
    return {
        "exposure_ev": float(rng.uniform(0.5, 2.0)),
        "contrast": float(rng.uniform(0.8, 1.2)),
        "has_tonemapping": bool(rng.choice([True, False])),
        "film_grain_intensity": float(rng.uniform(0.0, 0.03)),
    }


_FACTOR_RANDOMIZERS: dict[str, object] = {
    "F01_viewpoint": _rand_f01,
    "F02_background": _rand_f02,
    "F03_lighting": _rand_f03,
    "F04_material": _rand_f04,
    "F05_geometry": _rand_f05,
    "F06_animation": _rand_f06,
    "F07_occlusion": _rand_f07,
    "F08_scale": _rand_f08,
    "F09_dof": _rand_f09,
    "F10_motion_blur": _rand_f10,
    "F11_postprocess": _rand_f11,
}


# ── public API ─────────────────────────────────────────────────────


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization.

    Attributes:
        num_variants: Minimum number of scene variants to generate.
        seed: Master random seed for reproducibility.
        frame_count_rule: Constraint on T; ``"4k+1"`` means T ∈ {5, 9, …, 97}.
        base_scenes: Optional list of base scene dicts; uses built-in
            defaults when empty.
    """

    num_variants: int = 300
    seed: int = 42
    frame_count_rule: str = "4k+1"
    base_scenes: list[dict[str, str]] = field(default_factory=list)


class DomainRandomizer:
    """11-factor parametric scene variant generator for training data diversity.

    Generates ``≥ config.num_variants`` scene variant JSON descriptors by
    randomising all 11 rendering condition factors defined in
    ``synapse.bench.conditions``.

    Args:
        config: Optional configuration; uses defaults when *None*.
    """

    FACTOR_KEYS: list[str] = FACTOR_KEYS

    def __init__(self, config: DomainRandomizationConfig | None = None) -> None:
        self._cfg = config or DomainRandomizationConfig()
        self._rng = np.random.default_rng(self._cfg.seed)
        self._base_scenes = (
            self._cfg.base_scenes if self._cfg.base_scenes else _DEFAULT_BASE_SCENES
        )

    # ── core generation ────────────────────────────────────────────

    def generate_variants(
        self,
        base_scenarios: list[dict] | None = None,
    ) -> list[dict]:
        """Generate ≥ ``num_variants`` scene variant JSON descriptors.

        Each variant dict contains:

        - ``"scene_id"``: unique identifier
        - ``"base_scene"``: source scenario name
        - ``"frame_count"``: T satisfying ``T = 4k + 1``
        - ``"factors"``: dict with all 11 F01–F11 keys
        - ``"seed"``: per-variant seed for reproducibility

        Args:
            base_scenarios: Optional override for the base scene list.

        Returns:
            A list of variant dicts (length ``≥ num_variants``).
        """
        scenes = base_scenarios if base_scenarios else self._base_scenes
        num = self._cfg.num_variants

        # Reset RNG to guarantee identical output for same seed.
        rng = np.random.default_rng(self._cfg.seed)

        variants: list[dict] = []
        for idx in range(num):
            base = scenes[idx % len(scenes)]
            base_name = base.get("name", f"scene_{idx}")

            # Per-variant seed derived deterministically.
            variant_seed = int(rng.integers(0, 2**31))
            variant_rng = np.random.default_rng(variant_seed)

            frame_count = int(variant_rng.choice(_4K_PLUS_1))

            factors: dict[str, dict] = {}
            for key in FACTOR_KEYS:
                randomizer = _FACTOR_RANDOMIZERS[key]
                factors[key] = randomizer(variant_rng)  # type: ignore[operator]

            variants.append(
                {
                    "scene_id": f"{base_name}_v{idx:04d}",
                    "base_scene": base_name,
                    "frame_count": frame_count,
                    "factors": factors,
                    "seed": variant_seed,
                }
            )

        logger.info(
            "Generated %d domain-randomised scene variants (seed=%d)",
            len(variants),
            self._cfg.seed,
        )
        return variants
