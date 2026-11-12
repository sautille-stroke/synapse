"""Rendering condition factor system for Synapse-Bench.

Encodes the 11-dimensional × 4-level orthogonal factor space that
controls how GT scenes are rendered.  This allows systematic evaluation
of how each rendering condition (viewpoint, lighting, material, …)
affects forward-generation and inverse-reconstruction quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ConditionLevel(str, Enum):
    """Severity level for a rendering condition factor."""

    L1 = "L1"  # Baseline / easiest
    L2 = "L2"  # Moderate
    L3 = "L3"  # Challenging
    L4 = "L4"  # Extreme / stress-test


_VALID_LEVELS = frozenset({"L1", "L2", "L3", "L4"})

RENDERING_CONDITION_FACTORS: dict[str, dict[str, Any]] = {
    "F01_viewpoint": {
        "name": "Viewpoint Coverage",
        "levels": {
            "L1": {
                "label": "full_orbit_360",
                "description": "Full 360° orbit, 15° elevation, 3° azimuth step",
                "orbit_range_deg": 360,
                "elevation_deg": 15,
                "azimuth_step_deg": 3,
            },
            "L2": {
                "label": "partial_orbit_180",
                "description": "Front hemisphere 180°, 5° step",
                "orbit_range_deg": 180,
                "elevation_deg": 15,
                "azimuth_step_deg": 5,
            },
            "L3": {
                "label": "sparse_8view",
                "description": "8 evenly-spaced views, 45° apart",
                "orbit_range_deg": 360,
                "elevation_deg": 15,
                "azimuth_step_deg": 45,
            },
            "L4": {
                "label": "sparse_3view",
                "description": "3 views at 0°/120°/240°",
                "orbit_range_deg": 360,
                "elevation_deg": 15,
                "azimuth_step_deg": 120,
            },
        },
    },
    "F02_background": {
        "name": "Background",
        "levels": {
            "L1": {
                "label": "neutral_gray",
                "description": "Uniform 18% gray background",
                "background_color": [0.18, 0.18, 0.18],
            },
            "L2": {
                "label": "gradient",
                "description": "Smooth gradient (studio backdrop)",
                "background_type": "gradient",
            },
            "L3": {
                "label": "hdri_indoor",
                "description": "Indoor HDRI environment map",
                "background_type": "hdri",
                "hdri_category": "indoor",
            },
            "L4": {
                "label": "cluttered_scene",
                "description": "Cluttered background with distractors",
                "background_type": "scene",
                "num_distractors": 5,
            },
        },
    },
    "F03_lighting": {
        "name": "Lighting",
        "levels": {
            "L1": {
                "label": "single_directional",
                "description": "Single directional key light, 45° azimuth, 30° elevation",
                "num_lights": 1,
                "light_type": "directional",
                "key_azimuth_deg": 45,
                "key_elevation_deg": 30,
            },
            "L2": {
                "label": "three_point",
                "description": "Classic three-point (key + fill + rim)",
                "num_lights": 3,
                "light_type": "three_point",
            },
            "L3": {
                "label": "multi_area",
                "description": "4+ area lights with colour temperature variation",
                "num_lights": 4,
                "light_type": "area",
                "color_temperature_range": [3200, 6500],
            },
            "L4": {
                "label": "complex_env_emissive",
                "description": "HDRI env + emissive meshes + coloured spot lights",
                "num_lights": 6,
                "light_type": "mixed",
                "has_emissive": True,
            },
        },
    },
    "F04_material": {
        "name": "Material Complexity",
        "levels": {
            "L1": {
                "label": "diffuse_lambert",
                "description": "Pure Lambertian diffuse (roughness=1, metallic=0)",
                "roughness": 1.0,
                "metallic": 0.0,
            },
            "L2": {
                "label": "roughness_metal",
                "description": "Varying roughness + metallic (PBR baseline)",
                "roughness_range": [0.1, 0.9],
                "metallic_range": [0.0, 1.0],
            },
            "L3": {
                "label": "textured_pbr",
                "description": "Full PBR textures (albedo + normal + roughness + metallic)",
                "has_textures": True,
            },
            "L4": {
                "label": "multilayer_coat_sss",
                "description": "Multi-layer (clear coat + SSS + anisotropy)",
                "has_coat": True,
                "has_sss": True,
                "has_anisotropy": True,
            },
        },
    },
    "F05_geometry": {
        "name": "Geometric Complexity",
        "levels": {
            "L1": {
                "label": "primitive",
                "description": "Geometric primitive (cube, sphere, cylinder)",
                "vertex_count_range": [8, 500],
            },
            "L2": {
                "label": "low_poly",
                "description": "Low-poly model (< 10K triangles)",
                "vertex_count_range": [500, 10000],
            },
            "L3": {
                "label": "mid_poly",
                "description": "Mid-poly model (10K-100K triangles)",
                "vertex_count_range": [10000, 100000],
            },
            "L4": {
                "label": "high_poly_scan",
                "description": "High-poly scan / CAD (> 100K triangles)",
                "vertex_count_range": [100000, 1000000],
            },
        },
    },
    "F06_animation": {
        "name": "Animation Type",
        "levels": {
            "L1": {
                "label": "static",
                "description": "No animation, static pose",
                "has_animation": False,
            },
            "L2": {
                "label": "rigid_transform",
                "description": "Rigid-body rotation/translation",
                "animation_type": "rigid",
            },
            "L3": {
                "label": "skeletal",
                "description": "Skeletal animation (walk, run, idle)",
                "animation_type": "skeletal",
                "skeleton_bones": 53,
            },
            "L4": {
                "label": "physics_sim",
                "description": "Physics simulation (cloth, ragdoll, fluid)",
                "animation_type": "physics",
            },
        },
    },
    "F07_occlusion": {
        "name": "Occlusion",
        "levels": {
            "L1": {
                "label": "none",
                "description": "No occlusion, object fully visible",
                "occlusion_ratio": 0.0,
            },
            "L2": {
                "label": "partial_self",
                "description": "Self-occlusion from viewing angle",
                "occlusion_ratio": 0.15,
            },
            "L3": {
                "label": "partial_other",
                "description": "Partially occluded by other objects",
                "occlusion_ratio": 0.30,
            },
            "L4": {
                "label": "heavy_mutual",
                "description": "Heavy mutual occlusion (> 50%)",
                "occlusion_ratio": 0.50,
            },
        },
    },
    "F08_scale": {
        "name": "Scale & Distance",
        "levels": {
            "L1": {
                "label": "normal",
                "description": "Object fills ~60% of frame",
                "frame_fill_ratio": 0.6,
                "camera_distance_multiplier": 1.0,
            },
            "L2": {
                "label": "medium_far",
                "description": "Object fills ~30% of frame",
                "frame_fill_ratio": 0.3,
                "camera_distance_multiplier": 2.0,
            },
            "L3": {
                "label": "close_up",
                "description": "Extreme close-up, object exceeds frame",
                "frame_fill_ratio": 1.5,
                "camera_distance_multiplier": 0.5,
            },
            "L4": {
                "label": "extreme_far",
                "description": "Object fills < 10% of frame",
                "frame_fill_ratio": 0.08,
                "camera_distance_multiplier": 5.0,
            },
        },
    },
    "F09_dof": {
        "name": "Depth of Field",
        "levels": {
            "L1": {
                "label": "all_sharp",
                "description": "Pinhole camera, everything in focus",
                "aperture": 0.0,
            },
            "L2": {
                "label": "mild_f8",
                "description": "f/8 aperture, subtle background blur",
                "aperture": 8.0,
            },
            "L3": {
                "label": "moderate_f2_8",
                "description": "f/2.8 aperture, noticeable bokeh",
                "aperture": 2.8,
            },
            "L4": {
                "label": "shallow_f1_4",
                "description": "f/1.4 aperture, extreme shallow DOF",
                "aperture": 1.4,
            },
        },
    },
    "F10_motion_blur": {
        "name": "Motion Blur",
        "levels": {
            "L1": {
                "label": "none",
                "description": "No motion blur, instant shutter",
                "shutter_angle_deg": 0,
            },
            "L2": {
                "label": "mild_90",
                "description": "90° shutter angle (subtle)",
                "shutter_angle_deg": 90,
            },
            "L3": {
                "label": "standard_180",
                "description": "180° shutter angle (cinematic standard)",
                "shutter_angle_deg": 180,
            },
            "L4": {
                "label": "heavy_360",
                "description": "360° shutter angle (heavy blur)",
                "shutter_angle_deg": 360,
            },
        },
    },
    "F11_postprocess": {
        "name": "Post-Processing",
        "levels": {
            "L1": {
                "label": "none",
                "description": "No post-processing, raw linear output",
                "has_tonemapping": False,
                "has_grain": False,
            },
            "L2": {
                "label": "basic_tonemap",
                "description": "ACES tonemapping + sRGB gamma only",
                "has_tonemapping": True,
                "tonemapper": "aces",
            },
            "L3": {
                "label": "color_graded",
                "description": "Tonemapping + LUT color grade + mild vignette",
                "has_tonemapping": True,
                "has_lut": True,
                "has_vignette": True,
            },
            "L4": {
                "label": "heavy_stylized",
                "description": "Heavy grade + film grain + chromatic aberration + bloom",
                "has_tonemapping": True,
                "has_lut": True,
                "has_vignette": True,
                "has_grain": True,
                "has_chromatic_aberration": True,
                "has_bloom": True,
            },
        },
    },
}

_VALID_FACTOR_KEYS = frozenset(RENDERING_CONDITION_FACTORS.keys())


@dataclass
class Conditions:
    """Rendering condition configuration for a scenario."""

    F01_viewpoint: ConditionLevel | None = None
    F02_background: ConditionLevel | None = None
    F03_lighting: ConditionLevel | None = None
    F04_material: ConditionLevel | None = None
    F05_geometry: ConditionLevel | None = None
    F06_animation: ConditionLevel | None = None
    F07_occlusion: ConditionLevel | None = None
    F08_scale: ConditionLevel | None = None
    F09_dof: ConditionLevel | None = None
    F10_motion_blur: ConditionLevel | None = None
    F11_postprocess: ConditionLevel | None = None

    def to_dict(self) -> dict[str, str]:
        """Return non-None factors as ``{factor_id: level}``."""
        result: dict[str, str] = {}
        for f in self.__dataclass_fields__:
            v = getattr(self, f)
            if v is not None:
                result[f] = v.value
        return result

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> Conditions:
        """Construct from ``{factor_id: level_str}``.

        Raises
        ------
        ValueError
            If *d* contains an unknown factor key or a level value
            outside ``{L1, L2, L3, L4}``.
        """
        kwargs: dict[str, ConditionLevel] = {}
        for k, v in d.items():
            if k not in _VALID_FACTOR_KEYS:
                raise ValueError(
                    f"Unknown condition factor key '{k}'. "
                    f"Valid keys: {sorted(_VALID_FACTOR_KEYS)}"
                )
            if v not in _VALID_LEVELS:
                raise ValueError(
                    f"Illegal condition level '{v}' for factor '{k}'. "
                    f"Must be one of {sorted(_VALID_LEVELS)}"
                )
            kwargs[k] = ConditionLevel(v)
        return cls(**kwargs)

    def resolve(self) -> dict[str, dict]:
        """Expand to full factor parameter dicts."""
        resolved: dict[str, dict] = {}
        for factor_id, factor_def in RENDERING_CONDITION_FACTORS.items():
            field_name = factor_id
            level = getattr(self, field_name, None) or ConditionLevel.L1
            level_data = factor_def["levels"].get(level.value, {})
            resolved[factor_id] = {
                "name": factor_def["name"],
                "level": level.value,
                **level_data,
            }
        return resolved

    def get_scale_multiplier(self) -> float:
        """Return the camera distance multiplier from F08_scale."""
        if self.F08_scale is None:
            return 1.0
        level_data = RENDERING_CONDITION_FACTORS["F08_scale"]["levels"].get(
            self.F08_scale.value, {}
        )
        return float(level_data.get("camera_distance_multiplier", 1.0))

    def get_aperture_fstop(self) -> float:
        """Return the aperture f-stop from F09_dof."""
        if self.F09_dof is None:
            return 0.0
        level_data = RENDERING_CONDITION_FACTORS["F09_dof"]["levels"].get(
            self.F09_dof.value, {}
        )
        return float(level_data.get("aperture", 0.0))

    def get_shutter_angle_deg(self) -> float:
        """Return the shutter angle in degrees from F10_motion_blur."""
        if self.F10_motion_blur is None:
            return 180.0
        level_data = RENDERING_CONDITION_FACTORS["F10_motion_blur"]["levels"].get(
            self.F10_motion_blur.value, {}
        )
        return float(level_data.get("shutter_angle_deg", 180.0))


SINGLE_FACTOR_CONDITIONS: list[dict[str, str]] = []
for _fid in RENDERING_CONDITION_FACTORS:
    for _level in ["L2", "L3", "L4"]:
        SINGLE_FACTOR_CONDITIONS.append({_fid: _level})

CROSS_CONDITIONS: list[dict[str, str]] = [
    {"F01_viewpoint": "L3", "F03_lighting": "L3"},
    {"F04_material": "L3", "F08_scale": "L3"},
    {"F06_animation": "L3", "F10_motion_blur": "L3"},
    {"F02_background": "L3", "F07_occlusion": "L3"},
    {"F09_dof": "L3", "F11_postprocess": "L3"},
    {"F01_viewpoint": "L4", "F03_lighting": "L4", "F08_scale": "L3"},
    {"F06_animation": "L3", "F04_material": "L3", "F10_motion_blur": "L2"},
    {"F05_geometry": "L3", "F07_occlusion": "L3", "F02_background": "L4"},
]
