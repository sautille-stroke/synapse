"""Map StyleDescriptor to engine-compatible formats.

Provides conversions to StyleProfile v2.0.0 dict format, YAML/JSON
serialisation, and engine light override parameter generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synapse.style.descriptor import StyleDescriptor


def to_style_profile_dict(desc: StyleDescriptor) -> dict[str, Any]:
    """Convert a StyleDescriptor to StyleProfile v2.0.0 dict format.

    Args:
        desc: Extracted style descriptor.

    Returns:
        Nested dict conforming to ``style_profile.v2.0.0`` schema.
    """
    return {
        "schema_version": "style_profile.v2.0.0",
        "metadata": {
            "working_colorspace": desc.working_colorspace,
        },
        "shading_config": {
            "key_light_direction": desc.key_light_dir_world,
            "key_light_color": desc.key_light_rgb,
            "ambient_ratio": desc.ambient_ratio,
        },
        "post_process_config": {
            "color_temp_kelvin": desc.color_temp_kelvin,
            "tint": desc.tint,
            "exposure_ev": desc.exposure_ev,
            "contrast": desc.contrast,
            "saturation": desc.saturation,
            "grain_sigma": desc.grain_sigma,
            "vignette_strength": desc.vignette_strength,
            "tone_curve_lut_1d": desc.tone_curve_lut_1d,
        },
    }


def save_style_profile_yaml(
    desc: StyleDescriptor,
    path: str | Path,
) -> Path:
    """Save a StyleDescriptor as a YAML file (JSON fallback if no pyyaml).

    Args:
        desc: Extracted style descriptor.
        path: Output file path (suffix overridden to ``.json`` on fallback).

    Returns:
        Path to the written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = to_style_profile_dict(desc)

    try:
        import yaml  # type: ignore[import-untyped]
        p.write_text(yaml.dump(d, default_flow_style=False), encoding="utf-8")
    except ImportError:
        p = p.with_suffix(".json")
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")

    return p


def to_engine_light_overrides(desc: StyleDescriptor) -> dict[str, Any]:
    """Generate engine-compatible light override parameters.

    Args:
        desc: Extracted style descriptor.

    Returns:
        Dict with ``directional_light``, ``ambient_light``, and
        ``exposure`` keys suitable for engine scene override injection.
    """
    return {
        "directional_light": {
            "direction": desc.key_light_dir_world,
            "color": desc.key_light_rgb,
            "intensity": 1.0 - desc.ambient_ratio,
        },
        "ambient_light": {
            "color": [1.0, 1.0, 1.0],
            "intensity": desc.ambient_ratio,
        },
        "exposure": {
            "ev_compensation": desc.exposure_ev,
        },
    }
