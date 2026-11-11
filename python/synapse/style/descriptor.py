"""Style descriptor data model (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StyleDescriptor(BaseModel):
    """Extracted visual style parameters.

    All colour values are assumed to be in the ``working_colorspace``
    (default: ACEScg).  Scalars are unitless unless noted.
    """

    model_config = {"arbitrary_types_allowed": True}

    working_colorspace: str = "ACEScg"
    color_temp_kelvin: float = 6500.0
    tint: float = 0.0
    exposure_ev: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    key_light_dir_world: list[float] = Field(
        default_factory=lambda: [0.0, 1.0, 0.0],
    )
    key_light_rgb: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
    )
    ambient_ratio: float = 0.2
    grain_sigma: float = 0.0
    vignette_strength: float = 0.0
    tone_curve_lut_1d: list[float] | None = None
