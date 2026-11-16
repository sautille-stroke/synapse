"""Style descriptor data model (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StyleDescriptor(BaseModel):
    """Extracted visual style parameters.

    All colour values are assumed to be in the ``working_colorspace``
    (default: ACEScg).  Scalars are unitless unless noted.
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    working_colorspace: str = "ACEScg"
    color_temp_kelvin: float = Field(default=6500.0, ge=1000.0, le=40000.0)
    tint: float = Field(default=0.0, ge=-100.0, le=100.0)
    exposure_ev: float = Field(default=0.0, ge=-20.0, le=20.0)
    contrast: float = Field(default=1.0, ge=0.0, le=10.0)
    saturation: float = Field(default=1.0, ge=0.0, le=10.0)
    key_light_dir_world: list[float] = Field(
        default_factory=lambda: [0.0, 1.0, 0.0],
    )
    key_light_rgb: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
    )
    ambient_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    grain_sigma: float = Field(default=0.0, ge=0.0)
    vignette_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    tone_curve_lut_1d: list[float] | None = None
