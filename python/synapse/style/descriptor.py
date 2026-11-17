"""Style descriptor data model (Pydantic v2)."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


def _validate_3_element_finite_nonneg(v: list[float], field_name: str) -> list[float]:
    """Validate that *v* has exactly 3 finite non-negative elements."""
    if len(v) != 3:
        raise ValueError(f"{field_name} must have exactly 3 elements, got {len(v)}")
    for i, x in enumerate(v):
        if not math.isfinite(x):
            raise ValueError(f"{field_name}[{i}] must be finite, got {x}")
        if x < 0.0:
            raise ValueError(f"{field_name}[{i}] must be non-negative, got {x}")
    return v


def _validate_3_element_finite(v: list[float], field_name: str) -> list[float]:
    """Validate that *v* has exactly 3 finite elements (may be negative)."""
    if len(v) != 3:
        raise ValueError(f"{field_name} must have exactly 3 elements, got {len(v)}")
    for i, x in enumerate(v):
        if not math.isfinite(x):
            raise ValueError(f"{field_name}[{i}] must be finite, got {x}")
    return v


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

    @field_validator("key_light_dir_world")
    @classmethod
    def _validate_light_dir(cls, v: list[float]) -> list[float]:
        return _validate_3_element_finite(v, "key_light_dir_world")

    @field_validator("key_light_rgb")
    @classmethod
    def _validate_light_rgb(cls, v: list[float]) -> list[float]:
        return _validate_3_element_finite_nonneg(v, "key_light_rgb")
