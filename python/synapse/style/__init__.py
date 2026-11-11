"""synapse.style — Style extraction, descriptor & render alignment.

Public API:
    - ``StyleDescriptor``: Pydantic model of extracted visual style.
    - ``extract_style_optical``: Optical-only extraction from sRGB frames.
    - ``extract_style_fgpp_intrinsic``: FGPP-enriched extraction.
    - ``to_style_profile_dict``: Convert to StyleProfile v2.0.0 dict.
    - ``save_style_profile_yaml``: Serialise to YAML / JSON.
    - ``to_engine_light_overrides``: Engine light override params.
    - ACEScg colour science utilities in ``synapse.style.aces_color``.
"""

from synapse.style.descriptor import StyleDescriptor
from synapse.style.extractor import (
    extract_style_fgpp_intrinsic,
    extract_style_optical,
)
from synapse.style.profile_mapper import (
    save_style_profile_yaml,
    to_engine_light_overrides,
    to_style_profile_dict,
)

__all__ = [
    "StyleDescriptor",
    "extract_style_fgpp_intrinsic",
    "extract_style_optical",
    "save_style_profile_yaml",
    "to_engine_light_overrides",
    "to_style_profile_dict",
]
