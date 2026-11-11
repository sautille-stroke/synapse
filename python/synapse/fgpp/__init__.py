"""synapse.fgpp — Feature Guidance Package Protocol SDK.

Mirrors the nexus//fgpp SSOT (RFC 0044 §4.1) channel specification.
"""

from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

__all__ = ["CameraParams", "FGPPFrame", "FGPPSequence"]
