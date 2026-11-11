"""synapse.fgpp — Feature Guidance Package Protocol SDK.

Implements the FGPP channel specification (RFC 0044 §4.1).
Synapse is now the authoritative SSOT for the FGPP Python SDK.
"""

from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

__all__ = ["CameraParams", "FGPPFrame", "FGPPSequence"]
