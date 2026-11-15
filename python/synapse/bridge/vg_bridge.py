# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Thin bridge to geometry and material processing backends.

This module provides access to geometry and material processing backends.
The :class:`MockVgBackend` is always available for testing; production routes to
:class:`~synapse.recon.backend.SynapseReconBackend` which is self-contained with
native C++20 geometry operators and Python Oracle.

Abstract interface :class:`VgBackend` and :class:`MockVgBackend` are hosted in
:mod:`synapse.render_core.backend` to preserve a clean, unidirectional import DAG.
"""

from __future__ import annotations

import os

from synapse.errors import ContractValidationError
from synapse.fgpp.schema import (
    MIN_CHARACTER_WHITEBOX_VERTICES,
    validate_whitebox_coarseness_floor,
)
from synapse.render_core.backend import MockVgBackend, VgBackend

# NOTE: ``SynapseReconBackend`` and ``get_recon_backend`` live in
# ``synapse.recon.backend`` (Tier 2).  bridge is Tier 0 and must NOT
# import Tier 2 at module scope.  They are lazy-imported inside
# ``get_vg_backend()`` and re-exported via ``__getattr__`` below.

__all__ = [
    "MockVgBackend",
    "SynapseReconBackend",  # noqa: F822 — lazy via __getattr__
    "VgBackend",
    "enforce_distillation_geometry_gate",
    "get_recon_backend",  # noqa: F822 — lazy via __getattr__
    "get_vg_backend",
]


def __getattr__(name: str) -> object:
    """Lazy re-export of Tier-2 symbols to keep bridge at Tier 0."""
    if name in ("SynapseReconBackend", "get_recon_backend"):
        from synapse.recon.backend import SynapseReconBackend, get_recon_backend

        globals()["SynapseReconBackend"] = SynapseReconBackend
        globals()["get_recon_backend"] = get_recon_backend
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_vg_backend(mock: bool = False, fallback_to_recon: bool = True) -> VgBackend:
    """Obtain a :class:`VgBackend` instance.

    Synapse is self-contained with native C++20 geometry operators and
    the :class:`SynapseReconBackend`. External ``vg-core`` is deprecated and
    removed.

    Args:
        mock: If ``True`` (or env var ``SYNAPSE_VG_MOCK=1``), return a
            :class:`MockVgBackend`.
        fallback_to_recon: Maintained for backward compatibility. Always
            delegates to :class:`SynapseReconBackend`.

    Returns:
        A ready-to-use backend instance.
    """
    if mock or os.environ.get("SYNAPSE_VG_MOCK", "") == "1":
        return MockVgBackend()

    from synapse.recon.backend import SynapseReconBackend

    return SynapseReconBackend()


# -----------------------------------------------------------------------
# Track-B whitebox anti-paradox gate (RFC 0044 §4.1 & §4.4)
# -----------------------------------------------------------------------


def enforce_distillation_geometry_gate(
    track: str,
    has_gen3d_highpoly: bool,
    vertex_count: int,
    has_articulated_limbs: bool,
    is_character: bool,
) -> None:
    """Enforce the Track-B whitebox anti-paradox geometry gate.

    Track-B coarse whitebox proxies **must** pass through a generative
    3D reconstruction step (TRELLIS / Hunyuan3D) before AI-video texture
    back-projection is permitted.  Character proxies additionally must
    satisfy the whitebox coarseness floor (≥1500 vertices, articulated
    limbs).

    Args:
        track: Pipeline track identifier (e.g. ``"track_b_whitebox"``).
        has_gen3d_highpoly: Whether the asset has been through
            generative 3D high-poly reconstruction.
        vertex_count: Mesh vertex count.
        has_articulated_limbs: Whether the mesh has articulated joints.
        is_character: Whether this is a character asset.

    Raises:
        ContractValidationError: If the gate check fails (exit_code=10).
    """
    if track != "track_b_whitebox":
        return  # Non Track-B pipelines are unconstrained

    if not has_gen3d_highpoly:
        raise ContractValidationError(
            "Track-B whitebox proxy must pass generative 3D "
            "reconstruction before texture back-projection",
            agent_fix_hint=(
                "Run foundary TRELLIS/Hunyuan3D reconstruction and "
                "quad_remesh before texture back-projection."
            ),
        )

    if is_character and not validate_whitebox_coarseness_floor(
        vertex_count, has_articulated_limbs, is_character
    ):
        raise ContractValidationError(
            f"Track-B character whitebox proxy failed coarseness floor: "
            f"vertex_count={vertex_count} "
            f"(min={MIN_CHARACTER_WHITEBOX_VERTICES}), "
            f"articulated_limbs={has_articulated_limbs}",
            agent_fix_hint=(
                "Run foundary TRELLIS/Hunyuan3D reconstruction and "
                "quad_remesh before texture back-projection."
            ),
        )
