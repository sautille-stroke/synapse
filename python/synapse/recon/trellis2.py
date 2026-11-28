"""TRELLIS.2 O-Voxel reconstruction adapter and PBR asset decoding.

Upgrades the existing ``TrellisReconstructionAdapter`` (v1) to support
the TRELLIS 2 (4B-parameter) architecture which introduces:

- **O-Voxel representation**: Occupancy voxels with per-voxel PBR
  attributes (base colour, roughness, metallic, opacity) that are
  decoded into a polygon mesh + PBR textures.
- **Open-surface topology**: Unlike v1 which produces only closed
  manifolds, TRELLIS.2 can emit thin surfaces (cloth, leaves, hair
  cards) that are topologically open.  These use the new
  :class:`TopologyPolicy` to pass mesh validation gates.
- **Full PBR extraction**: Roughness, metallic, and opacity maps
  are extracted alongside base colour for downstream USD/glTF export.

Transport backends:

- :class:`MockTrellis2Backend` — deterministic CPU-only mock.
- :class:`FoundaryTrellis2Backend` — routes to foundary ``trellis``
  service endpoint (v2 API).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.errors import BackendUnavailableError, ContractValidationError
from synapse.recon.types import MeshGeometry

# ---------------------------------------------------------------------------
# Topology policy
# ---------------------------------------------------------------------------


class TopologyPolicy(enum.Enum):
    """Mesh topology validation policy.

    Controls how mesh validation gates (``check_manifold``,
    ``check_watertight``) treat open surfaces vs closed volumes.

    Attributes:
        STRICT: Require watertight closed manifold (default for rigid
            body assets like props, vehicles, buildings).
        OPEN_SURFACE_ALLOWED: Accept manifold meshes with boundary
            edges (cloth, foliage, hair cards, decals).
        PERMISSIVE: Accept any mesh — only check for NaN/Inf and
            degenerate faces. For prototyping only.
    """

    STRICT = "strict"
    OPEN_SURFACE_ALLOWED = "open_surface_allowed"
    PERMISSIVE = "permissive"


# ---------------------------------------------------------------------------
# O-Voxel PBR result
# ---------------------------------------------------------------------------


@dataclass
class OVoxelPBRMaps:
    """PBR material maps extracted from TRELLIS.2 O-Voxel decoding.

    All texture maps share the same UV layout as the decoded mesh.

    Attributes:
        base_color: ``[H, W, 3]`` float32 linear RGB base colour.
        roughness: ``[H, W]`` float32 perceptual roughness in [0, 1].
        metallic: ``[H, W]`` float32 metalness in [0, 1].
        opacity: ``[H, W]`` float32 opacity in [0, 1] (1.0 = fully opaque).
        normal_map: Optional ``[H, W, 3]`` float32 tangent-space normal map.
        metadata: Backend metadata (timings, model version, etc.).
    """

    base_color: np.ndarray  # [H, W, 3] float32
    roughness: np.ndarray  # [H, W] float32
    metallic: np.ndarray  # [H, W] float32
    opacity: np.ndarray  # [H, W] float32
    normal_map: np.ndarray | None = None  # [H, W, 3] float32
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trellis2Result:
    """Full reconstruction result from TRELLIS.2.

    Attributes:
        mesh: Decoded polygon mesh (may be open or closed depending
            on the asset type).
        pbr_maps: Per-texel PBR material decomposition.
        topology_policy: Topology classification assigned by the
            backend based on asset analysis.
        confidence: Reconstruction confidence score in [0, 1].
        metadata: Backend metadata.
    """

    mesh: MeshGeometry
    pbr_maps: OVoxelPBRMaps
    topology_policy: TopologyPolicy = TopologyPolicy.STRICT
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transport ABC
# ---------------------------------------------------------------------------


class Trellis2Transport(ABC):
    """Abstract transport for TRELLIS.2 O-Voxel reconstruction."""

    @abstractmethod
    def reconstruct(
        self,
        keyframes: np.ndarray,
        *,
        prompts: list[str] | None = None,
        topology_hint: TopologyPolicy = TopologyPolicy.STRICT,
    ) -> Trellis2Result:
        """Reconstruct a 3D asset from input keyframes.

        Args:
            keyframes: ``[K, H, W, 3]`` uint8 sRGB keyframe images.
            prompts: Optional text prompts guiding reconstruction.
            topology_hint: Desired topology policy — the backend
                may override based on O-Voxel analysis.

        Returns:
            :class:`Trellis2Result` with mesh and PBR maps.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend is ready to accept work."""
        ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


def _generate_mock_mesh(
    subdivisions: int = 20,
    radius: float = 0.6,
    open_surface: bool = False,
) -> MeshGeometry:
    """Generate a synthetic mesh for testing.

    Args:
        subdivisions: Number of latitude/longitude subdivisions.
        radius: Sphere radius.
        open_surface: If True, remove bottom cap to create an open surface.

    Returns:
        :class:`MeshGeometry` with vertices, faces, normals, and UVs.
    """
    verts = []
    uvs = []
    n_lat = subdivisions
    n_lon = subdivisions

    # Generate vertices
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon + 1):
            phi = 2.0 * np.pi * j / n_lon
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.cos(theta)
            z = radius * np.sin(theta) * np.sin(phi)
            verts.append([x, y, z])
            uvs.append([j / n_lon, i / n_lat])

    verts_arr = np.array(verts, dtype=np.float32)
    uvs_arr = np.array(uvs, dtype=np.float32)

    # Generate faces
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            # Skip bottom cap for open surface
            if open_surface and i == n_lat - 1:
                continue
            a = i * (n_lon + 1) + j
            b = a + n_lon + 1
            c = b + 1
            d = a + 1
            faces.append([a, b, c])
            faces.append([a, c, d])

    faces_arr = np.array(faces, dtype=np.int32)

    # Compute normals
    normals = verts_arr / (np.linalg.norm(verts_arr, axis=1, keepdims=True) + 1e-8)

    return MeshGeometry(
        vertices=verts_arr,
        faces=faces_arr,
        normals=normals.astype(np.float32),
        uvs=uvs_arr,
    )


def _generate_mock_pbr_maps(
    tex_size: int = 64,
    has_transparency: bool = False,
) -> OVoxelPBRMaps:
    """Generate synthetic PBR maps for testing.

    Args:
        tex_size: Texture resolution (square).
        has_transparency: Whether to include semi-transparent regions.

    Returns:
        :class:`OVoxelPBRMaps` with deterministic content.
    """
    rng = np.random.RandomState(42)
    base_color = rng.rand(tex_size, tex_size, 3).astype(np.float32) * 0.8 + 0.1
    roughness = np.full((tex_size, tex_size), 0.45, dtype=np.float32)
    metallic = np.zeros((tex_size, tex_size), dtype=np.float32)

    if has_transparency:
        opacity = np.ones((tex_size, tex_size), dtype=np.float32)
        # Create a gradient transparency band
        opacity[tex_size // 4 : tex_size // 2, :] = 0.5
        opacity[:tex_size // 8, :] = 0.0
    else:
        opacity = np.ones((tex_size, tex_size), dtype=np.float32)

    # Tangent-space normal map (default flat = [0.5, 0.5, 1.0] in sRGB convention)
    normal_map = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    normal_map[..., 0] = 0.5
    normal_map[..., 1] = 0.5
    normal_map[..., 2] = 1.0

    return OVoxelPBRMaps(
        base_color=base_color,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
        normal_map=normal_map,
        metadata={"backend": "mock", "tex_size": tex_size},
    )


class MockTrellis2Backend(Trellis2Transport):
    """Deterministic CPU-only mock for TRELLIS.2.

    Generates a UV sphere (open or closed depending on topology hint)
    with synthetic PBR maps.
    """

    def reconstruct(
        self,
        keyframes: np.ndarray,
        *,
        prompts: list[str] | None = None,
        topology_hint: TopologyPolicy = TopologyPolicy.STRICT,
    ) -> Trellis2Result:
        """Generate a mock reconstruction result.

        Args:
            keyframes: ``[K, H, W, 3]`` uint8 sRGB keyframes.
            prompts: Ignored in mock mode.
            topology_hint: Controls whether mesh is open or closed.

        Returns:
            Deterministic :class:`Trellis2Result`.
        """
        if len(keyframes) == 0:
            raise ContractValidationError(
                "TRELLIS.2 requires >= 1 keyframe.",
                agent_fix_hint="Select keyframes via select_adaptive_keyframes() first.",
            )

        is_open = topology_hint == TopologyPolicy.OPEN_SURFACE_ALLOWED
        mesh = _generate_mock_mesh(
            subdivisions=20,
            radius=0.6,
            open_surface=is_open,
        )
        pbr = _generate_mock_pbr_maps(
            tex_size=64,
            has_transparency=is_open,
        )

        return Trellis2Result(
            mesh=mesh,
            pbr_maps=pbr,
            topology_policy=topology_hint,
            confidence=0.95,
            metadata={
                "backend": "mock",
                "version": "trellis_2.0_4b",
                "keyframe_count": len(keyframes),
                "open_surface": is_open,
            },
        )

    def is_available(self) -> bool:
        """Mock backend is always available."""
        return True


# ---------------------------------------------------------------------------
# Foundary-backed transport
# ---------------------------------------------------------------------------


class FoundaryTrellis2Backend(Trellis2Transport):
    """TRELLIS.2 transport via the foundary container service.

    Submits reconstruction requests to the ``trellis`` service
    endpoint (v2 API) registered in the foundary container.

    Args:
        foundary_url: Base URL of the foundary container.
    """

    def __init__(self, foundary_url: str = "") -> None:
        self._url = foundary_url

    def reconstruct(
        self,
        keyframes: np.ndarray,
        *,
        prompts: list[str] | None = None,
        topology_hint: TopologyPolicy = TopologyPolicy.STRICT,
    ) -> Trellis2Result:
        """Submit reconstruction job to foundary.

        Raises:
            BackendUnavailableError: Always — not yet implemented.
        """
        raise BackendUnavailableError(
            "FoundaryTrellis2Backend.reconstruct not yet "
            "connected to live endpoint.",
            agent_fix_hint=(
                "Use MockTrellis2Backend for testing, or "
                "configure foundary_url to point at a running "
                "trellis v2 service."
            ),
        )

    def is_available(self) -> bool:
        """Return ``False`` — live endpoint not yet connected."""
        return False


# ---------------------------------------------------------------------------
# Topology-aware mesh validation
# ---------------------------------------------------------------------------


def validate_mesh_topology(
    mesh: MeshGeometry,
    policy: TopologyPolicy = TopologyPolicy.STRICT,
    *,
    min_vertices: int = 100,
) -> dict[str, Any]:
    """Validate mesh geometry against a topology policy.

    Applies the appropriate level of topology validation based on
    the policy:

    - ``STRICT``: Must be a closed watertight 2-manifold with
      ``>= min_vertices`` vertices.
    - ``OPEN_SURFACE_ALLOWED``: Must be 2-manifold (boundary edges
      OK) with ``>= min_vertices`` vertices.
    - ``PERMISSIVE``: Only checks for NaN/Inf vertices and minimum
      vertex count.

    Args:
        mesh: Mesh to validate.
        policy: Topology policy to enforce.
        min_vertices: Minimum vertex count threshold.

    Returns:
        Dict with validation results:
        - ``"valid"``: bool — overall pass/fail.
        - ``"policy"``: str — policy name applied.
        - ``"vertex_count"``: int — number of vertices.
        - ``"face_count"``: int — number of faces.
        - ``"is_manifold"``: bool | None.
        - ``"is_watertight"``: bool | None.
        - ``"has_finite_vertices"``: bool.
        - ``"errors"``: list[str] — failure reasons.
    """
    from synapse.recon.geometry.topology import check_manifold, check_watertight

    errors: list[str] = []
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)

    has_finite = bool(np.all(np.isfinite(v))) if len(v) > 0 else False
    if not has_finite:
        errors.append("Vertices contain NaN or Inf values.")

    vc = mesh.vertex_count
    fc = mesh.face_count
    if vc < min_vertices:
        errors.append(f"Vertex count {vc} < minimum {min_vertices}.")

    is_manifold_result: bool | None = None
    is_watertight_result: bool | None = None

    if policy == TopologyPolicy.STRICT:
        is_manifold_result = check_manifold(v, f) if len(v) > 0 and len(f) > 0 else False
        is_watertight_result = check_watertight(v, f) if len(v) > 0 and len(f) > 0 else False
        if not is_manifold_result:
            errors.append("Mesh is not 2-manifold (STRICT policy).")
        if not is_watertight_result:
            errors.append("Mesh is not watertight (STRICT policy).")

    elif policy == TopologyPolicy.OPEN_SURFACE_ALLOWED:
        is_manifold_result = check_manifold(v, f) if len(v) > 0 and len(f) > 0 else False
        if not is_manifold_result:
            errors.append("Mesh is not 2-manifold (OPEN_SURFACE_ALLOWED policy).")
        # Watertight is NOT required — boundary edges are acceptable
        is_watertight_result = check_watertight(v, f) if len(v) > 0 and len(f) > 0 else False

    elif policy == TopologyPolicy.PERMISSIVE:
        # Only basic checks
        pass

    return {
        "valid": len(errors) == 0,
        "policy": policy.value,
        "vertex_count": vc,
        "face_count": fc,
        "is_manifold": is_manifold_result,
        "is_watertight": is_watertight_result,
        "has_finite_vertices": has_finite,
        "errors": errors,
    }


__all__ = [
    "FoundaryTrellis2Backend",
    "MockTrellis2Backend",
    "OVoxelPBRMaps",
    "TopologyPolicy",
    "Trellis2Result",
    "Trellis2Transport",
    "validate_mesh_topology",
]
