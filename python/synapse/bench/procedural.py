"""Procedural GT geometry generators for Synapse-Bench.

These provide ground-truth 3D meshes with known geometry, allowing
pixel-perfect evaluation without external asset dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class ProceduralMeshType(str, Enum):
    """Available procedural mesh types."""

    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    TORUS = "torus"
    PLANE = "plane"


@dataclass
class ProceduralMesh:
    """A programmatically generated triangle mesh."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uvs: Optional[np.ndarray] = None
    mesh_type: str = "unknown"

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) of the axis-aligned bbox."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def centroid(self) -> np.ndarray:
        """Compute the mesh centroid."""
        return self.vertices.mean(axis=0)

    def save_obj(self, path: str | Path) -> Path:
        """Export to Wavefront OBJ format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"# Synapse-Bench procedural mesh: {self.mesh_type}"]
        lines.append(f"# Vertices: {self.num_vertices}, Faces: {self.num_faces}")

        for v in self.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

        for n in self.normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

        if self.uvs is not None:
            for uv in self.uvs:
                lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

        for f in self.faces:
            if self.uvs is not None:
                lines.append(
                    f"f {f[0]+1}/{f[0]+1}/{f[0]+1} "
                    f"{f[1]+1}/{f[1]+1}/{f[1]+1} "
                    f"{f[2]+1}/{f[2]+1}/{f[2]+1}"
                )
            else:
                lines.append(
                    f"f {f[0]+1}//{f[0]+1} "
                    f"{f[1]+1}//{f[1]+1} "
                    f"{f[2]+1}//{f[2]+1}"
                )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _generate_cube(size: float = 1.0, **kwargs) -> ProceduralMesh:
    """Unit cube centered at origin."""
    s = size / 2.0
    vertices = []
    normals = []
    uvs = []
    faces = []

    face_defs = [
        ([0, 0, 1], [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
        ([0, 0, -1], [(-s, s, -s), (s, s, -s), (s, -s, -s), (-s, -s, -s)]),
        ([0, 1, 0], [(-s, s, -s), (-s, s, s), (s, s, s), (s, s, -s)]),
        ([0, -1, 0], [(-s, -s, s), (-s, -s, -s), (s, -s, -s), (s, -s, s)]),
        ([1, 0, 0], [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),
        ([-1, 0, 0], [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),
    ]

    uv_quad = [(0, 0), (1, 0), (1, 1), (0, 1)]

    for normal, corners in face_defs:
        base = len(vertices)
        for j, c in enumerate(corners):
            vertices.append(c)
            normals.append(normal)
            uvs.append(uv_quad[j])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="cube",
    )


def _generate_sphere(
    radius: float = 1.0,
    segments: int = 32,
    rings: int = 16,
    **kwargs,
) -> ProceduralMesh:
    """UV sphere centered at origin."""
    vertices = []
    normals = []
    uvs = []
    faces = []

    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments + 1):
            theta = 2.0 * math.pi * j / segments

            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)

            nx, ny, nz = x / radius, y / radius, z / radius
            u = j / segments
            v = i / rings

            vertices.append([x, y, z])
            normals.append([nx, ny, nz])
            uvs.append([u, v])

    stride = segments + 1
    for i in range(rings):
        for j in range(segments):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1

            if i > 0:
                faces.append([a, c, b])
            if i < rings - 1:
                faces.append([b, c, d])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="sphere",
    )


def _generate_cylinder(
    radius: float = 0.5,
    height: float = 2.0,
    segments: int = 32,
    **kwargs,
) -> ProceduralMesh:
    """Cylinder centered at origin, axis along Y."""
    vertices = []
    normals = []
    uvs = []
    faces = []
    half_h = height / 2.0

    for i in range(2):
        y = -half_h + i * height
        for j in range(segments + 1):
            theta = 2.0 * math.pi * j / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            nx, nz = math.cos(theta), math.sin(theta)

            vertices.append([x, y, z])
            normals.append([nx, 0.0, nz])
            uvs.append([j / segments, i])

    stride = segments + 1
    for j in range(segments):
        a = j
        b = j + 1
        c = j + stride
        d = j + stride + 1
        faces.append([a, c, b])
        faces.append([b, c, d])

    top_center_idx = len(vertices)
    vertices.append([0, half_h, 0])
    normals.append([0, 1, 0])
    uvs.append([0.5, 0.5])
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        idx = len(vertices)
        vertices.append([x, half_h, z])
        normals.append([0, 1, 0])
        uvs.append([0.5 + 0.5 * math.cos(theta), 0.5 + 0.5 * math.sin(theta)])
        if j > 0:
            faces.append([top_center_idx, idx - 1, idx])
    faces.append([top_center_idx, len(vertices) - 1, top_center_idx + 1])

    bot_center_idx = len(vertices)
    vertices.append([0, -half_h, 0])
    normals.append([0, -1, 0])
    uvs.append([0.5, 0.5])
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        idx = len(vertices)
        vertices.append([x, -half_h, z])
        normals.append([0, -1, 0])
        uvs.append([0.5 + 0.5 * math.cos(theta), 0.5 + 0.5 * math.sin(theta)])
        if j > 0:
            faces.append([bot_center_idx, idx, idx - 1])
    faces.append([bot_center_idx, bot_center_idx + 1, len(vertices) - 1])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="cylinder",
    )


def _generate_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    major_segments: int = 32,
    minor_segments: int = 16,
    **kwargs,
) -> ProceduralMesh:
    """Torus centered at origin, lying in XZ plane."""
    vertices = []
    normals = []
    uvs = []
    faces = []

    for i in range(major_segments + 1):
        theta = 2.0 * math.pi * i / major_segments
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        for j in range(minor_segments + 1):
            phi = 2.0 * math.pi * j / minor_segments
            cos_p, sin_p = math.cos(phi), math.sin(phi)

            r = major_radius + minor_radius * cos_p
            x = r * cos_t
            y = minor_radius * sin_p
            z = r * sin_t

            nx = cos_p * cos_t
            ny = sin_p
            nz = cos_p * sin_t

            vertices.append([x, y, z])
            normals.append([nx, ny, nz])
            uvs.append([i / major_segments, j / minor_segments])

    stride = minor_segments + 1
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="torus",
    )


def _generate_plane(
    size: float = 2.0,
    subdivisions: int = 1,
    **kwargs,
) -> ProceduralMesh:
    """Flat XZ plane centered at origin (Y=0)."""
    n = subdivisions + 1
    half = size / 2.0
    vertices = []
    normals = []
    uvs = []
    faces = []

    for i in range(n + 1):
        for j in range(n + 1):
            x = -half + size * j / n
            z = -half + size * i / n
            vertices.append([x, 0.0, z])
            normals.append([0.0, 1.0, 0.0])
            uvs.append([j / n, i / n])

    stride = n + 1
    for i in range(n):
        for j in range(n):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="plane",
    )


_GENERATORS = {
    ProceduralMeshType.CUBE: _generate_cube,
    ProceduralMeshType.SPHERE: _generate_sphere,
    ProceduralMeshType.CYLINDER: _generate_cylinder,
    ProceduralMeshType.TORUS: _generate_torus,
    ProceduralMeshType.PLANE: _generate_plane,
}


def generate_procedural_mesh(
    mesh_type: ProceduralMeshType | str,
    **params,
) -> ProceduralMesh:
    """Generate a procedural mesh of the given type."""
    if isinstance(mesh_type, str):
        mesh_type = ProceduralMeshType(mesh_type)

    generator = _GENERATORS.get(mesh_type)
    if generator is None:
        raise ValueError(
            f"Unsupported mesh type: {mesh_type}. "
            f"Available: {[t.value for t in ProceduralMeshType]}"
        )

    return generator(**params)


# -------------------------------------------------------------------
# Humanoid-53 proxy generator
# -------------------------------------------------------------------

# Segment definitions: (name, shape, dimensions, parent, joint_offset)
# shape: "cylinder" or "box" or "sphere"
# dimensions: (radius, height) for cylinder, (sx, sy, sz) for box,
#             (radius,) for sphere
# parent: parent segment name or None for root
# joint_offset: (x, y, z) offset from parent joint to this joint

_HUMANOID_SEGMENTS: list[
    tuple[str, str, tuple[float, ...], str | None, tuple[float, float, float]]
] = [
    # Torso (root) — centered at origin, tall box
    ("torso", "cylinder", (0.18, 0.50), None, (0.0, 0.0, 0.0)),
    # Pelvis — below torso
    ("pelvis", "cylinder", (0.16, 0.15), "torso", (0.0, -0.325, 0.0)),
    # Head — above torso
    ("neck", "cylinder", (0.06, 0.10), "torso", (0.0, 0.30, 0.0)),
    ("head", "sphere", (0.12,), "neck", (0.0, 0.11, 0.0)),
    # Left arm chain
    ("left_upper_arm", "cylinder", (0.05, 0.28),
     "torso", (-0.23, 0.20, 0.0)),
    ("left_forearm", "cylinder", (0.04, 0.25),
     "left_upper_arm", (0.0, -0.265, 0.0)),
    ("left_hand", "box", (0.04, 0.08, 0.02),
     "left_forearm", (0.0, -0.165, 0.0)),
    # Right arm chain
    ("right_upper_arm", "cylinder", (0.05, 0.28),
     "torso", (0.23, 0.20, 0.0)),
    ("right_forearm", "cylinder", (0.04, 0.25),
     "right_upper_arm", (0.0, -0.265, 0.0)),
    ("right_hand", "box", (0.04, 0.08, 0.02),
     "right_forearm", (0.0, -0.165, 0.0)),
    # Left leg chain
    ("left_thigh", "cylinder", (0.07, 0.40),
     "pelvis", (-0.10, -0.275, 0.0)),
    ("left_shin", "cylinder", (0.05, 0.38),
     "left_thigh", (0.0, -0.39, 0.0)),
    ("left_foot", "box", (0.06, 0.04, 0.15),
     "left_shin", (0.0, -0.21, 0.04)),
    # Right leg chain
    ("right_thigh", "cylinder", (0.07, 0.40),
     "pelvis", (0.10, -0.275, 0.0)),
    ("right_shin", "cylinder", (0.05, 0.38),
     "right_thigh", (0.0, -0.39, 0.0)),
    ("right_foot", "box", (0.06, 0.04, 0.15),
     "right_shin", (0.0, -0.21, 0.04)),
]


def _make_rotation_x(angle: float) -> np.ndarray:
    """3x3 rotation matrix around X axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float64)


def _make_rotation_y(angle: float) -> np.ndarray:
    """3x3 rotation matrix around Y axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)


def _make_rotation_z(angle: float) -> np.ndarray:
    """3x3 rotation matrix around Z axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _generate_segment_mesh(
    shape: str,
    dims: tuple[float, ...],
    segments: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and faces for a single body segment.

    Returns (vertices [N, 3], faces [F, 3]) centered at origin.
    All generated meshes are watertight (closed manifold with shared
    vertices).
    """
    if shape == "sphere":
        radius = dims[0]
        # Build a watertight UV sphere with shared vertices
        rings = max(6, segments // 2)
        vertices: list[list[float]] = []
        faces_list: list[list[int]] = []

        # Top pole
        vertices.append([0.0, radius, 0.0])
        # Ring vertices
        for i in range(1, rings):
            phi = math.pi * i / rings
            for j in range(segments):
                theta = 2.0 * math.pi * j / segments
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                vertices.append([x, y, z])
        # Bottom pole
        bot_idx = len(vertices)
        vertices.append([0.0, -radius, 0.0])

        # Top cap triangles
        for j in range(segments):
            j_next = (j + 1) % segments
            faces_list.append([0, 1 + j, 1 + j_next])

        # Middle quads
        for i in range(rings - 2):
            base = 1 + i * segments
            for j in range(segments):
                j_next = (j + 1) % segments
                a = base + j
                b = base + j_next
                c = base + segments + j
                d = base + segments + j_next
                faces_list.append([a, c, b])
                faces_list.append([b, c, d])

        # Bottom cap triangles
        last_ring_base = 1 + (rings - 2) * segments
        for j in range(segments):
            j_next = (j + 1) % segments
            faces_list.append([
                last_ring_base + j,
                bot_idx,
                last_ring_base + j_next,
            ])

        return (
            np.array(vertices, dtype=np.float64),
            np.array(faces_list, dtype=np.int32),
        )

    if shape == "cylinder":
        radius, height = dims[0], dims[1]
        half_h = height / 2.0
        vertices = []
        faces_list = []

        # Top center
        top_c = 0
        vertices.append([0.0, half_h, 0.0])
        # Top ring
        top_ring_start = len(vertices)
        for j in range(segments):
            theta = 2.0 * math.pi * j / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append([x, half_h, z])

        # Bottom ring
        bot_ring_start = len(vertices)
        for j in range(segments):
            theta = 2.0 * math.pi * j / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append([x, -half_h, z])

        # Bottom center
        bot_c = len(vertices)
        vertices.append([0.0, -half_h, 0.0])

        # Top cap (fan)
        for j in range(segments):
            j_next = (j + 1) % segments
            faces_list.append([
                top_c,
                top_ring_start + j_next,
                top_ring_start + j,
            ])

        # Side quads
        for j in range(segments):
            j_next = (j + 1) % segments
            a = top_ring_start + j
            b = top_ring_start + j_next
            c = bot_ring_start + j
            d = bot_ring_start + j_next
            faces_list.append([a, b, d])
            faces_list.append([a, d, c])

        # Bottom cap (fan)
        for j in range(segments):
            j_next = (j + 1) % segments
            faces_list.append([
                bot_c,
                bot_ring_start + j,
                bot_ring_start + j_next,
            ])

        return (
            np.array(vertices, dtype=np.float64),
            np.array(faces_list, dtype=np.int32),
        )

    if shape == "box":
        # Build a watertight box with 8 shared vertices
        sx, sy, sz = dims[0], dims[1], dims[2]
        verts = np.array([
            [-sx, -sy, -sz], [sx, -sy, -sz],
            [sx, sy, -sz], [-sx, sy, -sz],
            [-sx, -sy, sz], [sx, -sy, sz],
            [sx, sy, sz], [-sx, sy, sz],
        ], dtype=np.float64)
        faces = np.array([
            # Front  (+Z)
            [4, 5, 6], [4, 6, 7],
            # Back   (-Z)
            [1, 0, 3], [1, 3, 2],
            # Top    (+Y)
            [3, 7, 6], [3, 6, 2],
            # Bottom (-Y)
            [0, 1, 5], [0, 5, 4],
            # Right  (+X)
            [1, 2, 6], [1, 6, 5],
            # Left   (-X)
            [0, 4, 7], [0, 7, 3],
        ], dtype=np.int32)
        return verts, faces

    raise ValueError(f"Unknown segment shape: {shape}")


def _compute_walk_pose(
    phase: float,
) -> dict[str, tuple[float, float, float]]:
    """Compute joint rotation angles for a walking pose.

    Parameters
    ----------
    phase:
        Walk cycle phase in radians [0, 2*pi).

    Returns
    -------
    dict mapping segment name to (rx, ry, rz) Euler angles in radians.
    """
    # Amplitude of swing in radians
    leg_swing = 0.45  # ~26 degrees
    arm_swing = 0.35  # ~20 degrees
    shin_bend = 0.30
    forearm_bend = 0.20

    s = math.sin(phase)
    c = math.cos(phase)

    rotations: dict[str, tuple[float, float, float]] = {}

    # Legs swing opposite to each other
    rotations["left_thigh"] = (leg_swing * s, 0.0, 0.0)
    rotations["right_thigh"] = (-leg_swing * s, 0.0, 0.0)

    # Shins bend when leg swings back (always positive X rotation)
    left_shin_x = max(0.0, shin_bend * (-s))
    right_shin_x = max(0.0, shin_bend * s)
    rotations["left_shin"] = (left_shin_x, 0.0, 0.0)
    rotations["right_shin"] = (right_shin_x, 0.0, 0.0)

    # Arms swing opposite to legs (counter-rotation)
    rotations["left_upper_arm"] = (-arm_swing * s, 0.0, 0.0)
    rotations["right_upper_arm"] = (arm_swing * s, 0.0, 0.0)

    # Forearms slight bend during swing
    left_fore_x = max(0.0, forearm_bend * s)
    right_fore_x = max(0.0, forearm_bend * (-s))
    rotations["left_forearm"] = (left_fore_x, 0.0, 0.0)
    rotations["right_forearm"] = (right_fore_x, 0.0, 0.0)

    # Slight torso rotation
    rotations["torso"] = (0.0, 0.05 * s, 0.0)
    rotations["pelvis"] = (0.0, -0.03 * s, 0.0)

    return rotations


def generate_humanoid_proxy(
    pose_params: dict[str, float] | None = None,
    target_vertices: int = 1820,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Generate a Humanoid-53 articulated proxy mesh.

    Parameters
    ----------
    pose_params:
        Optional dict with ``"phase"`` key (float, radians [0, 2pi))
        for walking animation pose. Default is T-pose (phase=0 gives
        slight walk start; omit or set None for neutral standing).
    target_vertices:
        Target vertex count. Segment tessellation is adjusted to
        approximate this target. Must be >= 1500.

    Returns
    -------
    tuple of:
        - vertices: ``np.ndarray`` shape ``[N, 3]``, float64
        - faces: ``np.ndarray`` shape ``[F, 3]``, int32
        - bone_indices: ``dict[str, list[int]]`` mapping segment names
          to their vertex index ranges
    """
    if target_vertices < 1500:
        raise ValueError(
            f"target_vertices must be >= 1500, got {target_vertices}"
        )

    # Adjust tessellation: need enough vertices across all segments.
    # Watertight cylinder with s segments: 2*s + 2 verts.
    # Watertight sphere with s segments, s/2 rings:
    #   2 + (s/2 - 1) * s verts.
    # Box: always 8 verts.
    # With 16 body parts (10 cyls, 1 sphere, 3 boxes, 2 cyls):
    #   at seg=28: 12 cyls * 58 + 1 sphere * 366 + 3 boxes * 8
    #   = 696 + 366 + 24 = 1086 (not enough)
    # Bump higher: seg=32 -> 12*66 + 1*(2+(15)*32) + 24
    #   = 792 + 482 + 24 = 1298 (still short)
    # Need seg=40: 12*82 + 1*(2+19*40) + 24
    #   = 984 + 762 + 24 = 1770
    n_segments = len(_HUMANOID_SEGMENTS)
    # Scale segments proportional to target
    cyl_segs = max(40, int(target_vertices / (n_segments * 2.5)))

    # Get walk pose rotations
    phase = 0.0
    if pose_params is not None:
        phase = pose_params.get("phase", 0.0)

    walk_rotations = _compute_walk_pose(phase)

    # Build each segment: compute world transform via parent chain
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    bone_indices: dict[str, list[int]] = {}

    # Joint world positions and rotations
    joint_world_pos: dict[str, np.ndarray] = {}
    joint_world_rot: dict[str, np.ndarray] = {}

    vertex_offset = 0

    for seg_name, shape, dims, parent, joint_off in _HUMANOID_SEGMENTS:
        # Generate local mesh
        seg_verts, seg_faces = _generate_segment_mesh(
            shape, dims, segments=cyl_segs,
        )

        # Compute world transform
        local_rot_angles = walk_rotations.get(
            seg_name, (0.0, 0.0, 0.0),
        )
        local_rot = (
            _make_rotation_z(local_rot_angles[2])
            @ _make_rotation_y(local_rot_angles[1])
            @ _make_rotation_x(local_rot_angles[0])
        )

        if parent is None:
            # Root segment
            world_pos = np.array(joint_off, dtype=np.float64)
            world_rot = local_rot
        else:
            parent_pos = joint_world_pos[parent]
            parent_rot = joint_world_rot[parent]
            offset = np.array(joint_off, dtype=np.float64)
            world_pos = parent_pos + parent_rot @ offset
            world_rot = parent_rot @ local_rot

        joint_world_pos[seg_name] = world_pos
        joint_world_rot[seg_name] = world_rot

        # Transform vertices to world space
        transformed = (world_rot @ seg_verts.T).T + world_pos

        # Track bone indices
        n_v = transformed.shape[0]
        bone_indices[seg_name] = list(
            range(vertex_offset, vertex_offset + n_v),
        )

        # Offset face indices
        offset_faces = seg_faces + vertex_offset

        all_vertices.append(transformed)
        all_faces.append(offset_faces)
        vertex_offset += n_v

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float64)
    faces = np.concatenate(all_faces, axis=0).astype(np.int32)

    return vertices, faces, bone_indices
