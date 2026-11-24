"""Subprocess client and ``.fmesh.bin`` wire codec for ``@aether//tools:aether_geo_worker``.

Matches the binary wire format defined in upstream
``aether/asset_ir/include/aether/asset_ir/fmesh_wire.h``:

- ``kFMeshMagic = 0x48534D46`` ("FMSH")
- 64-byte packed ``FMeshHeader`` (``<IHHIIII3f3f16s``)
- 16-byte packed ``FMeshStreamDescriptor`` entries (``<HHIII``)
- 16-byte aligned stream payloads (Position, Normal, TexCoord0, Indices)

CRITICAL CONTRACT (``NEG-100-08``):
Upstream ``aether_geo_worker --algorithm quadriflow`` is currently a stub
returning exit code ``1``.  :meth:`AetherGeoWorkerClient.quadriflow` MUST fail
closed with :class:`~synapse.errors.NotImplementedStageError` (exit code ``50``)
whenever the worker returns a stub error or the binary is unavailable — never
silently succeed with exit code ``0``.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synapse.errors import (
    BackendUnavailableError,
    ContractValidationError,
    NotImplementedStageError,
)

__all__ = [
    "FMESH_MAGIC",
    "FMESH_VERSION",
    "AetherGeoWorkerClient",
    "FMeshData",
    "read_fmesh_bin",
    "write_fmesh_bin",
]

FMESH_MAGIC: int = 0x48534D46  # "FMSH" little-endian
FMESH_VERSION: int = 2
_HEADER_FMT = "<IHHIIII3f3f16s"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 64 bytes
_STREAM_DESC_FMT = "<HHIII"
_STREAM_DESC_SIZE = struct.calcsize(_STREAM_DESC_FMT)  # 16 bytes

# Semantic IDs matching aether::asset_ir::FMeshSemanticId
_SEM_POSITION = 1
_SEM_NORMAL = 2
_SEM_TEXCOORD0 = 4
_SEM_INDICES = 8

# Component types matching FMeshStreamDescriptor
_TYPE_F32 = 0
_TYPE_U16 = 1
_TYPE_U32 = 2


def _align16(offset: int) -> int:
    return (offset + 15) & ~15


@dataclass
class FMeshData:
    """Decoded triangle mesh payload from a ``.fmesh.bin`` wire file."""

    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int32
    normals: np.ndarray | None = None  # (V, 3) float32
    uvs: np.ndarray | None = None  # (V, 2) float32


def write_fmesh_bin(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
) -> Path:
    """Encode mesh buffers into a binary ``.fmesh.bin`` file."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ContractValidationError(
            f"Expected vertices with shape (V, 3), got {v.shape}.",
            agent_fix_hint="Provide (V, 3) float32 vertex positions.",
        )
    if f.ndim != 2 or f.shape[1] != 3:
        raise ContractValidationError(
            f"Expected triangle faces with shape (F, 3), got {f.shape}.",
            agent_fix_hint="Triangulate polygon faces to (F, 3) before encoding .fmesh.bin.",
        )

    streams: list[tuple[int, int, int, bytes]] = []
    streams.append((_SEM_POSITION, 3, _TYPE_F32, v.tobytes()))
    streams.append((_SEM_INDICES, 1, _TYPE_U32, f.reshape(-1).tobytes()))

    if normals is not None:
        n_arr = np.ascontiguousarray(normals, dtype=np.float32)
        if n_arr.shape != v.shape:
            raise ContractValidationError(
                f"Normals shape {n_arr.shape} must match vertices {v.shape}.",
                agent_fix_hint="Ensure per-vertex normals match vertex count.",
            )
        streams.append((_SEM_NORMAL, 3, _TYPE_F32, n_arr.tobytes()))

    if uvs is not None:
        uv_arr = np.ascontiguousarray(uvs, dtype=np.float32)
        if uv_arr.shape != (v.shape[0], 2):
            raise ContractValidationError(
                f"UVs shape {uv_arr.shape} must match ({v.shape[0]}, 2).",
                agent_fix_hint="Ensure per-vertex UVs match vertex count.",
            )
        streams.append((_SEM_TEXCOORD0, 2, _TYPE_F32, uv_arr.tobytes()))

    if len(v) > 0:
        bbox_min = tuple(float(x) for x in np.min(v, axis=0))
        bbox_max = tuple(float(x) for x in np.max(v, axis=0))
    else:
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (0.0, 0.0, 0.0)

    header = struct.pack(
        _HEADER_FMT,
        FMESH_MAGIC,
        FMESH_VERSION,
        0,
        int(v.shape[0]),
        int(f.size),
        1,
        len(streams),
        bbox_min[0],
        bbox_min[1],
        bbox_min[2],
        bbox_max[0],
        bbox_max[1],
        bbox_max[2],
        b"\x00" * 16,
    )

    data_offset = _align16(_HEADER_SIZE + len(streams) * _STREAM_DESC_SIZE)
    descriptors = bytearray()
    payloads = bytearray()
    cur_offset = data_offset

    # Pad gap between descriptors and first aligned data block
    header_end = _HEADER_SIZE + len(streams) * _STREAM_DESC_SIZE
    pre_pad = data_offset - header_end

    for sem_id, comp_cnt, comp_type, raw_bytes in streams:
        descriptors.extend(
            struct.pack(
                _STREAM_DESC_FMT,
                sem_id,
                comp_cnt,
                comp_type,
                cur_offset,
                len(raw_bytes),
            )
        )
        payloads.extend(raw_bytes)
        next_offset = _align16(cur_offset + len(raw_bytes))
        pad = next_offset - (cur_offset + len(raw_bytes))
        if pad > 0:
            payloads.extend(b"\x00" * pad)
        cur_offset = next_offset

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + bytes(descriptors) + (b"\x00" * pre_pad) + bytes(payloads))
    return out_path


def read_fmesh_bin(path: str | Path) -> FMeshData:
    """Decode a binary ``.fmesh.bin`` file into :class:`FMeshData`."""
    p = Path(path)
    if not p.is_file():
        raise ContractValidationError(
            f".fmesh.bin file not found: {p}",
            agent_fix_hint="Verify the binary mesh output path exists.",
        )

    blob = p.read_bytes()
    if len(blob) < _HEADER_SIZE:
        raise ContractValidationError(
            f"Truncated .fmesh.bin header ({len(blob)} bytes < {_HEADER_SIZE}).",
            agent_fix_hint="Regenerate the .fmesh.bin asset.",
        )

    unpacked = struct.unpack_from(_HEADER_FMT, blob, 0)
    magic, version, _, vertex_count, index_count, _, stream_count = unpacked[:7]
    if magic != FMESH_MAGIC:
        raise ContractValidationError(
            f"Invalid .fmesh.bin magic: 0x{magic:08X} (expected 0x{FMESH_MAGIC:08X}).",
            agent_fix_hint="Ensure the file conforms to aether::asset_ir FMeshHeader.",
        )
    if version not in (1, 2):
        raise ContractValidationError(
            f"Unsupported .fmesh.bin version {version}; expected 1 or 2.",
            agent_fix_hint="Regenerate .fmesh.bin with schema version 1 or 2.",
        )

    table_end = _HEADER_SIZE + stream_count * _STREAM_DESC_SIZE
    if len(blob) < table_end:
        raise ContractValidationError(
            "Truncated .fmesh.bin stream descriptor table.",
            agent_fix_hint="Verify .fmesh.bin file integrity.",
        )

    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None

    for i in range(stream_count):
        sem_id, comp_cnt, comp_type, byte_off, byte_len = struct.unpack_from(
            _STREAM_DESC_FMT, blob, _HEADER_SIZE + i * _STREAM_DESC_SIZE
        )
        if byte_off + byte_len > len(blob):
            raise ContractValidationError(
                f"Stream {i} out of bounds ({byte_off}+{byte_len} > {len(blob)}).",
                agent_fix_hint="Verify .fmesh.bin file integrity.",
            )
        chunk = blob[byte_off : byte_off + byte_len]
        if sem_id == _SEM_POSITION and comp_type == _TYPE_F32:
            vertices = np.frombuffer(chunk, dtype=np.float32).reshape(vertex_count, comp_cnt).copy()
        elif sem_id == _SEM_NORMAL and comp_type == _TYPE_F32:
            normals = np.frombuffer(chunk, dtype=np.float32).reshape(vertex_count, comp_cnt).copy()
        elif sem_id == _SEM_TEXCOORD0 and comp_type == _TYPE_F32:
            uvs = np.frombuffer(chunk, dtype=np.float32).reshape(vertex_count, comp_cnt).copy()
        elif sem_id == _SEM_INDICES:
            if comp_type == _TYPE_U16:
                flat_idx = np.frombuffer(chunk, dtype=np.uint16).astype(np.int32)
            else:
                flat_idx = np.frombuffer(chunk, dtype=np.uint32).astype(np.int32)
            if len(flat_idx) != index_count:
                raise ContractValidationError(
                    f"Index stream length {len(flat_idx)} != header index_count {index_count}.",
                    agent_fix_hint="Verify .fmesh.bin index stream consistency.",
                )
            faces = flat_idx.reshape(-1, 3).copy()

    if vertices is None or faces is None:
        raise ContractValidationError(
            ".fmesh.bin missing required Position or Indices stream.",
            agent_fix_hint="Include semantic_id 1 (Position) and 8 (Indices) streams.",
        )

    return FMeshData(vertices=vertices, faces=faces, normals=normals, uvs=uvs)


class AetherGeoWorkerClient:
    """Subprocess client invoking ``@aether//tools:aether_geo_worker`` on ``.fmesh.bin`` files."""

    def __init__(self, worker_bin: str | Path | None = None) -> None:
        self._explicit_bin = Path(worker_bin) if worker_bin is not None else None

    def resolve_binary(self) -> Path:
        """Locate ``aether_geo_worker`` executable or raise BackendUnavailableError (exit 30)."""
        candidates: list[Path] = []
        if self._explicit_bin is not None:
            candidates.append(self._explicit_bin)
        env_bin = os.environ.get("AETHER_GEO_WORKER_BIN")
        if env_bin:
            candidates.append(Path(env_bin))
        aether_root = os.environ.get("AETHER_ROOT")
        if aether_root:
            candidates.append(Path(aether_root) / "bazel-bin" / "tools" / "aether_geo_worker")
        # Sibling checkout relative to synapse
        sibling_aether = Path(__file__).resolve().parents[4] / "aether"
        candidates.append(sibling_aether / "bazel-bin" / "tools" / "aether_geo_worker")

        for cand in candidates:
            if cand.is_file():
                return cand

        which_bin = shutil.which("aether_geo_worker")
        if which_bin:
            return Path(which_bin)

        raise BackendUnavailableError(
            "aether_geo_worker binary not found.",
            agent_fix_hint=(
                "Build '//tools:aether_geo_worker' in ../aether or set AETHER_GEO_WORKER_BIN."
            ),
        )

    def _run_algorithm(
        self,
        algorithm: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        extra_args: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> FMeshData:
        worker_path = self.resolve_binary()
        with tempfile.TemporaryDirectory(prefix="aether_geo_") as tmp:
            in_path = Path(tmp) / "input.fmesh.bin"
            out_path = Path(tmp) / "output.fmesh.bin"
            write_fmesh_bin(in_path, vertices, faces, normals=normals, uvs=uvs)
            cmd = [
                str(worker_path),
                "--algorithm",
                algorithm,
                "--input",
                str(in_path),
                "--output",
                str(out_path),
            ]
            if extra_args:
                cmd.extend(extra_args)

            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                shell=False,
                check=False,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or proc.stdout or "").strip()
                if algorithm == "quadriflow" or "not implemented" in stderr.lower() or "stub" in stderr.lower():
                    raise NotImplementedStageError(
                        f"aether_geo_worker --algorithm {algorithm} is unimplemented / failed closed "
                        f"(exit_code={proc.returncode}): {stderr or 'contract stub'}",
                        agent_fix_hint=(
                            "Implement upstream @aether//tools:aether_geo_worker quadriflow backend "
                            "or invoke quad_remesh with strict_native=False."
                        ),
                        details={"algorithm": algorithm, "worker_returncode": proc.returncode},
                    )
                raise ContractValidationError(
                    f"aether_geo_worker --algorithm {algorithm} failed (rc={proc.returncode}): {stderr}",
                    agent_fix_hint="Inspect input mesh topology and aether_geo_worker stderr.",
                )
            return read_fmesh_bin(out_path)

    def remediate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        timeout_s: float = 60.0,
    ) -> FMeshData:
        """Run native topology remediation (``--algorithm remediate``)."""
        return self._run_algorithm(
            "remediate",
            vertices,
            faces,
            normals=normals,
            uvs=uvs,
            timeout_s=timeout_s,
        )

    def quadriflow(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        target_faces: int = 1500,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        timeout_s: float = 120.0,
    ) -> FMeshData:
        """Invoke native quadriflow remeshing (fail-closed ``NEG-100-08`` exit 50 on stub).

        Raises:
            NotImplementedStageError: Always raised with ``exit_code=50`` when the
                native worker binary is unavailable or exits non-zero as a contract stub.
        """
        try:
            return self._run_algorithm(
                "quadriflow",
                vertices,
                faces,
                normals=normals,
                uvs=uvs,
                extra_args=["--target-faces", str(int(target_faces))],
                timeout_s=timeout_s,
            )
        except BackendUnavailableError as exc:
            raise NotImplementedStageError(
                f"Native quadriflow worker unavailable: {exc}",
                agent_fix_hint=(
                    "Upstream quadriflow in aether_geo_worker is a contract stub. "
                    "Use Python Oracle quad_remesh(strict_native=False) or build worker."
                ),
            ) from exc
