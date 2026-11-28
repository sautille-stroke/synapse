"""KHR_gaussian_splatting glTF 2.0 (.glb / SPZ) encoder, decoder & .ply converter.

Implements Khronos ``KHR_gaussian_splatting`` binary glTF 2.0 (``.glb``)
serialization and deserialization, optional compact SPZ v2 payload encoding
via ``KHR_gaussian_splatting_compression_spz_2``, standard 3DGS ``.ply`` I/O,
and bidirectional ``.ply`` <-> ``.glb`` / ``.s3dgs`` / ``.npz`` conversion.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.metrics.image_quality import compute_psnr
from synapse.recon.gs_quantizer import (
    GaussianSplatCloud,
    _pack_smallest_three_quats,
    _unpack_smallest_three_quats,
    dequantize_3dgs,
    load_s3dgs,
    quantize_3dgs,
    render_3dgs_preview,
    save_s3dgs,
)

KHR_GS_EXTENSION = "KHR_gaussian_splatting"
KHR_GS_SPZ_EXTENSION = "KHR_gaussian_splatting_compression_spz_2"

GLTF_MAGIC = 0x46546C67  # b"glTF"
GLTF_VERSION = 2
GLTF_CHUNK_JSON = 0x4E4F534A  # b"JSON"
GLTF_CHUNK_BIN = 0x004E4942  # b"BIN\0"

GLTF_COMPONENT_FLOAT = 5126
GLTF_ARRAY_BUFFER = 34962
GLTF_POINTS_MODE = 0

SPZ_MAGIC = 0x5053474E  # b"NGSP"
SPZ_VERSION = 2

_SH_BAND_COUNTS: tuple[tuple[int, int], ...] = (
    (1, 3),
    (2, 5),
    (3, 7),
)


@dataclass(frozen=True)
class GltfSplatExportReport:
    """Summary returned when exporting a splat cloud to KHR_gaussian_splatting .glb."""

    output_path: Path
    num_gaussians: int
    sh_degree: int
    compressed_spz: bool
    raw_bytes: int
    encoded_bytes: int
    compression_ratio: float
    roundtrip_psnr_db: float


@dataclass(frozen=True)
class GltfConversionReport:
    """Summary returned by bidirectional 3DGS format conversions."""

    input_path: Path
    output_path: Path
    source_format: str
    target_format: str
    num_gaussians: int
    sh_degree: int
    compressed_spz: bool
    input_bytes: int
    output_bytes: int
    compression_ratio: float
    roundtrip_psnr_db: float


def _infer_sh_degree(sh_band_count: int) -> int:
    if sh_band_count <= 0:
        return 0
    if sh_band_count <= 3:
        return 1
    if sh_band_count <= 8:
        return 2
    return 3


def _quantize_affine_uint8(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantize an array to uint8 using global min/max affine range."""
    vals = np.asarray(arr, dtype=np.float32)
    vmin = float(np.min(vals)) if vals.size > 0 else 0.0
    vmax = float(np.max(vals)) if vals.size > 0 else 0.0
    span = max(vmax - vmin, 1e-8)
    scaled = np.clip(np.rint(((vals - vmin) / span) * 255.0), 0.0, 255.0)
    return scaled.astype(np.uint8), vmin, span


def _dequantize_affine_uint8(q: np.ndarray, vmin: float, span: float) -> np.ndarray:
    return (q.astype(np.float32) / 255.0) * span + vmin


def _quantize_affine_uint16(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize an (N, 3) array per-column to uint16 for sub-millimetre precision."""
    vals = np.asarray(arr, dtype=np.float32)
    vmin = np.min(vals, axis=0) if vals.shape[0] > 0 else np.zeros(3, dtype=np.float32)
    vmax = np.max(vals, axis=0) if vals.shape[0] > 0 else np.zeros(3, dtype=np.float32)
    span = np.maximum(vmax - vmin, 1e-8).astype(np.float32)
    scaled = np.clip(np.rint(((vals - vmin) / span) * 65535.0), 0.0, 65535.0)
    return scaled.astype("<u2"), vmin.astype(np.float32), span


def _dequantize_affine_uint16(
    q: np.ndarray,
    vmin: np.ndarray,
    span: np.ndarray,
) -> np.ndarray:
    return (q.astype(np.float32) / 65535.0) * span + vmin


def _normalize_quats_wxyz(rots: np.ndarray) -> np.ndarray:
    arr = np.asarray(rots, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    safe = np.where(norms > 1e-8, norms, 1.0)
    out = np.where(norms > 1e-8, arr / safe, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    return out.astype(np.float32)


def _compute_preview_psnr(orig: GaussianSplatCloud, decoded: GaussianSplatCloud) -> float:
    img_a = render_3dgs_preview(orig, width=64, height=64)
    img_b = render_3dgs_preview(decoded, width=64, height=64)
    if np.allclose(img_a, img_b, atol=1e-6):
        return 99.0
    psnr = float(compute_psnr(img_a, img_b, data_range=1.0))
    return min(psnr, 99.0) if np.isfinite(psnr) else 99.0


# ---------------------------------------------------------------------------
# Compact SPZ v2 Payload Codec (KHR_gaussian_splatting_compression_spz_2)
# ---------------------------------------------------------------------------


def encode_spz_payload(cloud: GaussianSplatCloud) -> bytes:
    """Encode a :class:`GaussianSplatCloud` into compact SPZ v2 bytes."""
    n = cloud.num_gaussians
    k_rest = cloud.sh_band_count
    sh_deg = _infer_sh_degree(k_rest)

    q_xyz, xyz_min, xyz_span = _quantize_affine_uint16(cloud.xyz)
    q_rots = _pack_smallest_three_quats(_normalize_quats_wxyz(cloud.rotations))
    q_sh_dc, dc_min, dc_span = _quantize_affine_uint8(cloud.sh_dc)
    q_scales, sc_min, sc_span = _quantize_affine_uint8(cloud.scales)
    q_opac, op_min, op_span = _quantize_affine_uint8(cloud.opacities)

    if cloud.sh_rest is not None and k_rest > 0:
        q_rest, rest_min, rest_span = _quantize_affine_uint8(cloud.sh_rest)
        rest_bytes = np.ascontiguousarray(q_rest).tobytes()
    else:
        rest_min, rest_span = 0.0, 1.0
        rest_bytes = b""

    exact_hdr = struct.pack("<IIIIff", SPZ_MAGIC, SPZ_VERSION, n, k_rest, rest_min, rest_span)
    bounds_bytes = struct.pack(
        "<12f",
        float(xyz_min[0]),
        float(xyz_min[1]),
        float(xyz_min[2]),
        float(xyz_span[0]),
        float(xyz_span[1]),
        float(xyz_span[2]),
        dc_min,
        dc_span,
        sc_min,
        sc_span,
        op_min,
        op_span,
    )
    body = b"".join(
        [
            bounds_bytes,
            np.ascontiguousarray(q_xyz).tobytes(),
            np.ascontiguousarray(q_rots.astype("<u4")).tobytes(),
            np.ascontiguousarray(q_sh_dc).tobytes(),
            np.ascontiguousarray(q_scales).tobytes(),
            np.ascontiguousarray(q_opac).tobytes(),
            rest_bytes,
        ]
    )
    _ = sh_deg
    compressed_body = zlib.compress(body, level=6)
    return exact_hdr + compressed_body


def decode_spz_payload(payload: bytes) -> GaussianSplatCloud:
    """Decode compact SPZ v2 bytes back into a :class:`GaussianSplatCloud`."""
    hdr_size = struct.calcsize("<IIIIff")
    if len(payload) < hdr_size:
        raise ContractValidationError(
            "Truncated SPZ payload header",
            agent_fix_hint="Re-export the glTF .glb with valid SPZ payload.",
        )
    magic, version, n, k_rest, rest_min, rest_span = struct.unpack_from("<IIIIff", payload, 0)
    if magic != SPZ_MAGIC or version != SPZ_VERSION:
        raise ContractValidationError(
            f"Unsupported SPZ header magic={magic:#x} version={version}",
            agent_fix_hint="Expected SPZ v2 NGSP container.",
        )

    try:
        raw_body = zlib.decompress(payload[hdr_size:])
    except zlib.error as exc:
        raise ContractValidationError(
            f"Corrupt SPZ compressed payload: {exc}",
            agent_fix_hint="Re-generate the .glb asset.",
        ) from exc

    bounds_size = struct.calcsize("<12f")
    bounds = struct.unpack_from("<12f", raw_body, 0)
    xyz_min = np.array(bounds[0:3], dtype=np.float32)
    xyz_span = np.array(bounds[3:6], dtype=np.float32)
    dc_min, dc_span, sc_min, sc_span, op_min, op_span = bounds[6:12]

    offset = bounds_size
    n_xyz = n * 3 * 2
    n_rot = n * 4
    n_dc = n * 3
    n_sc = n * 3
    n_op = n
    n_rest = n * k_rest * 3

    expected_total = offset + n_xyz + n_rot + n_dc + n_sc + n_op + n_rest
    if len(raw_body) != expected_total:
        raise ContractValidationError(
            f"SPZ body byte length mismatch: expected {expected_total}, got {len(raw_body)}",
            agent_fix_hint="Re-export the SPZ glTF file.",
        )

    q_xyz = np.frombuffer(raw_body, dtype="<u2", count=n * 3, offset=offset).reshape(n, 3)
    offset += n_xyz
    q_rots = np.frombuffer(raw_body, dtype="<u4", count=n, offset=offset)
    offset += n_rot
    q_dc = np.frombuffer(raw_body, dtype=np.uint8, count=n * 3, offset=offset).reshape(n, 3)
    offset += n_dc
    q_sc = np.frombuffer(raw_body, dtype=np.uint8, count=n * 3, offset=offset).reshape(n, 3)
    offset += n_sc
    q_op = np.frombuffer(raw_body, dtype=np.uint8, count=n, offset=offset)
    offset += n_op

    sh_rest: np.ndarray | None = None
    if k_rest > 0:
        q_rest = np.frombuffer(raw_body, dtype=np.uint8, count=n * k_rest * 3, offset=offset).reshape(
            n, k_rest, 3
        )
        sh_rest = _dequantize_affine_uint8(q_rest, rest_min, rest_span).astype(np.float32)

    return GaussianSplatCloud(
        xyz=_dequantize_affine_uint16(q_xyz, xyz_min, xyz_span).astype(np.float32),
        sh_dc=_dequantize_affine_uint8(q_dc, dc_min, dc_span).astype(np.float32),
        sh_rest=sh_rest,
        scales=_dequantize_affine_uint8(q_sc, sc_min, sc_span).astype(np.float32),
        rotations=_unpack_smallest_three_quats(q_rots),
        opacities=_dequantize_affine_uint8(q_op, op_min, op_span).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# glTF 2.0 Binary (.glb) + KHR_gaussian_splatting Encoder / Decoder
# ---------------------------------------------------------------------------


def _pad4(data: bytes, fill: bytes) -> bytes:
    rem = len(data) % 4
    if rem == 0:
        return data
    return data + (fill * (4 - rem))


def export_gltf_gaussian_splat(
    cloud: GaussianSplatCloud,
    output_path: str | Path,
    *,
    compress_spz: bool = False,
) -> tuple[Path, GltfSplatExportReport]:
    """Export a :class:`GaussianSplatCloud` to a valid glTF 2.0 ``.glb`` file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = cloud.num_gaussians
    k_rest = cloud.sh_band_count
    sh_deg = _infer_sh_degree(k_rest)

    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    attributes: dict[str, int] = {}
    bin_chunks: list[bytes] = []
    byte_offset = 0

    def _append_accessor(
        arr: np.ndarray,
        accessor_type: str,
        *,
        include_minmax: bool = False,
    ) -> int:
        nonlocal byte_offset
        f32 = np.ascontiguousarray(arr, dtype="<f4")
        raw = f32.tobytes()
        padded = _pad4(raw, b"\x00")
        bv_idx = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": byte_offset,
                "byteLength": len(raw),
                "target": GLTF_ARRAY_BUFFER,
            }
        )
        acc: dict[str, Any] = {
            "bufferView": bv_idx,
            "componentType": GLTF_COMPONENT_FLOAT,
            "count": n,
            "type": accessor_type,
        }
        if include_minmax and n > 0:
            acc["min"] = [float(x) for x in np.min(f32, axis=0).tolist()]
            acc["max"] = [float(x) for x in np.max(f32, axis=0).tolist()]
        acc_idx = len(accessors)
        accessors.append(acc)
        bin_chunks.append(padded)
        byte_offset += len(padded)
        return acc_idx

    # Convert Synapse WXYZ quaternions -> glTF KHR_gaussian_splatting XYZW order
    rots_wxyz = _normalize_quats_wxyz(cloud.rotations)
    rots_xyzw = np.concatenate([rots_wxyz[:, 1:4], rots_wxyz[:, 0:1]], axis=-1)

    ext_used = [KHR_GS_EXTENSION]
    khr_ext_obj: dict[str, Any] = {
        "kernel": "ellipse",
        "colorSpace": "srgb_rec709_display",
        "sortingMethod": "cameraDistance",
        "projection": "perspective",
    }

    if compress_spz:
        ext_used.append(KHR_GS_SPZ_EXTENSION)
        spz_bytes = encode_spz_payload(cloud)
        padded_spz = _pad4(spz_bytes, b"\x00")
        bv_idx = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(spz_bytes),
            }
        )
        bin_chunks.append(padded_spz)
        byte_offset = len(padded_spz)

        # Lightweight metadata accessor indices declared logically under primitive extension
        khr_ext_obj["extensions"] = {
            KHR_GS_SPZ_EXTENSION: {
                "bufferView": bv_idx,
                "numGaussians": n,
                "shDegree": sh_deg,
            }
        }
        # Record minimal POSITION bounds for downstream spatial culling without storing float32 positions
        if n > 0:
            khr_ext_obj["boundsMin"] = [float(x) for x in np.min(cloud.xyz, axis=0).tolist()]
            khr_ext_obj["boundsMax"] = [float(x) for x in np.max(cloud.xyz, axis=0).tolist()]
    else:
        attributes["POSITION"] = _append_accessor(cloud.xyz, "VEC3", include_minmax=True)
        attributes[f"{KHR_GS_EXTENSION}:ROTATION"] = _append_accessor(rots_xyzw, "VEC4")
        attributes[f"{KHR_GS_EXTENSION}:SCALE"] = _append_accessor(cloud.scales, "VEC3")
        attributes[f"{KHR_GS_EXTENSION}:OPACITY"] = _append_accessor(
            cloud.opacities.reshape(n, 1), "SCALAR"
        )
        attributes[f"{KHR_GS_EXTENSION}:SH_DEGREE_0_COEF_0"] = _append_accessor(
            cloud.sh_dc, "VEC3"
        )
        if cloud.sh_rest is not None and k_rest > 0:
            coef_cursor = 0
            for deg, deg_count in _SH_BAND_COUNTS:
                for m_idx in range(deg_count):
                    if coef_cursor >= k_rest:
                        break
                    attr_name = f"{KHR_GS_EXTENSION}:SH_DEGREE_{deg}_COEF_{m_idx}"
                    attributes[attr_name] = _append_accessor(
                        cloud.sh_rest[:, coef_cursor, :], "VEC3"
                    )
                    coef_cursor += 1

    bin_payload = b"".join(bin_chunks)
    primitive: dict[str, Any] = {
        "mode": GLTF_POINTS_MODE,
        "attributes": attributes,
        "extensions": {
            KHR_GS_EXTENSION: khr_ext_obj,
        },
    }
    gltf_doc: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "synapse.recon.gltf_splat",
        },
        "extensionsUsed": ext_used,
        "extensionsRequired": [KHR_GS_EXTENSION],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "gaussian_splats"}],
        "meshes": [{"name": "gaussian_splats", "primitives": [primitive]}],
        "buffers": [{"byteLength": len(bin_payload)}],
        "bufferViews": buffer_views,
    }
    if accessors:
        gltf_doc["accessors"] = accessors

    json_bytes = _pad4(
        json.dumps(gltf_doc, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        b" ",
    )
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_payload)
    header = struct.pack("<III", GLTF_MAGIC, GLTF_VERSION, total_len)
    json_chunk_hdr = struct.pack("<II", len(json_bytes), GLTF_CHUNK_JSON)
    bin_chunk_hdr = struct.pack("<II", len(bin_payload), GLTF_CHUNK_BIN)

    out.write_bytes(header + json_chunk_hdr + json_bytes + bin_chunk_hdr + bin_payload)

    decoded = load_gltf_gaussian_splat(out)
    psnr = _compute_preview_psnr(cloud, decoded)
    raw_bytes = cloud.raw_bytes
    encoded_bytes = out.stat().st_size
    ratio = float(raw_bytes) / float(max(encoded_bytes, 1))
    report = GltfSplatExportReport(
        output_path=out.resolve(),
        num_gaussians=n,
        sh_degree=sh_deg,
        compressed_spz=compress_spz,
        raw_bytes=raw_bytes,
        encoded_bytes=encoded_bytes,
        compression_ratio=ratio,
        roundtrip_psnr_db=psnr,
    )
    return out.resolve(), report


def _parse_glb(raw: bytes) -> tuple[dict[str, Any], bytes]:
    if len(raw) < 20:
        raise ContractValidationError(
            "Truncated .glb file header",
            agent_fix_hint="Provide a valid glTF 2.0 binary (.glb) file.",
        )
    magic, version, total_len = struct.unpack_from("<III", raw, 0)
    if magic != GLTF_MAGIC or version != GLTF_VERSION:
        raise ContractValidationError(
            f"Invalid GLB magic/version: magic={magic:#x}, version={version}",
            agent_fix_hint="Ensure input file is glTF 2.0 binary (.glb).",
        )
    if total_len > len(raw):
        raise ContractValidationError(
            f"GLB declared length {total_len} exceeds file size {len(raw)}",
            agent_fix_hint="Re-export the corrupted .glb file.",
        )

    offset = 12
    gltf_json: dict[str, Any] | None = None
    bin_data = b""
    while offset + 8 <= total_len:
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk_bytes = raw[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == GLTF_CHUNK_JSON:
            gltf_json = json.loads(chunk_bytes.decode("utf-8").rstrip(" \x00"))
        elif chunk_type == GLTF_CHUNK_BIN:
            bin_data = chunk_bytes

    if gltf_json is None:
        raise ContractValidationError(
            "GLB missing JSON chunk",
            agent_fix_hint="Provide a valid glTF 2.0 .glb with KHR_gaussian_splatting.",
        )
    return gltf_json, bin_data


def _read_accessor_array(
    gltf: dict[str, Any],
    bin_data: bytes,
    acc_idx: int,
) -> np.ndarray:
    accessors = gltf.get("accessors", [])
    buffer_views = gltf.get("bufferViews", [])
    acc = accessors[acc_idx]
    bv = buffer_views[int(acc["bufferView"])]
    comp_type = int(acc["componentType"])
    if comp_type != GLTF_COMPONENT_FLOAT:
        raise ContractValidationError(
            f"Unsupported glTF accessor componentType {comp_type}; expected FLOAT (5126)",
            agent_fix_hint="Export KHR_gaussian_splatting attributes as FLOAT or SPZ.",
        )
    acc_type = str(acc["type"])
    width_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    width = width_map[acc_type]
    count = int(acc["count"])
    bv_offset = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    arr = np.frombuffer(bin_data, dtype="<f4", count=count * width, offset=bv_offset)
    if width == 1:
        return np.asarray(arr, dtype=np.float32).copy()
    return np.asarray(arr.reshape(count, width), dtype=np.float32).copy()


def load_gltf_gaussian_splat(path: str | Path) -> GaussianSplatCloud:
    """Load a ``KHR_gaussian_splatting`` ``.glb`` asset into a :class:`GaussianSplatCloud`."""
    p = Path(path)
    if not p.is_file():
        raise AssetNotFoundError(
            f"glTF splat file not found: {p}",
            agent_fix_hint="Verify the .glb path exists before loading.",
        )
    gltf, bin_data = _parse_glb(p.read_bytes())
    meshes = gltf.get("meshes", [])
    if not meshes or not meshes[0].get("primitives"):
        raise ContractValidationError(
            "glTF file contains no mesh primitives",
            agent_fix_hint="Provide a glTF asset exported with KHR_gaussian_splatting.",
        )
    prim = meshes[0]["primitives"][0]
    khr_ext = prim.get("extensions", {}).get(KHR_GS_EXTENSION, {})

    # SPZ compressed extension payload
    spz_meta = khr_ext.get("extensions", {}).get(KHR_GS_SPZ_EXTENSION)
    if isinstance(spz_meta, dict) and "bufferView" in spz_meta:
        bv = gltf["bufferViews"][int(spz_meta["bufferView"])]
        bv_off = int(bv.get("byteOffset", 0))
        bv_len = int(bv["byteLength"])
        return decode_spz_payload(bin_data[bv_off : bv_off + bv_len])

    attrs = prim.get("attributes", {})
    req = ("POSITION", f"{KHR_GS_EXTENSION}:ROTATION", f"{KHR_GS_EXTENSION}:SCALE", f"{KHR_GS_EXTENSION}:OPACITY", f"{KHR_GS_EXTENSION}:SH_DEGREE_0_COEF_0")
    for key in req:
        if key not in attrs:
            raise ContractValidationError(
                f"glTF primitive missing required KHR_gaussian_splatting attribute '{key}'",
                agent_fix_hint="Export .glb using KHR_gaussian_splatting attribute conventions.",
            )

    xyz = _read_accessor_array(gltf, bin_data, int(attrs["POSITION"]))
    rots_xyzw = _read_accessor_array(
        gltf, bin_data, int(attrs[f"{KHR_GS_EXTENSION}:ROTATION"])
    )
    # glTF XYZW -> Synapse WXYZ
    rots_wxyz = np.concatenate([rots_xyzw[:, 3:4], rots_xyzw[:, 0:3]], axis=-1)
    scales = _read_accessor_array(
        gltf, bin_data, int(attrs[f"{KHR_GS_EXTENSION}:SCALE"])
    )
    opacities = _read_accessor_array(
        gltf, bin_data, int(attrs[f"{KHR_GS_EXTENSION}:OPACITY"])
    ).reshape(-1)
    sh_dc = _read_accessor_array(
        gltf, bin_data, int(attrs[f"{KHR_GS_EXTENSION}:SH_DEGREE_0_COEF_0"])
    )

    sh_rest_cols: list[np.ndarray] = []
    for deg, deg_count in _SH_BAND_COUNTS:
        for m_idx in range(deg_count):
            attr_name = f"{KHR_GS_EXTENSION}:SH_DEGREE_{deg}_COEF_{m_idx}"
            if attr_name in attrs:
                sh_rest_cols.append(
                    _read_accessor_array(gltf, bin_data, int(attrs[attr_name]))
                )

    sh_rest = np.stack(sh_rest_cols, axis=1).astype(np.float32) if sh_rest_cols else None
    return GaussianSplatCloud(
        xyz=xyz,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        scales=scales,
        rotations=rots_wxyz,
        opacities=opacities,
    )


# ---------------------------------------------------------------------------
# Standard 3DGS .ply Reader / Writer
# ---------------------------------------------------------------------------


def save_ply_3dgs(cloud: GaussianSplatCloud, output_path: str | Path) -> Path:
    """Save a :class:`GaussianSplatCloud` to standard binary little-endian 3DGS ``.ply``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = cloud.num_gaussians
    k_rest = cloud.sh_band_count

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    n_rest_scalars = k_rest * 3
    for i in range(n_rest_scalars):
        header_lines.append(f"property float f_rest_{i}")
    header_lines.extend(
        [
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header\n",
        ]
    )
    header_bytes = "\n".join(header_lines).encode("ascii")

    cols = [
        cloud.xyz.astype("<f4"),
        cloud.sh_dc.astype("<f4"),
    ]
    if cloud.sh_rest is not None and k_rest > 0:
        # Inria PLY stores f_rest in channel-major planar order: [N, 3, K] flattened
        cols.append(
            np.transpose(cloud.sh_rest, (0, 2, 1)).reshape(n, n_rest_scalars).astype("<f4")
        )
    cols.extend(
        [
            cloud.opacities.reshape(n, 1).astype("<f4"),
            cloud.scales.astype("<f4"),
            _normalize_quats_wxyz(cloud.rotations).astype("<f4"),
        ]
    )
    vertex_table = np.ascontiguousarray(np.concatenate(cols, axis=1), dtype="<f4")
    out.write_bytes(header_bytes + vertex_table.tobytes())
    return out.resolve()


def load_ply_3dgs(path: str | Path) -> GaussianSplatCloud:
    """Load a standard 3D Gaussian Splatting ``.ply`` file (binary LE or ASCII)."""
    p = Path(path)
    if not p.is_file():
        raise AssetNotFoundError(
            f"3DGS .ply file not found: {p}",
            agent_fix_hint="Verify the .ply path exists before conversion.",
        )

    raw = p.read_bytes()
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        marker = b"end_header\r\n"
        idx = raw.find(marker)
    if idx < 0:
        raise ContractValidationError(
            f"Invalid PLY file (missing end_header): {p}",
            agent_fix_hint="Ensure input file is a valid 3DGS .ply.",
        )

    hdr_text = raw[:idx].decode("ascii", errors="strict")
    body = raw[idx + len(marker) :]
    fmt = "ascii"
    num_vertices = 0
    props: list[tuple[str, str]] = []
    in_vertex = False

    for line in hdr_text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                num_vertices = int(parts[2])
        elif parts[0] == "property" and in_vertex and len(parts) >= 3:
            props.append((parts[1], parts[2]))

    if fmt not in ("binary_little_endian", "ascii"):
        raise ContractValidationError(
            f"Unsupported PLY format '{fmt}'",
            agent_fix_hint="Convert PLY to binary_little_endian or ascii.",
        )

    prop_names = [name for _, name in props]
    name_to_idx = {name: i for i, name in enumerate(prop_names)}
    required = (
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    for req in required:
        if req not in name_to_idx:
            raise ContractValidationError(
                f"3DGS PLY missing required property '{req}'",
                agent_fix_hint="Export standard 3DGS PLY with x/y/z, f_dc_*, opacity, scale_*, rot_*.",
            )

    if fmt == "binary_little_endian":
        type_map = {
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
            "uchar": "|u1",
            "uint8": "|u1",
            "int": "<i4",
            "int32": "<i4",
        }
        dt = np.dtype([(name, type_map.get(pt, "<f4")) for pt, name in props])
        structured = np.frombuffer(body, dtype=dt, count=num_vertices)

        def _col(name: str) -> np.ndarray:
            return np.asarray(structured[name], dtype=np.float32)

    else:
        rows = [
            line.split()
            for line in body.decode("ascii").splitlines()
            if line.strip()
        ][:num_vertices]
        arr = np.asarray(rows, dtype=np.float32)

        def _col(name: str) -> np.ndarray:
            return np.asarray(arr[:, name_to_idx[name]], dtype=np.float32)

    xyz = np.stack([_col("x"), _col("y"), _col("z")], axis=-1)
    sh_dc = np.stack([_col("f_dc_0"), _col("f_dc_1"), _col("f_dc_2")], axis=-1)
    scales = np.stack([_col("scale_0"), _col("scale_1"), _col("scale_2")], axis=-1)
    opacities = _col("opacity")
    rots = _normalize_quats_wxyz(
        np.stack([_col("rot_0"), _col("rot_1"), _col("rot_2"), _col("rot_3")], axis=-1)
    )

    f_rest_names = [n for n in prop_names if n.startswith("f_rest_")]
    f_rest_names.sort(key=lambda s: int(s.split("_")[-1]))
    sh_rest: np.ndarray | None = None
    if f_rest_names:
        k_rest = len(f_rest_names) // 3
        if k_rest > 0:
            planar = np.stack([_col(n) for n in f_rest_names[: k_rest * 3]], axis=-1)
            sh_rest = np.transpose(planar.reshape(num_vertices, 3, k_rest), (0, 2, 1)).astype(
                np.float32
            )

    return GaussianSplatCloud(
        xyz=xyz,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        scales=scales,
        rotations=rots,
        opacities=opacities,
    )


# ---------------------------------------------------------------------------
# Bidirectional Converter (.ply <-> .glb / .s3dgs / .npz)
# ---------------------------------------------------------------------------


def load_any_3dgs_cloud(path: str | Path) -> tuple[GaussianSplatCloud, str]:
    """Load a :class:`GaussianSplatCloud` from ``.ply``, ``.glb``, ``.s3dgs``, or ``.npz``."""
    p = Path(path)
    if not p.is_file():
        raise AssetNotFoundError(
            f"3DGS asset file not found: {p}",
            agent_fix_hint="Check the path passed to '--input'.",
        )
    ext = p.suffix.lower()
    if ext == ".ply":
        return load_ply_3dgs(p), "ply"
    if ext in (".glb", ".gltf"):
        return load_gltf_gaussian_splat(p), "glb"
    if ext == ".s3dgs":
        return load_s3dgs(p), "s3dgs"
    if ext == ".npz":
        with np.load(p, allow_pickle=False) as data:
            for req in ("xyz", "sh_dc", "scales", "rotations", "opacities"):
                if req not in data.files:
                    raise ContractValidationError(
                        f"Missing required 3DGS array '{req}' in {p}",
                        agent_fix_hint="Ensure archive defines xyz, sh_dc, scales, rotations, opacities.",
                    )
            sh_rest_arr = data["sh_rest"] if "sh_rest" in data.files else None
            if sh_rest_arr is not None and sh_rest_arr.ndim == 3 and sh_rest_arr.shape[1] == 0:
                sh_rest_arr = None
            return (
                GaussianSplatCloud(
                    xyz=np.asarray(data["xyz"], dtype=np.float32),
                    sh_dc=np.asarray(data["sh_dc"], dtype=np.float32),
                    sh_rest=(
                        np.asarray(sh_rest_arr, dtype=np.float32)
                        if sh_rest_arr is not None
                        else None
                    ),
                    scales=np.asarray(data["scales"], dtype=np.float32),
                    rotations=np.asarray(data["rotations"], dtype=np.float32),
                    opacities=np.asarray(data["opacities"], dtype=np.float32),
                ),
                "npz",
            )
    raise ContractValidationError(
        f"Unsupported 3DGS format extension '{p.suffix}'",
        agent_fix_hint="Use .ply, .glb, .s3dgs, or .npz.",
    )


def convert_3dgs_asset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    compress_spz: bool = False,
) -> GltfConversionReport:
    """Convert a 3DGS asset bidirectionally between ``.ply``, ``.glb``, ``.s3dgs``, and ``.npz``."""
    inp = Path(input_path)
    out = Path(output_path)
    cloud, src_fmt = load_any_3dgs_cloud(inp)
    in_bytes = inp.stat().st_size

    tgt_ext = out.suffix.lower()
    if tgt_ext == ".glb":
        resolved, _ = export_gltf_gaussian_splat(cloud, out, compress_spz=compress_spz)
        tgt_fmt = "glb"
    elif tgt_ext == ".ply":
        resolved = save_ply_3dgs(cloud, out)
        tgt_fmt = "ply"
    elif tgt_ext == ".s3dgs":
        q = quantize_3dgs(cloud, require_draco=False)
        resolved = save_s3dgs(q, out)
        tgt_fmt = "s3dgs"
    elif tgt_ext == ".npz":
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            xyz=cloud.xyz,
            sh_dc=cloud.sh_dc,
            sh_rest=(
                cloud.sh_rest
                if cloud.sh_rest is not None
                else np.zeros((cloud.num_gaussians, 0, 3), dtype=np.float32)
            ),
            scales=cloud.scales,
            rotations=cloud.rotations,
            opacities=cloud.opacities,
        )
        resolved = out.resolve()
        tgt_fmt = "npz"
    else:
        raise ContractValidationError(
            f"Unsupported 3DGS output extension '{out.suffix}'",
            agent_fix_hint="Use .glb, .ply, .s3dgs, or .npz as output.",
        )

    decoded, _ = load_any_3dgs_cloud(resolved)
    out_bytes = resolved.stat().st_size
    psnr = _compute_preview_psnr(cloud, decoded)
    ratio = float(in_bytes) / float(max(out_bytes, 1))
    _ = dequantize_3dgs
    return GltfConversionReport(
        input_path=inp.resolve(),
        output_path=resolved,
        source_format=src_fmt,
        target_format=tgt_fmt,
        num_gaussians=cloud.num_gaussians,
        sh_degree=_infer_sh_degree(cloud.sh_band_count),
        compressed_spz=compress_spz and tgt_fmt == "glb",
        input_bytes=in_bytes,
        output_bytes=out_bytes,
        compression_ratio=ratio,
        roundtrip_psnr_db=psnr,
    )
