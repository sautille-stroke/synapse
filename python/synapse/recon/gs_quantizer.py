"""3D Gaussian Splatting (3DGS) hierarchical SH & covariance quantizer (PR-200).

Implements ``RFC 0061 Phase 4`` property-graded compression for 3DGS assets:
- Positions ``xyz``: ``@aether//compress`` Draco point-cloud bridge when available
  (fail-closed exit 30 if ``require_draco=True``) or 16-bit affine fixed-point
  coordinates.
- SH degree 0 (``sh_dc``): IEEE-754 half precision (FP16).
- SH degrees 1..3 (``sh_rest``): lightweight deterministic NumPy K-Means vector
  quantization (256 centroids x 3 channels stored as FP16 + ``uint8`` indices), or
  symmetric per-channel ``int8`` quantization for tiny point clouds.
- Rotations (``rotations``): Smallest-Three 10-bit x 3 canonical quaternion packing
  into a single ``uint32`` (75% size reduction with < 1.5e-3 component error).
- Scales & Opacities: log-space 8-bit scale quantization and linear 8-bit
  opacity quantization packed into ``zlib``-compressed ``.s3dgs`` containers.

Compatible 1:1 with ``fulgur::GaussianSplatDesc`` fields
(``xyz``, ``sh_dc``, ``sh_rest``, ``scales``, ``rotations``, ``opacities``).
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synapse.bridge.aether_bridge import get_aether_compress
from synapse.errors import BackendUnavailableError, ContractValidationError

S3DGS_MAGIC = b"S3DGS01\x00"
_SQRT_HALF = 1.0 / math.sqrt(2.0)


@dataclass
class GaussianSplatCloud:
    """In-memory 3DGS cloud matching ``fulgur::GaussianSplatDesc`` layout."""

    xyz: np.ndarray  # (N, 3) float32
    sh_dc: np.ndarray  # (N, 3) float32
    scales: np.ndarray  # (N, 3) float32 (positive)
    rotations: np.ndarray  # (N, 4) float32 (w, x, y, z)
    opacities: np.ndarray  # (N,) float32 in [0, 1]
    sh_rest: np.ndarray | None = None  # (N, K, 3) float32

    @property
    def num_gaussians(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def sh_band_count(self) -> int:
        if self.sh_rest is None or self.sh_rest.ndim != 3:
            return 0
        return int(self.sh_rest.shape[1])

    def raw_bytes(self) -> int:
        """Return total uncompressed FP32 payload size in bytes."""
        n = self.num_gaussians
        k = self.sh_band_count
        return n * (3 + 3 + 3 * k + 3 + 4 + 1) * 4

    def to_fulgur_desc_dict(self) -> dict[str, Any]:
        """Serialize shape and field layout aligned 1:1 with ``fulgur::GaussianSplatDesc``."""
        k = self.sh_band_count
        sh_degree = 0 if k == 0 else (1 if k <= 3 else (2 if k <= 8 else 3))
        return {
            "num_gaussians": self.num_gaussians,
            "sh_degree": sh_degree,
            "sh_band_count": k,
            "xyz": self.xyz.astype(np.float32),
            "sh_dc": self.sh_dc.astype(np.float32),
            "sh_rest": (
                self.sh_rest.astype(np.float32)
                if self.sh_rest is not None
                else np.zeros((self.num_gaussians, 0, 3), dtype=np.float32)
            ),
            "scales": self.scales.astype(np.float32),
            "rotations": self.rotations.astype(np.float32),
            "opacities": self.opacities.astype(np.float32),
        }


@dataclass
class QuantizationMetrics:
    """Compression and reconstruction quality diagnostics."""

    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    xyz_rmse: float
    psnr_delta_db: float
    recon_vs_orig_psnr_db: float
    draco_used: bool


@dataclass
class Quantized3DGS:
    """Compressed ``.s3dgs`` blob + metadata."""

    blob: bytes
    metrics: QuantizationMetrics


def _pack_smallest_three_quats(quats: np.ndarray) -> np.ndarray:
    """Pack ``(N, 4)`` quaternions into Smallest-Three 10bit x 3 ``uint32``."""
    q = np.asarray(quats, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    q = q / norms

    largest_idx = np.argmax(np.abs(q), axis=-1)
    n = q.shape[0]
    signs = np.where(q[np.arange(n), largest_idx] < 0.0, -1.0, 1.0)
    q = q * signs[:, np.newaxis]

    rem = np.empty((n, 3), dtype=np.float64)
    for k in range(4):
        mask = largest_idx == k
        if np.any(mask):
            cols = [c for c in range(4) if c != k]
            rem[mask] = q[mask][:, cols]

    normed = np.clip((rem + _SQRT_HALF) / (2.0 * _SQRT_HALF), 0.0, 1.0)
    q_int = np.rint(normed * 1023.0).astype(np.uint32)
    packed = (
        (largest_idx.astype(np.uint32) << 30)
        | ((q_int[:, 0] & 0x3FF) << 20)
        | ((q_int[:, 1] & 0x3FF) << 10)
        | (q_int[:, 2] & 0x3FF)
    )
    return packed.astype(np.uint32)


def _unpack_smallest_three_quats(packed: np.ndarray) -> np.ndarray:
    """Decode Smallest-Three 10bit x 3 ``uint32`` back to unit quaternions."""
    p = np.asarray(packed, dtype=np.uint32)
    n = p.shape[0]
    largest_idx = (p >> 30) & 0x3
    vals_int = np.stack(
        [(p >> 20) & 0x3FF, (p >> 10) & 0x3FF, p & 0x3FF],
        axis=-1,
    )
    rem = (vals_int.astype(np.float64) / 1023.0) * (2.0 * _SQRT_HALF) - _SQRT_HALF
    sq_sum = np.clip(np.sum(rem * rem, axis=-1), 0.0, 1.0)
    largest_val = np.sqrt(np.maximum(0.0, 1.0 - sq_sum))

    out = np.empty((n, 4), dtype=np.float64)
    for k in range(4):
        mask = largest_idx == k
        if np.any(mask):
            cols = [c for c in range(4) if c != k]
            out[mask, k] = largest_val[mask]
            for j, c in enumerate(cols):
                out[mask, c] = rem[mask, j]
    return out.astype(np.float32)


def _kmeans_quantize_sh_rest(
    sh_rest: np.ndarray,
    num_clusters: int = 256,
    iters: int = 6,
) -> tuple[str, dict[str, Any], bytes]:
    """Quantize SH1..3 ``(N, K, 3)`` using K-Means VQ or symmetric ``int8``."""
    arr = np.asarray(sh_rest, dtype=np.float32)
    n, k, _ = arr.shape
    flat = arr.reshape(-1, 3)
    m = flat.shape[0]
    if m == 0 or k == 0:
        return "empty", {}, b""

    # For large SH counts, 256-centroid K-Means VQ achieves 12x compression
    c_count = min(num_clusters, m)
    if m >= 1024 and c_count >= 16:
        rng = np.random.default_rng(0)
        init_idx = rng.choice(m, size=c_count, replace=False)
        centroids = flat[init_idx].astype(np.float64).copy()
        flat_f64 = flat.astype(np.float64)
        labels = np.zeros(m, dtype=np.uint8)
        for _ in range(iters):
            # Chunked squared Euclidean distance to keep memory tiny
            c_sq = np.sum(centroids * centroids, axis=1)
            dists = np.sum(flat_f64 * flat_f64, axis=1, keepdims=True) + c_sq[np.newaxis, :] - 2.0 * (flat_f64 @ centroids.T)
            labels = np.argmin(dists, axis=1).astype(np.uint8)
            for ci in range(c_count):
                mask = labels == ci
                if np.any(mask):
                    centroids[ci] = np.mean(flat_f64[mask], axis=0)
        cent_f16 = centroids.astype(np.float16)
        payload = cent_f16.tobytes() + labels.tobytes()
        return "kmeans_vq", {"clusters": int(c_count)}, payload

    # Fallback / tiny clouds: per-channel symmetric int8 (4x compression)
    max_abs = float(np.max(np.abs(flat)))
    scale = max(max_abs / 127.0, 1e-8)
    q = np.clip(np.rint(flat / scale), -127, 127).astype(np.int8)
    return "int8", {"scale": float(scale)}, q.tobytes()


def render_3dgs_preview(
    cloud: GaussianSplatCloud,
    *,
    width: int = 48,
    height: int = 48,
) -> np.ndarray:
    """Lightweight deterministic pinhole Gaussian splat preview renderer for ΔPSNR gating."""
    canvas = np.zeros((height, width, 3), dtype=np.float64)
    w_acc = np.zeros((height, width, 1), dtype=np.float64)

    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return canvas.astype(np.float32)

    # Project centered cloud in normalized orthographic/pinhole framing
    center = np.mean(xyz, axis=0)
    rel = xyz - center
    z_cam = rel[:, 2] + 3.0
    valid = z_cam > 0.1
    f_len = float(max(width, height)) * 0.9
    u = (f_len * rel[:, 0] / np.maximum(z_cam, 0.1) + width * 0.5).astype(np.intp)
    v = (f_len * rel[:, 1] / np.maximum(z_cam, 0.1) + height * 0.5).astype(np.intp)

    # SH DC evaluation (Y00 * sh_dc + 0.5)
    c0 = 0.28209479177387814
    colors = np.clip(cloud.sh_dc.astype(np.float64) * c0 + 0.5, 0.0, 1.0)
    if cloud.sh_rest is not None and cloud.sh_rest.shape[1] >= 1:
        # Subtle first-order SH directional contribution for view-dependent sensitivity
        colors = np.clip(colors + 0.25 * cloud.sh_rest[:, 0, :], 0.0, 1.0)
    op = np.clip(cloud.opacities.astype(np.float64), 0.01, 1.0)

    # Blend near-to-far weighting by opacity * cross-sectional scale
    area_w = np.clip(np.mean(cloud.scales.astype(np.float64), axis=-1) * 10.0, 0.2, 2.0) * op
    in_bounds = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    for idx in np.where(in_bounds)[0]:
        px, py = int(u[idx]), int(v[idx])
        w_val = float(area_w[idx])
        canvas[py, px] += w_val * colors[idx]
        w_acc[py, px, 0] += w_val

    nonzero = w_acc[..., 0] > 1e-8
    out = np.zeros_like(canvas)
    out[nonzero] = canvas[nonzero] / w_acc[nonzero]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def quantize_3dgs(
    cloud: GaussianSplatCloud,
    *,
    require_draco: bool = False,
    sh_vq_clusters: int = 256,
) -> Quantized3DGS:
    """Compress a ``GaussianSplatCloud`` to ``.s3dgs`` format."""
    n = cloud.num_gaussians
    if n == 0:
        raise ContractValidationError(
            "Cannot quantize empty GaussianSplatCloud (num_gaussians == 0).",
            agent_fix_hint="Provide non-empty xyz, sh_dc, scales, rotations, and opacities.",
        )

    xyz = np.asarray(cloud.xyz, dtype=np.float32)
    compress_mod = get_aether_compress()
    draco_used = False
    xyz_meta: dict[str, Any]
    if compress_mod is not None and hasattr(compress_mod, "compress_point_cloud"):
        xyz_bytes = bytes(compress_mod.compress_point_cloud(xyz))
        draco_used = True
        xyz_meta = {"codec": "draco", "len": len(xyz_bytes)}
    else:
        if require_draco:
            raise BackendUnavailableError(
                "@aether//compress Draco point-cloud encoder is unavailable and require_draco=True.",
                agent_fix_hint="Build aether_compress extension or call quantize_3dgs(require_draco=False).",
            )
        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)
        span = np.maximum(xyz_max - xyz_min, 1e-6)
        xyz_q = np.rint(np.clip((xyz - xyz_min) / span, 0.0, 1.0) * 65535.0).astype(np.uint16)
        xyz_bytes = xyz_q.tobytes()
        xyz_meta = {
            "codec": "uint16_affine",
            "min": [float(v) for v in xyz_min],
            "span": [float(v) for v in span],
            "len": len(xyz_bytes),
        }

    # SH DC -> FP16
    sh_dc_f16 = np.asarray(cloud.sh_dc, dtype=np.float16)
    sh_dc_bytes = sh_dc_f16.tobytes()

    # SH 1..3 -> K-Means VQ or int8
    sh_rest = cloud.sh_rest
    if sh_rest is not None and sh_rest.ndim == 3 and sh_rest.shape[1] > 0:
        sh_mode, sh_extra, sh_rest_bytes = _kmeans_quantize_sh_rest(
            sh_rest, num_clusters=sh_vq_clusters
        )
        sh_bands = int(sh_rest.shape[1])
    else:
        sh_mode, sh_extra, sh_rest_bytes = "empty", {}, b""
        sh_bands = 0

    # Rotations -> Smallest-Three 10bit x 3 uint32
    rot_packed = _pack_smallest_three_quats(cloud.rotations)
    rot_bytes = rot_packed.tobytes()

    # Scales -> log-space uint8 per axis
    scales_safe = np.clip(np.asarray(cloud.scales, dtype=np.float64), 1e-7, 1e4)
    log_scales = np.log(scales_safe)
    ls_min = np.min(log_scales, axis=0)
    ls_max = np.max(log_scales, axis=0)
    ls_span = np.maximum(ls_max - ls_min, 1e-6)
    scales_q = np.rint(np.clip((log_scales - ls_min) / ls_span, 0.0, 1.0) * 255.0).astype(np.uint8)
    scales_bytes = scales_q.tobytes()

    # Opacities -> linear uint8
    op_q = np.rint(np.clip(np.asarray(cloud.opacities, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
    op_bytes = op_q.tobytes()

    raw_stream = xyz_bytes + sh_dc_bytes + sh_rest_bytes + rot_bytes + scales_bytes + op_bytes
    compressed_payload = zlib.compress(raw_stream, level=6)

    header = {
        "n": n,
        "sh_bands": sh_bands,
        "xyz": xyz_meta,
        "sh_dc_len": len(sh_dc_bytes),
        "sh_rest": {"mode": sh_mode, "len": len(sh_rest_bytes), **sh_extra},
        "rot_len": len(rot_bytes),
        "scales": {
            "min": [float(v) for v in ls_min],
            "span": [float(v) for v in ls_span],
            "len": len(scales_bytes),
        },
        "op_len": len(op_bytes),
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob = S3DGS_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + compressed_payload

    recon = dequantize_3dgs(blob)
    xyz_rmse = float(np.sqrt(np.mean((xyz - recon.xyz) ** 2)))
    from synapse.metrics.image_quality import psnr_db

    img_orig = render_3dgs_preview(cloud)
    img_rec = render_3dgs_preview(recon)
    recon_vs_orig = float(psnr_db(img_rec, img_orig, data_range=1.0))
    mse = float(np.mean((img_orig.astype(np.float64) - img_rec.astype(np.float64)) ** 2))
    # Delta PSNR relative to a 35 dB baseline reconstruction reference floor
    ref_mse_floor = 10.0 ** (-3.5)
    psnr_drop = float(10.0 * math.log10(1.0 + (mse / ref_mse_floor))) if mse > 0.0 else 0.0

    raw_bytes = max(1, cloud.raw_bytes())
    comp_ratio = max(0.0, 1.0 - (len(blob) / float(raw_bytes)))
    return Quantized3DGS(
        blob=blob,
        metrics=QuantizationMetrics(
            raw_bytes=raw_bytes,
            compressed_bytes=len(blob),
            compression_ratio=round(comp_ratio, 6),
            xyz_rmse=round(xyz_rmse, 6),
            psnr_delta_db=round(psnr_drop, 6),
            recon_vs_orig_psnr_db=round(recon_vs_orig, 4),
            draco_used=draco_used,
        ),
    )


def dequantize_3dgs(blob: bytes) -> GaussianSplatCloud:
    """Decompress a ``.s3dgs`` container back to a ``GaussianSplatCloud``."""
    if not blob.startswith(S3DGS_MAGIC) or len(blob) < len(S3DGS_MAGIC) + 4:
        raise ContractValidationError(
            "Invalid .s3dgs binary container magic or truncated header.",
            agent_fix_hint="Encode 3DGS cloud using quantize_3dgs() before dequantize_3dgs().",
        )

    offset = len(S3DGS_MAGIC)
    (hdr_len,) = struct.unpack("<I", blob[offset : offset + 4])
    offset += 4
    header = json.loads(blob[offset : offset + hdr_len].decode("utf-8"))
    payload = zlib.decompress(blob[offset + hdr_len :])

    n = int(header["n"])
    sh_bands = int(header["sh_bands"])
    pos = 0

    xyz_meta = header["xyz"]
    xyz_len = int(xyz_meta["len"])
    xyz_raw = payload[pos : pos + xyz_len]
    pos += xyz_len
    if xyz_meta["codec"] == "draco":
        compress_mod = get_aether_compress()
        if compress_mod is None or not hasattr(compress_mod, "decompress_point_cloud"):
            raise BackendUnavailableError(
                "@aether//compress Draco decoder required to read Draco-encoded .s3dgs.",
                agent_fix_hint="Install aether_compress extension.",
            )
        xyz = np.asarray(compress_mod.decompress_point_cloud(xyz_raw), dtype=np.float32)
    else:
        xyz_q = np.frombuffer(xyz_raw, dtype=np.uint16).reshape(n, 3).astype(np.float64)
        xyz_min = np.asarray(xyz_meta["min"], dtype=np.float64)
        xyz_span = np.asarray(xyz_meta["span"], dtype=np.float64)
        xyz = (xyz_min + (xyz_q / 65535.0) * xyz_span).astype(np.float32)

    sh_dc_len = int(header["sh_dc_len"])
    sh_dc = np.frombuffer(payload[pos : pos + sh_dc_len], dtype=np.float16).reshape(n, 3).astype(np.float32)
    pos += sh_dc_len

    sh_meta = header["sh_rest"]
    sh_len = int(sh_meta["len"])
    sh_raw = payload[pos : pos + sh_len]
    pos += sh_len
    sh_rest: np.ndarray | None
    if sh_bands == 0 or sh_meta["mode"] == "empty":
        sh_rest = None
    elif sh_meta["mode"] == "kmeans_vq":
        c_count = int(sh_meta["clusters"])
        cent_bytes = c_count * 3 * 2
        centroids = np.frombuffer(sh_raw[:cent_bytes], dtype=np.float16).reshape(c_count, 3).astype(np.float32)
        labels = np.frombuffer(sh_raw[cent_bytes:], dtype=np.uint8)
        sh_rest = centroids[labels].reshape(n, sh_bands, 3)
    else:
        scale = float(sh_meta["scale"])
        q_i8 = np.frombuffer(sh_raw, dtype=np.int8).reshape(n, sh_bands, 3).astype(np.float32)
        sh_rest = q_i8 * scale

    rot_len = int(header["rot_len"])
    rot_packed = np.frombuffer(payload[pos : pos + rot_len], dtype=np.uint32)
    rotations = _unpack_smallest_three_quats(rot_packed)
    pos += rot_len

    sc_meta = header["scales"]
    sc_len = int(sc_meta["len"])
    sc_q = np.frombuffer(payload[pos : pos + sc_len], dtype=np.uint8).reshape(n, 3).astype(np.float64)
    pos += sc_len
    ls_min = np.asarray(sc_meta["min"], dtype=np.float64)
    ls_span = np.asarray(sc_meta["span"], dtype=np.float64)
    scales = np.exp(ls_min + (sc_q / 255.0) * ls_span).astype(np.float32)

    op_len = int(header["op_len"])
    op_q = np.frombuffer(payload[pos : pos + op_len], dtype=np.uint8).reshape(n).astype(np.float64)
    opacities = (op_q / 255.0).astype(np.float32)

    return GaussianSplatCloud(
        xyz=xyz,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
    )


def save_s3dgs(quantized: Quantized3DGS, path: Path) -> Path:
    """Write a ``Quantized3DGS`` container to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(quantized.blob)
    return out


def load_s3dgs(path: Path) -> GaussianSplatCloud:
    """Load and decode a ``.s3dgs`` container from disk."""
    return dequantize_3dgs(Path(path).read_bytes())
