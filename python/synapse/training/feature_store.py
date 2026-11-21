"""Offline latent pre-encoding and WebDataset-compatible shard builder.

Provides Protocol classes for pluggable latent/text encoders,
deterministic test implementations, a ``FeatureStoreBuilder`` that
pre-encodes video+caption samples into ``.tar`` shards, and a streaming
iterator ``iter_feature_shards()`` for reading them back.

Heavy dependencies are lazy-imported inside methods.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from synapse.errors import AssetNotFoundError, ContractValidationError

logger = logging.getLogger(__name__)

_SAMPLE_ID_REGEX = re.compile(r"^[A-Za-z0-9_\-.]+$")

# -------------------------------------------------------------------
# Structured caption formatter
# -------------------------------------------------------------------


def format_structured_caption(
    trigger_token: str,
    subject_desc: str,
    lighting_env: str,
    camera_motion: str,
    art_style_tag: str,
) -> str:
    """Build a five-segment deterministic caption string.

    Format::

        "{trigger_token}, {subject_desc}, {lighting_env},
         {camera_motion}, {art_style_tag}"

    Args:
        trigger_token: LoRA trigger word (e.g. ``"sks"``).
        subject_desc: Subject description.
        lighting_env: Lighting / environment note.
        camera_motion: Camera motion label.
        art_style_tag: Art style tag.

    Returns:
        Concatenated caption string.

    Raises:
        ContractValidationError: If any field is empty.
    """
    fields = {
        "trigger_token": trigger_token,
        "subject_desc": subject_desc,
        "lighting_env": lighting_env,
        "camera_motion": camera_motion,
        "art_style_tag": art_style_tag,
    }
    for name, value in fields.items():
        if not value or not value.strip():
            raise ContractValidationError(
                f"Caption field '{name}' must not be empty",
                agent_fix_hint=(f"Provide a non-empty value for '{name}'."),
            )

    return f"{trigger_token}, {subject_desc}, {lighting_env}, {camera_motion}, {art_style_tag}"


# -------------------------------------------------------------------
# Encoder protocols
# -------------------------------------------------------------------


@runtime_checkable
class LatentEncoderProtocol(Protocol):
    """Protocol for video-to-latent encoders."""

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode ``[T, H, W, C]`` uint8 to latent tensor.

        Returns:
            ``[C_latent, tau, H/8, W/8]`` float32 latent.
        """
        ...


@runtime_checkable
class TextEncoderProtocol(Protocol):
    """Protocol for text-to-embedding encoders."""

    def encode(self, text: str) -> np.ndarray:
        """Encode text to ``[D]`` float32 embedding vector."""
        ...


# -------------------------------------------------------------------
# Deterministic test encoders
# -------------------------------------------------------------------


class LinearProjectionEncoder:
    """Deterministic latent encoder for CI/testing.

    Down-samples spatially by 8x (average pooling) and temporally
    by 4x, then projects to 16 latent channels via a fixed random
    matrix seeded with 42.

    Args:
        latent_channels: Number of output latent channels.
        seed: RNG seed for the projection matrix.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        seed: int = 42,
    ) -> None:
        self._channels = latent_channels
        self._seed = seed

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode ``[T, H, W, C]`` uint8 frames.

        Returns:
            ``[C_latent, tau, H//8, W//8]`` float32.
        """
        frames = np.asarray(frames, dtype=np.float32) / 255.0
        t, h, w, c = frames.shape

        # Temporal compression: 4x
        tau = (t - 1) // 4 + 1
        temporal = np.zeros(
            (tau, h, w, c),
            dtype=np.float32,
        )
        for i in range(tau):
            src_start = i * 4
            src_end = min(src_start + 4, t)
            temporal[i] = frames[src_start:src_end].mean(axis=0)

        # Spatial compression: 8x average pooling
        h8 = h // 8
        w8 = w // 8
        spatial = np.zeros(
            (tau, h8, w8, c),
            dtype=np.float32,
        )
        for yi in range(h8):
            for xi in range(w8):
                block = temporal[
                    :,
                    yi * 8 : (yi + 1) * 8,
                    xi * 8 : (xi + 1) * 8,
                    :,
                ]
                spatial[:, yi, xi, :] = block.mean(axis=(1, 2))

        # Linear projection: [C] -> [C_latent]
        rng = np.random.RandomState(self._seed)
        proj = rng.randn(c, self._channels).astype(np.float32)
        proj *= 0.1

        # spatial: [tau, h8, w8, C] -> [tau, h8, w8, C_latent]
        flat = spatial.reshape(-1, c)
        projected = (flat @ proj).reshape(
            tau,
            h8,
            w8,
            self._channels,
        )

        # Rearrange to [C_latent, tau, h8, w8]
        return np.asarray(projected.transpose(3, 0, 1, 2))


class LinearProjectionTextEncoder:
    """Deterministic text encoder for CI/testing.

    Hashes the input text to produce a fixed-length float32 vector.

    Args:
        embed_dim: Output embedding dimension.
    """

    def __init__(self, embed_dim: int = 512) -> None:
        self._dim = embed_dim

    def encode(self, text: str) -> np.ndarray:
        """Encode *text* to ``[D]`` float32 embedding.

        Uses SHA-256 hash expansion for determinism.
        """
        digest = hashlib.sha256(
            text.encode("utf-8"),
        ).digest()

        # Expand hash to fill embed_dim floats
        result = np.zeros(self._dim, dtype=np.float32)
        hash_floats = (
            np.frombuffer(
                digest,
                dtype=np.uint8,
            ).astype(np.float32)
            / 255.0
        )

        n_hash = len(hash_floats)
        for i in range(self._dim):
            idx = i % n_hash
            # Combine with position for variation
            result[i] = hash_floats[idx] * (0.5 + 0.5 * np.sin(float(i) * 0.1))

        return result


# -------------------------------------------------------------------
# FeatureStoreBuilder
# -------------------------------------------------------------------

_BUFFERED_SAMPLE = tuple[
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    str,
]


class FeatureStoreBuilder:
    """Build WebDataset-compatible ``.tar`` shards from pre-encoded samples.

    Args:
        latent_encoder: Encoder for video frames -> latent tensor.
        text_encoder: Encoder for caption text -> embedding.
        max_shard_bytes: Maximum bytes per shard (default 2 GiB).
        streaming: If ``True``, write samples directly to disk shards
            instead of buffering in memory.
        output_dir: **Required when ``streaming=True``**.  Directory for
            streaming shard files.  Enables true O(1)-memory ingestion
            without calling ``build_shards`` first.
    """

    def __init__(
        self,
        latent_encoder: LatentEncoderProtocol,
        text_encoder: TextEncoderProtocol,
        max_shard_bytes: int = 2 * 1024**3,
        streaming: bool = False,
        output_dir: Path | str | None = None,
    ) -> None:
        self._lat_enc = latent_encoder
        self._txt_enc = text_encoder
        self._max_shard = max_shard_bytes
        self._samples: list[_BUFFERED_SAMPLE] = []
        self.rejected_count: int = 0
        self._streaming = streaming
        self._stream_tar: tarfile.TarFile | None = None
        self._stream_path: Path | None = None
        self._stream_shard_idx: int = 0
        self._stream_shard_size: int = 0
        self._stream_shard_paths: list[Path] = []
        self._stream_output_dir: Path | None = None

        if output_dir is not None:
            self._stream_output_dir = Path(output_dir)
            self._stream_output_dir.mkdir(parents=True, exist_ok=True)

    def add_sample(
        self,
        sample_id: str,
        video_frames: np.ndarray,
        caption: str,
        fgpp_cond: np.ndarray | None = None,
        camera_matrices: np.ndarray | None = None,
        source_sha256: str = "",
    ) -> None:
        """Pre-encode and buffer a training sample.

        Applies 5 quality ingestion gates before encoding.  If any gate
        fails the sample is silently skipped and ``rejected_count`` is
        incremented.

        Args:
            sample_id: Unique sample identifier.
            video_frames: ``[T, H, W, 3]`` uint8 frames.
            caption: Caption text to encode.
            fgpp_cond: Optional FGPP condition tensor.  Accepts either
                raw ``[T, H, W, 15]`` FGPP pixel tensors **or**
                pre-built ``[7, T, H, W]`` Wan control tensors.  The
                two layouts are distinguished by shape (see
                ``_validate_ingestion_quality``).
            camera_matrices: Optional ``[T, 4, 4]`` camera-to-world
                matrices.
            source_sha256: Optional source sequence SHA-256.
        """
        if (
            not sample_id
            or not _SAMPLE_ID_REGEX.match(sample_id)
            or ".." in sample_id
            or "/" in sample_id
            or "\\" in sample_id
        ):
            raise ContractValidationError(
                f"Invalid sample_id '{sample_id}': must be alphanumeric/dashes/underscores with no path traversal",
                agent_fix_hint="Use a valid sample_id without slashes or '..'.",
            )

        passed, reason = self._validate_ingestion_quality(
            {
                "sample_id": sample_id,
                "video_frames": video_frames,
                "fgpp_cond": fgpp_cond,
                "camera_matrices": camera_matrices,
            }
        )
        if not passed:
            self.rejected_count += 1
            logger.warning(
                "Sample '%s' rejected by ingestion gate: %s",
                sample_id,
                reason,
            )
            return

        latent = self._lat_enc.encode(video_frames)
        prompt = self._txt_enc.encode(caption)

        if self._streaming and self._stream_output_dir is not None:
            # Streaming mode: write directly to tar shard, no buffering
            entries = self._serialize_sample(
                sample_id, latent, prompt, fgpp_cond, source_sha256,
            )
            sample_size = sum(len(d) for _, d in entries)
            # Rotate shard if needed
            if (
                self._stream_tar is not None
                and self._stream_shard_size > 0
                and self._stream_shard_size + sample_size > self._max_shard
            ):
                self._stream_tar.close()
                self._stream_tar = None
            if self._stream_tar is None:
                name = f"shard-{self._stream_shard_idx:05d}.tar"
                self._stream_path = self._stream_output_dir / name
                self._stream_tar = tarfile.open(self._stream_path, "w")  # noqa: SIM115
                self._stream_shard_idx += 1
                self._stream_shard_paths.append(self._stream_path)
                self._stream_shard_size = 0
            for entry_name, data in entries:
                info = tarfile.TarInfo(name=entry_name)
                info.size = len(data)
                self._stream_tar.addfile(info, io.BytesIO(data))
            self._stream_shard_size += sample_size
        else:
            self._samples.append(
                (sample_id, latent, prompt, fgpp_cond, source_sha256),
            )

    # ── quality ingestion gates ────────────────────────────────────

    @staticmethod
    def _is_wan_control_layout(arr: np.ndarray) -> bool:
        """Return *True* when *arr* looks like a ``[7, T, H, W]`` Wan control tensor (C=7, T>=1)."""
        if arr.ndim != 4:
            return False
        c, t, h, w = arr.shape
        return bool(c == 7 and t >= 1 and h >= 1 and w >= 1 and arr.shape[-1] != 15)

    @staticmethod
    def _validate_ingestion_quality(
        sample: dict[str, Any],
    ) -> tuple[bool, str]:
        """Apply 5 quality pre-filters before ingesting a sample.

        Accepts two distinct ``fgpp_cond`` layouts:

        - **Raw FGPP**: ``[T, H, W, 15]`` — 15-scalar channel tensor per
          ``fgpp_types.h``.  Gates 1-4 apply pixel-level checks;
          Gate 5 requires a separate ``camera_matrices`` ``[T, 4, 4]``
          entry in *sample*.
        - **Wan control**: ``[7, T, H, W]`` — pre-built 7-channel
          adapter tensor.  Pixel-level Gates 1-4 are skipped because
          the channel semantics no longer match raw FGPP columns.
          Gate 5 still runs when ``camera_matrices`` is supplied.

        Gates:
            1. Foreground coverage ≥ 5 % — via Cryptomatte mask if available.
            2. Depth span ≥ 0.1 m — ``max(depth) - min(depth)`` in valid region.
            3. DirectRadiance 95th-percentile ≤ 65 504 — no half-float overflow.
            4. Normal unit-length ratio ≥ 95 % — ``|n| ∈ [0.99, 1.01]`` for
               ≥ 95 % of pixels.
            5. Rotation determinant ``|det(R) − 1| < 1e-5`` — SO(3) validity
               on ``camera_matrices[0, :3, :3]``.

        Args:
            sample: Dict with at least ``"video_frames"`` (``[T, H, W, C]``
                uint8/float) and optional ``"fgpp_cond"`` (either
                ``[T, H, W, 15]`` raw FGPP **or** ``[7, T, H, W]`` Wan
                control tensor) and optional ``"camera_matrices"``
                (``[T, 4, 4]`` float32 camera-to-world matrices).

        Returns:
            ``(passed, reason)`` — *True* if every gate passes, else
            *False* with a human-readable failure reason.
        """
        fgpp = sample.get("fgpp_cond")
        cam_mats = sample.get("camera_matrices")

        # Detect layout:  [7, T, H, W] Wan control  vs  [T, H, W, 15] raw FGPP
        is_wan = fgpp is not None and FeatureStoreBuilder._is_wan_control_layout(fgpp)
        is_raw_fgpp = fgpp is not None and not is_wan

        # ── Gates 1-4 apply only on raw [T, H, W, 15] FGPP ─────────

        # Gate 1 — Foreground coverage (Cryptomatte channel, index 14).
        if is_raw_fgpp and fgpp is not None and fgpp.shape[-1] >= 15:
            crypto = fgpp[..., 14]  # Cryptomatte.ID
            fg_mask = crypto > 0
            coverage = float(fg_mask.mean())
            if coverage < 0.05:
                return False, (f"Gate 1 (foreground coverage): {coverage:.3f} < 0.05")

        # Gate 2 — Depth span (CameraLinearDepth.Z, index 0).
        if is_raw_fgpp and fgpp is not None and fgpp.shape[-1] >= 1:
            depth = fgpp[..., 0]
            valid = depth[depth > 0]
            if valid.size > 0:
                span = float(valid.max() - valid.min())
                if span < 0.1:
                    return False, (f"Gate 2 (depth span): {span:.4f} m < 0.1 m")

        # Gate 3 — DirectRadiance 95th percentile (indices 1-3).
        if is_raw_fgpp and fgpp is not None and fgpp.shape[-1] >= 4:
            radiance = fgpp[..., 1:4]
            p95 = float(np.percentile(radiance, 95))
            if p95 > 65504.0:
                return False, (f"Gate 3 (radiance overflow): p95={p95:.1f} > 65504")

        # Gate 4 — Normal unit-length (WorldNormal indices 4-6).
        if is_raw_fgpp and fgpp is not None and fgpp.shape[-1] >= 7:
            normals = fgpp[..., 4:7]
            norms = np.linalg.norm(normals, axis=-1)
            # Only check pixels that have a non-zero normal.
            active = norms > 0.01
            if active.sum() > 0:
                unit_ok = np.abs(norms[active] - 1.0) < 0.01
                ratio = float(unit_ok.mean())
                if ratio < 0.95:
                    return False, (f"Gate 4 (normal unit-length): {ratio:.3f} < 0.95")

        # ── Wan 7ch control tensor validation (depth span & normal unit-length) ──
        if is_wan and fgpp is not None:
            # Gate-2w zero tensor rejection: reject all-zero Wan control tensors
            if np.all(fgpp == 0):
                return False, "Gate 2w (Wan zero tensor): entire 7ch control tensor is zero"

            # Wan control: [7, T, H, W] — ch0=depth (may be log-encoded [0,1])
            wan_depth = fgpp[0]  # [T, H, W]
            valid_d = wan_depth[wan_depth > 0]
            if valid_d.size > 0:
                d_min, d_max = float(valid_d.min()), float(valid_d.max())
                # Log-encoded depth in [0,1]: check log-domain span, not metric span.
                # A span of 0.005 in log-depth corresponds to negligible depth variation.
                if d_max <= 1.0 and d_min >= 0.0:
                    log_span = d_max - d_min
                    if log_span < 0.005:
                        return False, (
                            f"Gate 2w (Wan log-depth span): {log_span:.4f} < 0.005"
                        )
                else:
                    # Metric-space depth: apply standard 0.1m span check
                    span = d_max - d_min
                    if span < 0.1:
                        return False, (
                            f"Gate 2w (Wan depth span): {span:.4f} m < 0.1 m"
                        )
            wan_normals = fgpp[1:4]  # [3, T, H, W]
            wan_norms = np.linalg.norm(wan_normals, axis=0)  # [T, H, W]
            active_w = wan_norms > 0.01
            if active_w.sum() > 0:
                unit_ok_w = np.abs(wan_norms[active_w] - 1.0) < 0.01
                ratio_w = float(unit_ok_w.mean())
                if ratio_w < 0.95:
                    return False, (f"Gate 4w (Wan normal unit-length): {ratio_w:.3f} < 0.95")

        # ── Gate 5 — Rotation determinant and orthogonality ──────────
        if cam_mats is not None:
            cam = np.asarray(cam_mats)
            if cam.ndim == 3 and cam.shape[1] >= 3 and cam.shape[2] >= 3:
                for t in range(cam.shape[0]):
                    rot = cam[t, :3, :3]
                    det = float(np.linalg.det(rot))
                    ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3), ord="fro"))
                    if abs(det - 1.0) >= 1e-4 or ortho_err >= 1e-4:
                        return False, (
                            f"Gate 5 (rotation det): frame {t} |det(R)-1| = "
                            f"{abs(det - 1.0):.2e} >= 1e-4, ortho_err = {ortho_err:.2e}"
                        )
            elif cam.ndim == 2 and cam.shape[0] >= 3 and cam.shape[1] >= 3:
                rot = cam[:3, :3]
                det = float(np.linalg.det(rot))
                ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3), ord="fro"))
                if abs(det - 1.0) >= 1e-4 or ortho_err >= 1e-4:
                    return False, (
                        f"Gate 5 (rotation det): |det(R)-1| = "
                        f"{abs(det - 1.0):.2e} >= 1e-4, ortho_err = {ortho_err:.2e}"
                    )
        # Dead-code fallback for shape[-1] >= 16 removed (FGPP is 15ch only)

        return True, ""

    def build_shards(self, output_dir: Path) -> list[Path]:
        """Write buffered samples to ``.tar`` shard files.

        In streaming mode (``streaming=True``), returns the list of shard
        files that were incrementally written during ``add_sample`` calls
        and flushes any open shard.  Call this to finalize.

        Each sample in the tar contains:

        - ``{sample_id}.latent.npy`` — latent tensor
        - ``{sample_id}.prompt.npy`` — text embedding
        - ``{sample_id}.fgpp_cond.npy`` — condition tensor
          (if provided)
        - ``{sample_id}.meta.json`` — metadata

        Shards split when the current shard would exceed
        *max_shard_bytes*.

        Args:
            output_dir: Directory to write shard files into.

        Returns:
            List of created shard file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Streaming mode: set output_dir for future add_sample calls
        # and flush any existing open shard
        if self._streaming:
            if self._stream_output_dir is None:
                self._stream_output_dir = output_dir
            # Flush remaining open tar
            if self._stream_tar is not None:
                self._stream_tar.close()
                self._stream_tar = None
            # Return streaming shard paths + any buffered samples
            paths = list(self._stream_shard_paths)
            self._stream_shard_paths.clear()
            # Also flush any non-streaming buffered samples
            if self._samples:
                paths.extend(self._build_buffered_shards(output_dir))
            return paths

        return self._build_buffered_shards(output_dir)

    def _build_buffered_shards(self, output_dir: Path) -> list[Path]:
        """Write internally buffered samples to ``.tar`` shard files."""

        if not self._samples:
            return []

        shard_paths: list[Path] = []
        shard_idx = 0
        current_size = 0
        current_tar: tarfile.TarFile | None = None
        current_path: Path | None = None

        def _open_new_shard() -> tuple[tarfile.TarFile, Path]:
            nonlocal shard_idx
            name = f"shard-{shard_idx:05d}.tar"
            path = output_dir / name
            tf = tarfile.open(path, "w")  # noqa: SIM115
            shard_idx += 1
            shard_paths.append(path)
            return tf, path

        current_tar, current_path = _open_new_shard()

        for sid, lat, prm, cond, sha in self._samples:
            entries = self._serialize_sample(
                sid,
                lat,
                prm,
                cond,
                sha,
            )
            sample_size = sum(len(d) for _, d in entries)

            # Check if we need a new shard
            if current_size > 0 and current_size + sample_size > self._max_shard:
                current_tar.close()
                current_tar, current_path = _open_new_shard()
                current_size = 0

            for entry_name, data in entries:
                info = tarfile.TarInfo(name=entry_name)
                info.size = len(data)
                current_tar.addfile(info, io.BytesIO(data))

            current_size += sample_size

        if current_tar is not None:
            current_tar.close()

        self._samples.clear()
        return shard_paths

    @staticmethod
    def _serialize_sample(
        sample_id: str,
        latent: np.ndarray,
        prompt: np.ndarray,
        fgpp_cond: np.ndarray | None,
        source_sha256: str,
    ) -> list[tuple[str, bytes]]:
        """Serialize a single sample to (name, bytes) pairs."""
        entries: list[tuple[str, bytes]] = []

        # Latent
        buf = io.BytesIO()
        np.save(buf, latent)
        entries.append(
            (f"{sample_id}.latent.npy", buf.getvalue()),
        )

        # Prompt
        buf = io.BytesIO()
        np.save(buf, prompt)
        entries.append(
            (f"{sample_id}.prompt.npy", buf.getvalue()),
        )

        # FGPP condition (optional)
        if fgpp_cond is not None:
            buf = io.BytesIO()
            np.save(buf, fgpp_cond)
            entries.append(
                (f"{sample_id}.fgpp_cond.npy", buf.getvalue()),
            )

        # Meta
        meta = {
            "sample_id": sample_id,
            "source_sha256": source_sha256,
            "latent_shape": list(latent.shape),
            "prompt_shape": list(prompt.shape),
        }
        if fgpp_cond is not None:
            meta["fgpp_cond_shape"] = list(fgpp_cond.shape)
        meta_bytes = json.dumps(
            meta,
            indent=2,
        ).encode("utf-8")
        entries.append(
            (f"{sample_id}.meta.json", meta_bytes),
        )

        return entries


# -------------------------------------------------------------------
# Shard reader / iterator
# -------------------------------------------------------------------


def iter_feature_shards(
    shard_paths: list[Path],
) -> Iterator[dict[str, Any]]:
    """Stream samples from WebDataset-compatible ``.tar`` shards.

    Yields dicts with keys:

    - ``"sample_id"``: str
    - ``"latent"``: np.ndarray
    - ``"prompt"``: np.ndarray
    - ``"fgpp_cond"``: np.ndarray or None
    - ``"meta"``: dict

    Args:
        shard_paths: List of ``.tar`` shard file paths to read.

    Yields:
        One dict per sample across all shards.

    Raises:
        AssetNotFoundError: If any shard file does not exist.
    """
    import re

    _SID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
    _MAX_MEMBER_BYTES = 512 * 1024 * 1024  # 512 MiB per-member limit

    known_suffixes = (
        ".latent.npy",
        ".prompt.npy",
        ".fgpp_cond.npy",
        ".meta.json",
    )

    for shard_path in shard_paths:
        shard_path = Path(shard_path)
        if not shard_path.is_file():
            raise AssetNotFoundError(
                f"Shard file not found: {shard_path}",
                agent_fix_hint=f"Ensure shard file exists at '{shard_path}' before iterating.",
            )

        # Streaming O(1) memory: read entries one by one and yield when
        # a sample boundary is reached (all entries of one sample_id are
        # adjacent in a well-formed shard).
        current_sid: str | None = None
        current_parts: dict[str, bytes] = {}

        with tarfile.open(shard_path, "r|") as tf:  # streaming mode
            for member in tf:
                if not member.isfile():
                    continue
                if member.size > _MAX_MEMBER_BYTES:
                    continue  # skip oversized members

                name = member.name
                f = tf.extractfile(member)
                if f is None:
                    continue
                with f:
                    data = f.read()

                # Parse: {sample_id}.{suffix} matching known suffixes from the right
                matched_sid: str | None = None
                matched_suffix: str | None = None
                for suf in known_suffixes:
                    if name.endswith(suf):
                        matched_sid = name[: -len(suf)]
                        matched_suffix = suf[1:]  # e.g. "latent.npy"
                        break
                if matched_sid is None or matched_suffix is None:
                    continue

                # Validate sample_id format
                if not _SID_RE.match(matched_sid):
                    continue

                # Flush previous sample if we hit a new sample_id boundary
                if current_sid is not None and matched_sid != current_sid:
                    sample = _assemble_sample(current_sid, current_parts)
                    if sample is not None:
                        yield sample
                    current_parts = {}

                current_sid = matched_sid
                current_parts[matched_suffix] = data

        # Flush the last sample in the shard
        if current_sid is not None and current_parts:
            sample = _assemble_sample(current_sid, current_parts)
            if sample is not None:
                yield sample


def _assemble_sample(
    sid: str,
    parts: dict[str, bytes],
) -> dict[str, Any] | None:
    """Assemble a training sample from raw byte parts.

    Returns ``None`` if the minimum required keys (``latent``, ``prompt``)
    are missing.
    """
    latent = None
    prompt = None
    fgpp_cond = None
    meta: dict[str, Any] = {}

    if "latent.npy" in parts:
        latent = np.load(io.BytesIO(parts["latent.npy"]), allow_pickle=False)
    if "prompt.npy" in parts:
        prompt = np.load(io.BytesIO(parts["prompt.npy"]), allow_pickle=False)
    if "fgpp_cond.npy" in parts:
        fgpp_cond = np.load(
            io.BytesIO(parts["fgpp_cond.npy"]),
            allow_pickle=False,
        )
    if "meta.json" in parts:
        meta = json.loads(
            parts["meta.json"].decode("utf-8"),
        )

    if latent is not None and prompt is not None:
        return {
            "sample_id": sid,
            "latent": latent,
            "prompt": prompt,
            "fgpp_cond": fgpp_cond,
            "meta": meta,
        }
    return None
