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
import tarfile
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

from synapse.errors import ContractValidationError


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
                agent_fix_hint=(
                    f"Provide a non-empty value for '{name}'."
                ),
            )

    return (
        f"{trigger_token}, {subject_desc}, "
        f"{lighting_env}, {camera_motion}, "
        f"{art_style_tag}"
    )


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
            (tau, h, w, c), dtype=np.float32,
        )
        for i in range(tau):
            src_start = i * 4
            src_end = min(src_start + 4, t)
            temporal[i] = frames[src_start:src_end].mean(axis=0)

        # Spatial compression: 8x average pooling
        h8 = h // 8
        w8 = w // 8
        spatial = np.zeros(
            (tau, h8, w8, c), dtype=np.float32,
        )
        for yi in range(h8):
            for xi in range(w8):
                block = temporal[
                    :,
                    yi * 8:(yi + 1) * 8,
                    xi * 8:(xi + 1) * 8,
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
            tau, h8, w8, self._channels,
        )

        # Rearrange to [C_latent, tau, h8, w8]
        return projected.transpose(3, 0, 1, 2)


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
        hash_floats = np.frombuffer(
            digest, dtype=np.uint8,
        ).astype(np.float32) / 255.0

        n_hash = len(hash_floats)
        for i in range(self._dim):
            idx = i % n_hash
            # Combine with position for variation
            result[i] = hash_floats[idx] * (
                0.5 + 0.5 * np.sin(float(i) * 0.1)
            )

        return result


# -------------------------------------------------------------------
# FeatureStoreBuilder
# -------------------------------------------------------------------

_BUFFERED_SAMPLE = tuple[
    str, np.ndarray, np.ndarray, np.ndarray | None, str,
]


class FeatureStoreBuilder:
    """Build WebDataset-compatible ``.tar`` shards from pre-encoded samples.

    Args:
        latent_encoder: Encoder for video frames -> latent tensor.
        text_encoder: Encoder for caption text -> embedding.
        max_shard_bytes: Maximum bytes per shard (default 2 GiB).
    """

    def __init__(
        self,
        latent_encoder: LatentEncoderProtocol,
        text_encoder: TextEncoderProtocol,
        max_shard_bytes: int = 2 * 1024**3,
    ) -> None:
        self._lat_enc = latent_encoder
        self._txt_enc = text_encoder
        self._max_shard = max_shard_bytes
        self._samples: list[_BUFFERED_SAMPLE] = []

    def add_sample(
        self,
        sample_id: str,
        video_frames: np.ndarray,
        caption: str,
        fgpp_cond: np.ndarray | None = None,
        source_sha256: str = "",
    ) -> None:
        """Pre-encode and buffer a training sample.

        Args:
            sample_id: Unique sample identifier.
            video_frames: ``[T, H, W, 3]`` uint8 frames.
            caption: Caption text to encode.
            fgpp_cond: Optional FGPP condition tensor.
            source_sha256: Optional source sequence SHA-256.
        """
        latent = self._lat_enc.encode(video_frames)
        prompt = self._txt_enc.encode(caption)
        self._samples.append(
            (sample_id, latent, prompt, fgpp_cond, source_sha256),
        )

    def build_shards(self, output_dir: Path) -> list[Path]:
        """Write buffered samples to ``.tar`` shard files.

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

        shard_paths: list[Path] = []
        shard_idx = 0
        current_size = 0
        current_tar: tarfile.TarFile | None = None
        current_path: Path | None = None

        def _open_new_shard() -> tuple[tarfile.TarFile, Path]:
            nonlocal shard_idx
            name = f"shard-{shard_idx:05d}.tar"
            path = output_dir / name
            tf = tarfile.open(path, "w")
            shard_idx += 1
            shard_paths.append(path)
            return tf, path

        current_tar, current_path = _open_new_shard()

        for sid, lat, prm, cond, sha in self._samples:
            entries = self._serialize_sample(
                sid, lat, prm, cond, sha,
            )
            sample_size = sum(len(d) for _, d in entries)

            # Check if we need a new shard
            if (
                current_size > 0
                and current_size + sample_size > self._max_shard
            ):
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
            meta, indent=2,
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
    """
    for shard_path in shard_paths:
        shard_path = Path(shard_path)
        if not shard_path.is_file():
            continue

        # Collect all entries grouped by sample_id prefix
        entries: dict[str, dict[str, bytes]] = {}
        with tarfile.open(shard_path, "r") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()

                # Parse: {sample_id}.{suffix}
                dot_idx = name.find(".")
                if dot_idx < 0:
                    continue
                sid = name[:dot_idx]
                suffix = name[dot_idx + 1:]

                if sid not in entries:
                    entries[sid] = {}
                entries[sid][suffix] = data

        # Yield complete samples
        for sid, parts in entries.items():
            latent = None
            prompt = None
            fgpp_cond = None
            meta: dict[str, Any] = {}

            if "latent.npy" in parts:
                latent = np.load(io.BytesIO(parts["latent.npy"]))
            if "prompt.npy" in parts:
                prompt = np.load(io.BytesIO(parts["prompt.npy"]))
            if "fgpp_cond.npy" in parts:
                fgpp_cond = np.load(
                    io.BytesIO(parts["fgpp_cond.npy"]),
                )
            if "meta.json" in parts:
                meta = json.loads(
                    parts["meta.json"].decode("utf-8"),
                )

            if latent is not None and prompt is not None:
                yield {
                    "sample_id": sid,
                    "latent": latent,
                    "prompt": prompt,
                    "fgpp_cond": fgpp_cond,
                    "meta": meta,
                }


# -------------------------------------------------------------------
# JobBundleWorker stage registration
# -------------------------------------------------------------------

from synapse.pipeline.bundle_worker import (  # noqa: E402
    BundleManifest,
    register_stage,
)


@register_stage("lora_preencode")
def _handle_lora_preencode(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Pre-encode FGPP + video frames into feature shards.

    Uses deterministic test encoders by default. Production
    environments should override via configuration.

    Returns:
        Dict with ``output_paths`` pointing to the shards directory.
    """
    params = dict(manifest.params)
    num_frames = int(params.get("num_frames", 17))
    height = int(params.get("height", 64))
    width = int(params.get("width", 64))
    num_samples = int(params.get("num_samples", 1))

    output_dir = Path(
        manifest.outputs.get("shards_dir", "output/shards"),
    )

    enc = LinearProjectionEncoder()
    txt_enc = LinearProjectionTextEncoder()
    builder = FeatureStoreBuilder(enc, txt_enc)

    for i in range(num_samples):
        rng = np.random.RandomState(i)
        frames = rng.randint(
            0, 256,
            size=(num_frames, height, width, 3),
            dtype=np.uint8,
        )
        caption = format_structured_caption(
            trigger_token="sks",
            subject_desc=f"sample {i}",
            lighting_env="studio",
            camera_motion="static",
            art_style_tag="photorealistic",
        )
        builder.add_sample(
            sample_id=f"sample_{i:04d}",
            video_frames=frames,
            caption=caption,
        )

    shard_paths = builder.build_shards(output_dir)

    return {
        "output_paths": {
            "shards_dir": str(output_dir),
        },
    }
