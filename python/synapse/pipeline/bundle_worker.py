"""Stateless RFC 0039 JobBundleWorker with proactive OOM fallback.

Implements the ``synapse worker run --bundle <path>`` execution model:
a zero-daemon, single-execution worker that reads a ``BundleManifest``,
dispatches to a registered stage handler, and writes a ``ResultManifest``
with SHA256-checksummed output files.

GPU memory forecasting for the ``neural_video_generate`` stage uses
:class:`~synapse.telemetry.MemoryForecaster` to proactively select a
fallback level (spatial tiling or short causal window) before any CUDA
allocation, avoiding hard OOM crashes.

Exit codes follow AGENTS.md section 3::

    0  -- success
    10 -- contract validation failure
    20 -- missing asset
    30 -- GPU OOM / backend unreachable
    40 -- circuit breaker / quality gate
    50 -- not implemented
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ClosureType(StrEnum):
    """RFC 0039 closure type identifiers from ``render-farm`` / ``grand-studio``.

    Maps upstream job closure semantics to synapse stage names.
    """

    NEURAL_VIDEO = "neural_video"
    SURGICAL_INPAINT = "surgical_inpaint"
    MESH_RECONSTRUCT = "mesh_reconstruct"
    OPTICAL_POLISH = "optical_polish"
    MOCAP_IK = "mocap_ik"
    DELIGHTING = "delighting"


CLOSURE_TYPE_MAP: dict[str, str] = {
    ClosureType.NEURAL_VIDEO: "neural_video_generate",
    ClosureType.SURGICAL_INPAINT: "surgical_composite",
    ClosureType.MESH_RECONSTRUCT: "video_to_3d_invert",
    ClosureType.OPTICAL_POLISH: "style_extract_and_map",
    ClosureType.MOCAP_IK: "kinematic_distill",
    ClosureType.DELIGHTING: "delight_albedo",
}
"""Mapping from RFC 0039 ``closure_type`` to synapse stage names."""

# W3C traceparent pattern: version-trace_id-parent_id-trace_flags
_TRACEPARENT_RE = re.compile(
    r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"
)


class HardwareBudget(BaseModel):
    """Hardware resource budget for a single job execution.

    Attributes:
        free_vram_gb: Available GPU VRAM in GiB.
        max_cpu_cores: Maximum CPU cores the worker may use.
    """

    model_config = {"extra": "forbid"}

    free_vram_gb: float = 24.0
    max_cpu_cores: int = 8

    @field_validator("free_vram_gb")
    @classmethod
    def _check_vram(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"free_vram_gb must be >= 0, got {v}"
            )
        return v

    @field_validator("max_cpu_cores")
    @classmethod
    def _check_cores(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"max_cpu_cores must be >= 1, got {v}"
            )
        return v


class EngineSpec(BaseModel):
    """Upstream engine specification from RFC 0039 JobBundle.

    Attributes:
        name: Engine name (e.g. ``"fulgur"``, ``"caelum"``).
        version: Engine version string.
        mode: Execution mode (e.g. ``"realtime"``, ``"offline"``).
    """

    model_config = {"extra": "allow", "populate_by_name": True}

    name: str = ""
    version: str = ""
    mode: str = ""


class OutputSink(BaseModel):
    """Upstream output sink specification from RFC 0039 JobBundle.

    Attributes:
        format: Output format (e.g. ``"mp4"``, ``"exr"``, ``"npy"``).
        channels: Requested output channels.
        write_capability: Where output is written.
    """

    model_config = {"extra": "allow", "populate_by_name": True}

    format: str = ""
    channels: list[str] = Field(default_factory=list)
    write_capability: str = ""


class SceneSpec(BaseModel):
    """Upstream scene specification from RFC 0039 JobBundle.

    Attributes:
        resolution: ``[width, height]`` pixel dimensions.
        frame_range: ``[start, end]`` frame indices (inclusive).
        camera: Camera identifier or parameters.
        crop_window: Optional ``[x0, y0, x1, y1]`` normalised crop region.
        target_crypto_ids: Cryptomatte entity IDs to focus on.
    """

    model_config = {"extra": "allow", "populate_by_name": True}

    resolution: list[int] | None = None
    frame_range: list[int] | None = Field(default=None, alias="frameRange")
    camera: str | dict[str, Any] | None = None
    crop_window: list[float] | None = Field(default=None, alias="cropWindow")
    target_crypto_ids: list[int] | None = Field(
        default=None, alias="targetCryptoIds"
    )


class BundleManifest(BaseModel):
    """RFC 0039 Job Bundle manifest describing a single stage execution.

    Attributes:
        schema_version: Wire-format version identifier.
        job_id: Unique job identifier from the upstream scheduler.
        stage: Registered stage name to execute.
        inputs: Mapping of logical input names to file paths / URIs.
        outputs: Mapping of logical output names to file paths / URIs.
        params: Stage-specific parameter overrides.
        hardware_budget: Resource budget for this execution.
        traceparent: Optional W3C traceparent header value.
        tracestate: Optional W3C tracestate header value.
    """

    model_config = {"extra": "forbid"}

    schema_version: str = "synapse.job_bundle.v1.0"
    job_id: str
    stage: str
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    hardware_budget: HardwareBudget = Field(
        default_factory=HardwareBudget,
    )
    traceparent: str | None = None
    tracestate: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: str) -> str:
        if v != "synapse.job_bundle.v1.0":
            raise ValueError(
                f"Unsupported schema_version '{v}'; expected 'synapse.job_bundle.v1.0'"
            )
        return v

    @field_validator("traceparent")
    @classmethod
    def _check_traceparent(cls, v: str | None) -> str | None:
        if v is not None and not _TRACEPARENT_RE.match(v):
            raise ValueError(
                f"Invalid W3C traceparent format: {v!r}. "
                "Expected: <version>-<trace-id>-<parent-id>-<flags> "
                "(hex, e.g. '00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01')."
            )
        return v

    @classmethod
    def from_rfc0039_job_bundle(
        cls,
        raw: dict[str, Any],
    ) -> BundleManifest:
        """Create a ``BundleManifest`` from an RFC 0039 ``render-farm`` / ``grand-studio`` JobBundle.

        Accepts the upstream ``{job_id, project_name, closure_type, engine,
        scene_spec, asset_manifest, output_sink, payload}`` wire format,
        maps ``closure_type`` via :data:`CLOSURE_TYPE_MAP`, validates
        ``asset_manifest[].sha256``, and returns a normalised
        ``BundleManifest``.

        Args:
            raw: Raw JSON-decoded dict from the upstream scheduler.

        Returns:
            Normalised ``BundleManifest``.

        Raises:
            :class:`~synapse.errors.ContractValidationError`:
                Unknown ``closure_type`` or missing required fields.
            :class:`~synapse.errors.AssetNotFoundError`:
                SHA-256 mismatch in ``asset_manifest``.
        """
        from synapse.errors import ContractValidationError

        job_id = raw.get("job_id", "")
        if not job_id:
            raise ContractValidationError(
                "RFC 0039 JobBundle missing 'job_id'.",
                agent_fix_hint="Ensure 'job_id' is set in the JobBundle payload.",
            )

        closure_type = raw.get("closure_type", "")
        stage = CLOSURE_TYPE_MAP.get(closure_type)
        if stage is None:
            raise ContractValidationError(
                f"Unknown closure_type '{closure_type}'. "
                f"Valid types: {sorted(CLOSURE_TYPE_MAP.keys())}",
                agent_fix_hint=(
                    f"Use one of the supported closure types: {sorted(CLOSURE_TYPE_MAP.keys())}"
                ),
            )

        # Defer asset_manifest SHA-256 validation until after workspace
        # sandbox is established (see below) to prevent reads of
        # arbitrary host files outside the sandbox.
        asset_manifest = raw.get("asset_manifest", [])

        scene_spec_raw = raw.get("scene_spec", {}) or {}
        payload_raw = raw.get("payload", {}) or {}
        inputs_raw: dict[str, str] = raw.get("inputs", {}) or {}
        input_sha256s: dict[str, str] = raw.get("input_sha256s", {}) or {}

        # Build normalised inputs/outputs/params from scene_spec + payload
        output_sink_raw = raw.get("output_sink", {}) or {}

        inputs: dict[str, str] = {str(k): str(v) for k, v in inputs_raw.items() if isinstance(v, str)}
        outputs: dict[str, str] = {}
        params: dict[str, Any] = {}
        if input_sha256s:
            params["_input_sha256s"] = dict(input_sha256s)

        # Parse SceneSpec and map string fields to inputs, structured fields to params
        if isinstance(scene_spec_raw, dict):
            try:
                scene_spec = SceneSpec.model_validate(scene_spec_raw)
            except Exception:
                scene_spec = None

            for k, v in scene_spec_raw.items():
                if isinstance(v, str):
                    inputs[k] = v
                else:
                    params[k] = v

            # Also propagate parsed SceneSpec fields into params for downstream
            if scene_spec is not None:
                if scene_spec.resolution is not None:
                    params.setdefault("resolution", scene_spec.resolution)
                if scene_spec.frame_range is not None:
                    params.setdefault("frame_range", scene_spec.frame_range)
                if scene_spec.crop_window is not None:
                    params.setdefault("crop_window", scene_spec.crop_window)
                if scene_spec.target_crypto_ids is not None:
                    params.setdefault("target_crypto_ids", scene_spec.target_crypto_ids)

        # Merge payload into params (payload: null is acceptable)
        if isinstance(payload_raw, dict):
            params.update(payload_raw)

        # Parse OutputSink and map actual file paths to outputs
        if isinstance(output_sink_raw, dict):
            try:
                output_sink = OutputSink.model_validate(output_sink_raw)
            except Exception:
                output_sink = None

            # Only extract values that look like file paths, not format
            # strings like "mp4" or channel lists
            _OUTPUTSINK_META_KEYS = frozenset({
                "format", "channels", "write_capability",
            })
            for k, v in output_sink_raw.items():
                if isinstance(v, str) and k not in _OUTPUTSINK_META_KEYS:
                    # Only treat as a path if it contains a separator or extension
                    if os.sep in v or "/" in v or "." in v:
                        outputs[k] = v

            # Store parsed output_sink metadata in params for downstream access
            if output_sink is not None:
                if output_sink.format:
                    params.setdefault("output_format", output_sink.format)
                if output_sink.channels:
                    params.setdefault("output_channels", output_sink.channels)
                if output_sink.write_capability:
                    params.setdefault("write_capability", output_sink.write_capability)

        # Parse EngineSpec if present
        engine_raw = raw.get("engine", {}) or {}
        if isinstance(engine_raw, dict) and engine_raw:
            try:
                engine_spec = EngineSpec.model_validate(engine_raw)
                if engine_spec.name:
                    params.setdefault("engine_name", engine_spec.name)
                if engine_spec.version:
                    params.setdefault("engine_version", engine_spec.version)
                if engine_spec.mode:
                    params.setdefault("engine_mode", engine_spec.mode)
            except Exception:
                pass  # Non-critical: best-effort engine metadata

        # Parse HardwareBudget if present
        hw_budget_raw = raw.get("hardware_budget", {}) or {}
        hardware_budget = HardwareBudget()
        if isinstance(hw_budget_raw, dict) and hw_budget_raw:
            try:
                hardware_budget = HardwareBudget.model_validate(hw_budget_raw)
            except Exception:
                pass  # Fall back to defaults

        # Extract traceparent from raw, payload or env
        traceparent = (
            raw.get("traceparent") or payload_raw.get("traceparent") or os.environ.get("TRACEPARENT")
        )

        # Validate traceparent format if present
        if traceparent and not _TRACEPARENT_RE.match(traceparent):
            traceparent = None  # Silently drop invalid traceparent from upstream

        # Propagate asset_manifest into params so downstream stages can
        # perform deferred SHA-256 validation inside the workspace sandbox.
        if asset_manifest:
            params.setdefault("_asset_manifest", asset_manifest)

        return cls(
            schema_version="synapse.job_bundle.v1.0",
            job_id=str(job_id),
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            params=params,
            hardware_budget=hardware_budget,
            traceparent=traceparent,
        )


class OutputFileEntry(BaseModel):
    """Metadata for a single output file in the result manifest.

    Attributes:
        path: Absolute or relative path to the output file.
        sha256: Hex-encoded SHA256 digest of the file contents.
        size_bytes: File size in bytes.
    """

    path: str
    sha256: str
    size_bytes: int


class ResultManifest(BaseModel):
    """Result manifest emitted after stage execution.

    Attributes:
        job_id: Echoed job identifier.
        stage: Echoed stage name.
        status: Execution outcome.
        exit_code: Process exit code per AGENTS.md section 3.
        output_files: Checksummed output file entries.
        reanchor_requests: Optional re-anchor suggestions.
        timings: Named duration measurements in seconds.
        fallback_level: Applied OOM fallback tier name, if any.
        agent_fix_hint: Machine-readable remediation hint on error.
        error_type: Error class name on failure (empty on success).
        message: Human-readable error message on failure (empty on success).
    """

    job_id: str
    stage: str
    status: Literal["success", "fallback", "error"]
    exit_code: int
    output_files: dict[str, OutputFileEntry] = Field(
        default_factory=dict,
    )
    reanchor_requests: list[dict[str, Any]] | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    fallback_level: str | None = None
    agent_fix_hint: str | None = None
    error_type: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Stage handler registry
# ---------------------------------------------------------------------------

_STAGE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {}
_STAGE_ADAPTERS_LOADED: bool = False


def _ensure_stage_adapters_loaded() -> None:
    """Import ``synapse.pipeline.stages`` once to register all stage adapters."""
    global _STAGE_ADAPTERS_LOADED
    if not _STAGE_ADAPTERS_LOADED:
        _STAGE_ADAPTERS_LOADED = True
        import synapse.pipeline.stages  # noqa: F401


def register_stage(
    name: str,
) -> Callable[
    [Callable[..., dict[str, Any]]],
    Callable[..., dict[str, Any]],
]:
    """Decorator that registers a stage handler function.

    Usage::

        @register_stage("my_stage")
        def _handle_my_stage(
            manifest: BundleManifest,
        ) -> dict[str, Any]:
            ...

    Args:
        name: Unique stage name used in ``BundleManifest.stage``.

    Returns:
        Decorator that registers the handler and returns it unchanged.
    """

    def _decorator(
        fn: Callable[..., dict[str, Any]],
    ) -> Callable[..., dict[str, Any]]:
        if name in _STAGE_HANDLERS and _STAGE_HANDLERS[name] is not fn:
            existing = _STAGE_HANDLERS[name]
            raise RuntimeError(
                f"Duplicate stage registration: '{name}' is already "
                f"registered to {existing.__module__}.{existing.__qualname__}; "
                f"cannot re-register to {fn.__module__}.{fn.__qualname__}."
            )
        _STAGE_HANDLERS[name] = fn
        return fn

    return _decorator


def get_registered_stages() -> list[str]:
    """Return sorted list of registered stage names."""
    _ensure_stage_adapters_loaded()
    return sorted(_STAGE_HANDLERS.keys())


# ---------------------------------------------------------------------------
# Built-in stage handlers
# ---------------------------------------------------------------------------


@register_stage("fgpp_validate")
def _handle_fgpp_validate(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Validate an FGPP sequence directory.

    Expects ``manifest.inputs["fgpp_dir"]`` to point to a directory
    containing ``metadata.json``.

    Returns:
        Dict with ``valid``, ``schema_version``, ``num_frames``, and
        ``output_paths`` (empty for a read-only validation stage).
    """
    from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

    fgpp_dir = Path(manifest.inputs.get("fgpp_dir", ""))
    if not fgpp_dir.is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"FGPP directory does not exist: {fgpp_dir}",
            agent_fix_hint=(
                f"Ensure '{fgpp_dir}' exists and contains "
                "metadata.json produced by fulgur or synapse-bench."
            ),
        )

    meta_path = fgpp_dir / "metadata.json"
    if not meta_path.is_file():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"metadata.json not found in {fgpp_dir}",
            agent_fix_hint=(f"Run the FGPP exporter to generate metadata.json in '{fgpp_dir}'."),
        )

    raw = json.loads(meta_path.read_text(encoding="utf-8"))

    schema_ver = raw.get("schema_version", "")
    if schema_ver != FGPP_SCHEMA_VERSION:
        from synapse.errors import ContractValidationError

        raise ContractValidationError(
            f"Schema version mismatch: expected '{FGPP_SCHEMA_VERSION}', got '{schema_ver}'",
            agent_fix_hint=(
                "Re-export with a compatible FGPP producer "
                f"(expected schema '{FGPP_SCHEMA_VERSION}')."
            ),
        )

    num_frames = raw.get(
        "num_frames",
        raw.get("total_frames", 0),
    )

    return {
        "valid": True,
        "schema_version": schema_ver,
        "num_frames": num_frames,
        "output_paths": {},
    }


@register_stage("neural_video_generate")
def _handle_neural_video_generate(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Generate video frames using a neural adapter with OOM protection.

    OOM fallback logic:

    1. Estimate peak VRAM via ``MemoryForecaster.estimate_dit_peak_gb``.
    2. Compute ratio R = estimated / free_vram_gb.
    3. If 0.80 < R <= 1.05: inject ``spatial_tile="2x2"``.
    4. If R > 1.05: switch to ``window_frames=49``.
    5. If still over hard limit: reject with exit_code 30.

    Returns:
        Dict with ``output_paths``, ``fallback_level``, and
        ``reanchor_requests``.
    """
    from synapse.adapters import MockVideoAdapter, create_adapter
    from synapse.adapters.base import (
        VideoGenerationConfig,
        VideoResult,
    )
    from synapse.telemetry import MemoryForecaster

    params = dict(manifest.params)
    budget = manifest.hardware_budget

    # Extract generation parameters
    num_frames = int(params.get("num_frames", 97))
    height = int(params.get("height", 512))
    width = int(params.get("width", 512))
    model_variant = str(params.get("model_variant", "wan2.1_14b"))
    adapter_name = str(params.get("adapter", "mock"))

    # --- OOM forecasting ---
    forecast = MemoryForecaster.estimate_dit_peak_gb(
        num_frames=num_frames,
        height=height,
        width=width,
        model_variant=model_variant,
    )

    fallback_level: str | None = None
    ratio = forecast.peak_gb / max(budget.free_vram_gb, 0.01)

    if ratio > 1.05:
        # Level 2: short causal window
        params["window_frames"] = 49
        fallback_level = "LEVEL_2_CAUSAL_SHORT_WINDOW_49F"

        # Re-estimate with reduced frames
        forecast2 = MemoryForecaster.estimate_dit_peak_gb(
            num_frames=49,
            height=height,
            width=width,
            model_variant=model_variant,
        )
        ratio2 = forecast2.peak_gb / max(budget.free_vram_gb, 0.01)
        if ratio2 > 1.05:
            from synapse.errors import BackendUnavailableError

            raise BackendUnavailableError(
                f"Estimated peak VRAM {forecast2.peak_gb:.1f} GiB "
                f"exceeds budget {budget.free_vram_gb:.1f} GiB "
                "even after fallback to 49-frame window",
                agent_fix_hint=(
                    "Increase hardware_budget.free_vram_gb or reduce resolution/frame count."
                ),
            )
    elif ratio > 0.80:
        # Level 1: spatial tiling
        params["spatial_tile"] = "2x2"
        fallback_level = "LEVEL_1_SPATIAL_TILE_2X2"

    # --- Create adapter and generate ---
    adapter: Any
    if adapter_name == "mock":
        adapter = MockVideoAdapter()
    else:
        adapter = create_adapter(adapter_name)

    gen_cfg = VideoGenerationConfig(
        seed=int(params.get("seed", 42)),
        fps=float(params.get("fps", 24.0)),
        width=width,
        height=height,
        num_frames=num_frames,
        extra={
            k: v
            for k, v in params.items()
            if k
            not in {
                "num_frames",
                "height",
                "width",
                "model_variant",
                "adapter",
                "seed",
                "fps",
            }
        },
    )

    # Validate and load FGPP conditions
    fgpp_dir_str = manifest.inputs.get("fgpp_dir", "")
    if not fgpp_dir_str or not Path(fgpp_dir_str).is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"FGPP directory not found: {fgpp_dir_str!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['fgpp_dir'] pointing to a "
                "valid FGPP directory with metadata.json and EXR frames."
            ),
        )

    from synapse.fgpp import import_fgpp

    fgpp_result = import_fgpp(fgpp_dir_str, lazy=False)
    # lazy=False always returns FGPPSequence; narrow the union for mypy.
    assert not isinstance(fgpp_result, Iterator)
    fgpp_seq = fgpp_result
    if not fgpp_seq.frames:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No EXR frames found in FGPP directory: {fgpp_dir_str!r}",
            agent_fix_hint=(
                "Ensure the FGPP directory contains frame_*.exr files alongside metadata.json."
            ),
        )
    conditions = adapter.prepare_conditions(fgpp_seq, gen_cfg)

    result: VideoResult = adapter.generate(
        conditions=conditions,
        config=gen_cfg,
    )

    # Write output
    output_path = manifest.outputs.get(
        "video",
        str(Path("output") / "video.npy"),
    )
    out_p = Path(output_path)
    # np.save auto-appends .npy if suffix missing; normalise beforehand
    if out_p.suffix != ".npy":
        out_p = out_p.with_suffix(out_p.suffix + ".npy")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        np.save(f, result.frames_srgb_u8)

    output_paths: dict[str, str] = {"video": str(out_p)}
    reanchor: list[dict[str, Any]] | None = None

    return {
        "output_paths": output_paths,
        "fallback_level": fallback_level,
        "reanchor_requests": reanchor,
    }


@register_stage("style_extract_and_map")
def _handle_style_extract_and_map(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run style extraction and mapping pipeline.

    Expects ``manifest.inputs["frames_dir"]`` and/or
    ``manifest.inputs["fgpp_dir"]`` to be set.

    Returns:
        Dict with ``output_paths`` to the style profile YAML.
    """
    from synapse.style import (
        extract_style_optical,
        save_style_profile_yaml,
    )

    params = dict(manifest.params)
    frames_dir = Path(
        manifest.inputs.get("frames_dir", ""),
    )

    # Build sRGB frames for style extraction — no synthetic fallback
    num_frames = int(params.get("num_frames", 8))

    if not frames_dir.is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Frames directory does not exist: {frames_dir}",
            agent_fix_hint=(
                "Provide manifest.inputs['frames_dir'] pointing to a "
                "directory containing .png frame files."
            ),
        )

    frames: list[np.ndarray] = []
    for img_path in sorted(frames_dir.glob("*.png"))[:num_frames]:
        # Lazy import: PIL is optional
        try:
            from PIL import Image

            img = np.array(
                Image.open(img_path).convert("RGB"),
                dtype=np.uint8,
            )
            frames.append(img)
        except ImportError:
            from synapse.errors import BackendUnavailableError

            raise BackendUnavailableError(
                "PIL (Pillow) is required for loading frame images but is not installed.",
                agent_fix_hint=("Install Pillow: pip install 'Pillow>=9.0'"),
            ) from None

    if not frames:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No .png frame files found in {frames_dir}",
            agent_fix_hint=("Ensure frames_dir contains .png frame files for style extraction."),
        )

    descriptor = extract_style_optical(np.asarray(frames, dtype=np.uint8))

    output_path = manifest.outputs.get(
        "style_profile",
        str(Path("output") / "style_profile.yaml"),
    )
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    saved_path = save_style_profile_yaml(descriptor, out_p)

    return {"output_paths": {"style_profile": str(saved_path)}}


@register_stage("surgical_composite")
def _handle_surgical_composite(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run surgical compositing pipeline.

    Expects ``manifest.inputs["rendered_dir"]`` and
    ``manifest.inputs["neural_dir"]`` with frame sequences.

    Returns:
        Dict with ``output_paths`` to the composited output.
    """
    from synapse.composite import (
        composite_surgical_sequence,
        generate_surgical_mask,
    )

    params = dict(manifest.params)
    num_frames = int(params.get("num_frames", 4))

    # Load real rendered and neural frames — no synthetic fallback
    rendered_dir_str = manifest.inputs.get("rendered_dir", "")
    neural_dir_str = manifest.inputs.get("neural_dir", "")

    if not rendered_dir_str or not Path(rendered_dir_str).is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Rendered frames directory not found: {rendered_dir_str!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['rendered_dir'] pointing to "
                "a directory containing rendered .png frame files."
            ),
        )

    if not neural_dir_str or not Path(neural_dir_str).is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Neural frames directory not found: {neural_dir_str!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['neural_dir'] pointing to "
                "a directory containing neural .png frame files."
            ),
        )

    try:
        from PIL import Image
    except ImportError:
        from synapse.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "PIL (Pillow) is required for loading frame images but is not installed.",
            agent_fix_hint="Install Pillow: pip install 'Pillow>=9.0'",
        ) from None

    rendered_dir = Path(rendered_dir_str)
    neural_dir = Path(neural_dir_str)

    rendered_list: list[np.ndarray] = []
    for img_path in sorted(rendered_dir.glob("*.png"))[:num_frames]:
        img = np.array(
            Image.open(img_path).convert("RGB"),
            dtype=np.uint8,
        )
        rendered_list.append(img)

    neural_list: list[np.ndarray] = []
    for img_path in sorted(neural_dir.glob("*.png"))[:num_frames]:
        img = np.array(
            Image.open(img_path).convert("RGB"),
            dtype=np.uint8,
        )
        neural_list.append(img)

    if not rendered_list:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No .png frames found in rendered_dir: {rendered_dir}",
            agent_fix_hint=("Ensure rendered_dir contains .png frame files."),
        )

    if not neural_list:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No .png frames found in neural_dir: {neural_dir}",
            agent_fix_hint=("Ensure neural_dir contains .png frame files."),
        )

    rendered_frames = np.stack(rendered_list, axis=0)
    neural_frames = np.stack(neural_list, axis=0)

    height, width = rendered_frames.shape[1], rendered_frames.shape[2]

    # Generate per-frame surgical masks from compositing module
    mask_input_path = manifest.inputs.get("masks", "")
    if mask_input_path and Path(mask_input_path).is_file():
        masks = np.load(mask_input_path).astype(np.float32)
    else:
        # Generate masks from the neural/rendered frame differences
        masks = np.zeros(
            (rendered_frames.shape[0], height, width),
            dtype=np.float32,
        )
        for t in range(rendered_frames.shape[0]):
            diff = np.abs(
                rendered_frames[t].astype(np.float32) - neural_frames[t].astype(np.float32),
            )
            binary_mask = diff.mean(axis=-1) > 10.0
            masks[t] = generate_surgical_mask(binary_mask)

    # Composite
    composited = composite_surgical_sequence(
        neural_frames,
        rendered_frames,
        masks,
    )

    output_path = manifest.outputs.get(
        "composited",
        str(Path("output") / "composited.npy"),
    )
    out_p = Path(output_path)
    if out_p.suffix != ".npy":
        out_p = out_p.with_suffix(out_p.suffix + ".npy")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        np.save(f, composited)

    return {"output_paths": {"composited": str(out_p)}}


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------


def _compute_sha256(path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Ancestor symlink check
# ---------------------------------------------------------------------------


def _check_ancestor_symlinks(p: Path) -> str | None:
    """Check all ancestor components of *p* for symlink traversal.

    Returns an error message string if any ancestor is a symlink,
    or ``None`` if the path is clean.
    """
    # Walk from root to leaf, checking each component
    parts = p.parts
    for i in range(1, len(parts) + 1):
        ancestor = Path(*parts[:i])
        if ancestor.is_symlink():
            return (
                f"Symlink traversal blocked: ancestor '{ancestor}' "
                f"in path '{p}' is a symlink."
            )
    return None


# ---------------------------------------------------------------------------
# JobBundleWorker
# ---------------------------------------------------------------------------


class JobBundleWorker:
    """Stateless, single-execution RFC 0039 job bundle worker.

    Reads a :class:`BundleManifest`, dispatches to the registered stage
    handler, computes SHA256 checksums for output files, and returns a
    :class:`ResultManifest`.

    Args:
        worker_id: Identifier for this worker instance.
        workspace_root: Optional workspace root for path sandboxing.
            When set, all input/output paths must be ``is_relative_to``
            this directory.
    """

    def __init__(
        self,
        worker_id: str = "local",
        workspace_root: str | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.workspace_root: Path | None = (
            Path(workspace_root).resolve() if workspace_root else None
        )

    def execute(
        self,
        manifest: BundleManifest,
    ) -> ResultManifest:
        """Execute a single job bundle stage.

        Args:
            manifest: The job bundle manifest to execute.

        Returns:
            A :class:`ResultManifest` with status, checksums, and
            timing information.
        """
        t_start = time.monotonic()
        _ensure_stage_adapters_loaded()

        # Work on a deep copy to avoid mutating the caller's manifest
        manifest = manifest.model_copy(deep=True)

        # Path normalization, symlink rejection (full ancestor check),
        # and workspace sandboxing for inputs/outputs and params paths.
        for label, mapping in [("inputs", manifest.inputs), ("outputs", manifest.outputs)]:
            for key, val in mapping.items():
                # Reject empty strings that degrade to Path(".") CWD
                if not val or not val.strip():
                    return ResultManifest(
                        job_id=manifest.job_id,
                        stage=manifest.stage,
                        status="error",
                        exit_code=10,
                        error_type="ContractValidationError",
                        message=(
                            f"Empty path in {label}['{key}']. "
                            "Paths must be non-empty strings."
                        ),
                        timings={"total_seconds": time.monotonic() - t_start},
                        agent_fix_hint=(
                            f"Provide a non-empty path for {label}['{key}']."
                        ),
                    )

                raw = Path(val)

                # Check the leaf path itself
                if raw.is_symlink():
                    return ResultManifest(
                        job_id=manifest.job_id,
                        stage=manifest.stage,
                        status="error",
                        exit_code=10,
                        error_type="ContractValidationError",
                        message=(
                            f"Symlinks are forbidden in {label}: "
                            f"'{key}' -> '{val}' is a symlink. "
                            "Use real paths only."
                        ),
                        timings={"total_seconds": time.monotonic() - t_start},
                        agent_fix_hint=(
                            f"Symlinks are forbidden in {label}: "
                            f"'{key}' -> '{val}' is a symlink. "
                            "Use real paths only."
                        ),
                    )

                # Check all ancestor components for symlinks
                ancestor_err = _check_ancestor_symlinks(raw)
                if ancestor_err:
                    return ResultManifest(
                        job_id=manifest.job_id,
                        stage=manifest.stage,
                        status="error",
                        exit_code=10,
                        error_type="ContractValidationError",
                        message=ancestor_err,
                        timings={"total_seconds": time.monotonic() - t_start},
                        agent_fix_hint=(
                            "Remove symlinks from path ancestors. "
                            "Use real paths only."
                        ),
                    )

                resolved = raw.resolve()

                # Workspace root sandboxing
                if self.workspace_root is not None:
                    if not resolved.is_relative_to(self.workspace_root):
                        return ResultManifest(
                            job_id=manifest.job_id,
                            stage=manifest.stage,
                            status="error",
                            exit_code=10,
                            error_type="ContractValidationError",
                            message=(
                                f"Path escapes workspace sandbox: "
                                f"{label}['{key}'] = '{val}' resolves to "
                                f"'{resolved}' which is outside workspace "
                                f"root '{self.workspace_root}'."
                            ),
                            timings={"total_seconds": time.monotonic() - t_start},
                            agent_fix_hint=(
                                f"Ensure all {label} paths are within "
                                f"the workspace root '{self.workspace_root}'."
                            ),
                        )

                mapping[key] = str(resolved)

        # Also check params for path-like values
        for pkey, pval in manifest.params.items():
            if isinstance(pval, str) and (os.sep in pval or "/" in pval):
                raw_p = Path(pval)
                if raw_p.is_symlink():
                    return ResultManifest(
                        job_id=manifest.job_id,
                        stage=manifest.stage,
                        status="error",
                        exit_code=10,
                        error_type="ContractValidationError",
                        message=(
                            f"Symlinks are forbidden in params: "
                            f"'{pkey}' -> '{pval}' is a symlink."
                        ),
                        timings={"total_seconds": time.monotonic() - t_start},
                        agent_fix_hint="Use real paths only in params.",
                    )
                ancestor_err = _check_ancestor_symlinks(raw_p)
                if ancestor_err:
                    return ResultManifest(
                        job_id=manifest.job_id,
                        stage=manifest.stage,
                        status="error",
                        exit_code=10,
                        error_type="ContractValidationError",
                        message=ancestor_err,
                        timings={"total_seconds": time.monotonic() - t_start},
                        agent_fix_hint="Remove symlinks from path ancestors.",
                    )
                if self.workspace_root is not None:
                    resolved_p = raw_p.resolve()
                    if not resolved_p.is_relative_to(self.workspace_root):
                        return ResultManifest(
                            job_id=manifest.job_id,
                            stage=manifest.stage,
                            status="error",
                            exit_code=10,
                            error_type="ContractValidationError",
                            message=(
                                f"Path escapes workspace sandbox: "
                                f"params['{pkey}'] = '{pval}' resolves to "
                                f"'{resolved_p}' which is outside workspace "
                                f"root '{self.workspace_root}'."
                            ),
                            timings={"total_seconds": time.monotonic() - t_start},
                            agent_fix_hint=(
                                f"Ensure all params paths are within "
                                f"the workspace root '{self.workspace_root}'."
                            ),
                        )

        # ---- Deferred SHA-256 validation (after sandbox is enforced) ----
        # The asset_manifest was deferred from from_rfc0039_job_bundle()
        # so it only reads files inside the established workspace sandbox.
        deferred_manifest = manifest.params.get("_asset_manifest", [])
        if deferred_manifest:

            for entry in deferred_manifest:
                if not isinstance(entry, dict):
                    continue
                expected_sha = entry.get("sha256", "")
                asset_path = entry.get("path", "") or entry.get("logical_path", "")
                if expected_sha and asset_path:
                    p = Path(asset_path).resolve()
                    # Enforce sandbox for asset paths too
                    if self.workspace_root is not None:
                        if not p.is_relative_to(self.workspace_root):
                            return ResultManifest(
                                job_id=manifest.job_id,
                                stage=manifest.stage,
                                status="error",
                                exit_code=10,
                                error_type="ContractValidationError",
                                message=(
                                    f"Asset path escapes workspace sandbox: "
                                    f"'{asset_path}' resolves to '{p}' which is "
                                    f"outside workspace root '{self.workspace_root}'."
                                ),
                                timings={"total_seconds": time.monotonic() - t_start},
                                agent_fix_hint=(
                                    f"Ensure all asset paths are within "
                                    f"the workspace root '{self.workspace_root}'."
                                ),
                            )
                    if p.is_file():
                        actual = _compute_sha256(p)
                        if actual != expected_sha:
                            return ResultManifest(
                                job_id=manifest.job_id,
                                stage=manifest.stage,
                                status="error",
                                exit_code=20,
                                error_type="AssetNotFoundError",
                                message=(
                                    f"SHA-256 mismatch for asset '{asset_path}': "
                                    f"expected {expected_sha[:16]}..., "
                                    f"got {actual[:16]}..."
                                ),
                                timings={"total_seconds": time.monotonic() - t_start},
                                agent_fix_hint=(
                                    f"Re-download or re-export '{asset_path}' "
                                    "with the correct contents."
                                ),
                            )
                    elif not p.exists():
                        return ResultManifest(
                            job_id=manifest.job_id,
                            stage=manifest.stage,
                            status="error",
                            exit_code=20,
                            error_type="AssetNotFoundError",
                            message=f"Asset file not found: {asset_path!r}",
                            timings={"total_seconds": time.monotonic() - t_start},
                            agent_fix_hint=(
                                f"Ensure '{asset_path}' exists on disk "
                                "before submitting the JobBundle."
                            ),
                        )

        handler = _STAGE_HANDLERS.get(manifest.stage)
        if handler is None:
            return ResultManifest(
                job_id=manifest.job_id,
                stage=manifest.stage,
                status="error",
                exit_code=10,
                error_type="ContractValidationError",
                message=(
                    f"Unknown stage '{manifest.stage}'. "
                    f"Registered stages: {get_registered_stages()}"
                ),
                timings={
                    "total_seconds": time.monotonic() - t_start,
                },
                agent_fix_hint=(
                    f"Unknown stage '{manifest.stage}'. "
                    f"Registered stages: "
                    f"{get_registered_stages()}"
                ),
            )

        try:
            result_data = handler(manifest)
        except Exception as exc:
            t_end = time.monotonic()

            # Map synapse errors to their exit codes
            exit_code = 10
            from synapse.errors import SynapseError

            if isinstance(exc, SynapseError):
                exit_code = int(exc.exit_code)

            return ResultManifest(
                job_id=manifest.job_id,
                stage=manifest.stage,
                status="error",
                exit_code=exit_code,
                error_type=type(exc).__name__,
                message=str(exc),
                timings={
                    "total_seconds": t_end - t_start,
                },
                agent_fix_hint=(getattr(exc, "agent_fix_hint", "") or str(exc)),
            )

        t_end = time.monotonic()

        # Build output file entries with SHA256
        output_paths: dict[str, str] = result_data.get(
            "output_paths",
            {},
        )
        output_files: dict[str, OutputFileEntry] = {}
        for key, fpath in output_paths.items():
            p = Path(fpath)
            if p.is_file():
                output_files[key] = OutputFileEntry(
                    path=str(p),
                    sha256=_compute_sha256(p),
                    size_bytes=p.stat().st_size,
                )
            elif p.is_dir():
                subfiles = sorted(f for f in p.rglob("*") if f.is_file())
                hasher = hashlib.sha256()
                total_bytes = 0
                for sf in subfiles:
                    rel_p = str(sf.relative_to(p)).replace("\\", "/")
                    hasher.update(rel_p.encode("utf-8"))
                    with open(sf, "rb") as fh:
                        while True:
                            chunk = fh.read(65536)
                            if not chunk:
                                break
                            hasher.update(chunk)
                    total_bytes += sf.stat().st_size
                output_files[key] = OutputFileEntry(
                    path=str(p),
                    sha256=hasher.hexdigest(),
                    size_bytes=total_bytes,
                )

        fallback_level = result_data.get("fallback_level")
        status: Literal["success", "fallback", "error"] = (
            "fallback" if fallback_level else "success"
        )

        return ResultManifest(
            job_id=manifest.job_id,
            stage=manifest.stage,
            status=status,
            exit_code=0,
            output_files=output_files,
            reanchor_requests=result_data.get(
                "reanchor_requests",
            ),
            timings={
                "total_seconds": t_end - t_start,
            },
            fallback_level=fallback_level,
        )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def run_worker_cli(
    bundle_path: str,
    json_mode: bool = False,
    trace_ctx: Any | None = None,
    workspace_root: str | None = None,
) -> int:
    """Execute a job bundle from a JSON manifest file.

    Reads ``bundle_manifest.json`` from *bundle_path*, runs the worker,
    and writes ``result_manifest.json`` alongside the input.

    Automatically detects RFC 0039 upstream ``JobBundle`` wire format
    (by presence of ``closure_type`` key) and routes through
    :meth:`BundleManifest.from_rfc0039_job_bundle` to normalise it.

    Args:
        bundle_path: Path to the bundle directory or manifest file.
        json_mode: If ``True``, emit result JSON to stdout.
        trace_ctx: Optional :class:`~synapse.telemetry.TraceContext`
            for correlation.  Falls back to env ``TRACEPARENT`` or
            manifest ``traceparent``.
        workspace_root: Optional workspace root path for sandboxing.

    Returns:
        Process exit code from the result manifest.
    """
    import sys

    from synapse.telemetry import TraceContext

    if trace_ctx is None:
        trace_ctx = TraceContext.from_env_or_dict(dict(os.environ))

    bp = Path(bundle_path)
    if bp.is_dir():
        manifest_file = bp / "bundle_manifest.json"
    else:
        manifest_file = bp
        bp = bp.parent

    if not manifest_file.is_file():
        err = {
            "status": "error",
            "exit_code": 20,
            "error_type": "AssetNotFoundError",
            "message": f"Bundle manifest not found: {manifest_file}",
            "trace_id": trace_ctx.trace_id,
            "agent_fix_hint": (f"Ensure '{manifest_file}' exists and is a valid JSON file."),
        }
        if json_mode:
            json.dump(err, sys.stdout, default=str)
            print()
        else:
            print(
                f"Error: {err['message']}",
                file=sys.stderr,
            )
        return 20

    try:
        raw = json.loads(
            manifest_file.read_text(encoding="utf-8"),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        err = {
            "status": "error",
            "exit_code": 10,
            "error_type": "ContractValidationError",
            "message": f"Invalid JSON in bundle manifest: {exc}",
            "trace_id": trace_ctx.trace_id,
            "agent_fix_hint": (
                f"Fix the JSON syntax in the bundle manifest file '{manifest_file}' and retry."
            ),
        }
        if json_mode:
            json.dump(err, sys.stdout, default=str)
            print()
        else:
            print(
                f"Error: {err['message']}",
                file=sys.stderr,
            )
        return 10

    # Auto-detect RFC 0039 upstream JobBundle format by presence of
    # ``closure_type`` key and route through the normaliser.
    try:
        if "closure_type" in raw:
            manifest = BundleManifest.from_rfc0039_job_bundle(raw)
        else:
            manifest = BundleManifest(**raw)
    except Exception as exc:
        exit_code = 10
        from synapse.errors import SynapseError

        if isinstance(exc, SynapseError):
            exit_code = int(exc.exit_code)
        err = {
            "status": "error",
            "exit_code": exit_code,
            "error_type": type(exc).__name__,
            "message": f"Bundle manifest validation failed: {exc}",
            "trace_id": trace_ctx.trace_id,
            "agent_fix_hint": (
                getattr(exc, "agent_fix_hint", "")
                or "Ensure the bundle manifest conforms to the "
                "synapse.job_bundle.v1.0 schema or is a valid "
                "RFC 0039 JobBundle with 'closure_type'. Check required "
                "fields: job_id, stage/closure_type, schema_version."
            ),
        }
        if json_mode:
            json.dump(err, sys.stdout, default=str)
            print()
        else:
            print(
                f"Error: {err['message']}",
                file=sys.stderr,
            )
        return exit_code

    # Update trace_ctx from manifest if available
    if manifest.traceparent and not trace_ctx.trace_id:
        trace_ctx = TraceContext.from_env_or_dict(
            {"TRACEPARENT": manifest.traceparent},
        )

    worker = JobBundleWorker(workspace_root=workspace_root)
    result = worker.execute(manifest)

    # Write result manifest
    result_file = bp / "result_manifest.json"
    result_file.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    if json_mode:
        payload = {
            "trace_id": trace_ctx.trace_id,
            **result.model_dump(),
        }
        json.dump(payload, sys.stdout, default=str)
        print()
    else:
        status_label = result.status.upper()
        print(
            f"[{status_label}] job={result.job_id} "
            f"stage={result.stage} "
            f"exit_code={result.exit_code}"
        )
        if result.fallback_level:
            print(
                f"  Fallback: {result.fallback_level}",
            )
        if result.error_type:
            print(
                f"  Error: {result.error_type}: {result.message}",
            )
        if result.agent_fix_hint:
            print(
                f"  Hint: {result.agent_fix_hint}",
            )
        for key, entry in result.output_files.items():
            print(
                f"  {key}: {entry.path} ({entry.size_bytes} bytes, sha256={entry.sha256[:16]}...)",
            )

    return result.exit_code
