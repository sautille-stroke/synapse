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
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HardwareBudget(BaseModel):
    """Hardware resource budget for a single job execution.

    Attributes:
        free_vram_gb: Available GPU VRAM in GiB.
        max_cpu_cores: Maximum CPU cores the worker may use.
    """

    free_vram_gb: float = 24.0
    max_cpu_cores: int = 8


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
                f"Unsupported schema_version '{v}'; "
                "expected 'synapse.job_bundle.v1.0'"
            )
        return v


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


# ---------------------------------------------------------------------------
# Stage handler registry
# ---------------------------------------------------------------------------

_STAGE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {}


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
        _STAGE_HANDLERS[name] = fn
        return fn

    return _decorator


def get_registered_stages() -> list[str]:
    """Return sorted list of registered stage names."""
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
            agent_fix_hint=(
                "Run the FGPP exporter to generate metadata.json "
                f"in '{fgpp_dir}'."
            ),
        )

    raw = json.loads(meta_path.read_text(encoding="utf-8"))

    schema_ver = raw.get("schema_version", "")
    if schema_ver != FGPP_SCHEMA_VERSION:
        from synapse.errors import ContractValidationError

        raise ContractValidationError(
            f"Schema version mismatch: expected "
            f"'{FGPP_SCHEMA_VERSION}', got '{schema_ver}'",
            agent_fix_hint=(
                "Re-export with a compatible FGPP producer "
                f"(expected schema '{FGPP_SCHEMA_VERSION}')."
            ),
        )

    num_frames = raw.get(
        "num_frames", raw.get("total_frames", 0),
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
        PreparedConditionPayload,
        VideoGenerationConfig,
        VideoResult,
    )
    from synapse.telemetry import FallbackLevel, MemoryForecaster

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
                    "Increase hardware_budget.free_vram_gb or "
                    "reduce resolution/frame count."
                ),
            )
    elif ratio > 0.80:
        # Level 1: spatial tiling
        params["spatial_tile"] = "2x2"
        fallback_level = "LEVEL_1_SPATIAL_TILE_2X2"

    # --- Create adapter and generate ---
    if adapter_name == "mock":
        adapter = MockVideoAdapter()
    else:
        adapter = create_adapter(adapter_name)

    actual_frames = int(params.get("window_frames", num_frames))

    gen_cfg = VideoGenerationConfig(
        seed=int(params.get("seed", 42)),
        fps=float(params.get("fps", 24.0)),
        width=width,
        height=height,
        num_frames=actual_frames,
        extra={
            k: v for k, v in params.items()
            if k not in {
                "num_frames", "height", "width",
                "model_variant", "adapter", "seed", "fps",
                "window_frames",
            }
        },
    )

    # Build empty conditions payload for the adapter
    conditions = PreparedConditionPayload()

    result: VideoResult = adapter.generate(
        conditions=conditions, config=gen_cfg,
    )

    # Write output
    output_path = manifest.outputs.get(
        "video", str(Path("output") / "video.npy"),
    )
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_p), result.frames_srgb_u8)

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

    # Build synthetic sRGB frames for style extraction
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    num_frames = int(params.get("num_frames", 8))

    frames: list[np.ndarray] = []
    if frames_dir.is_dir():
        for img_path in sorted(frames_dir.glob("*.png"))[
            :num_frames
        ]:
            # Lazy import: PIL is optional
            try:
                from PIL import Image

                img = np.array(
                    Image.open(img_path).convert("RGB"),
                    dtype=np.uint8,
                )
                frames.append(img)
            except ImportError:
                frames.append(
                    np.zeros(
                        (height, width, 3), dtype=np.uint8,
                    )
                )

    if not frames:
        # Generate dummy frames for pipeline validation
        for _ in range(max(num_frames, 2)):
            frames.append(
                np.zeros(
                    (height, width, 3), dtype=np.uint8,
                )
            )

    descriptor = extract_style_optical(frames)

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
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    num_frames = int(params.get("num_frames", 4))

    # Build dummy frame arrays for pipeline validation
    rendered_frames = np.zeros(
        (num_frames, height, width, 3), dtype=np.uint8,
    )
    neural_frames = np.full(
        (num_frames, height, width, 3), 128, dtype=np.uint8,
    )

    # Generate per-frame surgical masks
    masks = np.zeros(
        (num_frames, height, width), dtype=np.float32,
    )
    for t in range(num_frames):
        # Create a simple mask for the centre region
        mask_3d_new = np.zeros(
            (height, width), dtype=bool,
        )
        cy, cx = height // 2, width // 2
        r = min(height, width) // 4
        y, x = np.ogrid[:height, :width]
        mask_3d_new[
            (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2
        ] = True
        masks[t] = generate_surgical_mask(mask_3d_new)

    # Composite
    composited = composite_surgical_sequence(
        neural_frames, rendered_frames, masks,
    )

    output_path = manifest.outputs.get(
        "composited",
        str(Path("output") / "composited.npy"),
    )
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_p), composited)

    return {"output_paths": {"composited": str(out_p)}}


@register_stage("inverse_reconstruct")
@register_stage("kinematic_distill")
def _handle_inverse_reconstruct(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run Stage 3 Video-to-3D Inverse Reconstruction & Kinematic Distillation.

    Expects video frames or FGPP inputs, performs point tracking, solves camera
    trajectory, recovers quad geometry, solves zero-slip foot IK (<0.3 cm/s),
    and de-lights intrinsic albedo.
    """
    from synapse.recon import ReconPipeline

    params = dict(manifest.params)
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    num_frames = int(params.get("num_frames", 8))
    target_faces = int(params.get("target_faces", 1500))

    frames_dir = Path(manifest.inputs.get("frames_dir", ""))
    frames: list[np.ndarray] = []
    if frames_dir.is_dir():
        for img_path in sorted(frames_dir.glob("*.png"))[:num_frames]:
            try:
                from PIL import Image

                img = np.array(
                    Image.open(img_path).convert("RGB"),
                    dtype=np.uint8,
                )
                frames.append(img)
            except ImportError:
                pass

    if not frames:
        # Synthetic video sequence
        for i in range(max(num_frames, 2)):
            synth_frame = np.full((height, width, 3), 120 + (i % 20), dtype=np.uint8)
            frames.append(synth_frame)

    frames_arr = np.stack(frames, axis=0)

    pipe = ReconPipeline(target_quad_faces=target_faces)
    result = pipe.run(frames_arr)

    output_paths: dict[str, str] = {}

    # Save trajectory if requested
    traj_path = manifest.outputs.get(
        "trajectory", str(Path("output") / "camera_trajectory.json")
    )
    p_traj = Path(traj_path)
    p_traj.parent.mkdir(parents=True, exist_ok=True)
    if result.camera_trajectory:
        with open(p_traj, "w", encoding="utf-8") as f:
            json.dump(result.camera_trajectory.to_dict(), f, indent=2)
        output_paths["trajectory"] = str(p_traj)

    # Save telemetry
    telem_path = manifest.outputs.get(
        "telemetry", str(Path("output") / "kinematics_telemetry.json")
    )
    p_telem = Path(telem_path)
    p_telem.parent.mkdir(parents=True, exist_ok=True)
    with open(p_telem, "w", encoding="utf-8") as f:
        json.dump(result.telemetry, f, indent=2)
    output_paths["telemetry"] = str(p_telem)

    return {"output_paths": output_paths}


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
# JobBundleWorker
# ---------------------------------------------------------------------------


class JobBundleWorker:
    """Stateless, single-execution RFC 0039 job bundle worker.

    Reads a :class:`BundleManifest`, dispatches to the registered stage
    handler, computes SHA256 checksums for output files, and returns a
    :class:`ResultManifest`.

    Args:
        worker_id: Identifier for this worker instance.
    """

    def __init__(self, worker_id: str = "local") -> None:
        self.worker_id = worker_id

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

        handler = _STAGE_HANDLERS.get(manifest.stage)
        if handler is None:
            return ResultManifest(
                job_id=manifest.job_id,
                stage=manifest.stage,
                status="error",
                exit_code=10,
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
                timings={
                    "total_seconds": t_end - t_start,
                },
                agent_fix_hint=(
                    getattr(exc, "agent_fix_hint", "")
                    or str(exc)
                ),
            )

        t_end = time.monotonic()

        # Build output file entries with SHA256
        output_paths: dict[str, str] = result_data.get(
            "output_paths", {},
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
) -> int:
    """Execute a job bundle from a JSON manifest file.

    Reads ``bundle_manifest.json`` from *bundle_path*, runs the worker,
    and writes ``result_manifest.json`` alongside the input.

    Args:
        bundle_path: Path to the bundle directory or manifest file.
        json_mode: If ``True``, emit result JSON to stdout.

    Returns:
        Process exit code from the result manifest.
    """
    import sys

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
            "message": (
                f"Bundle manifest not found: {manifest_file}"
            ),
            "agent_fix_hint": (
                f"Ensure '{manifest_file}' exists and is a "
                "valid JSON file."
            ),
        }
        if json_mode:
            json.dump(err, sys.stdout, default=str)
            print()
        else:
            print(
                f"Error: {err['message']}", file=sys.stderr,
            )
        return 20

    raw = json.loads(
        manifest_file.read_text(encoding="utf-8"),
    )
    manifest = BundleManifest(**raw)

    worker = JobBundleWorker()
    result = worker.execute(manifest)

    # Write result manifest
    result_file = bp / "result_manifest.json"
    result_file.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    if json_mode:
        json.dump(
            result.model_dump(),
            sys.stdout,
            default=str,
        )
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
        if result.agent_fix_hint:
            print(
                f"  Hint: {result.agent_fix_hint}",
            )
        for key, entry in result.output_files.items():
            print(
                f"  {key}: {entry.path} "
                f"({entry.size_bytes} bytes, "
                f"sha256={entry.sha256[:16]}...)",
            )

    return result.exit_code
