"""Pydantic manifest schemas and RFC 0039 normalization for JobBundleWorker."""

from __future__ import annotations

import logging
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from synapse.pipeline.sandbox import _check_ancestor_symlinks, _compute_sha256

logger = logging.getLogger(__name__)


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
            raise ValueError(f"free_vram_gb must be >= 0, got {v}")
        return v

    @field_validator("max_cpu_cores")
    @classmethod
    def _check_cores(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_cpu_cores must be >= 1, got {v}")
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


class PresignedEndpoint(BaseModel):
    """RFC 0039 time-limited presigned HTTP endpoint matching Go ``render-farm``."""

    model_config = {"extra": "allow", "populate_by_name": True}

    method: str = ""
    url: str = ""
    expires_at: str = ""


class OutputSink(BaseModel):
    """Upstream output sink specification from RFC 0039 JobBundle.

    Attributes:
        format: Output format (e.g. ``"mp4"``, ``"exr"``, ``"npy"``).
        channels: Requested output channels.
        write_capability: Presigned endpoint (Go ``PresignedEndpoint``) or URI string.
    """

    model_config = {"extra": "allow", "populate_by_name": True}

    format: str = ""
    channels: list[str] = Field(default_factory=list)
    write_capability: str | PresignedEndpoint | dict[str, Any] = ""


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
        *,
        workspace_root: str | Path | None = None,
    ) -> BundleManifest:
        """Create a ``BundleManifest`` from an RFC 0039 ``render-farm`` / ``grand-studio`` JobBundle.

        Accepts the upstream ``{job_id, project_name, closure_type, engine,
        scene_spec, asset_manifest, output_sink, payload}`` wire format,
        maps ``closure_type`` via :data:`CLOSURE_TYPE_MAP`, validates
        ``asset_manifest[].sha256``, and returns a normalised
        ``BundleManifest``.

        Args:
            raw: Raw JSON-decoded dict from the upstream scheduler.
            workspace_root: Optional workspace sandbox directory; when set,
                ``asset_manifest`` paths are validated against symlinks and
                containment before reading.

        Returns:
            Normalised ``BundleManifest``.

        Raises:
            :class:`~synapse.errors.ContractValidationError`:
                Unknown ``closure_type``, missing required fields, or sandbox violation.
            :class:`~synapse.errors.AssetNotFoundError`:
                SHA-256 mismatch in ``asset_manifest``.
        """
        from synapse.errors import AssetNotFoundError, ContractValidationError

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

        asset_manifest = raw.get("asset_manifest", []) or []
        ws_root_resolved = Path(workspace_root).resolve() if workspace_root is not None else None
        for entry in asset_manifest:
            if not isinstance(entry, dict):
                continue
            expected_sha = str(entry.get("sha256", "") or "")
            asset_path = str(entry.get("path", "") or entry.get("logical_path", "") or "")
            if not (expected_sha and asset_path):
                continue
            raw_asset = Path(asset_path)
            if raw_asset.is_symlink() or _check_ancestor_symlinks(raw_asset):
                raise ContractValidationError(
                    f"Symlinks are forbidden in asset_manifest: '{asset_path}'.",
                    agent_fix_hint="Provide real non-symlink paths in asset_manifest.",
                )
            resolved_asset = raw_asset.resolve()
            if ws_root_resolved is not None and not resolved_asset.is_relative_to(ws_root_resolved):
                raise ContractValidationError(
                    f"Asset path escapes workspace sandbox: '{asset_path}' resolves to "
                    f"'{resolved_asset}' outside '{ws_root_resolved}'.",
                    agent_fix_hint=f"Ensure all asset_manifest paths are inside '{ws_root_resolved}'.",
                )
            if resolved_asset.is_file():
                actual_sha = _compute_sha256(resolved_asset)
                if actual_sha != expected_sha:
                    raise AssetNotFoundError(
                        f"SHA-256 mismatch for asset '{asset_path}': "
                        f"expected {expected_sha[:16]}..., got {actual_sha[:16]}...",
                        agent_fix_hint=(
                            f"Re-download or re-export '{asset_path}' with matching SHA-256."
                        ),
                    )

        scene_spec_raw = raw.get("scene_spec", {}) or {}
        payload_raw = raw.get("payload", {}) or {}
        inputs_raw: dict[str, str] = raw.get("inputs", {}) or {}
        input_sha256s: dict[str, str] = raw.get("input_sha256s", {}) or {}
        output_sink_raw = raw.get("output_sink", {}) or {}

        inputs: dict[str, str] = {str(k): str(v) for k, v in inputs_raw.items() if isinstance(v, str)}
        outputs: dict[str, str] = {}
        params: dict[str, Any] = {}
        if input_sha256s:
            params["_input_sha256s"] = dict(input_sha256s)

        if isinstance(scene_spec_raw, dict):
            try:
                scene_spec = SceneSpec.model_validate(scene_spec_raw)
            except ValidationError as exc:
                raise ContractValidationError(
                    f"Invalid scene_spec in RFC 0039 JobBundle: {exc}",
                    agent_fix_hint=(
                        "Fix scene_spec field types (resolution, frame_range, "
                        "crop_window, target_crypto_ids)."
                    ),
                ) from exc

            for k, v in scene_spec_raw.items():
                if isinstance(v, str):
                    inputs[k] = v
                else:
                    params[k] = v

            if scene_spec is not None:
                if scene_spec.resolution is not None:
                    params.setdefault("resolution", scene_spec.resolution)
                    if len(scene_spec.resolution) == 2:
                        params.setdefault("width", int(scene_spec.resolution[0]))
                        params.setdefault("height", int(scene_spec.resolution[1]))
                if scene_spec.frame_range is not None:
                    params.setdefault("frame_range", scene_spec.frame_range)
                    if (
                        len(scene_spec.frame_range) == 2
                        and scene_spec.frame_range[1] >= scene_spec.frame_range[0]
                    ):
                        params.setdefault(
                            "num_frames",
                            int(scene_spec.frame_range[1] - scene_spec.frame_range[0] + 1),
                        )
                if scene_spec.crop_window is not None:
                    params.setdefault("crop_window", scene_spec.crop_window)
                if scene_spec.target_crypto_ids is not None:
                    params.setdefault("target_crypto_ids", scene_spec.target_crypto_ids)
            if isinstance(scene_spec_raw.get("frames"), int):
                params.setdefault("num_frames", int(scene_spec_raw["frames"]))

        if isinstance(payload_raw, dict):
            nested = payload_raw.get(str(closure_type))
            for pk, pv in payload_raw.items():
                if pk == str(closure_type) and isinstance(pv, dict):
                    continue
                params[pk] = pv
            if isinstance(nested, dict):
                params.update(nested)

        if isinstance(output_sink_raw, dict):
            try:
                output_sink = OutputSink.model_validate(output_sink_raw)
            except ValidationError as exc:
                raise ContractValidationError(
                    f"Invalid output_sink in RFC 0039 JobBundle: {exc}",
                    agent_fix_hint=(
                        "Fix output_sink field types (format, channels, write_capability)."
                    ),
                ) from exc

            outputsink_meta_keys = frozenset({"format", "channels", "write_capability"})
            for k, v in output_sink_raw.items():
                if isinstance(v, str) and k not in outputsink_meta_keys:
                    if os.sep in v or "/" in v or "." in v:
                        outputs[k] = v

            if output_sink is not None:
                if output_sink.format:
                    params.setdefault("output_format", output_sink.format)
                if output_sink.channels:
                    params.setdefault("output_channels", output_sink.channels)
                if output_sink.write_capability:
                    if isinstance(output_sink.write_capability, PresignedEndpoint):
                        params.setdefault(
                            "write_capability",
                            output_sink.write_capability.model_dump(),
                        )
                    else:
                        params.setdefault("write_capability", output_sink.write_capability)

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
            except ValidationError as exc:
                logger.warning("Ignoring malformed optional EngineSpec metadata: %s", exc)

        hw_budget_raw = raw.get("hardware_budget", {}) or {}
        hardware_budget = HardwareBudget()
        if isinstance(hw_budget_raw, dict) and hw_budget_raw:
            try:
                hardware_budget = HardwareBudget.model_validate(hw_budget_raw)
            except ValidationError as exc:
                logger.warning(
                    "Invalid optional HardwareBudget (%s); falling back to defaults.",
                    exc,
                )

        traceparent = (
            raw.get("traceparent") or payload_raw.get("traceparent") or os.environ.get("TRACEPARENT")
        )
        if traceparent and not _TRACEPARENT_RE.match(traceparent):
            traceparent = None

        if asset_manifest:
            params.setdefault("_asset_manifest", asset_manifest)

        tb_cfg = raw.get("traceboard")
        if isinstance(tb_cfg, dict) and tb_cfg.get("log_dir"):
            params.setdefault("_traceboard_dir", str(tb_cfg["log_dir"]))
        elif isinstance(tb_cfg, str) and tb_cfg:
            params.setdefault("_traceboard_dir", tb_cfg)

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
