"""3D Inspection pipeline stage handlers (asset_inspect_3d, inspect_refine, inspect_export_rl)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synapse.errors import BackendUnavailableError
from synapse.inspect.flywheel_exporter import (
    export_rl_flywheel_dataset,
    run_targeted_refinement,
)
from synapse.inspect.jev_critic import (
    JevTransportProtocol,
    MockJevTransport,
    VisualJevCritic,
)
from synapse.pipeline.bundle_worker import (
    BundleManifest,
    register_stage,
)


@register_stage("asset_inspect_3d")
def handle_asset_inspect_3d(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run 3D asset inspection using Visual Jev critic."""
    params = dict(manifest.params)
    asset_id = str(params.get("asset_id", "unknown"))
    asset_name = str(params.get("asset_name", "Unknown Asset"))
    image_uris = list(params.get("image_uris", []))

    endpoint_url = params.get("vlm_endpoint_url")
    if endpoint_url:
        try:
            from synapse.inspect.vlm_transport import (  # type: ignore[import-not-found]
                HttpVlmTransport,
            )
        except ImportError as exc:
            raise BackendUnavailableError(
                f"VLM endpoint '{endpoint_url}' was requested, "
                "but 'synapse.inspect.vlm_transport' is not "
                "available. Silent fallback to MockJevTransport "
                "is FORBIDDEN.",
                agent_fix_hint=(
                    "Install VLM transport extra or omit "
                    "'vlm_endpoint_url' in CI mock mode."
                ),
            ) from exc
        transport: JevTransportProtocol = HttpVlmTransport(
            endpoint_url=endpoint_url,
        )
    else:
        transport = MockJevTransport(
            defect_overrides=params.get("defect_overrides"),
        )
    critic = VisualJevCritic(transport=transport)
    report = critic.evaluate(
        asset_id,
        asset_name,
        image_uris,
    )

    return {
        "output_paths": {},
        "report": report.model_dump(),
    }


@register_stage("inspect_refine")
def handle_inspect_refine(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run targeted refinement from a job bundle manifest."""
    params = dict(manifest.params)
    report = params.get("report", {})

    result = run_targeted_refinement(report=report)
    return {
        "output_paths": {},
        **result,
    }


@register_stage("inspect_export_rl")
def handle_inspect_export_rl(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Export RL flywheel data from a job bundle manifest."""
    params = dict(manifest.params)
    log_dir = Path(params.get("log_dir", "."))
    output_path = Path(params.get("output_path", "rl_data.jsonl"))
    fmt = params.get("format", "dpo")

    result = export_rl_flywheel_dataset(
        log_dir=log_dir,
        output_path=output_path,
        format=fmt,
    )
    return {
        "output_paths": {
            "rl_data": str(output_path),
        },
        **result,
    }
