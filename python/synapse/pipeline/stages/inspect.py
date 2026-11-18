"""3D Inspection pipeline stage handlers (asset_inspect_3d, inspect_refine, inspect_export_rl)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synapse.errors import NotImplementedStageError
from synapse.inspect.flywheel_exporter import (
    export_rl_flywheel_dataset,
    run_targeted_refinement,
)
from synapse.inspect.jev_critic import (
    HttpJevTransport,
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
        transport: JevTransportProtocol = HttpJevTransport(
            endpoint_url=str(endpoint_url),
            model=str(params.get("vlm_model", "default")),
            api_key=str(params.get("vlm_api_key", "")),
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

    output_paths: dict[str, str] = {}
    report_target = manifest.outputs.get("report")
    if report_target:
        p_rep = Path(report_target)
        p_rep.parent.mkdir(parents=True, exist_ok=True)
        p_rep.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        output_paths["report"] = str(p_rep)

    return {
        "output_paths": output_paths,
        "report": report.model_dump(),
    }


@register_stage("inspect_refine")
def handle_inspect_refine(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run targeted refinement from a job bundle manifest."""
    from synapse.errors import ContractValidationError

    params = dict(manifest.params)
    report = params.get("report", {})
    if not report or not isinstance(report, dict):
        raise ContractValidationError(
            "inspect_refine stage requires a non-empty 'report' dict "
            "in manifest.params (from a prior asset_inspect_3d run).",
            agent_fix_hint=(
                "Run 'asset_inspect_3d' first and pass its output report "
                "as params.report in the inspect_refine manifest."
            ),
        )

    result = run_targeted_refinement(report=report)
    if result.get("unfixed_auto_count", 0) > 0:
        raise NotImplementedStageError(
            f"inspect_refine stage found {result['unfixed_auto_count']} auto-fixable defect(s), "
            "but no auto_fix_fn was provided in runtime.",
            agent_fix_hint="Provide an auto_fix_fn backend or escalate to manual HITL.",
        )

    output_paths: dict[str, str] = {}
    refined_target = manifest.outputs.get("refined_mesh") or manifest.outputs.get("refinement")
    if refined_target:
        p_ref = Path(refined_target)
        p_ref.parent.mkdir(parents=True, exist_ok=True)
        p_ref.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        output_paths["refined_mesh"] = str(p_ref)

    return {
        "output_paths": output_paths,
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
