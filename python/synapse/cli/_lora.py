"""LoRA subcommand handlers: prepare-data, train, eval, merge."""

from __future__ import annotations

from synapse.cli.common import (
    Any,
    AssetNotFoundError,
    ContractValidationError,
    ExitCode,
    NotImplementedStageError,
    Path,
    QualityGateError,
    SynapseError,
    _run_with_json_mode,
    argparse,
    json,
    sys,
)


def _cmd_lora_prepare_data(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Prepare training data for LoRA fine-tuning."""
    input_dir = Path(getattr(args, "input", "."))
    output_dir = Path(getattr(args, "output", "."))
    raise NotImplementedStageError(
        "CLI 'synapse lora prepare-data' is not yet wired to "
        "FeatureStoreBuilder; refusing to emit silent success.",
        agent_fix_hint=(
            "Use 'synapse.training.FeatureStoreBuilder' API directly "
            "to encode WebDataset shards."
        ),
        details={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
    )


def _cmd_lora_train(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run LoRA training from CLI config."""
    from synapse.training.trainer import LoRATrainConfig

    config_str = getattr(args, "config", "{}")
    try:
        config_dict = json.loads(config_str)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            f"Invalid JSON passed to --config: {exc}",
            agent_fix_hint=(
                "Pass a valid JSON object string to --config, e.g. "
                '\'{"category": "style", "rank": 8}\'.'
            ),
        ) from exc

    if not isinstance(config_dict, dict) or "category" not in config_dict:
        raise ContractValidationError(
            "LoRA train --config JSON must include explicit 'category'.",
            agent_fix_hint=(
                "Specify 'category' in --config (one of: style, "
                "identity, control, whitebox, motion)."
            ),
        )

    try:
        validated = LoRATrainConfig(**config_dict)
    except Exception as exc:
        raise ContractValidationError(
            f"LoRATrainConfig validation failed: {exc}",
            agent_fix_hint="Fix --config fields per LoRATrainConfig schema.",
        ) from exc

    raise NotImplementedStageError(
        "CLI 'synapse lora train' standalone backbone execution is "
        "not yet implemented; refusing to emit silent success.",
        agent_fix_hint=(
            "Use 'synapse.training.StatelessLoRATrainer' in Python "
            "with an injected DiT model."
        ),
        details={"validated_config": validated.model_dump()},
    )


def _cmd_lora_eval(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate a LoRA bundle against quality gates."""
    bundle_dir = Path(args.bundle)
    eval_path = bundle_dir / "evaluation_report.json"
    if not eval_path.is_file():
        raise AssetNotFoundError(
            f"evaluation_report.json not found in {bundle_dir}",
            agent_fix_hint=(
                "Run 'synapse lora train' first to create "
                "a complete bundle with evaluation_report.json."
            ),
        )

    report = json.loads(eval_path.read_text(encoding="utf-8"))
    passed = report.get("passed", False)
    waived = report.get("waived", False)

    if waived and not getattr(args, "allow_waived", False):
        raise QualityGateError(
            "LoRA bundle evaluation was waived (skip_bench=True); "
            "promotion rejected by fail-closed gate.",
            agent_fix_hint=(
                "Run full Synapse-Bench evaluation to populate "
                "bench_metrics before promoting LoRA weights."
            ),
        )

    if not passed:
        failures = report.get("failures", [])
        hint = report.get(
            "agent_fix_hint",
            "Review training hyperparameters.",
        )
        raise QualityGateError(
            "LoRA bundle failed quality gate: "
            + "; ".join(failures),
            agent_fix_hint=hint,
        )

    return {"passed": True, "waived": waived, "report": report}


def _cmd_lora_merge(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Merge LoRA weights into base model."""
    bundle_dir = Path(args.bundle)
    output_dir = Path(args.output)
    raise NotImplementedStageError(
        "CLI 'synapse lora merge' is not yet wired to a base model "
        "checkpoint; refusing to emit silent success.",
        agent_fix_hint=(
            "Use 'synapse.adapters.lora.MultiLoRAManager' to merge "
            "and apply LoRA adapters onto a loaded model."
        ),
        details={
            "bundle_dir": str(bundle_dir),
            "output_dir": str(output_dir),
        },
    )


def register_lora_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``lora`` subcommand tree."""
    lora_parser = subparsers.add_parser(
        "lora", help="LoRA fine-tuning pipeline operations",
    )
    lora_sub = lora_parser.add_subparsers(dest="lora_command")

    lora_prep = lora_sub.add_parser(
        "prepare-data",
        help="Prepare training data for LoRA fine-tuning",
    )
    lora_prep.add_argument(
        "--input", default=".",
        help="Input data directory",
    )
    lora_prep.add_argument(
        "--output", default=".",
        help="Output directory for prepared data",
    )

    lora_train = lora_sub.add_parser(
        "train",
        help="Run LoRA training from config",
    )
    lora_train.add_argument(
        "--config", default="{}",
        help="JSON training configuration string",
    )

    lora_eval = lora_sub.add_parser(
        "eval",
        help="Evaluate a LoRA bundle against quality gates",
    )
    lora_eval.add_argument(
        "--bundle", required=True,
        help="Path to LoRA bundle directory",
    )

    lora_merge = lora_sub.add_parser(
        "merge",
        help="Merge LoRA weights into base model",
    )
    lora_merge.add_argument(
        "--bundle", required=True,
        help="Path to LoRA bundle directory",
    )
    lora_merge.add_argument(
        "--output", required=True,
        help="Output directory for merged model",
    )
    return lora_parser


def dispatch_lora(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    lora_parser: argparse.ArgumentParser,
) -> int:
    """Dispatch ``synapse lora <sub-command>``."""
    lora_cmd = getattr(parsed, "lora_command", None)
    if lora_cmd == "prepare-data":
        return _run_with_json_mode(
            _cmd_lora_prepare_data, parsed,
            json_mode, trace_ctx,
        )
    elif lora_cmd == "train":
        return _run_with_json_mode(
            _cmd_lora_train, parsed,
            json_mode, trace_ctx,
        )
    elif lora_cmd == "eval":
        return _run_with_json_mode(
            _cmd_lora_eval, parsed,
            json_mode, trace_ctx,
        )
    elif lora_cmd == "merge":
        return _run_with_json_mode(
            _cmd_lora_merge, parsed,
            json_mode, trace_ctx,
        )
    else:
        if json_mode:
            err = SynapseError(
                "Missing lora sub-command "
                "(prepare-data|train|eval|merge)",
                ExitCode.CONTRACT_ERROR,
                "Use 'synapse lora <sub-command> --help'.",
            )
            json.dump(
                err.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
            return int(err.exit_code)
        lora_parser.print_help()
        return 0
