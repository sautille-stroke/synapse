---
type: Design
title: "Traceboard Integration Design — Dual-Track Observability for Synapse"
description: "Integration design for aether-traceboard Python SDK within synapse, covering ProbeProtocol/StageProtocol, Fail-Safe isolation, Tag Taxonomy, monotonic stepping, W3C traceparent correlation, CLI routing, Colab NVMe-First staging, and dual-track bench export."
tags: [synapse, traceboard, observability, aether]
status: active
stale_after: 2028-11-26T00:00:00Z
---
# Traceboard Integration Design — Dual-Track Observability for Synapse

> **Related Plan & Remediation Ledger**:
> - [PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md](plans/PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md)
> - [PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md](plans/PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md)

## 1. Overview

Traceboard is a lightweight, structured experiment-tracking and observability
platform originating in `aether` (Layer 0) as a C++20 SDK (`@aether//traceboard`)
and distributed for Python as `aether-traceboard` (released at `traceboard-python-v0.1.0`
with zero runtime dependencies beyond `numpy>=1.24.0`). This design documents
how Traceboard integrates into `synapse` (Layer 2) while strictly preserving the
DervGen ecosystem's unidirectional dependency rule (Layer 0 -> Layer 2).

## 2. Architecture & Lifecycle Contracts

```
┌──────────────────────────────────────────────────────────────────────┐
│                        synapse (Layer 2)                              │
│                                                                      │
│  synapse.observability (Tier 0, PEP 562 lazy)                        │
│    ├── ProbeProtocol / StageProtocol (@runtime_checkable Protocols)  │
│    ├── init_traceboard()      ← RLock + rollback on partial failure   │
│    ├── get_probe()            ← Thread-safe global Probe accessor    │
│    ├── shutdown_traceboard(status=...) ← RunEnd status propagation   │
│    ├── traceboard_stage()     ← Exception-isolating RAII stage CM    │
│    ├── safe_log_{scalar,text,table,artifact,image,histogram,comparison}
│    └── bound_image_u8()       ← Bounded max_dim<=512 uint8 helper    │
│                                                                      │
│  CLI & Execution Entry Points:                                       │
│    ├── synapse [--traceboard LOG_DIR] ...                            │
│    ├── synapse bench [--traceboard LOG_DIR] {list|run|...}           │
│    ├── synapse bench run --tb-media --traceboard LOG_DIR             │
│    ├── synapse worker run --bundle ... [--traceboard LOG_DIR | SYNAPSE_TRACEBOARD_DIR]
│    ├── scripts/colab_runner.py --traceboard [LOG_DIR]                │
│    └── scripts/run_cpu_verification.py --traceboard [LOG_DIR]        │
│                                                                      │
│  Dual-Track Export:                                                  │
│    └── synapse.bench.traceboard_export.export_reports_to_traceboard  │
│        (reuses SSOT compute_pillar_scores() from bench/report.py)    │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ optional extra: pip install "synapse[traceboard]"
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  aether-traceboard v0.1.0 (Layer 0)                  │
│                                                                      │
│  Probe / StageContext / FileCollector / read_file / iter_events      │
│  Wire parity verified bi-directionally (C++ <-> Python Golden Gate)  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Fail-Safe & Thread-Safe Boundary

1. **Module-level State Ownership**: `synapse.observability` maintains a
   module-local `_probe` reference protected by `threading.RLock()` and
   explicitly separates its lifecycle from `aether_traceboard.get_probe()`.
2. **Atomic Initialization & Rollback**: `init_traceboard()` builds local
   `Probe` + `FileCollector`, registers the collector, activates, calls
   `begin_run(run_name, engine=..., trace_id=..., **config)`, and only assigns
   `_probe` after every step succeeds. On exception, the collector is closed via
   `collector.on_run_close()`, `_probe` stays `None`, and `_initialized` marks
   completion so callers never hang or crash.
3. **Strict Exception Isolation**: `traceboard_stage()` isolates stage-entry
   and `__exit__` telemetry errors while always re-raising caller exceptions
   intact. All `safe_log_*` helpers filter `NaN`/`±Inf` floats (`math.isfinite`),
   coerce table cells to strings, enforce `size_bytes` in artifact metadata,
   and swallow collector I/O failures with debug logs.
4. **Typed Contracts for Tests (`ProbeProtocol`, `StageProtocol`)**: Both
   protocols are `@runtime_checkable` and mirror `aether_traceboard.Probe` and
   `StageContext` method signatures (including `add_collector`,
   `remove_collector`, `stage.set_fail(reason=...)`). All unit tests mock via
   `MagicMock(spec=ProbeProtocol)` / `MagicMock(spec=StageProtocol)` rather than
   unconstrained `MagicMock()`.

## 3. Tag Taxonomy & Low-Cardinality Constraints

All telemetry tags follow a hierarchical prefix (`{domain}/{component}/{metric}`)
with **bounded `O(1)` unique tag cardinality**. Per-sample or per-asset IDs
must NEVER be interpolated into scalar tag names; instead, emit fixed metric
tags keyed by monotonic `step` and record IDs in structured `TABLE` events.

| Domain / Subsystem | Stage Prefix | Scalar / Rich-Media Tags | Step / Cardinality Contract |
|---|---|---|---|
| **bench / runner** | `bench/run_all`, `bench/{scenario}` | `bench/scenario/{name}/passed`, `bench/scenario/{name}/num_frames`, `bench/scenario/{name}/metrics/{k}`, `bench/scenario/{name}/timing/{k}`, `bench/scenario/{name}/keyframe` (`IMAGE`), `bench/scenario/{name}/depth_residual_hist` (`HISTOGRAM`), `bench/summary/*`, `bench/pillar/*` | Bounded scenarios (`O(S)`); images capped via `bound_image_u8(max_dim<=512)` |
| **bench / compare & report** | `bench/compare` | `bench/compare/leaderboard` (`TABLE`), `bench/report` (`ARTIFACT`, `text/html`), `bench/compare_report` (`ARTIFACT`, `text/html`) | Single HTML artifact per report; `size_bytes` required |
| **training** | `train/lora`, `train/package` | `train/loss`, `train/step`, `train/lr_multiplier`, `train/dpo_preferred_err`, `train/dpo_rejected_err`, `train/checkpoint_step`, `train/package/*` | `probe.set_step(step)` + explicit `step=step` strictly monotonic |
| **rsi** | `rsi/gate`, `rsi/curriculum`, `rsi/verifier` | `rsi/promotion/*`, `rsi/promotion/quarantine` (`TABLE`), `rsi/curriculum/*`, `rsi/verifier/physics_passed`, `rsi/verifier/prm_score`, `rsi/verifier/{check}`, `rsi/verifier/verdicts` (`TABLE`) | `round_idx` step for curriculum; per-instance `_traceboard_step` for verifier (`O(1)` tags) |
| **inspect** | `inspect/jev` | `inspect/jev/overall_quality`, `inspect/jev/defect_count`, `inspect/jev/requires_hitl`, `inspect/jev/defects` (`TABLE`) | Fixed scalar tags + monotonic `_traceboard_step`; `asset_id` column in table |
| **adapters & scheduler** | `adapter/wan`, `adapter/scheduler` | `adapter/wan/*`, `adapter/scheduler/{window_idx,overlap_frames,shot_cut_reanchor,window_elapsed_s}` | `step=window_index` (`0, 1, 2, ...`) per causal window |
| **refine (RFC 0061)** | `refine` | `refine/border_psnr_db`, `refine/border_ciede2000`, `refine/temporal_warp_error`, `refine/passed`, `refine/anatomical_patch` (`COMPARISON`) | `uint8` RGB patches capped at `max_dim<=512` with thresholds & pass verdict |
| **worker (RFC 0039)** | `worker/{stage}` | `worker/exit_code`, `worker/elapsed_s`, `worker/output_file_count`, `worker/succeeded`, `worker/trace_id` (`TEXT`) | Covers all exit codes `0, 10, 20, 30, 40, 50` and `TRACEPARENT` |
| **runners** | `colab/{exp}`, `cpu_verify/{exp}` | `colab/gpu/vram_gb`, `colab/{exp}/*`, `cpu_verify/{exp}/passed`, `cpu_verify/elapsed_sec` | `finally:` guarantees `RUN_END` (`finished`/`failed`/`cancelled`) |

## 4. CLI Routing, W3C `traceparent`, and Colab NVMe-First Staging

### 4.1 Subcommand CLI Flag Routing

`get_json_parent_parser()` registers `--traceboard LOG_DIR` (`default=argparse.SUPPRESS`,
`metavar="LOG_DIR"`), enabling `--traceboard` at any parser level (`synapse --traceboard DIR bench list`,
`synapse bench --traceboard DIR list`, `synapse bench run --traceboard DIR`,
`synapse-bench --traceboard DIR`, `synapse worker run --traceboard DIR`). Helper scripts
(`scripts/colab_runner.py`, `scripts/run_cpu_verification.py`) additionally accept
`--traceboard [LOG_DIR]` (`nargs="?"`, `const="__AUTO__"`). CLI initialization is deferred
until a valid subcommand is confirmed so bare `synapse --traceboard DIR` exits `0`
without creating empty `.tbevt` files.

### 4.2 W3C `traceparent` Correlation & Worker Env Config

CLI dispatch extracts `trace_ctx.trace_id` from `TRACEPARENT` or generates a
32-hex ID and forwards `trace_id=...` into `init_traceboard()`. For stateless
container workers (`synapse worker run`), `run_worker_cli()` additionally honors
`SYNAPSE_TRACEBOARD_DIR` and RFC 0039 `JobBundle.traceboard` (`_traceboard_dir`).

### 4.3 Colab NVMe-First Staging (Google Drive SSOT V2.0)

In `scripts/colab_runner.py`, when the resolved telemetry directory lies on a
Google Drive FUSE mount (`/drive/` or `MyDrive`), `.tbevt` writes are staged
exclusively on local NVMe (`/content/work/traceboard/{user}_{date}_{job_id}/`,
or `$TMPDIR/synapse_tb_staging/...`). After each experiment checkpoint and again
inside `finally:` after `shutdown_traceboard(status=...)`, `_sync_traceboard_to_drive()`
flushes active buffers and atomically copies `.tbevt` files to
`{job_dir}/traceboard/` via `.tmp` + `replace()`, preventing FUSE tail corruption
across preemptions (`SIGTERM` exit `143` -> `status="cancelled"`).

## 5. Dual-Track Strategy & Dependency Model

- **HTML & Traceboard Parity**: `bench/report.py` exposes `compute_pillar_scores()`
  as the Single Source of Truth reused by `bench/traceboard_export.py`. Both
  `generate_html_report()` and `render_comparison_html()` emit self-contained HTML
  artifacts (`bench/report`, `bench/compare_report`) alongside live scalars.
- **Optional Zero-Bloat Dependency**: Declared in `pyproject.toml` under
  `[project.optional-dependencies].traceboard = ["aether-traceboard>=0.1.0"]`.
  `import synapse` remains PEP 562 lazy-isolated with zero startup overhead.

## 6. Wire Format & Cross-Language Golden Contract

The Python SDK implements `.tbevt` v1 wire parity matching C++ `@aether//traceboard`:
- `RecordHeader.payloadLength = tagLength + len(payload)`
- IEEE CRC-32 (`0xEDB88320`) across the full `payloadLength` body (`tag + payload`)
- Strict RFC 8259 JSON (`NaN`/`±Inf` sanitized to `0.0`, `TABLE` cells coerced to `str`)
- `STAGE_END` JSON includes `"status"` (`StageStatus`) + optional `"reason"` and `"metrics"`
- `ARTIFACT` metadata normalizes `"mime"` -> `"mime_type"` and enforces `"size_bytes"`
- Verified bi-directionally via C++ `//traceboard:test_golden_interop` reading Python
  `python_reference.tbevt` and Python `pytest traceboard/tests/golden/` reading C++
  `cpp_reference.tbevt`.
