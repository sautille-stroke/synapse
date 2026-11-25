---
type: Design
title: "Traceboard Integration Design — Dual-Track Observability for Synapse"
description: "Integration design for aether-traceboard Python SDK within synapse, covering the dual-track migration strategy, module instrumentation, Colab/CI integration, and bench.report → Traceboard export."
tags: [synapse, traceboard, observability, aether]
status: active
stale_after: 2028-11-25T00:00:00Z
---
# Traceboard Integration Design — Dual-Track Observability for Synapse

> **Related Plan & Landing Summary**: [PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md](plans/PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md)

## 1. Overview

Traceboard is a lightweight, structured experiment-tracking and observability
platform that originated in `aether` (Layer 0) as a C++20 SDK + Go server.
This design documents how Traceboard is integrated into `synapse` (Layer 2)
through its pure-Python SDK (`aether-traceboard`), following the DervGen
ecosystem's unidirectional dependency rule (Layer 0 → Layer 2).

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     synapse (Layer 2)                            │
│                                                                 │
│  synapse.observability          ← Thin wrapper module           │
│    ├── init_traceboard()        ← One-shot initialization       │
│    ├── get_probe()              ← Global probe accessor         │
│    ├── shutdown_traceboard()    ← Graceful teardown              │
│    └── traceboard_stage()       ← Context manager for stages    │
│                                                                 │
│  Instrumented Modules:                                          │
│    ├── synapse.bench            ← Scenario metrics, reports     │
│    ├── synapse.pipeline         ← Bench runner timings          │
│    ├── synapse.training         ← LoRA training losses          │
│    ├── synapse.rsi              ← Curriculum metrics            │
│    ├── synapse.recon            ← Reconstruction metrics        │
│    ├── synapse.inspect          ← JEV critique scoring          │
│    ├── synapse.adapters         ← WanAdapter generate timings   │
│    ├── synapse.style            ← Style extraction metrics      │
│    └── synapse.composite        ← Surgical composition metrics  │
│                                                                 │
│  synapse.bench.traceboard_export  ← Dual-track export utility   │
│    └── export_reports_to_traceboard()                            │
│                                                                 │
│  CLI Flags:                                                     │
│    ├── synapse --traceboard      ← Enable for CLI runs          │
│    ├── synapse-bench --traceboard                                │
│    ├── colab_runner.py --traceboard                              │
│    └── run_cpu_verification.py --traceboard                     │
│                                                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │ pip install aether-traceboard
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               aether-traceboard (Layer 0 SDK)                   │
│                                                                 │
│  Probe        ← Event emitter with scalar/artifact/stage APIs   │
│  FileCollector ← Writes .tbevt files to disk                    │
│  begin_run()  ← Opens a named run with engine/config metadata   │
│  end_run()    ← Flushes and closes the event stream             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Dual-Track Migration Strategy

### Existing Path (unchanged)

`synapse.bench.report.generate_html_report()` produces self-contained HTML
reports with inline SVG radar and bar charts. This path is fully preserved.

### Parallel Traceboard Path (new)

`synapse.bench.traceboard_export.export_reports_to_traceboard()` takes the
same `list[ScenarioRunReport]` and writes structured `.tbevt` events:

| Data                      | Traceboard Key Pattern                        |
|---------------------------|-----------------------------------------------|
| Per-scenario pass/fail    | `scenario/{name}/passed`                      |
| Per-scenario metrics      | `scenario/{name}/metrics/{metric_key}`        |
| Per-scenario timing       | `scenario/{name}/timing/{phase}`              |
| Aggregate summary         | `summary/total_scenarios`, `summary/pass_rate`|
| 4-pillar scores           | `pillar/temporal`, `pillar/fgpp_fidelity`, …  |

### Why Dual-Track?

- **Zero regression risk**: The HTML report remains the primary output.
- **Incremental adoption**: Teams can opt into Traceboard visualization
  without changing existing workflows.
- **Eventual convergence**: Once the Go Traceboard server matures to v1.0,
  the HTML report can be generated from `.tbevt` data as a downstream view.

## 4. Instrumentation Guard Pattern

All instrumentation follows a consistent, non-invasive pattern:

```python
from synapse.observability import get_probe

probe = get_probe()
if probe is not None and probe.active:
    probe.log_scalar("metrics/psnr", 38.7)
```

Key properties:
- `get_probe()` returns `None` when `aether-traceboard` is not installed.
- The import is lazy — no side effects at module load time.
- Guard checks cost ~50ns per call (attribute lookup + branch), negligible
  compared to GPU inference or CPU-heavy metric computation.

## 5. Dependency Model

```
pyproject.toml:
  [project.optional-dependencies]
  traceboard = ["aether-traceboard>=0.1.0"]
  full = ["synapse[...,traceboard,...]"]
```

- `aether-traceboard` is an **optional** dependency.
- `import synapse` does NOT trigger traceboard initialization.
- All Traceboard code paths are guarded by try/except ImportError or
  the `get_probe() is not None` pattern.

## 6. File Manifest

| File                                        | Purpose                                                      |
|---------------------------------------------|--------------------------------------------------------------|
| `python/synapse/observability.py`           | Core wrapper (`init_traceboard`/`get_probe`/`shutdown_traceboard`/`traceboard_stage`) |
| `python/synapse/bench/traceboard_export.py` | Dual-track export utility (`export_reports_to_traceboard`)   |
| `tests/test_observability.py`               | 31 unit + E2E tests for `synapse.observability` and CLI      |
| `tests/test_traceboard_export.py`           | 11 export, pillar-score parity, and real-SDK E2E tests       |
| `tests/test_training_traceboard.py`         | 6 LoRA / DPO / packager instrumentation tests                |
| `tests/test_rsi_recon_traceboard.py`        | 13 RSI / ReconPipeline / VisualJevCritic tests               |
| `tests/test_adapters_style_traceboard.py`   | 7 WanAdapter / StyleExtractor / composite tests              |
| `tools/import_dag_guard.py`                 | Registers `observability` in Tier 0                          |
| `scripts/colab_runner.py`                   | `--traceboard` boolean flag (logs to `{job_dir}/traceboard/`)|
| `scripts/run_cpu_verification.py`           | `--traceboard` boolean flag (logs beside report)             |
| `.github/workflows/ci.yml`                  | Installs optional SDK from sibling `../aether` or PyPI       |
| `workenv/google-colab/setup_colab.sh`       | Colab optional SDK install                                   |
| `docs/DESIGN_TRACEBOARD_INTEGRATION.md`     | This design document                                         |
| `docs/plans/PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md` | Full plan and merged PR landing summary              |

## 7. Wire Format Compatibility

The Python SDK writes `.tbevt` files in the same wire format as the C++
SDK — both are readable by the Go Traceboard server. This enables:

- C++ engine runs (fulgur) and Python algorithm runs (synapse) to be
  visualized in the same Traceboard dashboard.
- Cross-language golden tests in `aether/traceboard/tests/golden/` verify
  format parity.

## 8. Future Work

1. **Traceboard Server UI**: Dashboard for browsing `.tbevt` runs with
   interactive charts, comparison views, and regression detection.
2. **HTML Report from Traceboard**: Generate the existing HTML report format
   as a downstream view of `.tbevt` data, completing the convergence.
3. **Remote Collector**: Stream events to a remote Traceboard server for
   real-time monitoring of GPU inference runs.
