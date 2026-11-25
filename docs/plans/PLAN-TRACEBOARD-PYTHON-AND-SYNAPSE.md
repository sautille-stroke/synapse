---
type: Guide
title: "Traceboard Python SDK 与 Synapse 可观测性集成：完整 Plan 与落地总结"
description: "跨 aether (Layer 0) 与 synapse (Layer 2) 引入纯 Python Traceboard SDK、CLI --traceboard、全模块非侵入埋点、Colab/CI 集成与双轨导出的工程计划、实际落地台账及 Reviewer 审查指南。"
tags: [synapse, traceboard, observability, aether, plan, implementation]
status: active
stale_after: 2027-11-26T00:00:00Z
---
# Traceboard Python SDK 与 Synapse 可观测性集成：完整 Plan 与落地总结

> **文档用途**: 作为 **Plan + Implementation Landing Record (Review Baseline)**，供团队成员对 Traceboard Python 化与 Synapse 全链路接入进行 Code Review、架构复核与上线验收。  
> **涉及仓库**: `dervgen/aether` (Layer 0) + `sautille-stroke/synapse` (Layer 2)  
> **合入范围**: `aether#523` – `#525`（3 PRs） + `synapse#204` – `#212`（9 PRs，含落地后修复闭环 `#212`），**总计 12 个已合入 PR**  
> **设计参考**: [DESIGN_TRACEBOARD_INTEGRATION.md](../DESIGN_TRACEBOARD_INTEGRATION.md)  
> **总体状态**: ✅ **核心能力与全量 PR 已合入两仓 `main`**（详见 §5 计划 vs 实态差异与待办说明）

---

## 0. 执行摘要 (Executive Summary)

为使 `synapse`（Python 算法 SDK / 无状态 Worker / Synapse-Bench）能够复用 `aether` 生态现有的结构化实验追踪能力（`.tbevt` 线格式），而不反向依赖重型 C++ 构建环境，本次改造分 **4 个 Horizon（原计划 12 个 PR + 1 个落地后修复闭环 PR `#212`）** 完成跨仓落地：

| 维度 | 统计指标（已合入 `main` 验证） | 说明 |
|---|---|---|
| **`aether` 侧交付** | `3 PRs` (`#523`, `#524`, `#525`) · `27 files, +3,950` 行 | 纯 Python SDK `aether-traceboard v0.1.0`、50 项跨语言黄金测试 + `python_reference.tbevt`、打包与 CI |
| **`synapse` 侧交付** | `9 PRs` (`#204` – `#212`) · `32 files, +2,939 / -114` 行 | `synapse.observability`、CLI/Colab/CI `--traceboard`、9 子系统 / 14 源码埋点、双轨导出、PR `#212` 修复闭环 |
| **双仓合计变更** | **`12 PRs` · `59 files` · `+6,889 / -114` 行** | 零引入硬依赖（`aether-traceboard` 缺失时静默降级，`get_probe()` 返回 `None`） |
| **新增/验证测试** | `aether`: **154 passed** · `synapse`: **68 项 Traceboard 专项测试** | 含真实 `.tbevt` 读写、`CRC-32` 校验、`RUN_START`/`STAGE`/`TABLE`/`ARTIFACT` E2E 闭环 |
| **零侵入与安全门禁** | `ruff` 0 错 · `mypy --strict` 0 错 · `import_dag_guard` clean | `observability` 归属 Tier 0；`import synapse` 保持 PEP 562 零副作用懒加载 |

---

## 1. 架构设计与核心约束 (Architecture & Invariants)

### 1.1 分层拓扑图

```mermaid
flowchart LR
    subgraph Layer0 ["Layer 0: dervgen/aether (SSOT)"]
        CPP["@aether//traceboard\nC++20 Probe SDK ✅"]
        PYSDK["aether-traceboard (v0.1.0)\n纯 Python SDK (仅依赖 numpy) ✅"]
        SPEC[".tbevt Wire Format Spec v1\n& Golden Contract (1,567B) ✅"]
        CPP --> SPEC
        PYSDK --> SPEC
    end

    subgraph Layer2 ["Layer 2: sautille-stroke/synapse"]
        OBS["Tier 0: synapse.observability\ninit_traceboard / get_probe /\nshutdown_traceboard / traceboard_stage ✅"]
        INST["9 个子系统 14 处非侵入埋点\nbench / pipeline / training / rsi /\nrecon / inspect / adapters / style / composite ✅"]
        EXP["bench.traceboard_export\n双轨导出 (.tbevt ∥ HTML) ✅"]
        CLI["CLI & Scripts --traceboard\nsynapse / synapse-bench /\ncolab_runner / run_cpu_verification ✅"]

        CLI --> OBS
        OBS --> INST
        EXP --> PYSDK
    end

    PYSDK -. "optional extra:\npip install synapse[traceboard]" .-> OBS
```

### 1.2 六大架构铁律 (Design Invariants)

| # | 约束条目 | 落地保障方式 |
|---|---|---|
| **1** | **SDK 归属 Layer 0 (`aether`)** | 包名 `aether-traceboard` 位于 `aether/traceboard/python/`，禁止在 `synapse` 重复造轮子；遵循 Layer 0 → Layer 2 单向依赖。 |
| **2** | **零基础依赖膨胀** | 在 [pyproject.toml](../../pyproject.toml) 作为 `[traceboard]` optional extra（含入 `[full]`），未安装时不阻塞安装与运行。 |
| **3** | **PEP 562 零导入副作用** | `observability` 登记在 [\_\_init\_\_.py](../../python/synapse/__init__.py) `_SUBMODULES` 延迟表；`import synapse` 或 `from synapse import observability` 均不触发初始化或 I/O（已由 `TestLazyImportIsolation` 断言锁死）。 |
| **4** | **缺失时 Graceful Degradation** | `init_traceboard()` 在缺包或初始化异常时捕获处理并置 `get_probe() -> None`；业务埋点统一走 `probe = get_probe(); if probe is not None and probe.active:`，无 Probe 时退化为单次空检查。 |
| **5** | **零重型依赖** | `aether-traceboard` 唯一运行时依赖为 `numpy>=1.24.0`（`pyproject.toml` 声明 `requires-python >= 3.10` + `py.typed`），不引入 `torch` / C++ 扩展；`synapse` 保持 `requires-python >= 3.12`。 |
| **6** | **`.tbevt` 线格式字节兼容** | 严格实现 `.tbevt` v1 文件头（`TBEV` magic、`uint32` version=1、`uint64` SHA-256 `runIdHash`）、记录头、payload 与 IEEE `CRC-32`（`zlib.crc32`），配合 `python_reference.tbevt` 防止跨语言漂移。 |

---

## 2. 原始分期规划与落地 PR 台账 (Plan vs Merged PR Ledger)

```mermaid
flowchart TD
    subgraph HT1 ["Horizon T1: aether Python SDK"]
        T01["PR-T01 -> aether#523\naether-traceboard SDK ✅"]
        T02["PR-T02 -> aether#524\nGolden 黄金测试 ✅"]
        T03["PR-T03 -> aether#525\n元数据与 Publish CI ✅"]
    end
    subgraph HT2 ["Horizon T2: synapse 集成层"]
        T045["PR-T04+T05 -> synapse#204\nobservability + CLI --traceboard ✅"]
        T06["PR-T06 -> synapse#205\n31 项核心单测与 E2E ✅"]
    end
    subgraph HT3 ["Horizon T3: 全模块渐进埋点"]
        T07["PR-T07 -> synapse#206\nbench + pipeline ✅"]
        T08["PR-T08 -> synapse#207\ntraining (LoRA/DPO) ✅"]
        T09["PR-T09 -> synapse#208\nrsi + recon + inspect ✅"]
        T10["PR-T10 -> synapse#209\nadapters + style + composite ✅"]
    end
    subgraph HT4 ["Horizon T4: 集成 / 导出 / 闭环修复"]
        T11["PR-T11 -> synapse#210\nColab & CI --traceboard ✅"]
        T12["PR-T12 -> synapse#211\n双轨导出 + 设计文档 ✅"]
        FIX["Post-Landing -> synapse#212\nDAG/API/CI/严格类型修复 ✅"]
    end

    T01 --> T02 --> T03 --> T045 --> T06
    T045 --> T07 & T08 & T09 & T10
    T07 --> T11 --> T12 --> FIX
```

| Plan 编号 | Horizon | 实际 PR | 合并 Commit | 核心内容摘要 | 规模 | 状态 |
|---|---|---|---|---|---:|---|
| **PR-T01** | T1 | `aether#523` | `de09b1d4` | `aether-traceboard` 纯 Python SDK（10 源码文件 + 6 单测模块） | `18 files, +2,569` | ✅ |
| **PR-T02** | T1 | `aether#524` | `d02b5af4` | Python 黄金生成器、5 阶段校验器、`python_reference.tbevt`（1,567B）、50 项 Golden 测试 | `6 files, +1,077` | ✅ |
| **PR-T03** | T1 | `aether#525` | `b3f179b4` | `aether-traceboard` 包文档、`LICENSE` (MIT)、CI 与 Publish workflow | `4 files, +305 / -1` | ✅ |
| **PR-T04 + T05** | T2 | `synapse#204` | `d36666a` | `synapse.observability` 薄封装、`[traceboard]` extra、`cli/common.py` & `cli/_app.py` `--traceboard` | `6 files, +376 / -45` | ✅ |
| **PR-T06** | T2 | `synapse#205` | `e296a0e` | `tests/test_observability.py` 扩充至 31 项单测（生命周期 / stage / 懒加载 / E2E CRC32） | `1 file, +467 / -57` | ✅ |
| **PR-T07** | T3 | `synapse#206` | `8248647` | `BenchPipelineRunner`、`compare_runs`、`generate_html_report` 埋点 | `3 files, +120 / -48` | ✅ |
| **PR-T08** | T3 | `synapse#207` | `d496a35` | `StatelessLoRATrainer`、`DiffusionDPOLoss`、`packager` 埋点与 6 项训练测试 | `4 files, +348 / -2` | ✅ |
| **PR-T09** | T3 | `synapse#208` | `dcf2ccf` | `promotion_gate`、`curriculum`、`dual_verifier`、`recon/pipeline`、`jev_critic` 埋点与 12 项测试 | `6 files, +511 / -4` | ✅ |
| **PR-T10** | T3 | `synapse#209` | `b1de94f` | `WanAdapter`、`extract_style_{optical,fgpp_intrinsic}`、`composite_surgical_sequence` 埋点与 7 项测试 | `4 files, +292 / -1` | ✅ |
| **PR-T11** | T4 | `synapse#210` | `0cf7b6c` | `colab_runner.py` / `run_cpu_verification.py` `--traceboard` 标志、`setup_colab.sh` 与 CI 安装 | `4 files, +97 / -1` | ✅ |
| **PR-T12** | T4 | `synapse#211` | `a7920b6` | [traceboard_export.py](../../python/synapse/bench/traceboard_export.py) 双轨导出（10 测试）与 [DESIGN_TRACEBOARD_INTEGRATION.md](../DESIGN_TRACEBOARD_INTEGRATION.md) | `3 files, +641` | ✅ |
| **Post-Landing Fix** | Remediation | `synapse#212` | `c57f135` | `import_dag_guard` 注册 `observability`、对齐真实 `Probe` API（`activate` 先于 `begin_run`、`log_table`/`log_artifact` 签名）、`ruff`/`mypy --strict`/CI/Colab 修复 | `16 files, +208 / -77` | ✅ |

---

## 3. 详细落地内容清单 (Detailed Landing Inventory)

### 3.1 `dervgen/aether` 侧：`aether-traceboard` Python SDK (`#523`–`#525`)

#### 核心包 (`aether/traceboard/python/aether_traceboard/`)

| 文件路径 | 核心职责与实现细节 |
|---|---|
| `aether_traceboard/__init__.py` | 导出公开契约：`Probe`, `get_probe`, `StageContext`, `ProbeCollector`, `FileCollector`, `EventType`, `ImageFormat`, `StageStatus`, `Record`, `FileHeader`, `read_file`, `read_events`, `__version__="0.1.0"`。 |
| `aether_traceboard/_types.py` | 定义 `EventType`（`SCALAR=0`, `IMAGE=1`, `HISTOGRAM=2`, `TEXT=3`, `COMPARISON=4`, `TABLE=5`, `ARTIFACT=6`, `SERIES_GROUP=7`, `RUN_START=240`, `RUN_END=241`, `STAGE_BEGIN=242`, `STAGE_END=243`, `CONFIG=244`）、`ImageFormat` 与 `StageStatus`。 |
| `aether_traceboard/_crc32.py` / `_hash.py` | 复用标准库 `zlib.crc32` 计算 IEEE CRC-32 与 SHA-256 截断生成小端 `uint64` `runIdHash`。 |
| `aether_traceboard/event.py` | 完整的 `.tbevt` v1 二进制封包/解包（`FileHeader` 16B、`RecordHeader` 32B、`pack_*/unpack_*` 全 13 类事件），严格小端对齐与 CRC 校验。 |
| `aether_traceboard/collector.py` / `file_collector.py` | `ProbeCollector` 抽象基类与 append-only `FileCollector`（按 run 生成 `.tbevt` 文件，包含实时 flush 与异常恢复边界处理）。 |
| `aether_traceboard/probe.py` | 进程级 `Probe` 单例与 RAII `StageContext`：支持 `log_scalar`, `log_image`, `log_histogram`, `log_text`, `log_comparison`, `log_table`, `log_artifact`, `log_series_group`, `log_config`, `begin_stage`（自动异常捕获标 FAIL）及多 Collector fan-out。 |
| `aether_traceboard/reader.py` | `.tbevt` 读取器与崩溃恢复（`read_events` / `read_file` 遇尾部截断记录时安全停止而不整文件失效）。 |
| `aether_traceboard/py.typed` | PEP 561 类型声明标记（`pyproject.toml` 启用 `mypy strict = true` + `ruff`）。 |

#### 跨语言黄金契约与打包 (`aether/traceboard/tests/golden/` & CI)

- **黄金测试套件** (`traceboard/tests/golden/`):
  - `generate_python_golden.py`: 在固定时间戳生成涵盖全部 13 类事件共 17 条记录的 `python_reference.tbevt`（`1,567 bytes`，已入库追踪）。
  - `verify_golden.py`: 独立 5 阶段校验脚本（文件头、记录内容、CRC-32 扫描、manifest 对齐、字节级确定性重生成对比）。
  - `test_golden.py`: 17 个测试类 / 43 个测试函数（经参数化展开 50 项用例），额外锁死 4 组 SHA-256 `run_id_hash` 跨语言参考向量。
- **发布元数据与 CI**: `aether/traceboard/python/{pyproject.toml, README.md, LICENSE}` 与 `.github/workflows/traceboard-python-publish.yml`（Python `3.10`/`3.11`/`3.12` × `ubuntu`/`macos` 测试矩阵）。

---

### 3.2 `sautille-stroke/synapse` 侧：集成层、全模块埋点与双轨导出 (`#204`–`#212`)

#### A. 核心集成层、CLI 与工程护栏

| 模块/文件 | 落地内容 | 关键接口/标志 |
|---|---|---|
| [observability.py](../../python/synapse/observability.py) | Thin wrapper 单例管理与异常隔离；先 `activate()` 后 `begin_run()`（保证写入 `RUN_START`）。 | `init_traceboard()`, `get_probe()`, `shutdown_traceboard()`, `traceboard_stage()` |
| [\_\_init\_\_.py](../../python/synapse/__init__.py) | 在 `_SUBMODULES` / `TYPE_CHECKING` / `__all__` 注册 `"observability"`。 | PEP 562 延迟加载 |
| [pyproject.toml](../../pyproject.toml) | 新增 optional extra `traceboard = ["aether-traceboard>=0.1.0"]`、`[full]` 纳管，及 `mypy` override。 | `pip install -e ".[traceboard]"` |
| [cli/common.py](../../python/synapse/cli/common.py) & [cli/_app.py](../../python/synapse/cli/_app.py) | 为 `synapse` 与 `synapse-bench` 双入口注册 `--traceboard LOG_DIR`，在 `finally:` 确保 `shutdown_traceboard()` 刷盘。 | `synapse [--traceboard LOG_DIR] ...` |
| [tools/import_dag_guard.py](../../tools/import_dag_guard.py) & [ARCHITECTURE.md](../ARCHITECTURE.md) | `observability` 注册为 **Tier 0**，同步更新权威架构目录树以通过 `test_profiles_doc_sync.py`。 | `python tools/import_dag_guard.py --check` |

#### B. 9 个子系统 · 14 处模块埋点全览

所有埋点均采用延迟局部或安全门卫模式：`probe = get_probe(); if probe is not None and probe.active:`，无 Probe 时开销仅单次 `None` 判断：

| 子系统 | 源码文件 | 埋点函数 / 方法 | 记录的 Traceboard 事件与 Tag 命名空间 |
|---|---|---|---|
| **pipeline** | [bench_runner.py](../../python/synapse/pipeline/bench_runner.py) | `run_scenario()`, `run_all()` | `traceboard_stage("bench/run_all")`, `traceboard_stage("bench/{scenario}")`, `log_scalar("bench/{scenario}/{metric\|timing\|num_frames}")`, `stage.set_pass()` / `stage.set_fail()` |
| **bench** | [compare.py](../../python/synapse/bench/compare.py) | `compare_runs()` | `traceboard_stage("bench/compare")`, `log_table("bench/compare/leaderboard", columns, rows)` |
| **bench** | [report.py](../../python/synapse/bench/report.py) | `generate_html_report()` | `log_artifact("bench/report", {"filename": ..., "mime_type": "text/html"}, html_bytes)` |
| **bench** | [traceboard_export.py](../../python/synapse/bench/traceboard_export.py) | `export_reports_to_traceboard()` | 独立离线转换：`scenario/{name}/passed`, `scenario/{name}/metrics/*`, `scenario/{name}/timing/*`, `summary/*`, `pillar/*` |
| **training** | [trainer.py](../../python/synapse/training/trainer.py) | `inject_lora()`, `save_state_dict()`, `train_step()` | `log_scalar`: `train/lora_injected_layers`, `train/total_params`, `train/trainable_params`, `train/checkpoint_step`, `train/loss`, `train/step`, `train/lr_multiplier`, `train/dpo_beta` |
| **training** | [losses.py](../../python/synapse/training/losses.py) | `DiffusionDPOLoss.__call__()` | `log_scalar`: `train/dpo_preferred_err`, `train/dpo_rejected_err` |
| **training** | [packager.py](../../python/synapse/training/packager.py) | `package_and_gate_lora_bundle()` | `log_scalar`: `train/package_passed`, `train/package/{metric}` |
| **rsi** | [promotion_gate.py](../../python/synapse/rsi/promotion_gate.py) | `evaluate_monotonic_promotion()` | `log_scalar("rsi/promotion/*")`, `log_table("rsi/promotion/quarantine", columns, q_rows)` |
| **rsi** | [curriculum.py](../../python/synapse/rsi/curriculum.py) | `analyse_weaknesses()`, `sample_scenarios()`, `plan_round()` | `log_scalar("rsi/curriculum/{weight/*\|round_idx\|num_scenarios}")`, `log_text("rsi/curriculum/{weak_factors\|factor_distribution}")` |
| **rsi** | [dual_verifier.py](../../python/synapse/rsi/dual_verifier.py) | `verify_physics()`, `_log_verdict()` | `log_scalar("rsi/verifier/{physics_passed\|<check>\|prm_score}/{sample_id}")`, `log_text("rsi/verifier/{failures\|verdict}/{sample_id}")` |
| **recon** | [pipeline.py](../../python/synapse/recon/pipeline.py) | `ReconPipeline.run()` | `log_scalar`: `recon/pipeline/{elapsed_s\|num_frames\|mesh_vertex_count\|mesh_face_count\|is_manifold\|albedo_delta_e\|foot_slip_cm_s\|jev_quality_score}` |
| **inspect** | [jev_critic.py](../../python/synapse/inspect/jev_critic.py) | `VisualJevCritic.evaluate()` | `log_scalar("inspect/jev/{asset_id}/*")`, `log_table("inspect/jev/{asset_id}/defects", columns, rows)` |
| **adapters** | [wan.py](../../python/synapse/adapters/wan.py) | `WanAdapter.generate()` | `log_scalar`: `adapter/wan/{num_output_frames\|output_height\|output_width\|guidance_scale\|num_inference_steps\|seed\|gamma_flow_bias\|has_lora\|total_frames\|window_size}` |
| **style** | [extractor.py](../../python/synapse/style/extractor.py) | `extract_style_optical()`, `extract_style_fgpp_intrinsic()` | `log_scalar("style/optical/*")`, `log_scalar("style/fgpp/num_light_samples")`, `log_scalar("style/fgpp/ambient_ratio")`, `log_text("style/fgpp/key_light_dir")` |
| **composite** | [blend.py](../../python/synapse/composite/blend.py) | `composite_surgical_sequence()` | `log_scalar`: `composite/surgical/{num_frames\|height\|width\|ema_alpha\|mean_mask_coverage\|has_motion_vectors}` |

#### C. Colab / CPU 验证 / CI 与测试资产

- **脚本集成**:
  - [scripts/colab_runner.py](../../scripts/colab_runner.py): 新增 `--traceboard` 开关，激活后写入 `{job_dir}/traceboard/`，在 `finally` 块刷盘关闭（在 `--auto-release` 前执行）。
  - [scripts/run_cpu_verification.py](../../scripts/run_cpu_verification.py): 新增 `--traceboard` 开关，输出 `cpu_verify/*` 阶段与耗时指标到 `{report_out.parent}/traceboard/`。
  - [workenv/google-colab/setup_colab.sh](../../workenv/google-colab/setup_colab.sh) & [.github/workflows/ci.yml](../../.github/workflows/ci.yml): 优先使用同级 `../aether/traceboard/python` editable 安装；缺失 PyPI 包时优雅降级并告警，不阻断主流程。
- **测试套件（共 68 项 Traceboard 专项用例）**:
  - [tests/test_observability.py](../../tests/test_observability.py): 31 项（生命周期、单次/重init、stage RAII、懒加载隔离、CLI flag、4 项真 SDK `.tbevt` + CRC-32 E2E）。
  - [tests/test_rsi_recon_traceboard.py](../../tests/test_rsi_recon_traceboard.py): 13 项（RSI 门禁/课程/双验证、ReconPipeline、VisualJevCritic 及真 SDK `TABLE` E2E）。
  - [tests/test_traceboard_export.py](../../tests/test_traceboard_export.py): 11 项（双轨 `.tbevt` 导出、4 支柱得分算法与 `bench/report.py` + `bench/compare.py` 真 SDK `ARTIFACT`/`TABLE` E2E）。
  - [tests/test_adapters_style_traceboard.py](../../tests/test_adapters_style_traceboard.py): 7 项（`WanAdapter`、`extract_style_*`、`composite_surgical_sequence`）。
  - [tests/test_training_traceboard.py](../../tests/test_training_traceboard.py): 6 项（`StatelessLoRATrainer`、`packager`、`DiffusionDPOLoss`）。

---

## 4. 落地后审计与修复闭环 (`synapse#212` 核心发现)

在 `#204`–`#211` 合入后、准备交付 Review 前的全面核验（`ubuntu24` 实测 + GitHub Actions CI 日志审查）中，发现并已在 **PR `#212`** (`c57f135`) 彻底修复以下 **6 类真实阻断问题**：

1. **5-Tier Import DAG 漏注册导致 CI `python-slow` 全红**：`observability` 加入 `_SUBMODULES` 后未在 `tools/import_dag_guard.py` `TIER_MAP` 注册，引发 3 个致命报错；已注册为 `TIER 0` 并同步更新 [ARCHITECTURE.md](../ARCHITECTURE.md)。
2. **`Probe.begin_run()` 在 `activate()` 前调用导致 `RUN_START` 静默丢失**：`aether_traceboard.Probe.begin_run()` 检查 `if not self._active: return`。`observability.py` 与 `traceboard_export.py` 原先先 `begin_run()` 后 `activate()`，导致真实 SDK E2E `test_full_lifecycle` 失败；已调换顺序确保 `RUN_START` 写入首条记录。
3. **`log_table()` 与 `log_artifact()` 参数签名与真实 SDK 失配**：`bench/compare.py`、`inspect/jev_critic.py`、`rsi/promotion_gate.py` 原以 2 参数传 `list[dict]` 调用 `log_table`；`bench/report.py` 原用 `filename=/data=/mime_type=` 关键字调用 `log_artifact`。因原测试全部采用无约束 `MagicMock()` 而未在单测阶段暴露，接入真实 `aether_traceboard.Probe` 时会抛 `TypeError`。PR `#212` 已全部对齐为 `log_table(tag, columns, rows)` 与 `log_artifact(tag, metadata, blob)`，并补充真 SDK E2E 用例（`test_real_probe_report_and_compare_end_to_end`、`test_jev_critic_real_probe_e2e`）。
4. **`TestPackagerLogging.test_package_logs_gate_result` 阈值越界**：原先因无 `torch` 长期跳过，CI 带 `torch` 实跑暴露 `chamfer_mm=5.00 > RSI_PROMOTION["chamfer_mm"]=4.00`；已修正测试夹具为合法值 `2.5`。
5. **`ruff` / `mypy --strict` 与 `SYNAPSE_CI_STRICT_DEPS=1` 失败**：清理未使用导入 `F401`、移除会触发 `[unused-ignore]` 的冗余 `# type: ignore` 并消除 `[no-any-return]`、补全测试类型标注，并在真 SDK 跨仓测试类显式标注 `@pytest.mark.external_data`。
6. **CI/Colab 拉取尚未公开的 `aether-traceboard` 崩溃**：改为检出同级 `../aether/traceboard/python` 时走 `-e` 安装，否则警告优雅降级，防止 `pip` / `set -euo pipefail` 直接中断。

---

## 5. 计划 vs 实态差异与 Reviewer 关注项 (Known Gaps & Notes)

为保证信息 100% 透明，以下 **5 项计划偏差或后续跟进项** 请 Reviewer 重点知悉：

1. **Report 模板形态差异 (`PR-T12`)**：原计划设想在 `aether/traceboard/server/web/templates/` 新增服务端 HTML 模板；核实当前 `aether/traceboard/` 仅包含 C++ SDK（`src/`, `include/`, `tests/`）与 Python SDK（`python/`, `tests/golden/`），Go Server 源码不在当前 `aether` 目录树内。故 PR `#211` 改为在 `synapse` 侧落地双轨导出模块 [traceboard_export.py](../../python/synapse/bench/traceboard_export.py)，Go Server 端专用模板留待服务端仓后续对接。
2. **跨语言 Golden 验证目前覆盖 Python ↔ Spec，尚未跑通 C++/Go 对拍 (`PR-T02`)**：`aether/traceboard/tests/golden/` 已实现 Python 黄金文件生成、5 阶段校验器与 4 组 SHA-256 跨语言哈希向量，但 C++ 读入 `python_reference.tbevt` 的 GTest 及 Go `TestGoldenPython` 对拍尚未补全。
3. **`aether-traceboard` PyPI / GitHub Packages 尚未正式发布 wheel (`PR-T03`)**：
   - `aether/.github/workflows/traceboard-python-publish.yml` 中 `publish` job 环境变量写了 `TWINE_REPOSITORY_URL: https://npm.pkg.github.com/`，而命令行传了 `--repository-url https://upload.pypi.org/legacy/`（存在目标仓库混用），且 `dervgen/aether` GitHub Actions 因组织账户计费限制未启动运行，尚未创建 `traceboard-python-v0.1.0` Tag；
   - 目前通过源码 editable 安装（`pip install -e ../aether/traceboard/python`）使用，待打包工作流修复并配置组织 PyPI/内部 Registry 凭据后即可发布 `0.1.0`。
4. **脚本 `--traceboard` 形态与 CI Artifact 归档 (`PR-T11`)**：
   - `scripts/colab_runner.py` 与 `scripts/run_cpu_verification.py` 的 `--traceboard` 实现为布尔开关（默认写入 `{job_dir}/traceboard/` / `{report_out.parent}/traceboard/`），而 `synapse` / `synapse-bench` 主 CLI 的 `--traceboard LOG_DIR` 接收目录路径；
   - `.github/workflows/ci.yml` 已接入依赖安装，尚未增加运行 `synapse bench run --traceboard` 并上传 `.tbevt` CI artifact 的独立 step。
5. **公网 Runner 上 `python-fast` 的 3 项非 Traceboard 遗留失败**：在 PR `#212` 的 GitHub Actions 验证中（见运行日志），`docs-and-bench-gate`、`electron-inspector`、`native-bazel`、`python-slow` 4 个 Job 全部通过，**全部 Traceboard 相关单测、`ruff`、`mypy --strict`、`import_dag_guard` 与 `wire_guard` 均 100% 通过**；`python-fast` 仅剩 `tests/test_wave4_rfc0060_rfc0061_parity.py` 中的 3 个既有（`#203` 遗留、无 `DERVGEN_PAT` 时缺少原生 `.so` 且未标 `external_data`）用例因 `SYNAPSE_CI_STRICT_DEPS=1` 失败，不属于本次 Traceboard 改动引入的问题。

---

## 6. 实测验证矩阵与复现指令 (Verification Ledger)

以下结果均于 PR `#212` 合入基线（`c57f135`）及 `aether` `main`（`b3f179b4` 之后）实际跑通观测：

| 仓库 / 检查项 | 执行环境 | 命令 | 实测结果 |
|---|---|---|---|
| **`aether` Python SDK + Golden** | `ubuntu24` | `PYTHONPATH=traceboard/python:traceboard/tests/golden /root/venv-synapse/bin/pytest traceboard/python/tests/ traceboard/tests/golden/ -v` | ✅ **154 passed in 1.16s** |
| **`synapse` Ruff Lint** | `ubuntu24` / macOS | `ruff check python/ tests/ scripts/ tools/` | ✅ **All checks passed (0 errors)** |
| **`synapse` 5-Tier DAG Guard** | `ubuntu24` | `python tools/import_dag_guard.py --check` | ✅ **OK: import DAG is clean** |
| **`synapse` Wire Guard (with aether)** | `ubuntu24` | `python tools/wire_guard.py --check --require-aether` | ✅ **RESULT: All checks passed (15ch, 34B/px)** |
| **`synapse` Markdown 断链扫描** | macOS / CI | `python3 scripts/check_doc_links.py` | ✅ **0 broken links** |
| **`synapse` Mypy Strict (13 Traceboard 相关文件，无 SDK)** | `ubuntu24` | `/usr/local/bin/mypy --strict --follow-imports=silent <13 traceboard/guard files>` | ✅ **Success: no issues found in 13 source files** |
| **`synapse` 单测 + 真 SDK E2E + DAG + 文档同步** | `ubuntu24` (with `aether_traceboard`, `SYNAPSE_CI_STRICT_DEPS=1`) | `PYTHONPATH=.:python:../aether/traceboard/python /root/venv-synapse/bin/pytest tests/test_observability.py tests/test_rsi_recon_traceboard.py tests/test_adapters_style_traceboard.py tests/test_traceboard_export.py tests/test_architecture_dag.py tests/test_profiles_doc_sync.py -v` | ✅ **82 passed in 6.50s (0 skipped)** |
| **`synapse` 降级模式验证 (无 `aether_traceboard`)** | `ubuntu24` (`/usr/bin/python3`, `SYNAPSE_CI_STRICT_DEPS=1`) | `SYNAPSE_CI_STRICT_DEPS=1 PYTHONPATH=.:python /usr/local/bin/pytest tests/test_observability.py tests/test_rsi_recon_traceboard.py tests/test_adapters_style_traceboard.py tests/test_traceboard_export.py tests/test_profiles_doc_sync.py -v` | ✅ **74 passed in 3.17s**（8 项 `external_data` E2E 自动跳过且受 strict-deps 允许） |
| **`synapse` GitHub Actions CI (`PR #212`)** | GitHub `ubuntu-latest` | `CI` workflow (`#212` `ff08852b`) | `docs-and-bench-gate` ✅ · `electron-inspector` ✅ · `native-bazel` ✅ · `python-slow` ✅ (`3m23s`) · `python-fast` ⚠️ `1287 passed`，仅 3 个非 Traceboard 遗留 `test_wave4_rfc0060_rfc0061_parity.py` 失败 |

### Reviewer 快速本地验证命令

```bash
# 1. 在 aether 验证纯 Python SDK 与 Golden 文件契约
cd ../aether/traceboard/python
pip install -e ".[dev]"
PYTHONPATH=.:../tests/golden pytest tests/ ../tests/golden/ -v

# 2. 在 synapse 验证集成、CLI 与静态护栏
cd ../synapse
pip install -e ".[dev]" -e ../aether/traceboard/python
ruff check python/ tests/ scripts/ tools/
python tools/import_dag_guard.py --check
python scripts/check_doc_links.py
SYNAPSE_CI_STRICT_DEPS=1 pytest \
  tests/test_observability.py \
  tests/test_training_traceboard.py \
  tests/test_rsi_recon_traceboard.py \
  tests/test_adapters_style_traceboard.py \
  tests/test_traceboard_export.py \
  tests/test_architecture_dag.py \
  tests/test_profiles_doc_sync.py -v

# 3. 端到端 CLI 验证
synapse bench --traceboard /tmp/tblog list
ls -lh /tmp/tblog/*.tbevt
```
