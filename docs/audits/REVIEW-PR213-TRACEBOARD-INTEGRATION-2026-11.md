---
type: Guide
title: "PR #213 (Aether Traceboard Python SDK 与 Synapse 集成) 深度代码评审与改进建议报告"
description: "对 sautille-stroke/synapse#204–#213 与 dervgen/aether#523–#525 全链路代码、.tbevt 线格式、打包发布与文档实态的深度评审报告，共识别 20 项缺陷（6 P0 / 9 P1 / 5 P2）并给出系统化改进建议。"
tags: [synapse, traceboard, observability, aether, review, audit]
status: active
stale_after: 2027-11-25T00:00:00Z
---
# PR #213 (`Aether Traceboard` Python SDK 与 `Synapse` 集成) 深度代码评审与改进建议报告

- **文档状态**: `Post-PR#213 Deep Code Review & Audit Report (v1.0)`
- **评审基线**: `synapse` `main` @ `1e3ad5f`（`#204`–`#213`） + `aether` `main` @ `ca2cbff4`（`#523`–`#525`）
- **评审日期**: `2026-11-25`
- **操作模型标识**: `[gemini4-neon]`
- **关联文档**:
  - 被评审基线：[PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md](../plans/PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md) · [DESIGN_TRACEBOARD_INTEGRATION.md](../DESIGN_TRACEBOARD_INTEGRATION.md)
  - 整改落地计划：[PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md](../plans/PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md)
  - 全局实施总纲：[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) · [REVIEW-POST-HORIZON10-2026-11.md](REVIEW-POST-HORIZON10-2026-11.md)

---

## 1. 执行摘要与综合判定 (Executive Summary & Verdict)

### 1.1 总体评审结论

> [!NOTE]
> **整改闭环状态更新 (`2026-11-26` — Full Remediation Closed ✅)**：本报告识别的全部 **20 项缺陷（`TRB-P0-01`–`06` · `TRB-P1-01`–`09` · `TRB-P2-01`–`05`）** 已通过 [PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md](../plans/PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md) 全部 3 Wave / 8 PR + 2 跨仓 PR 落地清零（`dervgen/aether#557`, `#558`, `#560`, Tag/Release `traceboard-python-v0.1.0`, `synapse#215`–`#219`, `caelum-core#557`, `all-overview#770`），**10 项反作弊门禁 (`TB-1 ~ TB-10`) 全绿封板**。以下保留原评审发现作为历史溯源记录。

> [!WARNING]
> **历史评审结论 (`2026-11-25`, 已完成整改)**：原基线（`synapse#213`）经逐行源码审查检出 20 项缺陷，现已全部修复归零。

核心架构亮点值得肯定：
1. **依赖边界正确**：纯 Python SDK 放在 Layer 0 `aether/traceboard/python/`，`synapse` 仅通过 `[traceboard]` optional extra 弱引用，严格遵守 Layer 0 → Layer 2 单向依赖与 PEP 562 零导入副作用铁律。
2. **PR `#212` 自查与 PR `#213` 透明披露**：主动补充 `import_dag_guard` Tier 0 注册、`Probe` 真机 E2E，并如实披露了 Go Server 不在当前 `aether` 目录树、PyPI 尚未发包等 5 项差异。

然而，以下 **4 项实证铁证** 表明当前代码尚不可直接发版或用于生产诊断：

### 1.2 实证复现铁证摘要表（`2026-11-25` 实测观测）

| # | 实证维度 | 复现方法与观测命令 | 实测原始输出 / 证据结论 |
| :---: | :--- | :--- | :--- |
| **1** | **`.tbevt` 线格式字节级不兼容** | 解析 `aether/traceboard/tests/golden/python_reference.tbevt`（1,567 B）Record 0 头部（`<IIdqHBB4x`） | `tag='golden_test_run'`（`tagLength=15`）、JSON payload 62 B，但 header 写入 `payloadLength=62`（违反 `wire_format_spec.md` §3 与 C++ `file_collector.cc:167-171` 要求的 `15+62=77`）。按规范切片 `data[48+15:48+62]` 仅得截断的 47 B：`b'{"engine":"fulgur","resolution":"1920x1080","ve'`；C++/Go 解析器首条记录即 CRC-32 报错。 |
| **2** | **Wheel 打包污染顶层包名** | 在 `ubuntu24` 对 `aether/traceboard/python` 执行 `setuptools.find_packages(where=".")` | 返回 `['aether_traceboard', 'tests']`；构建的 wheel 会向任意安装该包的下游 Python 环境注入顶层 `tests` 包。 |
| **3** | **`mypy --strict` 门禁剧场** | 在 `ubuntu24` 对 `aether/traceboard/python` 运行 `/usr/local/bin/mypy aether_traceboard/ --strict` | **实测报 4 errors in 2 files**：`aether_traceboard/_hash.py:17` `[no-any-return]`、`aether_traceboard/event.py:43` `[unused-ignore]`、`event.py:86` `[unused-ignore]` + `[no-any-return]`；CI 因 `traceboard-python-publish.yml:66` 设置 `continue-on-error: true` 掩盖了失败。 |
| **4** | **CLI 文档命令实跑崩溃** | 使用 `synapse.cli._app.main_synapse` 实跑 PR `#213` §6 Reviewer Quickstart 原文命令 | `synapse bench --traceboard /tmp/tblog list` 报 `argument bench_command: invalid choice: '/tmp/tblog'`（`SystemExit 2` / CLI 退出码 `10`）；`synapse bench list --traceboard /tmp/tblog` 报 `unrecognized arguments`；仅前置 `synapse --traceboard /tmp/tblog bench list` 可解析。 |

### 1.3 缺陷分布统计

| 归属域 | `P0` 阻断 | `P1` 高危 | `P2` 中优先级 | 合计 |
| :--- | :---: | :---: | :---: | :---: |
| **`dervgen/aether` (`aether-traceboard` SDK / Golden / CI)** | 3 | 3 | — | **6** |
| **`sautille-stroke/synapse` (Integration / CLI / Scripts / Tests)** | 3 | 6 | 3 | **12** |
| **文档失真与跨仓生态协同 (`docs` / `caelum-core` / `all-overview`)** | — | — | 2 | **2** |
| **合计** | **6** | **9** | **5** | **20** |

---

## 2. 三大深层系统性根因 (Root Cause Analysis)

1. **同源闭环验证盲区（Python 自写自读自验）**：`aether/traceboard/python` 中写入侧（`file_collector.py`）、读取侧（`reader.py`）、黄金生成器（`generate_python_golden.py`）、校验脚本（`verify_golden.py`）、Golden 单测（`test_golden.py`）、SDK 单测（`test_file_collector.py`, `test_reader.py`）以及 `synapse` 的 [test_observability.py](../../tests/test_observability.py) 第 483 行，**全链条共享了同一个错误的 `body_len = tag_len + payload_len` 假设**。因缺乏 **C++ ↔ Python 双向交叉读写 Golden 测试**，154 项 SDK 测试全部以自身 Bug 为真值跑绿。
2. **门禁剧场回潮（无 `spec` `MagicMock` + `continue-on-error` + `|| true` + 永真断言）**：与 H10 审计（[REVIEW-POST-HORIZON10-2026-11.md](REVIEW-POST-HORIZON10-2026-11.md) §2.2）点名治理的反模式完全一致：
   - 无约束 `MagicMock()` 不仅掩盖了 PR `#212` 修复的 `log_table`/`log_artifact` 签名错误，也掩盖了本次检出的 `BenchPipelineRunner` 调用 `stage.set_fail(reason)`（`TRB-P0-03`）必现 `TypeError`；
   - `traceboard-python-publish.yml:66` 配置 `mypy --strict` + `continue-on-error: true` 使 4 项严格类型错误绕过 CI；
   - `traceboard-python-publish.yml:131` 配置 `twine upload ... || true` 静默吞掉上传失败；
   - [test_traceboard_export.py](../../tests/test_traceboard_export.py) 第 215 行写 `assert isinstance(result, bool)` 永真断言掩盖了降级路径无效的测试漏洞。
3. **契约先行与洁净运行时脱节**：`aether-traceboard` 未对照 `wire_format_spec.md` 与 C++ `probe.cc`/`file_collector.cc` 做字段与并发安全逐项对账，也未在仅安装 wheel + `numpy` 的洁净 venv 中做冒烟导入测试，导致未声明运行期依赖 `typing_extensions`、路径穿越、同秒覆写文件与多写线程竞态直接合入 `main`。

---

## 3. P0 阻断级缺陷详单与改进意见 (`TRB-P0-01` ~ `TRB-P0-06`)

### `TRB-P0-01` [aether] `.tbevt` 线格式 `RecordHeader.payloadLength` 语义反置（100% 跨语言不兼容）

- **权威规范真值**：
  - `aether/traceboard/docs/wire_format_spec.md` §3（第 114–146 行）明确规定：`payloadLength` 为 **`tag + payload data` 的总字节数**；Payload 长度隐式为 `payloadLength - tagLength`；§4（第 175–176 行）规定 CRC-32 覆盖紧随 RecordHeader 之后的全部 `payloadLength` 字节。
  - C++ 实现 `aether/traceboard/src/file_collector.cc:167-171` 严格计算 `fullPayload = tag.size() + totalPayloadSize; rh.payloadLength = static_cast<u32>(fullPayload);`，且 C++ 单测 `aether/traceboard/tests/test_file_collector.cc:104-106` 显式断言 `expectedPayload = rec.tagLength + sizeof(f64); EXPECT_EQ(rec.payloadLength, expectedPayload);`。
- **Python 侧缺陷位置**：
  - `aether/traceboard/python/aether_traceboard/file_collector.py:79-91`：仅写入 `payload_len = len(payload)`（遗漏 `+ tag_len`）；
  - `aether/traceboard/python/aether_traceboard/reader.py:88-90`：错按 `body_len = tag_len + payload_len` 切片；
  - 同源污染扩展到 `generate_python_golden.py:64`、`verify_golden.py:235`、`test_golden.py:315`、`test_file_collector.py:95`、`test_reader.py:33`、提交入库的 `python_reference.tbevt`（1,567 B），以及 `synapse` 的 [test_observability.py](../../tests/test_observability.py) 第 483 行。
- **后果**：C++ 读取器与 Go Ingest Server 解析 Python SDK 写出的任意 `.tbevt` 文件时，会把 payload 末尾 `tagLength` 个字节误吞或当成下一条 RecordHeader 导致 **100% CRC-32 校验失败或解析脱轨**；反之 Python `read_events()` 读入 C++ 写出的 `.tbevt` 文件也会多读 `tagLength` 字节导致 **整条文件链错位**。
- **改进建议**：
  1. 修正 `FileCollector._write_record()` 为 `payload_length = tag_len + len(payload)`；
  2. 修正 `reader.py` 为 `body_len = payload_len`，并校核 `if tag_len > payload_len: break`；
  3. 同步修正 `generate_python_golden.py`、`verify_golden.py`、`test_golden.py`、`test_file_collector.py`、`test_reader.py` 与 `synapse` [test_observability.py](../../tests/test_observability.py)，重新生成 `python_reference.tbevt`；
  4. **强制补建 C++ ↔ Python 双向交叉读写 Golden Gate**：新增 C++ GTest 读入 Python `python_reference.tbevt`，以及 Python `read_events()` 读入 C++ `FileCollector` 产物文件的双向互验。

### `TRB-P0-02` [aether] 洁净环境 `import` 必崩（漏 `typing_extensions` 依赖）、Wheel 污染 `tests` 包及 4 项 `mypy --strict` 隐匿错误

- **缺陷位置**：
  1. `aether/traceboard/python/aether_traceboard/probe.py:23` 在顶层无条件执行 `from typing_extensions import Self`，但 `aether/traceboard/python/pyproject.toml:27-29` `dependencies` 仅声明 `["numpy>=1.24.0"]`；
  2. `aether/traceboard/python/pyproject.toml:43-44` `[tool.setuptools.packages.find]` 仅写 `where = ["."]`，未排除 `tests*`，`setuptools.find_packages(where=".")` 实测返回 `['aether_traceboard', 'tests']`；
  3. `.github/workflows/traceboard-python-publish.yml:65-66` 对 `python -m mypy aether_traceboard/ --strict` 配置了 `continue-on-error: true`，实测掩盖了 4 处严格类型错误（`_hash.py:17` `[no-any-return]`，`event.py:43,86` `[unused-ignore]` + `[no-any-return]`）。
- **后果**：在仅安装 `aether-traceboard` + `numpy` 的洁净容器/Colab/CI 环境中执行 `import aether_traceboard` 直接抛出 `ModuleNotFoundError: No module named 'typing_extensions'`；打包 wheel 会污染其他仓库的顶层 `tests` 命名空间。
- **改进建议**：
  1. 从 `probe.py` 移除运行期 `from typing_extensions import Self`（将 `__enter__` 返回注解改为 `StageContext` 或置于 `if TYPE_CHECKING:`，彻底保持运行期零非 `numpy` 依赖）；
  2. 在 `pyproject.toml` 配置 `include = ["aether_traceboard*"]` 与 `exclude = ["tests*"]`；
  3. 修复 `_hash.py:17`（`int(struct.unpack("<Q", digest[:8])[0])`）与 `event.py:43,86` 的 4 项 `mypy --strict` 报错，并移除 `traceboard-python-publish.yml:66` 的 `continue-on-error: true`；在发布前 CI 增加 **洁净 venv 安装 wheel + `import aether_traceboard` 冒烟门禁**。

### `TRB-P0-03` [synapse] `BenchPipelineRunner` 在异常/门禁失败路径调用 `stage.set_fail(reason)` 必现 `TypeError` 崩溃

- **缺陷位置**：
  - [bench_runner.py](../../python/synapse/pipeline/bench_runner.py) 第 199 行 `stage.set_fail(report.error)` 与第 204 行 `stage.set_fail("gate_failed")`；
  - 对比真实 `aether/traceboard/python/aether_traceboard/probe.py:419-421` 定义：`def set_fail(self) -> None:`（**不接收任何位置参数**）。
- **后果**：一旦通过 CLI/Colab 开启 `--traceboard`，**任何场景发生运行错误或质量门禁不通过时**（最需要可观测性定位故障的时刻），`run_scenario()` 在退出阶段会直接抛出 `TypeError: StageContext.set_fail() takes 1 positional argument but 2 were given`，中断整个 `run_all()` 批量评测循环并绕过报告归档！此前因缺少 `BenchPipelineRunner` 的 Traceboard 失败路径单测与 E2E，该 Bug 逃逸了 PR `#212`。
- **改进建议**：
  1. 短期在 `bench_runner.py` 改为无参调用 `stage.set_fail()`（并用 `probe.log_text(f"bench/{scenario_name}/error", ...)` 记录错误详情），同时在 `run_all()` 外层 `stage` 上也补齐 `set_pass()`/`set_fail()`；
  2. 在 `aether_traceboard.StageContext.set_fail(self, reason: str | None = None)` / `set_skip(self, reason: str | None = None)` 增加向后兼容的可选 `reason` 参数（非空时写入 `STAGE_END` JSON `"reason"` 字段）；
  3. 在 `synapse` 为 `BenchPipelineRunner` 补充正常、运行异常与门禁失败三态的 **真实 SDK E2E 测试**。

### `TRB-P0-04` [synapse] CLI `--traceboard` 挂载位置错位、空命令句柄泄漏且丢失 `trace_id`/失败状态

- **缺陷位置**：
  - [cli/\_app.py](../../python/synapse/cli/_app.py) 第 68 行与第 187 行仅在顶层 `parser` 挂载 `add_traceboard_args(parser)`，未进入子命令共享的 `json_parent`（[cli/common.py](../../python/synapse/cli/common.py) 第 62–76 行）；
  - [cli/\_app.py](../../python/synapse/cli/_app.py) 第 100–122 行 & 第 210–232 行：先调用 `init_traceboard(...)`，遇到 `parsed.command is None` 时直接 `return 0`，跳过第 168–170 / 234–238 行的 `try...finally: shutdown_traceboard()`；
  - 第 104 / 214 行 `run_name` 自带 `_{int(time.time())}` 与 `FileCollector` 自带 `.{ts}.tbevt` 产生双时间戳冗余；未传入 `trace_ctx.trace_id`；在异常/非零退出码时仍以默认 `status="finished"` 结束 run。
- **实测后果**：
  - 按 `PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md` §6 原文命令执行 `synapse bench --traceboard /tmp/tblog list` 直接触发 `SynapseArgumentParser.error()` 返回退出码 `10`（`invalid choice: '/tmp/tblog'`）；`synapse bench list --traceboard /tmp/tblog` 报 `unrecognized arguments`；
  - 无子命令调用 `synapse --traceboard /tmp/tblog` 泄漏未关闭文件句柄，且不会写入 `RUN_END`；
  - 在测试或嵌套进程内调用 `main_synapse()`/`main_bench()`（未传 `--traceboard`）也会无条件在 `finally` 执行 `shutdown_traceboard()` 误关外层调用方已初始化的 Probe。
- **改进建议**：
  1. 在 [cli/common.py](../../python/synapse/cli/common.py) 提供共享 `common_parent`（使用 `default=argparse.SUPPRESS`），同时注册 `--json` 与 `--traceboard LOG_DIR`，挂载到顶层与所有域子解析器，确保前置、中置、后置写法全部兼容；
  2. 仅在 `parsed.command is not None` 后初始化 Traceboard，并用局部变量 `started_traceboard = True` 控制 `finally:` 中的 `shutdown_traceboard(status="finished" if rc == 0 else "failed")`；
  3. 向 `init_traceboard(..., trace_id=trace_ctx.trace_id)` 透传 W3C 追踪上下文，打通 `synapse` 结构化日志与 Traceboard 运行流关联。

### `TRB-P0-05` [aether] 多项 Payload Schema 与 `wire_format_spec.md` / C++ `probe.cc` 漂移

逐项对比 `aether/traceboard/docs/wire_format_spec.md`、C++ `aether/traceboard/src/probe.cc` 与 Python `event.py`/`probe.py`/`generate_python_golden.py`，发现 **5 处 Schema 不一致**：

| # | 事件类型 | Spec / C++ `probe.cc` 真值 | Python 当前实现问题 |
| :---: | :--- | :--- | :--- |
| 1 | `STAGE_END` (`243`) | `probe.cc:917`：`{"status": 2}` | `probe.py:454` 与 `generate_python_golden.py:175` 写 `"_status"`，多出前导下划线，C++/Go 读不到状态码 |
| 2 | `ARTIFACT` (`6`) | Spec §6.4.3 & `probe.cc:774-775`：`{"filename", "mime_type", "size_bytes"}` | `generate_python_golden.py:155` 写 `{"filename", "mime", "size"}`；`Probe.log_artifact` 未自动补齐 `"size_bytes": len(blob)`，导致 [report.py](../../python/synapse/bench/report.py) 第 370 行漏传 `size_bytes` |
| 3 | `TABLE` (`5`) | Spec §6.4.2 & `probe.cc:742`：单元格为字符串数组 `[][]string`；标准 RFC 8259 JSON（无 `NaN`/`Inf`） | `probe.py:306` 与 `generate_python_golden.py:150` 允许 `float`/`int` 混入，且 `event.py` 全局 `json.dumps()` 未禁用 `allow_nan=False`，一旦含 `NaN`/`Inf` 会写出非标 JSON 导致 Go `encoding/json` 解析失败 |
| 4 | `COMPARISON` (`4`) | `probe.cc:655-698`：含 `width/height/channels/format/thresholds`；按 `_min`/`_max`/上界指标集合自动计算 `pass`；无 `diff_image` 时**省略** `diff_image_tag` | `probe.py:292` 在 `passed=None` 时直接盲置 `True`，彻底忽略传入的 `thresholds`；第 285 行无 `diff_image` 时写 `"diff_image_tag": ""`；`generate_python_golden.py:136-144` 缺 `width/height/channels/format/thresholds` |
| 5 | `RUN_START` / `CONFIG` / `SERIES_GROUP` | Spec §6.9 强要求 `CONFIG`/`RUN_START` 所有值必须为 JSON 字符串；Spec §6.4.4 `SERIES_GROUP` 含 `"steps"` 数组 | `event.py` `pack_run_start`/`pack_config` 未做 `str()` 强转；`generate_python_golden.py:164-167` `SERIES_GROUP` 缺少 `"steps"` |

- **改进建议**：在 `event.py`、`probe.py` 与 `generate_python_golden.py` 全量对齐 `wire_format_spec.md` 与 `probe.cc`；引入统一 `_sanitized_json_dumps()`（`allow_nan=False`，先将 `NaN`/`±Inf` 按 C++ `formatJsonFloat` 清洗为 `0.0`）；在 `Probe.log_comparison` 实现与 C++ 对等的阈值自动判决；在 `Probe.log_artifact` 自动注入 `size_bytes` 并兼容规范化 `mime_type`。

### `TRB-P0-06` [aether] `FileCollector` 路径穿越、同秒覆写、句柄泄漏、线程竞态与越界缺失

对比 C++ `aether/traceboard/src/file_collector.cc`，Python `aether/traceboard/python/aether_traceboard/file_collector.py` 存在 **5 项安全与可靠性缺陷**：
1. **路径穿越与目录逃逸**：`file_collector.py:46` 直接拼接 `filename = f"{run_name}.{ts}.tbevt"`，未像 C++ `file_collector.cc:72-84` 将非 `[a-zA-Z0-9_-]` 字符清洗为 `_`（`run_name="../../evil"` 可穿越 `logdir` 写任意文件；`run_name="bench/run"` 直接报 `FileNotFoundError`）。
2. **同秒运行静默覆盖**：`int(time.time())` 仅精确到秒且未实现 C++ `file_collector.cc:96-98` 的 `_{suffix}` 去重循环，同秒内连续调用 `on_run_open` 用 `"wb"` 截断覆盖前序实验日志。
3. **句柄泄漏**：`on_run_open()` 若已有打开的 `_fp` 不先关闭直接覆写属性，且缺少 `__enter__`/`__exit__`/`__del__` 兜底关闭。
4. **多线程非线程安全**：`FileCollector`（3 次独立 `self._fp.write()` 写一条记录）与 `Probe`（`_step`、`_collectors`）没有任何 `threading.RLock()`，多线程训练/评估场景下会发生记录交错撕裂与 CRC 损坏。
5. **缺失长度守卫与冗余整块拷贝**：缺少 `tag_len <= 65535` 截断守卫与 `payload_length <= 0xFFFFFFFF` 溢位检查；第 82 行 `compute_crc32(tag_bytes + payload)` 对数 MB 图像/HTML Artifact 做整块字节拼接拷贝，应改用已有 `update_crc32(compute_crc32(tag_bytes), payload)` 实现零拼接增量 CRC；同时补齐与 C++ 对称的 `current_file_path` 属性。

---

## 4. P1 高危与工程硬化缺陷详单 (`TRB-P1-01` ~ `TRB-P1-09`)

### `TRB-P1-01` [aether] `Probe` 与 `StageContext` 生命周期行为与输入校验缺口

- **缺陷位置**：`aether/traceboard/python/aether_traceboard/probe.py`
  1. `begin_run()`（第 125 行）不主动激活 Probe，导致直接用官方用法先 `begin_run` 再 `activate` 的用户产出“只有 16B Header、丢掉 `RUN_START`”的半残文件；`end_run()`（第 135–146 行）不调用 `deactivate()` 且不清空 `_run_name`；
  2. 缺少与 C++ `ProbeContext` 对称的 `remove_all_collectors()`、`increment_step()`、`step` 与 `run_name` 属性；
  3. `StageContext.__exit__()`（第 445–455 行）：正常无异常退出且未显式调 `set_pass()`/`set_fail()`/`set_skip()` 时，状态永远停留在 `StageStatus.RUNNING (1)`（`with traceboard_stage(...)` 常见用法导致全部 stage 被标为永久运行中），且用非单调 `time.time()` 计算耗时（受系统时钟/NTP 跳变影响可能产出负 duration）；
  4. `log_image()`（第 188–197 行）未校验 `channels in (1, 3, 4)` 与 `h > 0, w > 0`，且在大端或非本机字节序 NumPy 数组时 `tobytes()` 不做 `<f4`/`<f2` 小端强转；
  5. `log_histogram()`（第 223–226 行）未校验 `len(edges) == len(counts) + 1`、`1 <= num_bins <= 65536`（与 C++ `file_collector.cc:238` 一致），空直方图或维度不匹配时静默写出损坏记录。
- **改进建议**：对齐 C++ `ProbeContext` 生命周期与校验规则：`begin_run()` 完成后自动 `activate()`（保留 `auto_activate: bool = True` 参数）；`end_run()` 调用 `deactivate()`、flush 后关闭 collectors 并重置 `_run_name`；`StageContext` 改用 `time.monotonic()` 测耗时并在无异常退出且状态仍为 `RUNNING` 时默认落为 `StageStatus.PASS`（保留显式 `set_fail`/`set_skip` 覆盖）；补齐 `log_image`/`log_histogram` 契约校验与小端字节序强转。

### `TRB-P1-02` [aether] `reader.py` 前向兼容失效与大文件整读 OOM 风险

- **缺陷位置**：`aether/traceboard/python/aether_traceboard/reader.py` & `__init__.py`
  1. 第 82–86 行 `unpack_record_header()` 直接调用 `EventType(et)`，遇 Spec §9 定义的未来扩展事件码（如 `128..239`）抛 `ValueError` 后直接 `break` 丢弃文件后续全部合法事件——直接违反 `wire_format_spec.md` §9「遇到未知 EventType 必须按 `payloadLength` 跳过并继续读取」的强制条款；
  2. 第 66 行忽略 `FileHeader.version` 校验（不检查 `version > 1`）；且仅返回 `list[Record]`，未返回 `FileHeader`（`run_id_hash`/`version`）及截断/CRC 损坏统计（Spec §8.2）；
  3. 第 59 行 `Path(path).read_bytes()` 一次性将整文件读入内存，若含 `IMAGE` 或大 `ARTIFACT` 易引发内存峰值飙升；同时 `reader.py` 中的 `Record`、`FileHeader`、`read_file` 在 `__init__.py` `__all__` 中未实际定义或导出（`PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md` §3.1 文档已提前声明了 `FileHeader` 与 `read_file`）。
- **改进建议**：提供流式生成器 `iter_events(path)`（逐条读取并按 `payload_len` 跳过未知 `EventType`），补齐 `FileHeader` / `ReadStats` 数据类与 `read_file(path) -> tuple[FileHeader, list[Record], ReadStats]`，并在 `__init__.py` 完整导出。

### `TRB-P1-03` [aether] `traceboard-python-publish.yml` 存在仓库混用、错误静默吞掉与触发条件互斥

- **缺陷位置**：`aether/.github/workflows/traceboard-python-publish.yml`
  1. 第 127 行环境变量 `TWINE_REPOSITORY_URL: https://npm.pkg.github.com/` 与第 130 行 `--repository-url https://upload.pypi.org/legacy/` + `secrets.GITHUB_TOKEN` 混用（向 PyPI 传 `GITHUB_TOKEN` 必报 403，且 `npm.pkg.github.com` 仅支持 npm/Python 无关格式）；
  2. 第 131 行 `--skip-existing || true` 在上传失败时仍以 `0` 退出并打印 `"Package published"`（典型 CI 假绿）；
  3. 第 10–15 行 `push` 同时设置 `branches: [main]`、`tags: ["traceboard-python-v*"]` 与 `paths:` 过滤（在 GitHub Actions 中 tag push 若带 `paths:` 过滤常存在非预期跳过风险，且 `paths:` 漏列了 `traceboard/tests/golden/**` 与 `traceboard/src/**`）；
  4. 未对接 DervGen 生态三级分发架构（`files.dervgen.com` 私有源 / GitHub Releases / PyPI OIDC Trusted Publishing）。
- **改进建议**：拆分为 `test`（对 PR/push 生效）与 `publish`（仅由 `tags: ["traceboard-python-v*"]` 无 `paths` 限制触发），移除 `|| true` 与 `npm.pkg.github.com`，增加 `twine check` + 洁净 venv wheel 安装冒烟，并规范发布产物至 GitHub Release Assets 与可选 PyPI OIDC / 内部 wheel 源。

### `TRB-P1-04` [synapse] `synapse.observability` 初始化残留坏状态、缺锁保护与埋点无异常隔离

- **缺陷位置**：[observability.py](../../python/synapse/observability.py) 第 86–146 行及全库 14 处业务埋点
  1. **失败路径残留坏单例**：第 103 行先赋值全局 `_probe = ProbeCls()`，再于第 104–108 行构造 `FileCollector` 与调 `begin_run()`。若 `log_dir` 不可写或 `begin_run()` 抛异常，第 112–117 行虽捕获异常返回 `None`，却**把半初始化/未关闭的 `_probe` 留在了全局变量中**，导致后续所有 `get_probe()` 拿到坏 Probe 并持续报错或泄漏文件句柄；
  2. **缺锁保护**：`init_traceboard()` / `shutdown_traceboard()` 无 `threading.Lock`；
  3. **`shutdown_traceboard()` 缺少运行状态参数**：无法向 `probe.end_run(status=...)` 透传 `"finished"` / `"failed"` / `"cancelled"`；
  4. **业务埋点未做 Fail-Safe 隔离**：14 处源码埋点直接调用 `probe.log_scalar(...)` / `traceboard_stage(...)`（如 [bench_runner.py](../../python/synapse/pipeline/bench_runner.py)、[trainer.py](../../python/synapse/training/trainer.py)），若 Traceboard I/O（磁盘满、权限变动、非有限值序列化等）抛异常会直接打断核心渲染/训练/评测业务；
  5. **双全局状态边界**：`synapse.observability._probe` 与 `aether_traceboard.get_probe()` 为两份独立全局实例，需在文档与 API 层面明确禁止混用（或统一桥接）。
- **改进建议**：先在局部变量构建 `probe`、加锁并在 `begin_run()` 成功后原子提交 `_probe = probe`（失败时显式回收关闭）；为 `shutdown_traceboard(status: str = "finished")` 增加状态入参；在 `traceboard_stage()` 与新增 `safe_log_*` 辅助中隔离日志异常，确保可观测性故障零影响主流程。

### `TRB-P1-05` [synapse] 训练与课程模块缺失 `step` 递增且存在重复日志

- **缺陷位置**：
  - [trainer.py](../../python/synapse/training/trainer.py) 第 497–500 行（`save_state_dict`）与第 719–725 行（`train_step(..., step=...)`）：虽然拿到了训练 `step`，但从未调用 `probe.set_step(step)` 也未向 `log_scalar(..., step=step)` 传参；
  - [losses.py](../../python/synapse/training/losses.py) 第 364–373 行（`DiffusionDPOLoss.__call__`）：依赖当前步数但 `trainer` 未先 `set_step`；
  - [curriculum.py](../../python/synapse/rsi/curriculum.py) 第 251–262 行（`sample_scenarios(..., round_idx=...)`）与第 330–341 行（`plan_round()`）：未传 `step=round_idx`，且 `plan_round()` 在调用 `self._sampler.sample_scenarios()` 后**原样重复写了一遍完全相同的 `round_idx`/`num_scenarios`/`factor_distribution` 事件**。
- **后果**：多步训练/多轮课程博弈的所有 `train/loss`、`train/dpo_preferred_err`、`rsi/curriculum/*` 全部塌缩在 `step=0`，Traceboard 时间序列图退化为单点垂直堆叠。
- **改进建议**：在 `train_step()` 与 `save_state_dict()` 显式调用 `probe.set_step(step)` 与传参 `step=step`；在 `curriculum.py:sample_scenarios()` 传 `step=round_idx` 并删除 `plan_round()` 中的重复埋点块。

### `TRB-P1-06` [synapse] 高基数 ID 拼接 Tag 造成 Tag 空间爆炸

- **缺陷位置**：
  - [dual_verifier.py](../../python/synapse/rsi/dual_verifier.py) 第 382–394 & 471–472 行：`rsi/verifier/physics_passed/{sample_id}`、`rsi/verifier/{check_name}/{sample_id}`、`rsi/verifier/failures/{sample_id}`、`rsi/verifier/prm_score/{sample_id}`、`rsi/verifier/verdict/{sample_id}`；
  - [jev_critic.py](../../python/synapse/inspect/jev_critic.py) 第 579–598 行：`inspect/jev/{asset_id}/overall_quality` 等与 `inspect/jev/{asset_id}/defects`。
- **后果**：每验证 1,000 个样本或质检 500 个 3D 资产就会衍生数千上万个单点孤立 Tag，无法按指标聚合趋势曲线，且会撑爆前端看板侧边栏与查询索引。
- **改进建议**：收敛为**低基数固定指标 Tag + 单调递增 `step`**（如 `rsi/verifier/prm_score`、`inspect/jev/overall_quality`），并将每个 `sample_id` / `asset_id` 明细汇总或追加进结构化 `TABLE`（如 `rsi/verifier/verdicts`、`inspect/jev/defects`）。

### `TRB-P1-07` [synapse] Bench 实时埋点与离线导出 Schema 分裂、逻辑重复及资源泄漏

- **缺陷位置**：
  - **两套命名空间互不兼容**：[bench_runner.py](../../python/synapse/pipeline/bench_runner.py) 第 186–194 行写入扁平 `bench/{scenario}/{metric|timing}`（且未过滤 `NaN`/`Inf`），而 [traceboard_export.py](../../python/synapse/bench/traceboard_export.py) 第 97–127 行写入 `scenario/{scenario}/metrics|timing/{key}` + `summary/*` + `pillar/*`，导致 `--traceboard` 实时输出与离线导出报告无法在 Traceboard 看板同图对比；
  - **DRY 违反**：`traceboard_export.py:144-219` 整段复制了 [report.py](../../python/synapse/bench/report.py) 第 377–455 行的 `_compute_pillar_scores`（已有两处独立维护风险）；
  - **Docstring 与实现不符 + FD 泄漏**：`traceboard_export.py:37` 声称 "one stage per scenario"，实际并未创建任何 `begin_stage`；且 `close_on_finish=False` 分支直接返回 `True` 并丢弃局部变量 `probe` 引用，导致未关闭的 `FileCollector` 文件描述符泄漏。
- **改进建议**：将 `report.py` 的 4 支柱得分算法提升为公开 SSOT 函数供 `traceboard_export.py` 复用；统一两轨 Tag 规范（`bench/scenario/{name}/metrics/{k}` 等）与 `NaN` 过滤；在 `traceboard_export.py` 按场景真实打开 `begin_stage`，并移除悬空 `close_on_finish=False`（或改返回已关闭 `.tbevt` 文件路径 / 支持注入外部 `Probe`）。

### `TRB-P1-08` [synapse] `colab_runner.py` 违反存储分离铁律与 `run_cpu_verification.py` 缺失 `finally` 刷盘

- **缺陷位置**：
  - [colab_runner.py](../../scripts/colab_runner.py) 第 846 行：当 `job_dir` 位于 Google Drive FUSE（`/content/drive/MyDrive/dervgen/...`）时，直接向 Drive FUSE 路径写高频 append-only `.tbevt` 违反 `AGENTS.md` §2 与 `.agents/skills/google-colab-environment/SKILL.md`「计算在 NVMe `/content/`，完成后再持久化回 Drive」铁律（FUSE 断流易损毁尾部记录）；第 811–813 行 `--help` 文案承诺记录 `"GPU profile, and experiment results"`，但代码**从未记录 GPU 配置与各实验指标/Stage**；异常与 SIGTERM 抢占退出路径也未上报 `"failed"` / `"cancelled"` 状态；
  - [run_cpu_verification.py](../../scripts/run_cpu_verification.py) 第 505–620 行：E1–E5 验证主体与 `shutdown_traceboard()` 未放入 `try...finally`，任一实验异常或 SHA-256 校验失败（第 542、550 行 `return 20`）会直接跳过 `shutdown_traceboard()`，且全程未创建 E1–E5 分实验 stage 与分阶段耗时。
- **改进建议**：`colab_runner.py` 在 `/content/work/{job_id}/traceboard/` 写入 `.tbevt` 并在每次实验 checkpoint 及 `finally` 刷盘后原子同步回 `{job_dir}/traceboard/`，补齐 `gpu_profile` 配置与 `colab/{exp}` stage 埋点；`run_cpu_verification.py` 用 `try...finally` 保证无论成功或早退（`20`/异常）均执行 `shutdown_traceboard(status=...)` 并记录分实验 stage 耗时。

### `TRB-P1-09` [synapse] 测试覆盖盲区（无 `spec` Mock、永真断言、缺 Bench 失败路径与 CI 产物校验）

- **缺陷位置**：
  1. [test_traceboard_export.py](../../tests/test_traceboard_export.py) 第 214–215 行 `test_returns_false_when_not_installed` 写永真断言 `assert isinstance(result, bool)`（`True` 和 `False` 都通过！）；
  2. [test_training_traceboard.py](../../tests/test_training_traceboard.py)、[test_rsi_recon_traceboard.py](../../tests/test_rsi_recon_traceboard.py)、[test_adapters_style_traceboard.py](../../tests/test_adapters_style_traceboard.py) 全部使用裸 `MagicMock()` 而无 `spec`，无法在缺少真实 SDK 时拦截类似 `TRB-P0-03` 的签名错误；
  3. **0 项** `BenchPipelineRunner` Traceboard 埋点单测与真 SDK 失败路径测试；
  4. [.github/workflows/ci.yml](../../.github/workflows/ci.yml) 未增加运行 `synapse --traceboard ... bench run` 并校验产出 `.tbevt` 完整性与上传 artifact 的 CI 步骤（另记录：公网 Runner `python-fast` 上 `tests/test_wave4_rfc0060_rfc0061_parity.py` 存在 3 项 `#203` 既有未标 `@pytest.mark.external_data` 的原生扩展失败，需独立止血使 CI 全绿）。
- **改进建议**：消除永真断言；定义 `ProbeProtocol` / `StageProtocol` 并将测试 mock 切换为 `MagicMock(spec=...)`；补全 `BenchPipelineRunner` 三态 E2E、`StatelessLoRATrainer` 多步 `step` 递增验证以及 CI `.tbevt` 产线校验与 artifact 归档。

---

## 5. P2 中优先级改进、文档失真与跨仓生态协同 (`TRB-P2-01` ~ `TRB-P2-05`)

### `TRB-P2-01` PR #213 与关联文档实态偏差对照表

| 文档路径 | 原文陈述 | 实际代码 / 工具验证真值 | 整改要求 |
| :--- | :--- | :--- | :--- |
| [PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md](../plans/PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md) §6 (L266) | `synapse bench --traceboard /tmp/tblog list` | `argparse` 报 `invalid choice: '/tmp/tblog'` 退出码 `10` | 修复 CLI 共享参数后再验证并修正示例 |
| [PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md](../plans/PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md) §3.1 (L129) | 导出 `Record`, `FileHeader`, `read_file`, `read_events` | `aether_traceboard/__init__.py` `__all__` **未导出**这 4 项，`FileHeader`/`read_file` **尚未实现** | `PR-TR02` 补齐实现与导出 |
| [PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md](../plans/PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md) §0 (L29) | `aether`: 154 passed · `mypy --strict` 0 错 | `aether_traceboard` 在 `ubuntu24` 实测 `mypy --strict` **报 4 errors** | `PR-TR03` 清零报错并去除 CI `continue-on-error` |
| `aether/traceboard/python/README.md` (L3, L25, L91, L106) | 引用 `../../docs/traceboard_architecture.md`、`../../docs/wire_format_spec.md`、`server/`、`https://npm.pkg.github.com/` | 上述路径（相对位置多了一层 `../`）与 `server/` 目录**均不存在**，且 GitHub Packages 不支持 npm registry 托管 Python wheel | `PR-TR03` 修正为 `../docs/wire_format_spec.md` 与真实分发地址 |
| `aether/traceboard/python/pyproject.toml` (L41) | `Documentation = .../traceboard/docs/traceboard_architecture.md` | 该文档不存在（存在的是 `wire_format_spec.md`） | `PR-TR03` 修正为 `wire_format_spec.md` |
| `aether/traceboard/docs/wire_format_spec.md` (L81, L560) | 指向 `../python/traceboard/probe.py`、`../python/traceboard/event.py` | 实际包目录为 `../python/aether_traceboard/` | `PR-TR03` 同步路径 |

### `TRB-P2-02` 脚本与主 CLI `--traceboard` 形态统一

- [scripts/colab_runner.py](../../scripts/colab_runner.py) 与 [scripts/run_cpu_verification.py](../../scripts/run_cpu_verification.py) 当前为布尔开关，而主 CLI 为 `--traceboard LOG_DIR`。建议统一升级为 `--traceboard [LOG_DIR]`（`nargs="?"`, `const="__AUTO__"`, `default=None`）：不传路径时延续当前默认目录，传入路径时尊重用户指定目录，兼顾自动化脚本易用性与 CI/自动化目录可控性。

### `TRB-P2-03` 核心缺失链路与富媒体事件（`IMAGE` / `COMPARISON` / `HISTOGRAM`）零使用

当前 9 子系统埋点几乎全部停留在标量与零星文本/表格，Traceboard 专为渲染/视觉生成设计的 **`IMAGE`、`COMPARISON`、`HISTOGRAM`、`SERIES_GROUP` 四大核心能力在 `synapse` 中使用率为 0%**：
1. **`synapse worker run --bundle`（RFC 0039 生产入口）零埋点**：[bundle_worker.py](../../python/synapse/pipeline/bundle_worker.py) 未支持从 `JobBundle` 或环境变量读取 `traceboard_dir`，生产容器单步 GPU 任务无法产出 `.tbevt`；
2. **`ConditionScheduler`（97 帧因果滑窗）零埋点**：[scheduler.py](../../python/synapse/adapters/scheduler.py) 缺失切窗索引、重叠帧相位、`shot_cut` 触发率与窗耗时埋点；
3. **RFC 0061 局部修补前后对比未接 `log_comparison`**：`synapse.refine` 最适合通过 `probe.log_comparison(before, after, metrics, thresholds)` 产出可视化 A/B 对比（需限制分辨率上限与抽帧开关以防 `.tbevt` 体积膨胀）；
4. **`bench_runner` 关键帧与深度残差分布**：在显式开启时抽样记录关键帧 `log_image` 与深度误差 `log_histogram`；`compare_runs` / `render_comparison_html` 补记 4 轨道对比 HTML artifact。

### `TRB-P2-04` 架构设计文档补强

- [DESIGN_TRACEBOARD_INTEGRATION.md](../DESIGN_TRACEBOARD_INTEGRATION.md) 需补充：**Tag 命名规范（Tag Taxonomy & 低基数约束）**、**Step 单调性规约**、**W3C `traceparent` 与 `runId` 关联规范**、**非阻塞 Fail-Safe 异常边界**以及 **Colab NVMe-First 同步策略**。

### `TRB-P2-05` 跨仓生态协同项

1. **`caelum-core` 遗留 API 脱节**：同生态 `caelum-core/tests/helpers/traceboard_integration.py` 仍在尝试 `from traceboard import Probe` 并调用已废弃签名 `Probe(logdir, run_name=..., config=...)`、`probe.scalar()`、`probe.image()`、`probe.close()`，与 `aether-traceboard` 完全不兼容，需提 Issue / PR 协同迁移；
2. **`aether` 加密原语去重护栏 (`DUP-17`)**：`all-overview/architecture/cross-repo-code-duplication-and-dedup-todo.md` 规划将 `aether/traceboard/src/event.cc` 散装 `SHA-256` / `CRC-32` 收口至 `@aether//crypto`，务必在 `PR-TR01` 建立的 C++ ↔ Python 双向黄金测试之后再重构 `event.cc`；
3. **Go Traceboard Server 独立仓对接**：待服务端仓可见后，补跑 Go `TestGoldenPython` 与 Python 产出的 `TABLE`/`ARTIFACT`/`COMPARISON`/`STAGE_END` 全事件解析联调。

---

## 6. 改进路线与验收门禁总览

针对上述 20 项问题，已同步制定可分批原子落地、带严格前置依赖与防静默失败门禁的实施蓝图：[PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md](../plans/PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md)：

- **Wave TR-1 (`dervgen/aether`, `PR-TR01 ~ PR-TR03`)**：修复 `.tbevt` `payloadLength` 线格式、对齐 5 类 Payload Schema 与 C++ ↔ Python 双向黄金测试；加固 `FileCollector`/`Probe`/`Reader` 并发与安全边界；修复包依赖、`mypy --strict` 与 Publish CI + 洁净 venv 冒烟。
- **Wave TR-2 (`sautille-stroke/synapse`, `PR-TR04 ~ PR-TR06`)**：止血 `BenchPipelineRunner` `set_fail` 崩溃与 CLI 子命令 `--traceboard` 挂载；收敛 `step` 递增、低基数 Tag 与双轨 SSOT；修复 `colab_runner.py` NVMe 分离、`run_cpu_verification.py` `finally` 刷盘、消除假测试并落地 CI `.tbevt` 端到端校验。
- **Wave TR-3 (`synapse` + 跨仓收口, `PR-TR07 ~ PR-TR08`)**：打通 `bundle_worker`、因果滑窗与 RFC 0061 `log_comparison` 富媒体埋点；全量通过 **10 项反作弊封板门禁 (`TB-1 ~ TB-10`)**，刷新文档台账、发布 `aether-traceboard` 正式包并推动 `caelum-core` 对齐。
