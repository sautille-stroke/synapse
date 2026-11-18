# Synapse Horizon-5 收官后全库深度复审报告（2026-11）

- **文档状态**: `Accepted Audit Report (v1.0)`
- **审计基线**: `main` @ `e37bfa6`（PR #69 `"docs(skills): sync ecosystem skills and mark Horizon-5 100% complete"` 合入后全量代码基线）
- **审计日期**: `2026-11-15`
- **操作模型标识**: `[opus4.6]`
- **关联修复计划**: [docs/plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md)（Horizon 6: `M21–M24` / `PR-74 ~ PR-91`）
- **关联总计划**: [docs/IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md)
- **前序基线审计**: [AUDIT-POST-REFACTOR-2026-11.md](AUDIT-POST-REFACTOR-2026-11.md) · [AUDIT-RFC0063-AND-CODEBASE-2026-11.md](AUDIT-RFC0063-AND-CODEBASE-2026-11.md)

---

## 1. 执行摘要与 `ubuntu24` 实测基线 (`e37bfa6`)

### 1.1 代码库定量规模统计

在 `HEAD = e37bfa6fec8f2b154c430c3ed05090d634711e25` 基线上对仓库全目录进行了只读代码扫描与 AST/C++/JS/CI/文档交叉验证，同时对照上游 `../aether` 与 `../all-overview` SSOT 契约，得到当前代码规模与结构组成如下：

| 目录 / 子系统 | 文件数 / 规模 | 职责分布 |
| :--- | :--- | :--- |
| **`python/synapse/`** | **135 文件 / 34,433 LOC**（19 个包域） | `recon` (4,192) · `inspect` (3,503) · `adapters` (3,264) · `refine` (2,764) · `cli` (2,695) · `render_core` (2,381) · `pipeline` (2,247) · `bench` (2,201) · `metrics` (1,834) · `training` (1,740) · `composite` (1,361) · `rsi` (1,324) · `fgpp` (1,195) · `viewer` (924) · `style` (808) · `generators` (616) · `<root>` (550) · `models` (418) · `bridge` (416) |
| **`csrc/geometry/`** | **516 LOC** C++20（Bazel 9.3.1） | 几何校验导出与 pybind11 绑定（头文件委托 `@aether//geo` + `@aether//math` `DVec3`） |
| **`tools/data-inspector/`** | **1,341 LOC** C++20 + **1,873 LOC** JS | `native/`（独立 Bazel）+ `electron/` + `renderer/` + `test/` |
| **`scripts/` & `tools/*.py`** | **2,767 LOC** (`scripts/`) + **894 LOC** (`tools/`) | 实验/资产脚本、`wire_guard.py`、`import_dag_guard.py` |
| **`tests/`** | **56 Python 测试文件 / 20,494 LOC** | `ubuntu24` pytest 实收集 **1,098** 项（`1,062` fast + `36` slow）；另有 `tools/data-inspector/test/` JS 测试与 2 组 C++ 测试 |

### 1.2 `ubuntu24` 云端主力机实跑基线（`e37bfa6`）

| 检查项目 | `ubuntu24` 执行命令 | 实测结果 | SLA / 契约要求 | 判定 |
| :--- | :--- | :--- | :--- | :---: |
| **快速单元套件** | `/root/venv-synapse/bin/pytest tests/ -m 'not slow' -q --durations=10` | `1062 passed, 36 deselected, 1 warning in 64.89s`（`metrics/temporal.py:222` 触发 `LinAlgWarning: Matrix is singular`） | `<= 30s`（RFC-0063 目标 `<= 20s`） | ❌ **超标 2.16x–3.24x** |
| **慢速套件** | `/root/venv-synapse/bin/pytest tests/ -m 'slow' -q --durations=10` | `36 passed, 1062 deselected in 518.38s (0:08:38)`（3 个全量测试重复 CPU 软光栅化全部 30 场景：`198.63s + 114.97s + 66.02s`） | 共享夹具去重 | ⚠️ **三重冗余耗时** |
| **Ruff 静态检查** | `/root/venv-synapse/bin/ruff check python/ tests/ scripts/ tools/` | `All checks passed!`（但 `[tool.ruff.lint] select` 仅启 `E4,E7,E9,F,I`） | 全规约检查 | ✅ / ⚠️ **规则偏窄** |
| **Mypy 默认配置** | `/root/venv-synapse/bin/mypy python/synapse` | `Success: no issues found in 135 source files`（非 `--strict`） | 全仓 `strict` | ⚠️ **掩盖 strict 报错** |
| **Mypy `--strict`** | `/root/venv-synapse/bin/mypy --strict python/synapse` | `Found 79 errors in 25 files`（`synapse.bench.*` 与 `bridge.hitl` 缺 `__all__` 触发 `[attr-defined]`、`[type-arg]`、`[unused-ignore]`、`[redundant-cast]`） | 0 error | ❌ **违反 strict 门禁** |
| **Mypy tests/scripts/tools** | `/root/venv-synapse/bin/mypy tests/ scripts/ tools/` | 立即 **`EXIT=2` 崩溃**（`c_client.runfiles` 缺 `exclude` 过滤）；加上 `--exclude 'bazel-\|\.pytest_tmp\|runfiles'` 后仍报 `Found 63 errors in 17 files` | 0 error（PR-72 宣称） | ❌ **门禁失真** |
| **跨语言线守卫** | `/root/venv-synapse/bin/python tools/wire_guard.py --check` | 退出码 `0`（但 `_RE_PIXEL_TYPE` 匹配 0 行、`@aether//style` 不存在而静默跳过、`bundle` 版本分裂漏检） | 全契约硬校验 | ⚠️ **False-Green 盲区** |
| **包级 DAG 守卫** | `/root/venv-synapse/bin/python tools/import_dag_guard.py --check` | 退出码 `0`（但跳过 `synapse/__init__.py`、遗漏相对导入、放行 `inspect->bench` 且未查 5-Tier 跨层逆向边） | 5-Tier 无环与单向 | ⚠️ **False-Green 盲区** |
| **GitHub Actions CI** | 检查 `.github/workflows/` | **目录完全不存在** | PR 自动门禁 | ❌ **零自动 CI** |
| **Inspector 测试副作用** | `ubuntu24` 工作区 `git status` | `tools/data-inspector/test/fixtures/{full,minimal}-bundle/bundle.json` 被测试原地改写时间戳，且未跟踪 `native/MODULE.bazel.lock` | 测试只读不脏树 | ⚠️ **污染工作区** |

### 1.3 重构正向成果（What Went Right）

1. **包边界重构与 CLI 模块化基本成型**：原先 2,211 行的单体 `cli/__init__.py` 已拆分为 11 个职责清晰的领域子模块；`synapse.render_core`（2,381 LOC）成功从 `bench/` 剥离了软光栅器、相机与基础几何生成器。
2. **顶层零重型依赖纪律稳固**：全量 `python/synapse/` 源文件均保持 `torch`、`openexr`、`cv2`、`mitsuba`、`safetensors` 函数内按需懒加载，基础导入无显存/权重隐式分配。
3. **Agent-Native 错误编码体系完备**：[errors.py](../../../python/synapse/errors.py) 与 [telemetry.py](../../../python/synapse/telemetry.py) 统一实现了确定性退出码 `ExitCode (0/10/20/30/40/50)`、`agent_fix_hint` 与 W3C `TraceContext`。
4. **C++20 原生几何核减重完成**：[csrc/geometry](../../../csrc/geometry/) 已将底层几何检查去重委托给 `@aether//geo`，补齐了 N-gon 扇形剖分与 pybind11 `gil_scoped_release`。

### 1.4 多维健康度评分卡（Post-Horizon-5 真值校准）

| 审查维度 | 得分 | 核心诊断结论 |
| :--- | :---: | :--- |
| **跨仓 Wire 契约与 Schema 一致性** | **64 / 100** | `Cryptomatte.ID` 在 EXR 中误写 `FLOAT` 存在 sNaN 位损；`InspectionBundle` 四方契约分裂；`wire_guard` 存在多项正绿盲区 |
| **仓内 5-Tier 依赖拓扑 (DAG)** | **72 / 100** | 直接环路已破，但 `synapse/__init__.py` 仍全量急切导入全部 13 个子包，5-Tier 跨层逆向依赖（`bridge->recon`、`inspect->bench,rsi`、`recon->refine`）与相对导入漏洞仍在 |
| **算法真实性与防静默失败三问** | **54 / 100** | `MultiLoRAManager` 注入随机噪声、`FoundaryHttpTransport` 丢弃 3D 条件、`bundle_worker` 仅读元数据零像素、`delight` 恒等 `rad/rad=1`、`FeatureStore` 100% 误拒、`FVD` 9 维 RGB 均值替身 |
| **原生 C++20、Bazel 与 Data Inspector** | **50 / 100** | `inspector_app.cc` 违背铁律 #9 泄露 `<poll.h>/<unistd.h>` + Windows 死锁 + OBJ 堆越界读 `SIGSEGV` + 缺 Z-Buffer/GLB/EXR；`test_inspector.cc` 在 `-DNDEBUG` 下断言 100% 蒸发；Electron SHM 在 macOS/Windows 100% 失效 |
| **测试质量、Colab/L-Asset 与 CI 基建** | **58 / 100** | 缺失 `.github/workflows/`；fast pytest `64.89s`；存在恒真/失明单测；`colab_runner` E3 必崩且偏离 SSOT V2.0；`stage_from_remote_huggingface` 为空桩 |
| **文档、RFC 与治理真实性** | **68 / 100** | `SYN-RFC-0061..0063` canonical 指针仍指向冲突编号；`PR-59/62/63/64/65/66/67/68/69/70/72` 均留有未闭环缺口 |
| **综合健康度总评** | **61 / 100** | **未达生产闭环（Needs M21–M24 Hardening）** —— 在进入 `foundary` / `render-farm` 真实投产前须完成 Horizon-6 (`PR-74 ~ PR-91`) |

---

## 2. Horizon 5 (`PR-58 ~ PR-73`) 宣称 vs 代码实态逐项核验表

> 按 `AGENTS.md §1` 红线 #4（**存在问 · 接线问 · 执行问**）逐一对照 `HEAD = e37bfa6` 源码与 `ubuntu24` 实测输出重新定级：

| PR | 计划宣称状态 | 存在 | 接线 | 执行 | 审计实态 | 实测未闭环 / 回归缺口（均已转入 Horizon 6） |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`PR-58`** | ✅ 已合入 (#51) | ✅ | ✅ | ⚠️ | **⚠️ 部分合入** | `docs/RFC-0061..0063` 第 4 行仍写 `Ecosystem canonical: all-overview/rfcs/0061..0063`（指向压缩/L-Asset/Runtime 等无关生态提案）；`RFC-0063` 仍缺作者/日期/备选方案元数据 (`PR-91`) |
| **`PR-59`** | ✅ 已合入 (#53) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `wire_guard.py:411` `_RE_PIXEL_TYPE` 匹配 `0` 行（`pixel_type_values == {}`），未校验 C++ 逐通道类型与顺序，`@aether//style` 目录不存在而静默绿灯；`bundle.v1.0.schema.json` 与 `bundle_exporter.py` 版本常量互斥 (`PR-74`) |
| **`PR-60`** | ✅ 已合入 (#54) | ✅ | ✅ | ⚠️ | **⚠️ 部分合入** | `cli/_recon`/`_refine` 移除随机数，但仍构造全零/中心 50% ROI 与 `MockJevTransport`；`refine/hand.py` / `face.py` 无先验默认无操作直通 (`PR-81`, `PR-87`) |
| **`PR-61`** | ✅ 已合入 (#55) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `recon/delight.py` 仍将 `rad` 同时作 `albedo` 与 `radiance`（`rad / rad == 1.0` 恒等空操作）、HDR 浮点误除 `255.0`；`texture_extractor.py` 非对称 TV、`_spatial_dims` 维度反转 (`PR-79`, `PR-82`) |
| **`PR-62`** | ✅ 已合入 (#56) | ✅ | ❌ | ❌ | **⚠️ 部分合入** | `scripts/build_native.py` 非 Bazel 直编漏加 `aether/core/include` 与 `aether/spatial/include`，真实编译必失败（被 `--dry-run` 单测掩盖），且隐式偷跑 `pip install` (`PR-84`) |
| **`PR-63`** | ✅ 已合入 (#57) | ✅ | ⚠️ | ⚠️ | **⚠️ 部分合入** | `import_dag_guard.py` 显式跳过 `synapse/__init__.py`（该文件急切导入全部 13 包）、漏检相对导入与动态 `import_module`，并继续放行 `inspect -> bench` 等跨 Tier 边 (`PR-89`) |
| **`PR-64`** | ✅ 已合入 (#58) | ✅ | ❌ | ❌ | **⚠️ 部分合入** | `inspector_app.cc` 泄露 `<poll.h>/<unistd.h>`（违背铁律 #9）并在 Windows 析构死锁；OBJ 加载越界读 `SIGSEGV`、缺 Z-Buffer、零 `.glb/.exr`；`test_inspector.cc` 用 `<cassert>` 在 `opt` 下 100% 蒸发 (`PR-85`) |
| **`PR-65`** | ✅ 已合入 (#59) | ✅ | ❌ | ❌ | **⚠️ 部分合入** | `shm-reader.js` 找临时目录文件在 macOS/Windows 上 100% 失效、Linux 仍整帧 IPC 拷贝；`app.js` 渲染 `[object Object]` 且点击 Tab 触发 `set_debug_view` 未捕获异常 (`PR-86`) |
| **`PR-66`** | ✅ 已合入 (#61) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `stage_from_remote_huggingface` 返回 `[]` 空桩；`setup_colab.sh` 自造 `data/video/base_models` 并向 Drive 写合成 OBJ 违反 SSOT V2.0；`assets_manifest.json` 缺 `vital_file` (`PR-88`) |
| **`PR-67`** | ✅ 已合入 (#62) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `compute_fvd_from_videos` 实际计算 9 维 RGB 均值/差分伪 FVD；`f_score_5mm` 与 `watertight_pct` 漏入 `_METRIC_DIRECTIONS` 导致 `100%` 水密判 FAIL；`fvd_feature_distance` 缺正则触发奇异告警 (`PR-83`) |
| **`PR-68`** | ✅ 已合入 (#63) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `bench_runner.evaluate_bundle` 因 `ndarray or ndarray` 抛 `ValueError` 崩溃，跳过未产出指标；`model_evaluator` 丢弃 16 项中 15 项门禁且 GSO `*/meshes/model.obj` 重名覆盖 (`PR-87`) |
| **`PR-69`** | ✅ 已合入 (#64) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `cli/_bench.py:119` 访问不存在的 `failures`（应为 `violations`）抛 `AttributeError` 误返 `10` 而非 `40`；`cli/_fgpp.py:158` 缺 OpenEXR 返回 `exr_validated: True`；trailing `--json` 报 exit `2` (`PR-74`, `PR-75`) |
| **`PR-70`** | ✅ 已合入 (#65) | ✅ | ❌ | ❌ | **⚠️ 部分合入** | `reanchor_physical_frame` 零生产调用；`wan.py:423` 将 `depth_norm_f64` 截回 `float32`；`scheduler.py` 48 倍数帧覆写消除真实切镜；`validate_output_non_degenerate` 展平方差放行纯红帧且未接 `Cloud/Mock` (`PR-76`, `PR-83`) |
| **`PR-71`** | ✅ 已合入 (#67) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `colab_runner.py --exp all` 因缺 97 帧包回退 17 帧后 E3 在 `frames[64]` 必崩 `IndexError`，且忽略 `--mode real-gpu` 全程跑 Mock (`PR-88`) |
| **`PR-72`** | ✅ 已合入 (#68) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `ubuntu24` 实测 `pytest -m 'not slow'` 需 **`64.89s`**（>2x 超标 `<= 30s`）；`mypy --strict python/synapse` 79 错；`mypy tests/ scripts/ tools/` `EXIT=2`（过滤后仍 63 错） (`PR-90`) |
| **`PR-73`** | ✅ 已合入 (#69) | ✅ | ✅ | ⚠️ | **⚠️ 部分合入** | `README.md` 与技能文档仍停留在旧版描述、存在断链与不存在测试路径；`.github/workflows/` 仍为空 (`PR-91`) |

---

## 3. 全量分级缺陷台账 (`H6-P0` / `H6-P1` / `H6-P2` / `H6-P3`)

### 3.1 P0 阻断级缺陷（数据损坏 · 崩溃死锁 · 静默伪执行）

#### `[H6-P0-01]` `Cryptomatte.ID` 在 OpenEXR 头中误写为 `FLOAT` 导致 `UINT32` 实体哈希位损坏，且 `cli/_fgpp` 放行并缺依赖假绿
- **精确位置**:
  - [python/synapse/fgpp/io.py:134-142](../../../python/synapse/fgpp/io.py#L134-L142)
  - [python/synapse/cli/_fgpp.py:25-28, 55-60, 158](../../../python/synapse/cli/_fgpp.py#L25-L60)
  - 上游 SSOT: `aether/fgpp/include/aether/fgpp/fgpp_types.h:85` (`PixelType::Uint32`)
- **代码实证与根因**:
  - `export_fgpp()` 将 `uint32` 数组通过 `.view(np.float32)` 位重解释并以 `Imath.PixelType(Imath.PixelType.FLOAT)` 写入 OpenEXR 头；当 MurmurHash3 产生的实体哈希高 9 位落在 IEEE-754 signaling NaN 区间（概率约 `1/128`）时，底层 OpenEXR/NumPy 位拷贝或 FPU 读写可能将静默触发 NaN 规范化篡改 bit22，直接违反 `AGENTS.md §1` 红线 #2（`Cryptomatte.ID` `UINT32` 严禁插值与位失真）。
  - `cli/_fgpp.py:55` 允许 `PixelType.UINT32` 通道接受 `FLOAT`，掩盖了写出端错误；且在无 `OpenEXR` 时仅校验 `metadata.json` 并返回 `"exr_validated": True`（`line 158`）。
- **修复建议（`M21 PR-74`）**:
  - 在 `export_fgpp()` 中按 `PixelType.UINT32` 使用 `Imath.PixelType(Imath.PixelType.UINT)` 原生写出 `uint32.tobytes()`（并保持对遗留 `FLOAT` EXR 读取时的兼容解码与告警）；收紧 `_fgpp.py` 逐通道 `PixelType` 严苛比对，缺失 `OpenEXR` 时在校验命令抛出 `BackendUnavailableError (30)`，禁止返回 `"exr_validated": True`。

#### `[H6-P0-02]` `MultiLoRAManager` 注入伪随机噪声、`Cryptomatte` 掩膜丢弃及 `FoundaryHttpTransport` 丢失全部 3D 条件
- **精确位置**:
  - [python/synapse/adapters/lora/manager.py:212-240](../../../python/synapse/adapters/lora/manager.py#L212-L240)
  - [python/synapse/adapters/lora/patcher.py:22-76](../../../python/synapse/adapters/lora/patcher.py#L22-L76)
  - [python/synapse/adapters/wan.py:130-141, 471, 579-582](../../../python/synapse/adapters/wan.py#L130-L141)
- **代码实证与根因**:
  - `MultiLoRAManager._compute_delta()`（`manager.py:212-240`）完全不读取 `manifest.adapter_path` 或校验 `adapter_sha256`，直接依据 `hash(manifest.name)` 用 `torch.randn(...) * 0.01` 生成伪随机权重矩阵；
  - `wan.py:579-580` 调用 `build_spatial_lora_masks(...)` 后立即丢弃返回值，`LoRAPatchHook`（且 `patcher.py:52` 硬编码 `torch.float32`，遇到 `bf16/fp16` 权重直接报错）与 `orthogonalize_conflicting_loras` 零生产调用，`conditions.cryptomatte_id` 在链路中始终为 `None`；
  - `FoundaryHttpTransport.submit()`（`wan.py:130-141`）构造的 HTTP JSON 载荷遗漏 `payload.control_tensor`、`keyframe_images` 与 `payload.extra`，使远端推理完全丧失 3D 几何/深度/法线控制，并重新生成随机 `traceparent` 切断 W3C 链路追踪。
- **修复建议（`M21 PR-76`）**:
  - 实现 `.safetensors` 真实加载与 SHA-256 完整性校验，接通正交化与 `dtype` 自适应 `LoRAPatchHook`，把 `Cryptomatte.ID` 透传并绑定空间掩膜到 `WanAdapter` 推理前向；完善 `FoundaryHttpTransport` 完整载荷序列化（二进制/张量引用的显式编码）与上游 `TraceContext` 透传。

#### `[H6-P0-03]` `JobBundleWorker` 零像素空跑、`bench_runner.evaluate_bundle` 崩溃与 CLI `bench run` 误返退出码 `10`
- **精确位置**:
  - [python/synapse/pipeline/bundle_worker.py:372-375, 856-940](../../../python/synapse/pipeline/bundle_worker.py#L372-L375)
  - [python/synapse/pipeline/bench_runner.py:259, 360](../../../python/synapse/pipeline/bench_runner.py#L259-L360)
  - [python/synapse/cli/_bench.py:119](../../../python/synapse/cli/_bench.py#L119) · [python/synapse/cli/_app.py:138](../../../python/synapse/cli/_app.py#L138)
- **代码实证与根因**:
  - `bundle_worker.py:372` 在 `neural_video_generate` 阶段调用 `FGPPSequence.load_metadata()` 而非 `import_fgpp()`，向生成器传入全零像素帧；`run_worker_cli`（`856` 行起）对损坏 JSON 或 Schema 不匹配未包裹结构化错误处理，直接未捕获崩溃返回 `exit 1`；
  - `bench_runner.py:360` 在 `evaluate_bundle()` 中写出 `np.ndarray or np.ndarray` 布尔求值表达式，遇到数组输入必抛 `ValueError: The truth value of an array with more than one element is ambiguous`，且 `run_scenario()` 未传全量 `required_keys` 导致缺项跳过；
  - `cli/_bench.py:119` 访问不存在的 `r.gate_verdict.failures`（`GateVerdict` 字段实为 `violations`），门禁未通过时抛出 `AttributeError` 被外层兜底为退出码 `10` 而非 `40`；且子命令后置 `--json` 参数触发 argparse exit `2` 无 JSON 输出。
- **修复建议（`M21 PR-75`, `M23 PR-87`）**:
  - 在 `PR-75` 中修复 `bundle_worker` 真读 EXR 像素、`run_worker_cli` 错误信封、`_app.py` 共享父解析器 `--json` 与 `_bench.py:119` `violations` -> exit `40`；在 `PR-87` 中修复 `evaluate_bundle` 数组判定与 `required_keys` fail-closed。

#### `[H6-P0-04]` `FeatureStoreBuilder` 100% 误拒真实样本，且 `training/losses.py` 断梯度未接入训练器
- **精确位置**:
  - [python/synapse/training/feature_store.py:299-381](../../../python/synapse/training/feature_store.py#L299-L381)
  - [python/synapse/training/losses.py:34-210](../../../python/synapse/training/losses.py#L34-L210)
  - [python/synapse/training/trainer.py:149-503](../../../python/synapse/training/trainer.py#L149-L503)
  - [python/synapse/training/packager.py:140-300](../../../python/synapse/training/packager.py#L140-L300)
- **代码实证与根因**:
  - `FeatureStoreBuilder._validate_ingestion_quality()`（`feature_store.py:322-380`）把 `fgpp_cond` 按 `[T, H, W, 16]` 切片并读取像素列 `flat[0, 7:16]` 校验 `|det(R) - 1| < 1e-5`——实际控制条件通道与相机矩阵维度完全不符，导致任何真实 FGPP 序列 100% 被门禁拒绝，写出 0 样本 `.tar` 分片；
  - `training/losses.py` 使用 `np.asarray` 计算指标并在物理上截断 PyTorch `autograd` 梯度图，且从未被 `StatelessLoRATrainer` 导入调用；
  - `training/packager.py` 打包阶段未执行 `.safetensors` 头部与 `LoRAManifest` 靶向层白名单强校验。
- **修复建议（`M21 PR-77`）**:
  - 统一 `fgpp_cond`（`[7,T,H,W]` 或显式 `[T,H,W,C]` 描述）与独立 `camera_matrices` 字段校验；将 `losses.py` 全量改造为纯 PyTorch 张量算子并真实接入 `StatelessLoRATrainer` 反向传播与梯度单测；在 `packager.py` 落盘前强校验 `.safetensors` 头部与白名单。

#### `[H6-P0-05]` `SoftwareRasterizer` 零运动矢量与负 `w_clip` 倒投影 + `composite` 全图模糊与假逆色调映射
- **精确位置**:
  - [python/synapse/render_core/rasterizer.py:181-183, 215, 286-316](../../../python/synapse/render_core/rasterizer.py#L181-L316)
  - [python/synapse/composite/blend.py:198-290, 408-455, 581-636](../../../python/synapse/composite/blend.py#L198-L636)
  - [python/synapse/composite/detector.py:293-331](../../../python/synapse/composite/detector.py#L293-L331)
  - [python/synapse/style/aces_color.py:119-129](../../../python/synapse/style/aces_color.py#L119-L129)
- **代码实证与根因**:
  - `SoftwareRasterizer` 从未向 `mv_buf` 写入像素（`line 303` `_pts_world` 标注 `# noqa: F841`，导出运动矢量全 0），跨越近裁剪面三角形未裁剪即除以负 `w_clip` 造成屏幕倒投影与非透视线性深度插值；
  - `composite_optical_increments()`（`blend.py:581`）用分段常数 Cryptomatte 作为 `_guided_filter_box` 引导对**整幅视频底片 RGB** 进行 `9×9` 双盒滤波（造成背景全局糊化），且调用的 `_fallback_soft_knee_inverse_tonemap`（`aces_color.py:123-129`，`threshold=4.0` 叠加 `tanh` 压缩）实为前向非线性压缩器，在 `[0,1]` 上恒等无逆色调映射展开效果；
  - `border_ring_lab_match` 在掩膜边缘内 16px 产生 1px 阶跃接缝，`_forward_warp_bilinear`（`408` 行）与 `_detect_ground_violation`（`detector.py:293`）分别存在光流符号相反及反投影 `+Y/+Z` 坐标轴颠倒。
- **修复建议（`M21 PR-78`）**:
  - 为 `SoftwareRasterizer` 实现近裁剪面 Sutherland-Hodgman 裁剪、透视校正 $1/w$ 插值及前后帧 MVP 真实屏幕空间运动矢量输出；修正 `composite_optical_increments` 仅滤波/衰减焦散高光增量、实现真正单调可逆 `soft_knee_inverse_tonemap` 并统一光流符号与世界系反投影。

#### `[H6-P0-06]` `recon/delight.py` 恒等相除空操作与多视角逆向管线断裂
- **精确位置**:
  - [python/synapse/recon/delight.py:148, 160, 185-216, 288-359](../../../python/synapse/recon/delight.py#L148-L359)
  - [python/synapse/recon/tracking.py:16-133](../../../python/synapse/recon/tracking.py#L16-L133)
  - [python/synapse/recon/multiview.py:115-117, 429, 522](../../../python/synapse/recon/multiview.py#L115-L522)
  - [python/synapse/pipeline/stages/recon.py:64-65](../../../python/synapse/pipeline/stages/recon.py#L64-L65)
  - [python/synapse/inspect/geometry_gate.py:125-180](../../../python/synapse/inspect/geometry_gate.py#L125-L180) · [python/synapse/recon/geometry.py:292-394](../../../python/synapse/recon/geometry.py#L292-L394)
- **代码实证与根因**:
  - `decompose_intrinsic()` 与 `delight_multiview_albedo()`（`delight.py:185, 288`）都将 `rad` 同时作为 `albedo` 与 `radiance` 传入 `delight(rad, norm, rad)`（本质计算 `rad / rad == 1.0`，高光与阴影 100% 原封不动），对 2D 视角图片不做相机反投 UV 直接平均，还对 `max > 1.5` 的线性 HDR 浮点输入执行 `/ 255.0` 压黑，并用 `A * (I / A) == I` 计算恒零 $\Delta E$ 残差；
  - `solve_point_tracks()` 使用全局 R/G 强度均值差代替真实点追踪；`tracking.py` 输出 `c2w` 而 `multiview.py` 当作 `w2c`；空体素 Marching Cubes 返回 `zeros((3,3))` 退化面；`stages/recon.py:64` 仅写盘 `vertices` 丢失 `faces`；`geometry_gate.py:125-180` 访问 `_c_geometry` 返回字典的 `.is_manifold` 属性触发 `AttributeError` 后静默回退 Python；`quad_remesh` 对奇数剩余三角形生成 `[a,b,c,c]` 非流形退化边。
- **修复建议（`M22 PR-79`, `M22 PR-80`）**:
  - `PR-79` 用法线球谐辐照度拟合 + 多视角 UV 投影融合重构 `recon/delight.py`；`PR-80` 实现块匹配/金字塔光流追踪、统一 `w2c`/`c2w` 约定、空体素 fail-closed、完整网格序列化、修复 `_c_geometry` 字典解包与合法 `quad_remesh`。

#### `[H6-P0-07]` `refine/` 默认原图直通空操作与 `ScalpelStitcher` 纯黑光晕 / HDR 截断
- **精确位置**:
  - [python/synapse/refine/hand.py:50, 141-195](../../../python/synapse/refine/hand.py#L50-L195)
  - [python/synapse/refine/face.py:190, 213-254](../../../python/synapse/refine/face.py#L190-L254)
  - [python/synapse/refine/stitcher.py:143-178, 287, 328, 402](../../../python/synapse/refine/stitcher.py#L143-L287)
  - [python/synapse/refine/fusion.py:183, 314](../../../python/synapse/refine/fusion.py#L183-L314)
  - [python/synapse/refine/pipeline.py:194, 260, 301](../../../python/synapse/refine/pipeline.py#L194-L301)
- **代码实证与根因**:
  - 当 `priors=None` 或 `gallery=None` 时，`HandRefiner.refine()` 与 `FaceRefiner.refine()` 不做像素修改即返回“成功”；
  - `ScalpelStitcher` 在 `stitcher.py:143` 将前景裁切粘贴进全零画布 `fg_full`，随后通过 `generate_surgical_mask` 向外膨胀 `8–24px`，使全零（纯黑）外圈像素参与 `border_ring_lab_match` 与 `laplacian_pyramid_blend`，在精修 ROI 四周引入永久性纯黑光晕，且 `_acescg_to_srgb_f32` 在拼合前后把全幅线性 ACEScg HDR 强行截断至 `[0,1]`；
  - `fusion.py` `LATENT_TIMESTEP_SWITCH` 高低频权重之和仅 `0.5` 导致输出整体亮度减半；`pipeline.py:301` 弃用算好的 `smoothed_bbox` 仍写回原始 `roi.bbox_xyxy`。
- **修复建议（`M22 PR-81`）**:
  - 缺失先验时严禁静默原图直通；在 `fg_mask_bool` 内缩而非零填充区外扩展融合边界、保留非 ROI 区域线性 HDR 数值、归一化 `LATENT_TIMESTEP_SWITCH` 权重并按 `track_id` 独立消费 `smoothed_bbox`。

#### `[H6-P0-08]` 9 维 RGB 均值冒充 FVD 与 3D 关键指标门禁方向反转
- **精确位置**:
  - [python/synapse/metrics/temporal.py:222, 237-290](../../../python/synapse/metrics/temporal.py#L222-L290)
  - [python/synapse/metrics/profiles.py:31-59, 230](../../../python/synapse/metrics/profiles.py#L31-L59)
- **代码实证与根因**:
  - `compute_fvd_from_videos()`（`temporal.py:237-290`）在函数内导入 `torch` 与 `torchvision`（标记 `# noqa: F401` 并丢弃），转而提取每帧 RGB 均值/标准差/帧间差共 **9 维纯颜色统计量**充当 I3D 时空特征，对形状变化和时序闪屏完全盲区；同时 `fvd_feature_distance()`（`222` 行）未给协方差加 $\epsilon I$ 正则，全套测试运行稳定触发 `LinAlgWarning: Matrix is singular`；
  - `profiles.py:31-59` 的 `_METRIC_DIRECTIONS` 遗漏 `f_score_5mm` 与 `watertight_pct`；第 `230` 行 `.get(name, _LOWER)` 默认回落至 `lower_is_better`——**结果导致 `watertight_pct = 100.0%` 与阈值比较为 `100 > 95 -> FAIL`，而 `0%` 破洞模型 `0 <= 95 -> PASS`**！
- **修复建议（`M22 PR-83`）**:
  - 补全 `_METRIC_DIRECTIONS` 对 `f_score_5mm` / `watertight_pct` 的 `"higher_is_better"` 注册；加入 $\epsilon I$ 协方差稳定项并接入真实预训练 3D 视频特征骨干（缺权重在严格模式抛 `ExitCode 30`）。

#### `[H6-P0-09]` `inspector_app.cc` 泄露 POSIX 头文件 + Windows 死锁 + OBJ 越界读 `SIGSEGV` + `-DNDEBUG` 测试断言全蒸发
- **精确位置**:
  - [tools/data-inspector/native/inspector_app.cc:8-11, 42-45, 113-174, 220-237, 474-578](../../../tools/data-inspector/native/inspector_app.cc#L8-L45)
  - [tools/data-inspector/native/test_inspector.cc:7, 25-185](../../../tools/data-inspector/native/test_inspector.cc#L7-L185)
- **代码实证与根因**:
  - 直接 `#include <poll.h>` / `<unistd.h>`，直接违反生态 Always-On Rule #9（OS 差异必须收敛至 `@aether//pal`）；在 Windows 上 `poll()` 被宏屏蔽后 `std::getline(std::cin, ...)` 永久阻塞，导致析构阶段 `m_stdinThread.join()` 死锁；
  - OBJ 解析虽跳过越界索引的面法线计算，但未把非法面从 `m_mesh.faces` 删除，使 `renderFrame()`（`line 479`）发生堆越界读 `SIGSEGV`；渲染器缺少 Z-Buffer（正背面直接覆盖）、不支持负相对索引及 Quad 第 4 顶点、完全不解析 `.glb` 与 `.exr`（加载真实 Bundle 静默画 8 顶点方块）；
  - `test_inspector.cc` 全部使用 `<cassert>` `assert(...)`，在 DervGen 默认 `opt` (`-DNDEBUG`) 下 100% 断言与部分被测表达式直接被剥离。
- **修复建议（`M23 PR-85`）**:
  - 改用 `@aether//pal` `aether::pal::pollStdinBytes`、过滤非法面与扇形三角化、加入浮点 Z-Buffer 与 `.glb`/`.exr` 真实解码、将 `test_inspector.cc` 迁至 GoogleTest。

#### `[H6-P0-10]` Electron SHM 读取在 macOS/Windows 100% 失效 + `InspectionBundle` 四方契约分裂
- **精确位置**:
  - [tools/data-inspector/electron/shm-reader.js:36-49, 114-156](../../../tools/data-inspector/electron/shm-reader.js#L36-L156) · [electron/main.js:134](../../../tools/data-inspector/electron/main.js#L134)
  - [schemas/bundle.v1.0.schema.json:14](../../../schemas/bundle.v1.0.schema.json#L14) · [python/synapse/bench/bundle_exporter.py:36, 95-98](../../../python/synapse/bench/bundle_exporter.py#L36-L98)
  - [tools/data-inspector/renderer/src/app.js:67-69, 113-122, 193-196](../../../tools/data-inspector/renderer/src/app.js#L67-L196)
- **代码实证与根因**:
  - `shm-reader.js` 在非 Linux 平台查找 `os.tmpdir()/shmName` 磁盘文件，而 `@aether//pal:shm` 调用内核 `shm_open` / `CreateFileMappingW` 从不写入 `/tmp` 或 `%TEMP%`，导致 macOS/Windows 下 `fs.existsSync` 恒返回 `false`（且 macOS `PSHMNAMLEN=31`）；Linux 仍每帧拷贝整块 RGBA 经 Electron IPC 发送；
  - `bundle.v1.0.schema.json` 锁死 `"const": "synapse.bundle.v1.0"`，`bundle_exporter.py` 锁死 `"inspection.v1.0"`，而 `test/fixtures/*/bundle.json` 使用裸字符串 `sources` 令两者都失败；UI `app.js` 直接字符串化 `SourceEntry` 显示 `[object Object]`，点击视图 Tab 向 `validateNativeCommand` 发送 `'source'|'animation'` 必然触发未捕获 Promise 异常。
- **修复建议（`M21 PR-74`, `M23 PR-86`）**:
  - 在 `PR-74` 统一 `schema_version = "synapse.bundle.v1.0"`；在 `PR-86` 通过 `@aether//pal:shm` N-API 绑定或跨平台文件后备 mmap 实现全平台零拷贝读帧，并解耦 UI 面板 Tab 与 `set_debug_view` AOV 下拉。

---

### 3.2 P1 重要架构、守卫、安全与测试基建缺陷 (`H6-P1-01` ~ `H6-P1-10`)

| 编号 | 类别 | 精确位置 | 实证问题与影响 | 修复 PR |
| :--- | :---: | :--- | :--- | :---: |
| **`H6-P1-01`** | WIRE | `tools/wire_guard.py:220-248, 275-387, 411-495` | `_RE_PIXEL_TYPE` 正则匹配 `fgpp_types.h` 0 行（`pixel_type_values == {}` 且 `Uint32` vs `UINT32` 大小写失配）；未校验 C++ 逐通道 `PixelType`、顺序与 `allow_spatial_filtering`；`aether/style/` 不存在而静默报告 OK；未校验 Python 模型实例化默认 `schema_version` | `PR-74` |
| **`H6-P1-02`** | ARCH | `tools/import_dag_guard.py:63-81, 116-129, 223`<br>`python/synapse/__init__.py:33-47` | `synapse/__init__.py` 急切导入全部 13 个子包（含 Tier 4 `cli`/`pipeline`）却被守卫显式跳过；仅查环路与 `->bench`，放行 `inspect->bench` 且存在 `bridge->recon`、`inspect->rsi`、`recon->refine` 跨 Tier 逆向边；漏检 `node.level > 0` 相对导入与动态 `import_module` | `PR-89` |
| **`H6-P1-03`** | ADAPTER | `adapters/wan.py:423, 476-523`<br>`adapters/scheduler.py:448-468`<br>`metrics/image_quality.py:273-414`<br>`style/extractor.py:135-175, 234` | `reanchor_physical_frame` 零生产调用；`depth_norm_f64` 在 `wan.py:423` 被 cast 回 `float32`；`scheduler.py` 在 48 倍数帧强覆写 `is_physical_anchor=True` 抹除真实切镜 `has_discontinuity`；`validate_output_non_degenerate` 展平三通道放行纯红帧（`Var=0.222`）且未接 `Cloud`/`Mock`；`lpips()` 稀释掩膜区误差；`style/extractor` 背向法线污染 SH 且对浮点误除 `255.0` | `PR-76`<br>`PR-78`<br>`PR-83` |
| **`H6-P1-04`** | RECON | `recon/loop/forward_guide.py:152-158`<br>`mitsuba_optimizer.py` · `texture_extractor.py` | `_spatial_dims` 维度 `(T,H)` 互换错读时序帧数为高度；预卷帧拼接裁切长度边界脆弱；`texture_extractor.py` 缺切线空间 UV 法线烘焙且 TV 梯度非对称 | `PR-82` |
| **`H6-P1-05`** | BENCH/CLI | `inspect/model_evaluator.py:43, 66, 430`<br>`cli/{_recon.py, _refine.py:89, _inspect.py:121, _rsi.py:179}` | `model_evaluator` 丢弃 `profiles.py` 16 项门禁中的 15 项，且 GSO `*/meshes/model.obj` 全覆写进同一个 `"model"` 键；CLI 子命令仍残余伪造法线/ROI、`MockJevTransport` 及 `_rsi` `scenarios=[]` 与三轴同分 | `PR-87` |
| **`H6-P1-06`** | NATIVE | `scripts/build_native.py:90-205`<br>`csrc/geometry/.bazelrc` · `native/.bazelrc`<br>`inspector_shm_server.h/.cc` | `build_native.py --backend=direct` 缺 `aether/core/include` 与 `aether/spatial/include` 真实编译必败（被 `--dry-run` 测试掩盖）且私跑 `pip install`；两处 `.bazelrc` 缺 `build --config=opt`、禁用平台自动配置、缺私服 BCR；Native SHM `CreateOrOpen` 零化在役共享内存、非原子竞态、JSON 注入与 `stdout` 混入日志 | `PR-84`<br>`PR-85` |
| **`H6-P1-07`** | SEC/HYG | `viewer/server.py:281-289`<br>`tools/data-inspector/test/test-bundle-loading.js:35-251` | `viewer/server.py` 将未转义 `json.dumps()` 插入 `<script>` 存在 `</script>` 存储型 XSS；`test-bundle-loading.js` 每次运行原地覆写 Git 跟踪 `fixtures/*/bundle.json` 且失败时泄漏临时目录 | `PR-75`<br>`PR-86` |
| **`H6-P1-08`** | COLAB | `scripts/colab_runner.py:238-266, 322-358`<br>`workenv/google-colab/setup_colab.sh:99`<br>`scripts/prepare_benchmark_assets.py:317` | `colab_runner.py --exp all` 因 97f 未打入束而在 E3 `frames[64]` 必崩 `IndexError`，忽略 `--mode real-gpu` 全程跑 Mock 且缺 `SIGTERM` `state.json` 续跑；`setup_colab.sh` 自造 `data/video/base_models` 违反 SSOT V2.0；`stage_from_remote_huggingface` 返回 `[]` 空桩 | `PR-88` |
| **`H6-P1-09`** | CI/PERF | 缺失 `.github/workflows/`<br>`pyproject.toml` · `tests/` | 仓库零 GitHub Actions 自动门禁；`mypy --strict python/synapse` 79 错；`mypy tests/ scripts/ tools/` `EXIT=2` 且过滤后仍 63 错；fast pytest `64.89s` 超 2x SLA；slow pytest `518.38s` 3 次重复软光栅 | `PR-90`<br>`PR-91` |
| **`H6-P1-10`** | TEST/RFC | `tests/{test_recon_loop_optimizer.py:366, test_lora_cli_gate.py:576, test_roundtrip_reconstruction.py:222, test_inspect_align.py:300}`<br>`docs/RFC-0061..0063` | 4 处典型恒真/失明/伪测试（纯 NumPy 自减恒 0、只读 18 行 `cli/__init__.py` 垫片、用 `gt+2mm` 噪声充当 `ReconPipeline`、丢弃 `best_pose`）；`SYN-RFC-0061..0063` canonical 指针仍与 `all-overview/rfcs/0061..0063` 撞号 | `PR-82`<br>`PR-90`<br>`PR-91` |

---

### 3.3 P2 与 P3 一般工程债务 (`H6-P2-01` ~ `H6-P2-05`, `H6-P3-01` ~ `H6-P3-02`)

- **`[H6-P2-01]` Electron 壳安全、UI 布局与死代码** (`tools/data-inspector/electron/main.js:18-59`, `renderer/index.html:55`, `renderer/src/views/source-model.js`):
  - 缺少 `will-navigate` 与 `setWindowOpenHandler` 外跳拦截及开发态 CDP 调试端口；macOS `.toolbar` (`padding: 0 12px`) 与 `(16, 16)` 交通灯碰撞；`#viewport-canvas` 缺 `ResizeObserver` / DPR 自适应；`views/source-model.js`（162 行）从未被引入。
- **`[H6-P2-02]` 可选依赖静默跳过、固定测试端口与重复用例** (`tests/test_geometry_oracle_parity.py`, `test_dataset_and_viewer.py:190,241`, `test_cli.py:528-575`):
  - 17 个 `OpenEXR`/`torch`/`_c_geometry` 核心对标用例可在依赖缺失时静默 `skip`，缺 `SYNAPSE_CI_STRICT_DEPS=1` 强制 fail 门禁；`ViewerServer` 测试写死 `9876/9877` 端口且存在长阻塞 `serve_forever()`；`test_cli.py` 存在重复用例及 5 个零 `assert` 冒烟用例。
- **`[H6-P2-03]` RSI 闭环课程调度与隔离目录接线未完全穿透** (`rsi/curriculum.py`, `rsi/promotion_gate.py`):
  - `CurriculumManager` 与失败样本 Quarantine 归档在全链路评估 CLI 中接线覆盖不完整（并受当前 `inspect -> rsi` 逆向耦合牵制）。
- **`[H6-P2-04]` 文档断链、过期模块描述与治理字段缺漏** (`AGENTS.md`, `README.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/plans/*`, `.agents/skills/*`):
  - `PLAN-M13-M16:257` 引用已删除的 `csrc/geometry/src/mesh_checks.cc`；`PLAN-M17-M20` 与 `IMPLEMENTATION_PLAN §4.5` 引用不存在的 `tests/test_temporal.py` / `tests/test_roundtrip_metrics.py` / `tests/test_fgpp.py` / `tests/test_scheduler.py` / `tools/data-inspector/tests/`（实为 `test/`）；`README.md` / `AGENTS.md` / Skills 仍称 `csrc/geometry` 自包含（未更新 `@aether//geo` 委托）、把尚未建立的 `@aether//style` 写成已存在模块、遗漏 `synapse.render_core` 与 `RFC-0063`。
- **`[H6-P2-05]` 表面采样 Chamfer 缺失与软光栅/指标协议标注** (`metrics/roundtrip.py`, `inspect/model_evaluator.py`):
  - 顶点直接最近邻对稀疏高模分布敏感，需补齐面积加权均匀表面采样并在评测报告显式声明协议边界。
- **`[H6-P3-01]` C++20 命名风格微调**: `inspector_shm_server.h:38-48` POD 结构体混用 `snake_case` 与 `m_reserved`，`csrc/geometry/src/bindings.cc:53` 使用 `verts_per_face`。
- **`[H6-P3-02]` 仓库卫生与规则扩充**: `.gitignore` 第 `57`/`66` 行重复 `data/datasets/` 且漏 `**/MODULE.bazel.lock`、`tools/data-inspector/node_modules/`；`schemas/inspection.v1.0.schema.json` 文件名与 `"2.0.0"` 版本常量不匹配；`pyproject.toml` Ruff 可扩至 `["E4", "E7", "E9", "F", "I", "UP", "B", "SIM", "RUF"]`。

---

## 4. Horizon 6 (`M21–M24`, `PR-74 ~ PR-91`) 整改映射矩阵

| 审计缺陷编号 | 严重度 | 归属里程碑 | 目标 PR | 关键关闭判据（防静默失败三问） |
| :--- | :---: | :---: | :---: | :--- |
| `H6-P0-01`, `H6-P1-01` | P0/P1 | **M21** | [`PR-74`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-74) | EXR 原生 `UINT32` 写出与 bit 零失真；`wire_guard` 逐通道 C++ 解析与 `@aether//style` 显式状态；Bundle Schema 统一 `"synapse.bundle.v1.0"` |
| `H6-P0-03`, `H6-P1-07` | P0/P1 | **M21** | [`PR-75`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-75) | `bundle_worker` 真读 EXR 像素；`run_worker_cli` 结构化退错；后置 `--json` 生效；`_bench.py` 门禁失败稳定 exit `40`；`viewer` HTML XSS 转义 |
| `H6-P0-02`, `H6-P1-03` | P0/P1 | **M21** | [`PR-76`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-76) | `MultiLoRAManager` 加载真实 `.safetensors` + SHA256 与正交化；`Cryptomatte` 掩膜生效；`FoundaryHttp` 传 3D 张量与 Trace；48f 重锚真接入 + `float64` 精度 |
| `H6-P0-04` | P0 | **M21** | [`PR-77`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-77) | `FeatureStoreBuilder` 修复 `[7,T,H,W]` 与相机矩阵校验消除 100% 误拒；纯 PyTorch 可微 `losses.py` 接通训练器；`packager` 强制校验 safetensors/白名单 |
| `H6-P0-05`, `H6-P1-03` | P0/P1 | **M21** | [`PR-78`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-78) | `SoftwareRasterizer` 近裁剪面 + $1/w$ 深度 + 真实 `motion_vectors`；`composite` 不糊底片 RGB、真逆色调映射、统一光流/地面反投影；SH 去背向与去误除 255 |
| `H6-P0-06` | P0 | **M22** | [`PR-79`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-79) | 废除 `delight(rad,norm,rad)` 自除空操作，实现多视角 UV + SH 辐照度去光照与 HDR 保真 |
| `H6-P0-06` | P0 | **M22** | [`PR-80`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-80) | 光流块追踪、`w2c/c2w` 统一、空体素拒退化、全网格序列化、`_c_geometry` 字典修复与合法 `quad_remesh` |
| `H6-P0-07` | P0 | **M22** | [`PR-81`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-81) | `HandRefiner`/`FaceRefiner` 无先验 fail-closed；`ScalpelStitcher` 消除纯黑光晕与 HDR 截断；权重和归一化与 Kalman `track_id` 独立跟踪 |
| `H6-P1-04`, `H6-P1-10` | P1 | **M22** | [`PR-82`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-82) | `_spatial_dims` 与预卷帧裁切修复；UV 法线烘焙与对称 TV 散度；测试直连 `synapse.recon.loop` 算子 |
| `H6-P0-08`, `H6-P1-03` | P0/P1 | **M22** | [`PR-83`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-83) | `_METRIC_DIRECTIONS` 修正 `f_score_5mm`/`watertight_pct`；$\epsilon I$ 消警与真 3D 特征 FVD；掩膜局部 LPIPS；逐通道时序-空间 `Var(X)` 门禁覆盖全适配器 |
| `H6-P1-06` | P1 | **M23** | [`PR-84`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-84) | `build_native.py` 补全 `@aether//core,spatial` 与 Colab 自包含直编；`.bazelrc` 对齐 `opt` 与私服 BCR；非 `--dry-run` 真实编译单测 |
| `H6-P0-09`, `H6-P1-06`, `H6-P3-01` | P0/P1/P3 | **M23** | [`PR-85`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-85) | `@aether//pal` 替换 `<poll.h>/<unistd.h>`；OBJ 安全解析 + Z-Buffer + `.glb/.exr`；原子 SHM + `nlohmann::json` + `cerr`；GoogleTest 替代 `assert` |
| `H6-P0-10`, `H6-P1-07`, `H6-P2-01` | P0/P1/P2 | **M23** | [`PR-86`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-86) | macOS/Windows/Linux 跨平台零拷贝 SHM；修复 `app.js` `SourceEntry` 与视图/AOV 解耦；Electron 守卫、红绿灯边距、只读 fixture |
| `H6-P0-03`, `H6-P1-05`, `H6-P2-05` | P0/P1/P2 | **M23** | [`PR-87`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-87) | `bench_runner` 数组修复与全 `required_keys`；`model_evaluator` 全指标门禁 + 相对键防撞 + 表面采样；清理 CLI 残余伪输入与 `MockJevTransport` |
| `H6-P1-08` | P1 | **M23** | [`PR-88`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-88) | 97f 入束修复 `colab_runner` E3 `IndexError`、`--mode real-gpu` 与 `SIGTERM` 续跑；SSOT V2.0 对齐；真实 HF 下载 + `vital_file` |
| `H6-P1-02`, `H6-P2-03` | P1/P2 | **M24** | [`PR-89`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-89) | `__init__.py` PEP 562 惰性化；`import_dag_guard` 强制 5-Tier `TIER_MAP`、相对/动态导入检测；打通 `CurriculumManager`/Quarantine 接线 |
| `H6-P1-09`, `H6-P1-10`, `H6-P2-02` | P1/P2 | **M24** | [`PR-90`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-90) | 清零 `mypy --strict` 79 错与 `tests/scripts/tools` 63 错；修复 4 处假单测、`port=0` 与 `SYNAPSE_CI_STRICT_DEPS`；fast pytest `<= 25s`、`slow <= 90s` |
| `H6-P1-09`, `H6-P1-10`, `H6-P2-04`, `H6-P3-02` | P1/P2/P3 | **M24** | [`PR-91`](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#pr-91) | 新增 `.github/workflows/ci.yml` 全自动 CI；协同 `all-overview` 消除 `0061–0063` 生态碰撞；同步文档/技能/断链与 `.gitignore`/Ruff |
