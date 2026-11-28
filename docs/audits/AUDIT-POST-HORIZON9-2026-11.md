---
type: Guide
title: "Synapse Horizon-9 + RFC-0071 合流后全库整体审查、改进战略与真值复核报告（2026-11）"
tags: [synapse]
status: active
stale_after: 2027-11-19T00:00:00Z
---
# Synapse Horizon-9 + RFC-0071 合流后全库整体审查、改进战略与真值复核报告（2026-11）

- **文档状态**: `Accepted Holistic Review & Audit Report (v1.0)`
- **审计基线**: `main` @ `5c6e3e6`（`aa84b5b` 将 `v4.0.0` 分支与 `RFC-0071` `@aether` 原生去重主干三方合流后的最新基线）
- **审计日期**: `2026-11-20`
- **操作模型标识**: `[gemini4]`
- **关联演进计划**: [docs/plans/PLAN-M37-M40-POST-RECONCILIATION-TRUE-CLOSURE.md](../plans/PLAN-M37-M40-POST-RECONCILIATION-TRUE-CLOSURE.md)（Horizon 10: `M37–M40` / `PR-153 ~ PR-172`）
- **关联总计划**: [docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)（`v3.8`）
- **前序审计链**: [AUDIT-POST-REFACTOR-2026-11.md](../archive/audits/AUDIT-POST-REFACTOR-2026-11.md) · [AUDIT-RFC0063-AND-CODEBASE-2026-11.md](../archive/audits/AUDIT-RFC0063-AND-CODEBASE-2026-11.md) · [AUDIT-POST-HORIZON5-2026-11.md](../archive/audits/AUDIT-POST-HORIZON5-2026-11.md) · [AUDIT-POST-HORIZON6-2026-11.md](../archive/audits/AUDIT-POST-HORIZON6-2026-11.md) · [AUDIT-POST-HORIZON7-2026-11.md](../archive/audits/AUDIT-POST-HORIZON7-2026-11.md) · [AUDIT-POST-HORIZON8-AND-HOLISTIC-REVIEW-2026-11.md](../archive/audits/AUDIT-POST-HORIZON8-AND-HOLISTIC-REVIEW-2026-11.md)

---

## 1. 执行摘要、全仓定量盘点与 14 项反作弊硬门禁独立复核

### 1.1 代码库定量规模盘点 (`5c6e3e6`)

在 `HEAD = 5c6e3e6` 基线上对全仓五个正交代码与治理切片开展逐行只读审计、跨仓 SSOT（`../aether`、`../render-farm`、`../all-overview`）源码交叉比对，并在云端主力机 `ubuntu24` 及远端 GitHub Actions（`Run #44643198391`）实跑复核全部门禁，当前全仓定量规模如下：

| 目录 / 子系统 | 文件数 / 规模 | 职责分布与实态 |
| :--- | :--- | :--- |
| [python/synapse/](../../python/synapse/__init__.py) | **`142` `.py` 文件 (`46,367` LOC) + `30` 场景 `.json` (`1,217` LOC)** | 20 个顶层领域子包：`fgpp` · `render_core` · `bridge` · `style` · `composite` · `adapters`（含 `lora`） · `training` · `recon`（含 `loop`） · `refine` · `inspect` · `bench` · `metrics` · `rsi` · `pipeline`（含 `stages`） · `cli` · `viewer` · `models` · `generators` · `errors` · `telemetry` |
| `csrc/geometry/` | **已删除 (`0` 文件)** | 在 `RFC-0071`（`b0eaa27`）整体迁往 `@aether//python:aether_geo`，但 CI、Colab、活文档与历史归档断链未同步清理 |
| [tools/](../../tools/wire_guard.py) | **`2` Python 守卫 (`1,360` LOC) + `data-inspector` (`17` 源码文件 / `6,283` LOC)** | `wire_guard.py` / `import_dag_guard.py`；桌面检查器含 Native C++20 SHM 视口、Electron 主进程/预加载、前端与 Bundle/SHM 测试 |
| [scripts/](../../scripts/build_native.py) | **`10` Python 脚本 (`4,097` LOC)** | `build_native.py` · `prepare_datasets.py` · `run_cpu_verification.py` · `colab_runner.py` · `run_roundtrip_verification.py` · `prepare_benchmark_assets.py` · `evaluate_3d_models.py` · `dev_preview_viewer.py` · `check_doc_links.py` |
| `schemas/` & `docs/` & `workenv/` | **`8` JSON Schemas (`981` LOC) + `44` Markdown (`14,262` LOC) + `6` Colab 文件 (`901` LOC)** | 契约 Schema、架构/线格式 SSOT、`SYN-RFC-0060..0063`、计划/审计归档与 Colab 套件 |
| `tests/` | **`62` Python 测试文件 (`31,277` LOC)** | `ubuntu24` 实收集 **`1,510` 项 pytest 用例**（`1,341` fast + `169` slow） |

### 1.2 远端 CI 与 `ubuntu24` 14 项反作弊硬门禁实测真值表 (`5c6e3e6`)

> [!CAUTION]
> **门禁真值核查结论**：`aa84b5b` 将 `v4.0.0` 分支（53 个 Horizon-9 提交）与 `RFC-0071` 主干（删除 `csrc/geometry/`、移除兄弟目录探测、强制 `aether_geo`）三方合并 200 文件后，**远端 GitHub Actions 5/5 Job 全红失败**；`ubuntu24` 独立实测 **14 项硬门禁中仅 1 项基本达标、9 项直接失败或超时、4 项存在规则空转/漏检假绿**。

| 门禁 | 宣称门槛 (`v4.0.0`) | `ubuntu24` / GitHub Actions (`5c6e3e6`) 独立实测真值 | 核查判定 |
| :---: | :--- | :--- | :---: |
| **Gate 1a** | 单进程 `pytest -m 'not slow'` `0 fail/0 skip`，`<= 20s` | `1341 passed, 0 skipped`，耗时 **`71.58s`（`real 75.50s`，超标 `3.58x`）**；`test_recon.py` 两项未标 `slow` 用例独占 `24.78s`；自身包导入触发 `4` 条 `DeprecationWarning` | ❌ **超时 `3.58x`** |
| **Gate 1b** | 并行 `pytest -m 'not slow' -n auto` `<= 10s` | `1341 passed, 64 warnings in 37.65s`（`real 38.89s`，**超标 `3.77x`**） | ❌ **超时 `3.77x`** |
| **Gate 2** | 并行 `pytest -m slow -n auto` `0 fail/0 skip`，`<= 40s` | **`1 failed, 168 passed in 96.23s`（超标 `2.41x`）**：`test_real_bazel_build_and_load` 因 `aether` `downloader.cfg` 路径错误退 `30`；`test_roundtrip_reconstruction.py` 在多进程重复执行 5 次 30~46s `setup` | ❌ **失败 + 超时** |
| **Gate 3** | `ruff check` `0` 错，解禁 `RUF059/E501/W/PT/S*` | 退出 `0`，但 [pyproject.toml](../../pyproject.toml#L142-L150) 将 `E501, W, PT, S108, S301, S310, S602, S603` **同时写入 `select` 与 `ignore`**（且 `RUF059` 仍在 `ignore`），解禁实为 `0` 生效空操作 | ⚠️ **规则空转** |
| **Gate 4** | 全仓 4 目录 `mypy --strict` `>= 207` 文件 `0 errors` | 裸跑 `mypy` 或 CI 命令因 `tools/import_dag_guard.py` 双重模块名直接 **`exit 2`（实检 `0` 文件）**；加 `MYPYPATH=python --explicit-package-bases` 穿透后检出 **`20 errors in 8 files`**（`216` 文件） | ❌ **崩溃 + 20 错** |
| **Gate 5** | `import_dag_guard.py --check` 零逆向边、零绕过 | 静态通过，但 [import_dag_guard.py](../../tools/import_dag_guard.py#L77-L79) 对 `aether_bridge.py` 整文件豁免任意动态导入，且漏检 `import_module(".cli", "synapse")`、`name=` 关键字参数与 `builtins.__import__` | ⚠️ **留 4 绕过** |
| **Gate 6** | `wire_guard.py --check --require-aether` 全量 Schema 双向对齐 | 15ch `34 B/px` 对齐 C++ 通过；但仅校验 `6/8` 个 Schema（漏 `fgpp.v1.0` 与 `style_profile.v2.0.0`），掩盖 [profile_mapper.py](../../python/synapse/style/profile_mapper.py#L26-L46) 嵌套输出与扁平 Schema 100% 失配 | ⚠️ **漏检 2 Schema** |
| **Gate 7** | 原生几何核编译、安装与双轨一致性 | `--check` 命中旧缓存通过；默认执行 [build_native.py](../../scripts/build_native.py#L351-L355) 不拷贝 `.so` 且 [aether_bridge.py](../../python/synapse/bridge/aether_bridge.py#L35-L44) 已删 `../aether` 探测形成构建-加载死锁；未编译 `aether_metrics`；`aether_geo.marching_cubes` 轴转置+面汤 P0 | ❌ **死锁 + 算子错** |
| **Gate 8** | `data-inspector` Native Bazel + Node SHM | `ubuntu24` 实跑 `bazel test //:test_inspector` 因 `downloader.cfg` 相对路径退 **`2`**；源码实查 **15ch EXR、多实体槽与 4-Tab 均未接入 `renderFrame()`**，首次 `resize` 必冻屏 | ❌ **构建败 + 假接线** |
| **Gate 9** | `prepare_datasets.py` + `run_cpu_verification.py` 实跑全绿 | **双双实跑崩溃**：[prepare_datasets.py](../../scripts/prepare_datasets.py#L178) 回退缺 5 个 EXR 通道抛 `ContractValidationError`；[run_cpu_verification.py](../../scripts/run_cpu_verification.py#L344) E4 传绝对路径抛 `LoRAManifest` `ValidationError` | ❌ **双脚本崩溃** |
| **Gate 10** | 反恒等跨仓契约测试 | 用例通过，但 [bundle_worker.py](../../python/synapse/pipeline/bundle_worker.py#L362-L364) 仍把 `OutputSink.format="mp4"` 当路径致 `CWD != workspace_root` 沙箱自锁，SHM 测试仅断言 `pixels.shape` | ⚠️ **覆盖失真** |
| **Gate 11** | 全仓 Markdown 相对路径 `0 broken links` | [check_doc_links.py](../../scripts/check_doc_links.py) 实跑 **`exit 1`（`11~16` 处指向已删 `csrc/geometry/` 的断链）**；新立 SSOT [WIRE_CONTRACTS.md](../WIRE_CONTRACTS.md#L13-L31) 存在 **16 通道 38B/px 幻觉** | ❌ **断链 + SSOT 错** |
| **Gate 12** | `hypothesis` 属性测试与畸形二进制防御 | `test_property_contracts.py` 运行通过（但含 4 处 `mypy` 错且 C++ `loadObjMesh` 缺体积上限） | ⚠️ **部分达标** |
| **Gate 13** | Server-side Zero-Python 红线 #1 与懒加载纪律 | [synapse/__init__.py](../../python/synapse/__init__.py#L86-L89) PEP 562 惰性加载、`viewer/server.py` 纯静态导出合规 | ✅ **基本达标** |
| **Gate 14** | `.github/workflows/ci.yml` 远端全 Job 绿灯 | 远端 `Run #44643198391` **5/5 Jobs 全部失败 (`X`)**：`dervgen/aether` 404、`csrc/geometry` 目录不存在、Python Job 缺 Bazel、断链扫描失败 | ❌ **5/5 全红** |

### 1.3 七维健康度评分卡 (`5c6e3e6` 实测真值)

| 审查维度 | 当前得分 | 核心诊断结论 |
| :--- | :---: | :--- |
| **1. CI/CD、原生构建与静态工程门禁** | **`48 / 100`** | GitHub Actions 5/5 全挂；`mypy` 双模块名崩溃且藏 20 错；`pyproject.toml` Ruff 规则假启用；`build_native.py` 与 `aether_bridge.py` 寻址死锁；`data-inspector` Bazel 因 `downloader.cfg` 失败 |
| **2. 跨仓 Wire 契约、相机系与色彩空间** | **`66 / 100`** | `fgpp/frame.py` 修 `cx` 却反了 `cy = H - cy`；Vulkan `RH_ZO` / 有限 Reversed-Z 误判为 OpenGL；`NaN` 旋转矩阵穿透 `SO(3)` 校验；`aether_bridge` SIMD 色彩在 `>=148x148` 图像上 100% 触发 65536 上限退回慢速 `float64`；`WIRE_CONTRACTS.md` 严重失真 |
| **3. 架构分层、纯 Python 基座与内存效率** | **`72 / 100`** | `camera_math.py` 与 `procedural.py` 删光 NumPy 回退强依赖 `aether_geo`（违反 `AGENTS.md` §3.3）且引入球极点 64 简并面与非水密圆锥；3 套 Mesh 容器并存；`FeatureStoreBuilder(streaming=True)` 实则全量驻留内存 |
| **4. 视频生成适配、因果滑窗与 PEFT 训练** | **`56 / 100`** | `DiffusersWanBackend` 为不可运行空壳（错模型 ID、向 `WanPipeline` 传不存在的 `control_tensor` 必崩、`_vae` 零调用）；`WanAdapter` 忽略 `conditions.fgpp_frames` 与切镜 `needs_reanchor`；`trainer.py` 反向三元表达式必崩、`lora train` 硬编码 `64x64` 玩具 MLP 产出非法权重键 |
| **5. 3D 逆向重建、可微闭环与解剖精修** | **`58 / 100`** | `aether_geo.marching_cubes` 输出非索引面汤 + X/Z 转置 + 法线内翻；xatlas UV 在闭合网格 100% 被丢弃；红线 #3 因 `vertex_count >= 1500` 自动置 `has_gen3d_highpoly=True` 被自旁路；`Mitsuba` 断梯度；`refine` 检测置信度当健康度全旁路、3 种融合核代数退化为简单加权平均、快运动 `w_curr=0.30` 反向加重鬼影 |
| **6. `Synapse-Bench`、`Inspect`、`RSI` 与 `Data Inspector`** | **`62 / 100`** | `bench_runner` `light_dir_error_deg(gt, gt)` 自比较 + 24/30 场景因未算 `chamfer_mm` 必挂 Profile；`model_evaluator` `normalize()` 抹除世界尺度；`rsi/curriculum.py` 把 `0.0`/崩溃算成 `0.85`；`data-inspector` 15ch EXR/多实体/4-Tab 未接入 `renderFrame()` 且缩放冻屏 |
| **7. 测试反恒等质量、耗时 SLA 与安全基线** | **`64 / 100`** | 快测 `71.6s`（超 `20s` SLA `3.58x`）、慢测 `96.2s`（超 `40s` SLA `2.41x`）；存在 `or "ACEScg" in combined` 永真断言与 `MagicMock` 遮蔽；`bundle_worker` 沙箱前任意文件 SHA256 Oracle、`trainer` `weights_only=False` Pickle RCE、`vscode_tunnel.sh` `curl -k`、`HttpJevTransport` SSRF |
| **综合健康度总评** | **`61 / 100`** | **双主线合流后急需 Horizon 10 (`M37–M40`, `PR-153 ~ PR-172`) 完成彻底止血合龙与真接线演进，目标推升至 `>= 95/100` (`Release v4.1.0`)** |

---

## 2. 系统性根因剖析与七大战略级改进建议

### 2.1 三大深层结构性根因

1. **双主线三方强合流缺乏合流后全链路复跑**：`v4.0.0` 分支围绕本地 `csrc/geometry/` 构建 CI、文档与测试，而 `RFC-0071` 分支删除 `csrc/geometry/`、移除 `aether_bridge` 兄弟目录探测并强绑 `aether_geo`。`aa84b5b` 合并 200 文件后仅运行局部检查，未跑通 4 目录 `mypy`、断链扫描、`prepare_datasets.py`、`run_cpu_verification.py` 与远端 Actions，也未核对 `@aether//geo` C++ 算子的数组维度序（`(dim_z, dim_y, dim_x)` vs `(nx, ny, nz)`）与拓扑契约。
2. **“门禁剧场（Gate Theater）”与 `MagicMock` 遮蔽未完工接线**：`pyproject.toml` 同时 `select` 与 `ignore` 同一批 Ruff 规则、`ci.yml` 后缀 `|| true`、`test_profiles_doc_sync.py` 永真 `or` 断言、`test_wan_adapter.py` 用 `MagicMock` 吞掉 `WanPipeline` 签名报错、`data-inspector` 解析 15ch EXR 却零引用进 `renderFrame()`，系统性违背 `AGENTS.md` 红线 #4（防静默失败三问）。
3. **安全红线与质量门禁的自消解逻辑**：`ReconPipeline` 用白模及格线 `vertex_count >= 1500` 自动置 `has_gen3d_highpoly = True`（红线 #3a 架空红线 #3b）；`TiledBypassExecutor` 把检测框置信度 `0.9` 当作解剖健康度跳过全部精修；`rsi/curriculum.py` 用 `val or 0.85` 将 `0.0` 与崩溃运行洗白为高分。

### 2.2 七大战略级整体改进支柱

| 战略支柱 | 核心目标 | 关键落地措施 | 承接 PR |
| :--- | :--- | :--- | :---: |
| **支柱一：真绿远端 CI、双扩展自动装配与零盲区静态门禁** | 恢复每次 PR 5/5 GitHub Actions 真绿阻断 | 修正 `sautille-stroke/aether` 与 Bazel 配置、`build_native.py` 默认安装 `aether_geo` + `aether_metrics` 至 `python/`；修复 `mypy` 包基清零 216 文件 20 错、真启用 Ruff 规则、封堵 `import_dag_guard` 4 处绕过 | `PR-153`, `PR-154`, `PR-155` |
| **支柱二：纯 Python 基础包铁律与 `@aether` 跨语言算子校准** | 守住 `AGENTS.md` §3.1/§3.3 并激活全图 SIMD 加速 | 恢复 `requires-python = ">=3.10"` 与轻量纯 NumPy 回退；修复球 64 简并面与圆锥非水密；取消 `aether_bridge` `(-1, 1)` 展平激活全分辨率 SIMD 色彩并接线 `native_compute_ssim`/`ciede2000` | `PR-157` |
| **支柱三：FGPP 相机几何、有界软肩与 RFC 0039 沙箱安全边界** | 消灭矩阵/色域错算与越界读取/SSRF 漏洞 | 修正 `cy=(1+p12)H/2`、Vulkan/RevZ 检测、`NaN` 旋转与深度阻断、`StyleProfile` 双向校验；前置 `workspace_root` 沙箱与 `OutputSink` 修复、SSRF IP 强绑定与 `HITL` 原子锁；修复 16px EDT 羽化与有界软肩 | `PR-158`, `PR-159`, `PR-160` |
| **支柱四：真白盒 `DiffusersWanBackend`、因果滑窗与真 PEFT 训练** | 终结空壳后端与 `64x64` 玩具 MLP | 对齐 `Wan2.1-T2V-1.3B/14B-Diffusers` 真签名与潜空间 `AutoencoderKLWan` 融合；接通 `fgpp_frames` / 切镜 `needs_reanchor` / `backend.model`；修 `trainer` 反向三元式与 `weights_only=True`、真流式 `FeatureStore` 与白名单 `lora {train,merge}` | `PR-161`, `PR-162` |
| **支柱五：`@aether//geo` 焊接网格/UV/QEM、红线 #3 与非退化精修** | 兑现 `SYN-RFC-0060/0061/0062` 真实闭环 | 修复 MC 顶点焊接+轴序+绕序、保留闭合网格 xatlas 拆点 UV、QEM 减面与水密一致细分、6-DoF 尺度；封堵红线 #3 自旁路、接通 Mitsuba 真 AD 梯度与世界 UV 去光照；重构独立健康分、3 类真融合核与 `w_curr -> 1.0` 防鬼影缝合 | `PR-163`, `PR-164`, `PR-165` |
| **支柱六：真 30 场景基准、世界尺度质检、防洗白 RSI 与真 4-Tab 检查器** | 让评测、飞轮与桌面检查器真实可用 | 激活 T1–T5 多物体/骨骼/多光源与真 `light_dir_error_deg`、修 `applicable_keys` 与 `evaluate_bundle` EXR 真读、修 FVD `uint8` 缩放；保留世界尺度、阻断 RSI `0.0`/崩溃洗白与 Viewer XSS；`Data Inspector` 接通 15ch EXR/多实体/4-Tab 并根治 `resize` 冻屏 | `PR-166`, `PR-167`, `PR-168`, `PR-169` |
| **支柱七：反恒等真测试、SLA `<=20s/40s` 与权威 SSOT 收口** | 彻底终结假绿测试与文档幻觉 | 重写 `WIRE_CONTRACTS.md` 并清零 16 处断链；清除永真断言与 `MagicMock` 伪验证；清零自身 `DeprecationWarning` 并通过 `slow` 分层与 `xdist_group` 达成单进程快测 `<= 20s` / 慢测 `<= 40s`，14 门禁全绿封板 `v4.1.0` | `PR-156`, `PR-170`, `PR-171`, `PR-172` |

---

## 3. 全量分级实证缺陷台账（`22×P0 + 18×P1 + 16×P2 = 56` 项）

### 3.1 P0 阻断、崩溃、安全与红线违背级缺陷（`H10-P0-01 ~ H10-P0-22`）

| 编号 | 精确位置 | 实证缺陷与根因 | 承接 PR |
| :---: | :--- | :--- | :---: |
| **`H10-P0-01`** | [.github/workflows/ci.yml](../../.github/workflows/ci.yml#L26-L38), [L167-L169](../../.github/workflows/ci.yml#L167-L169), [L267-L272](../../.github/workflows/ci.yml#L267-L272) | GitHub Actions 5/5 全挂：`dervgen/aether` 404（应为 `sautille-stroke/aether`）、缺 `id: checkout-aether`、`native-bazel` 进入已删 `synapse/csrc/geometry` 秒崩、`python-fast`/`slow` 缺 Bazel 调 `build_native` 退 `30`、`--timeout=60 \|\| true` 假门禁 | `PR-153` |
| **`H10-P0-02`** | [pyproject.toml](../../pyproject.toml#L91-L96), [L141-L150](../../pyproject.toml#L141-L150), [build_native.py](../../scripts/build_native.py#L166-L210) | 裸 `mypy` 与 4 目录 `mypy --strict` 因 `tools/import_dag_guard.py` 双模块名直接 `exit 2`（检 `0` 文件），穿透后存在 `20 errors in 8 files`；Ruff `E501/W/PT/S*` 同时位于 `select` 与 `ignore` 且 `RUF059` 仍在 `ignore` | `PR-154` |
| **`H10-P0-03`** | [build_native.py](../../scripts/build_native.py#L83-L109), [L351-L355](../../scripts/build_native.py#L351-L355), [aether_bridge.py](../../python/synapse/bridge/aether_bridge.py#L35-L45), [setup_colab.sh](../../workenv/google-colab/setup_colab.sh#L67-L73) | `build_native.py` 默认不安装 `.so` 至 `python/` 而 `eb5a010` 删去 `../aether` 探测造成构建-加载死锁，且漏编 `//python:aether_metrics`；`setup_colab.sh` 无 Bazelisk 且探针仍引用已删 `synapse._c_geometry` | `PR-153`, `PR-155` |
| **`H10-P0-04`** | [tools/data-inspector/native/.bazelrc](../../tools/data-inspector/native/.bazelrc#L5-L29) | 子目录相对 `downloader.cfg` 在 `ubuntu24` 报错 `exit 2`/`30`；首选私有 IP `8.138.167.168:8443` 致公网 CI 挂起，`--config=opt` 双重展开 | `PR-153`, `PR-168` |
| **`H10-P0-05`** | [docs/WIRE_CONTRACTS.md](../WIRE_CONTRACTS.md#L13-L31), [L56-L69](../WIRE_CONTRACTS.md#L56-L69), [L108-L117](../WIRE_CONTRACTS.md#L108-L117) | 权威线格式 SSOT 出现 16 通道（幻觉 `RoughnessMetallic.B`，总计 `38 B/px`）、`MotionVector.XY` 错标 `FLOAT(4B)`、`.R`/`.G` 语义颠倒、伪造 `job.json`/`job_completed.flag` 与全错 SHM 偏移量；全仓另有 `11~16` 处已删 `csrc/geometry/` 断链 | `PR-156` |
| **`H10-P0-06`** | [camera_math.py](../../python/synapse/render_core/camera_math.py#L34-L107), [procedural.py](../../python/synapse/render_core/procedural.py#L82-L207), [roundtrip.py](../../python/synapse/metrics/roundtrip.py#L43-L81) | 强制 `require_aether_geo()` 删光纯 NumPy 回退违反 `AGENTS.md` §3.1/§3.3；`aether::geo::createSphere` 产生 64 个零面积极点简并面，`createCone` 顶底分离致非水密 | `PR-157` |
| **`H10-P0-07`** | [frame.py](../../python/synapse/fgpp/frame.py#L180-L195), [L244-L255](../../python/synapse/fgpp/frame.py#L244-L255), [L454](../../python/synapse/fgpp/frame.py#L454), [schema.py](../../python/synapse/fgpp/schema.py#L146-L148) | `load_metadata` `cy = (1.0 - p12)*H/2` 垂直翻转（应为 `1.0 + p12`）；Vulkan `RH_ZO` / 有限 Reversed-Z 误判为 OpenGL（`z_near` 错 `2x`）；全 `NaN` 旋转矩阵穿透 `SO(3)` 校验；`NaN` 深度返 `1.0` 违背 C++ `fgpp_types.h:143` (`0.0f`) | `PR-158` |
| **`H10-P0-08`** | [bundle_worker.py](../../python/synapse/pipeline/bundle_worker.py#L256-L396), [L1391-L1392](../../python/synapse/pipeline/bundle_worker.py#L1391-L1392) | `from_rfc0039_job_bundle` 在沙箱建立前对任意宿主机路径算 SHA256 并泄露前 16 位哈希；将 `OutputSink.format="mp4"` 当输出路径致 `CWD != workspace_root` 必触沙箱自锁；丢弃 `scene_spec.resolution`/`frame_range` 与 `asset_manifest` 映射 | `PR-159` |
| **`H10-P0-09`** | [wan_diffusers.py](../../python/synapse/adapters/wan_diffusers.py#L42-L53), [L195-L229](../../python/synapse/adapters/wan_diffusers.py#L195-L229), [colab_runner.py](../../scripts/colab_runner.py#L257-L279) | `DiffusersWanBackend` 默认模型 ID `Wan-AI/Wan2.1-T2V-1.4B` 不存在，向 `WanPipeline.__call__` 传不存在的 `control_tensor`/`warm_start_frames` 必抛 `TypeError`，`_vae` 零引用；Colab E1 实例化无后端 `WanAdapter()` 必退 `30` | `PR-161` |
| **`H10-P0-10`** | [trainer.py](../../python/synapse/training/trainer.py#L445), [L723](../../python/synapse/training/trainer.py#L723), [_lora.py](../../python/synapse/cli/_lora.py#L148-L196), [stages/training.py](../../python/synapse/pipeline/stages/training.py#L175-L226) | `trainer.py:723` `z_pred if "z_pred" not in dir() else v_pred` 反向三元式必抛 `UnboundLocalError`；`weights_only=False` 存 Pickle RCE；CLI/Stage 硬编码 `64x64` `_SmallDiT` 产出非法 `"linear"` 键且 `lora merge` 未调 `MultiLoRAManager` | `PR-162` |
| **`H10-P0-11`** | [multiview.py](../../python/synapse/recon/multiview.py#L687-L704)（对照 `../aether/geo/src/marching_cubes.cc:459` 与 `bindings_geo.cc:264`） | `aether_geo.marching_cubes` 输出非索引三角形面汤（`V=3F`）、将 `(nx,ny,nz)` 按 `(dim_z,dim_y,dim_x)` 转置 X/Z 轴并翻转 CW 内向法线，破坏水密性与 `quad_remesh`/`delight` | `PR-163` |
| **`H10-P0-12`** | [recon/pipeline.py](../../python/synapse/recon/pipeline.py#L145-L156) | 红线 #3 自旁路：`has_gen3d_highpoly=None` 时自动按 `vertex_count >= 1500` 置 `True`，令所有合法角色白模自动豁免 Track-B Gen-3D 重建门禁 | `PR-164` |
| **`H10-P0-13`** | [mitsuba_optimizer.py](../../python/synapse/recon/loop/mitsuba_optimizer.py#L454-L466), [forward_guide.py](../../python/synapse/recon/loop/forward_guide.py#L224-L252), [closed_loop_runner.py](../../python/synapse/recon/loop/closed_loop_runner.py#L191-L228) | `MitsubaInverseOptimizer` 用 `np.array` 截断 AD 图调 `dr.backward` 零梯度；`ForwardGuide` 调不存在的 `set_cut_scene` 与错参 `adapter.generate`；CLI/Worker 默认对着 `0.5` 灰帧反演材质并贴回原视频 | `PR-164` |
| **`H10-P0-14`** | [tracking.py](../../python/synapse/recon/tracking.py#L650-L707), [recon/pipeline.py](../../python/synapse/recon/pipeline.py#L107-L114) | 单位范数 `F` 奇异值纯旋转检测失效；相邻帧尺度恢复错用帧 `0` 与帧 `t-1` 矩阵（`t=1` 零基线）；`ReconPipeline` 丢弃 `visibility` | `PR-163` |
| **`H10-P0-15`** | [tiled_executor.py](../../python/synapse/refine/tiled_executor.py#L100-L106), [refine/pipeline.py](../../python/synapse/refine/pipeline.py#L313-L320), [L479](../../python/synapse/refine/pipeline.py#L479) | 把 ROI 检测置信度（默认 `0.9`）当解剖健康分（`>0.8` 旁路），致默认 CLI/Worker 100% 跳过精修，并吞掉 `strict=True` 门禁错误伪装成功返回原帧 | `PR-165` |
| **`H10-P0-16`** | [fusion.py](../../python/synapse/refine/fusion.py#L80-L126), [L235-L274](../../python/synapse/refine/fusion.py#L235-L274) | `LAPLACIAN_FREQUENCY`、`CASCADE`（模糊项抵消且权重倒挂 `4/9`）、`POISSON_GRADIENT`（不动点初始化致 200 次迭代为零操作）三大融合算子代数退化为平凡线性加权 | `PR-165` |
| **`H10-P0-17`** | [bench_runner.py](../../python/synapse/pipeline/bench_runner.py#L358-L403), [L504-L629](../../python/synapse/pipeline/bench_runner.py#L504-L629), [L826-L843](../../python/synapse/pipeline/bench_runner.py#L826-L843) | 导入不存在模块 `synapse.style.intrinsic` 且 `light_dir_error_deg(gt_ld, gt_ld)` 自比较恒 `0°`；`applicable_keys` 含未算 `chamfer_mm` 致 24/30 场景挂 Profile 必败；`evaluate_bundle` 只看第 0 帧且不读 `.exr` | `PR-166` |
| **`H10-P0-18`** | [model_evaluator.py](../../python/synapse/inspect/model_evaluator.py#L276-L297), [L309](../../python/synapse/inspect/model_evaluator.py#L309) | `_default_load_mesh` 各自独立 `mesh.normalize()` 抹除 100x 单位尺度和刚体位移；`evaluate_single_model` 无参考模式默认仍强求参考网格而必败 | `PR-167` |
| **`H10-P0-19`** | [curriculum.py](../../python/synapse/rsi/curriculum.py#L352-L438) | Python `or` 把 `0.0` 分与 `rep.error` 崩溃任务洗白为 `0.85` 均分，允许崩溃候选通过 `PromotionGate` | `PR-167` |
| **`H10-P0-20`** | [inspector_app.cc](../../tools/data-inspector/native/inspector_app.cc#L619-L752), [L1097-L1354](../../tools/data-inspector/native/inspector_app.cc#L1097-L1354), [app.js](../../tools/data-inspector/renderer/src/app.js#L319-L345) | 15ch EXR (`m_fgppAov`)、多实体槽 (`m_entities[1..N]`) 与 4-Tab (`m_activeTab`) 在 `renderFrame()` 中零引用（违反红线 #4 接线问），AOV 视图仍画假面片索引 | `PR-168` |
| **`H10-P0-21`** | [prepare_datasets.py](../../scripts/prepare_datasets.py#L178), [L550-L555](../../scripts/prepare_datasets.py#L550-L555), [run_cpu_verification.py](../../scripts/run_cpu_verification.py#L344) | `prepare_datasets.py` 回退缺 5 通道崩溃且伪造 `caelum-core`/`spp=256` 溯源；`run_cpu_verification.py` E4 传绝对路径给 `LoRAManifest` 崩溃 | `PR-169` |
| **`H10-P0-22`** | [synapse_colab_workflow.ipynb](../../workenv/google-colab/synapse_colab_workflow.ipynb#L113)（及本地新增 `vscode_tunnel.sh`） | Notebook Cell 4 f-string 内含未转义物理换行触发 `SyntaxError` 令防扣费看门狗失效；`vscode_tunnel.sh` 使用 `curl -k` 禁 TLS 校验执行远端二进制 | `PR-155` |

### 3.2 P1 重要功能、物理精度与安全缺陷（`H10-P1-01 ~ H10-P1-18`）

- **`H10-P1-01`** [aether_bridge.py](../../python/synapse/bridge/aether_bridge.py#L160-L190)：`reshape(-1, 1)` 超过 C++ `kMaxDim=65536` 上限，令全部 `>=148x148` 图像 SIMD 色彩静默失效；`native_compute_ssim`/`ciede2000` 未接线（`PR-157`）。
- **`H10-P1-02`** [blend.py](../../python/synapse/composite/blend.py#L296-L305)：`d_out` 在 `mask` 内恒为 `0`，致 16px 内环羽化权重恒等于 `1.0`（硬阶跃接缝）（`PR-160`）。
- **`H10-P1-03`** [aces_color.py](../../python/synapse/style/aces_color.py#L149-L166)：`soft_knee_forward_tonemap` 按 $\sim\sqrt{y}$ 无界增长，导致焦散 HDR 高光被硬截断偏色（`PR-160`）。
- **`H10-P1-04`** [profile_mapper.py](../../python/synapse/style/profile_mapper.py#L26-L46) & [wire_guard.py](../../tools/wire_guard.py#L267-L292)：`StyleProfile v2.0.0` 嵌套输出与扁平 Schema 冲突且未注册进 `_SCHEMA_FILES`（`PR-158`）。
- **`H10-P1-05`** [import_dag_guard.py](../../tools/import_dag_guard.py#L77-L79,L202-L256)：`aether_bridge.py` 整文件豁免及相对导入/关键字参数/`builtins.__import__` 4 处绕过（`PR-154`）。
- **`H10-P1-06`** [http_security.py](../../python/synapse/bridge/http_security.py#L89-L158) & [jev_critic.py](../../python/synapse/inspect/jev_critic.py#L121-L135)：DNS Rebinding TOCTOU、DNS 异常 Fail-Open、未禁代理与 `HttpJevTransport`/`registry.py` 裸调 `urlopen` SSRF（`PR-159`, `PR-166`, `PR-167`）。
- **`H10-P1-07`** [hitl.py](../../python/synapse/bridge/hitl.py#L153-L218)：损坏账本清零 900s 计时、`record_action` 锁句柄泄漏、`recon` 漏接账本（`PR-159`, `PR-164`, `PR-165`）。
- **`H10-P1-08`** [wan.py](../../python/synapse/adapters/wan.py#L654-L907)：忽略 `conditions.fgpp_frames`、`self._backend.model` 与切镜 `needs_reanchor`，未对齐 `4k+1` 尾窗与 `16px` 分辨率（`PR-161`）。
- **`H10-P1-09`** [cloud.py](../../python/synapse/adapters/cloud.py#L514-L871)：标准调用丢失 `text_prompt` 并拼出非法含点号环境变量名 `SEEDANCE2.5_API_URL`（`PR-161`）。
- **`H10-P1-10`** [manager.py](../../python/synapse/adapters/lora/manager.py#L78,L556-L609) & [patcher.py](../../python/synapse/adapters/lora/patcher.py#L125-L163)：按 CWD 解析相对路径、部分块 LoRA 跨层错配、假 Register Token 与非因果池化（`PR-162`）。
- **`H10-P1-11`** [feature_store.py](../../python/synapse/training/feature_store.py#L142-L150,L267,L329) & [trainer.py](../../python/synapse/training/trainer.py#L525,L622-L690)：`streaming=True` 仍全量缓冲 RAM、逆因果时域池化、`batch["t"]` 原地冻结、未训练 `LatentGeometryProbe` 与 CPU 模型触发 CUDA `GradScaler` 崩溃（`PR-162`）。
- **`H10-P1-12`** [recon/geometry.py](../../python/synapse/recon/geometry/__init__.py)：闭合网格 xatlas UV 100% 丢弃、`quad_remesh` 缺 QEM 减面致面数反向暴增且存 1-Ring T-junction 破洞、ICP 每轮重建 2 次 KD-Tree（`PR-163`）。
- **`H10-P1-13`** [multiview.py](../../python/synapse/recon/multiview.py#L437-L455,L653) & [delight.py](../../python/synapse/recon/delight.py#L849-L962)：相机位置均值张角恒 `180°`、`np.roll` 周期穿透与屏幕空间多视角叠影（`PR-163`, `PR-164`）。
- **`H10-P1-14`** [stitcher.py](../../python/synapse/refine/stitcher.py#L198-L200,L677-L686) & [refine/pipeline.py](../../python/synapse/refine/pipeline.py#L419-L423)：快运动反向降 `w_curr=0.30` 加重残影、HDR 环带硬跳变与 Kalman 框尺寸差 1px 致 LPIPS 失败（`PR-165`）。
- **`H10-P1-15`** [bench_runner.py](../../python/synapse/pipeline/bench_runner.py#L213-L270,L483-L486) & [temporal.py](../../python/synapse/metrics/temporal.py#L379,L526)：T1–T5 多物体/骨骼/多光源参数丢弃、FVD 代理 `uint8` `65,025x` 放大与 `is_proxy_metric=False` 误报（`PR-166`）。
- **`H10-P1-16`** [_rsi.py](../../python/synapse/cli/_rsi.py#L289-L290) · [stages/refine.py](../../python/synapse/pipeline/stages/refine.py#L114) · [viewer/server.py](../../python/synapse/viewer/server.py#L92,L166-L173)：报告模式覆写 `min_scenarios=0`、`.npy` 走 `json.loads` 崩溃、`viewer` 先 `resolve()` 破坏 symlink 防御与二次 `.replace()` XSS（`PR-165`, `PR-167`）。
- **`H10-P1-17`** [inspector_app.cc](../../tools/data-inspector/native/inspector_app.cc#L798-L816,L1186) · [shm-reader.js](../../tools/data-inspector/electron/shm-reader.js#L243) · [test-bundle-loading.js](../../tools/data-inspector/test/test-bundle-loading.js#L591-L598)：缩放冻屏、Seqlock 缓冲提前轮转、`-fno-exceptions` `abort()`、无穷 Bresenham 循环、OBJ 无体积上限与 macOS 漏 `--shm-file`（`PR-168`）。
- **`H10-P1-18`** [colab_runner.py](../../scripts/colab_runner.py#L291-L380,L448) & [prepare_benchmark_assets.py](../../scripts/prepare_benchmark_assets.py#L294-L305,L572)：E4 零训练步空壳、`SIGTERM` 卡死 `RUNNING`、VBench 生成 SHA256 与清单不一致（`PR-169`）。

### 3.3 P2 中低危去重、性能与测试卫生项（16 项汇总）

包括：3 套 Mesh 容器去重（`PR-160`）、`bench/__init__.py` 与 `generators/__init__.py` 自身导入触发 4 条 `DeprecationWarning`（`PR-171`）、`SynapseError.to_json_dict` 不支持 `ndarray`/`set`（`PR-159`）、`MemoryForecaster` `256x` 像素/潜空间偏差（`PR-161`）、`detector.py` `3.22 GB` 4K 缓冲（`PR-160`）、`torch_bridge` `int64` 视图错位（`PR-158`）、`cli/common.py` `"status"` 键被覆盖（`PR-167`）、`test_profiles_doc_sync.py` 永真断言与 `test_roundtrip_reconstruction.py` 全黑输入/5x 重复 `setup`（`PR-170`, `PR-171`）。

---

## 4. 演进移交结论

本报告确认的全部 **`22×P0 + 18×P1 + 16×P2 = 56`** 项缺陷及七大战略改进支柱已 100% 映射为可并发落地的 **Horizon 10 (`M37–M40`, `PR-153 ~ PR-172`)** 详细实施蓝图，详见 [docs/plans/PLAN-M37-M40-POST-RECONCILIATION-TRUE-CLOSURE.md](../plans/PLAN-M37-M40-POST-RECONCILIATION-TRUE-CLOSURE.md)。
