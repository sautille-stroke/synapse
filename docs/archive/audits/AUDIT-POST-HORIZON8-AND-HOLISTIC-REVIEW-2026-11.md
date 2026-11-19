# Synapse Horizon-8 后全库整体审查、改进战略与真值复核报告（2026-11）

- **文档状态**: `Accepted Holistic Review & Audit Report (v1.0)`
- **审计基线**: `main` @ `df93ac7`（`df93ac7577141b9008799c023d9a3c51dded773e`，PR `#130` / `#131` `"Release v3.6.0 Sealed"` 合入后全仓基线）
- **审计日期**: `2026-11-17`
- **操作模型标识**: `[gemini4]`
- **关联演进计划**: [docs/plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md)（Horizon 9: `M33–M36` / `PR-133 ~ PR-152`）
- **关联总计划**: [docs/IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md)（`v3.7`）
- **前序审计链**: [AUDIT-POST-REFACTOR-2026-11.md](AUDIT-POST-REFACTOR-2026-11.md) · [AUDIT-RFC0063-AND-CODEBASE-2026-11.md](AUDIT-RFC0063-AND-CODEBASE-2026-11.md) · [AUDIT-POST-HORIZON5-2026-11.md](AUDIT-POST-HORIZON5-2026-11.md) · [AUDIT-POST-HORIZON6-2026-11.md](AUDIT-POST-HORIZON6-2026-11.md) · [AUDIT-POST-HORIZON7-2026-11.md](AUDIT-POST-HORIZON7-2026-11.md)

---

## 1. 执行摘要、全仓定量盘点与 `ubuntu24` 12 项硬门禁独立复核

### 1.1 代码库定量规模盘点 (`df93ac7`)

在 `HEAD = df93ac7577141b9008799c023d9a3c51dded773e` 基线上对全仓五个正交代码与治理切片开展逐行只读审计、跨仓 SSOT（`../aether`、`../render-farm`、`../grand-studio`、`../fulgur`、`../caelum-core`、`../foundary`、`../all-overview`）交叉比对，并在云端主力机 `ubuntu24` 实跑复核全部门禁，当前全仓定量规模如下：

| 目录 / 子系统 | 文件数 / 规模 | 职责分布与明细 |
| :--- | :--- | :--- |
| **[python/synapse/](../../../python/synapse/)** | **`138` `.py` 文件 (`42,005` LOC) + `1` `py.typed` + `30` 场景 `.json` = `169` 文件 / `43,222` 文本行** | 20 个顶层领域子包/模块：`fgpp` · `render_core` · `style` · `composite` · `adapters`（含 `lora`） · `training` · `bridge` · `models` · `cli` · `pipeline`（含 `stages`） · `recon`（含 `loop`） · `refine` · `inspect` · `bench` · `rsi` · `metrics` · `viewer` · `generators` · `errors` · `telemetry` |
| **[csrc/geometry/](../../../csrc/geometry/)** | **`10` 文件 (`550` LOC)** | Tier 0 C++20 原生几何拓扑校验核与 pybind11 绑定（Bazel 委托 `@aether//geo`、`@aether//math`） |
| **[tools/](../../../tools/)** | **`2` Python 守卫 (`1,286` LOC) + `data-inspector` (`34` 文件 / `7,446` 文本行)** | `wire_guard.py` / `import_dag_guard.py`；桌面检查器含 Native C++20 SHM 视口、Electron 主进程/预加载、前端与 Bundle/SHM 测试 |
| **[scripts/](../../../scripts/)** | **`9` Python 脚本 (`4,092` LOC)** | `prepare_datasets.py` · `colab_runner.py` · `run_cpu_verification.py` · `run_roundtrip_verification.py` · `prepare_benchmark_assets.py` · `evaluate_3d_models.py` · `dev_preview_viewer.py` · `build_native.py` |
| **[schemas/](../../../schemas/)** | **`6` JSON Schemas (`608` LOC)** | `bundle.v1.0` · `inspection.v1.0` · `inspection.v2.0` · `job_bundle.v1.0` · `lora.v1.0` · `refine_session.v1.0` |
| **[tests/](../../../tests/)** | **`60` Python 测试文件 (`29,179` LOC)** | `ubuntu24` 实收集 **`1,408`** 项 pytest 用例（`1,274` fast + `134` slow）；另含 `2` 组 Bazel GoogleTest 与 `23` 项 Node Bundle/SHM 断言 |
| **[docs/](../../../docs/) & [workenv/](../../../workenv/)** | **`40` Markdown 文档 (`12,745` LOC) + `8` Colab 工作区文件 (`898` LOC)** | `SYN-RFC-0060..0063` · `DESIGN_*` · `plans/` · `audits/` · `archive/` · Colab 自动化装配与 Notebook 套件 |

### 1.2 `ubuntu24` 12 项反作弊硬门禁独立实测复核表 (`v3.6.0` 宣称 vs `df93ac7` 实跑真值)

> [!CAUTION]
> **门禁真值核查结论**：Horizon 8 (`PR-112 ~ PR-129`) 确实在**静态类型与代码卫生**（`209` 文件 `mypy --strict` `0` 错、Ruff 解禁 8 条核心规则且 `0` 报错）、**部分物理与安全细节**（`UINT32` EXR、$C^1$ 有理软肩、CCW 256 项 MC 表、去除局部 `alpha^2`、修复 `roundtrip.py` Quad 邻边水密与 HDR 禁除 255、`0 broken links`）取得实质进展；但对 `HEAD = df93ac7` 在 `ubuntu24` 无竞争环境下的独立实测以及五切片逐行复审表明，12 项硬门禁中仅有 **3 项完全达标**，**5 项失败（含 4 项 Horizon-8 新引入的回归崩溃）**、**4 项部分通过或存在恒等自证（Tautological）掩盖**。

| 门禁 | 宣称门槛 (`PLAN-M29-M32 §6`) | `ubuntu24` (`df93ac7`) 独立实测真值 | 核查判定 |
| :---: | :--- | :--- | :---: |
| **Gate 1a** | 单进程严禁 `-n auto`，`-m "not slow"` `0 fail/0 skip`，**`<= 25s`**（`PR-126` 目标 `<= 20s`） | 无竞争实测 `1274 passed, 134 deselected in 38.60s`（`real 41.80s`），超 `25s` 门槛 **`1.67x`**（超 `PR-126` `20s` 目标 `2.09x`） | ❌ **耗时超标 `1.67x`** |
| **Gate 1b** | 并行 `-m "not slow" -n auto` `0 fail/0 skip`，**`<= 12s`** | 无竞争实测 `1274 passed in 16.03s`（`real 17.14s`），超 `12s` 门槛 `1.43x` | ❌ **耗时超标 `1.43x`** |
| **Gate 2** | 并行 `-m slow -n auto` `0 fail/0 skip`，**`<= 45s`** | 无竞争实测 `134 passed in 90.75s`（`real 92.36s`）：`test_run_all_scenarios` 单函数串行 `71.65s`、`test_real_direct_compile_and_load` `69.10s`、`test_inspect_align` 两项各 `66.59s`/`66.09s`、`test_humanoid_53_scenarios_no_error` `50.44s` | ❌ **耗时超标 `2.05x`** |
| **Gate 3** | `ruff check` `0 errors`，`pyproject.toml` 移出 8 条核心规则 | `All checks passed!` 且 `B008/B017/B023/B904/B905/SIM115/RUF012/RUF100` 确已移出；但 `RUF059` 仍屏蔽（掩盖 `import_dag_guard.py:435` 未使用 `pkg_files`），且缺 `E501`/`W`/`PT`/`S301`/`S310` | ✅ **通过（有规则盲区）** |
| **Gate 4** | 全仓 `mypy --strict python/synapse tests/ scripts/ tools/` `>= 207` 文件 `0 errors` | `Success: no issues found in 209 source files` | ✅ **真实达标 (`209` 文件)** |
| **Gate 5** | `import_dag_guard.py --check` 覆盖全作用域、`import_module` 与全部包目录 | 退出 `0`（静态逆向边清零），但 `tools/import_dag_guard.py:435` 解包 `pkg_files` 后从未读取（孤立未注册包逃逸），且 `_extract_synapse_pkg("synapse")` 返回 `None` 放行裸 `import synapse; synapse.pipeline...` | ⚠️ **静态通过但留 2 绕过** |
| **Gate 6** | `wire_guard.py --check --require-aether` `0 SKIPPED` + `model_json_schema()` 双向校验 | 15ch `34 B/px` 与 `fgpp_types.h` 校验通过；但仍输出 `SKIPPED (upstream @aether//style not present)`，且 `wire_guard.py` 零调用 `model_json_schema()`（漏检 `LoRAManifest.adapter_path` 放行 `../` 穿越） | ⚠️ **基础通过（缺深度校验）** |
| **Gate 7** | `csrc/geometry` `//:test_geometry` (`--config=opt`) + `build_native.py --backend=compiler --check --json` | `//:test_geometry PASSED in 0.1s` + 单 JSON `{"status": "ok"}`；但两处 `.bazelrc` 漏写 `build:opt --compilation_mode=opt`（脱离 `all-overview` 时退回 `-O0`），`build_native.py:62,662` 硬编码 `timeout=15.0` 忽略 `--timeout` | ⚠️ **通过但默认退化 `-O0`** |
| **Gate 8** | `(cd tools/data-inspector/native && bazel test //:test_inspector)` + `node tools/data-inspector/test/test-bundle-loading.js` (`0 skipped`) | `//:test_inspector PASSED in 0.2s`；但字面执行 `node` 测试时因未先构建 `//:data-inspector-native` 直接 **`22 passed, 1 failed`**（提示错靶标 `//:inspector_app`）；手动先 `bazel build //:data-inspector-native` 后 Node 23/23 通过，**但实查代码发现 macOS/Win 三重 `--shm-file` 断链导致非 Linux 视口 100% 黑屏** | ❌ **字面命令失败 + 跨端断裂** |
| **Gate 9** | `prepare_datasets.py` + `run_cpu_verification.py` 零 `SIGSEGV` 并产出真实 15ch EXR | **`prepare_datasets.py` 实跑崩溃 (`CalledProcessError: bazel build //course:102-pbr-direct returned exit status 255`)**；`L139` `VK_ICD_FILENAMES=""` 禁用 Lavapipe；回退写全零帧未调 `SoftwareRasterizer`；`L424-427` 仍伪造 `"generator": "caelum-core", "spp": 256` | ❌ **实跑崩溃 (`exit 255`) + 伪造溯源** |
| **Gate 10** | `tests/test_cross_repo_contracts.py` 实跑 `@aether`, `render-farm`, `inspector` SHM | `11 passed in 0.61s`；**但 Contract 1 复制错式 `p02=(2*cx-W)/W` 恒等自证、Contract 3 未调 `JobBundleWorker.execute()` 且未对齐 Go `models.go`、Contract 4 仅用 Python 自写自读临时文件（零 C++/Node）** | ❌ **恒等假绿掩盖真 P0** |
| **Gate 11** | 全仓 Markdown 相对路径扫描 `0 broken links` | 独立扫描全仓 `62` 个 `.md` 文件共 `1,156` 条相对链接，**`broken=0`** | ✅ **真实达标 (`1,156` 链接)** |
| **Gate 12** | `test -f .github/workflows/ci.yml && gh run list -L 3` 全 Job `success` | **`.github/` 目录不存在**，`gh run list` 返回 `no runs found`，且 `.github/workflows/ci.yml` 存在仓内嵌套 `./aether` checkout 必败等 5 处致命缺陷 | ❌ **零 GitHub CI** |

### 1.3 Horizon 1–8 已沉淀的正向架构资产（肯定项）

尽管仍存未合龙断点与回归，`synapse` 历经 M1–M32 已积累出清晰且有价值的核心工程底座：
1. **严格守住 Server-side Zero-Python 红线**：全仓无常驻 HTTP/WebSocket 微服务或任务队列守护进程，[python/synapse/viewer/server.py](../../../python/synapse/viewer/server.py) 保持纯静态 HTML 导出，定位清晰收敛为无状态单步 CLI Worker 与 Python 算法 SDK。
2. **静态工程纪律显著增强**：全仓 `209` 个 Python 源文件达成 `mypy --strict` 零错误，Ruff 解禁 8 条关键规则零报错，静态包级 5-Tier DAG 保持无环。
3. **底层物理协议与数值原语具备坚实基础**：15 通道（`34 B/px`）EXR 读写与 `Cryptomatte.ID` 原生 `UINT32` 无损往返、`@aether//color` `kSrgbToAcesCg` 对齐与 $C^1$ 有理软肩色调映射、4D 齐次近面裁剪与 $O(HW)$ Felzenszwalb-Huttenlocher 精确欧氏距离变换、256 项外向绕序 Marching Cubes 拓扑表、参考锚定 Diffusion-DPO 闭式梯度、安全流式 Tar 读取以及 Electron 安全沙箱基线均已真实落盘。

### 1.4 全仓七维健康度评分卡 (`df93ac7` 实测真值)

| 审查维度 | H7 得分 | H8 (`df93ac7`) 得分 | 核心诊断结论 |
| :--- | :---: | :---: | :--- |
| **跨仓 Wire 契约、坐标系与色彩空间一致性** | `66` | **`65 / 100`** | `p02` 主点水平镜像仍在且被假合同测试锁死；`run_worker_cli` 未接 `from_rfc0039_job_bundle` 且与 Go `models.go` 结构体不一致；仓内分裂 OpenGL `-Z` / Vulkan `[0,1]` / OpenCV `+Z` 与 AP1/Rec.709 混用 |
| **仓内分层架构、内聚度与内存效率** | `88` | **`74 / 100`** | 静态 DAG 清零但存裸 `import synapse` 与未使用 `pkg_files` 绕过；3 套网格、3 套软光栅与重复色彩/Warp 并存；97f 4K 序列峰值内存因全量 `float64` 放大至 `38+ GB` |
| **生成适配、长视频调度与 PEFT 训练真实度** | — | **`62 / 100`** | `WanAdapter` 双重注入 LoRA、对 `uint8` 像素套用零均值潜空间融合致 `1.41x` 闪白、错取 `[94:97]` 未来帧且零 `diffusers` 真实接入；`StatelessLoRATrainer` 在 CUDA 上 `cpu/cuda` 必崩且用潜速度前 3 维对齐 3D 法线 |
| **3D 逆向重建、可微闭环与解剖精修成熟度** | `64` | **`64 / 100`** | 红线 #3（白模底线/反投门禁）全仓生产零调用；TSDF 未按 `weight > 0` 门控出双壁空腔、`quad_remesh` 一环 T-junction、`delight` 自抵消、Refiner/Stitcher 旁路 `PSNR=0` 与反转 EMA；`csrc/geometry` 未绑定 `@aether//geo` 11 项现有 C++20 核 |
| **原生 C++20 几何核与三端 `Data Inspector`** | `65` | **`68 / 100`** | `@aether//pal` SHM 基础骨架与安全沙箱落地 ✅；但 `loadGlbMesh` 存 `uint32_t idxCount * 4` 32 位溢出堆越界读，macOS/Win 三重 `--shm-file` 断链 100% 黑屏，3/4 标签页为空壳且零真 EXR 渲染 |
| **测试反恒等有效性、性能 SLA 与 CI 基建** | `58` | **`66 / 100`** | 209 文件 `mypy` 与 1,156 链接真绿 ✅；但 `.github/workflows/` 仍为空，单进程快测 `38.6s`/慢测 `90.8s` 超 SLA `2x`，`roundtrip`/`cross_repo` 存在假绿恒等测试，4 项 Horizon-8 回归崩溃逃过单测 |
| **文档治理、SSOT 同步与知识库可维护性** | `70` | **`72 / 100`** | 相对链接零断链 ✅；但 40 份文档（11 份历史 Plan + 5 份 Audit + 10 份重叠 Design）膨胀失步，`README.md` 状态表停在 `PR-113..124`，`io_spec.md` 与 `DESIGN_REVIEW_AND_ALIGNMENT.md` 仍留错字段与旧 `tanh` 描述 |
| **综合健康度总评** | **`68`** | **`67 / 100`** | **工程底座更稳，但距离“真生产闭环”仍差临门一脚** — 全仓共实证定位 **`18×P0 + 24×P1 + 12×P2 + 4×P3 = 58`** 项缺陷及 7 大系统性架构瓶颈，建议通过 **Horizon 9 (`M33–M36`, `PR-133 ~ PR-152`)** 完成“止血纠偏 + 架构演进合龙” |

---

## 2. 系统性根因复盘与七大维度整体改进建议（Holistic Diagnosis & Strategic Pillars）

### 2.1 为何历经 Horizon 5 → 8 四轮收口仍出现“门禁失真与局部回归”？（五大结构性根因）

1. **缺乏远端 GitHub Actions 机器强门禁阻断**：`.github/workflows/ci.yml` 始终未生效，本地/人工运行子集命令易发生漏建二进制（Gate 8）、旧缓存或未测目录遗漏（Gate 9/12），缺乏提交即阻断的客观裁判。
2. **恒等自证（Tautological）与过度 Mock 测试遮蔽真实断裂**：包括复制错误公式（`p02`）、Mock 掉核心权重补丁（`test_wan_lora_wiring`）、传入全黑帧 + 真值 `initial_mesh` 跳过重建（`test_roundtrip_reconstruction`）、跳过 `JobBundleWorker.execute()`（`test_cross_repo_contracts`），以及缺少 `device="meta"` / CUDA 多设备张量校验与 CLI 参数分支覆盖（导致 `_rsi.py` `UnboundLocalError`、`bench_runner` `applicable_keys`、`colab_runner` `real-gpu` 回归直接流入 `main`）。
3. **“CPU 启发式玩具代理 vs 生产级模型/算子”结构性代差**：`WanAdapter` 仍停留在 NumPy 预处理与抽象 Transport（`diffusers` 引用数为 `0`），`StatelessLoRATrainer` 用随机矩阵模拟编码器并用潜空间通道充当法线，`MultiViewInitializer` 在 Python 中做 `32³` (`6.25 cm`) 粗体素、`Hand/FaceRefiner` 以单候选直通绕过 5 大多模型融合算子，导致“全单测通过”与“真实 GPU 产线可用”之间存在显著鸿沟。
4. **跨仓协议靠手写镜像而非上游 Schema/Fixture 驱动**：未直接对齐 Go `../render-farm/internal/domain/models.go` 的 `JobBundle` 结构与 `../fulgur` Vulkan/Reversed-Z 投影矩阵；同时仓内隐含了 OpenGL `-Z`（`fgpp`）、Vulkan `[0,1]`（`fulgur`）、OpenCV `+Z`（`recon`）三套未显式转换的相机系，以及 AP1 `ACEScg`、Linear `sRGB`、Display `sRGB` 的混用。
5. **内部重复造轮子、全量 `float64` 内存爆炸与文档膨胀失步**：三套网格类型、三套软光栅入口、多份色彩矩阵与 Warp 实现各自演化；全序列 `float64` 转化将 97 帧 4K 推理/评测峰值内存推高至 `38+ GB`；`csrc/geometry` 仅封装 `@aether//geo` 的极小子集而把重计算留在 Python 循环；40 份平级设计/审计/子计划文档导致维护成本随迭代线性恶化。

### 2.2 七大战略级整体改进建议（7 Strategic Improvement Pillars）

| 战略支柱 | 核心目标 | 关键落地措施 | 承接 PR |
| :--- | :--- | :--- | :---: |
| **支柱一：真 CI 阻断与反恒等属性测试体系** | 消灭假绿幻觉与局部回归，实现每次 PR 自动双端强校验 | 1. 激活 `.github/workflows/ci.yml` 并退役 `.ci/`，修通同级 `synapse`/`aether` 检出、前置构建 `//:data-inspector-native` 与 `mypy` 错误日志回显；<br>2. 恢复 `.bazelrc` `build:opt --compilation_mode=opt` 与 `build_native.py` `--timeout` 透传；<br>3. 引入 `hypothesis` 属性测试与畸形输入 Fuzzing（偏心/正交矩阵投影、EXR 极值、`jsonschema` 等价性、`.glb` 越界头、C++/Python 随机网格一致性）；<br>4. 重写 `test_cross_repo_contracts` 与 `test_roundtrip_reconstruction` 为真端到端投影/Worker/SHM/重建验证，并发提速使单进程快测 **`<= 20s`**、慢测 **`<= 40s`**。 | `PR-133`, `PR-148`, `PR-149`, `PR-150` |
| **支柱二：跨仓 Wire 契约、坐标系与色彩空间大一统** | 与 `aether` / `fulgur` / `caelum-core` / `render-farm` / `grand-studio` 100% 字节级与几何级零歧义互通 | 1. 修复 `p02 = (W - 2*cx)/W`，建立显式 `CameraConvention`（统一 OpenGL `-Z`、Vulkan `RH_ZO`/`Infinite Reversed-Z`、OpenCV `+Z`）与显式色域标签（AP1 `ACEScg` vs Linear `sRGB` vs Display `sRGB`）；<br>2. `run_worker_cli` 原生路由 `from_rfc0039_job_bundle`，1:1 对齐 Go `models.go` 与 TS `grand-studio` 字段并丰富 `ResultManifest` 错误信封；<br>3. 补齐 `fgpp.v1.0` / `style_profile.v2.0.0` Schema，在 `wire_guard.py` 启用 `model_json_schema()` 全量校验并封堵 `import_dag_guard.py` 2 处绕过；<br>4. 强化 `http_security.py`（`getaddrinfo` + `ipv4_mapped` 解包 + IP 绑定）与 `JobBundleWorker` 全祖先路径防穿越。 | `PR-134`, `PR-135`, `PR-137`, `PR-139` |
| **支柱三：核心 SDK 去重瘦身与流式 `< 1.5 GB` 低内存架构** | 消除重复轮子，突破 1080p/4K 长序列内存墙 | 1. 合并 `MeshData` / `ProceduralMesh` / `ModelAsset` 为单一网格基元（焊接水密几何体 + 无自交 `Humanoid-53` + 保接缝 `load_obj`），收敛重复光栅/色彩/Warp/金字塔实现至 `render_core` 与 `style.aces_color`；<br>2. 落地流式 `iter_fgpp_frames` 与 `style`/`ssim` 逐帧 `float32`（加空间步进采样），彻底消除 `style/extractor.py` `38.6 GB` 与 `ssim` `20 GB` `float64` 峰值，将 97f 4K 内存压降至 **`< 1.5 GB`**。 | `PR-137`, `PR-138` |
| **支柱四：真白盒 `DiffusersWanBackend` 与物理可控 PEFT 管线** | 摆脱纯 Mock/占位编码器，使 E1–E4 在 Colab/GPU 真实可训可推 | 1. 修复 `WanAdapter` 双重 LoRA 注入、多窗 Hook 越界、`[stride-3:stride]` 暖启动错位与非零均值像素融合闪烁；<br>2. 新增可选 `DiffusersWanBackend` 真实适配 `AutoencoderKLWan` + `UMT5` + `WanTransformer3DModel` (`attn1`/`attn2`) 并在潜空间完成滑窗融合；<br>3. 修复 `StatelessLoRATrainer` CUDA/CPU 撕裂与 `T -> tau` 时域池化，用轻量潜空间探针头/分块 VAE 解码替代 `v_pred[..., :3]` 伪法线损失，流式构建 `FeatureStore` 并打通 `synapse lora {prepare-data,train,merge}`。 | `PR-139`, `PR-140`, `PR-147` |
| **支柱五：`@aether//geo` C++20 核扩围与真 3D 逆向/精修闭环** | 告别 `32³` 玩具体素与单候选直通，兑现 `SYN-RFC-0060/0061/0062` 与红线 #3 | 1. 将 `@aether//geo` 现成的 `MarchingCubes`、`SubdMesh`、`MeshUVUnwrapper (xatlas)`、QEM 减面与 `Bvh` 最近点全面绑定进 `csrc/geometry`（加速 `50–200x` 并消灭 `60 GB` ICP 广播）；<br>2. 修复 TSDF `weight > 0` 空腔、`quad_remesh` 一致细分与世界空间 UV 去光照，强制挂接 `enforce_distillation_geometry_gate` 与 `asset_3d` 15m 熔断并导出 `.obj`/`.usda`+OpenPBR；<br>3. 落地真实 `MitsubaInverseOptimizer` (`[invrender]`) 与 GGX/SH 解析优化器及推理前 $\kappa=1.0$ 切镜；<br>4. 修复 `refine` 全旁路门禁、反转 EMA、`POISSON`/`CASCADE` 融合、`PersistentHitlLedger` 持久化并接入多候选 `AnatomicalModelEnsemble`。 | `PR-141`, `PR-142`, `PR-143` |
| **支柱六：`Synapse-Bench` 30 场景真值化、RSI 自动桥接与三端 `Data Inspector`** | 让评测、质检、RSI 飞轮与桌面端真正闭环可用 | 1. 修复 `bench_runner` `applicable_keys` 回归与 `model_evaluator` 无参考/世界尺度，打通 22/30 场景被丢弃的相机/多物体/光照/骨骼动画并接入 GSO/Objaverse 扫描资产；<br>2. 打通 `HttpJevTransport`（`top_logprobs`）与 `CurriculumManager <-> BenchPipelineRunner <-> PromotionGate` 自动飞轮桥，修复 `cli/_rsi.py` 崩溃与 `PreferencePair` 溯源洗白；<br>3. 彻底封堵 `loadGlbMesh` 32/64 位整数溢出堆越界读，修通三端 `--shm-file`、3 槽无撕裂 `Buffer`、多模型源隔离、真 15ch EXR AOV 与四标签页同步视图。 | `PR-144`, `PR-145`, `PR-146` |
| **支柱七：文档体系三层收敛与自动化防漂移治理** | 终结 40 份文档膨胀与“封板与代码脱节”现象 | 1. 归档已完成历史子计划（`PLAN-M1..M32`）与阶段审计至 `docs/archive/{plans,audits}/`；<br>2. 整合为 4 份核心活文档（`ARCHITECTURE.md`、`WIRE_CONTRACTS.md`、`SYN-RFC-0060..0063`、主路线图）；<br>3. 统一修正 `README.md`、`io_spec.md` 与 `synapse-colab-{cpu,gpu}` 命名，扩展 `test_profiles_doc_sync.py` 自动化守护文档与代码实时一致。 | `PR-147`, `PR-151`, `PR-152` |

---

## 3. Horizon-8 (`PR-112 ~ PR-129`, `#112 ~ #131`) 宣称 vs 代码实态核验表

| 计划 PR | GitHub PR & Commit | 存在 | 接线 | 执行 | 审计实态 | 实测未闭环 / 回归 / 假绿缺口 | 移交 H9 |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **`PR-112`** | `#112` (`22be586`) | ✅ | ⚠️ | ⚠️ | **⚠️ 部分合入** | 209 文件 `mypy --strict` 与 8 条 Ruff 规则真清零 ✅；但 `.github/workflows/` 仍不存在，`.ci/` 有嵌套 `./aether` 必败等 5 错，`RUF059` 掩盖死变量 | `PR-133` |
| **`PR-113`** | `#113` (`11d872d`) | ✅ | ⚠️ | ⚠️ | **⚠️ 部分闭合** | 缺深度默认拦截与 26 项 `wire_schemas` 落地 ✅；但 `fgpp/frame.py:80` `p02` 符号反转**根本未改**，`import_dag_guard` 漏 `pkg_files`/裸 `import synapse`，`wire_guard` 未调 `model_json_schema()`，`LoRAManifest` 放行 `../` | `PR-134` |
| **`PR-114`** | `#114` (`8c60241`) | ✅ | ✅ | ⚠️ | **⚠️ 部分闭合** | $C^1$ 有理软肩与 `morphology` 修复落地 ✅；但 `blend.py:297-317` 仍用双向腐蚀造内环 `1->0` 接缝、`composite_optical_increments` 2px 引导仍失效、`extractor.py` 未转 ACEScg 且 IRLS/Fréchet 有缺陷 | `PR-137`, `PR-138` |
| **`PR-115`** | `#115` (`2a063b9`, `8c3a646`) | ✅ | ⚠️ | ⚠️ | **⚠️ 部分闭合** | `from_rfc0039_job_bundle`、`register_stage` 去重与流式哈希落地 ✅；**但 `run_worker_cli` (`L1052`) 从未调用 `from_rfc0039_job_bundle`** 且字段未对齐 Go `models.go`；`_inspect`/`_recon` 前置 `--json` 仍污染 stdout；`viewer` 先 `resolve()` 致 symlink 检测失效 | `PR-135`, `PR-137` |
| **`PR-116`** | `#116` (`9b6d892`) | ✅ | ⚠️ | ❌ | **❌ 接线缺陷/必崩** | `http_security.py` 与 `PersistentHitlLedger` 落地 ⚠️；**但 `WanAdapter` 先调全局 `apply_merged_weights` 再套 `LoRAPatchHook`（前景 2x / 背景 100% 泄漏）**，跨窗 Hook 尺寸必崩，用非零均值 `uint8` 像素做潜空间方差融合闪白，`warm_start` 错切片 `[94:97]`；`::ffff:169.254.169.254` 绕过 SSRF | `PR-139` |
| **`PR-117`** | `#117` (`fd0eba6`) | ✅ | ⚠️ | ❌ | **❌ CUDA 必崩** | 流式 `FeatureStore` 落地 ✅；**但 `StatelessLoRATrainer` 在 CUDA 下把 `depth`/`normal` 建在 CPU 必崩**、`T != tau` 必崩、用潜速度前 3 维对齐 3D 法线、`_dpo_step` 仍缺 `t`；Gate 2w 误拒合法对数深度 | `PR-140` |
| **`PR-118`** | `#118` (`139bff0`) | ✅ | ❌ | ❌ | **❌ 大部未落地** | CCW 256 项 MC 表与外 pad 落地 ✅；但 TSDF 未按 `weight > 0` 门控仍出双壁空腔、Visual Hull 漏雕视锥角、`_validate_cameras` 原点 `90°` 假通过、`quad_remesh` 外推 T-junction、`delight.py` 全套旧缺陷仍在、`stages/recon.py` `np.save` 序列化 `DelightResult` 物件 | `PR-141`, `PR-142` |
| **`PR-119`** | `#119` (`08c0e73`) | ✅ | ⚠️ | ❌ | **❌ 门禁必败** | 移除 $\alpha^2$ 与 `TiledBypassExecutor` 末块修复 ✅；**但全旁路仍报 `PSNR=0.0` 挂 `strict=True`**、快速 EMA 仍降 `w_curr=0.30`、未传 `motion_vectors`、`stages/refine` CLI JSON 崩溃且零写入 `PersistentHitlLedger` | `PR-143` |
| **`PR-120`** | `#120` (`43bf050`) | ✅ | ✅ | ⚠️ | **⚠️ 部分闭合** | Quad 邻边环水密、扇形三角化与 HDR 禁除 `255` 真实修复 ✅；但 `profiles.py` 别名产生双倍违规、缺 `histogram_intersection` 方向、`lpips` 平坦噪声放大、`validate_output_non_degenerate` 放行黑白频闪、`watertight_pct` 跳过顶点检查 | `PR-144` |
| **`PR-121`** | `#121` (`5127dd5`) | ✅ | ✅ | ⚠️ | **⚠️ 部分闭合** | 分步编译与 Python $O(d)$ 伞环落地 ✅；但两处 `.bazelrc` 仍缺 `build:opt --compilation_mode=opt`、`build_native.py` 仍硬编码 `timeout=15s`、C++ `@aether//geo` 伞环仍为 $O(d^2)$ | `PR-133`, `PR-141` |
| **`PR-122`** | `#122` (`2140fb1`) | ✅ | ⚠️ | ❌ | **❌ 内存安全 P0** | `--shm-file` 后端接口在 `InspectorShmServer` 落地 ⚠️；**但 `loadGlbMesh` `uint32_t idxCount * 4` 溢出绕回仍可越界读 4 GiB 堆内存**、`main.cc` 未解析 `--shm-file`、多 GLB 互相覆盖、零真 EXR 渲染 | `PR-136`, `PR-146` |
| **`PR-123`** | `#123` (`491c950`) | ✅ | ❌ | ❌ | **❌ 跨端断裂** | Electron 安全静态断言与 Node 23 项通过（Linux）✅；**但 `native-process.js` 写 `/tmp/synapse-shm-*` 而 `ShmFrameReader` 读 `/tmp/inspector-frame-*` 且 `main.cc` 忽略 `--shm-file`**，macOS/Win 100% 黑屏；共享 `_pixelBuf` 存撕裂竞态；3/4 Tab 仍为空壳 | `PR-136`, `PR-146` |
| **`PR-124`** | `#124` (`7f5d39f`) | ✅ | ⚠️ | ❌ | **❌ 3 项回归必崩** | 移除 `bench_runner` 假常数与 `PromotionGate` 防绕过 ✅；**但引入 3 项 H8 必崩回归**: `cli/_rsi.py:269` `UnboundLocalError`、`bench_runner` `applicable_keys` 未收敛致挂 Profile 100% 失败、`model_evaluator` 无参考必退 `40` + `normalize()` 毁尺度；22/30 场景仍丢参数 | `PR-135`, `PR-144`, `PR-145` |
| **`PR-125`** | `#125` (`c4ffeb2`) | ✅ | ⚠️ | ❌ | **❌ 实跑崩溃/回归** | `assets_manifest` 归档与文件哈希分离 ✅；**但 `prepare_datasets.py` 裸调 `bazel build` 实跑退 `255` 且仍伪造 `caelum-core`/`spp=256`**；**H8 回归**: `colab_runner --mode=real-gpu` E1 缺后端、E4 `inject_lora(Linear, ["weight"])` 匹配 0 层必崩，失败覆写 `{}` | `PR-147` |
| **`PR-126`** | `#126` (`8599cd5`) | ⚠️ | ❌ | ❌ | **❌ SLA/真值未达** | `test_metrics:282` `monkeypatch` 落地 ✅；**但 `test_roundtrip_reconstruction` 仍传全黑帧 + 真值 `initial_mesh` 绕过重建**，且单进程快测 `38.60s`（限 `25s`）、慢测 `90.75s`（限 `45s`）均大幅超时 | `PR-148`, `PR-150` |
| **`PR-127`** | `#127` (`485f573`) | ✅ | ✅ | ⚠️ | **⚠️ 部分校准** | Skills 7ch 语义与 `AGENTS.md` `/var/cache/bazel` 对齐 ✅；但 **`README.md` 状态表完全遗漏更新**（停在 `待 PR-113..124`）、`io_spec.md` 仍存 6 处幻觉、`DESIGN_REVIEW_AND_ALIGNMENT` 仍写旧 `tanh` | `PR-147`, `PR-151` |
| **`PR-128`** | `#128` (`f2dbde1`) | ✅ | ❌ | ❌ | **❌ 恒等假测试** | 11 项通过 ✅；但 Contract 1 复制错式 `p02`、Contract 3 未运行 `JobBundleWorker`、Contract 4 纯 Python 自 pack/unpack，未验证真实跨仓端到端 | `PR-148` |
| **`PR-129`** | `#129`/`#131` (`30ce3a5`, `df93ac7`) | ⚠️ | ❌ | ❌ | **❌ 过早封板** | `209`-file `mypy` 与 `1,156` 链接通过；但 Gate 1a/1b/2/8/9/10/12 共 7 项未真实过关即封板 `v3.6.0` | `PR-152` |

---

## 4. 全量分级实证缺陷台账（`H9-P0` / `H9-P1` / `H9-P2` / `H9-P3` — 共 58 项）

> 全量 **58 项** 实证缺陷均在 `HEAD = df93ac7` 上逐行核验源码或 `ubuntu24` 实测输出，按 **`18 P0 + 24 P1 + 12 P2 + 4 P3`** 精确映射至 **Horizon 9 (`M33–M36`, `PR-133 ~ PR-152`)**。

### 4.1 P0 阻断与回归级缺陷（`H9-P0-01 ~ H9-P0-18`）

#### `[H9-P0-01]` `.github/workflows/ci.yml` 至今不存在，且 `.github/workflows/ci.yml` 含仓内嵌套 `./aether` 必败等 5 处缺陷 (`PR-133`)
- **位置**: `.github/workflows/ci.yml:1-150`（缺失 `.github/workflows/`）
- **根因**: (1) GitHub Actions 仅执行 `.github/workflows/*.yml`，仓库无 `.github/` 目录致 `gh run list` 为空；(2) `python-fast` 先将 `synapse` 检出至 `.`，又将 `dervgen/aether` 检出至 `path: aether`（仓内嵌套目录），导致探测 `../aether` 的 `wire_guard.py --require-aether` (`L63`) 与 `build_native.py` 确定性失败；(3) `OUTPUT=$(mypy ...)` (`L54`) 在 `bash -e` 下非零即退出，跳过 `echo "$OUTPUT"` 吞没全部诊断；(4) 未安装 `pybind11` / `libopenexr-dev` / `libgl1` 且 `continue-on-error: true` 掩盖编译失败；(5) `electron-inspector` (`L148`) 未预先构建 `//:data-inspector-native`，必败于 Section 8；跨仓 `dervgen/aether` 未配只读凭证。

#### `[H9-P0-02]` `fgpp/frame.py` `p02` 水平主点符号反转仍在，`test_cross_repo_contracts.py` 复制错式恒等自证 (`PR-134, PR-148`)
- **位置**: [python/synapse/fgpp/frame.py](../../../python/synapse/fgpp/frame.py#L80), [frame.py:339](../../../python/synapse/fgpp/frame.py#L339), [tests/test_cross_repo_contracts.py](../../../tests/test_cross_repo_contracts.py#L160)
- **根因**: OpenGL 右手系 `-Z` 前望相机下，光轴点 $(0, 0, -d, 1)^T$ 经 `proj[3, 2] = -1` 产生 $x_{\text{ndc}} = -p_{02}$。`frame.py:80` 至今仍为错号 `(2.0 * self.cx - w) / w`，将 $(0,0,-d)$ 投影至镜像像素 `sx = W - cx`；`frame.py:339` 用同一错号逆解，且 `test_cross_repo_contracts.py:160` 未做 3D 点投影而是直接断言 `expected_p02 = (2.0 * cx - width) / width` 完成自我循环证明。

#### `[H9-P0-03]` `run_worker_cli` 从未接通 RFC 0039 `JobBundle`，且转换器与 Go `render-farm` 真实结构体不兼容 (`PR-135, PR-148`)
- **位置**: [python/synapse/pipeline/bundle_worker.py](../../../python/synapse/pipeline/bundle_worker.py#L120-L250), [bundle_worker.py:1052](../../../python/synapse/pipeline/bundle_worker.py#L1052)；对照 `../render-farm/internal/domain/models.go#L426-L534`
- **根因**: (1) `run_worker_cli` (`L1052`) 直接调 `BundleManifest(**raw)`（`extra="forbid"`），**从未调用 `from_rfc0039_job_bundle`**，令所有真实 `render-farm`/`grand-studio` 载荷在 CLI 入口即以 `ExitCode 10` 拒收；(2) 即便手动调用 `from_rfc0039_job_bundle`，Go `AssetManifestEntry` 键名为 `logical_path`（而非 `path`）、`OutputSink` 为嵌套 `{format, channels, write_capability: {url}}`（非扁平 `dict[str, str]`）、`SceneSpec` 为结构化对象、`Payload` 可能为 JSON `null`（`raw.get("payload", {})` 返 `None` 触发 `AttributeError`）且 `EngineSpec` 被丢弃；(3) `input_sha256s` 错在 `payload` 而非 `scene_spec` 寻址且静默放行缺失文件。

#### `[H9-P0-04]` **[Horizon-8 回归]** `synapse rsi gate-promote --baseline-report ... --candidate-report ...` 确定性 `UnboundLocalError` 崩溃 (`PR-135`)
- **位置**: [python/synapse/cli/_rsi.py](../../../python/synapse/cli/_rsi.py#L169-L271)
- **根因**: PR #125 在 `else:` 分支（`L233-234`）定义 `base_scenarios_raw` / `cand_scenarios_raw`，却在分支外无条件执行 `if not base_scenarios_raw and not cand_scenarios_raw:`（`L269`），导致使用规范标志 `--baseline-report` + `--candidate-report` 时 100% 触发 `UnboundLocalError`；此外 `:267` 硬编码 `raise_on_reject=True` 无视 `--strict` 标志，`:421` `--scenarios` 为死参数。

#### `[H9-P0-05]` **[Horizon-8 回归]** `BenchPipelineRunner` 挂任意 `--profile` 100% 门禁必败 (`PR-144`)
- **位置**: [python/synapse/pipeline/bench_runner.py](../../../python/synapse/pipeline/bench_runner.py#L361-L392), [bench_runner.py:593-601](../../../python/synapse/pipeline/bench_runner.py#L593-L601), [bench_runner.py:784-825](../../../python/synapse/pipeline/bench_runner.py#L784-L825)
- **根因**: PR #125 删去 `_compute_metrics` 中的假指标常数（并在 `L823` 注释需同步收缩 `applicable_keys`），却未更新 `L362-385` 的 `applicable_keys`——仍无条件要求未计算的 `light_dir_error_deg`、所有 T1–T3 场景的 `foot_sliding_cm`/`bone_length_drift_pct` 及 24 场景 `expected.chamfer_mm`（`evaluate_gate` 判为 `NaN` 违规），同时用未焊接的输入 `ProceduralMesh` 算出 `watertight_pct=0%`；`evaluate_bundle`（`L595`）仅算 4 项指标却要求全量 16+ Profile 键，严格模式 100% 必败；`L754` 未将 AP1 `ACEScg` 转为 Linear `sRGB` 即编码 sRGB。

#### `[H9-P0-06]` **[Horizon-8 回归]** `inspect/model_evaluator.py` 无参考模式 100% 退 `40` 且 `normalize()` 毁掉世界尺度 (`PR-145`)
- **位置**: [python/synapse/inspect/model_evaluator.py](../../../python/synapse/inspect/model_evaluator.py#L157-L181), [model_evaluator.py:278-291](../../../python/synapse/inspect/model_evaluator.py#L278-L291), [model_evaluator.py:455-462](../../../python/synapse/inspect/model_evaluator.py#L455-L462)
- **根因**: `L456` 注释承诺不传 `--reference-dir` 时仅评自身拓扑不因缺 `chamfer_mm` 失败，但 `L167` 硬编码 `base_required = {"is_manifold", "chamfer_mm"}` 且 `L461` 无区别传入导致全数退 `40`；`_default_load_mesh` (`L280,291`) 无条件调用 `mesh.normalize()` 归零平移与缩放，使 10 倍尺度和位置偏差 `chamfer_mm=0.0`，单颗离群噪点却毁掉全模型评分。

#### `[H9-P0-07]` **[Horizon-8 回归]** `tools/data-inspector` 三重 `--shm-file` 断链致 macOS/Windows 视口 100% 黑屏 (`PR-136`)
- **位置**: [tools/data-inspector/electron/native-process.js](../../../tools/data-inspector/electron/native-process.js#L116,L127,L209-L213), [tools/data-inspector/native/main.cc](../../../tools/data-inspector/native/main.cc#L69-L81), [tools/data-inspector/electron/shm-reader.js](../../../tools/data-inspector/electron/shm-reader.js#L70-L107)
- **根因**: `native-process.js:210` 传 `--shm-file=${os.tmpdir()}/synapse-shm-${pid}`，但 `native/main.cc:69-81` **从未解析 `--shm-file` 参数**（恒走内核 `shm_open`/Named Mapping），且 `native-process.js:127` 又将 `/inspector-frame-${pid}` 传给 `ShmFrameReader` 探测不同文件名，导致 Linux 外全部失效；字面 Gate 8 未先建 `//:data-inspector-native` 亦在 Linux 失败。

#### `[H9-P0-08]` `loadGlbMesh` 存在 32 位 `uint32_t idxCount * 4` 乘法溢出与 64 位偏移绕回堆越界读 (`PR-136`)
- **位置**: [tools/data-inspector/native/inspector_app.cc](../../../tools/data-inspector/native/inspector_app.cc#L385-L408), [inspector_app.cc:439-486](../../../tools/data-inspector/native/inspector_app.cc#L439-L486)
- **根因**: `idxCount` 为 `uint32_t`（由 `int64_t rawIdxCount` 直接截断而来），`L474` 执行 `idxCount * 4` 与 `L461` `idxCount * 2` 先在 32 位无符号下溢出绕回再提升为 `size_t`：畸形 `.glb` 设 `count = 0x40000001` 时 `idxCount * 4 == 4 <= binChunk.size()` 通过检查，随后 `size_t i` 循环读取 **4 GiB 堆外内存**；`posViewOffset + posAccOffset` 与 `vi * byteStride` 亦存 64 位无符号加法绕回隐患。

#### `[H9-P0-09]` `WanAdapter.generate()` 双重 LoRA 注入、多窗 Hook 维度崩溃及对 `uint8` 像素套用零均值潜空间公式闪烁 (`PR-139`)
- **位置**: [python/synapse/adapters/wan.py](../../../python/synapse/adapters/wan.py#L740-L856), [python/synapse/adapters/lora/patcher.py](../../../python/synapse/adapters/lora/patcher.py#L77-L161), [python/synapse/adapters/scheduler.py](../../../python/synapse/adapters/scheduler.py#L515-L595)
- **根因**: (1) `wan.py:746-769` 丢弃 `build_spatial_lora_masks` 并调无掩膜全局 `apply_merged_weights`，紧接着又挂载 `manage_spatial_hooks`，使目标实体叠加 **2x LoRA** 且背景承受 **100% 风格污染**；(2) `manage_spatial_hooks` 挂在滑窗外（全长 `T_total`），与单窗 `97f` 序列长度冲突崩溃，且 `patcher.py:120` 奇数 `H//8`（如 `540p`）未减半、`L77` 未跳过 Diffusers `attn2.to_{k,v}`；(3) `wan.py:846-856` 将解码后非零均值 `[0, 255]` `uint8` 像素直接送入零均值潜空间保方差融合 `fuse_overlapping_latents`，在平坦灰墙（`rho=0`）中点放大 `1/sqrt(0.5) = 1.41x` 造成周期性白屏闪爆。

#### `[H9-P0-10]` `StatelessLoRATrainer` 在 CUDA 上 `cpu/cuda` 设备冲突必崩、`T != tau` 必崩且用潜速度对齐 3D 法线 (`PR-140`)
- **位置**: [python/synapse/training/trainer.py](../../../python/synapse/training/trainer.py#L418-L420), [trainer.py:556-650](../../../python/synapse/training/trainer.py#L556-L650), [python/synapse/training/losses.py](../../../python/synapse/training/losses.py#L92-L115), [losses.py:197-237](../../../python/synapse/training/losses.py#L197-L237)
- **根因**: `train_step` 在调用 `_sft_step` 前移至 CUDA，但 `_extract_auxiliary_from_fgpp` 转回 NumPy 在 **CPU** 新建 `depth`/`normal`，触发 `cuda:0 vs cpu` `RuntimeError`；同时仅空间下采样 `::8` 未做 `T -> tau=(T-1)//4+1` 时域压缩（`5D` 与 `4D` 维度错位必崩，且 `5D` Batch 时 `ndim==4` 静默跳过）；`L584` 取 16ch 潜速度前 3 维与欧氏世界法线算余弦相似度在物理空间无意义。

#### `[H9-P0-11]` **[Horizon-8 回归]** `scripts/colab_runner.py --mode=real-gpu` E1/E4 双重必崩且失败覆写清空已跑指标 (`PR-147`)
- **位置**: [scripts/colab_runner.py](../../../scripts/colab_runner.py#L184-L200), [colab_runner.py:256-278](../../../scripts/colab_runner.py#L256-L278), [colab_runner.py:322-353](../../../scripts/colab_runner.py#L322-L353), [colab_runner.py:597-608](../../../scripts/colab_runner.py#L597-L608)
- **根因**: `_verify_e1_real_gpu` 实例化无后端 `WanAdapter()` (`_backend=None`) 必退 `30`；`_verify_e4_real_gpu` 对裸 `nn.Linear(64, 64)` 匹配 `target_patterns=["weight"]`（`named_modules()` 仅含 `""`，匹配 `0` 层必退 `50`），且后续构 `LoRAManifest(category="style", target_modules=["spatial_attn.to_q"])` 违反风格白名单必抛 `ValidationError`；`except SynapseError` (`L597`) 传入 `{}` 清空已完成阶段指标并硬编码写入 `exit_code=1`。

#### `[H9-P0-12]` `scripts/prepare_datasets.py` 在 `ubuntu24` 裸调 `bazel build` 崩溃 (`exit 255`)、禁用 Lavapipe 且仍伪造 `caelum-core` (`PR-147`)
- **位置**: [scripts/prepare_datasets.py](../../../scripts/prepare_datasets.py#L37-L95), [prepare_datasets.py:119,139](../../../scripts/prepare_datasets.py#L119), [prepare_datasets.py:243-431](../../../scripts/prepare_datasets.py#L243-L431)
- **根因**: `L119` 在 `try` 块外裸跑 `bazel build //course:102-pbr-direct` 实跑当场崩溃退出 `255`；`L139` `VK_ICD_FILENAMES=""` 主动关闭系统软件 Vulkan ICD；`_export_software_rasterizer_fallback` 伪称使用 `SoftwareRasterizer` 实写全零 `direct_radiance`/`cryptomatte_id`；`_export_caelum_track_a_17f` 仍忽略 `caelum_repo` 伪造 `"generator": "caelum-core", "spp": 256`。

#### `[H9-P0-13]` `border_ring_lab_match` 内环 `1->0` 硬接缝未修 + `viewer/server.py` 先 `resolve()` 致符号链接防御失效 (`PR-137`)
- **位置**: [python/synapse/composite/blend.py](../../../python/synapse/composite/blend.py#L297-L317), [python/synapse/viewer/server.py](../../../python/synapse/viewer/server.py#L52-L64), [server.py:94-109](../../../python/synapse/viewer/server.py#L94-L109)
- **根因**: `blend.py` 对双边均为 `False` 的环形带 `inner_band` 做形态学剥离，令 `inner_band` 最内圈直接出现 `1.0 -> 0.0` 跳变硬缝且多跑 16 次腐蚀；`viewer/server.py:52` 在 `L60` `is_symlink()` 检查前先行 `.resolve()` 使目录符号链接守卫完全成死代码，且未拦截 `frames/`/`visualizations/` 内指向敏感文件的子符号链接。

#### `[H9-P0-14]` `stages/recon.py` 序列化 `DelightResult` 物件损坏 `.npy`；红线 #3 全仓生产零调用；`cli/_recon.py` 丢弃重建网格 (`PR-142`)
- **位置**: [python/synapse/pipeline/stages/recon.py](../../../python/synapse/pipeline/stages/recon.py#L111,L291), [python/synapse/cli/_recon.py](../../../python/synapse/cli/_recon.py#L44-L73), [python/synapse/bridge/vg_bridge.py](../../../python/synapse/bridge/vg_bridge.py#L119-L170)
- **根因**: `stages/recon.py:111,291` 将 `DelightResult` dataclass 直接传给 `np.save`，写出 `dtype=object` Pickle 数组导致下游 `allow_pickle=False` 读取崩溃；`validate_whitebox_coarseness_floor`、`enforce_distillation_geometry_gate` 与 `evaluate_hitl_circuit_breaker(..., domain="asset_3d")` 在 `python/synapse/` 生产路径调用数为 **`0`**；`cli/_recon.py` 仅写 `camera_trajectory.json` 丢弃 `result.mesh` 与 PBR 贴图。

#### `[H9-P0-15]` TSDF 双壁空腔、Visual Hull 漏雕视锥角、`_validate_cameras` 原点 `90°` 假通过与 `quad_remesh` 一环 T-junction 裂缝 (`PR-142`)
- **位置**: [python/synapse/recon/multiview.py](../../../python/synapse/recon/multiview.py#L415-L442), [multiview.py:588-613](../../../python/synapse/recon/multiview.py#L588-L613), [multiview.py:691-695](../../../python/synapse/recon/multiview.py#L691-L695), [python/synapse/recon/geometry.py](../../../python/synapse/recon/geometry.py#L360-L484)
- **根因**: TSDF 初值为 `+1.0` 且 `valid` 截断 `sdf > -sdf_trunc`，内部深处 `weight == 0` 体素保持 `+1.0` 抽出内外双壳；Visual Hull `carve` 未置 `~in_bounds`；`_validate_cameras` 对原点 `[0,0,0]` 相机算 `acos(0) = 90°` 放行零基线输入；`quad_remesh` 仅细分接触未配对三角形的四边面，把 T-junction 裂缝推向外一环仍判非水密且令 `target_faces` 面数翻倍并丢弃 UV。

#### `[H9-P0-16]` `recon/delight.py` 仍保留 `(rad/E)*E` 自抵消、屏幕空间重影、相机系 SH 抵消与 Quad 丢半 (`PR-142`)
- **位置**: [python/synapse/recon/delight.py](../../../python/synapse/recon/delight.py#L414-L436), [delight.py:598-671](../../../python/synapse/recon/delight.py#L598-L671), [delight.py:719-834](../../../python/synapse/recon/delight.py#L719-L834)
- **根因**: `relit = (rad / E) * E` 恒等零残差、5 点稀疏投影覆盖极低、旋转视角直接在屏幕 `(H,W)` 空间与相机坐标系平均反照率/SH 引起重影与方向光互消，以及 `f_arr[:, :3]` 截断丢半 Quad 面片、未经反伽马对 `uint8` sRGB 算线性亮度。

#### `[H9-P0-17]` `refine` 全旁路健康视频得 `PSNR=0` 必败、`ScalpelStitcher` 高速 EMA 反转造重影、`stages/refine` JSON 崩溃且不记录 HITL 账本 (`PR-143`)
- **位置**: [python/synapse/refine/stitcher.py](../../../python/synapse/refine/stitcher.py#L260-L266), [stitcher.py:665-675](../../../python/synapse/refine/stitcher.py#L665-L675), [python/synapse/refine/pipeline.py](../../../python/synapse/refine/pipeline.py#L277-L356), [python/synapse/pipeline/stages/refine.py](../../../python/synapse/pipeline/stages/refine.py#L55-L142)
- **根因**: 全旁路 `measured==0` 返回 `mean_border_psnr_db=0.0 < 38.0` 挂 `strict=True`；`pipeline.py` 未传 `motion_vectors` 且拿未 Warp 帧差充当 `temporal_warp_error`；`stitcher.py` 高速运动时将当前帧权重降至 `0.30`（70% 历史重影）且全图均分静默光流；`stages/refine.py` 迭代 flat CLI ROI dict 触发 `TypeError` 且从不调 `ledger.record_action()` 使 15m 熔断跨作业永不累加。

#### `[H9-P0-18]` `Synapse-Bench` 22/30 场景参数静默失效且 `test_roundtrip_reconstruction.py` 喂全黑帧 + 真值 `initial_mesh` 假重构 (`PR-144, PR-148`)
- **位置**: [python/synapse/render_core/camera.py](../../../python/synapse/render_core/camera.py#L241-L333), [python/synapse/pipeline/bench_runner.py](../../../python/synapse/pipeline/bench_runner.py#L212-L305), [tests/test_roundtrip_reconstruction.py](../../../tests/test_roundtrip_reconstruction.py#L263-L334)
- **根因**: 8 个 `fixed` 与 `pan` 场景球坐标在 `**kwargs` 被吞，7 个 T3/T5 多物体只渲单个体，T2 `animation`（如 `idle`/`walk`）与 `Conditions.resolve()` 未接入；`test_roundtrip_reconstruction.py` 用 `np.zeros((12,64,64,3))` + GT `initial_mesh` 调用 `ReconPipeline.run()` 完全跳过多视角重建（`_make_sphere_depth_maps` 为死代码）。

---

### 4.2 P1 严重功能 / 安全 / 数学 / 性能缺陷（`H9-P1-01 ~ H9-P1-24`）

| 审计编号 | 精确位置 | 根因与后果摘要 | 目标 PR |
| :--- | :--- | :--- | :---: |
| **`H9-P1-01`** | [tools/import_dag_guard.py](../../../tools/import_dag_guard.py#L81-L93,L435-L446); [tools/wire_guard.py](../../../tools/wire_guard.py#L247-L413); [manifest.py](../../../python/synapse/adapters/lora/manifest.py#L112-L122) | `import_dag_guard` 丢弃 `pkg_files`（未注册孤包逃逸）且放行 `import synapse; synapse.pipeline`；`wire_guard` 未调 `model_json_schema()`，漏检 `LoRAManifest.adapter_path` 放行 `../`/绝对路径与 `CameraPose.rotation` 允许长度 `5..8` | `PR-134` |
| **`H9-P1-02`** | [fgpp/schema.py](../../../python/synapse/fgpp/schema.py#L135-L146); [fgpp/frame.py](../../../python/synapse/fgpp/frame.py#L315-L370); [fgpp/io.py](../../../python/synapse/fgpp/io.py#L139-L286) | `linear_to_log_normalized_depth` 对 `NaN`/`-Inf` 返 `1.0`（C++ 返 `0.0`）；含 `"camera"` 时跳过 `det(R)` 正交校验且未解 Vulkan `Infinite Reversed-Z`；`_read_exr_frame` 不校验 `dataWindow` 与 `FLOAT`/`HALF` `PixelType`，`export_fgpp` 默许写出残缺通道 EXR | `PR-134` |
| **`H9-P1-03`** | [generators/dataset.py](../../../python/synapse/generators/dataset.py#L231-L332); [generators/rasterizer.py](../../../python/synapse/generators/rasterizer.py#L104-L106) | `DatasetGenerator` 将 Linear sRGB 渲染图再套 `acescg_to_linear_srgb` 偏色、缺传 `base_color`/`roughness_metallic` 写出 10ch 非标 EXR、`except (ImportError, OSError)` 漏接 `BackendUnavailableError`；`generators/rasterizer` 漏 sRGB 伽马 | `PR-138` |
| **`H9-P1-04`** | [composite/blend.py](../../../python/synapse/composite/blend.py#L714-L738); [composite/detector.py](../../../python/synapse/composite/detector.py#L240-L390) | `_cryptomatte_boundary_guide` 2px 脉冲令引导滤波跨轮廓糊成一团；`_detect_silhouette_drift` 在有纹理图上报 100% 假阳性且漏检消失物体；`_detect_ground_violation` 将垂直 `fov_degrees` 当水平 FoV 算出 `1.78x` 焦距且跨全帧合并离散帧报错 | `PR-137` |
| **`H9-P1-05`** | [style/extractor.py](../../../python/synapse/style/extractor.py#L55-L77,L188-L194,L253-L260,L308-L403) | 全模块从不转 ACEScg（违背 `working_colorspace="ACEScg"`）且未估 `key_light_rgb`/暗角；IRLS 对背光面整行乘 `0.01` 弃用环境光 $a_0$ 观测；`_frechet_mean_s2` 在 $\theta\approx\pi$ 出现 `31,416 rad` 爆炸；`_estimate_cct_mccamy` 分母 `0.1858-y` 可零除，`exposure_ev` 未钳位 `[-20, 20]` | `PR-137` |
| **`H9-P1-06`** | [render_core/camera.py](../../../python/synapse/render_core/camera.py#L142-L333); [render_core/mesh.py](../../../python/synapse/render_core/mesh.py#L125-L158); [bench/registry.py](../../../python/synapse/bench/registry.py#L177-L243) | 全部 6 种相机轨迹生成器吞掉 `z_near`/`z_far`/`aperture_fstop`/`shutter_angle_deg`；`load_obj` 破坏 UV 接缝且遇无索引 `vn` 时产出全零法线；`cone` 基元坍缩造 32+ 退化面，远程资产下载允许 `file://` 无超时且 SHA256 失败不删残包 | `PR-138`, `PR-144` |
| **`H9-P1-07`** | [adapters/wan.py](../../../python/synapse/adapters/wan.py#L564-L569,L685,L705-L721,L793-L838); [adapters/base.py](../../../python/synapse/adapters/base.py#L155-L169) | 滑窗 1 误取上窗末 3 帧 `[94:97]`（超前 30 帧）且无视切镜清空；标准 `VideoModelAdapter.generate()` 缺 `fgpp_frames` 绕过突变检测；`gamma_flow_bias` 无上游光流为死代码；`LoRAManifest` `fullmatch` 拒收 `blocks.0.*` / `attn1.*` | `PR-139` |
| **`H9-P1-08`** | [adapters/cloud.py](../../../python/synapse/adapters/cloud.py#L94-L108,L555-L583,L766-L784); [bridge/http_security.py](../../../python/synapse/bridge/http_security.py#L15-L105) | `CloudAdapters` 丢弃 `conditions.text_prompt`、`camera_trajectory`、`Idempotency-Key` 并硬截 HDR `>1.0`；`validate_safe_url` 被 `http://[::ffff:169.254.169.254]/`、整数 IP、DNS 重绑定与云元数据地址绕过 SSRF | `PR-139` |
| **`H9-P1-09`** | [training/trainer.py](../../../python/synapse/training/trainer.py#L354-L366,L509-L539,L668); [training/feature_store.py](../../../python/synapse/training/feature_store.py#L320,L381-L424); [training/packager.py](../../../python/synapse/training/packager.py#L224-L306) | `_dpo_step` 漏设 `batch["t"] = t`；5D `[B,1,1,1,1]` `t` 与缺 `encoder_hidden_states` 不兼容 Diffusers 且 `_call_module` 吞 `TypeError`；`flush_gradients` 漏 `GradScaler`；Gate 2w 将无量纲 `[0,1]` 对数深度当米制误拒有效样本；`add_sample` 内存积压 `1.5GB/clip`；`packager` 先落盘后门禁 | `PR-140` |
| **`H9-P1-10`** | [pipeline/bundle_worker.py](../../../python/synapse/pipeline/bundle_worker.py#L282-L292,L376,L470-L505,L858-L911); [pipeline/stages/{inspect,recon,training}.py](../../../python/synapse/pipeline/stages/inspect.py#L44) | `is_symlink()` 仅查末级放行父目录符号链接穿越；`Path("")` 变 CWD；`ResultManifest` 丢弃异常原文；OOM 回退把短序列扩到 `49` 帧反增显存；`asset_inspect_3d` 静默用 Mock 返 `0`、`lora_preencode` 用随机高斯写 0 分片也返 `0`、`kinematic_distill` 无动作导出 | `PR-135`, `PR-140`, `PR-142` |
| **`H9-P1-11`** | [cli/common.py](../../../python/synapse/cli/common.py#L61-L105); [cli/_inspect.py](../../../python/synapse/cli/_inspect.py#L51,L349); [cli/_recon.py](../../../python/synapse/cli/_recon.py#L33,L147,L294); [cli/_refine.py](../../../python/synapse/cli/_refine.py#L92,L123); [cli/_generate.py](../../../python/synapse/cli/_generate.py#L95-L225); [cli/_bench.py](../../../python/synapse/cli/_bench.py#L102-L114) | `synapse --json inspect/recon` 被子解析器覆写污染 stdout；缺包报 `10`（应 `30`）、缺参报 `20`（应 `10`）、`main_data_gen`/`main_viewer` 用裸 `argparse` 退 `1/2`；`closed-loop` 伪造屏幕 UV；`refine` BBox 越界不对齐；`bench --export-bundle` 读不存在的 `sc.asset.path` 必崩 | `PR-135`, `PR-142`, `PR-143`, `PR-144` |
| **`H9-P1-12`** | [recon/geometry.py](../../../python/synapse/recon/geometry.py#L759-L789); `../aether/geo/src/mesh_checks.cc:341-375`; [csrc/geometry/.bazelrc](../../../csrc/geometry/.bazelrc#L14); [tools/data-inspector/native/.bazelrc](../../../tools/data-inspector/native/.bazelrc#L13); [scripts/build_native.py](../../../scripts/build_native.py#L62,L662) | Python `check_self_intersections` `_MAX_CELLS_PER_FACE=512` 漏检穿过大三角形的小面片；C++ `analyzeManifold` 仍为 $O(d^2)$；两处 `.bazelrc` 缺 `build:opt --compilation_mode=opt`；`build_native.py` 硬编码 `15s` 超时、传重 `--config=<os>` 并编译冗余 `mesh_forensics.cc` | `PR-133`, `PR-141` |
| **`H9-P1-13`** | [recon/tracking.py](../../../python/synapse/recon/tracking.py#L214-L253,L508-L728); [recon/pipeline.py](../../../python/synapse/recon/pipeline.py#L80-L130); [recon/kinematics.py](../../../python/synapse/recon/kinematics.py#L224-L243) | `tracking` 丢 `visibility`/无三角化尺度/用 Frame 0 `P1` 算 `reprojection_error_px`/LK `patch_p` 整数漂移；3 模块分裂 `fx=w`/`1050`/`max(h,w)` 且 OpenCV `+Z` 与 FGPP `-Z` 混用；`kinematics` 在 IK 钳位前算滑移且 `ReconPipeline` 伪造 15 帧方波步态 | `PR-142` |
| **`H9-P1-14`** | [recon/loop/forward_guide.py](../../../python/synapse/recon/loop/forward_guide.py#L104-L121); [closed_loop_runner.py](../../../python/synapse/recon/loop/closed_loop_runner.py#L202-L398); [texture_extractor.py](../../../python/synapse/recon/loop/texture_extractor.py#L121-L286); [mitsuba_optimizer.py](../../../python/synapse/recon/loop/mitsuba_optimizer.py#L147-L350) | `ForwardGuide` $\kappa=1.0$ 为生成后死代码被截去；`ClosedLoopRunner` `[0,0,w,h]` 致外环为空；`bake_uv_normal_map` 未剔除 `[0,0,0]` 背景且用与 UV 无关的 Frisvad TBN；`AnalyticUV` 钳 `ndotl=0.01` 放大百倍并跨 UV 岛渗 `0.5` 灰边；`MitsubaInverseOptimizer` 100% 为 `50` 桩 | `PR-142` |
| **`H9-P1-15`** | [refine/fusion.py](../../../python/synapse/refine/fusion.py#L93-L248); [refine/stitcher.py](../../../python/synapse/refine/stitcher.py#L197-L235,L371-L390,L538); [refine/pipeline.py](../../../python/synapse/refine/pipeline.py#L261-L312); [refine/tiled_executor.py](../../../python/synapse/refine/tiled_executor.py#L110) | `CASCADE` 丢 50% 高频、`UNCERTAINTY` $K=2$ 退化、`POISSON` Neumann vs `reflect` 失配；`Stitcher` 用内环惩罚合法修复、分通道覆盖破坏 HDR 色相、`BORDER_CONSTANT=0` 造 1px 黑晕；`LocalRefinementPipeline` 错用未平滑框裁切、按索引匹配破坏多手/脸跟踪、单候选直通绕过融合且 `tile_and_execute` 零调用 | `PR-143` |
| **`H9-P1-16`** | [refine/hand.py](../../../python/synapse/refine/hand.py#L80,L267,L343-L358); [refine/face.py](../../../python/synapse/refine/face.py#L211,L299-L436) | `strict=True` 下无真权重仍静默跑 `heuristic`；`hand.py` 用 RGB 通道均值替代空间高斯滤波且顶部计数指法不抗旋转；`face.py` 无负偏航角、拿 ArcFace 切片给全脸染色、双眼混算质心漏检斜视 | `PR-143` |
| **`H9-P1-17`** | [metrics/profiles.py](../../../python/synapse/metrics/profiles.py#L31-L69,L234-L285); [metrics/image_quality.py](../../../python/synapse/metrics/image_quality.py#L308-L541); [metrics/roundtrip.py](../../../python/synapse/metrics/roundtrip.py#L287,L452,L613); [metrics/fgpp_fidelity.py](../../../python/synapse/metrics/fgpp_fidelity.py#L25-L188) | 别名导致 `GateViolation` 双重计分；`histogram_intersection` 方向反转；`lpips` 平坦 1-LSB 噪声炸至 `0.30` 且未调 `torch`；`validate_output_non_degenerate` 混算时空方差放行纯黑/纯白频闪；`watertight_pct` 跳过顶点无限性校验；`f_score` 同种子采样失真；`albedo_delta_e00`/边界环遇到 `uint8` 指数爆炸且缺形态/掩膜校验 | `PR-144` |
| **`H9-P1-18`** | [inspect/macro.py](../../../python/synapse/inspect/macro.py#L158-L201); [inspect/align.py](../../../python/synapse/inspect/align.py#L270-L364); [inspect/schemas.py](../../../python/synapse/inspect/schemas.py#L37-L40); [inspect/flywheel_exporter.py](../../../python/synapse/inspect/flywheel_exporter.py#L499-L549); [inspect/jev_critic.py](../../../python/synapse/inspect/jev_critic.py#L378-L390); [pipeline/stages/inspect.py](../../../python/synapse/pipeline/stages/inspect.py#L87-L96) | `macro` 帧内 min-max 归一化放大 1-LSB 噪声为高危缺陷且未屏蔽背景法线；`align` 锚定原点非质心使偏移网格失败且非方图崩溃；`inspect_refine` 无修复算子却标 `waived=True` 退 `0`；`flywheel_exporter` 跨会话错配 DPO 对；`VisualJevCritic` 缺真实 HTTP 传输且置信度混淆概率与熵 | `PR-145` |
| **`H9-P1-19`** | [rsi/dual_verifier.py](../../../python/synapse/rsi/dual_verifier.py#L229-L263); [rsi/pair_builder.py](../../../python/synapse/rsi/pair_builder.py#L59-L95); [rsi/promotion_gate.py](../../../python/synapse/rsi/promotion_gate.py#L189-L280); [rsi/curriculum.py](../../../python/synapse/rsi/curriculum.py#L170); [pipeline/stages/rsi.py](../../../python/synapse/pipeline/stages/rsi.py#L31-L34) | `DualVerifier` 拒收 `bool` `is_manifold_watertight` 却允许省略绕过（`stages/rsi.py` 报错键名）；`PreferencePair` 放行空 URI 且 `from_dpo_record` 将 `source="self"` 洗白成 `"human"`；`PromotionGate` 总分与场景分脱钩且静默去重；`CurriculumManager` 每轮同种子复现相同题库且无闭环连接器 | `PR-145` |
| **`H9-P1-20`** | [tools/data-inspector/electron/shm-reader.js](../../../tools/data-inspector/electron/shm-reader.js#L229-L253); [main.js](../../../tools/data-inspector/electron/main.js#L136-L156,L199-L222); [native-process.js](../../../tools/data-inspector/electron/native-process.js#L210,L301); [native/inspector_shm_server.cc](../../../tools/data-inspector/native/inspector_shm_server.cc#L132); [native/inspector_app.cc](../../../tools/data-inspector/native/inspector_app.cc#L670-L686,L872); [renderer/src/app.js](../../../tools/data-inspector/renderer/src/app.js#L235-L262) | 共享 `_pixelBuf` 在 33ms 节流期间被下一帧写失败撕裂；`/tmp` SHM 文件 `0666` 无 `O_NOFOLLOW` 且退出不清理；Seqlock 持锁跨整帧光栅；多 GLB 源互相覆盖单一 `m_mesh`；`ResizeObserver` 与异步 `export-screenshot` 为假指令 | `PR-136`, `PR-146` |
| **`H9-P1-21`** | [scripts/run_roundtrip_verification.py](../../../scripts/run_roundtrip_verification.py#L161-L234,L382,L478); [scripts/prepare_benchmark_assets.py](../../../scripts/prepare_benchmark_assets.py#L515,L742-L760); [assets/assets_manifest.json](../../../assets/assets_manifest.json#L27-L81) | `run_roundtrip_verification` 喂纯色噪声未投影 GT 网格（3 个渲染函数全死代码）、丢弃受损相机外参并恒退 `0`；`prepare_benchmark_assets` 离线重拉 `vbench_prompts` 必挂、缺 `huggingface_hub` 静默返回空且 `sha256_verified` 全 `false` | `PR-147` |
| **`H9-P1-22`** | [workenv/google-colab/setup_colab.sh](../../../workenv/google-colab/setup_colab.sh#L27,L60); [workenv/google-colab/io_spec.md](../../../workenv/google-colab/io_spec.md#L15-L190); [workenv/google-colab/benchmarks.json](../../../workenv/google-colab/benchmarks.json#L29-L106) | `setup_colab.sh` `libgl1-mesa-glx` 在 Ubuntu 24 必挂且不克隆 `aether` 致 `_c_geometry` 永编不出；`io_spec.md` 仍存 6 处幻觉字段/命令/类名；`benchmarks.json` 与 `assets_manifest.json` 源仓分裂 | `PR-147` |
| **`H9-P1-23`** | [README.md](../../../README.md#L24-L66); [docs/DESIGN_REVIEW_AND_ALIGNMENT.md](../../../docs/DESIGN_REVIEW_AND_ALIGNMENT.md#L3-L294); [docs/AUDIT_AND_ARCHITECTURE.md](../../../docs/AUDIT_AND_ARCHITECTURE.md#L264-L294); [.agents/skills/google-colab-environment](../../../AGENTS.md) | `README.md` 遗漏 `v3.6.0` 更新仍写 `待 PR-113..124`；`DESIGN_REVIEW_AND_ALIGNMENT` 与 `AUDIT_AND_ARCHITECTURE` 仍留 `tanh(L_th=4, max=12)` 软肩、`ViewerServer` HTTP 与旧 CLI `--input/--output-usd`；Colab CPU/GPU 共用主机名 `synapse-colab-gpu` 碰撞 | `PR-151` |
| **`H9-P1-24`** | [tests/test_wan_lora_wiring.py](../../../tests/test_wan_lora_wiring.py#L168-L186); [tests/test_refine_fusion.py](../../../tests/test_refine_fusion.py#L191-L343); [tests/test_metrics.py](../../../tests/test_metrics.py#L308-L321); [tests/conftest.py](../../../tests/conftest.py#L16,L73-L98) | 关键接线测用 `MagicMock` 替真实合并与张量流；`test_refine_fusion` 自写欧氏 RGB 伪冒 $\Delta E_{00}$ 并放行 `2.0x` 能量膨胀；`test_metrics` 吞全部 `Exception`；全仓 `0` 个 `hypothesis` 属性与畸形二进制 Fuzz 测试 | `PR-148`, `PR-149` |

---

### 4.3 P2 架构收敛 / 性能 / 内存瓶颈 & P3 工程卫生（`H9-P2-01 ~ H9-P2-12`, `H9-P3-01 ~ H9-P3-04`）

| 审计编号 | 级别 | 位置与根因分析 | 目标 PR |
| :--- | :---: | :--- | :---: |
| **`H9-P2-01`** | **P2** | **97f 4K 峰值内存爆炸 (`38.6 GB`)**：[style/extractor.py](../../../python/synapse/style/extractor.py#L294,L344) 连续两次整视频转 `float64`；[fgpp/io.py](../../../python/synapse/fgpp/io.py#L292) 冗余 `.copy()`；[render_core/rasterizer.py](../../../python/synapse/render_core/rasterizer.py#L310-L319) 分配 `104 B/px` 且 `color_buf` (`24 B/px`) 为死内存；[metrics/image_quality.py](../../../python/synapse/metrics/image_quality.py#L158-L198) `ssim` 占 `20 GB` `float64` | `PR-138`, `PR-144` |
| **`H9-P2-02`** | **P2** | **稠密 ICP `60 GB` 广播与纯 Python 内循环**：[recon/geometry.py](../../../python/synapse/recon/geometry.py#L273-L276) 构 `(N, M, 3)` 距阵在 5 万顶点下爆内存；`rasterizer.py:329`、`pyramid.py:157`、`morphology.py:184` 纯 Python 循环阻滞吞吐 | `PR-138`, `PR-141` |
| **`H9-P2-03`** | **P2** | **三套重复网格/光栅与色彩算子**：`MeshData` / `ProceduralMesh` / `ModelAsset`、`render_core` / `bench` / `generators` 光栅器、`aces_color` / `extractor` / `radiometric` / `delight` 矩阵各自独立，且 `bench` 弃用垫片零告警 | `PR-138`, `PR-142` |
| **`H9-P2-04`** | **P2** | **`csrc/geometry` 未充分利用 `@aether//geo`**：仅绑定 `mesh_checks.h`（6 函数），同级 `aether/geo` 已落地的 `marching_cubes.h`, `subdivision.h`, `uv_unwrapper.h`, `lod_selector.h`, `bvh.h` 等 11 个 C++20 核未露出，且 [recon/backend.py](../../../python/synapse/recon/backend.py#L64-L84) 先截断为 `float32` 再让 C++ 升回 `float64` | `PR-141` |
| **`H9-P2-05`** | **P2** | **[synapse/\_\_init\_\_.py](../../../python/synapse/__init__.py#L36-L90) 漏列 `errors` / `telemetry`**：冷启动 `import synapse; synapse.errors` 报错（加载顺序敏感） | `PR-137` |
| **`H9-P2-06`** | **P2** | **[telemetry.py](../../../python/synapse/telemetry.py#L41-L47,L221-L238) 显存策略与 Trace 大小写**：当静态模型权重已超 `available_vram_gb` 时误推 `SPATIAL_TILE_2X2`（应推荐 CPU Offload）；`TraceContext` 未将大写 `trace_id` 转为 W3C 小写 | `PR-137` |
| **`H9-P2-07`** | **P2** | **缺少真 `DiffusersWanBackend` 与 3D 场景 scanned 资产**：全仓 `0` `diffusers` 接入、`30/30` 场景全无纹理白模（未接线 `SYN-RFC-0063` 扫描资产），[metrics/temporal.py](../../../python/synapse/metrics/temporal.py#L454) 仅用 CPU 单条跑 R3D-18 | `PR-139`, `PR-144` |
| **`H9-P2-08`** | **P2** | **[pyproject.toml](../../../pyproject.toml#L24-L60,L82-L140) 缺失运行依赖与规则**：`[gpu]` 缺 `diffusers`/`accelerate`/`transformers`/`torchvision`、`[train]` 缺 `webdataset`、`[dev]` 缺 `pybind11`/`hypothesis`；Ruff 缺 `E501`/`PT`/`S301`/`S310`；[`.gitignore`](../../../.gitignore) 缺 `uv.lock` 与 `eval_output/` | `PR-133` |
| **`H9-P2-09`** | **P2** | **`Synapse-Bench` 测试串行长尾 (`Gate 1a=38.6s`, `Gate 2=90.8s`)**：`test_run_all_scenarios` (`71.65s`)、`test_real_direct_compile_and_load` (`69.10s`)、`test_inspect_align` (`66.59s`) 与快测 18 处 `subprocess.run` 拖垮 SLA | `PR-150` |
| **`H9-P2-10`** | **P2** | **`data-inspector` 3/4 标签页为空壳且缺真 EXR AOV**：`Animation`、`Video + FGPP`、`Comparison` 无交互视图；`BUILD.bazel` 未接 `tinyexr` 渲染真实 EXR 通道 | `PR-146` |
| **`H9-P2-11`** | **P2** | **40 份文档膨胀失步**：每次迭代新增 600 行计划与 300 行审计导致多头文档难以协同维护 | `PR-151` |
| **`H9-P2-12`** | **P2** | **[refine/types.py](../../../python/synapse/refine/types.py#L55-L70) & [refine/gates.py](../../../python/synapse/refine/gates.py#L148)**：AP1 线性色直接套 sRGB 伽马/直传 `compute_lpips` 未做 `acescg_to_linear_srgb` | `PR-143` |
| **`H9-P3-01`** | **P3** | **[bench/report.py](../../../python/synapse/bench/report.py#L189,L361-L432) & [domain_randomizer.py](../../../python/synapse/bench/domain_randomizer.py#L178)**：遗漏 `normal_angular_error_deg`/`chamfer_mm`、`NaN` 耗时崩溃；随机化器 `frame_count_rule` 死字段 | `PR-144` |
| **`H9-P3-02`** | **P3** | **[scripts/run_cpu_verification.py](../../../scripts/run_cpu_verification.py#L372-L430) & [evaluate_3d_models.py](../../../scripts/evaluate_3d_models.py#L34)**：`--output-dir` 映射文件路径崩 `IsADirectoryError`；缺独立执行时的 `sys.path` 引导 | `PR-147` |
| **`H9-P3-03`** | **P3** | **[recon/types.py](../../../python/synapse/recon/types.py#L109) & [refine/hand.py](../../../python/synapse/refine/hand.py#L368)**：`is_quad_dominant` 命名偏差；纯 Python 栈标记连通域重复入栈 $4\times$ | `PR-142`, `PR-143` |
| **`H9-P3-04`** | **P3** | **[csrc/geometry/tests/test_geometry.cc](../../../csrc/geometry/tests/test_geometry.cc)** & [refine/face.py](../../../python/synapse/refine/face.py#L147)：C++ 缺 Quad(`vpf=4`)/高阶伞环单测；人脸角度插值未用 $S^2$ 测地距离 | `PR-141`, `PR-143` |

---

## 5. 本地工作区未提交改动风险提示（Workspace Drift Advisory）

只读审计期间确认 `/Volumes/work/dervgen/synapse` 主工作区尚存 7 个已修改但未提交的本地文件及 5 个未追踪路径，本审查严格遵循“不修改代码”原则通过独立 Git Worktree 提交文档 PR（未干扰主工作区）。以下本地草稿在后续合入前必须清理或修复：
1. `workenv/google-colab/synapse_colab_workflow.ipynb:113`：Cell 4 的 JSON 字符串含未转义换行（`print(f"\n🚨...")` 解码后直接跨行），运行必报 Python `SyntaxError`；
2. `workenv/google-colab/vscode_tunnel.sh:17`（未追踪）：`curl -sLk` 使用 `--insecure` 禁用 TLS 校验并解压至 `/usr/local/bin`，存在供应链劫持风险；
3. `tests/test_metrics.py:284`：将 `is_proxy_metric is True` 弱化为 `isinstance(..., bool)`，削弱了 `8599cd5` 已锁死的回归断言；
4. `AGENTS.md` / Skills / Colab README 本地改动回写了陈旧单测数 `1,098`（当前实测为 `1,408`），且 `uv.lock` 未列入 `.gitignore`。

---

## 6. 缺陷 → Horizon-9 (`M33–M36`, `PR-133 ~ PR-152`) 闭环映射总表

| Horizon-9 PR | 里程碑 / 波次 | 主责包域与核心交付 | 闭环审计项与战略支柱 |
| :--- | :---: | :--- | :--- |
| **[PR-133](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-133)** | M33 / W32 | 激活 `.github/workflows/ci.yml`（删 `.ci/`）、`.bazelrc` `opt`、`build_native` 超时与 `pyproject`/`.gitignore` | `H9-P0-01`, `H9-P1-12`, `H9-P2-08` · 支柱一 |
| **[PR-134](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-134)** | M33 / W32 | `fgpp` `p02` 修复/相机系与投影解析、EXR/深度非有限对齐、`import_dag_guard` 堵洞与 `wire_guard` 全量 Schema 校验 | `H9-P0-02`, `H9-P1-01`, `H9-P1-02` · 支柱二 |
| **[PR-135](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-135)** | M33 / W32 | `run_worker_cli` 原生对齐 Go `render-farm` RFC 0039 `JobBundle`、全祖先符号链接沙箱、前置 `--json` 与 `_rsi.py` `UnboundLocalError` | `H9-P0-03`, `H9-P0-04`, `H9-P1-10`, `H9-P1-11` · 支柱二 |
| **[PR-136](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-136)** | M33 / W32 | `loadGlbMesh` 32/64 位整数溢出堆越界封堵、打通三端 `--shm-file` 与 3 槽无撕裂 `Buffer` 环 | `H9-P0-07`, `H9-P0-08`, `H9-P1-20` · 支柱六 |
| **[PR-137](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-137)** | M33 / W32 | `blend` EDT 内环平滑与实体哈希引导、`detector`、`style` ACEScg/IRLS/Fréchet、`viewer` 预 `resolve` 防穿越、`__init__`/`telemetry` | `H9-P0-13`, `H9-P1-04`, `H9-P1-05`, `H9-P2-05`, `H9-P2-06` · 支柱二/三 |
| **[PR-138](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-138)** | M34 / W33 | 统一网格/光栅/色彩/Warp 轮子（`DatasetGenerator` 15ch、流式 `iter_fgpp_frames`、逐帧 `float32` 降内存至 `<1.5GB`、透传相机参数与保接缝 `load_obj`） | `H9-P1-03`, `H9-P1-06`, `H9-P2-01`, `H9-P2-02`, `H9-P2-03` · 支柱三 |
| **[PR-139](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-139)** | M34 / W33 | `WanAdapter` 去重注入/分窗 Hook/`[stride-3:stride]` 暖启动/真 `DiffusersWanBackend`、`http_security` SSRF 硬加固与 `CloudAdapters` | `H9-P0-09`, `H9-P1-07`, `H9-P1-08`, `H9-P2-07` · 支柱二/四 |
| **[PR-140](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-140)** | M34 / W33 | `StatelessLoRATrainer` CUDA/CPU 对齐、`T->tau` 与潜空间探针头、`_dpo_step` `t`、`FeatureStore` 流式 Tar 及 `synapse lora` CLI 闭环 | `H9-P0-10`, `H9-P1-09`, `H9-P1-10` · 支柱四 |
| **[PR-141](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-141)** | M34 / W33 | `@aether//geo` 绑定扩围（`MarchingCubes`, `SubdMesh`, `MeshUVUnwrapper`, `Bvh`）、C++ $O(d)$ 伞环、Python 大面片漏检与零复制 `float64` 桥接 | `H9-P1-12`, `H9-P2-02`, `H9-P2-04`, `H9-P3-04` · 支柱五 |
| **[PR-142](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-142)** | M34 / W33 | TSDF `weight>0`、Visual Hull、`quad_remesh` 一致细分与 UV、`delight` 世界 SH+UV、`tracking`/`kinematics`、红线 #3、`Mitsuba`+GGX 优化器与 `stages/recon` 修复 | `H9-P0-14`, `H9-P0-15`, `H9-P0-16`, `H9-P1-13`, `H9-P1-14`, `H9-P3-03` · 支柱五 |
| **[PR-143](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-143)** | M35 / W34 | `refine` 全旁路 `PSNR=100`、反转 EMA 修复、`motion_vectors` 接线、`CASCADE`/`POISSON`/`UNCERTAINTY`、外环 HDR 保色、多候选融合、HITL 账本写盘 | `H9-P0-17`, `H9-P1-15`, `H9-P1-16`, `H9-P2-12`, `H9-P3-03`, `H9-P3-04` · 支柱五 |
| **[PR-144](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-144)** | M35 / W34 | `BenchPipelineRunner` `applicable_keys` 回归、22/30 场景参数与扫描纹理资产、`metrics` 别名/方向/`lpips`/防频闪/`watertight` 修复 | `H9-P0-05`, `H9-P0-18`, `H9-P1-06`, `H9-P1-17`, `H9-P2-01`, `H9-P3-01` · 支柱六 |
| **[PR-145](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-145)** | M35 / W34 | `model_evaluator` 无参考/世界尺度、`macro`/`align`/`inspect_refine`/`flywheel_exporter`、`HttpJevTransport`、`RSI` 闭环桥与 `bool`/溯源修复 | `H9-P0-06`, `H9-P1-18`, `H9-P1-19` · 支柱六 |
| **[PR-146](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-146)** | M35 / W34 | `data-inspector` 多源网格隔离、15ch 真 EXR AOV 渲染、同步截图/`resize` 与 Electron 四标签页/场景树交互完备化 | `H9-P1-20`, `H9-P2-10` · 支柱六 |
| **[PR-147](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-147)** | M35 / W34 | `prepare_datasets` 零崩与诚实溯源、`colab_runner` E1/E4 真注入与状态保留、`run_roundtrip`/`assets` 离线闭环、`setup_colab` 与 `io_spec` 修复 | `H9-P0-11`, `H9-P0-12`, `H9-P1-21`, `H9-P1-22`, `H9-P3-02` · 支柱四/六 |
| **[PR-148](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-148)** | M36 / W35 | 反恒等跨仓联调（偏心 `sx==cx`、Go `models.go` + `run_worker_cli`、真实 Native→SHM）、真 `ReconPipeline` 往返与去过度 Mock | `H9-P0-02`, `H9-P0-03`, `H9-P0-18`, `H9-P1-24` · 支柱一/二 |
| **[PR-149](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-149)** | M36 / W35 | `hypothesis` 属性测试 + `.glb`/EXR/几何畸形 Fuzz 套件与 C++ ASan/UBSan 回归门禁 | `H9-P1-24`, `H9-P0-08` · 支柱一 |
| **[PR-150](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-150)** | M36 / W35 | 并发参数化 30 场景与 C++ 加速搜索使单进程快测 **`<= 20s`**、`-n auto` **`<= 10s`**、慢测 `-n auto` **`<= 40s`** | `H9-P2-09`（Gate 1a/1b/2 SLA）· 支柱一 |
| **[PR-151](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-151)** | M36 / W35 | 全仓文档三层归档收敛（`ARCHITECTURE.md`, `WIRE_CONTRACTS.md`, `archive/`）、`README.md` 同步与文档防漂移自动化断言 | `H9-P1-23`, `H9-P2-11` · 支柱七 |
| **[PR-152](../../plans/PLAN-M33-M36-HOLISTIC-REMEDIATION-AND-EVOLUTION.md#pr-152)** | M36 / W35 | Horizon-9 全量 **14 项反作弊硬门禁**（`ubuntu24` + GitHub Actions 远端绿灯双证）终验封板与 `Release v4.0.0` | 全域 14 项硬门禁终验封板 |
