# 细分执行计划 5：Phase M8 `InspectionBundle` 契约与 Electron + `fulgur` 原生可视化检查器 (`PR-26` ~ `PR-29`)

- **所属总计划**: [docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) (`v2.0` Horizon 2)
- **前置设计文档**: [docs/DESIGN_DATA_INSPECTOR.md](../DESIGN_DATA_INSPECTOR.md) · [docs/DESIGN_REVIEW_AND_ALIGNMENT.md](../DESIGN_REVIEW_AND_ALIGNMENT.md) (`RV-02`, `RV-03`)
- **覆盖范围**: `PR-26` ~ `PR-29`（共 4 个原子 PR，预估 7.0 人天）
- **核心架构约束**:
  1. **桌面端零 Python 运行时依赖（NG-5）**：Electron 进程严禁内嵌 Python 解释器或 PyTorch；指标重算统一通过子进程调用 `synapse-bench evaluate --bundle <dir> --json`；
  2. **独立 Bazel Bzlmod 子工作区（修复 `RV-03`）**：`tools/data-inspector/native/` 自带 `MODULE.bazel`，仅依赖 `@aether` 与 `@fulgur` 公开引擎目标，严禁引用 `@fulgur//course/...` 内部示例目标；
  3. **UI 规范对齐**：严格遵守 `forge-ui-design` 与 `electron-desktop-dev` 规范（`ContextBridge` 进程隔离、Forge Dark Neutral 主题令牌、防双层边框嵌套）。

---

<a id="pr-26"></a>
## 1. PR-26：`InspectionBundle (inspection.v1.0)` 标准契约校验与一键打包导出器

- **建议标题**: `feat(bench): add InspectionBundle v1.0 schema validator and exporter for Data Inspector [model-tag]`
- **前置依赖**: `PR-14a`, `PR-15`
- **代码净增预算**: `~360` 行
- **涉及文件清单**:
  - 新增：`python/synapse/bench/bundle_exporter.py`, `tests/test_inspection_bundle.py`
  - 修改：`python/synapse/pipeline/bench_runner.py`, `python/synapse/cli/__init__.py`

### 1.1 精确落地规格
1. **`InspectionBundleManifest` (`schema_version = "inspection.v1.0"`)**：
   - 定义 Pydantic v2 校验模型，锁死相对路径安全规范（拒绝任何含 `..` 路径穿越或绝对路径的 `sources` 条目！）；
   - 支持最小可用集（仅 `bundle.json` + `source/model.glb`）与全量往返集（含 `animations/*.glb`、`fgpp/`、`generated/video.mp4`、`reconstructed/model.glb`、`metrics/comparison.json`）。
2. **`export_inspection_bundle(...)` & CLI 接线**：
   - 为 `synapse-bench run` 增加 `--export-bundle <dir>` 参数，评测结束后一键将源模型、FGPP 序列、生成视频帧与 `comparison.json`（含 `defect_regions` 与物理锚点帧索引 `[0, 48, 96]`）整理输出为可直接被 Data Inspector 打开的标准目录。

### 1.2 验证门禁
- 单测验证生成最小 Bundle 与完整往返 Bundle 均通过 Schema 及相对路径安全校验，随后直接调用 `synapse-bench evaluate --bundle <dir> --json` 往返打分退出码 `0`。

---

<a id="pr-27"></a>
## 2. PR-27：`tools/data-inspector` 脚手架、独立 `native/MODULE.bazel` 渲染进程与 SHM 单模型视口

- **建议标题**: `feat(inspector): bootstrap Electron shell and standalone @fulgur native SHM viewport [model-tag]`
- **前置依赖**: `PR-26`
- **代码净增预算**: `~680` 行
- **涉及文件清单**:
  - 新增：
    - `tools/data-inspector/native/.bazelversion`, `MODULE.bazel`, `BUILD.bazel`, `main.cc`, `inspector_shm_server.h`, `inspector_shm_server.cc`, `inspector_app.h`, `inspector_app.cc`
    - `tools/data-inspector/package.json`, `electron/main.js`, `electron/preload.js`, `electron/native-process.js`, `electron/shm-reader.js`, `electron/bundle-manager.js`, `renderer/index.html`, `renderer/src/app.js`, `renderer/src/views/source-model.js`
    - `tools/data-inspector/test/test-bundle-loading.js`

### 2.1 精确落地规格
1. **C++20 原生渲染子进程 (`tools/data-inspector/native/`)**：
   - 遵循 Google C++20 规范（`-fno-exceptions`, `Result<T, E>`, `m_` 成员命名，零平台头文件泄漏至 `.h`）；
   - `MODULE.bazel` 通过 `local_path_override` 引用 `../../../../aether` 与 `../../../../fulgur`；支持命令行 `--rhi=vulkan|metal|d3d12|null --headless`（确保在 `ubuntu24` 无物理显示器 CI 环境下可通过 `--rhi=null` 执行无头协议回归！）；
   - `InspectorShmServer`：利用 `@aether//pal` 共享内存原语建立 RGBA8 双缓冲帧传输与 JSON 命令/事件环形队列，支持 `load_bundle`, `camera` (orbit/pan/zoom), `set_debug_view` (`lit`/`wireframe`/`normal`/`uv`), `set_entity_visible`, `export_screenshot`。
2. **Electron 安全壳与 View 1 (`Source Model`)**：
   - 开启 `contextIsolation: true, nodeIntegration: false`，通过 `preload.js` `contextBridge` 暴露受控 IPC 接口；
   - 实现左侧场景层级树、中央 60Hz SHM Canvas 视口与右侧网格/材质统计面板（去双层嵌套边框，适配 240px 紧凑侧边栏）。

### 2.2 验证门禁
- 在 `ubuntu24` 上执行 `cd ~/work/dervgen/synapse/tools/data-inspector/native && bazel build //:data-inspector-native` 零告警通过，以 `--rhi=null --headless` 模式发送 `load_bundle` + `camera` JSON 指令验证状态响应正常；运行 `node test/test-bundle-loading.js` 全绿。

---

<a id="pr-28"></a>
## 3. PR-28：骨骼动画播放器（八面体 Gizmo + 锚点标记）与生成视频 + FGPP 多通道同步叠加视图

- **建议标题**: `feat(inspector): add skeleton animation player, FGPP overlay canvas, and unified timeline [model-tag]`
- **前置依赖**: `PR-27`
- **代码净增预算**: `~620` 行
- **涉及文件清单**:
  - 新增：`tools/data-inspector/native/skeleton_overlay.h`, `skeleton_overlay.cc`, `tools/data-inspector/electron/exr-bridge.js`, `tools/data-inspector/renderer/src/views/animation-player.js`, `renderer/src/views/video-player.js`, `renderer/src/components/fgpp-overlay.js`, `renderer/src/components/timeline.js`, `tools/data-inspector/test/test-timeline-sync.js`
  - 修改：`tools/data-inspector/native/inspector_app.h`, `inspector_app.cc`

### 3.1 精确落地规格
1. **View 2：骨骼动画播放器 (`animation-player` + `skeleton_overlay.cc`)**：
   - 基于 `@fulgur//engine/animation` 实现多剪辑切换、逐帧 `seek`、倍速（`0.25x ~ 4.0x`）、八面体骨骼关节 Gizmo 与根运动地面投影轨迹绘制。
2. **View 3：视频与 FGPP 多通道叠加检视器 (`video-player.js` + `fgpp-overlay.js`)**：
   - 支持双路径加载 FGPP 帧：优先读取浮点缓冲以支持鼠标悬停显示精确米制深度 `CameraLinearDepth.Z (m)`、世界法线向量与 `Cryptomatte.ID` 十六进制值；当无原生 EXR 插件时自动降级调用 `synapse fgpp inspect --export-vis` 预渲染缓存；
   - 支持 `RGB Only`、`RGB + Depth Heatmap`、`RGB + Normal`、`RGB + Optical Flow Wheel`、`Side-by-Side (Video vs DirectRadiance)` 五种合成模式及 `defect_regions` 红色告警框标注。
3. **统一主从时间轴 (`TimelineController`)**：
   - 支持锁定联动动画帧与视频时间戳（帧索引映射误差 `0` 帧），并在时间轴刻度上显式高亮标注 **FGPP 物理强锚点帧（`△ 0`, `△ 48`, `△ 96`）**。

### 3.2 验证门禁
- `ubuntu24` 编译 C++ 骨骼叠加模块通过；`node test/test-timeline-sync.js` 验证 24fps / 30fps 下 97 帧序列的时间戳↔帧索引双向映射、锚点标记定位及悬停像素读值 100% 精确。

---

<a id="pr-29"></a>
## 4. PR-29：C++ `ComparisonEngine`（BVH 距离热力图/双视口同步/Ghost/Wipe）与指标仪表盘闭环

- **建议标题**: `feat(inspector): implement BVH ComparisonEngine, 4-mode compare view, and MetricsBridge dashboard [model-tag]`
- **前置依赖**: `PR-28`
- **代码净增预算**: `~650` 行
- **涉及文件清单**:
  - 新增：`tools/data-inspector/native/comparison_engine.h`, `comparison_engine.cc`, `tools/data-inspector/electron/metrics-bridge.js`, `electron/screenshot-exporter.js`, `renderer/src/views/comparison.js`, `renderer/src/components/metrics-dashboard.js`, `tools/data-inspector/test/test-comparison-metrics.js`
  - 修改：`tools/data-inspector/native/BUILD.bazel`, `inspector_app.h`, `inspector_app.cc`

### 4.1 精确落地规格
1. **C++ `ComparisonEngine`（基于 `@fulgur//engine/physics` BVH）**：
   - 加载 `source/model.glb` 与 `reconstructed/model.glb` 后，构建源网格三角形 BVH 树，毫秒级并行求取重建网格每个顶点的最近点面距离（毫米），映射为蓝（`0mm`）→ 绿（`1mm`）→ 黄（`5mm`）→ 红（`>=10mm`）热力图顶点色；
   - 单进程单帧缓冲内支持 4 种对比模式切换：`side_by_side`（左右双视口相机实时同步）、`ghost`（实体 + 半透明线框叠影）、`heatmap`（点到面距离色带）、`material_wipe`（可拖拽垂直分割线 A/B 材质对比）。
2. **`MetricsBridge` 与常驻底部仪表盘 (`metrics-dashboard.js`)**：
   - 自动加载 `metrics/comparison.json` 展示 9 项核心指标卡片及三级门禁状态徽标（`✅` / `❌`）；点击“Re-evaluate”时通过 `child_process.spawn("synapse-bench", ["evaluate", "--bundle", path, "--json"])` 异步刷新并回写磁盘；支持一键导出带指标水印的对比 PNG 截图。

### 4.2 验证门禁
- 在 `ubuntu24` 构造源单位球与整体偏移 `5.0mm` 的重建球测试 Bundle：1) C++ `get_comparison_metrics` 返回平均距离误差 `< 0.1mm`；2) `node test/test-comparison-metrics.js` 端到端跑通 `MetricsBridge` 子进程调用 `synapse-bench evaluate --bundle` 并成功解析 JSON 渲染状态徽标。
