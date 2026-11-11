# Synapse Data Inspector — 训练数据可视化工具设计方案

- **文档状态**: `Draft v1.0`
- **关联仓库**: `synapse`（宿主）, `fulgur`（渲染引擎）, `aether`（基础层）
- **前置依赖**: fulgur `explorer/` Electron↔C++ SHM 集成架构、`@dervgen/tecton-*` 设计系统
- **目标产物**: `synapse/tools/data-inspector/` — 独立 Electron 桌面应用

---

## 1. 动机与目标

### 1.1 问题陈述

Synapse 的神经-物理混合管线产出大量训练与评测数据：原始 3D 模型、骨骼动画、FGPP 多通道 EXR 序列、AI 生成视频、以及逆向重建的 3D 模型。当前 `synapse.viewer` 仅提供 2D 图片画廊式预览（纯 HTML + HTTP，无 3D 渲染能力），无法满足以下核心需求：

1. **原始资产检视**：交互式 3D 查看原始模型（网格、材质、包围盒）与骨骼动画（播放、时间轴 scrub、骨骼叠加可视化）
2. **生成结果审阅**：并排查看 AI 生成视频与 FGPP 先验通道（深度、法线、运动向量）叠加
3. **往返质量比对**：将逆向重建模型与原始模型在同一场景中直观对比（叠影、差异热力图、量化指标仪表盘）

### 1.2 目标用户

| 角色 | 核心诉求 |
|:---|:---|
| **算法研究员** | 快速诊断生成视频的时序一致性、几何保真度、风格漂移 |
| **3D 美术 / TD** | 检查重建模型的拓扑质量、UV 展开、材质保真度 |
| **QA 自动化 Agent** | 在评测流程中截取对比截图、自动生成差异报告 |
| **项目负责人** | 一览式仪表盘快速判定训练批次质量 |

### 1.3 设计原则

1. **复用 fulgur 渲染核**：不自研 WebGL/WebGPU 渲染器，直接嵌入 fulgur C++ 引擎通过 SHM 双缓冲输出帧，获得 PBR、骨骼动画、阴影、后处理等 AAA 级渲染能力
2. **统一 tecton 设计系统**：UI 层使用 `@dervgen/tecton-components` + `@dervgen/tecton-tokens`，与 fulgur explorer、grand-studio 保持视觉一致性
3. **桌面端零 Python 依赖**（NG-5）：Electron 应用不 `import synapse`，指标计算通过子进程调用 `synapse-bench` CLI（`--json`）获取结果
4. **数据驱动，约定优于配置**：通过标准化的 **Inspection Bundle** 目录结构加载数据，无需复杂配置

---

## 2. 架构概览

### 2.1 系统分层

```
┌──────────────────────────────────────────────────────────────────┐
│                    Electron Main Process                         │
│                                                                  │
│   ┌─────────────┐  ┌───────────────┐  ┌───────────────────────┐ │
│   │NativeProcess│  │ BundleWatcher │  │   MetricsBridge       │ │
│   │  Manager    │  │ (fs.watch)    │  │ (spawn synapse CLI)   │ │
│   └──────┬──────┘  └───────────────┘  └───────────────────────┘ │
│          │                                                       │
│     SHM Transport (双缓冲 RGBA + JSON 命令/事件环形缓冲区)       │
│          │                                                       │
│   ┌──────▼──────────────────────────────────────────────────┐   │
│   │        data-inspector-native (C++20, Bazel)             │   │
│   │                                                          │   │
│   │   ┌──────────┐  ┌───────────┐  ┌────────────────────┐  │   │
│   │   │  Scene   │  │ Animation │  │    Comparison      │  │   │
│   │   │ Renderer │  │  Player   │  │     Engine         │  │   │
│   │   │(fulgur)  │  │ (fulgur)  │  │ (双模型叠加渲染)    │  │   │
│   │   └──────────┘  └───────────┘  └────────────────────┘  │   │
│   │   ┌──────────────────────────────────────────────────┐  │   │
│   │   │              AssetManager (fulgur)                │  │   │
│   │   │   glTF / FBX / OBJ → MeshData + Skeleton + Anim │  │   │
│   │   └──────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────┘   │
│          │ SHM Frames (RGBA)                                     │
│   ┌──────▼──────────────────────────────────────────────────┐   │
│   │            Electron Renderer Process                     │   │
│   │                                                          │   │
│   │   ┌──────────────────────────────────────────────────┐  │   │
│   │   │           tecton UI Shell (HTML/CSS/JS)          │  │   │
│   │   │                                                  │  │   │
│   │   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐  │  │   │
│   │   │  │ Source │ │ Anim  │ │ Video │ │ Compare  │  │  │   │
│   │   │  │ Model  │ │Player │ │Player │ │  View    │  │  │   │
│   │   │  │  Tab   │ │  Tab  │ │  Tab  │ │   Tab    │  │  │   │
│   │   │  └────────┘ └────────┘ └────────┘ └──────────┘  │  │   │
│   │   │  ┌──────────────────────────────────────────┐    │  │   │
│   │   │  │    Metrics Dashboard (底部面板)           │    │  │   │
│   │   │  └──────────────────────────────────────────┘    │  │   │
│   │   └──────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈选型

| 层级 | 技术 | 理由 |
|:---|:---|:---|
| **桌面壳** | Electron 33+ | 复用 fulgur explorer 成熟的 SHM 集成方案，团队已有经验 |
| **UI 组件** | `@dervgen/tecton-components` / `@dervgen/tecton-tokens` | 与 grand-studio / fulgur explorer 视觉统一 |
| **3D 渲染** | fulgur C++20 引擎（HeadlessServer 模式） | PBR、骨骼动画、后处理，GPU 硬件加速 |
| **进程通信** | SHM 双缓冲 + JSON Ring Buffer | 已验证的 60fps 低延迟方案 |
| **视频播放** | HTML5 `<video>` + Canvas 帧同步 | 浏览器原生，无额外依赖 |
| **指标计算** | `synapse` CLI 子进程（`--json`） | 遵守 NG-5，不在桌面端引入 PyTorch |
| **构建** | Bazel（C++ native）+ npm（Electron） | 与 fulgur 统一构建系统 |

---

## 3. Inspection Bundle 数据契约

Data Inspector 通过打开一个标准化的 **Inspection Bundle** 目录来加载所有数据。

### 3.1 目录结构

```
<bundle_root>/
├── bundle.json                      # 必需：Bundle 元数据与索引
├── source/                          # 必需：原始资产
│   ├── model.glb                    #   原始 3D 模型（glTF 2.0）
│   ├── animations/                  #   动画剪辑目录（可选）
│   │   ├── idle.glb
│   │   ├── walk.glb
│   │   └── run.glb
│   └── thumbnail.png                #   缩略图（可选）
├── fgpp/                            # 可选：FGPP 多通道渲染序列
│   ├── metadata.json                #   FGPP 元数据（fgpp.v1.0 schema）
│   └── frame_000000.exr             #   OpenEXR 帧序列
├── generated/                       # 可选：AI 生成视频
│   ├── video.mp4                    #   生成视频（H.264/H.265）
│   ├── adapter_id.txt               #   生成所用适配器标识
│   └── frames/                      #   逐帧 PNG（备选，用于精确帧对齐）
│       └── frame_000000.png
├── reconstructed/                   # 可选：逆向重建模型
│   ├── model.glb                    #   重建 3D 模型
│   ├── textures/                    #   提取的纹理（可选）
│   └── reconstruction_log.json      #   重建管线元数据
└── metrics/                         # 可选：预计算指标
    └── comparison.json              #   synapse metrics 输出
```

### 3.2 `bundle.json` Schema

```jsonc
{
  "schema_version": "inspection.v1.0",
  "bundle_id": "t2_humanoid_walk_wan_20261110",
  "created_at": "2026-11-10T14:30:00Z",
  "description": "Humanoid-53 walk cycle — Wan 2.1 generation + TRELLIS reconstruction",

  // 数据源声明（控制 UI 中哪些 Tab 可用）
  "sources": {
    "source_model": "source/model.glb",
    "animations": ["source/animations/idle.glb", "source/animations/walk.glb"],
    "fgpp_sequence": "fgpp/",
    "generated_video": "generated/video.mp4",
    "generated_frames": "generated/frames/",
    "reconstructed_model": "reconstructed/model.glb",
    "metrics": "metrics/comparison.json"
  },

  // 场景信息（可选，用于自动配置相机）
  "scene": {
    "up_axis": "Y",
    "units": "meters",
    "scenario_id": "t2_humanoid_walk_orbit",
    "adapter_id": "wan_2.1_controlnext"
  }
}
```

> [!NOTE]
> 所有路径均为相对于 `bundle_root` 的相对路径。缺失的可选项不会显示对应 Tab。最小可用 Bundle 仅需 `bundle.json` + `source/model.glb`。

---

## 4. 五个核心视图设计

### 4.1 View 1：Source Model（原始模型检视）

**功能**：
- 加载并渲染原始 glTF/FBX/OBJ 模型，支持 PBR 材质、纹理、法线贴图完整显示
- 轨道相机（鼠标拖拽旋转、滚轮缩放、中键平移），支持快捷预设视角（前/后/左/右/上/下）
- 显示模式切换：实体 / 线框 / 线框+实体 / 法线可视化 / UV 检查器
- 侧栏信息面板：顶点数、面数、子网格列表、材质参数（PBR 属性、纹理预览）、包围盒尺寸
- 场景层级树（来自 glTF 节点层级），支持点击选中高亮、单独隐藏/显示

**fulgur 命令映射**：
| UI 操作 | JSON Command |
|:---|:---|
| 旋转/缩放/平移 | `{"cmd":"camera","type":"orbit","yaw":..,"pitch":..,"distance":..}` |
| 切换线框 | `{"cmd":"set_debug_view","mode":"wireframe"}` |
| 选中子网格 | `{"cmd":"pick_at","x":..,"y":..}` → 返回 `entity_id` + `material` |
| 隐藏节点 | `{"cmd":"set_entity_visible","id":..,"visible":false}` |

---

### 4.2 View 2：Animation Player（动画播放器）

**功能**：
- 加载模型绑定的骨骼与动画剪辑，支持多剪辑选择切换
- 时间轴控件：播放/暂停、逐帧步进（←→）、速度调节（0.25×~4×）、循环模式
- 时间轴 Scrub：拖拽到任意帧，引擎实时更新姿态
- 骨骼可视化叠加：八面体骨骼 Gizmo、关节标签、局部坐标轴（可独立开关）
- 根运动轨迹可视化：地面投影曲线显示角色位移路径
- 关键帧标记：在时间轴上标注 FGPP 物理锚点帧（第 0、48、96 帧）

**fulgur 命令映射**：
| UI 操作 | JSON Command |
|:---|:---|
| 播放/暂停 | `{"cmd":"animation","action":"play"}` / `"pause"` |
| Scrub 到帧 | `{"cmd":"animation","action":"seek","frame":48}` |
| 切换剪辑 | `{"cmd":"animation","action":"set_clip","clip":"walk"}` |
| 速度调节 | `{"cmd":"animation","action":"set_speed","speed":0.5}` |
| 骨骼叠加 | `{"cmd":"set_skeleton_overlay","enabled":true,"style":"octahedral"}` |

---

### 4.3 View 3：Generated Video Player（生成视频播放器）

**功能**：
- 播放 AI 生成视频（MP4 via HTML5 `<video>`），与 FGPP 帧序列精确同步
- 多通道叠加模式（通过 Canvas 合成，不经 fulgur）：
  - RGB Only（纯生成视频）
  - RGB + Depth Overlay（深度热力图半透明叠加）
  - RGB + Normal Overlay（法线可视化）
  - RGB + Motion Vector Overlay（光流 HSV 轮盘）
  - Side-by-side：生成视频 vs FGPP DirectRadiance
- FGPP 通道独立检视器：选择任意通道全屏查看、鼠标悬停显示像素值
- 时间轴同步：与 View 2 动画时间轴联动（锁定/解锁模式）
- 异常区域高亮：若 `metrics/comparison.json` 包含 `defect_regions`，用边框高亮标注闪烁/穿模/轮廓漂移区域

**实现要点**：
- 视频播放由 Renderer 进程 HTML5 原生处理，**不经 fulgur C++ 渲染**
- FGPP EXR 帧由 Electron Main 进程读取（使用 `openexr` N-API 原生模块或预转换为 PNG），通过 IPC 发送到 Renderer
- 帧同步策略：以视频播放时间戳为主时钟，查找最近 FGPP 帧索引

---

### 4.4 View 4：Reconstructed Model（重建模型检视）

**功能**：
- 与 View 1 共享同一套 3D 交互逻辑（轨道相机、显示模式、场景层级）
- 额外信息面板：重建来源（`TRELLIS` / `Hunyuan3D-2`）、重建分辨率、重拓扑状态
- 纹理质量检视器：UV 展开可视化、纹理分辨率与 Texel Density 热力图
- 重建置信度着色（若 `reconstruction_log.json` 提供逐顶点置信度）

---

### 4.5 View 5：Comparison（模型对比）

**核心视图**，也是此工具最大的差异化价值。

**布局模式**（用户可切换）：

#### A. 并排双视口 (Side-by-Side)
```
┌──────────────────┬──────────────────┐
│   原始模型 (L)   │   重建模型 (R)   │
│   轨道相机同步   │   轨道相机同步   │
└──────────────────┴──────────────────┘
```
- 两个 fulgur 视口，相机操作**实时同步**（拖拽左侧 = 右侧同步旋转）
- 各自独立的线框/材质模式切换
- 鼠标悬停左右联动高亮对应区域

#### B. 叠影模式 (Ghost Overlay)
- 单一视口，原始模型实体渲染 + 重建模型半透明线框叠加（或反之）
- 透明度滑块控制叠加权重
- 顶点位移箭头可视化（可选，仅在顶点数相近时启用）

#### C. 差异热力图 (Difference Heatmap)
- 单一视口渲染重建模型，表面颜色映射为与原始模型的**点到面距离**
- 色带：蓝（0mm）→ 绿（1mm）→ 黄（5mm）→ 红（>10mm）
- 距离计算由 fulgur C++ 侧在加载时预计算（BVH 加速最近点查询），存入顶点色通道

#### D. 材质对比切片 (Material A/B Wipe)
- 单一视口，可拖拽的垂直分割线，左半显示原始材质、右半显示重建材质
- 相同相机、相同光照，仅材质/纹理不同

**指标仪表盘（底部常驻面板）**：

```
┌─────────────────────────────────────────────────────────────────┐
│  ● Chamfer (mm)    ● F-Score@1mm    ● Normal Error (°)         │
│    2.34 ✅           0.87 ✅          8.7° ✅                    │
│                                                                  │
│  ● Volume Breath.  ● Albedo ΔE      ● Bone Drift (%)           │
│    3.2% ✅           4.1 ✅           0.8% ✅                    │
│                                                                  │
│  ● Foot Slide (cm) ● CIEDE2000      ● Light Dir Err (°)        │
│    0.22 ✅           2.1 ✅           6.3° ✅                    │
└─────────────────────────────────────────────────────────────────┘
        ✅ = 通过门禁阈值    ❌ = 超出阈值（红色高亮）
```

- 指标数据来源：优先加载 `metrics/comparison.json`（预计算）；若缺失，提供"Run Metrics"按钮通过 IPC 调用 `synapse-bench evaluate --bundle <path> --json` 实时计算

---

## 5. C++ Native Binary 设计（独立 Bazel 子工作区规范）

### 5.1 `MODULE.bazel` 与 Target 依赖（修复跨仓可见性契约）

由于 `synapse` 仓库根目录为纯 Python 包（无根级 `MODULE.bazel`），`synapse/tools/data-inspector/native/` 作为独立 Bazel 9.3.1 Bzlmod 子工作区构建，通过 `local_path_override` 引用同级 `@aether` 与 `@fulgur` 公开模块，**严禁依赖 `@fulgur//course/...` 内部私有目标**：

```python
# synapse/tools/data-inspector/native/MODULE.bazel
module(name = "synapse_data_inspector_native", version = "0.1.0")

bazel_dep(name = "aether", version = "0.1.0")
bazel_dep(name = "fulgur", version = "0.1.0")
local_path_override(module_name = "aether", path = "../../../../aether")
local_path_override(module_name = "fulgur", path = "../../../../fulgur")

# synapse/tools/data-inspector/native/BUILD.bazel
cc_binary(
    name = "data-inspector-native",
    srcs = [
        "main.cc",
        "inspector_app.cc",
        "comparison_engine.cc",
        "inspector_shm_server.cc",
    ],
    hdrs = [
        "inspector_app.h",
        "comparison_engine.h",
        "inspector_shm_server.h",
    ],
    deps = [
        "@aether//core",
        "@aether//pal",
        "@fulgur//engine/rhi",
        "@fulgur//engine/renderer",
        "@fulgur//engine/asset",
        "@fulgur//engine/animation",
        "@fulgur//engine/physics",  # BVH 用于点到面 Chamfer 距离预计算
    ],
)
```

### 5.2 `InspectorApp` 核心类

```cpp
class InspectorApp {
public:
  /// 初始化渲染器、资源管理器
  void init(const InspectorConfig& config);

  /// 加载 Inspection Bundle
  Result<void, std::string> loadBundle(const std::filesystem::path& bundleRoot);

  /// 主循环（由 HeadlessServer 驱动）
  void update(float dt);
  void render(rhi::ICommandList& cmd, rhi::TextureHandle target);

  /// JSON 命令处理（扩展 HeadlessServer 命令集）
  void handleCommand(const json& cmd);

private:
  // 场景管理
  struct LoadedScene {
    asset::MeshData meshData;
    rhi::MeshHandle gpuMesh;
    animation::Skeleton skeleton;
    std::vector<animation::AnimationClip> clips;
    animation::AnimationPlayer player;
    physics::BVH bvh;                    // 用于最近点查询
  };

  LoadedScene m_sourceScene;              // 原始模型
  LoadedScene m_reconScene;               // 重建模型（可选）

  // 对比引擎
  ComparisonEngine m_comparison;          // 点到面距离、顶点色映射

  // 视口模式
  enum class ViewMode { Source, Animation, Comparison };
  ViewMode m_activeView = ViewMode::Source;

  // 对比布局
  enum class CompareLayout { SideBySide, GhostOverlay, DiffHeatmap, MaterialWipe };
  CompareLayout m_compareLayout = CompareLayout::SideBySide;
};
```

### 5.3 `ComparisonEngine`

```cpp
class ComparisonEngine {
public:
  /// 预计算原始↔重建模型的逐顶点距离（BVH 加速）
  void precompute(const asset::MeshData& source, const asset::MeshData& recon);

  /// 获取重建模型逐顶点的距离值（用于热力图着色）
  std::span<const float> getVertexDistances() const;

  /// 获取聚合指标
  struct AggregateMetrics {
    float chamferMm;
    float fScore1mm;
    float meanNormalErrorDeg;
    float volumeBreathingPct;
  };
  AggregateMetrics getAggregateMetrics() const;

  /// Side-by-Side 模式：同步双视口相机
  void syncCameras(renderer::Camera& left, renderer::Camera& right);
};
```

### 5.4 扩展 JSON 命令集

在 HeadlessServer 标准命令基础上，新增以下命令：

```jsonc
// 加载 Bundle
{"cmd": "load_bundle", "path": "/path/to/bundle"}

// 切换主视图
{"cmd": "set_view", "view": "source" | "animation" | "comparison"}

// 对比布局切换
{"cmd": "set_compare_layout", "layout": "side_by_side" | "ghost" | "heatmap" | "material_wipe"}

// Ghost 模式透明度
{"cmd": "set_ghost_opacity", "opacity": 0.4}

// Material Wipe 分割线位置
{"cmd": "set_wipe_position", "x_ratio": 0.5}

// 热力图色带范围
{"cmd": "set_heatmap_range", "min_mm": 0.0, "max_mm": 10.0}

// 请求聚合指标（C++ 侧计算的几何指标）
{"cmd": "get_comparison_metrics"}
// → Event: {"event": "comparison_metrics", "data": {"chamfer_mm": 2.34, ...}}

// 导出对比截图
{"cmd": "export_screenshot", "path": "/tmp/compare.png", "views": ["source", "recon", "heatmap"]}
```

---

## 6. Electron Shell 与 tecton UI 设计

### 6.1 窗口布局

```
┌──────────────────────────────────────────────────────────────────┐
│  ◉ ◉ ◉  Synapse Data Inspector — t2_humanoid_walk_wan    ─ □ ✕ │
├──────────────────────────────────────────────────────────────────┤
│  File ▾   View ▾   Compare ▾   Tools ▾                   Help ▾ │
├──┬───────────────────────────────────────────────────────────┬───┤
│  │  [Source Model] [Animation] [Generated Video] [Compare]  │ ℹ │
│  │                                                           │   │
│  │                                                           │ P │
│  │                                                           │ r │
│  │              ┌─────────────────────────┐                  │ o │
│S │              │                         │                  │ p │
│c │              │    3D Viewport / Video   │                  │ e │
│e │              │      (Canvas)            │                  │ r │
│n │              │                         │                  │ t │
│e │              └─────────────────────────┘                  │ i │
│  │                                                           │ e │
│T │  ◀ ▶ ⏸  ━━━━━━━●━━━━━━━━━━━━━━━ 48/97  1.0×  🔁        │ s │
│r │  △0    △48    △96          物理锚点帧标记                  │   │
│e │                                                           │   │
│e ├───────────────────────────────────────────────────────────┤   │
│  │  Metrics Dashboard                                        │   │
│  │  Chamfer: 2.34mm ✅ │ F-Score: 0.87 ✅ │ Normal: 8.7° ✅  │   │
│  │  Foot Slide: 0.22cm ✅ │ CIEDE2000: 2.1 ✅ │ ...          │   │
└──┴───────────────────────────────────────────────────────────┴───┘
```

**布局区域**：
- **左侧栏（Scene Tree）**：场景层级树（模型节点、骨骼关节、子网格），支持展开/折叠/选中/隐藏
- **中央区域**：3D 视口（fulgur SHM Canvas）或视频播放器，Tab 切换
- **右侧栏（Properties）**：选中对象的属性面板（网格信息、材质参数、动画状态、对比指标详情）
- **底部面板（Timeline + Metrics）**：统一时间轴 + 指标仪表盘，可折叠

### 6.2 Electron 进程架构

```
Main Process
  ├── NativeProcessManager          # 管理 data-inspector-native 子进程
  │     ├── spawn / kill / restart
  │     ├── SHM frame polling (60Hz)
  │     └── JSON command/event ring buffer
  ├── BundleManager                 # 加载/校验 bundle.json，管理文件路径
  ├── MetricsBridge                 # 调用 synapse CLI 计算指标
  │     └── child_process.spawn("synapse-bench", ["evaluate", ...])
  ├── ExrReader                     # 读取 FGPP EXR 帧（N-API 原生模块或预处理）
  │     └── 输出 Float32Array 按通道分离，IPC 发送到 Renderer
  └── ScreenshotExporter            # 导出对比截图、HTML 报告片段

Renderer Process
  ├── App Shell (tecton-components)
  │     ├── <tc-tab-bar>            # 四个主视图 Tab
  │     ├── <tc-split-pane>         # 三栏布局
  │     └── <tc-toolbar>            # 工具栏
  ├── NativeFrameRenderer           # 接收 SHM RGBA → Canvas 绘制
  ├── VideoPlayer                   # HTML5 <video> + 帧同步
  ├── FgppOverlayRenderer           # Canvas 2D 叠加 FGPP 通道可视化
  ├── TimelineController            # 统一时间轴（驱动动画 + 视频 + FGPP）
  ├── MetricsDashboard              # 指标仪表盘组件
  └── SceneTreePanel / PropertiesPanel
```

---

## 7. 与 synapse CLI 的集成

### 7.1 指标计算

Data Inspector 通过子进程调用 synapse CLI 获取量化指标，严格遵守 **NG-5（桌面端零 Python 依赖）**：

```javascript
// MetricsBridge.js (Main Process)
const { spawn } = require('child_process');

async function computeMetrics(bundlePath) {
  return new Promise((resolve, reject) => {
    const proc = spawn('synapse-bench', [
      'evaluate',
      '--bundle', bundlePath,
      '--json'
    ]);

    let stdout = '';
    proc.stdout.on('data', d => stdout += d);
    proc.on('close', code => {
      if (code === 0) resolve(JSON.parse(stdout));
      else reject(new Error(`synapse-bench exited with code ${code}`));
    });
  });
}
```

### 7.2 FGPP EXR 预处理

对于 FGPP 通道叠加显示，有两种方案：

| 方案 | 实现 | 优劣 |
|:---|:---|:---|
| **A. 运行时 N-API 模块** | 编写 `exr-reader` N-API 原生模块直接读取 OpenEXR | 实时、按需加载，但需要编译原生模块 |
| **B. 预处理为 PNG** | 首次打开 Bundle 时调用 `synapse fgpp inspect --export-vis` 生成可视化 PNG | 零原生依赖，但需要首次等待预处理 |

**推荐方案 A**（N-API 模块），因为需要访问原始浮点深度值用于鼠标悬停像素值显示。方案 B 作为 fallback（当 synapse 环境可用时使用 CLI 预处理）。

---

## 8. 实施阶段规划

### Phase 1：项目脚手架 + 单模型视口（2 周）

| 任务 | 交付物 |
|:---|:---|
| 初始化 `tools/data-inspector/` 目录结构（Electron + native） | `package.json`, `BUILD.bazel`, 基本骨架 |
| 从 fulgur explorer 复制并适配 SHM 传输层 | `shm-reader.js`, `native-process.js`, `shm_addon.c` |
| 实现 `InspectorApp` 基础版：加载单个 glTF，轨道相机 | `inspector_app.cc`, `main.cc` |
| tecton UI Shell：Tab 栏 + 3D Canvas + 侧栏 | `renderer/` 全套 |
| Bundle 加载器：解析 `bundle.json`，校验路径 | `BundleManager.js` |
| **验证门禁**：打开含 `source/model.glb` 的 Bundle，3D 视口实时旋转查看 | |

### Phase 2：动画 + 视频播放（2 周）

| 任务 | 交付物 |
|:---|:---|
| 动画播放器：加载骨骼 + 剪辑，时间轴 scrub | Animation Tab |
| 骨骼可视化叠加（八面体 Gizmo） | `skeleton_overlay.cc` |
| HTML5 视频播放器 + FGPP 帧同步 | Video Tab |
| FGPP 通道叠加渲染（深度/法线/光流 Canvas 合成） | `FgppOverlayRenderer.js` |
| 统一时间轴控件（动画 + 视频 + FGPP 联动） | `TimelineController.js` |
| **验证门禁**：播放动画同时叠加骨骼可视化；视频播放与 FGPP 深度叠加精确同步 | |

### Phase 3：对比视图 + 指标（2 周）

| 任务 | 交付物 |
|:---|:---|
| `ComparisonEngine`：双模型加载、BVH 距离计算 | `comparison_engine.cc` |
| 并排双视口（相机同步） | Compare Tab — Side-by-Side |
| 差异热力图着色 | Compare Tab — Heatmap |
| Ghost 叠影模式 + Material A/B Wipe | Compare Tab — Ghost & Wipe |
| MetricsBridge：synapse CLI 调用 + 仪表盘 UI | `MetricsBridge.js`, `MetricsDashboard` |
| **验证门禁**：加载原始+重建模型，热力图颜色正确映射距离，指标仪表盘数据与 CLI 一致 | |

### Phase 4：打磨 + 打包（1 周）

| 任务 | 交付物 |
|:---|:---|
| 异常区域高亮（从 metrics 读取 `defect_regions`） | Video Tab 增强 |
| 截图导出（对比截图 + 指标文字水印） | `ScreenshotExporter.js` |
| 物理锚点帧标记（△ 标记在时间轴上） | Timeline 增强 |
| 快捷键系统（Space 播放、1-4 切 Tab、F 聚焦、W 线框） | KeyboardManager |
| Electron Builder 打包配置（Windows .exe + macOS .dmg） | `electron-builder.yml` |
| **验证门禁**：完整 Bundle 端到端体验，导出截图，打包安装后独立运行 | |

---

## 9. 目标文件结构

```
synapse/tools/data-inspector/
├── package.json                     # Electron + tecton 依赖
├── electron-builder.yml             # 打包配置
├── README.md                        # 使用说明
│
├── electron/                        # Electron 主进程
│   ├── main.js                      # 窗口创建、生命周期
│   ├── preload.js                   # contextBridge API 暴露
│   ├── native-process.js            # fulgur C++ 子进程管理
│   ├── shm-reader.js                # SHM 双缓冲帧读取
│   ├── bundle-manager.js            # Inspection Bundle 加载/校验
│   ├── metrics-bridge.js            # synapse CLI 子进程调用
│   └── exr-reader.js                # FGPP EXR 读取（或 N-API）
│
├── renderer/                        # Electron 渲染进程
│   ├── index.html                   # 入口 HTML
│   ├── src/
│   │   ├── app.js                   # 应用初始化
│   │   ├── views/
│   │   │   ├── source-model.js      # View 1: 原始模型
│   │   │   ├── animation-player.js  # View 2: 动画播放
│   │   │   ├── video-player.js      # View 3: 生成视频
│   │   │   └── comparison.js        # View 5: 模型对比
│   │   ├── components/
│   │   │   ├── native-renderer.js   # SHM → Canvas
│   │   │   ├── fgpp-overlay.js      # FGPP 通道叠加
│   │   │   ├── timeline.js          # 统一时间轴
│   │   │   ├── metrics-dashboard.js # 指标仪表盘
│   │   │   ├── scene-tree.js        # 场景层级树
│   │   │   └── properties-panel.js  # 属性面板
│   │   └── styles/
│   │       └── inspector.css        # tecton tokens 主题
│   └── assets/                      # UI 图标、默认纹理
│
├── native/                          # C++ 原生渲染二进制
│   ├── BUILD.bazel                  # Bazel 构建规则
│   ├── main.cc                      # 入口 + HeadlessServer 初始化
│   ├── inspector_app.h/cc           # InspectorApp 核心
│   ├── comparison_engine.h/cc       # 双模型对比引擎
│   ├── inspector_commands.h/cc      # 扩展 JSON 命令处理
│   └── shm-transport/               # 复用 fulgur explorer SHM 头文件
│       └── fulgur_frame_shm.h
│
└── test/                            # 测试
    ├── fixtures/                    # 测试 Bundle（最小 glTF + bundle.json）
    └── test-bundle-loading.js       # Bundle 加载/校验测试
```

---

## 10. 非目标与边界 (Non-Goals)

| 编号 | 不做的事 | 正确归属 |
|:---|:---|:---|
| NG-I-1 | ❌ 不在 Data Inspector 中编辑或修改 3D 模型 | `asset-forge` |
| NG-I-2 | ❌ 不运行 AI 视频生成或模型重建推理 | `foundary` 容器 / `render-farm` |
| NG-I-3 | ❌ 不直接 `import synapse` 或加载 PyTorch | 通过 CLI 子进程调用（NG-5） |
| NG-I-4 | ❌ 不实现在线协作或多用户审阅 | `grand-studio` |
| NG-I-5 | ❌ 不替代 `synapse.viewer` 的轻量 Web 预览功能 | `viewer/` 保留用于无 GPU 环境 |

---

## 11. 开放设计问题 (Open Questions)

| # | 问题 | 候选方案 | 建议 |
|:---|:---|:---|:---|
| Q1 | Data Inspector 是放在 `synapse/tools/` 还是独立仓库？ | A) `synapse/tools/data-inspector/` B) 独立 `data-inspector` 仓库 | **A**：与 synapse 数据格式紧密耦合，且 C++ native 部分通过 Bazel 直接依赖 fulgur，放在 synapse 内更易维护 |
| Q2 | 对比视图的 Side-by-Side 是否需要两个独立 fulgur 进程？ | A) 单进程双 RenderTarget B) 双进程双 SHM | **A**：单进程内通过 RenderGraph 渲染两个视口到同一帧缓冲区，避免双进程同步开销 |
| Q3 | EXR 读取使用 N-API 原生模块还是 synapse CLI 预处理？ | A) N-API `openexr` B) CLI 预处理 PNG | **A**（主路径）+ **B**（fallback），见 §7.2 |
| Q4 | 是否需要支持同时打开多个 Bundle 做跨 Bundle 对比？ | A) 单 Bundle B) 多 Bundle Tab | **A**（v1.0 先做单 Bundle，多 Bundle 留 v2.0） |
