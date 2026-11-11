# Synapse 3D 建模智能质检、多尺度对齐与精修数据飞轮设计规范 (3D Quality Inspection, Multi-scale Alignment & Refinement Flywheel Specification)

- **文档状态**: `Draft v1.0`
- **关联仓库**: `synapse`（执行核心、CLI 与评测 SDK）、`fulgur`（C++20 AAA 渲染引擎与离屏 FGPP 提取器）、`aether`（底层类型、数学库与 FGPP SSOT）、`forgekit` / `tecton`（3D 编辑器交互、TimeWarp 与数据采集）、`foundary`（GPU 训练与容器调度）
- **关联生态规约**:
  - **RFC 0019**: `CommandIR` 序列化操作协议与离线同步机制
  - **RFC 0020**: 人机协同操作时间轴与模型训练数据闭环 (`IntentGroup`, `OperatorIdentity`)
  - **RFC 0021**: 模型训练数据采集完整性与跨系统对齐 (`MultiAxisRating`, `StepAnnotation`, `ReasoningTrace`)
  - **RFC 0039**: 资产解耦与自洽作业闭包 (`JobBundle`)
  - **RFC 0044**: 神经-物理混合工业流水线、FGPP 与人机协同 (HITL 15 分钟熔断与白模粗糙度下限)
- **目标受众**: 神经几何算法工程师、3D 引擎技术美术 (TA/TD)、自动化质检与训练 Agent

---

## 1. 动机与核心定位

### 1.1 背景与行业痛点

在 3D 内容生产（生成式 3D 重建、神经视频反投、数字资产入库）中，资产质量控制（QA）一直面临三大结构性矛盾：

1. **几何硬规则与语义/美学缺陷的断层**：
   - 传统的几何检查脚本（如 `trimesh`、`libigl`）只能检测非流形（Non-manifold）、简并面（Degenerate faces）和孤立点等纯拓扑问题；
   - 无法识别“面部凹陷塌陷”、“解剖学肢体畸形”、“布料褶皱严重失真”、“UV 拉伸破损”或“浮动几何碎片”等高层语义和美术视觉缺陷。
2. **通用漫谈式大模型（Chat VLM）的高成本与低可靠性**：
   - 使用通用多模态大语言模型（如 GPT-4o、Claude 3.5 Sonnet）做质检，依赖漫长的自回归文本生成，单次推理耗时数秒甚至数十秒；
   - 输出格式易碎（JSON 解析偶发失败），且缺乏校准的数值置信度，难以作为自动化流水线的硬性门禁。
3. **全局视角对微观高频瑕疵的遮蔽**：
   - 在全景渲染视口（如 1024×1024 或 2K）中，微小的顶点自穿模、局部法线反转、贴图缝合露白往往只占几个像素，全局评估极易漏检；
   - 美术在人工审查时通常会“拉近镜头看特写”，而现有的自动化质检管线缺乏动态调动微距相机（Macro Virtual Camera）的能力。
4. **质检与修复数据的孤岛化**：
   - 现有的质检仅作为“通过/驳回”的被动过滤器，人工或算法在精修过程中产生的“缺陷定位 $\to$ 修复动作 $\to$ 修复后状态”高质量演进数据未被系统化回收，无法反哺模型进化。

### 1.2 核心解法：Visual Jev 判别决策 + 多尺度对齐 + RL 数据飞轮

本规范定义了一套**自动化、强类型、端到端的 3D 资产质检与修复数据飞轮**：
* **单图逆向位姿对齐（Pose Inversion & Alignment）**：通过 DINOv2 球面粗检索与 6DoF 姿态估计（FoundationPose / 可微光栅化），自动计算参考图片到 3D 模型的对应相机机位；
* **同位多通道解耦（Isotopic Multi-Channel GBuffer）**：利用 `fulgur` 渲染核同时输出 RGB、世界法线（World Normal）、米制线性深度（Linear Depth）和线框拓扑（Wireframe），实现几何与材质的物理级解耦；
* **Visual Jev 强类型决策（System 1 Decision Engine）**：基于共享视觉上下文（Shared Visual Context）技术，在百毫秒内对批量质检提问直接从 Logits 读取校准概率，输出零幻觉的结构化判决；
* **视锥动态微距放大（Macro Zoom-in & 3D Grounding）**：基于全局残差热力图定位潜在瑕疵，自适应调度虚拟微距相机拉近拍摄高倍局部切片（Patch Crop），并将 2D 缺陷反投回 3D 顶点空间；
* **强化学习闭环反哺（RL & Flywheel Integration）**：全面接入 RFC 0020 / RFC 0021 规范，将“检测-微距-修复”轨迹记录为 `IntentGroup` 五元组与细粒度 `StepAnnotation`，持续反哺训练 Process Reward Models (PRM)、DPO 偏好对与 3D 精修 Agent。

---

## 2. 端到端五阶质检与精修管线架构

整个流水线划分为 5 个串行阶段，每个阶段均包含严格的输入契约、处理算子与输出量化门禁。

```
 [输入: 参考图 + 3D Mesh]
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: 姿态解算与视锥逆向对齐 (Camera Inversion & Alignment)         │
│ - 算子: DINOv2 球面粗检索 (32~64 视角) + FoundationPose / Nvdiffrast 微调  │
│ - 产出: 相机外参 [R | t]、内参 (FOV/焦距)                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ [Gate 1]: Silhouette IoU >= 0.88
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: 同位多通道与微距离屏渲染 (Isotopic Multi-Channel Rendering)    │
│ - 引擎: fulgur C++20 无头服务器 / 软光栅化器                           │
│ - 产出: RGB 图、FGPP 几何通道 (WorldNormal, LinearDepth, Wireframe, AO)  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: 多级视觉残差解耦比对 (Multi-level Residual & Decoupling)       │
│ - 比对: 几何剪影 XOR、DINOv2 Patch 距离、法线角偏差、CIEDE2000 色差    │
│ - 产出: 全局相似度分值 + 2D 异常残差热力图 (Heatmap)                    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ [Gate 2]: 全局得分临界判定 (若存在异常热区)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 4: 动态微距视锥放大与 Visual Jev 强类型裁决 (Macro Jev QA)       │
│ - 动作: 虚拟微距相机推近聚焦 (Zoom-in), 抓取 512x512 高倍局部 Patch    │
│ - 判决: Visual Jev 批量执行强类型问答, 输出 Calibrated Confidence       │
│ - 投影: 结合深度图 Unproject 回溯三维空间, 生成 3D BoundingBox / Mask   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 5: 缺陷分流、精修编排与数据导出 (Refinement & Data Flywheel)     │
│ - 自动修复: Trellis-Refine / Inpainting 局部网格与贴图生成              │
│ - 人工回退: 超过阈值触发 RFC 0044 15 分钟熔断, 导出 CommandIR 质检报告 │
│ - 数据回收: 沉淀为 RFC 0020 IntentGroup 五元组与 RFC 0021 PRM 训练样本  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 分阶段评价机制与量化指标库 (Quantitative Metrics Suite)

流水线的每一个流转节点均受量化指标约束，禁止通过模糊经验放行。

### 3.1 Stage 1：姿态对齐度指标（评价机位是否严密对齐）

在比对 3D 模型与参考图之前，必须确保虚拟相机外参与内参已收敛到参考机位，否则像素比对将产生假阳性误报。

| 指标字段 (Key) | 数学定义 / 算法 | 目标及格阈值 | 物理与业务意义 |
|:---|:---|:---:|:---|
| `silhouette_iou` | $\frac{\|M_{\text{ref}} \cap M_{\text{render}}\|}{\|M_{\text{ref}} \cup M_{\text{render}}\|}$ | $\ge 0.88$ | **核心自监督对齐门禁**。通过二值化前景 Mask 的交并比检验轮廓重合度，无需先验 3D 位姿真值。 |
| `add_s_norm` | $\frac{1}{m} \sum_{x \in \mathcal{M}} \min_{y \in \mathcal{M}} \|(Rx+t) - (R_{\text{gt}}y+t_{\text{gt}})\|$ | $\le 0.05 \cdot d_{\text{obj}}$ | 归一化平均点距离（带对称性）。位姿偏差在模型外接球直径的 5% 以内。 |
| `reprojection_error_px` | $\sqrt{\frac{1}{K}\sum_{k=1}^K \|p_k^{\text{ref}} - \pi(P_k, K, [R\|t])\|^2}$ | $\le 4.0\text{ px}$ (1K) | 2D 骨骼关节/特征关键点的重投影均方根像素漂移。 |

### 3.2 Stage 2 & 3：全局跨模态比对指标（评价几何轮廓与材质解耦）

在相同位姿渲染后，系统对多通道信息进行分层解耦比对：

| 指标分类 | 指标字段 (Key) | 算法与定义 | 目标及格阈值 | 作用与质检维度 |
|:---|:---|:---|:---:|:---|
| **语义与轮廓** | `dino_cosine_sim` | $\frac{\phi(\mathbf{I}_{\text{ref}}) \cdot \phi(\mathbf{I}_{\text{render}})}{\|\phi(\mathbf{I}_{\text{ref}})\| \|\phi(\mathbf{I}_{\text{render}}\|}$ (DINOv2 ViT-L/14) | $\ge 0.85$ | 衡量资产的高层语义、部件拓扑与风格一致性，对光照与阴影变化天然鲁棒。 |
| **感知保真度** | `lpips_vgg` | 深层感知特征距离加权和 | $\le 0.15$ | 评估视觉失真度，较 PSNR/SSIM 更符合人类视觉审美容差。 |
| **表面曲面法线** | `normal_angular_error_deg` | $\frac{1}{N}\sum_{p=1}^N \arccos(\mathbf{n}_{\text{ref}}(p) \cdot \mathbf{n}_{\text{render}}(p))$ | $\le 15.0^\circ$ | **物理表面朝向门禁**。若法线夹角过大，表明存在面法线反转、平滑组缺失或几何凹陷。 |
| **固有色与材质** | `albedo_delta_e00` | CIEDE2000 色差公式计算解耦后的 BaseColor | $\le 2.5$ | 剥离直接光照影响，直接度量漫反射反照率贴图的真实色差。 |

### 3.3 Stage 4：局部微距高频与 Jev 强类型决策指标

当全局指标落入临界区间（Marginal Zone），系统自动调用虚拟微距相机推近放大，触发局部精检：

| 指标字段 (Key) | 计算方式 | 门禁阈值 | 捕获的典型 3D 质量问题 |
|:---|:---|:---:|:---|
| `patch_lpips` | 在 $512\times512$ 局部特写切片上单独计算的 LPIPS | $\le 0.12$ | 局部纹理过度模糊（Over-smoothing）、高频噪点伪影。 |
| `seam_discontinuity_deg` | 沿 UV 接缝或网格拓扑缝合线两侧法线的夹角跳跃均值 | $\le 5.0^\circ$ | 贴图拼缝露底露白、法线贴图接缝断层（Seam Discontinuity）。 |
| `surface_roughness_var` | 局部微表面离散曲率的方差梯度 $\text{Var}(\kappa_{\text{mean}})$ | $\le 0.08$ | 异常几何起伏、未平滑的阶梯网格、点云噪波凹坑。 |
| `jev_confidence_clean` | Visual Jev 直接从 Logits 读取的“无缺陷”校准概率 | $\ge 0.85$ | 解剖学粘连（多指/粘肢）、非自然生物穿模、浮动几何碎块。 |

### 3.4 Stage 5：精修前后往返保真度指标（评价修复质量增益）

精修模块介入后，必须进行往返差异对比，确保在修补缺陷的同时没有破坏资产整体基底：

| 指标字段 (Key) | 数学公式 / 算法 | 门禁约束 | 质检意义 |
|:---|:---|:---:|:---|
| `delta_fscore_1mm` | $\text{F-Score@1mm}_{\text{after}} - \text{F-Score@1mm}_{\text{before}}$ | $> 0$ | 几何精度净增益。模型表面距离参考目标 $1\text{mm}$ 内的双向点云覆盖率必须提升。 |
| `volume_breathing_pct` | $\frac{\|V_{\text{after}} - V_{\text{before}}\|}{V_{\text{before}}} \times 100\%$ | $\le 3.0\%$ | **体积呼吸率红线**。防止精修模型在局部修补时导致资产全身膨胀或坍缩。 |
| `non_manifold_edges` | 拓扑半边数据结构中共享面不等于 2 的边总数 | $\equiv 0$ | **物理硬门禁**。精修后产物必须严格满足 2-流形网格标准。 |

---

## 4. 强化学习与训练数据采集闭环 (RL Data Flywheel)

系统将每一次质检发现缺陷、调度微距相机、执行精修的过程，转化为标准化强化学习训练样本。

### 4.1 MDP 强化学习形式化映射

| 强化学习要素 | 3D 质检与精修管线中的物理实体 | 对应的系统数据结构 |
|:---|:---|:---|
| **状态 ($S_t$)** | 当前 3D 网格、FGPP 15 通道多视点渲染图、局部微距特写 Patch、Jev 缺陷预测向量 | `InspectionState { meshSnapshot, fgppBuffers, macroPatches, defectReports }` |
| **动作 ($A_t$)** | 精修操作序列：修复工具选取、修复 Mask 锚定、生成式 Inpainting 参数 | `CommandIR[]` (如 `mesh_relax`, `texture_inpaint`, `quad_remesh`) |
| **转移 ($P(S_{t+1}\|S_t, A_t)$)**| 几何算子变形与重新离屏渲染生成的新状态 | 执行 `CommandIR` $\to$ 更新 Mesh $\to$ `fulgur` 重新渲染 |
| **奖励 ($R_t$)** | 多源稠密复合奖励：几何指标增益 + Jev 判定增益 + 人工采纳/撤销信号 | $R_t = \Delta \text{Metric} + \Delta P_{\text{jev}} + R_{\text{human}}$ |

### 4.2 对接 RFC 0020 & RFC 0021 数据协议

#### 1. 采集为 `IntentGroup` 事务五元组 (RFC 0020)
在 3D 编辑器或无状态 Worker 中，一次自动/半自动精修被打包为一个不可变的 `IntentGroup`：

```json
{
  "intent_group_id": "ig_3d_repair_20261111_0042",
  "workspace_id": "ws_char_hero_v3",
  "source": "agent_auto_repair",
  "prompt": "修复左前臂贴图拉伸与局部网格褶皱",
  "operator": {
    "kind": "agent",
    "id": "agent_mesh_refiner_01",
    "model_ref": {
      "urn": "internal/trellis-refine:v2.1",
      "provider": "synapse",
      "model_id": "trellis-refine",
      "tier": "custom"
    }
  },
  "scene_before": {
    "mesh_hash": "sha256:4a8b7c9e...",
    "defect_anchor_3d": [0.35, 1.12, -0.05],
    "macro_camera_pose": { "eye": [0.45, 1.20, 0.15], "target": [0.35, 1.12, -0.05], "fov": 30.0 }
  },
  "command_ids": ["cmd_001_mask_select", "cmd_002_quad_remesh", "cmd_003_tex_inpaint"],
  "scene_after": {
    "mesh_hash": "sha256:9f2c1d3e...",
    "volume_delta_pct": 0.42
  },
  "multi_axis_rating": {
    "overall": 5,
    "spatial_accuracy": 5,
    "aesthetic_quality": 4,
    "efficiency": 5
  }
}
```

#### 2. 细粒度 `StepAnnotation` 支持 PRM (RFC 0021 §决议 5)
对于由多步 `CommandIR` 组成的复杂修复，系统对每一步记录原子标注：
* `cmd_001 (mask_select)`: `correctness = "+"`
* `cmd_002 (quad_remesh)`: `correctness = "+"`
* `cmd_003 (tex_inpaint)`: `correctness = "+"`
直接导出为 **过程奖励模型（Process Reward Model, PRM）** 的 step-by-step 训练数据。

#### 3. 隐式负反馈自动构建 DPO 偏好对 (RFC 0021 §决议 7)
* 若算法模型执行自动修复后，美术在编辑器中按下了 `Ctrl+Z`（Undo），或人工手动重新捏塑：
* 系统自动捕获 `ImplicitFeedback.undo_triggered`，将模型修复的产物标记为 `rejected`，将人工最终修改的产物标记为 `chosen`，零人工标注成本沉淀为高质量 **DPO 训练对**。

---

## 5. 核心数据模型契约 (Pydantic / TypeScript Schemas)

所有接口与序列化格式严格使用类型安全模型定义。

### 5.1 缺陷检测报告数据契约 (`InspectionReport`)

```python
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

class CameraPose(BaseModel):
    eye: Tuple[float, float, float]
    target: Tuple[float, float, float]
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 45.0

class DefectAnnotation3D(BaseModel):
    defect_id: str = Field(..., description="唯一缺陷 ID，格式 DEF-{seq}")
    category: Literal[
        "non_manifold",
        "mesh_crumple",
        "texture_stretch",
        "seam_tearing",
        "floating_debris",
        "anatomical_distortion"
    ]
    severity: float = Field(..., ge=0.0, le=1.0, description="严重程度，0 为轻微，1 为阻断")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Jev 校准置信度")
    center_3d: Tuple[float, float, float] = Field(..., description="三维空间中心锚点")
    bounding_radius_m: float = Field(..., gt=0.0, description="三维缺陷包围球半径(米)")
    affected_vertices_count: int = Field(..., ge=1)
    macro_camera: CameraPose = Field(..., description="微距特写机位参数")
    macro_crop_uri: str = Field(..., description="局部微距渲染切片路径")
    suggested_action: Literal["auto_remesh", "auto_inpaint_texture", "manual_hitl"]

class InspectionReport(BaseModel):
    schema_version: str = "2.0.0"
    asset_id: str
    timestamp_utc: int
    alignment_metrics: dict = Field(..., description="包含 silhouette_iou, reprojection_error 等")
    global_metrics: dict = Field(..., description="包含 dino_cosine_sim, lpips, normal_angular_error 等")
    defects: List[DefectAnnotation3D] = Field(default_factory=list)
    overall_passed: bool
    requires_hitl: bool = False
```

---

## 6. CLI 命令行与容器执行规范 (Stateless CLI Specification)

遵循 `synapse` 无服务端铁律，质检与修复作为无状态 CLI 命令运行在容器内。

### 6.1 常用命令清单

```bash
# 1. 单图相机位姿对齐与视锥反求
synapse inspect align \
  --ref-image assets/concept_art.png \
  --mesh assets/model.glb \
  --out-pose run/aligned_camera.json \
  --json

# 2. 同机位离屏多通道渲染与多级比对
synapse inspect eval \
  --ref-image assets/concept_art.png \
  --mesh assets/model.glb \
  --camera run/aligned_camera.json \
  --jev-model autotrust/JEV-27B-VL \
  --out-report run/inspection_report.json \
  --json

# 3. 根据质检报告自动执行微距精修
synapse inspect refine \
  --mesh assets/model.glb \
  --report run/inspection_report.json \
  --refine-backend trellis-refine \
  --out-mesh assets/model_refined.glb \
  --json

# 4. 从历史编辑器操作日志导出强化学习 (DPO/PRM) 训练数据集
synapse inspect export-rl-data \
  --journal-dir /data/journals/ \
  --format dpo \
  --out-dataset train_dpo_pairs.jsonl \
  --json
```

### 6.2 确定性退出状态码 (Exit Codes)

| 退出码 | 状态含义 | 触发场景 | `agent_fix_hint` 处理指引 |
|:---:|:---|:---|:---|
| **`0`** | 执行成功 | 质检通过，或缺陷已由精修模块成功自动修复并通过往返验证 | 无需干预，直接流入下游资产库 |
| **`10`** | 契约/参数校验失败 | 输入的 Mesh 格式损坏，或相机内参缺失 | 检查 glTF/OBJ 文件完整性与 JSON 参数格式 |
| **`20`** | 输入文件缺失 | 参考图路径不存在，或 VFS 资产哈希校验失败 | 确认输入资产已挂载并在 VFS 中建立索引 |
| **`30`** | GPU 算力故障 | 姿态估计显存不足 (OOM)，或 Jev 推理后端离线 | 降低批量视角切片尺寸，或检查 vLLM 实例健康度 |
| **`40`** | 质检熔断 / HITL 降级 | 存在重度结构缺陷且无法自动修复；或人工精修超 15 分钟 | 唤起 3D 编辑器微距机位卡片，分配技术美术手工精修 |

---

## 7. 跨仓边界澄清与主计划映射 (Phase M6: `PR-20` ~ `PR-23`)

> [!IMPORTANT]
> **与 `vg-core`（NG-3）及 `RFC 0053`（Agentic 3D Repair Harness）的去重边界**：
> - 密集几何拓扑校验（非流形边、半边流形度、BVH 距离、四边面重拓扑、足底零滑移 IK）严格委托 **`vg-core` C++20/CUDA** 经 `synapse.bridge.vg_bridge` 执行，严禁在 `synapse.inspect` 中用 Python 重写几何网格循环；
> - 交互式 MCP 修复沙箱（`RepairSandbox`）与 8 机位摄影棚评测归属 `forgekit` / `grand-studio` / `asset-forge`（RFC 0053）；`synapse.inspect` 聚焦于 **DINOv2 姿态反求包装、微距视锥规划、Visual Jev 校准 Logits 强类型裁决、以及 RFC 0020/0021 DPO/PRM 训练数据导出**。
> - 详细工程拆解见 [docs/plans/PLAN-M6-M7-INSPECTION-AND-RSI-FLYWHEEL.md](plans/PLAN-M6-M7-INSPECTION-AND-RSI-FLYWHEEL.md)。

| 原子 PR | 交付模块 | 核心成果 | 依赖项 | 预估工时 |
|:---:|:---|:---|:---|:---:|
| **`PR-20`** | `synapse.inspect.schemas` & `align` | `InspectionReport` v2.0 契约 + Fibonacci 球面 DINOv2 检索 + `vg.recon` 6DoF 精配准 | `PR-12`, `PR-13` | 1.5 天 |
| **`PR-21`** | `synapse.inspect.macro` | 同位多通道残差热力图计算 + 虚拟微距相机视锥规划与 2D→3D 包围球反投 | `PR-20` | 1.5 天 |
| **`PR-22`** | `synapse.inspect.jev_critic` | `Visual Jev` 共享视觉上下文客户端（校准 Logits 提取）+ `asset_inspect_3d` Worker 模式 | `PR-21`, `PR-14b` | 1.5 天 |
| **`PR-23`** | `synapse.inspect.flywheel_exporter`| `Trellis-Refine` 往返编排 + RFC 0020/0021 `IntentGroup` / `StepAnnotation` / Undo-DPO JSONL 导出 | `PR-12`, `PR-22` | 1.5 天 |
