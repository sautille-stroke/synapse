# 3D 生成模型质量提升方法论与高质量训练数据工程规范
## 3D Model Quality Enhancement & Training Data Specification Guide

- **文档状态**: `Approved Technical Guide v2.0`
- **发布日期**: 2026-11-12 (v2.0: 2026-11-12 评估体系完整性审计与补全)
- **所属生态**: `sautille-stroke` / `DervGen`
- **关联子系统**:
  - `synapse`（3D CG 渲染与神经生成互操作 SDK、无状态推理 Worker、`Synapse-Bench`）
  - `fulgur`（现代 C++20 AAA 实时渲染引擎、15 通道 FGPP 离屏真值提取器）
  - `synapse-native`（C++20/CUDA 核心几何算子、四边面重拓扑、物理去光照与刚体 IK）
  - `aether`（底层数学库、几何类型与 `FGPP` 跨语言线格式 SSOT）
  - `grand-studio` / `scene-forge` / `asset-forge`（人机协同质检、白模编辑与 `CommandIR` 采集）
- **交付清单与 Review 指南**: [REVIEW_3D_EVAL_DELIVERY.md](REVIEW_3D_EVAL_DELIVERY.md)
- **关联生态规约**:
  - **RFC 0019**: `CommandIR` 序列化操作协议与离线同步机制
  - **RFC 0020**: 人机协同操作时间轴与模型训练数据闭环 (`IntentGroup`, `OperatorIdentity`)
  - **RFC 0021**: 模型训练数据采集完整性与跨系统对齐 (`MultiAxisRating`, `StepAnnotation`)
  - **RFC 0039**: 资产解耦与自洽作业闭包 (`JobBundle`)
  - **RFC 0044**: 神经-物理混合工业流水线、FGPP 与人机协同 (HITL 15 分钟熔断与白模粗糙度下限)
  - **RFC 0058**: OpenUSD Hydra、MaterialX / OpenPBR 1.0 与 ACEScg 跨平台架构

---

## 1. 概述与核心痛点

随着生成式 3D 模型（如 Trellis、Hunyuan3D、TripoSR、LGM、Rodin 以及各类 Diffusion/Flow-Matching 3D 架构）的快速发展，**“单图/文本一键生成 3D 资产”** 在宏观粗模形态上已经达到极高的初稿可用性。然而，在真正导入下游游戏引擎（Unreal/Unity/Fulgur）、电影视效或具身仿真等生产级渲染管线时，直接生成的模型普遍存在严重的**质量断层（Quality Gap）**。

本文档系统性梳理：
1. **生成 3D 模型效果不佳的病理归因**（它到底坏在哪里？）；
2. **多层递进的质量提升方法论**（从零成本的确定性规则清洗，到局部可微反演精修，再到推理先验约束与参数微调）；
3. **高品质 3D 训练与微调数据工程规范**（模型进化到底需要怎样的数据？格式、通道、分布与标注契约是什么？）；
4. **工业落地路线图与验收自查清单**。

---

## 2. 3D 模型生成缺陷全景病理诊断

解决模型质量问题的前提是对缺陷进行**解耦与量化诊断**。生成式 3D 资产的常见缺陷可以划分为四个物理层级：

```
Level 4: 材质与外观层 (PBR Material & Textures)
  └── 烘焙假高光/阴影、贴图严重模糊、UV 撕裂拉伸、UV 接缝露白 (Seam Discontinuity)
        ▲
Level 3: 宏观语义与解剖学层 (Macro Semantics & Anatomy)
  └── 肢体多指粘连、无铰接结构、背面信息缺失引发的幻觉 (Hallucination)、比例失调
        ▲
Level 2: 表面微观与微分几何层 (Differential Surface Geometry)
  └── 面部/关键凹陷坍塌、法线反转/混杂、高频过平滑 (Over-smoothing)、Marching Cubes 阶梯波纹
        ▲
Level 1: 几何流形与拓扑层 (Geometric Manifold & Topology)
  └── 非流形边/顶点 (Non-manifold)、网格破洞开裂、自穿模、浮动游离碎片 (Floating Bits)、内部隐藏废面
```

### 2.1 缺陷分类与判定指标

| 缺陷层级 | 典型症状 | 根本成因 | 量化检测手段与指标 |
|:---|:---|:---|:---|
| **1. 几何流形层** | 浮动游离碎片、非流形边、三角面未缝合（非闭合水密）、网格内部残留体素废面 | 体素隐式场提取算法（如 Marching Cubes/Dual Contouring）在低密度或截断边界处的插值伪影 | 欧拉示性数 $\chi = V - E + F \neq 2(1-g)$、`non_manifold_edges > 0`、连通分支计数 $>1$ |
| **2. 表面微观层** | 面部五官塌陷、局部表面法线反转（凹凸反向）、过度平滑丢失机械边缘、表面凹坑噪波 | 隐式场梯度估算不准、SDF 零等值面正则化权重过低，或高频几何惩罚不足 | 法线角偏差均值 $\theta_{\text{normal}} > 15^\circ$、离散平均曲率方差 $\text{Var}(\kappa_{\text{mean}}) > 0.08$ |
| **3. 宏观语义层** | 手指粘连成掌、双腿粘连成裙、后脑勺或背部瞎猜结构、缺乏关节铰接拓扑 | 单目输入导致的背部严重遮挡（Ambiguity）、模型缺乏人体/通用刚体解剖学骨骼先验 | 剪影交并比 $\text{Silhouette IoU} < 0.88$、重投影漂移 $\text{RMSE} > 4.0\text{ px}$ |
| **3.1 毛发微观层** | 发丝黏连为塑料硬块（“头盔发”）、发际线浮空/穿颅、无发流方向感、无各向异性散射 | 通用 Marching Cubes 等值面包络一维曲线纤维失效、模型缺少 1D Strand 导引线先验与 Marschner BSDF | 发丝分缕度 $\text{StrandSeparation} < 0.7$、发流切线方差 $\text{Var}(\theta) > 0.08$ |
| **3.2 服饰解耦层** | 衣物与身体焊死（“衣肉一体”）、无内部躯干支撑、肢体弯曲时网格严重拉扯撕裂、内部穿模 | 缺少分层数字人（Layered Human）解耦架构，未区分 SMPL-X 裸模碰撞代理与外部 3D/2D 裁片壳体 | 身体穿透率 $\text{PenetrationRate} > 0.0\%$、最小碰撞间隙 $\delta < 2.0\,\text{mm}$ |
| **4. 材质与外观层** | BaseColor 贴图带有环境阴影与高光反射、UV 展开畸变拉伸严重、贴图拼缝处有白边/黑缝 | 缺少物理去光照（Delighting）、未导出标准 PBR 材质贴图（只有单张有光照扩散图）、UV 投影重叠 | CIEDE2000 色差 $\Delta E_{00} > 2.5$、UV 面积形变比率 $\sigma_{\text{uv}} > 1.5$、接缝法线跳跃角 $>5^\circ$ |

---

## 3. 四阶递进式质量提升方法论

针对上述病灶，质量提升应遵循**“低成本开箱即用后处理 $\to$ 局部针对性可微精修 $\to$ 推理期先验引导 $\to$ 终极自演进模型对齐”**的四阶体系。

```mermaid
flowchart TD
    subgraph Tier1 ["Tier 1: 确定性几何规则清洗 (开箱即用，0 显存/0 训练成本)"]
        Raw["生模原始 Mesh"] --> Filter_Islands["1. 连通域碎片过滤<br/>(剔除 <0.5% 体积游离噪点)"]
        Filter_Islands --> Watertight["2. 泊松/环绕数水密修复<br/>(封堵空洞，消除非流形边)"]
        Watertight --> Remesh["3. 四边面重拓扑 (Quad Remesh)<br/>(消除乱网格，构建规则布线)"]
        Remesh --> UV_Pack["4. 自动 UV 分缝与松弛打包<br/>(XAtlas/UV-Packer 消除拉伸)"]
    end

    subgraph Tier2 ["Tier 2: 视锥微距聚焦与可微光栅化反演 (局部针对性精修)"]
        UV_Pack --> Heatmap["1. 多视角残差比对与热力图定位<br/>(LPIPS + DINOv2 Patch 残差)"]
        Heatmap --> Macro_Zoom["2. 虚拟微距相机聚焦 (Macro Zoom-in)<br/>(截取 512x512 高倍局部 Patch)"]
        Macro_Zoom --> Inpainting["3. 2D/3D 生成式局部重绘<br/>(消除粘连与修复塌陷)"]
        Inpainting --> Diff_Raster["4. 可微光栅化反演优化 (Nvdiffrast)<br/>(基于参考图反向传播微调顶点位移与反照率)"]
    end

    subgraph Tier3 ["Tier 3: 推理期物理先验注入与测试时引导 (生成前/生成时治理)"]
        Diff_Raster --> Physics_Rig["1. 骨骼与刚体约束注入<br/>(SMPL-X / Humanoid-53 先验，足底零滑步)"]
        Physics_Rig --> Rejection["2. 物理与视觉双重判别器 Best-of-N<br/>(多候选采样，量化指标择优淘汰)"]
    end

    subgraph Tier4 ["Tier 4: 递归自我提升 (RSI) 与偏好对齐微调 (根本性能力重塑)"]
        Rejection -.->|"沉淀失败/修复对 (Rejected / Chosen)"| Pair_Store["偏好样本库 D_pair"]
        Pair_Store --> LoRA_Train["DPO / Flow-Matching 偏好微调<br/>(Wan2.1 / Trellis 权重升级)"]
        LoRA_Train -.->|"晋升正式基线权重"| Raw
    end
```

---

### 3.1 Tier 1：确定性几何规则自动化清洗流水线

此阶段不依赖大模型，运行速度极快（毫秒至数秒级），能够直接干掉 80% 的基础工业级几何报错：

1. **连通域体积过滤（Connected Component Cleaning）**：
   - 遍历网格三角面图，切分所有独立连通子集（Components）；
   - 计算各子集的外接球半径与封闭体积，**强制剔除体积小于主物体 $0.5\%$ 的孤立浮动碎块**。
2. **水密化与流形性修复（Watertight & Manifold Healing）**：
   - 基于**快速环绕数（Fast Winding Numbers）**或隐式泊松表面重构（Screened Poisson Reconstruction），重新提取具有严格 2-流形特性的闭合外壳；
   - 消除共面重叠面、零面积退化三角面（Degenerate Faces）与 T-Junction。
3. **四边面结构化重拓扑（Adaptive Quad Remeshing）**：
   - 原生 AI 生成模型（如 Marching Cubes）产物通常是杂乱无序的三角形态，无法直接用于骨骼蒙皮；
   - 调用如 `Instant Meshes` 或 `QuadriFlow` 算法，沿主曲率场方向对齐几何流向（Edge Flow），生成等距、规整、无尖锐夹角的轻量四边面网格。
4. **参数化 UV 自动切缝与松弛（UV Parameterization & Packing）**：
   - 利用 `XAtlas` 自动按曲率拐角切分 UV 孤岛（Islands）；
   - 执行能量最小化拉伸松弛（Least Squares Conformal Maps, LSCM），保证 UV 纹素密度（Texel Density）均匀，杜绝贴图拉伸变形。

---

### 3.2 Tier 2：视锥微距聚焦与可微光栅化反演精修

针对高频细节缺失、局部结构塌陷或局部贴图模糊，采用“局部放大镜”机制：

1. **多视角残差热力图解耦（Residual Decoupling）**：
   - 利用渲染核（如 `fulgur`）将当前 3D 模型在 $K$ 个标准视角（正、背、左、右、俯、仰）下渲染出 RGB、米制线性深度与法线图；
   - 与原始输入多视角/参考图进行特征比对（DINOv2 ViT-L/14 Patch 级余弦距离 + 轮廓 XOR 残差），生成 2D 异常残差热力图 $\mathcal{H}(u, v)$。
2. **虚拟微距相机自动巡航（Macro Zoom-in & 3D Grounding）**：
   - 在残差热力图峰值区域（如手部、面部、装饰纹理），计算其局部包围球；
   - 调度虚拟微距相机推近聚焦，抓取 $512\times512$ 高倍局部切片（Patch Crop）；
   - 结合深度图反投影（Unproject），将 2D 缺陷边界精确映射为 3D 网格顶点掩码（Vertex Mask）。
3. **可微光栅化几何反向反演（Differentiable Inverse Rendering）**：
   - 在局部遮罩区域内，基于 `Nvdiffrast` 构建端到端可微渲染管线；
   - 保持网格拓扑不变，以原始参考图的高分辨率特征为优化目标，以法线平滑度（Laplacian Regularizer）为物理先验，反向传播微调顶点位移偏移量（Displacement Offset）$\Delta \mathbf{V}$：
     $$\mathcal{L} = \mathcal{L}_{\text{rgb}}(\mathbf{I}_{\text{render}}, \mathbf{I}_{\text{ref}}) + \lambda_1 \mathcal{L}_{\text{lpips}} + \lambda_2 \|\Delta \mathbf{L} \mathbf{V}\|_2^2$$
4. **物理去光照与材质解耦（Delighting & Material Splitting）**：
   - 利用双向反射分布函数（BRDF）逆渲染网络，将单一贴图解耦为真正的 **漫反射反照率（BaseColor/Albedo）、法线贴图（Normal Map）与粗糙度（Roughness）**，彻底剥离烘焙在纹理上的环境假光。

---

### 3.3 Tier 3：推理期物理先验注入与测试时引导

防止模型在推理生成阶段“自由发挥”过度而产生严重畸变：

1. **骨骼刚体先验注入（Kinematic & Body Rigid Priors）**：
   - 针对人形/四足资产生成，引入参数化人体模型先验（如 SMPL-X / Humanoid-53）；
   - 强制约束关节长度对称性、脊柱曲率合理性，并在动态生成阶段施加**足底零滑动刚性约束（Zero-Slip Velocity Constraint: $\mathbf{v}_{\text{foot}} = \mathbf{0}$）**，杜绝滑步与浮空。
2. **物理验算器与 Best-of-N 拒绝采样（Dual Verifier Rejection Sampling）**：
   - 推理阶段同时并行生成 $N$ 组候选网格（如 $N=4$ 或 $8$）；
   - 由轻量离屏验算器进行无偏差快速物理排查：
     - 剪影交并比 $\text{Silhouette IoU} \ge 0.88$；
     - 水密性闭合测试（不存在开孔）；
     - 体积呼吸率（Volume Breathing）与参考粗模相比偏差 $\le 5\%$；
     - 法线一致性得分；
   - 直接剔除不及格候选，仅放行综合物理与视觉指标最高的样本。

---

### 3.4 Tier 4：递归自演进飞轮与参数微调（RSI & PEFT）

要从根本上让生成大模型“越来越聪明、不再犯同样的错误”，必须将质检与修复过程转化为**强化学习与微调数据飞轮**：

1. **构建三元组偏好对（Preference Triplet Construction）**：
   - 将模型生成的有缺陷初始网格记为负样本 $\mathbf{y}_{\text{rejected}}$；
   - 经过 Tier 1 自动算子清洗或美术轻量人工精修后的合格网格记为正样本 $\mathbf{y}_{\text{chosen}}$；
   - 组成偏好对：$\mathcal{D}_{\text{pair}} = \{(\mathbf{x}_{\text{prompt/cond}}, \mathbf{y}_{\text{rejected}}, \mathbf{y}_{\text{chosen}})\}$。
2. **DPO / Flow-Matching 偏好对齐训练**：
   - 在生成模型（如 Wan 2.1 DiT、Trellis 隐式骨干网络）上挂载 LoRA 参数模块；
   - 使用直接偏好优化（Direct Preference Optimization, DPO）对齐损失函数，拉大正负样本在隐空间中的概率密度差，使模型在单步采样时主动避开致错模式。
3. **单调演进安全门禁（Monotonic Evolution Gate）**：
   - 微调后模型必须通过全套基准套件（如 `Synapse-Bench`）测试；
   - 确保核心质量分值（如 F-Score@1mm、Silhouette IoU、LPIPS）单调提升，且在非微调类别上出现性能退化（Regression）时自动熔断回滚。

---

## 4. 高品质 3D 训练与微调数据工程规范

提升 3D 模型质量，**决定模型上限的是数据质量，而非算法网络结构的微小微调**。优质的 3D 生成训练数据必须满足以下四大核心维度契约。

---

### 4.1 多通道物理几何解耦数据标准（超越纯 RGB）

纯 RGB 图像存在严重的光照-几何单义歧义性（Illumination Ambiguity）。用于现代 3D 生成、条件对齐与精修模型训练的数据集，**必须具备像素级绝对对齐的多通道物理缓冲层**。

系统统一遵循 **FGPP v1.0**（`@aether//fgpp` SSOT）15 标量通道规范（`34 B/px`）：

```
FGPP 15 通道 OpenEXR 标准布局 (34 B/px, IEEE 754, 与 @aether//fgpp 100% 对齐):
┌───────────────────────────┬──────────────┬───────────────┬───────────────────────────────────────┐
│ 通道名称 (ASCII 排序)      │ 数据类型     │ 字节数        │ 物理含义与技术约束                    │
├───────────────────────────┼──────────────┼───────────────┼───────────────────────────────────────┤
│ BaseColor.R / G / B       │ HALF (fp16)  │ 6 B (3×2 B)   │ 物理去光照的真实漫反射反照率 (Albedo) │
│ CameraLinearDepth.Z       │ FLOAT (fp32) │ 4 B           │ 绝对米制相机空间线性深度 (严禁插值)   │
│ Cryptomatte.ID            │ UINT32       │ 4 B           │ 实体/材质独立整型索引 (严禁插值滤波)  │
│ DirectRadiance.R / G / B  │ HALF (fp16)  │ 6 B (3×2 B)   │ 直接光照线性辐射亮度 (ACEScg)         │
│ MotionVector.X / Y        │ HALF (fp16)  │ 4 B (2×2 B)   │ 像素级屏幕空间前向运动矢量            │
│ Roughness                 │ HALF (fp16)  │ 2 B           │ 感知粗糙度 [0, 1]                     │
│ Specular                  │ HALF (fp16)  │ 2 B           │ 电介质镜面反射率 F0 缩放              │
│ WorldNormal.X / Y / Z     │ HALF (fp16)  │ 6 B (3×2 B)   │ 世界空间归一化表面朝向向量 [-1, 1]    │
├───────────────────────────┼──────────────┼───────────────┼───────────────────────────────────────┤
│ 合计: 15 个标量通道        │ —            │ 34 B / px     │ Metallic 等扩展属性由资产 PBR 贴图承载 │
└───────────────────────────┴──────────────┴───────────────┴───────────────────────────────────────┘
```

#### 各通道的核心作用与训练价值：
1. **米制线性深度 (`CameraLinearDepth.Z`)**：
   - 必须为绝对米制单位（`float32`），而非归一化非线性屏幕空间深度；
   - 价值：消除多视角尺度不一致，给模型提供不可动摇的绝对距离基准，彻底消除透视塌陷。
2. **表面法线 (`WorldNormal.XYZ`)**：
   - 世界空间归一化向量（相机空间法线由 `metadata.json` 列主序外参矩阵无损变换可得，无需冗余存储），表征微表面朝向与微分几何特征；
   - 价值：让模型直接感知高频凹凸细节，约束几何法线不可反转、五官凹凸结构不可坍塌。
3. **物理去光照漫反射反照率 (`BaseColor.RGB`) 与直接辐射度 (`DirectRadiance.RGB`)**：
   - `BaseColor` 彻底剥离方向光、点光源高光与投射阴影，与 `DirectRadiance`、`Roughness`、`Specular` 组成能量自洽的内蕴分解监督对；
   - 价值：防止模型把阴影误学生长成黑斑，把高光误学成浮雕突起。
4. **实体掩码 (`Cryptomatte.ID`)**：
   - 无滤波的无符号整型 ID（`UINT32`），区分不同材质与独立骨骼部件（如头发、衣服、皮肤、手部）；
   - 价值：作为注意力掩码（Spatial Attention Mask），防止跨资产语义粘连（如衣服与肢体融合）。

---

### 4.2 物理流形真值 3D 资产入库准入门禁

作为 Ground Truth（真值）训练集进入数据湖的 3D 资产，必须通过以下自动化前置门禁扫描：

```mermaid
flowchart LR
    Asset["待入库 3D 资产"] --> Gate_Topo{"门禁 1: 拓扑流形度"}
    Gate_Topo --"存在非流形边/孔洞"--> Reject1["驳回清洗"]
    Gate_Topo --"Pass: 2-Manifold"--> Gate_PBR{"门禁 2: PBR 解耦度"}
    Gate_PBR --"存在烘焙阴影/单图"--> Reject2["驳回重烘焙"]
    Gate_PBR --"Pass: OpenPBR"--> Gate_Scale{"门禁 3: 尺度与轴向"}
    Gate_Scale --"中心漂移/非米制"--> Reject3["自动重对齐"]
    Gate_Scale --"Pass"--> Accept["准予入库归档"]
```

1. **流形与水密性门禁（Manifold & Watertight Gate）**：
   - 严格满足 2-流形网格标准：每条边被且仅被 2 个面共享，无共享顶点的非流形扇区；
   - 封闭水密（Watertight）：无外部漏光孔洞，可通过射线法稳定计算闭合符号距离场（SDF）；
   - 零退化面：无面积小于 $10^{-6}$ 的简并三角形，无零长度边。
2. **材质规范门禁（PBR Specification Gate）**：
   - 必须严格遵循 `MaterialX` / `OpenPBR 1.0` 或 Disney 原则；
   - 贴图分辨率满足幂次规格（$1024\times1024$ 或 $2048\times2048$）；
   - 纹理贴图必须完备：必须具备独立 Albedo、Normal（DirectX/OpenGL 标注明确）、Roughness、Metallic 贴图。
3. **几何尺度与坐标系标准化门禁（Scale & Alignment Gate）**：
   - **坐标轴对齐**：全库统一坐标系（如右手系 Y-up，Z-forward）；
   - **中心锚定**：资产包围盒底面中心点归零至原点 $(0, 0, 0)$；
   - **绝对物理尺寸标注**：包含元数据标明真实世界尺寸（如人物高度标注为 $1.75\text{m}$，汽车长度标注为 $4.8\text{m}$），严禁无尺度乱缩放。

---

### 4.3 稠密多视角球面与多尺度特写机位方案

仅靠单张图或正交 4 视角不足以训练出高鲁棒性模型。标准数据集采集必须包含**宏观球面机位 + 局部微距机位**：

```
                             [北极 俯角 85°]
                                   ▲
                                   │
              [中纬度仰角 30°] ─── 球面螺旋环绕 (32~64 机位)
                                   │
                                   ▼
                             [赤道视角 0°]
                                   │
                                   ▼
                             [南极 仰角 -60°]
                                   
      同时采集: 针对关键部件 (手部/面部/徽标) 的 4~8 个微距局部 Patch
```

1. **宏观球面多视角覆盖（Spherical Fibonacci Trajectory）**：
   - 采用斐波那契球面采样算法，在围绕资产包围球外围均匀分布 32～64 个渲染机位；
   - 俯仰角覆盖范围 $-60^\circ \sim +85^\circ$，保证头顶与脚底死角无遗漏；
   - 相机参数显式记录：内参矩阵 $\mathbf{K}$（FOV、焦距、主点）与外参矩阵 $[\mathbf{R} | \mathbf{t}]$ 严格锁死为 64-bit 浮点数。
2. **多尺度微距局部特写切片（Multi-Scale Macro Patches）**：
   - 在高曲率与高语义密度区域（如人脸五官、手指指节、服装接缝处），以 $2\times \sim 4\times$ 虚拟长焦微距相机拍摄高分辨率特写切片；
   - 训练模型具备感知毫米级高频拓扑接缝与法线细节的能力。

---

### 4.4 人机协同与精修偏好对构建标准（DPO/PRM）

用于奖励模型（PRM）和直接偏好微调（DPO）的数据不仅需要“最终正确答案”，更需要**“错误 $\to$ 修正”的动态演进轨迹**：

$$\text{DataPoint} = \left\langle \mathbf{x}, \mathbf{y}_{\text{rejected}}, \mathbf{y}_{\text{chosen}}, \mathcal{T}_{\text{repair}} \right\rangle$$

1. **输入条件 $\mathbf{x}$**：多视角参考图、提示词或粗白模网格；
2. **劣质模型产物 $\mathbf{y}_{\text{rejected}}$**：
   - 未经过滤的原始生成网格，保留真实存在的瑕疵（如面部微塌、破面、接缝开裂）；
3. **精修合格产物 $\mathbf{y}_{\text{chosen}}$**：
   - 经过算法修复或美术精修后的高质量网格；
   - **严格约束**：$\mathbf{y}_{\text{chosen}}$ 与 $\mathbf{y}_{\text{rejected}}$ 的体积呼吸率变化必须 $\le 3.0\%$，确保修复仅发生在局部缺陷区，未改变整体造型基底；
4. **精修操作序列轨迹 $\mathcal{T}_{\text{repair}}$（遵循 RFC 0019 `CommandIR`）**：
   - 记录精修动作的每一步操作算子与参数：
     ```json
     [
       {"op": "select_vertices_by_mask", "params": {"center": [0.12, 1.45, 0.05], "radius": 0.08}},
       {"op": "laplacian_smooth", "params": {"iterations": 5, "lambda": 0.4}},
       {"op": "bridge_edge_loops", "params": {"tolerance": 0.002}},
       {"op": "quad_remesh_local", "params": {"target_edge_length_mm": 2.5}}
     ]
     ```
   - 价值：使精修 Agent 不仅能做出好坏判别，还能直接预测出针对具体缺陷的精修动作参数。

---

### 4.5 极限挑战工况难例设计（Hard Negatives & Curriculum）

模型在简单凸多面体（如杯子、苹果、规整建筑）上表现良好，但在真实工业资产中极易崩塌。数据集必须包含**主动构造的挑战性难例（Hard Cases）**：

1. **极薄壁结构（Thin-Shell Structures）**：
   - 植物树叶、纸张、金属薄刃、伞面、衣物褶皱边；
   - 挑战点：体素截断容易产生双面破损、法线正反错乱；
2. **深度镂空与复杂亏格拓扑（High-Genus & Porous Meshes）**：
   - 镂空铁艺栏杆、机械齿轮箱、链条、网筛；
   - 挑战点：Marching Cubes 易将镂空区域误封堵为实心泥团；
3. **极高反光与半透明物理属性（Specular & Transmission）**：
   - 汽车烤漆金属反光、水体、玻璃杯、玉石散射；
   - 挑战点：高光移动在不同视角下破坏一致性假设；
4. **高动态交叉与自遮挡姿态（Severe Occlusion & Entangled Poses）**：
   - 双手合十、瑜伽扭曲姿态、复杂机械折叠连杆；
   - 挑战点：自穿模与多部件粘连高发。

---

### 4.6 工业分层数字人解耦数据标准（Layered Human SSOT）

针对“毛发、衣物与身体粘连”的行业通病，生产级角色数据**严禁入库单一闭合的“一体化网格（Fused Single Mesh）”**，必须强制解耦为自底向上的三层资产组合包（Layered Asset Bundle）：

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: 独立毛发/毛皮资产 (Strand Groom / Auto Hair Cards)                             │
│   - 几何形态: 500~1500 根 3D 引导样条线 (USD/Alembic Curves) 或 LOD0-3 Hair Cards 发带  │
│   - 生长根部: 严格锚定于 Layer 1 裸模头皮 UV 区域 (Scalp Follicles)                     │
│   - 物理材质: Marschner BSDF 切线场 T、纵向粗糙度 β_R、方位角粗糙度 β_TT、黑色素 Melanin │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Layer 2: 独立分层衣物资产 (Multi-Layer Garment Shells)                                  │
│   - 几何形态: 2D 缝合裁片 (Patterns & Seams) 或 3D 双层织物壳 (Shell Thickness >= 1.5mm)│
│   - 防穿模硬门禁: 距离 Layer 1 裸模外表面间隙恒满足 Signed Distance Field Φ_body >= 2mm   │
│   - 物理材质: OpenPBR Fabric 织物散射 (微纤维 Sheen 反射 + 经纬编织法线 + 布料抗拉刚度) │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: 标准解剖裸模基底 (Underlying Neutral Body Mesh)                                │
│   - 几何形态: SMPL-X / 53-Joint 标准拓扑网格，包含精准四肢关节旋转中心与肌肉变形场       │
│   - 物理职责: 充当骨骼运动学 SSOT 与布料/毛发的刚体/弹性碰撞代理 (Collision Proxy)      │
│   - 物理材质: 标准人体皮肤多层次表面漫反射 (Skin SSS / Dipole Diffusion)                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 训练监督损失与解耦约束：
1. **毛发解耦损失（Hair Decoupling Loss）**：
   $$\mathcal{L}_{\text{hair}} = \lambda_{\text{tangent}} \|\mathbf{T}_{\text{pred}} - \mathbf{T}_{\text{gt}}\|^2 + \lambda_{\text{follicle}} \text{BCE}(M_{\text{scalp}}^{\text{pred}}, M_{\text{scalp}}^{\text{gt}}) + \lambda_{\text{curv}} \|\kappa_{\text{strand}} - \kappa_{\text{gt}}\|^2$$
2. **衣体零穿模碰撞惩罚（Zero-Penetration Penalty）**：
   $$\mathcal{L}_{\text{cloth-body}} = \sum_{\mathbf{v} \in \mathcal{V}_{\text{cloth}}} \max\left(0, \delta_{\text{clearance}} - \Phi_{\text{body}}(\mathbf{v})\right)^2$$
   确保在任何姿态生成与微调反演中，衣物顶点绝不陷入身体肉体内部。

---

## 5. 工业落地演进路径与最佳实践 Checklist

针对不同研发阶段的团队，推荐以下递进落地演进路线：

### 5.1 三阶段实施计划

| 演进阶段 | 实施周期 | 核心动作 | 预期收益 |
|:---|:---|:---|:---|
| **Phase 1: 几何硬门禁与后处理接入** | 1～2 天 | 引入连通域过滤、泊松水密修复、四边面重拓扑（`QuadriFlow`/`XAtlas`）自动化脚本 | 消除全量浮动噪点碎块、消灭非流形破面，模型达到可导入引擎的基础及格线 |
| **Phase 2: 局部可微反演与质检门禁** | 1～2 周 | 搭建多视角离屏渲染、计算残差热图、基于 `Nvdiffrast` 的顶点位移微调与 PBR 去光照解耦 | 攻克面部塌陷、消除贴图假高光/假阴影，极大提升高频细节保真度 |
| **Phase 3: 偏好数据闭环与自演进微调** | 1～2 月 | 沉淀人机协同操作对（$\mathbf{y}_{\text{rejected}}, \mathbf{y}_{\text{chosen}}$），在底模上执行 DPO / LoRA 偏好对齐训练 | 模型自身生成能力质变，从源头减少 70% 以上的常见粘连与畸变 |

---

### 5.2 3D 资产生成与数据准入自查 Checklist

在模型交付或数据入库前，逐项核对以下指标：

- [ ] **流形度检查**：`non_manifold_edges == 0`，欧拉公式满足闭合流形；
- [ ] **连通域检查**：无体积 $<0.5\%$ 的游离浮动几何残渣；
- [ ] **水密性检查**：网格闭合无破孔，射线碰撞法正反面判定完全闭合；
- [ ] **多通道解耦**：训练数据具备米制线性深度、相机/世界法线与物理去光照反照率贴图；
- [ ] **法线一致性**：多视角渲染图与几何法线夹角偏差均值 $\le 15^\circ$，无翻转法线；
- [ ] **拓扑布线规范**：四边面占比合理，主要关节弯曲处具备环形布线（Edge Loops）；
- [ ] **PBR 规范性**：材质贴图满足 BaseColor、Normal、Roughness、Metallic 独立解耦，无烘焙假阴影；
- [ ] **空间归一化**：模型底面原点对齐 $(0,0,0)$，尺度单位为真实米制；
- [ ] **微距质量**：局部特征区域在 $2\times$ 微距特写下 LPIPS $\le 0.12$，无过度模糊；
- [ ] **偏好闭环**：所有人工精修修复步骤均已被序列化记录为 `CommandIR` 沉淀入数据湖。


---

## 6. 3D 模型评估指标全景矩阵与实现状态 (Evaluation Metrics Matrix)

> **审计基线**: 2026-11-13 v2.1。基于对 `synapse/python/synapse/metrics/`、`inspect/`、`recon/`、`rsi/`、`bench/` 与 `csrc/geometry/` 全量代码及 `profiles.py` 门禁 SSOT 的交叉校验。

### 6.1 已完整实现的门禁指标（16 个核心 Profile 指标 + 4 个 P0 渲染画质指标 + 10 个辅助质检算子）

#### 6.1.1 四支柱 16 核心 Profile 门禁指标（与 `metrics/profiles.py` 100% 对齐）

以下 16 个核心指标均已有完整的计算函数实现、单元测试覆盖，并严格锁死在 4 层递进门禁 Profile（`TRACK_B_VIDEO` → `TRACK_A_ASSET` → `RSI_PROMOTION` → `RSI_PROMOTION_TARGET`）中。

| # | 支柱 | 指标 (Key) | 计算位置 | 检测维度 | TRACK_B | TRACK_A | RSI_PROMO | RSI_TARGET |
|---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| 1 | Temporal | `temporal_warp_error` | `metrics/temporal.py` | 运动矢量时域翘曲 RMSE（含遮挡掩码） | ≤ 8.0 | ≤ 4.0 | ≤ 2.0 | ≤ 2.0 |
| 2 | Temporal | `flicker_score` | `metrics/temporal.py` | 二阶拉普拉斯时域闪烁不稳定度 | ≤ 0.025 | ≤ 0.020 | ≤ 0.018 | ≤ 0.008 |
| 3 | Temporal | `fvd` | `metrics/temporal.py` | Fréchet 时序特征距离 (`fvd_feature_distance`) | ≤ 300.0 | ≤ 150.0 | ≤ 50.0 | ≤ 50.0 |
| 4 | FGPP Fidelity | `depth_rmse_m` | `metrics/fgpp_fidelity.py` | 深度图 RMSE（绝对米制） | ≤ 0.05 | ≤ 0.04 | ≤ 0.03 | ≤ 0.02 |
| 5 | FGPP Fidelity | `scale_invariant_depth_error` | `metrics/fgpp_fidelity.py` | Eigen 对数尺度不变深度误差 | ≤ 0.30 | ≤ 0.10 | ≤ 0.05 | ≤ 0.05 |
| 6 | FGPP Fidelity | `normal_angular_error_deg` | `metrics/fgpp_fidelity.py` | 法线角度均值偏差（度） | ≤ 15.0 | ≤ 12.0 | ≤ 12.0 | ≤ 5.0 |
| 7 | FGPP Fidelity | `flow_endpoint_error_px` | `metrics/fgpp_fidelity.py` | 光流端点误差（像素） | ≤ 4.0 | ≤ 2.0 | ≤ 1.0 | ≤ 1.0 |
| 8 | FGPP Fidelity | `silhouette_iou` | `metrics/fgpp_fidelity.py` | 二值化剪影交并比 | ≥ 0.85 | ≥ 0.88 | ≥ 0.90 | ≥ 0.97 |
| 9 | Roundtrip | `chamfer_mm` | `metrics/roundtrip.py` | 双向 Chamfer 距离（mm） | ≤ 8.0 | ≤ 5.0 | ≤ 4.0 | ≤ 1.0 |
| 10 | Roundtrip | `f_score_1mm` | `metrics/roundtrip.py` | 1mm 阈值双向 F-score | ≥ 0.40 | ≥ 0.70 | ≥ 0.90 | ≥ 0.90 |
| 11 | Roundtrip | `volume_breathing_pct` | `metrics/roundtrip.py` | 有符号四面体体积变化百分比 | ≤ 5.0 | ≤ 3.0 | ≤ 2.5 | ≤ 1.5 |
| 12 | Roundtrip | `foot_sliding_cm` | `metrics/roundtrip.py` | 地面接触顶点水平滑动量（cm） | ≤ 0.5 | ≤ 0.3 | ≤ 0.2 | ≤ 0.1 |
| 13 | Roundtrip | `bone_length_drift_pct` | `metrics/roundtrip.py` | 骨骼段长度漂移百分比 | ≤ 5.0 | ≤ 2.0 | ≤ 0.5 | ≤ 0.5 |
| 14 | Roundtrip | `albedo_delta_e00` | `metrics/roundtrip.py` | CIEDE2000 反照率色差（ACEScg） | ≤ 5.0 | ≤ 2.5 | ≤ 2.5 | ≤ 2.0 |
| 15 | Style | `border_ring_ciede2000` | `metrics/style_consistency.py` | 形态膨胀边界环 CIEDE2000 色差 | ≤ 3.0 | ≤ 2.5 | ≤ 2.0 | ≤ 1.5 |
| 16 | Style | `light_dir_error_deg` | `metrics/style_consistency.py` | 主光方向角度误差（度） | ≤ 20.0 | ≤ 10.0 | ≤ 5.0 | ≤ 5.0 |

#### 6.1.2 P0 渲染画质核心指标与辅助质检算子（均已完整实现 ✅）

| 算子 / 指标 | 位置 | 功能与场景门禁支持 |
|:---|:---|:---|
| `psnr_db` | `metrics/image_quality.py` | 峰值信噪比 $10 \log_{10}(\text{MAX}^2 / \text{MSE})$，支持 `Synapse-Bench` `scenario_overrides` 与 3D 多视角评估 |
| `ssim` | `metrics/image_quality.py` | Wang et al. 2004 结构相似度（$11 \times 11$ 高斯窗），支持全部 T1–T5 场景门禁 |
| `ciede2000` | `metrics/image_quality.py` | 全图/掩码区域像素级精确 CIEDE2000 感知色差均值 |
| `lpips` | `metrics/image_quality.py` | 多尺度感知结构/色度块距离（零重型依赖确定性 CPU 实现，支持 T1–T5 门禁） |
| `tone_curve_rmse` | `metrics/style_consistency.py` | 亮度分位数色调映射曲线 RMSE |
| `histogram_intersection` | `metrics/style_consistency.py` | 归一化颜色直方图交集相似度 |
| `validate_mesh_geometry` | `inspect/geometry_gate.py` + `csrc/geometry/` | Tier 0 双轨几何硬门禁：2-流形（含扇面单伞连通检测）、水密性、退化/针状面索引、共面 2D SAT + Möller–Trumbore 自穿插对、连通分量与浮空碎块过滤 |
| `compute_silhouette_iou` | `inspect/align.py` | Z-buffer 软光栅化 + IoU 对齐 |
| `compute_isotopic_residual_heatmap` | `inspect/macro.py` | 多通道（RGB+深度+法线）异常融合热力图 |
| `VisualJevCritic` | `inspect/jev_critic.py` | VLM Logprob 6 类缺陷校准概率检测 |
| `DualVerifierEngine` | `rsi/dual_verifier.py` | 物理 + 神经双重验算引擎 |

### 6.2 设计文档已规约、后续演进待扩展的指标（P1/P2 Backlog）

以下指标已完成设计规约，排入后续迭代路线图（其中 P0 图像画质指标与 Tier 0 自包含双轨几何核已在 v2.1 落地）：

#### 6.2.1 感知与预训练骨干语义指标（需可选 GPU 权重）

| 指标 (Key) | 算法规格 | 目标位置 | 门禁阈值 | 行业对标 |
|:---|:---|:---|:---|:---|
| `dino_cosine_sim` | DINOv2 ViT-L/14 CLS Token 余弦相似度 | `metrics/perceptual.py` (待新建) | ≥ 0.85 | Zero123++, LGM, TripoSR 标配 |
| `lpips_vgg` | 预训练 VGG-16 特征骨干 LPIPS（补充现有轻量确定性 `lpips`） | `metrics/perceptual.py` (待新建) | ≤ 0.15 | GSO Benchmark, DreamFusion |

#### 6.2.2 P1 级：拓扑量化指标（已实现连通域/退化面/自穿插计数，待补欧拉特征与非流形边计数导出）

| 指标 (Key) | 算法规格 | 实现位置与状态 | 门禁阈值 |
|:---|:---|:---|:---|
| `connected_components_count` | 共享边面图并查集连通域计数 | ✅ `csrc/geometry` & `recon/geometry.py` (`find_connected_components`) | $= 1$ (单一主体) |
| `floating_islands_ratio` | 浮空碎块面占比阈值判定 | ✅ `csrc/geometry` & `recon/geometry.py` (`detect_floating_islands`) | $< 0.005$ |
| `degenerate_faces_count` | 面积 $< 10^{-8}$ 或高宽比 $> 100:1$ 的面数量 | ✅ `csrc/geometry` & `recon/geometry.py` (`check_degenerate_faces`) | $= 0$ |
| `self_intersecting_pairs` | AABB 树 + 共面 2D SAT + Möller–Trumbore 非邻接穿插对 | ✅ `csrc/geometry` & `recon/geometry.py` (`check_self_intersections`) | $= 0$ |
| `euler_characteristic` | $\chi = V - E + F$，验证 $\chi = 2(1-g)$ | 🟡 待扩展 (`recon/geometry.py`) | $= 2$ (genus-0 封闭网格) |
| `non_manifold_edges_count` | 共享面 $> 2$ 或扇面非单伞的精确计数导出 | 🟡 待扩展（布尔判定 `check_manifold` 已实现） | $= 0$ |

#### 6.2.3 P1 级：UV 与贴图质量指标

| 指标 (Key) | 算法规格 | 目标位置 | 门禁阈值 |
|:---|:---|:---|:---|
| `uv_area_distortion_ratio` | UV 面积 vs 3D 面积的 $\max / \min$ 比率 | `recon/geometry.py` (扩展) | $\le 1.5$ |
| `seam_discontinuity_deg` | UV 接缝两侧法线夹角跳跃均值 | `recon/geometry.py` (扩展) | $\le 5.0°$ |
| `texel_density_variance` | 纹素密度 (px/m) 的全局变异系数 (CV) | `recon/geometry.py` (扩展) | $\le 0.3$ |

#### 6.2.4 P1 级：微观曲面与表面质量指标

| 指标 (Key) | 算法规格 | 目标位置 | 门禁阈值 |
|:---|:---|:---|:---|
| `discrete_mean_curvature_variance` | Cotangent-weight 离散平均曲率方差 | `recon/geometry.py` (扩展) | $\le 0.08$ |
| `surface_roughness_var` | 局部微表面曲率方差梯度 | `recon/geometry.py` (扩展) | $\le 0.08$ |
| `patch_lpips` | 512×512 局部特写切片 LPIPS | `inspect/macro.py` (扩展) | $\le 0.12$ |

#### 6.2.5 P2 级：姿态估计与对齐指标

| 指标 (Key) | 算法规格 | 目标位置 | 门禁阈值 |
|:---|:---|:---|:---|
| `add_s_norm` | 归一化平均点距离（对称性） | `inspect/align.py` (扩展) | $\le 0.05 \cdot d_{\text{obj}}$ |
| `reprojection_error_px` | 2D 关节重投影 RMSE（像素） | `inspect/align.py` (扩展) | $\le 4.0$ px |

### 6.3 Synapse-Bench 场景指标名称映射表（已全部打通 ✅）

> [!NOTE]
> T1–T5 场景 JSON 中引用的渲染画质指标与风格指标别名均已在 `metrics/image_quality.py`、`metrics/profiles.py`（`METRIC_ALIASES`）与 `pipeline/bench_runner.py` 中完整实现并接入自动门禁判定。

| 场景 JSON 引用名 | 实现函数 / 规范指标名 | 状态 |
|:---|:---|:---:|
| `style_color_de` | `METRIC_ALIASES` $\to$ `border_ring_ciede2000` | ✅ 已桥接 |
| `style_light_error_deg` | `METRIC_ALIASES` $\to$ `light_dir_error_deg` | ✅ 已桥接 |
| `psnr_db` | `synapse.metrics.psnr_db` (`image_quality.py`) | ✅ 已实现 |
| `ssim` | `synapse.metrics.ssim` (`image_quality.py`) | ✅ 已实现 |
| `lpips` | `synapse.metrics.lpips` (`image_quality.py`) | ✅ 已实现 |
| `ciede2000` | `synapse.metrics.ciede2000` (`image_quality.py`) | ✅ 已实现 |

---

## 7. 推荐外部 3D 评测数据集接入规范

当前 Synapse-Bench 全部 30 套场景 100% 使用程序化几何（`AssetSource.PROCEDURAL`）。代码中已定义 `AssetSource.OBJAVERSE` 与 `AssetSource.KHRONOS` 枚举但从未使用。为覆盖真实世界长尾分布，应逐步引入以下外部数据集。

### 7.1 推荐数据集清单

| 数据集 | 规模 | 核心特征 | 评测价值 | 接入优先级 |
|:---|:---|:---|:---|:---:|
| **Google Scanned Objects (GSO)** | 1,030 扫描物体 | 高质量扫描 + HDRI 渲染，学术标准评测集 | 3D 生成论文标准对标（TripoSR, LGM, InstantMesh 均使用） | **P0** |
| **Objaverse-XL** | 10M+ 3D 资产 | 最大开源 3D 资产库，覆盖长尾品类 | 多样性与泛化能力测试 | **P0** |
| **ABO (Amazon Berkeley Objects)** | 8,222 产品 | 电商真实 3D + PBR 材质 + 参考照片 | 工业级 PBR 材质保真度验证 | **P1** |
| **OmniObject3D** | 6,000 扫描对象 | 190 类别标签 + 多视角渲染 + 点云 | 类别覆盖与分类鲁棒性 | **P1** |
| **Thingi10K** | 10,000 CAD 模型 | 极端拓扑挑战（镂空、薄壁、高亏格） | 拓扑门禁的压力测试 | **P1** |
| **Tanks and Temples** | 21 大场景 | MVS 重建标准基准 | 大场景几何重建精度 | **P2** |
| **CO3D (Common Objects 3D)** | 19K 视频序列 | 多视角视频 + SfM 重建 | 视频到 3D 管线端到端评测 | **P2** |
| **Stanford 3D Scanning** | Dragon, Bunny 等经典 | 超高密度点云扫描 | 高精度几何对比基准 | **P2** |

### 7.2 外部数据集接入契约

外部 3D 资产接入 Synapse-Bench 评测场景时，必须遵循以下统一预处理契约：

1. **坐标系归一化**：统一右手系 Y-up，模型底面中心对齐原点 $(0, 0, 0)$；
2. **尺度标准化**：资产包围球半径归一化至 $[0.8, 1.2]$ 范围或保留原始米制尺度（通过 `metadata.json` 中 `physical_scale_m` 字段标记）；
3. **拓扑预过滤**：入库前必须通过 Stage 0 几何硬门禁（2-manifold、watertight、零退化面），不合格资产不得作为 Ground Truth；
4. **材质约束**：必须具备独立 PBR 贴图（至少 Albedo + Normal），不接受仅有顶点色或烘焙光照的资产；
5. **许可证合规**：仅接入 CC-BY 4.0、CC0、Apache 2.0 或同等宽松许可证的资产。

### 7.3 场景 JSON 扩展格式示例

```json
{
  "id": "t1_gso_shoe_orbit",
  "tier": "T1",
  "description": "GSO high-quality scanned shoe, 360° orbit, neutral lighting",
  "asset": {
    "source": "gso",
    "gso_id": "Shoe_fccb813e",
    "fallback_procedural": "sphere"
  },
  "camera_params": {
    "trajectory": "orbit",
    "num_frames": 17,
    "elevation_deg": 15.0,
    "fov_deg": 50.0
  },
  "expected_metrics": {
    "psnr_db": 28.0,
    "ssim": 0.90,
    "lpips": 0.08,
    "silhouette_iou": 0.92,
    "depth_rmse_m": 0.05,
    "normal_angular_error_deg": 8.0,
    "dino_cosine_sim": 0.88
  }
}
```

> [!NOTE]
> `fallback_procedural` 字段确保在无法获取外部资产（如 CI 环境无网络）时，场景自动降级为对应的程序化几何体，保证测试不会因网络依赖而中断。

---

## 8. 评估体系补全实施路线图

### 8.1 修复优先级与工时估算

| 阶段 | 要做什么 | 状态 / 工时 | 产出 |
|:---|:---|:---|:---|
| **P0-Image & Alias** | 实现 `psnr_db` / `ssim` / `lpips` / `ciede2000` 计算函数（`metrics/image_quality.py`）；在 `metrics/profiles.py` 中增设 `METRIC_ALIASES`（`style_color_de`, `style_light_error_deg`）并接入 `bench_runner.py` 与 `model_evaluator.py` | **✅ 已完成** | 30 套基准场景的渲染画质指标与 T4 风格别名全部打通 |
| **P0-Perceptual** | 实现预训练骨干感知语义指标 `dino_cosine_sim` / `lpips_vgg`（`metrics/perceptual.py`，可选依赖 `torch`/`timm`） | **1~2 天** | 高层视觉语义嵌入距离对标行业论文 |
| **P1-Topo** | 基于已落地的 `csrc/geometry/` C++20 + `recon/geometry.py` 双轨几何核（已实现连通域、退化面、共面+MT 自穿插、浮空碎块），补充导出 `euler_characteristic` 与 `non_manifold_edges_count` | **1~2 天** | 3D 网格入库门禁全部量化指标完备 |
| **P1-UV** | 实现 `uv_area_distortion_ratio`、`seam_discontinuity_deg`、`texel_density_variance` | **2~3 天** | 材质贴图质量首次有自动化评测 |
| **P1-Curvature** | 实现 `discrete_mean_curvature_variance`、`surface_roughness_var`、`patch_lpips` | **2~3 天** | 微观表面缺陷检测（阶梯网格、凹坑、过平滑）可量化定位 |
| **P1-Data** | 集成 GSO + Objaverse 外部 3D 资产，新增 6~10 套 T1/T2 外部数据场景 | **3~5 天** | 评测从纯程序化几何扩展到真实长尾资产 |
| **P2-Misc** | 补充 `histogram_intersection` 门禁阈值；实现 `add_s_norm`、`reprojection_error_px` | **1~2 天** | 消除姿态与直方图分布评估盲区 |

### 8.2 各阶段完成验收标准

- **P0-Image & Alias 完成（✅ 已达成）**：运行 `pytest tests/test_metrics_image_quality.py tests/test_bench_runner.py -v` 产出 `psnr_db`、`ssim`、`ciede2000`、`lpips` 真实数值且 `METRIC_ALIASES` 自动映射通过；
- **P1-Topo 完成**：运行 `pytest tests/test_geometry_oracle_parity.py tests/test_geometry_tier0.py -v` 断言 C++20 原生扩展 `synapse._c_geometry` 与 Python Oracle 100% 一致；
- **P1-Data 完成**：Synapse-Bench 场景列表扩展至 36~40 套，其中至少 6 套使用 `AssetSource.OBJAVERSE` 或 `AssetSource.GSO`；
- **全部完成**：`evaluate_metrics_gate(results, "rsi_promotion")` 覆盖全部本文档声明的指标，不遗漏任何设计文档规定的评估维度。

### 8.3 3D 模型评估模块、CLI 与报告模板（已实施 ✅）

> **实施状态**: 已收敛至核心库 `synapse.inspect.model_evaluator` + `synapse.inspect.eval_report`，同时支持主 CLI `synapse inspect evaluate` 与脚本包装器 `scripts/evaluate_3d_models.py`。

#### 评估入口（统一 CLI 与脚本）

- **核心模块**: [`python/synapse/inspect/model_evaluator.py`](../python/synapse/inspect/model_evaluator.py) · **脚本入口**: [`scripts/evaluate_3d_models.py`](../scripts/evaluate_3d_models.py)
- **功能**: 对输入目录中的 3D 模型文件（`.obj`, `.glb`, `.ply` 等）执行 Tier 0 双轨拓扑/自穿插/浮空碎块检查、多视角截图渲染、参考网格几何精度与多视角渲染画质比对
- **CLI 接口**:
  ```bash
  synapse inspect evaluate \
    --input-dir ./models/ \
    --output-dir ./eval_output/ \
    --reference-dir ./reference_models/ \
    --profile TRACK_A_ASSET \
    --resolution 256 \
    --num-views 8 \
    --json
  ```
  > 兼容别名：`--gate-profile` 等价于 `--profile`，`--views` 等价于 `--num-views`；非法档位名严格按 Fail-Closed 返回退出码 `10`。
- **退出码**: `0`=全部通过, `10`=参数/档位校验失败, `20`=输入缺失, `40`=门禁未通过（`"status": "gate_failed"`，HTML/JSON 报告仍完整生成）
- **门禁配置（与 `metrics/profiles.py` 对齐）**: `TRACK_A_ASSET`（默认，`chamfer_mm <= 5.0`）、`TRACK_B_VIDEO`（宽松，`chamfer_mm <= 8.0`）、`RSI_PROMOTION`（严苛，`chamfer_mm <= 4.0`）

#### HTML 报告模板

- **路径**: [`python/synapse/inspect/eval_report.py`](../python/synapse/inspect/eval_report.py)
- **特性**:
  - 自包含 HTML（零外部依赖，内联 CSS/SVG）
  - 概要仪表盘（总量/通过/失败统计）
  - 聚合雷达图（SVG）展示跨模型平均质量得分（所有缺陷计数型指标严格按“越小越优 $\to 1.0$”正向归一化，剔除原始顶点/面数规模干扰）
  - 逐模型详情卡片:
    - **多视角截图画廊**（base64 内联 PNG，无 GPU 纯 NumPy 软件光栅化渲染）
    - **拓扑摘要表**（流形/水密/顶点数/面数/边界边/退化面/自穿插对）
    - **指标表**（指标值 + PASS/FAIL 徽章）
    - **逐模型雷达图**

#### 已实现的评估维度

| 评估维度 | 指标 | 实现状态 |
|:---|:---|:---|
| **Tier 0 拓扑与自穿插** | `is_manifold`, `is_watertight`, `connected_components`, `boundary_edges`, `num_degenerate_faces`, `self_intersecting_pairs`, `floating_island_faces` | ✅ 已实现（C++20 + Python 双轨互验） |
| **几何精度（含参考网格）** | `chamfer_mm`, `f_score_1mm` | ✅ 已实现（复用 `metrics/roundtrip.py`） |
| **多视角渲染画质（含参考网格）** | `psnr_db`, `ssim`, `ciede2000`, `lpips` | ✅ 已实现（复用 `metrics/image_quality.py`） |
| **视觉检查** | 8 视角软件光栅化截图（Front/Right/Back/Left 等） | ✅ 已实现 |
| **门禁配置** | `TRACK_A_ASSET` (`≤5mm`), `TRACK_B_VIDEO` (`≤8mm`), `RSI_PROMOTION` (`≤4mm`) | ✅ 已实现 |

#### 测试覆盖

- **路径**: [`tests/test_eval_report.py`](../tests/test_eval_report.py) · [`tests/test_geometry_oracle_parity.py`](../tests/test_geometry_oracle_parity.py) · [`tests/test_metrics_image_quality.py`](../tests/test_metrics_image_quality.py)
- **覆盖**: 数据模型、HTML 报告生成、雷达图正向归一化、双轨 C++/Python 几何一致性、参考网格几何+画质评估及 CLI 退出码契约测试
