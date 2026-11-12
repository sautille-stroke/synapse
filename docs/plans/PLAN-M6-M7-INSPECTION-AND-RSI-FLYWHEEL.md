# 细分执行计划 4：Phase M6–M7 3D 智能微距质检、Visual Jev 判别与物理驱动 RSI 自演进飞轮 (`PR-20` ~ `PR-25`)

- **所属总计划**: [docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) (`v2.0` Horizon 2)
- **前置设计文档**: [docs/DESIGN_3D_INSPECTION_AND_REFINEMENT.md](../DESIGN_3D_INSPECTION_AND_REFINEMENT.md) · [docs/DESIGN_RSI_FLYWHEEL.md](../DESIGN_RSI_FLYWHEEL.md) · [docs/DESIGN_REVIEW_AND_ALIGNMENT.md](../DESIGN_REVIEW_AND_ALIGNMENT.md) (`RV-04`, `RV-09`)
- **覆盖范围**: `PR-20` ~ `PR-25`（共 6 个原子 PR，预估 8.5 人天）
- **跨仓边界红线**:
  - 严禁在 `synapse.inspect` 或 `synapse.rsi` 中用 Python 手写半边网格流形遍历、BVH 加速结构或骨骼 IK 循环（全部通过 `synapse.bridge.vg_bridge` 调用 `vg-core` C++20 算子）；
  - 严禁使用纯神经自评做唯一裁判——任何进入 RSI 正样本池的数据**必须通过至少一项闭式物理几何验算**，且每轮训练集必须保留 **$\ge 20\%$ 人类专家 L0 黄金样本**。

---

<a id="pr-20"></a>
## 1. PR-20：3D 质检数据契约 (`InspectionReport v2.0`) 与单图逆向相机位姿对齐 (`align.py`)

- **建议标题**: `feat(inspect): add InspectionReport v2.0 schema and Fibonacci coarse-to-fine pose alignment [model-tag]`
- **前置依赖**: `PR-12` (`recon.geometry`), `PR-13` (`metrics`)
- **代码净增预算**: `~480` 行
- **涉及文件清单**:
  - 新增：`python/synapse/inspect/__init__.py`, `python/synapse/inspect/schemas.py`, `python/synapse/inspect/align.py`, `tests/test_inspect_align.py`

### 1.1 精确落地规格
1. **`inspect/schemas.py`**：
   - 定义 Pydantic v2 模型 `CameraPose`、`DefectAnnotation3D`（含 `defect_id`, `category`, `severity`, `confidence`, `center_3d`, `bounding_radius_m`, `affected_vertices_count`, `macro_camera`, `macro_crop_uri`, `suggested_action`）与 `InspectionReport`（`schema_version="2.0.0"`）。
2. **`inspect/align.py` (`align_reference_image_to_mesh`)**：
   - **Stage 1A 球面粗检索**：在目标网格外接球上生成 $N=48$ 个均匀 Fibonacci 螺旋候选机位，使用软光栅渲染剪影与法线图，通过可插拔特征提取器（默认剪影 Hu 矩/CHAMFER 轮廓距离 + 可选 DINOv2 余弦相似度）筛选 Top-3 候选机位；
   - **Stage 1B 精配准**：优先通过 `vg_bridge.refine_pose_6dof` 调用 C++ 位姿精化（纯轻量环境提供基于 Powell / Nelder-Mead 剪影 IoU 优化的无梯度回退器），输出最优 `CameraParams` 并强制校验 `Gate 1: silhouette_iou >= 0.88`（未收敛则抛出 `QualityGateError(40)` 提示参考图与模型姿态不匹配）。

### 1.2 验证门禁
- 对非对称几何模型（如带手柄茶壶或 Humanoid-53）在随机机位（方位角 `137°`, 仰角 `22°`）生成参考渲染图，运行 `align_reference_image_to_mesh` 恢复机位：断言重渲染剪影 `silhouette_iou >= 0.90`、相机位置误差 `< 5%` 外接球直径。

---

<a id="pr-21"></a>
## 2. PR-21：同位多通道残差热力图解耦与虚拟微距相机 3D 反投规划 (`macro.py`)

- **建议标题**: `feat(inspect): implement multi-channel residual heatmap and virtual macro camera 3D grounding [model-tag]`
- **前置依赖**: `PR-20`
- **代码净增预算**: `~460` 行
- **涉及文件清单**:
  - 新增：`python/synapse/inspect/macro.py`, `tests/test_inspect_macro.py`

### 2.1 精确落地规格
1. **同位多通道残差解耦 (`compute_isotopic_residual_heatmap`)**：
   - 在对齐机位下对比参考图与同位渲染的 RGB、深度边缘、世界法线连续性与 `BaseColor` 色差，融合生成归一化 2D 异常热力图 $H(u, v) \in [0, 1]$；
   - 使用连通域非极大值抑制（NMS）提取显著异常热区列表（最多 Top-$K=6$ 个局部 ROI）。
2. **三维空间反投与虚拟微距机位解算 (`plan_macro_camera_for_roi`)**：
   - 利用该像素区的 `CameraLinearDepth.Z` 与 `CameraParams` 内外参将 2D 异常中心 $(u_c, v_c, z_c)$ 反投影（Unproject）至三维世界坐标锚点 $\mathbf{p}_{\text{defect}} \in \mathbb{R}^3$ 及包围球半径 $r_{\text{defect}}$；
   - 沿局部表面平均法线 $\bar{\mathbf{n}}_{\text{world}}$ 方向外推距离 $d_{\text{macro}} = \frac{1.35 \cdot r_{\text{defect}}}{\tan(\text{fov}_{\text{macro}} / 2)}$（带视线遮挡光线投射自检：若被其他肢体遮挡则在 $\pm 35^\circ$ 圆锥内搜索无遮挡最佳观察角！），生成 $512\times 512$ 高倍微距 `CameraPose`。

### 2.2 验证门禁
- 在单位球表面已知三维坐标点注入一块半径 `0.08m` 的局部色斑/法线凹坑，运行热力图检测与微距反投：断言反求出的 `center_3d` 与真实缺陷中心欧氏距离 `< 5.0mm`，且生成的微距机位将缺陷完整置于 $512\times 512$ 视口中央且无前景遮挡。

---

<a id="pr-22"></a>
## 3. PR-22：`Visual Jev` 共享视觉上下文强类型判别客户端与容器质检模式 (`jev_critic.py`)

- **建议标题**: `feat(inspect): add Visual Jev calibrated logits critic and asset_inspect_3d worker stage [model-tag]`
- **前置依赖**: `PR-14b`, `PR-21`
- **代码净增预算**: `~450` 行
- **涉及文件清单**:
  - 新增：`python/synapse/inspect/jev_critic.py`, `tests/test_inspect_jev.py`
  - 修改：`python/synapse/pipeline/bundle_worker.py`, `python/synapse/cli/__init__.py`

### 3.1 精确落地规格
1. **`VisualJevCritic` 强类型 Logits 校准判别器**：
   - 对接 `foundary` 托管的 `vLLM` / OpenAI-Compatible `logprobs` 接口（利用 Shared Visual Context Prefix Caching 一次性编码多通道全局图 + 微距 Patch，批量下发 6 类缺陷二元判别探针：`mesh_crumple`, `texture_stretch`, `seam_tearing`, `floating_debris`, `anatomical_distortion`, `plastic_shading`）；
   - 直接从首 Token `Yes` / `No` 的 Top Logprobs 计算温度缩放校准概率 $P_{\text{clean}} = \frac{e^{z_{\text{no}}/T_c}}{e^{z_{\text{yes}}/T_c} + e^{z_{\text{no}}/T_c}}$（彻底杜绝自由文本 JSON 解析崩溃与长自回归延迟）；
   - 提供确定性 `MockJevTransport` 供离线测试验证温度校准、阈值分流（`auto_remesh` / `auto_inpaint_texture` / `manual_hitl`）与 `InspectionReport` 组装。
2. **接线 CLI 与 Worker**：
   - 落地 `synapse inspect align` 与 `synapse inspect eval` 子命令，并在 `JobBundleWorker` 注册 `stage = "asset_inspect_3d"`。

### 3.2 验证门禁
- 单测使用 `MockJevTransport` 校验 Logprobs → 校准置信度数学公式精度、连接失败抛 `30`、检出高危解剖学畸变（`severity >= 0.8`）时自动置 `requires_hitl=True` 并返回退出码 `40`。

---

<a id="pr-23"></a>
## 4. PR-23：靶向精修往返验证编排与 RFC 0020/0021 强化学习数据导出器 (`flywheel_exporter.py`)

- **建议标题**: `feat(inspect): add refinement roundtrip orchestrator and RFC 0020/0021 DPO/PRM data exporter [model-tag]`
- **前置依赖**: `PR-12`, `PR-22`
- **代码净增预算**: `~500` 行
- **涉及文件清单**:
  - 新增：`python/synapse/inspect/flywheel_exporter.py`, `tests/test_inspect_flywheel.py`
  - 修改：`python/synapse/cli/__init__.py`

### 4.1 精确落地规格
1. **自动精修与往返门禁编排 (`run_targeted_refinement`)**：
   - 根据 `InspectionReport` 中的 `suggested_action` 路由调用 `foundary_client`（`Trellis-Refine` / 局部贴图补全）与 `vg_bridge`（`quad_remesh` / `check_manifold`）；
   - 强制执行 **Stage 5 往返硬门禁**：`delta_fscore_1mm > 0`、`volume_breathing_pct <= 3.0%`、`non_manifold_edges == 0`；若自动修复未通过或超时则无缝转交 `hitl.py` 触发 L2 美术派单。
2. **RFC 0020 / RFC 0021 训练数据提纯器 (`export_rl_flywheel_dataset`)**：
   - 解析编辑器或自动精修产出的 `IntentGroup` 事务日志目录，支持两种标准导出模式：
     - `--format prm`：提取细粒度 `StepAnnotation`（`+` / `-` 步骤正确性标签 + 微距状态切片 URI + 指标增量 $\Delta R_t$），输出供 Process Reward Model 训练的 JSONL；
     - `--format dpo`：自动识别人工 `ImplicitFeedback.undo_triggered == True`（或算法修复未过门禁后由人工修好）的配对轨迹，将撤销前的算法状态记为 `rejected`、最终通过门禁的状态记为 `chosen`，零人工额外标注产出标准 DPO JSONL；
   - 挂载 CLI `synapse inspect refine` 与 `synapse inspect export-rl-data`。

### 4.2 验证门禁
- 构造含 3 条自动修复成功、2 条美术按下 `Undo` 后重修的模拟 `IntentGroup` 日志，运行 `synapse inspect export-rl-data --format dpo --json`，断言精确导出 2 对合法 `(prompt, state_before, chosen, rejected)` 记录且流形违背样本绝不混入 `chosen`。

---

<a id="pr-24"></a>
## 5. PR-24：RSI 自博弈课程生成器、物理+PRM 双重验算核与 20% 黄金锚点偏好配对器

- **建议标题**: `feat(rsi): implement curriculum sampler, dual physical+PRM verifier, and 20%-replay pair builder [model-tag]`
- **前置依赖**: `PR-19`, `PR-23`
- **代码净增预算**: `~560` 行
- **涉及文件清单**:
  - 新增：`python/synapse/rsi/__init__.py`, `python/synapse/rsi/curriculum.py`, `python/synapse/rsi/dual_verifier.py`, `python/synapse/rsi/pair_builder.py`, `tests/test_rsi_verifier_and_pairs.py`
  - 修改：`python/synapse/pipeline/bundle_worker.py`

### 5.1 精确落地规格
1. **`rsi/curriculum.py` (`CurriculumSampler`)**：
   - 根据上一轮 `Synapse-Bench` 薄弱维度得分自适应调整难度分布 $\Omega_{\text{task}}(\theta)$（动态提升薄弱条件因子如 `F01_viewpoint`、`F03_lighting`、`F06_occlusion` 到 `L3/L4` 的采样概率），输出标准化挑战 `Scenario` 清单供 `fulgur` / 软光栅烘焙条件帧。
2. **`rsi/dual_verifier.py` (`DualVerifierEngine`)**：
   - **绝对反崩溃红线 1（禁止纯神经回音室）**：硬编码断言每个候选样本必须先执行 `vg_bridge` / `fgpp_fidelity` 确定性物理检查 $V_{\text{phys}}$（流形水密、穿模体积比 $\le 0.001$、足底滑移速度 $\le 0.02\text{ m/s}$、`silhouette_iou >= 0.88`、`depth_rmse <= 0.03m`、`normal_angular_error <= 12.0°`），再结合 `VisualJevCritic` $V_{\text{prm}}$ 综合裁定；若缺失物理真值输入直接抛出 `ContractValidationError(10)`！
   - 按失败归因标签自动路由至五维正交 LoRA 桶（如 `SILHOUETTE_DRIFT` / `DEPTH_INCONSISTENCY` → `control` LoRA；`FOOT_SLIPPING` → `motion` LoRA；色偏 → `style` LoRA）。
3. **`rsi/pair_builder.py` (`PreferenceDatasetBuilder`)**：
   - 合并算法二次精修转绿的自学习偏好对 $\mathcal{D}_{\text{self}}$ 与 L2 美术精修偏好对 $\mathcal{D}_{\text{human}}$；
   - **绝对反崩溃红线 2（经验回放锚定）**：硬编码校验人类专家 L0 黄金基准样本在最终导出训练清单中的占比 `l0_golden_ratio >= 0.20`（低于 `20%` 时自动过采样 L0 黄金集或拒绝打包抛出 `ContractValidationError(10)`！）；
   - 注册 `JobBundleWorker` `stage = "rsi_dual_verify"`。

### 5.2 验证门禁
- 单测验证：1) 仅提供 Jev 高分但 `foot_sliding` 超标的样本 100% 被判为 `Hard_Negative` 并正确归入 `motion` LoRA 待训桶；2) 传入空 L0 黄金集调用 `build_pairs` 立即被拦截（退出码 `10`）。

---

<a id="pr-25"></a>
## 6. PR-25：单调递增防退化发布门禁 (`PromotionGate`)、隔离回滚清单与 `synapse rsi` CLI

- **建议标题**: `feat(rsi): add monotonic promotion gatekeeper, quarantine manifest, and synapse rsi CLI [model-tag]`
- **前置依赖**: `PR-24`
- **代码净增预算**: `~430` 行
- **涉及文件清单**:
  - 新增：`python/synapse/rsi/promotion_gate.py`, `tests/test_rsi_promotion_gate.py`
  - 修改：`python/synapse/cli/__init__.py`

### 6.1 精确落地规格
1. **`rsi/promotion_gate.py` (`evaluate_monotonic_promotion`)**：
   - 输入当前生产基线报告 $G_t$ 与候选权重全量 30 场景 `Synapse-Bench` 报告 $G_{t+1}$，严格验算四条合取条件：
     1. $\text{OverallScore}(G_{t+1}) > \text{OverallScore}(G_t) + \epsilon_{\text{min}}$；
     2. 几何聚合得分 $\text{GeomScore}(G_{t+1}) \ge \text{GeomScore}(G_t)$；
     3. 时序聚合得分 $\text{TemporalScore}(G_{t+1}) \ge \text{TemporalScore}(G_t)$；
     4. **单场景零退化硬红线**：$\forall s \in \text{Scenarios}_{1..30}, \; \text{Regression}(s) \le \tau_{\text{noise}}$（默认容忍浮点噪声 `0.0`，禁止拆东墙补西墙！）；
   - 若任一条件违背，自动生成 `quarantine_manifest.json`（记录退化场景 ID、退化指标差值、待回滚 LoRA SHA256 与诊断建议），并以退出码 `40` 阻断晋升。
2. **完整挂载 `synapse rsi` CLI 子命令族**：
   - `synapse rsi sample-curriculum`、`synapse rsi verify`、`synapse rsi build-pairs`、`synapse rsi gate-promote`（全部支持 `--json`）。

### 6.2 验证门禁
- 构造 30 场景基准报告：候选模型在 29 个场景提升 `+5%`，仅在 `t1_rigid_sphere_orbit` 深度误差恶化 `0.005m`，运行 `synapse rsi gate-promote --json` 断言 100% 触发退出码 `40` 并精准定位 `t1_rigid_sphere_orbit` 写入隔离清单。
