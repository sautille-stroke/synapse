# 细分执行计划：3D 模型多视角生成与逆向重建闭环验证计划
## (3D Round-Trip Reconstruction & Differential Verification Plan)

- **所属总计划**: [docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) · [PLAN-M9-M12-HORIZON3-LANDING.md](PLAN-M9-M12-HORIZON3-LANDING.md) (`M9: PR-30 ~ PR-31`)
- **关联设计规范**: [RFC 0060](../RFC-0060-REAL-VIDEO-TO-3D-DIGITAL-TWIN.md) · [DESIGN_REAL_VIDEO_TO_3D_INVERSE_RENDERING.md](../DESIGN_REAL_VIDEO_TO_3D_INVERSE_RENDERING.md) · [DESIGN_3D_INSPECTION_AND_REFINEMENT.md](../DESIGN_3D_INSPECTION_AND_REFINEMENT.md) · [DESIGN_3D_MODEL_ENHANCEMENT_AND_DATA_SPEC.md](../DESIGN_3D_MODEL_ENHANCEMENT_AND_DATA_SPEC.md)
- **文档状态**: `Active Plan v1.1 (Aligned with M9)`
- **修订日期**: 2026-11-13
- **协同模式**: 人机协同（Human-in-the-Loop, 关键节点人眼核签 + 算法脚本自动化）
- **核心目标**:
  1. 正向多视角图像 / 环视视频生成与人眼视觉确认 (Visual Audit)
  2. 逆向重建闭环与正向/负向差分敏感度报表验证（好数据高分 PASS vs 坏数据红灯 FAIL）
  3. 沉淀自动化保护测试 `tests/test_roundtrip_reconstruction.py`（阶段级往返闭环集成测试）

---

## 1. 核心工程哲学：差分敏感度与闭环度量

### 1.1 闭环链路 (The Round-Trip Loop)
本项工程串联从 3D 资产到多视角图像、再从图像逆向生成 3D 模型的完整往返过程：

```
┌─────────────────┐       正向多角度拍照 / 视频渲染       ┌────────────────────────┐
│ 原始 3D 黄金资产 │ ─────────────────────────────────> │ 多视点图像 / 环绕视频  │
└─────────────────┘                                    └────────────────────────┘
         │                                                          │
         │ (Ground Truth)                                           │ (输入逆向重建)
         ▼                                                          ▼
┌──────────────────────────────────────────────┐       ┌────────────────────────┐
│          多维几何度量与质检评估系统          │ <──── │   逆向重建的 3D 模型   │
│ (Chamfer / Normal / Watertight / Radar / 门禁)│       └────────────────────────┘
└──────────────────────────────────────────────┘
```

### 1.2 差分敏感度准则 (Differential Sensitivity Rule)
**“一套健全的评估体系，其价值不仅在于给好数据打高分，更在于必须能对坏数据敏锐亮红灯。”**

若无论输入清晰数据还是畸变损坏数据，报表输出的各项指标均无实质性变化，则说明质检体系是“迟钝或失明”的。我们必须通过两组对照实验（黄金组 vs 劣化组）验证报表的**断崖式差分**，从而证明质检与重建闭环具备真实的工业判决力。

> [!CAUTION]
> **前置代码阻断项 (`ReconPipeline` 空盒陷阱，由 `M9 PR-30` 解锁)**：
> 当前 `python/synapse/recon/pipeline.py` 中 `ReconPipeline.run()` 在 `initial_mesh=None` 时会直接生成硬编码 8 顶点默认立方体并忽略输入帧像素！因此**阶段二与阶段三严禁跳过 `PR-30`**——必须先落地 `synapse.recon.multiview.MultiViewInitializer` 真实从多视角深度/轮廓反投影构造网格，否则 Clean 与 Corrupted 两组会得到恒等网格（差分比 $= 1.0\times$ 伪 PASS）。

---

## 2. 人机协同分工矩阵 (Manual HITL vs Automated Matrix)

本计划强调**“人眼核验在前，代码固化在后”**。核心环节需要开发者手动审查与主观决断：

| 阶段 | 任务项 | 执行主体 | 关键产物 / 行为 | 验收方式 |
| :---: | :--- | :---: | :--- | :--- |
| **P1** | 选定黄金 3D 资产（DamagedHelmet / 几何体） | **手动 (人工)** | 确认网格与材质源文件就绪 | 人工选定 |
| **P1** | 执行正向多视角渲染与环视视频生成 | 自动化 (脚本) | 输出多角度 PNG 序列与 Turntable 视频 | 脚本产出非空 |
| **P1** | **正向数据人眼视觉确认 (Visual Audit)** | **手动 (人眼)** | 检查视点覆盖度、深度真实度、法线凸起感、无半透明插值污染 | **人工 Sign-off 核准** |
| **P2** | 黄金组数据输入逆向重建并生成报表 | 混合 (人工触发+脚本) | 产出重构模型 A 与 `report_clean.html` | 预期全绿 PASS |
| **P2** | 注入可控缺陷（视角缺失/视线模糊/破洞截断） | 混合 (人工定义+脚本) | 产出损坏输入源与劣质重构模型 B | 确定性缺陷注入 |
| **P2** | 劣化组逆向重建并生成对比报表 | 混合 (人工触发+脚本) | 产出 `report_corrupted.html` | 预期红灯 FAIL |
| **P2** | **双报表人眼差分核对与灵敏度评估** | **手动 (人眼)** | 对比 SVG 雷达图与 Chamfer 指标差距 | **确认出现显著断崖差距** |
| **P3** | 编写 `tests/test_roundtrip_reconstruction.py` | 自动化 (代码) | 阶段级闭环集成测试脚本 | `pytest` 跑通 |
| **P3** | 断言正向 PASS、负向 FAIL 与差分敏感度倍数 | 自动化 (CI 门禁) | 固化在全仓回归套件中 | 云端 `ubuntu24` 零卡秒级回归 |

---

## 3. 三阶段落地实施规划

### 阶段一：正向多角度数据生成与人眼视觉确认（重点第一步）

1. **输入准备**：
   - 资产路径：`data/datasets/damaged_helmet_orbit_60f/` 或指定 glTF/OBJ 资产。
   - 包含要素：几何顶点、法线贴图、PBR 材质。
2. **正向多角度采集生成**：
   - 采集方式：基于半球/球面均匀采样，生成 8~16 个环绕视角渲染图，并组合成 360° Turntable 视频与 Contact Sheet。
3. **【手动】人眼审查核心 Checkpoint**：
   - [ ] **视角完整性**：俯仰角（Pitch）与方位角（Yaw）是否全方位覆盖，有无死角丢失？
   - [ ] **深度图（Linear Depth）**：近处与远处是否有数值断层？是否存在逆深度/反转现象？
   - [ ] **法线图（Normal Map）**：法线半球映射是否符合约定？旋转时凹凸光影是否自然？
   - [ ] **掩膜边缘（Cryptomatte/Mask）**：边缘是否完全锐利？是否存在插值抗锯齿羽化？
   - [ ] **人工核签**：确认该数据集具备作为“Ground Truth 输入”的资格。

---

### 阶段二：逆向重建与差分敏感度报表实验

1. **正向黄金组（Baseline Group）**：
   - 输入：阶段一核签通过的高质量多视角数据。
   - 执行：调用逆向几何重构流程产出网格模型。
   - 评估：运行 `scripts/evaluate_3d_models.py`，计算 Chamfer 距离、流形度、水密性与法线一致性。
   - 产出：`eval_output/roundtrip_clean/report.html` 与 `metrics.json`。
   - 预期门禁：**PASS（达到 TRACK_A_ASSET 或 TRACK_B_VIDEO 准入标准）**。

2. **负向对抗组（Corrupted Group）**：
   - 确定性缺陷注入契约（供阶段三自动化测试复用）：
     - **缺陷 A（视点稀疏/共线盲区）**：仅保留连续 $\le 2$ 帧且相机夹角跨度 $< 15^\circ$（模拟严重单侧遮挡）；
     - **缺陷 B（深度/法线畸变）**：将 `WorldNormal` 整体取反（`-N`）并对线性深度注入 $\sigma = 0.08\text{m}$ 高斯噪声及 $3.5\text{px}$ 高斯模糊；
     - **缺陷 C（几何截断破损）**：屏蔽上半球掩膜像素（模拟头冠/关键结构缺失孔洞）。
   - 执行：相同参数跑 `MultiViewInitializer` + `ReconPipeline` 逆向重建流程。
   - 评估：生成 `eval_output/roundtrip_corrupted/report.html`。
   - 预期门禁：**FAIL（`Chamfer Distance` 恶化，非流形/低覆盖报警）**。

3. **【手动】人眼比对与结论审定**：
   - 打开两份 HTML 报告并排比对；
   - 检查雷达图缩放与覆盖面积变化；
   - 锁定最低差分判据：`CD_corrupted / max(CD_clean, 1e-6) >= 3.0`，且对称法线一致性由 Clean $\ge 0.90$ 跌落至 Corrupted $\le 0.50$。

---

### 阶段三：自动化测试固化 (`tests/test_roundtrip_reconstruction.py`, `M9 PR-31`)

在人眼确认正向数据质量与差分有效性后，将该流程固化为代码层面的守护门禁（绑定 `PR-31`）：

1. **测试定位**：
   - 文件位置：`tests/test_roundtrip_reconstruction.py`
   - 测试类别：**Pipeline Stage Integration Test（阶段级往返闭环集成测试）**
   - 依赖属性：基于纯 NumPy 软光栅与 `MultiViewInitializer` 轻量确定性算子，确保在无卡环境（`ubuntu24` / 本地）秒级运行完毕。
2. **测试用例编排**：
   - `test_roundtrip_golden_pipeline_passes_gate()`：验证标准 12 视角半球渲染输入下，重建网格通过门禁（`CD <= 0.008`, `PSNR >= 26.0 dB`）；
   - `test_roundtrip_corrupted_input_fails_gate()`：分别验证缺陷 A/B/C 输入下，质检系统必然拦截并判定 FAIL；
   - `test_differential_sensitivity_ratio()`：断言黄金组与破坏组满足 `CD_corrupted / CD_clean >= 3.0`（彻底锁定未退化回硬编码 `8` 顶点盒）。

---

## 4. 实施状态与追踪

- [x] **2026-11-12**: 制定人机协同闭环方案，明确差分对比与人眼核签机制。
- [ ] **Phase 1**: 正向多角度数据就绪，完成人眼视觉核签。
- [ ] **Phase 2**: 黄金组与负向组逆向重建完成，产出双份差分 HTML 报表并完成人眼确认。
- [ ] **Phase 3**: 编写并合入 `tests/test_roundtrip_reconstruction.py`，全量单测通过。
