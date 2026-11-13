# RFC 0060: 真实视频逆向 3D 动态数字孪生与联合反演架构规范

- **RFC 编号**: `RFC 0060`
- **文档状态**: `Accepted (v1.1 — Aligned with M9)`
- **责任体系**: sautille-stroke / DervGen 核心架构组 (`synapse`, `fulgur`, `caelum-core`, `aether`)
- **关联设计文档 (Layer B)**: [DESIGN_REAL_VIDEO_TO_3D_INVERSE_RENDERING.md](DESIGN_REAL_VIDEO_TO_3D_INVERSE_RENDERING.md)
- **关联实施计划 (Layer C)**: [PLAN-M9-M12-HORIZON3-LANDING.md](plans/PLAN-M9-M12-HORIZON3-LANDING.md) (`M9: PR-30 ~ PR-32`)
- **关联下游闭环**: [RFC 0062: 3D 动画引导生成、可微逆向渲染与局部修补双向闭环规范](RFC-0062-INVERSE-RENDERING-AND-3D-NEURAL-CLOSED-LOOP.md)
- **跨仓 SSOT 契约**: `@aether//fgpp` (`fgpp.v1.0`), `@aether//style`, RFC 0039 (`JobBundle`), RFC 0044 (白模粗糙度底线)

---

## 1. 背景与动机 (Context & Motivation)

传统 CG 管线中 `3D Scene -> FGPP -> Video` 为单向不可逆过程。一旦实拍素材或神经生成视频发生构图微调、角色动态偏移或局部质感突破，资产变化无法回流至 `fulgur`（Track-B）或 `caelum-core`（Track-A），导致多机位重拍与后期重光（Relighting）失去物理几何锚点。

`RFC 0060` 确立 **`synapse.recon` 真实视频逆向 3D 重建与联合反演体系**，在不破坏 `aether`/`fulgur` 底层前向实时渲染契约的前提下，从单目或多视角视频序列中联合解算：
1. 相机外参/内参轨迹与深度结构对齐；
2. 参数化拓扑网格与骨骼运动序列（Humanoid-53 / SMPL-X 兼容骨架）；
3. 去光照（De-lighting）后的 OpenPBR 贴图与环境 HDRI 球谐/贴图表示。

---

## 2. 目标与非目标 (Goals & Non-Goals)

### 2.1 核心目标 (Goals)
1. **消除静默假体重建**：为 `ReconPipeline` 补齐基于真实像素/特征的多视角几何初始化模块（`synapse.recon.multiview`），彻底消除当 `initial_mesh=None` 时退化为固定 8 顶点立方体导致的“伪往返 PASS”陷阱。
2. **双轨可微反演求解**：在不侵入 `fulgur` C++20 渲染管线的前提下，在 `[invrender]` 可选依赖下使用 `Mitsuba 3` 物理可微渲染核与纯 NumPy 解析 Jacobian CPU 代理，联合反演几何变形与 PBR 贴图。
3. **跨仓资产可重用**：输出严格符合 `OpenUSD (.usda)`、`glTF/OBJ`、`OpenPBR`（BaseColor/Roughness/Metallic/Normal）及 `FGPP v1.0` 契约的可重光照资产，无缝接入 `fulgur`、`caelum-core` 与 `RFC 0062` 主动生成闭环。
4. **CLI-First 与确定性契约**：统一以 `synapse recon invert-video` 作为入口，遵循全仓 `ExitCode`（`0/10/20/30/40/50`）与 `"status": "ok"` JSON 契约。

### 2.2 非目标 (Non-Goals)
- **不在 `fulgur` 中实现可微光栅化 Pass**：`fulgur` 专注前向/延迟实时硬件渲染（Vulkan/Metal/D3D12/Null），不引入反向传播磁带（AD Tape）或可微光栅器。
- **不编写常驻 Python 服务端**：`synapse.recon` 仅作为无状态库与单步执行即退出的 CLI / `JobBundle` Stage 运行（遵守 Server-side Zero-Python 铁律）。
- **禁止粗白模直接 UV 反投**：严格遵守 `RFC 0044 §4.1 & §4.4`，低于 `1500` 顶点或无铰接四肢的粗白模必须先经生成式 3D 重建与重拓扑门禁，严禁直接从视频反投纹理。

---

## 3. 核心架构契约与决策 (Architecture & Decisions)

### 3.1 统一模块归属：`synapse.recon`
所有视频逆向重建与反演能力统一收敛于 `python/synapse/recon/`（配合 `csrc/geometry/` 原生加速核），弃用早期的 `synapse.inversion` 假想命名。阶段职责边界如下：

| Stage | 模块职责 | 关键输出与约束 |
|---|---|---|
| **Stage 1: Camera & Depth** | 相机轨迹（DUSt3R/VGGT/PnP）与尺度对齐 | `P_t` 与连续尺度一致深度序列（拒绝退化全零基线） |
| **Stage 2: Multi-view Mesh Init** | `MultiViewInitializer`（双视角/多视角轮廓刻蚀 + 深度反投影 TSDF） | 真实视差驱动初始网格，替代 `8` 顶点硬编码盒 |
| **Stage 3: Joint Inverse Optimization** | `Mitsuba 3` 可微光追（GPU）或 `AnalyticUVInverseOptimizer`（CPU） | 辐射线性（ACEScg）L1 + 显示域（sRGB）LPIPS + 拉普拉斯正则 |
| **Stage 4: Rigging & USD Export** | 骨骼蒙皮绑定、OpenPBR 贴图解耦与 `.usda` 导出 | 合规资产包，支持直接回灌 `synapse recon roundtrip` 验证 |

### 3.2 光度学色彩空间一致性契约
- 视频帧 $I_t$ 为显示域非线性编码（sRGB / Rec.709），物理反演渲染图 $\hat{I}_t$ 为场景线性空间（ACEScg）。
- 反演损失 $\mathcal{L}_{\text{rgb}}$ 严禁直接将线性 $\hat{I}_t$ 与 sRGB $I_t$ 做差：
  - 像素残差 $\mathcal{L}_1$ 必须在去伽马/逆色调映射后的**场景线性空间**计算；
  - 感知项 $\mathcal{L}_{\text{LPIPS}}$ 必须将线性渲染结果经 `ACEScg -> Display sRGB` 色调映射后再与 $I_t$ 比对。

### 3.3 人工精修熔断与工程门禁
- 需进入 `asset-forge` / `scene-forge` 进行 L2 拓扑或 UV 精修的资产，由 `synapse.bridge.hitl.evaluate_hitl_circuit_breaker(elapsed_seconds, domain="asset_3d", max_seconds=900.0)` 守护；超时 `> 900s` 触发 `CircuitBreakerError`（退出码 `40`）并自动切换降级策略。

---

## 4. 备选方案权衡 (Alternatives Considered)

1. **在 `fulgur` C++20 中新增可微光栅 Pass vs. `synapse` 侧 Mitsuba 3 + 解析 Jacobian**：
   - *选择后者*。`fulgur` 跨 Vulkan/Metal/D3D12 维护实时 RenderGraph，引入反向自动微分会成倍增加 RHI 复杂度；`Mitsuba 3`（Dr.Jit）与 Python 解析微分核原生解耦，且通过 CPU 代理可在无卡 `ubuntu24` 节点实现确定性 CI 回归。
2. **纯 3D Gaussian Splatting (3DGS) 表达 vs. Articulated Mesh + OpenPBR**：
   - *选择后者*。纯 3DGS 难以接入 `fulgur` 阴影贴图、Jolt 碰撞体和 `anim_graph` 骨骼蒙皮；导出参数化 Mesh + PBR 贴图是打通 `RFC 0062` 与真实 CG 制作管线的唯一解。

---

## 5. 验收门禁与指标口径 (Quality Gates)

以下阈值遵循 `DESIGN_REVIEW_AND_ALIGNMENT.md` 统一评测口径（`temporal_warp_error` 采用 `[0, 1]` 归一化光流重投影 RMSE）：

| 维度 | 指标 | 验收阈值 | 验证方式 |
|---|---|---|---|
| **多视角几何精度** | Chamfer Distance (`CD`) | $\le 0.008$ | 破坏性差分测试：错位/噪声组劣化比 $\ge 3.0\times$ |
| **重渲染外观保真** | 线性 PSNR / 显示 sRGB SSIM | $\text{PSNR} \ge 26.0\text{ dB}$, $\text{SSIM} \ge 0.88$ | `synapse recon roundtrip --json` |
| **时序重投影一致性** | `temporal_warp_error` (`[0,1]`) | $\le 0.020$ | 双轨往返测试套件 (`tests/test_recon_roundtrip.py`) |
| **白模防呆守护** | `validate_whitebox_coarseness_floor` | 顶点 $\ge 1500$ 且含四肢关节 | 违规输入返回退出码 `10` (`CONTRACT_ERROR`) |
| **CLI 与 JSON 契约** | `synapse recon invert-video --json` | `"status": "ok"`, `"exit_code": 0` | `tests/test_cli.py` + `tests/test_recon_*.py` |

---

## 6. 实施计划绑定 (`M9: PR-30 ~ PR-32`)

本 RFC 的工程落地完全绑定于 [PLAN-M9-M12-HORIZON3-LANDING.md](plans/PLAN-M9-M12-HORIZON3-LANDING.md) 的 **M9 里程碑**：

| PR | 目标范围 | 核心交付物 |
|---|---|---|
| **PR-30** | 多视角几何初始化（解除 8 顶点盒陷阱） | `python/synapse/recon/multiview.py`、`ReconPipeline` 默认调用与真差测试 |
| **PR-31** | 双轨可微纹理/光照联合反演与往返真值门禁 | `Mitsuba` + CPU 解析 Jacobian 反演通道、`roundtrip` 破坏性差分对比（$\ge 3.0\times$） |
| **PR-32** | CLI 统一入口与 `.usda` / `FGPP v1.0` 回灌联调 | `synapse recon invert-video` 与 `JobBundle` `video_to_3d_invert` Stage 闭环 |
