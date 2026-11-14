# 细分执行计划 2：Phase M3–M4 风格对齐、局部手术刀合成、`vg-core` 桥接与 30 场景基准 (`PR-09` ~ `PR-15`)

- **所属总计划**: [docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) (`v3.1` Master Plan, Horizon 1: M3–M4)
- **前置对齐决议**: [docs/DESIGN_REVIEW_AND_ALIGNMENT.md](../DESIGN_REVIEW_AND_ALIGNMENT.md) (`RV-02`, `RV-07`, `RV-09`)
- **覆盖范围**: `PR-09` ~ `PR-15`（含 `PR-14a`, `PR-14b`，共 8 个原子 PR，预估 11.0 人天）
- **测试主力环境**: `ssh ubuntu24 "cd ~/work/dervgen/synapse && pytest ... && mypy --strict python/synapse && ruff check python/ tests/"`

---

<a id="pr-09"></a>
## 1. PR-09：双曲正切软肩 ACEScg 色彩科学、球谐光照提取与 RFC 0043 `StyleProfile` 映射

- **建议标题**: `feat(style): implement soft-knee ACEScg color pipeline, style extractor, and StyleProfile v2 mapper [model-tag]`
- **前置依赖**: `PR-03`（可与 `M2` 完全并行）
- **代码净增预算**: `~620` 行
- **涉及文件清单**:
  - 新增：`python/synapse/style/aces_color.py`, `python/synapse/style/descriptor.py`, `python/synapse/style/extractor.py`, `python/synapse/style/profile_mapper.py`, `tests/test_style.py`
  - 修改：`python/synapse/style/__init__.py`

### 1.1 精确落地规格
1. **`style/aces_color.py` 辐射能量守恒色彩管线**：
   - 实现精确分段 `srgb_to_linear` / `linear_to_srgb`（IEC 61966-2-1 阈值 `0.04045`）；
   - 实现 `linear_srgb_to_acescg` 与 `acescg_to_linear_srgb`（D65 Rec.709 ↔ D60 AP1 $3\times 3$ Bradford 色适应矩阵，往返误差 $< 10^{-6}$）；
   - **双曲正切软肩逆色调映射 (`soft_knee_inverse_tonemap`)**：对 LDR 线性亮度先计算基础逆映射，再施加 $\mathcal{C}^1$ 连续软肩约束：当 $L \le 4.0$ 时恒等输出；当 $L > 4.0$ 时输出 $4.0 + 8.0 \cdot \tanh\left(\frac{L - 4.0}{8.0}\right)$，确保全图最大辐射度严格 $\le 12.0$ 且零 `NaN/Inf`；
   - 向量化实现 `rgb_to_cielab` 与 `ciede2000_delta_e(lab1, lab2)`（完整包含 $S_L, S_C, S_H$ 与蓝色区域旋转项 $R_T$）。
2. **`style/descriptor.py` & `extractor.py` 双模风格提取**：
   - `StyleDescriptor`（Pydantic v2 模型，默认 `working_colorspace="ACEScg"`）：包含 `color_temp_kelvin`, `tint`, `exposure_ev`, `contrast`, `saturation`, `key_light_dir_world`, `key_light_rgb`, `ambient_ratio`, `grain_sigma`, `vignette_strength`, `tone_curve_lut_1d`；
   - `extract_style_optical(frames_srgb)`：跨帧中位数聚合估算色温（McCamy / Planckian 轨迹近似）、曝光偏移、1D 单调色调曲线与高频拉普拉斯残差颗粒度；
   - `extract_style_fgpp_intrinsic(frames_srgb, fgpp_seq)`：在有效前景掩膜（`cryptomatte_id > 0` 且漫反射主导区域）联立 `WorldNormal.XYZ` 与 `BaseColor.RGB`，通过带正则化的加权最小二乘求解一阶球谐光照（环境光 + 主平行光方向 `key_light_dir_world` 与辐射强度）。
3. **`style/profile_mapper.py` 独立内建 `StyleProfile v2.0.0` Schema（对齐 `RV-07`）**：
   - 不阻塞等待 C++ `@aether//style` 落盘，直接按 `DESIGN-0043-01` 七段式规范（`Metadata`, `ShadingConfig`, `MaterialRemapConfig`, `PostProcessConfig`, `AnimationStepConfig`, `EnvironmentModifier`, `AudioModifier`, `TransitionConfig`）定义 Pydantic 校验模型，提供 `to_style_profile_dict()` / `save_style_profile_yaml()` 及 `to_engine_light_overrides()`。

### 1.2 验证门禁
- 1) 纯白高光 `[255, 255, 255]` 输入 `soft_knee_inverse_tonemap` 输出严格 $\le 12.0$ 且有限，在 $L=4.0\pm 10^{-4}$ 两侧数值一阶导数连续；
- 2) 使用已知光源方向（方位角 `45°`、仰角 `30°`）与色温 `4500K` 软光栅渲染的球体序列运行 `extract_style_fgpp_intrinsic`，恢复的主光方向夹角误差 $< 8.0^\circ$。

---

<a id="pr-10"></a>
## 2. PR-10：自动异常区域检测器与 `3D ∪ SAM2` 运动自适应膨胀手术刀掩膜

- **建议标题**: `feat(composite): implement artifact detector and 3D+SAM2 motion-dilated surgical mask [model-tag]`
- **前置依赖**: `PR-03`
- **代码净增预算**: `~430` 行
- **涉及文件清单**:
  - 新增：`python/synapse/composite/detector.py`, `python/synapse/composite/mask.py`, `tests/test_composite_mask.py`
  - 修改：`python/synapse/composite/__init__.py`

### 2.1 精确落地规格
1. **`composite/detector.py` (`detect_video_artifacts`)**：
   - 输出强类型 `list[VideoDefectRegion]`（含 `defect_type`, `frame_range`, `bbox_xyxy`, `entity_id`, `severity_score`）：
     - `temporal_flicker`：沿 `MotionVector.XY` 反向扭曲对齐相邻帧，计算局部窗口二阶时间差分残差；
     - `silhouette_drift`：提取视频亮度/色度显著梯度边缘，与 FGPP `Cryptomatte.ID` / 深度跳变边缘计算局部距离变换偏差；
     - `ground_violation`：检测实体最低前景像素在世界空间 $Y < y_{\text{ground}} - \epsilon$（穿地）或触地帧悬空异常。
2. **`composite/mask.py` (`generate_surgical_mask`)**：
   - 严格实现 $M_{\text{surgical}}(t) = \text{Dilate}_{\mathbf{v}(t)}\left( M_{\text{3D}}^{\text{new}}(t) \cup M_{\text{SAM2}}^{\text{old}}(t) \right)$；
   - 先做半径 `r_base`（默认 `8px`）各向同性形态学膨胀，再根据局部前景平均光流向量 $\bar{\mathbf{v}} = (v_x, v_y)$ 构造沿运动方向拉长的有向核（长度 $\min(32, \|\bar{\mathbf{v}}\| \cdot k_v)$）执行运动扫掠膨胀，最后用余弦距离变换生成平滑过渡软掩膜 $\alpha_{\text{mask}} \in [0.0, 1.0]$（纯 NumPy/SciPy 默认实现，可选 `cv2` 加速）。

### 2.2 验证门禁
- 构造原视频残影超出新 3D 掩膜右方 `10px` 且存在向右 `12px/frame` 运动向量的测试帧，断言联合运动膨胀掩膜 `100%` 覆盖残影区且羽化过渡带单调平滑。

---

<a id="pr-11"></a>
## 3. PR-11：16px 边缘环带 CIE LAB 对齐、时域泊松融合与 ACEScg 高阶光学增量合成

- **建议标题**: `feat(composite): add 16px border-ring LAB matching, multi-band blend, and ACEScg optical pass [model-tag]`
- **前置依赖**: `PR-09`, `PR-10`
- **代码净增预算**: `~510` 行
- **涉及文件清单**:
  - 新增：`python/synapse/composite/blend.py`, `tests/test_composite_blend.py`
  - 修改：`python/synapse/composite/__init__.py`

### 3.1 精确落地规格
1. **`border_ring_lab_match(patch_srgb, bg_srgb, binary_mask, ring_width_px=16)`**：
   - 提取外围环形参考带 $\Omega_{\text{ring}} = \text{Dilate}_{16}(M) \setminus M$ 与掩膜内侧边缘带；
   - 在 CIE LAB 空间对 $L, a, b$ 三通道做带 $\epsilon = 10^{-4}$ 保护的均值-标准差仿射传递（限制最大缩放倍率 `[0.5, 2.0]` 防极端退化区色彩爆炸）。
2. **`composite_surgical_sequence(...)` 时空无缝融合**：
   - 空间域采用 4 层拉普拉斯金字塔多频段融合（Multi-Band Pyramid Blending，纯向量化高斯核实现，消除低频色阶切边同时保留高频纹理）；
   - 时间域对修补掩膜边界带施加相邻帧光流引导指数移动平均（EMA），抑制修补边缘闪烁。
3. **`composite_optical_increments(...)`（Pipeline 6 物理阴影与焦散叠加）**：
   - 将底片 LDR 视频经 `soft_knee_inverse_tonemap` + `linear_srgb_to_acescg` 升维至线性 ACEScg 空间；
   - 以 FGPP `Cryptomatte.ID` 边界与底片灰度图作引导图，执行半径 $r = 4\text{px}$ 的快速箱式引导滤波（Guided Filter），对齐 $\le 3\text{px}$ 的轮廓偏差；
   - 按物理辐射传输方程执行乘性接触阴影调制与加性 HDR 焦散叠加：$L_{\text{out}} = L_{\text{base}}^{\text{ACEScg}} \cdot \tilde{S}_{\text{contact}} + \tilde{L}_{\text{caustics}}^{\text{ACEScg}}$，最后映射回显示空间。

### 3.2 验证门禁
- 在前景补丁人为注入 `+15%` 亮度偏移与 `+1200K` 暖色偏后执行环带匹配与金字塔融合，实测边界环带平均 `CIEDE2000 < 2.5`；接触阴影叠加测试断言高光与暗部无死黑截断且零 `NaN`。

---

<a id="pr-12"></a>
## 4. PR-12：`vg-core` / `foundary` 薄桥接层、Track-B 白模反投硬拦截门禁与 HITL 15 分钟熔断器

- **建议标题**: `feat(bridge): implement thin vg-core/foundary bridges, Track-B anti-paradox gate, and 15m HITL circuit breaker [model-tag]`
- **前置依赖**: `PR-03`
- **代码净增预算**: `~480` 行
- **涉及文件清单**:
  - 新增：`python/synapse/bridge/__init__.py`, `python/synapse/bridge/vg_bridge.py`, `python/synapse/bridge/foundary_client.py`, `python/synapse/bridge/hitl.py`, `tests/test_bridge_and_hitl.py`

### 4.1 精确落地规格
1. **`bridge/vg_bridge.py`（严守 NG-3 零重写红线）**：
   - 封装对 `vg-core` `pybind11` 模块（`vg.material.delight`, `vg.geometry.quad_remesh`, `vg.geometry.check_manifold`, `vg.actor.solve_zero_slip_ik`, `vg.recon.refine_pose_6dof`）的零拷贝 Buffer/NumPy 调用；当环境未编译安装 `vg` 原生扩展时，支持注入测试 `MockVgBackend` 或在生产模式抛出 `BackendUnavailableError (exit_code=30)`；
   - **强制内建 Track-B 白模反投防呆门禁 (`enforce_distillation_geometry_gate`)**：校验资产元数据，凡 `track == "track_b_whitebox"` 且 `has_gen3d_highpoly is False`（或角色网格未通过 `validate_whitebox_coarseness_floor`），在进入 UV 贴图反投前 **100% 硬拦截并抛出 `ContractValidationError (exit_code=10)`**，附带 `agent_fix_hint="Run foundary TRELLIS/Hunyuan3D reconstruction and quad_remesh before texture back-projection."`。
2. **`bridge/foundary_client.py`**：
   - 定义调用 `foundary` 标准推理服务（`TRELLIS`、`Hunyuan3D-2`、`SAM2`、`GVHMR`、`Visual-Jev`）的请求/响应 Pydantic 模型、Presigned URL 载荷协议与本地 `MockFoundaryTransport`。
3. **`bridge/hitl.py`（RFC 0020 五元组与 15 分钟工程熔断器）**：
   - 定义 `OperatorIdentity(kind: Literal["human_l1", "human_l2", "agent"], operator_id: str, model_urn: str | None)` 与 `CommandIRRecord(intent_group_id, intent, state_before_uri, actions, state_after_uri, feedback, elapsed_seconds)`；
   - 实现 `evaluate_hitl_circuit_breaker(elapsed_seconds: float, domain: Literal["video_shot", "asset_3d"], max_seconds: float = 900.0)`：当 `elapsed_seconds > 900.0` 时抛出或返回熔断动作（`video_shot` → `FALLBACK_REBAKE_3D_FGPP`；`asset_3d` → `FALLBACK_SINGLE_VIEW_DECAL_OR_PROCEDURAL`，对应退出码 `40`）。

### 4.2 验证门禁
- 单测覆盖：1) `track_b_whitebox` 未经 `Gen3D` 直接调用贴图烘焙 100% 触发退出码 `10`；2) `elapsed_seconds = 900.0` 正常放行，`elapsed_seconds = 900.1` 精确触发熔断降级（退出码 `40`）。

---

<a id="pr-13"></a>
## 5. PR-13：四维 16 项跨模态评测指标体系与三级统一质量门禁档位

- **建议标题**: `feat(metrics): add 4-pillar 16-metric evaluation suite and 3-tier quality gate profiles [model-tag]`
- **前置依赖**: `PR-05`, `PR-09`
- **代码净增预算**: `~640` 行
- **涉及文件清单**:
  - 新增：`python/synapse/metrics/profiles.py`, `python/synapse/metrics/temporal.py`, `python/synapse/metrics/fgpp_fidelity.py`, `python/synapse/metrics/roundtrip.py`, `python/synapse/metrics/style_consistency.py`, `tests/test_metrics.py`
  - 修改：`python/synapse/metrics/__init__.py`

### 5.1 精确落地规格
1. **`metrics/profiles.py`（对齐 `DESIGN_REVIEW_AND_ALIGNMENT.md` §3.4）**：
   - 固化 `GateProfileName = Literal["track_b_video", "track_a_asset", "rsi_promotion"]` 三套权威门禁字典及统一判定器 `evaluate_metrics_gate(measured: dict[str, float], profile: GateProfileName, scenario_overrides: dict | None = None) -> GateVerdict`。
2. **四维 16 项指标纯 NumPy/SciPy 高精度实现（GPU 特征指标按需懒加载）**：
   - `temporal.py`：`temporal_warp_error`（含遮挡掩膜加权）、`flicker_score`（时域二阶拉普拉斯能量）、`fvd_feature_distance`（可选 `[gpu]` 特征钩子）；
   - `fgpp_fidelity.py`：`depth_rmse_m`、`scale_invariant_depth_error`、`normal_angular_error_deg`、`flow_endpoint_error_px`、`silhouette_iou`；
   - `roundtrip.py`：双边 KD-Tree (`scipy.spatial.cKDTree`) 加速的 `chamfer_mm` 与 `f_score_1mm`、有向四面体求积 `volume_breathing_pct`、触地相 `foot_sliding_cm`、骨骼层级 `bone_length_drift_pct`、去光照 `albedo_delta_e00`；
   - `style_consistency.py`：`border_ring_ciede2000`、`light_dir_error_deg`、`tone_curve_rmse`、`histogram_intersection`。

### 5.2 验证门禁
- 黄金解析解验证：1) GT 自身比对所有误差指标严格 `== 0.0`（相似度指标 `== 1.0`）；2) 将单位球顶点整体平移 `3.0mm`、法线旋转 `10.0°`、足底节点水平位移 `0.4cm`，实测值与解析真值绝对误差 $< 10^{-4}$。

---

<a id="pr-14a"></a>
## 6. PR-14a：离线评测串联器 `BenchPipelineRunner` 与 `synapse-bench evaluate` 桥接接口

- **建议标题**: `feat(pipeline): add offline BenchPipelineRunner and synapse-bench evaluate command [model-tag]`
- **前置依赖**: `PR-05`, `PR-13`
- **代码净增预算**: `~390` 行
- **涉及文件清单**:
  - 新增：`python/synapse/pipeline/bench_runner.py`, `tests/test_bench_runner.py`
  - 修改：`python/synapse/pipeline/__init__.py`, `python/synapse/cli/__init__.py`

### 6.1 精确落地规格
1. **`BenchPipelineRunner`**：串联 `Scenario 加载 → FGPP 加载或软光栅即时生成 → VideoModelAdapter 推理 → 四维指标计算 → GateProfile 准出判定`，输出结构化 `ScenarioRunReport`（含分阶段毫秒级耗时拆解）。
2. **补齐 CLI `synapse-bench run` 与 `synapse-bench evaluate --bundle <dir>`（对齐 `RV-02`）**：
   - `synapse-bench evaluate --bundle <dir> [--profile track_b_video] [--json]` 直接读取符合 `InspectionBundle` 目录约定的资产/帧/模型计算全套指标并写出 `metrics/comparison.json`，为 `Phase M8` Electron Data Inspector 提供零耦合后端契约。

### 6.2 验证门禁
- 在 `ubuntu24` 运行 `synapse-bench run --scenario t1_rigid_cube_orbit --adapter mock --json` 返回退出码 `0` 且各项指标非空合法。

---

<a id="pr-14b"></a>
## 7. PR-14b：无状态容器执行核 `JobBundleWorker`（RFC 0039 闭包与三级显存弹性降级）

- **建议标题**: `feat(pipeline): implement stateless RFC 0039 JobBundleWorker with proactive OOM fallback [model-tag]`
- **前置依赖**: `PR-07`, `PR-08`, `PR-11`, `PR-12`, `PR-14a`
- **代码净增预算**: `~480` 行
- **涉及文件清单**:
  - 新增：`python/synapse/pipeline/bundle_worker.py`, `tests/test_bundle_worker.py`
  - 修改：`python/synapse/pipeline/__init__.py`, `python/synapse/cli/__init__.py`

### 7.1 精确落地规格
1. **`BundleManifest` (`synapse.job_bundle.v1.0`) 强类型契约**：
   - 字段：`schema_version`, `job_id`, `stage`（首批支持 `fgpp_validate`, `neural_video_generate`, `style_extract_and_map`, `surgical_composite`，并采用可扩展 Stage Handler 注册表以便 `M5/M6/M7` 动态注册新阶段！）, `inputs`, `outputs`, `params`, `hardware_budget` (`free_vram_gb`), `traceparent`, `tracestate`。
2. **前瞻显存自愈调度**：在 `neural_video_generate` 启动前调用 `MemoryForecaster`；当 $0.80 < R \le 1.05$ 自动注入 `spatial_tile="2x2"`，当 $R > 1.05$ 自动切换 `window_frames=49` 短因果级联；若仍超硬上限则零崩溃退出 `30` 并写出 `result_manifest.json`（含输出文件 SHA256、`reanchor_requests`、耗时遥测与 `agent_fix_hint`）。
3. **挂载 CLI**：`synapse worker run --bundle <path> [--json]`，单次执行完毕立即退出，零常驻线程。

### 7.2 验证门禁
- 构造模拟本地挂载 `JobBundle` 测试夹具，子进程实跑全部 4 种 `stage` 及模拟 `free_vram_gb=16.0` 触发自动降级分支，验证 `result_manifest.json` SHA256 完整准确。

---

<a id="pr-15"></a>
## 8. PR-15：`Synapse-Bench` 扩充至 30 个 T1–T5 场景、Humanoid-53 铰接代理白模与自包含 HTML 报告

- **建议标题**: `feat(bench): expand to 30 T1-T5 scenarios, Humanoid-53 >=1500v proxy, and offline HTML report [model-tag]`
- **前置依赖**: `PR-01`, `PR-14a`
- **代码净增预算**: `~720` 行
- **涉及文件清单**:
  - 新增：`python/synapse/bench/registry.py`, `python/synapse/bench/report.py`, `python/synapse/bench/scenarios/t{1..5}_*.json`（补齐至 30 个）, `tests/test_bench_full_suite.py`
  - 修改：`python/synapse/bench/procedural.py`, `python/synapse/cli/__init__.py`

### 8.1 精确落地规格
1. **`procedural.py::generate_humanoid_proxy(pose_params=None, target_vertices=1820)`**：
   - 程序化构建具备头颈、躯干、骨盆、双上臂、双前臂、双手掌、双大腿、双小腿、双足独立分节拓扑且关节处水密缝合的三角网格（默认顶点数 `~1,820 >= 1500`，非流形边 `== 0`，100% 通过 `validate_whitebox_coarseness_floor`）；支持传入周期步行相位 `phase ∈ [0, 2π)` 生成无滑步刚体骨长恒定的动画序列！
2. **`registry.py`**：支持按 `SYNAPSE_ASSET_CACHE`（默认 `~/.cache/synapse/assets`）按需拉取可选公开 glTF 模型并强制校验 `sha256`（离线 CI 环境下 30 个标准场景默认全部可由内置程序化几何 + Humanoid-53 零网络运行！）。
3. **30 个标准场景矩阵**：`T1` 静态刚体（8 个）、`T2` 动态与骨骼（7 个）、`T3` 多体遮挡交互（5 个）、`T4` 局部重渲与冷暖大光比风格对齐（6 个）、`T5` 97 帧多锚点端到端短片（4 个）。
4. **`report.py` 零外部依赖 HTML 仪表盘**：内联 CSS + SVG 四维雷达图、30 场景 Pass/Fail 矩阵、各阶段耗时条形图与异常帧 Base64 缩略图。

### 8.2 验证门禁
- 在 `ubuntu24` 断网模式下执行 `synapse-bench run --all --adapter mock --report /tmp/synapse_bench.html --json`：30 个场景 100% 通过、CPU 总运行时间 `< 30s`、HTML 文件无任何 `http://`/`https://` 外部脚本样式引用且体积 `< 2MB`。
