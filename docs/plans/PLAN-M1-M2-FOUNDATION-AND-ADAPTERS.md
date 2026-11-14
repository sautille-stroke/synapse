# 细分执行计划 1：Phase M1–M2 地基归正、FGPP 跨语言契约闭环与双轨视频生成适配 (`PR-01` ~ `PR-08`)

- **所属总计划**: [docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) (`v3.1` Master Plan, Horizon 1: M1–M2)
- **前置对齐决议**: [docs/DESIGN_REVIEW_AND_ALIGNMENT.md](../DESIGN_REVIEW_AND_ALIGNMENT.md) (`RV-01`, `RV-02`, `RV-05`, `RV-06`, `RV-08`, `RV-10`)
- **覆盖范围**: `PR-01` ~ `PR-08`（共 8 个原子 PR，预估 10.5 人天）
- **测试主力环境**: `ssh ubuntu24 "cd ~/work/dervgen/synapse && pytest ... && mypy --strict python/synapse && ruff check python/ tests/"`

---

<a id="pr-01"></a>
## 1. PR-01：修复现存 `bench` 缺陷、$T=4k+1$ 帧数法则、纯 NumPy 软光栅整合与仓库卫生

- **建议标题**: `fix(bench): resolve camera/packaging bugs, add pure-NumPy rasterizer fallback, and align T=4k+1 [model-tag]`
- **前置依赖**: 无（Wave 1 首发）
- **代码净增预算**: `~550` 行
- **涉及文件清单**:
  - 修改：`.gitignore`, `pyproject.toml`, `python/synapse/bench/camera.py`, `python/synapse/bench/conditions.py`, `python/synapse/bench/scenario.py`, `python/synapse/viewer/server.py`, `tests/test_bench.py`, `tests/test_dataset_and_viewer.py`
  - 新增：`python/synapse/py.typed`, `python/synapse/bench/rasterizer.py`, `python/synapse/bench/model_loader.py`
  - 迁入：`data/scenarios/*.json` → `python/synapse/bench/scenarios/*.json`（保留根级 `data/scenarios/` 软链或同步兼容寻址）
  - 删除（整合完毕后清理空壳）：`python/synapse/generators/*`, `python/synapse/models/*`

### 1.1 精确落地规格
1. **环绕轨迹 360° 首尾不重合 (`camera.py::_generate_orbit`)**：
   - 签名新增 `endpoint: bool | None = None`；当 `endpoint is None` 时令 `endpoint = abs(azimuth_range_deg) < 360.0`；
   - 采样因子 `t = i / max(num_frames - 1, 1) if endpoint else i / max(num_frames, 1)`，保证 `num_frames=8, azimuth_range_deg=360` 精确产出 `0°, 45°, 90°, ..., 315°`。
2. **天顶/天底奇异矩阵防护 (`camera.py::_look_at`)**：
   - 计算 `forward = normalize(target - eye)`；若 `abs(np.dot(forward, up_norm)) > 0.9999`，自动切换正交参考轴 `up_fallback = np.array([0.0, 0.0, -1.0 if forward[1] > 0 else 1.0])`，并断言 `abs(np.linalg.det(view[:3, :3]) - 1.0) < 1e-6`。
3. **条件因子与相机参数打通 (`conditions.py` & `camera.py`)**：
   - `CameraParams` 新增字段 `aperture_fstop: float = 0.0`、`shutter_angle_deg: float = 180.0`；
   - `Conditions.from_dict(d)` 对未知因子键或非法 `L1~L4` 值立即抛出 `ValueError`；
   - `generate_cameras(scenario)` 自动读取 `F08_scale` 乘算相机半径/距离，`F09_dof` 写入 `aperture_fstop`，`F10_motion_blur` 写入 `shutter_angle_deg`。
4. **`Wan 2.1` $T = 4k + 1$ 场景规范与 Wheel 资源寻址 (`scenario.py` & `scenarios/*.json`)**：
   - `Scenario` 新增 `temporal_mode: Literal["continuous_clip", "discrete_views"] = "continuous_clip"`；当 `temporal_mode == "continuous_clip"` 时校验 `(num_frames - 1) % 4 == 0`；
   - 将连续镜头场景帧数统一更新为 `97`（对齐 `0, 48, 96` 锚点），`t1_rigid_sphere_sparse.json` 设为 `temporal_mode="discrete_views", num_frames=8`；
   - `pyproject.toml` 配置 `[tool.setuptools.package-data] synapse = ["py.typed", "bench/scenarios/*.json"]`，`scenario.py` 使用 `importlib.resources.files("synapse.bench") / "scenarios"` 为主路径。
5. **消除隐式重型依赖与重构零依赖软光栅 (`RV-08`)**：
   - `bench/rasterizer.py`：提供**纯 NumPy 向量化包围盒重心坐标光栅化**作为默认路径；若检测到已安装可选包 `numba`，才动态启用 JIT 核加速（确保仅安装 `numpy` + `pydantic` 时 `SoftwareRasterizer` 100% 可运行！）；同时直接输出完整 7 逻辑组 `FGPPFrame`（含 `MurmurHash3` `uint32` `cryptomatte_id`）；
   - `bench/model_loader.py`：内置零依赖纯 Python/NumPy `.obj` 解析器与程序化网格桥接；`trimesh` / `PIL` 改为函数内按需延迟导入，移除 `H:/dervgen/...` 硬编码路径；
   - `pyproject.toml` 新增可选组 `mesh = ["trimesh>=4.0.0"]`, `jit = ["numba>=0.58.0"]`。
6. **本地预览器安全绑定与 `.gitignore` 卫生 (`RV-10`)**：
   - `viewer/server.py` 将 `TCPServer(("", cur_port), ...)` 改为默认 `host: str = "127.0.0.1"`，`TCPServer((host, cur_port), ...)`；
   - `.gitignore` 补充忽略 `data/sessions/`, `data/datasets/`, `.pytest_tmp/`, `dist/`, `build/`, `*.egg-info/`。

### 1.2 验证门禁（防静默失败三问）
- 在 `ubuntu24` 创建仅含 `pip install ./dist/synapse-*.whl pytest`（**不装 `numba`、`trimesh`、`opencv`**）的洁净虚拟环境，`cd /tmp` 运行 `pytest`：验证 `list_scenarios()` 非空、`elevation=±90°` 不崩溃、`sparse_8view` 精确 45° 步进、`SoftwareRasterizer` 纯 NumPy 模式成功产出非零深度与法线帧。

---

<a id="pr-02"></a>
## 2. PR-02：`@aether//fgpp` 跨语言契约自动守卫与 `fgpp.v1.0` 列主序元数据互通

- **建议标题**: `feat(fgpp): add wire_guard codegen and align 15-channel @aether//fgpp contract [model-tag]`
- **前置依赖**: 无（与 `PR-01` 并行）
- **代码净增预算**: `~480` 行
- **涉及文件清单**:
  - 新增：`tools/wire_guard.py`, `python/synapse/fgpp/schema.py`, `tests/test_fgpp_contract.py`
  - 修改：`python/synapse/fgpp/frame.py`, `python/synapse/fgpp/__init__.py`

### 2.1 精确落地规格
1. **`tools/wire_guard.py` 跨语言契约抽取与校验器**：
   - 支持 `--aether-header <path>`（默认探测 `../aether/fgpp/include/aether/fgpp/fgpp_types.h`）与 `--check` 模式；
   - 解析 C++ `PixelType`、`ChannelNames`、`getLogicalChannelSpecs()`、`getAlphabeticalEXRChannelSpecs()`、`kMinCharacterWhiteboxVertices (1500)`、`kDefaultPhysicalAnchorIntervalFrames (48)`，与 `python/synapse/fgpp/schema.py` 逐项比对。
2. **`python/synapse/fgpp/schema.py` 强类型常量与物理守卫函数**：
   - `PixelType(IntEnum)`: `UINT32 = 0`, `HALF = 1`, `FLOAT = 2`；
   - 精确锁死 15 个 ASCII 排序标量通道（`BaseColor.B/G/R`, `CameraLinearDepth.Z`, `Cryptomatte.ID`, `DirectRadiance.B/G/R`, `MotionVector.X/Y`, `RoughnessMetallic.G/R`, `WorldNormal.X/Y/Z`）及像素总步长 `FGPP_BYTES_PER_PIXEL = 34`；
   - 向量化实现 `linear_to_log_normalized_depth(z_m, z_near_m, z_far_m)` 与逆变换（对 `z_near_m <= 0`、`z_far_m <= z_near_m`、`NaN/Inf` 零异常钳位）；
   - 实现 `validate_whitebox_coarseness_floor(vertex_count: int, has_articulated_limbs: bool, is_character: bool) -> bool` 与 `is_physical_anchor_frame(frame_idx: int, interval: int = 48) -> bool`。
3. **`python/synapse/fgpp/frame.py` 升级 `fgpp.v1.0` 相机矩阵主序转换**：
   - `CameraParams` 新增 `projection_matrix()` 计算方法、`focal_length_mm: float = 35.0`、`sensor_width_mm: float = 36.0`、`sensor_height_mm: float = 20.25`；
   - `FGPPSequence.save_metadata(output_dir)` 严格写出 `schema_version: "fgpp.v1.0"`, `fps`, `total_frames`, `resolution: [W, H]`, `z_near_meters`, `z_far_meters`, `anchor_interval_frames: 48`；逐帧写出 `frame_index`, `timestamp_sec`, `is_physical_anchor`, `fov_y_deg`, `view_matrix_col_major`（显式 `np.asarray(view_matrix, dtype=np.float64).T.reshape(-1).tolist()`）与 `proj_matrix_col_major`；
   - 新增 `FGPPSequence.load_metadata(input_dir)`：自动将 `_col_major` 16 元素数组 `reshape((4, 4)).T` 还原为 NumPy 行主序矩阵，并校验 $|\det(R) - 1.0| < 10^{-5}$。

### 2.2 验证门禁
- `python tools/wire_guard.py --check` 返回 `0`；`tests/test_fgpp_contract.py` 验证任意非对称相机位姿（含平移 `[1.2, -3.4, 5.6]` 与俯仰偏航旋转）序列化至 `metadata.json` 再读回的最大元素误差 $< 10^{-12}$，且与 C++ OpenGL/Fulgur 列主序内存布局完全一致。

---

<a id="pr-03"></a>
## 3. PR-03：多通道异构精度 OpenEXR 2.0 读写、边界消毒器与 Torch DLPack 零拷贝桥接

- **建议标题**: `feat(fgpp): implement multi-channel OpenEXR 2.0 I/O, sanitizer, and DLPack Torch bridge [model-tag]`
- **前置依赖**: `PR-02`
- **代码净增预算**: `~520` 行
- **涉及文件清单**:
  - 新增：`python/synapse/fgpp/io.py`, `python/synapse/fgpp/torch_bridge.py`, `tests/test_fgpp_io.py`
  - 修改：`python/synapse/fgpp/__init__.py`

### 3.1 精确落地规格
1. **`fgpp/io.py` 多通道原生字节读写**：
   - `export_fgpp(seq: FGPPSequence, output_dir: Path, compression: str = "ZIP") -> Path`：写出 `frame_{idx:06d}.exr` + `metadata.json`；严格按每个标量通道的原生 `np.float16` / `np.float32` / `np.uint32` `.tobytes()` 写入 OpenEXR `Header`（**严禁将 `Cryptomatte.ID` 经 `float32` 中转，防止 `> 2^24` 哈希值精度截断**）；
   - `sanitize_fgpp_frame(frame: FGPPFrame) -> FGPPFrame`：将 `depth` 中 `NaN / Inf / <= 0` 的天空/背景像素统一钳制为 `camera.z_far`，对应 `cryptomatte_id` 置 `0`，`world_normal` 单位化（零模长像素置 `[0, 0, 1]`）；
   - `resize_fgpp_frame(frame: FGPPFrame, target_hw: tuple[int, int]) -> FGPPFrame`：浮点通道使用边缘保持插值，**`cryptomatte_id` 硬编码强制使用最近邻（Nearest-Neighbor）采样**；
   - `import_fgpp(input_dir: Path, lazy: bool = True, channels: Sequence[str] | None = None)`：支持惰性帧迭代器与子通道按需加载，控制单帧峰值内存。
2. **`fgpp/torch_bridge.py` 零拷贝张量桥接**：
   - 延迟导入 `torch`；实现 `frame_to_tensors(frame: FGPPFrame, device: str = "cpu") -> dict[str, torch.Tensor]` 与 `tensors_to_frame(...)`；
   - CPU 连续数组使用 `torch.from_numpy` / `torch.from_dlpack` 共享底层内存；针对 PyTorch 旧版本对 `uint32` 算子支持受限的情况，提供无损 `.view(torch.int32)` 双向重解释包装，保证 `0` 字节拷贝与逐 bit 保真。

### 3.2 验证门禁
- 在 `ubuntu24` 构造含 `0xDEADBEEF`、`0xFFFFFFFF`、`0x80000001` 等大整数 ID 的 15 通道 `FGPPSequence`，导出 EXR 再读回断言 `np.array_equal(out.cryptomatte_id, gt.cryptomatte_id)` 100% 成立，`FLOAT32` 深度逐 bit 一致，`HALF` 通道最大误差 $\le 5 \times 10^{-4}$。

---

<a id="pr-04"></a>
## 4. PR-04：Agent-Native CLI 基座、确定性退出码、`MemoryForecaster` 与 W3C 追踪穿透

- **建议标题**: `feat(cli): build Agent-Native CLI with deterministic exit codes, MemoryForecaster, and W3C trace [model-tag]`
- **前置依赖**: `PR-01`, `PR-03`
- **代码净增预算**: `~460` 行
- **涉及文件清单**:
  - 新增：`python/synapse/errors.py`, `python/synapse/telemetry.py`, `tests/test_cli.py`
  - 修改：`python/synapse/cli/__init__.py`, `pyproject.toml`

### 4.1 精确落地规格
1. **`errors.py` 结构化异常与退出码体系**：
   - `ExitCode(IntEnum)`: `OK = 0`, `CONTRACT_ERROR = 10`, `RESOURCE_MISSING = 20`, `BACKEND_UNAVAILABLE = 30`, `CIRCUIT_BREAKER = 40`；
   - 基类 `SynapseError(code, message, exit_code, agent_fix_hint, details)` 及派生类 `ContractValidationError(10)`、`AssetNotFoundError(20)`、`BackendUnavailableError(30)`、`CircuitBreakerError(40)`、`QualityGateError(40)`，内置 `.to_json_dict(trace_ctx)`。
2. **`telemetry.py` 显存前瞻预测与 W3C Trace**：
   - `TraceContext.from_env_or_dict(...)`：解析标准 W3C `traceparent` (`00-<trace_id>-<span_id>-<flags>`) 与 `tracestate`；
   - `MemoryForecaster.estimate_dit_peak_gb(num_frames, height, width, model_variant="wan2.1_14b", dtype_bytes=2)`：按解析式估算权重 + 激活 + 时空注意力峰值显存，并给出三级弹性降级决议 `FallbackLevel`（`LEVEL_0_FULL` / `LEVEL_1_SPATIAL_TILE_2X2` / `LEVEL_2_CAUSAL_SHORT_WINDOW_49F` / `REJECT_OOM`）。
3. **双入口 CLI 收敛 (`pyproject.toml` & `cli/__init__.py`)**：
   - `[project.scripts]` 收敛为 `synapse = "synapse.cli:main_synapse"` 与 `synapse-bench = "synapse.cli:main_bench"`；同时调整 `pyproject.toml` 将 `torch>=2.0.0` 独立至 `[gpu]` 可选组，`[metrics]` 保持纯 CPU（`scipy`, `scikit-image`）；
   - 落地首批命令：`synapse fgpp validate`, `synapse fgpp inspect [--export-vis]`, `synapse generate`, `synapse view`, `synapse-bench list`, `synapse-bench export-fgpp`；
   - `--json` 模式下捕获一切 `SynapseError`，向 `stdout` 输出唯一合法 JSON 对象（含 `status`, `exit_code`, `error_code`, `message`, `agent_fix_hint`, `trace_id`），严禁混入 `print` 杂讯。

### 4.2 验证门禁
- `tests/test_cli.py` 使用 `subprocess.run` 实测 `0 / 10 / 20 / 30` 路径：所有命令 `json.loads(proc.stdout)` 零异常通过；`export-fgpp --scenario t1_rigid_cube_orbit` 导出的目录立即通过 `synapse fgpp validate --json`（退出码 `0`）。

---

<a id="pr-05"></a>
## 5. PR-05：`VideoModelAdapter` 插件注册表与零 GPU 确定性 `MockVideoAdapter`

- **建议标题**: `feat(adapters): add VideoModelAdapter registry and deterministic CPU MockVideoAdapter [model-tag]`
- **前置依赖**: `PR-03`
- **代码净增预算**: `~420` 行
- **涉及文件清单**:
  - 新增：`python/synapse/adapters/base.py`, `python/synapse/adapters/mock.py`, `tests/test_adapters_base.py`
  - 修改：`python/synapse/adapters/__init__.py`

### 5.1 精确落地规格
1. **`adapters/base.py`**：
   - 定义 `AdapterTier = Literal["whitebox", "blackbox", "mock"]`、`PreparedConditionPayload`、`VideoGenerationConfig(seed, fps, width, height, guidance_scale, num_inference_steps, extra)`、`VideoResult(frames_srgb_u8: np.ndarray, fps: float, seed: int, adapter_id: str, metadata: dict)`；
   - 抽象基类 `VideoModelAdapter`（`adapter_id`, `tier`, `prepare_conditions`, `generate`）及全局注册表 `@register_adapter` / `create_adapter(name, **kwargs)` / `list_adapters()`。
2. **`adapters/mock.py` (`MockVideoAdapter`)**：
   - 纯 NumPy 确定性着色合成：读取 `FGPPSequence` 的 `base_color`、`world_normal`、`direct_radiance` 与 `depth`，施加可配色温球谐光照与胶片微噪（由 `np.random.Generator(PCG64(seed))` 严格锁死）；
   - **可控缺陷注入器 (`DefectInjectionConfig`)**：支持按帧区间 `[t0, t1]` 与目标 `entity_id` 精确注入：(a) `silhouette_shift_px`（模拟穿模/轮廓偏移）、(b) `flicker_amplitude`（模拟帧间高频闪烁）、(c) `kelvin_shift`（模拟局部色温漂移），为后续 `M3` 合成与 `M4` 指标提供精确数学真值靶标。

### 5.2 验证门禁
- 相同 `seed` 两次调用 `MockVideoAdapter.generate()` 输出数组 `np.array_equal` 为 `True`；注入 `+12px` 轮廓偏移后仅目标 `Cryptomatte.ID` 邻域发生变化。

---

<a id="pr-06"></a>
## 6. PR-06：长镜头因果流式滑窗（含 3 帧前导暖启动）、拓扑突变重锚定与 3D VAE 光流池化

- **建议标题**: `feat(adapters): implement causal window scheduler with 3f warm-start, discontinuity detector, and latent flow pooling [model-tag]`
- **前置依赖**: `PR-05`
- **代码净增预算**: `~580` 行
- **涉及文件清单**:
  - 新增：`python/synapse/adapters/scheduler.py`, `tests/test_condition_scheduler.py`
  - 修改：`python/synapse/adapters/__init__.py`

### 6.1 精确落地规格
1. **因果滑窗切片与 3 帧暖启动对齐 (`CausalWindowPlan` & `ConditionScheduler`, 对齐 `RV-05`)**：
   - 标准参数：`window_frames = 97` ($\tau=25$)、`stride_frames = 64` ($\Delta\tau=16$)、`overlap_frames = 33` ($\tau_{\text{overlap}}=9$)；
   - 对第 $k \ge 1$ 个滑窗（名义帧 `[64k .. 64k+96]`），记录 `causal_warmup_prefix_indices = [64k-3, 64k-2, 64k-1]` 供 3D 因果 VAE 暖启动卷积状态；
   - 保方差 Sigmoid 因果退火函数 `fuse_overlapping_latents(z_prev, z_curr, lam=8.0)`：对重叠首步 $\tau'=0$ 强制 $\alpha(0) = 0.0$（保留 `z_prev[16]`），对 $\tau'=1..8$ 与 `z_prev[17..24]` 执行单位方差归一化加权 $\mathbf{z}_{\text{fused}} = \frac{(1-\alpha)\mathbf{z}_k + \alpha\mathbf{z}_{k+1}}{\sqrt{(1-\alpha)^2 + \alpha^2}}$。
2. **拓扑突变检测与两级重锚定 (`DiscontinuityDetector`, 对齐 `RV-06`)**：
   - 按公式计算逐帧突变指数 $\kappa(t) = 0.4 \cdot \text{norm\_motion\_jump}(t) + 0.4 \cdot \text{normal\_cosine\_drop}(t) + 0.2 \cdot (1 - \text{mask\_iou}(t))$；
   - 当 $\kappa(t) > 0.65$ 时，标记 `is_dynamic_anchor = True`、清空跨帧光流累积，并生成 `ReanchorRequest(frame_index=t, subframe_window=(-2, 2), kappa=kappa_val)`。
3. **3D Causal VAE 潜空间光流积分与 $8\times 8$ 最小深度池化 (`compute_latent_flow_operator`)**：
   - 输入屏幕光流 `[T, H, W, 2]` 与线性深度 `[T, H, W]`（校验 $(T-1) \bmod 4 == 0$）；
   - 执行跨 4 帧前向欧拉轨迹积分，结合残差阈值生成遮挡掩膜 $O_{\tau \to \tau'} \in \{0, 1\}$（突变帧处强制重置积分器）；
   - 在每个 $8\times 8$ 空间块内选取 `argmin(depth)` 所在前景像素的积分位移向量，输出 `[τ, H/8, W/8, 2]` 潜空间位移网格。

### 6.2 验证门禁
- 1) 构造 385 帧（5 个滑窗）恒定高斯单位方差潜空间张量序列，验证 `fuse_overlapping_latents` 输出方差保持在 `1.0 ± 0.02` 且拼接边界无跳变；
- 2) 构造半径 `32px` 前景球每帧右移 `8px`、背景静止的 17 帧序列，验证池化潜空间前景位移为 `4.0 ± 0.5` 潜步单位；注入单帧法线翻转突变验证 $\kappa > 0.65$ 精确触发积分清零。

---

<a id="pr-07"></a>
## 7. PR-07：Tier-1 白盒 `WanAdapter`（相机空间法线变换、ControlNeXt 张量与软光流偏置）

- **建议标题**: `feat(adapters): implement Tier-1 whitebox WanAdapter with ControlNeXt and soft flow-bias [model-tag]`
- **前置依赖**: `PR-06`
- **代码净增预算**: `~490` 行
- **涉及文件清单**:
  - 新增：`python/synapse/adapters/wan.py`, `tests/test_wan_adapter.py`
  - 修改：`python/synapse/adapters/__init__.py`

### 7.1 精确落地规格
1. **`WanAdapter` 条件张量构建 (`prepare_conditions`)**：
   - 深度：调用 `linear_to_log_normalized_depth` 转为 `[0, 1]`；
   - 法线：利用逐帧 `CameraParams.view_matrix[:3, :3]` 将 `WorldNormal.XYZ` 旋转至相机观察空间 `CamNormal.XYZ`，背景零法线保持 `[0, 0, 1]`；
   - 光照基调：对 `DirectRadiance.RGB` 做软肩色调归一化；拼接为 `[C_cond, T, H, W]` 控制张量；
   - 潜空间光流软偏置载荷：`gamma_flow_bias: float = 0.0`（**未挂载微调控制 LoRA 时默认 `0.0` 防分布崩塌**；挂载后支持 `[0.0, 2.5]` 门控）与遮挡自动释放掩膜；
   - 预留 `lora_manager: MultiLoRAManager | None = None` 插槽供 `Phase M5 (PR-16)` 无缝挂载。
2. **双后端无状态传输契约 (`WanBackendTransport`)**：
   - 提供 `FoundaryHttpTransport`（对接 `foundary` 容器内推理端点，携带 W3C `traceparent` 头）与 `LocalCallableBackend`（供容器内直接绑定 PyTorch Pipeline 或单测注入 Fake Pipeline）；端点超时或连接拒绝时统一抛出 `BackendUnavailableError (exit_code=30)`。

### 7.2 验证门禁
- 单测验证世界法线经正交旋转变换后模长保持 `1.0 ± 1e-3`、朝向相机球面中心点法线 $N_z > 0.99`；模拟后端断连断言精确抛出退出码 `30` 及 `agent_fix_hint`。

---

<a id="pr-08"></a>
## 8. PR-08：Tier-2 黑盒商业适配器（`SeedanceAdapter`、`KlingAdapter`、`VeoAdapter`）

- **建议标题**: `feat(adapters): add Tier-2 blackbox cloud adapters with FGPP anchor degradation [model-tag]`
- **前置依赖**: `PR-06`
- **代码净增预算**: `~450` 行
- **涉及文件清单**:
  - 新增：`python/synapse/adapters/cloud.py`, `tests/test_cloud_adapters.py`
  - 修改：`python/synapse/adapters/__init__.py`

### 8.1 精确落地规格
1. **FGPP 受控降级转换器 (`degrade_fgpp_for_blackbox`)**：
   - 自动按物理锚点（`0, 48, 96...` 及动态重锚点）将长序列切分为 `[0..48]`, `[48..96]` 子片段；
   - 提取段首与段尾帧的 `DirectRadiance.RGB`（经 ACEScg → sRGB 色调映射编码为内存 PNG 字节流）作为 I2V 首尾关键帧锚定；
   - 相机外参编码：既输出逐帧归一化 `[R|t]` 轨迹数组（供支持矩阵轨迹的 `Seedance 2.5` / `Veo` 使用），也通过最小二乘速度分解拟合成结构化运镜枚举（`pan_right`, `dolly_in`, `orbit_ccw` + 强度 `[0, 10]`，供 `Kling 1.5` 使用）；可选导出对数深度与法线边缘引导帧。
2. **安全传输抽象 (`CloudApiTransport`)**：
   - 内置 `client_request_token = sha256(scene_id + segment_range + seed)` 幂等键、指数退避重试与 `MockCloudTransport`（**严禁单元测试发起任何公网 HTTP 请求或读取生产密钥**；缺失 API Token 时抛出 `BackendUnavailableError(30)`）。

### 8.2 验证门禁
- 使用 `MockCloudTransport` 测试 97 帧序列被精确拆分为 `[0..48]` 与 `[48..96]` 两段请求，段间首尾共享第 `48` 帧锚点且合并后总帧数严格还原为 `97` 帧（无重复帧）。
