# Synapse Google Colab 各环节输入与输出规范字典 (I/O Specification)

> 本文档详细定义在 Google Colab GPU 节点上执行 Synapse 算法微调、推理与实验阶段（E1–E5）的数据线格式、张量维度、元数据 Schema、检查点结构及异常退出码契约。

---

## 一、通用契约与坐标系基准 (SSOT)

| 物理量 / 通道 | 规范要求 | 校验断言 |
| :--- | :--- | :--- |
| **空间分辨率** | 标准评测分辨率 $832 \times 480$（16:9，必须为 16 的整数倍） | `H % 16 == 0 and W % 16 == 0` |
| **时序帧率** | 默认 24.0 fps，因果窗口步长 $W=97$ 帧（约 4 秒） | `fps == 24.0` |
| **相机矩阵主序** | **列主序（Column-Major）**，且 $\det(\mathbf{R}) = +1$（右手系无镜像） | `abs(det(world_from_camera[:3, :3]) - 1.0) < 1e-4` |
| **线性深度** | 绝对米制正值 $z \in (0, +\infty)$，非投影空间深度 | `np.all(linear_depth > 0.0)` |
| **世界空间法线** | 单位模长向量 $\|\mathbf{n}\| = 1.0$，空背景填 `(0, 0, 0)` | `abs(norm(n) - 1.0) < 1e-3` (前景像素) |
| **实体 ID** | `Cryptomatte.ID` 必须为原生无符号 32 位整型（`UINT32`），**严禁双线性插值** | `dtype == np.uint32` |

---

## 二、各环节详细输入与输出定义

### 1. 数据准备与打包环节 (`scripts/prepare_datasets.py`)

#### 输入 (Inputs)
- `fulgur` 二进制：`bazel-bin/course/102-pbr-direct`（Linux / Vulkan 无头渲染）
- `caelum-core` 渲染资产：`cornell_spheres_17f`（离线路径追踪真值对）
- 命令行参数：`--fulgur-repo <path>`, `--out-dir <path>`, `--num-frames 17`

#### 输出 (Outputs)
- **Track-B 实时序列**：`data/datasets/track_b_fulgur/course_102_pbr_17f/`
  - `frame_000000.exr` .. `frame_000016.exr`：15 标量通道 OpenEXR (`34 B/px`, HALF/UINT32)
  - `metadata.json`：符合 `fgpp.v1.0` Schema，包含相机外参/内参、近远裁切面、渲染轨迹标签
- **Track-B 97 帧长时序序列**：`data/datasets/track_b_fulgur/course_102_pbr_97f/`
  - 包含 0..63 帧平滑运镜、第 64 帧切镜突变点，以及 t=0, 48, 96 处的物理重锚定关键帧
- **Track-A 离线真值对**：`data/datasets/track_a_caelum/cornell_spheres_17f/`
  - `frame_XXXXXX.exr` (15 通道 FGPP) + `beauty_XXXXXX.exr` (离线路径追踪 Ground Truth)
- **校验清单**：`data/manifests/golden_dataset_manifest.json`（各文件 SHA256 与字节大小）
- **Colab 独立数据包**：`synapse_colab_bundle_v1.tar.gz`（解压即用，大小约 300–450 MB）

---

### 2. E1 阶段：7 通道空间条件控制与消融 (`WanAdapter`)

#### 输入 (Inputs)
- `seq`: `FGPPSequence`（包含 $T$ 帧 15 通道 OpenEXR 与相机元数据）
- `cfg`: `VideoGenerationConfig(width=832, height=480, num_frames=T, fps=24.0)`

#### 输出 (Outputs)
- `payload`: `ConditionPayload`
  - `control_tensor`: 形状为 **`[7, T, H, W]`** 的 `float32` 张量：
    - `ch 0`: 对数归一化线性深度 $d_{\log} \in [0.0, 1.0]$
    - `ch 1..3`: 世界空间单位法线 $(n_x, n_y, n_z) \in [-1.0, 1.0]$（源自 FGPP `WorldNormal.XYZ`）
    - `ch 4..5`: 归一化帧间运动向量 $(v_x, v_y) \in [-1.0, 1.0]$
    - `ch 6`: 重投影遮挡掩码 $M_{\text{occ}} \in [0.0, 1.0]$
  - `camera_embedding`: Plücker 光线几何嵌入张量 `[6, T, H, W]`（**设计规划字段，当前 `PreparedConditionPayload` 未实现**）
  - `text_prompt`: 结构化提示词（可选）
- **消融子档输出**：
  - `E1-A_rgb_only`: 仅 RGB，控制通道全 0（L2 范数 = 0）
  - `E1-B_rgb_depth`: 激活通道 `[0]`
  - `E1-C_rgb_depth_normal`: 激活通道 `[0, 1, 2, 3]`
  - `E1-D_full_7ch`: 全通道激活 `[0, 1, 2, 3, 4, 5, 6]`

---

### 3. E2 阶段：ACEScg 内蕴风格提取与 16px 手术刀合成 (`synapse.style` & `synapse.composite`)

#### 输入 (Inputs)
- 原始场景 sRGB 帧张量：`[T, H, W, 3]` (`uint8`)
- 配对 `FGPPSequence`（提供直接光照、反照率及 `Cryptomatte.ID`）
- 瑕疵精修补丁图像：`patch_rgb` `[H, W, 3]` (`float32`, [0, 1])

#### 输出 (Outputs)
- **风格配置文件**：`StyleProfile` 字典（符合 `style_profile.v2.0.0` Schema）
  - `albedo_mean_acescg`: `[R, G, B]`
  - `key_light_color_temperature_kelvin`: 如 `5600.0`
  - `shadow_contrast_ratio`: 如 `0.18`
  - `highlight_rolloff`: 软肩压缩控制曲线参数
- **手术刀掩码**：`alpha_mask` `[H, W]` (`float32`, [0.0, 1.0])
  - 膨胀核 $r_{\text{base}} = 6\text{ px}$，羽化边缘 $w_{\text{feather}} = 4.0\text{ px}$
- **色彩对齐补丁**：`matched_patch`（经 16px 边界环 CIE LAB 均值/方差仿射修正）
- **最终合成帧**：`blended_rgb` `[H, W, 3]` (`float32`, [0.0, 1.0])
  - 4 层拉普拉斯金字塔跨频段保锐融合，边界接缝色差 $\Delta E_{00} < 1.5$

---

### 4. E3 阶段：97 帧长时序因果滑窗与切镜重锚定 (`synapse.adapters.scheduler`)

> **注意**：E3 当前仅在 `ubuntu24` 上通过 `scripts/run_cpu_verification.py` 执行无卡验证。`colab_runner.py --exp all` 不包含 E3，Colab GPU 版将在后续迭代中支持。

#### 输入 (Inputs)
- 97 帧长时序序列：`course_102_pbr_97f`（$T=97$ 或扩展至 $T=161$）
- 调度规划器配置：`CausalWindowPlan(window_frames=97, stride_frames=64, overlap_frames=33)`

#### 输出 (Outputs)
- **因果窗口切片列表**：`list[WindowSchedule]`
  - `Window 0`: `frames [0..97)`
  - `Window 1`: `frames [64..161)`（携带前序 3 帧 `[61..63]` 作为 3D VAE 潜空间因果热启动前缀）
- **潜空间重叠融合张量**：`fused_latents` `[9, 16, H/8, W/8]`
  - 在重叠潜步 $\tau_{\text{overlap}}=9$ 上使用保方差 Sigmoid 加权融合，潜变量能量衰减 $< 0.5\%$
- **切镜重锚定事件列表**：`list[ReanchorRequest]`
  - 当不连续指标 $\kappa(t) > 0.65$ 时自动触发，重置时序累积并在切镜点强行插入物理关键帧

---

### 5. E4 阶段：WebDataset 特征分片与 Multi-LoRA 训练 (`synapse.training` & `synapse.adapters.lora`)

#### 输入 (Inputs)
- 预编码潜变量生成器：`LinearProjectionEncoder` (16 通道) + `LinearProjectionTextEncoder`
- 多实体分割序列：包含物体 `Cryptomatte.ID` 的 FGPP 序列
- 训练超参数：`lora_rank=16/32`, `lr=1e-4`, `batch_size=1`, `max_steps=500`

#### 输出 (Outputs)
- **WebDataset 潜变量分片**：`shard_000000.tar`
  - 每个样本包含：`__key__.json` (元数据与 5 段式 Caption), `__key__.latent.npy` (`[16, 25, 60, 104]`), `__key__.control.npy` (`[7, 97, 480, 832]`)
- **LoRA 权重文件**：`adapter_model.safetensors`
  - 仅包含注意力投影权重增量，如 `lora_A` 与 `lora_B`
- **安全清单**：`manifest.json` 符合 `synapse.lora.v1.0` Schema：
  - 强制包含权重文件的 SHA256 校验和、基础模型兼容标识与模块白名单
- **空域多实体门控评估**：
  - 实体间注意力掩码非正交泄露率断言：$\text{leakage} < 3.0\%$

---

### 6. E5 阶段：30 个场景评测基准 (`Synapse-Bench`)

> **注意**：E5 为纯 CPU 软光栅评测，当前仅在 `ubuntu24` 上通过 `scripts/run_cpu_verification.py` 执行。`colab_runner.py` 不调度 E5。

#### 输入 (Inputs)
- 场景定义：`data/bench/scenarios/T1_sphere_direct.json` .. `T5_bistro_interior.json` (共 30 个场景)
- 角色白模代理：顶点数 $\ge 1500$ 且具备铰接四肢的 Humanoid-53 Mesh

#### 输出 (Outputs)
- **无卡软光栅真值集**：深度、法线与物体 ID
- **评测报告 JSON**：`synapse_bench_results.json`
  - 覆盖 Tier 1 到 Tier 5，指标包含 PSNR、SSIM、CIEDE2000、时序重投影漂移误差 $\text{E}_{\text{warp}}$
- **HTML 综合诊断报告**：可离线浏览的可视化卡片

---

## 三、Google Drive 断点续跑状态持久化契约

在 Colab 运行时中，实验进度定期持久化至 Google Drive（`colab_experiment_state.json`）：

```json
{
  "gpu_profile": {
    "cuda_available": true,
    "device_name": "NVIDIA A100-SXM4-40GB",
    "vram_gb": 40.0,
    "recommended_wan_model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "enable_model_cpu_offload": false,
    "dtype": "bfloat16",
    "lora_rank": 32
  },
  "completed": {
    "e1": {
      "status": "PASS",
      "track_b_tensor_shape": [7, 17, 480, 832],
      "depth_channel_range": [0.0, 1.0],
      "elapsed_sec": 4.12
    },
    "e2": {
      "status": "PASS",
      "style_profile_version": "2.0.0",
      "mask_foreground_ratio": 0.28,
      "mean_abs_diff_after": 0.012,
      "elapsed_sec": 3.85
    },
    "e4": {
      "status": "PASS",
      "latent_shard_keys": 1,
      "lora_attention_leakage_ratio": 0.015,
      "elapsed_sec": 5.43
    }
  }
}
```

---

## 四、CLI 确定性退出状态码字典

当通过 CLI 或脚本执行时，进程严格遵守以下退出状态码：

| 状态码 | 异常类名 | 触发条件 | `agent_fix_hint` 举例 |
| :---: | :--- | :--- | :--- |
| **0** | `Success` | 实验或推理正常结束，断言全绿 | 无 |
| **10** | `ContractValidationError` | FGPP 通道名不匹配、矩阵行列式 $\neq +1$、非 16 整数倍分辨率 | 检查 OpenEXR 通道是否遗漏或相机外参行列式是否为负 |
| **20** | `AssetNotFoundError` | `synapse_colab_bundle_v1.tar.gz` 缺失或 SHA256 校验和失效 | 重新在 `ubuntu24` 运行 `prepare_datasets.py` 导出数据包 |
| **30** | `BackendUnavailableError` | CUDA OOM 显存耗尽或 Diffusers 后端加载超时 | 启用 `--enable_model_cpu_offload`，或将模型从 14B 降级至 1.3B |
| **40** | `QualityGateCircuitBreakerError` | 白模顶点数 $<1500$、合成色差 $>3.0$ 或精修超时 15 分钟 | 触发工程熔断降级，回退至上一版稳定几何资产 |
