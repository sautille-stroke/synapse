# Synapse Wire Contracts & Interop Specifications (SSOT)

> 本文件为 Synapse 与上游渲染引擎（`fulgur`, `caelum-core`）、下游调度农场（`render-farm`, `narrative-rail`）及基础库（`@aether`）交互的**跨语言唯一事实源（Canonical Wire SSOT）**。
>
> **对齐版本**：FGPP v1.0 — `@aether//fgpp`（`aether/fgpp/include/aether/fgpp/fgpp_types.h`）

---

## 1. FGPP v1.0 15-通道 OpenEXR 线格式规约

FGPP（Forward-Guided Production Package）是 3D CG 渲染引擎导出供神经视频生成适配器消费的标准中间载体。

### 1.1 标量通道表与内存排布（34 字节/像素）

| ASCII 排序通道名 | 物理语义 | 数据类型 | 像素范围 / 单位 | 内存字节 | 默认填充 |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `BaseColor.B` | 漫反射/表面基色蓝分量 | `HALF` (float16) | $[0.0, 1.0]$ | 2 B | 0.0 |
| `BaseColor.G` | 漫反射/表面基色绿分量 | `HALF` (float16) | $[0.0, 1.0]$ | 2 B | 0.0 |
| `BaseColor.R` | 漫反射/表面基色红分量 | `HALF` (float16) | $[0.0, 1.0]$ | 2 B | 0.0 |
| `CameraLinearDepth.Z` | 相机空间正向米制线性深度 | `FLOAT` (float32) | $(0.0, z_{\text{far}}]$（米） | 4 B | $+\infty$ |
| `Cryptomatte.ID` | 实例分割多实体唯一哈希 ID | `UINT32` | $[0, 2^{32}-1]$ | 4 B | 0 |
| `DirectRadiance.B` | 未色调映射直接光蓝分量 | `HALF` (float16) | $[0.0, 65504.0]$ | 2 B | 0.0 |
| `DirectRadiance.G` | 未色调映射直接光绿分量 | `HALF` (float16) | $[0.0, 65504.0]$ | 2 B | 0.0 |
| `DirectRadiance.R` | 未色调映射直接光红分量 | `HALF` (float16) | $[0.0, 65504.0]$ | 2 B | 0.0 |
| `MotionVector.X` | 屏幕空间光流/运动增量 X | `HALF` (float16) | $[-W, +W]$（像素） | 2 B | 0.0 |
| `MotionVector.Y` | 屏幕空间光流/运动增量 Y | `HALF` (float16) | $[-H, +H]$（像素） | 2 B | 0.0 |
| `RoughnessMetallic.G` | 金属度 (Metallic) | `HALF` (float16) | $[0.0, 1.0]$ | 2 B | 0.0 |
| `RoughnessMetallic.R` | 表面微平面粗糙度 (Roughness) | `HALF` (float16) | $[0.04, 1.0]$ | 2 B | 0.5 |
| `WorldNormal.X` | 世界空间单位法线 X 分量 | `HALF` (float16) | $[-1.0, 1.0]$ | 2 B | 0.0 |
| `WorldNormal.Y` | 世界空间单位法线 Y 分量 | `HALF` (float16) | $[-1.0, 1.0]$ | 2 B | 1.0 |
| `WorldNormal.Z` | 世界空间单位法线 Z 分量 | `HALF` (float16) | $[-1.0, 1.0]$ | 2 B | 0.0 |

> **合计**：13×HALF (26 B) + 1×FLOAT (4 B) + 1×UINT32 (4 B) = **15 通道 / 34 B/px**

> **关键铁律**：
> 1. `Cryptomatte.ID` 必须以无符号 32 位整型存储，**严禁使用任何有损插值或双线性滤波**。
> 2. `WorldNormal.XYZ` 必须满足归一化条件：$\|\mathbf{N}\|_2 = 1.0 \pm 10^{-3}$。
> 3. `CameraLinearDepth.Z` 必须为物理米制正值，背景天球填充为 $+\infty$ 或 `z_far`。

### 1.2 相机矩阵约定与元数据契约 (`metadata.json`)

相机矩阵在 `metadata.json` 中统一使用**列主序 16 浮点数组（Column-Major Flat Array）**，包含：
- `projection_matrix`: $4 \times 4$ 投影矩阵（OpenGL 约定 NDC $Z \in [-1, 1]$ 或 Vulkan 约定 $Z \in [0, 1]$）。
- `view_matrix`: $4 \times 4$ 视图外参矩阵。
- 偏心主点恢复关系：
  $$\begin{aligned}
  c_x &= \frac{1.0 - p_{02}}{2} \cdot W \\
  c_y &= \frac{1.0 + p_{12}}{2} \cdot H
  \end{aligned}$$
- 旋转子矩阵 $\mathbf{R} \in \mathrm{SO}(3)$ 必须满足：$\det(\mathbf{R}) = 1.0 \pm 10^{-4}$，且 $\mathbf{R}^T \mathbf{R} = \mathbf{I}$。

---

## 2. RFC 0039 自洽任务包（JobBundle）契约

### 2.1 任务包目录拓扑与文件清单

```text
<job_dir>/
  ├── bundle_manifest.json    # RFC 0039 阶段编排与硬件预算声明 (synapse.job_bundle.v1.0)
  ├── fgpp/                   # 输入多通道 FGPP 序列
  │   ├── frame_000000.exr
  │   ├── frame_000001.exr
  │   └── metadata.json
  ├── checkpoints/            # LoRA 适配器权重与 Manifest
  │   ├── lora_manifest.json
  │   └── weights.safetensors
  └── output/                 # 单次运行产物与评测日志
      ├── generated.mp4
      └── report.json
```

### 2.2 统一确定性退出状态码

| 状态码 | 常量名 | 触发语义 | Agent 修复指引 |
| :---: | :--- | :--- | :--- |
| `0` | `SUCCESS` | 阶段顺利完成并产出有效非退化张量 | 无需干预 |
| `10` | `INVALID_ARGUMENT` | 命令行参数格式错误或 Schema 契约校验失败 | 检查输入 JSON 格式并依据错误定位修正 |
| `20` | `ASSET_NOT_FOUND` | 输入场景、网格、FGPP EXR 丢失或 SHA256 不匹配 | 检查资产挂载路径并执行前置资产同步 |
| `30` | `BACKEND_UNAVAILABLE` | GPU 显存不足（OOM）、驱动不可达或缺少深度学习框架 | 降级分辨率/批量大小或切换至 CPU 验证模式 |
| `40` | `QUALITY_GATE_FAILED` | 质量门禁不达标（如 PSNR/SSIM 不足或人体结构崩溃） | 调优引导权重或触发人工 HITL 精修工作流 |
| `50` | `NOT_IMPLEMENTED` | 请求的特性、算子或阶段尚未实现（严禁假冒 0 退出） | 审查待实现桩函数或确认环境依赖 |

---

## 3. ACEScg 线性色彩主链与有理软肩（Rational Soft-Shoulder）

### 3.1 色彩空间拓扑

全系统图像与色彩运算严格在 **ACEScg (AP1 Linear)** 色彩空间内进行，严禁混入 sRGB Gamma 压缩空间进行线性代数或卷积运算：
$$\text{sRGB Input} \xrightarrow{\text{sRGB-to-ACEScg}} \text{ACEScg Linear (Work Space)} \xrightarrow{\text{ACEScg-to-sRGB}} \text{sRGB Output}$$

### 3.2 有理软肩压缩算子 $T(y)$

取代旧有饱和度衰减严重的 `tanh` 算子，采用物理保真有理软肩压缩：
$$T(y) = \begin{cases}
y, & y \le y_{\text{knee}} \\
y_{\text{knee}} + \frac{y - y_{\text{knee}}}{1.0 + \frac{y - y_{\text{knee}}}{1.0 - y_{\text{knee}}}}, & y > y_{\text{knee}}
\end{cases}$$
其中 $y_{\text{knee}} = 0.85$。该算子确保 $T(y) \in [0, 1)$，在一阶导数处 $C^1$ 连续，并有效保留高光梯度。

---

## 4. 共享内存（SHM）与 Seqlock 无锁协议

桌面端检查器（`Data Inspector` Electron 前端）与 C++20 本地引擎（`InspectorApp`）通过共享内存实现 60 FPS 零拷贝预览。

### 4.1 SHM 结构体对齐（64 字节对齐）

```text
Offset 0x00: uint32_t seqlock_sequence   (写者自增：奇数=写中，偶数=就绪)
Offset 0x04: uint32_t width               (帧宽度)
Offset 0x08: uint32_t height              (帧高度)
Offset 0x0C: uint32_t stride              (行跨度，字节)
Offset 0x10: uint32_t channels            (通道数，通常为 4: RGBA)
Offset 0x14: float    fps                 (当前渲染帧率)
Offset 0x18: float    render_time_ms      (单帧耗时毫秒)
Offset 0x20: uint8_t  pixel_data[...]     (原始像素缓冲区，大小 = width * height * 4)
```

### 4.2 读者无锁校验伪代码

```cpp
uint32_t seq1, seq2;
do {
    seq1 = shm->sequence.load(std::memory_order_acquire);
    while (seq1 & 1) { // 正在写入中，短暂让出 CPU
        std::this_thread::yield();
        seq1 = shm->sequence.load(std::memory_order_acquire);
    }
    std::memcpy(local_buffer, shm->pixel_data, frame_size);
    seq2 = shm->sequence.load(std::memory_order_acquire);
} while (seq1 != seq2);
```
