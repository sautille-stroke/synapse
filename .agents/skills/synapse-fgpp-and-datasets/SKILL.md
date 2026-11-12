---
name: synapse-fgpp-and-datasets
description: >-
  FGPP v1.0 15 通道 OpenEXR 线格式规范与跨引擎黄金数据集构建指南。涵盖 34 B/px 通道表、矩阵列主序与旋转行列式校验、fulgur Track-B 实时导出（含 PickingPass Cryptomatte.ID）、caelum-core Track-A 离线光追真值对导出，以及 prepare_datasets.py 与 Colab 数据包打包流程。在处理 EXR 读写、跨引擎联调或准备实验数据时使用。
---

# FGPP v1.0 线格式与跨引擎黄金数据集指南

> `FGPP`（Feature Guidance Package Protocol，`fgpp.v1.0`）是连接正向 3D 渲染引擎（`fulgur` / `caelum-core`）与神经视频生成模型的 15 标量通道（`34 B/px`）OpenEXR 2.0 线格式契约。

---

## 一、15 通道 OpenEXR 线格式规范（ASCII 严格排序）

每个 `frame_XXXXXX.exr` 必须包含以下 15 个按 ASCII 字典序排列的标量通道（总计 **34 Bytes/Pixel**）：

| 序号 | OpenEXR 通道名 | 像素类型 (`PixelType`) | 字节数 | 物理语义与数值域约束 |
| :---: | :--- | :---: | :---: | :--- |
| 0–2 | `BaseColor.{B,G,R}` | `HALF` (`1`) | 6 B | 线性空间漫反射反照率 $\in [0, 1]^3$（禁止 sRGB 伽马编码） |
| 3 | `CameraLinearDepth.Z` | `FLOAT` (`2`) | 4 B | 相机空间米制线性深度 $z \in (0, z_{\text{far}}]$（无效/天空像素回填 $z_{\text{far}}$） |
| 4 | `Cryptomatte.ID` | `UINT` (`0`) | 4 B | 精确 32 位无符号整数实体哈希/ID（严禁双线性插值滤波；背景为 `0`） |
| 5–7 | `DirectRadiance.{B,G,R}` | `HALF` (`1`) | 6 B | 线性 ACEScg 直接光照辐射亮度 $\in [0, 65504]^3$ |
| 8–9 | `MotionVector.{X,Y}` | `HALF` (`1`) | 4 B | 屏幕空间亚像素帧间位移 $(dx, dy)$（朝向下一帧或上一帧一致约定） |
| 10 | `RoughnessMetallic.G` | `HALF` (`1`) | 2 B | 金属度 Metallic $\in [0, 1]$ |
| 11 | `RoughnessMetallic.R` | `HALF` (`1`) | 2 B | 微表面粗糙度 Roughness $\in [0.04, 1.0]$ |
| 12–14 | `WorldNormal.{X,Y,Z}` | `HALF` (`1`) | 6 B | 世界空间单位法线 $\in [-1, 1]^3$（前景 $\|\mathbf{n}\|_2 = 1$，背景回填 `[0, 0, 1]`） |

### 相机与时序元数据契约 (`metadata.json`)
- **坐标系**：右手系，`+Y` 向上，相机沿 `-Z` 观察，单位为米（`meters`）。
- **矩阵存储**：`view_matrix_col_major` 与 `proj_matrix_col_major` 为 **列主序（Column-Major）** 16 元素展平数组；Python 端通过 `np.array(col_flat).reshape((4, 4)).T` 还原行主序矩阵，并强制校验旋转子块 $|\det(R) - 1| < 10^{-5}$。
- **帧数对齐**：序列总帧数必须满足 3D 因果 VAE 时间下采样对齐公式 **$T = 4k + 1$**（如 `17`, `49`, `81`, `97` 帧）。
- **物理锚点帧**：每 `48` 帧（$t \in \{0, 48, 96, \dots\}$）标记 `is_physical_anchor = true`。

---

## 二、跨引擎导出工作流

### 1. Track-B：`fulgur` 实时光栅序列导出（含 GPU `PickingPass` `Cryptomatte.ID`）
在 `ubuntu24` 上通过 `--export-fgpp` 与 `--fgpp-frames` 直接从任意 `SampleApp` 课程导出：

```bash
ssh ubuntu24 "cd ~/work/dervgen/fulgur && bazel run //course:102-pbr-direct -- \
  --headless --width=832 --height=480 \
  --export-fgpp=/root/work/datasets/synapse/track_b_fulgur/course_102_pbr_17f \
  --fgpp-frames=17"
```
- **实现机制**：当开启 `--export-fgpp` 且 `m_diagnosticScene != nullptr` 时，`SampleApp::executeFrame()` 自动挂载 `PickingPass` 输出 `Editor_PickingEntityId`（`R32_UINT`，编码 `(entityId << 12) | submeshIndex`），由 `FgppExportPass` 回读写入 `Cryptomatte.ID`。

### 2. Track-A：`caelum-core` 离线路径追踪 FGPP + Beauty GT 配对导出
使用 `caelum.camera.fgpp_exporter.export_fgpp_sequence` 同时导出 15 通道 `frame_XXXXXX.exr` 与配对路径追踪真值 `beauty_XXXXXX.exr`：

```python
from caelum.camera import FgppFrameInput, export_fgpp_sequence

result = export_fgpp_sequence(
    frames=[FgppFrameInput(aov_buffer=aov, camera=cam, frame_index=i) for i in range(17)],
    output_dir="/root/work/datasets/synapse/track_a_caelum/cornell_spheres_17f",
    fps=24.0,
    scene_id="caelum_cornell_spheres_17f",
    spp=256,
    max_bounces=8,
    export_beauty_gt=True,
)
```

---

## 三、一键生成标准黄金数据集与 Colab 离线包

在 `ubuntu24` 上运行 [`scripts/prepare_datasets.py`](scripts/prepare_datasets.py)，自动完成 Track-B（`17f` + `97f`）与 Track-A（`17f`）数据集构建、SHA256 清单更新及 Colab `.tar.gz` 归档：

```bash
ssh ubuntu24 "cd ~/work/dervgen/synapse && /root/venv-synapse/bin/python scripts/prepare_datasets.py"
```

产出路径速查：
- **数据集根目录（`ubuntu24`）**：`/root/work/datasets/synapse/`
- **Colab 便携归档包（`ubuntu24`）**：`/root/work/datasets/synapse/synapse_colab_bundle_v1.tar.gz`（约 `12.8 MB`）
- **Git 锁定的 SHA256 清单**：[`data/manifests/golden_dataset_manifest.json`](data/manifests/golden_dataset_manifest.json)
