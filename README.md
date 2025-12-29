# WSI推理工具使用指南

本工具提供了完整的端到端（E2E）WSI（全切片图像）处理流程，从原始slide文件到最终的推理结果。

## 目录

- [功能概述](#功能概述)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [详细流程说明](#详细流程说明)
- [提取UNI2特征](#提取uni2特征)
- [参数说明](#参数说明)
- [输出文件结构](#输出文件结构)

## 功能概述

本工具实现了完整的WSI处理流水线：

1. **组织分割（Segmentation）**：识别WSI中的组织区域
2. **坐标提取（Coordinate Extraction）**：提取组织区域的patch坐标
3. **特征提取（Feature Extraction）**：使用patch encoder提取patch特征
4. **Slide编码（Slide Encoding）**：使用slide encoder对patch特征进行聚合，得到slide级别的表示
5. **推理输出（Inference）**：基于slide特征进行下游任务推理

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：

- PyTorch
- torchvision
- timm
- h5py
- opencv-python
- 其他相关库

### 2. 模型权重准备

根据您使用的模型，需要准备相应的权重文件：

- **Patch Encoder权重**：如UNI2权重文件（`.pt`或`.pth`格式）
- **Slide Encoder权重**：如ABMIL、THREADS等模型的权重文件
- **Segmentation模型权重**（可选）：如需自定义分割模型

权重文件可以通过以下方式获取：

- 从Hugging Face Hub自动下载（需要网络连接和访问权限）
- 手动下载并指定路径

## 快速开始

### 基本使用示例

```bash
python run_slide_encoder_inference.py \
    --slide_path /path/to/slide.svs \
    --job_dir /path/to/output \
    --mag 20 \
    --patch_size 256 \
    --patch_encoder uni_v2 \
    --patch_encoder_weights_path /path/to/uni2_weights.pt \
    --slide_encoder gfy_abmil \
    --slide_encoder_weights_path /path/to/slide_encoder_weights.pt
```

### 仅提取UNI2特征（不进行slide编码）

使用 `--extract_features_only` 参数可以跳过slide encoder推理，仅提取并保存patch特征：

```bash
python run_slide_encoder_inference.py \
    --slide_path /path/to/slide.svs \
    --job_dir /path/to/output \
    --mag 20 \
    --patch_size 256 \
    --patch_encoder uni_v2 \
    --patch_encoder_weights_path /path/to/uni2_weights.pt \
    --extract_features_only
```

此模式下不需要指定 `--slide_encoder` 和 `--slide_encoder_weights_path` 参数。

## 详细流程说明

### 步骤1: 加载WSI

工具支持多种WSI格式：

- `.svs` (Aperio)
- `.tiff` / `.tif`
- 其他OpenSlide支持的格式

```python
from trident import load_wsi

slide = load_wsi(
    slide_path=args.slide_path,
    lazy_init=False,
    custom_mpp_keys=args.custom_mpp_keys
)
```

### 步骤2: 组织分割

识别并提取WSI中的组织区域，可选的分割器：

- `hest`：默认分割器
- `grandqc`：高质量分割器

分割结果保存为GeoJSON格式，如果已存在分割结果，会自动跳过此步骤。

```bash
# 使用自定义分割权重
--seg_weights_path /path/to/seg_weights.ckpt \
--segmenter hest \
--seg_conf_thresh 0.5
```

### 步骤3: 提取组织坐标

从分割的组织区域中提取patch坐标：

- `--mag`: 提取的放大倍数（5, 10, 20, 40）
- `--patch_size`: Patch大小（默认256像素）
- `--overlap`: Patch之间的重叠像素数（默认0）

坐标信息保存为H5格式，包含每个patch的坐标位置。

### 步骤4: 提取Patch特征

使用指定的patch encoder提取每个patch的特征表示。

**支持的Patch Encoder包括**：

- `uni_v2`: UNI2（推荐用于医学图像）
- `uni_v1`: UNI v1
- `conch_v1`, `conch_v15`: CONCH系列
- `ctranspath`: CTransPath
- `phikon`, `phikon_v2`: Phikon系列
- `virchow`, `virchow2`: Virchow系列
- `gigapath`: GigaPath
- 以及其他多种模型

特征提取结果保存为H5格式，包含：

- `coords`: Patch坐标数组
- `features`: Patch特征矩阵 (N × feature_dim)
- `coords_attrs`: 坐标相关属性（如patch_size, mag等）

### 步骤5: Slide编码和推理（可选）

如果未使用 `--extract_features_only` 参数，将使用slide encoder对patch特征进行聚合，得到slide级别的表示，并进行下游任务推理。

**支持的Slide Encoder包括**：

- `gfy_abmil`: ABMIL-based slide encoder
- `threads`: THREADS模型
- `titan`: Titan模型
- `prism`: PRISM模型
- `chief`: CHIEF模型
- `gigapath`: GigaPath slide encoder
- `mean-*`: 简单的平均池化

推理结果以概率分布的形式输出。

**注意**：如果使用 `--extract_features_only`，将跳过此步骤，仅保存patch特征文件。

## 提取UNI2特征

### 方法1: 仅提取Patch特征

使用 `--extract_features_only` 参数，仅提取并保存patch特征，不执行slide encoder推理：

```bash
python run_slide_encoder_inference.py \
    --slide_path /path/to/slide.svs \
    --job_dir /path/to/output \
    --mag 20 \
    --patch_size 256 \
    --patch_encoder uni_v2 \
    --patch_encoder_weights_path /path/to/uni2_weights.pt \
    --extract_features_only \
    --batch_size 32
```

输出结果：

- Patch特征文件：`{job_dir}/{mag}x_{patch_size}px_{overlap}px_overlap/features_uni_v2/{slide_name}.h5`
- H5文件包含：
  - `coords`: (N, 2) 坐标数组
  - `features`: (N, 1536) UNI2特征矩阵（1536维）
  - `coords_attrs`: 元数据

### 方法2: 使用Python API加载特征

```python
import h5py
import numpy as np

# 加载特征文件
with h5py.File('path/to/slide.h5', 'r') as f:
    coords = f['coords'][:]  # (N, 2)
    features = f['features'][:]  # (N, 1536)
    attrs = dict(f['coords'].attrs)
    
print(f"提取了 {len(coords)} 个patches")
print(f"特征维度: {features.shape[1]}")
```

### UNI2模型说明

UNI2 (UNI v2) 是一个大型视觉Transformer模型，专门用于医学图像分析：

- **架构**: ViT-Giant (patch_size=14, embed_dim=1536, depth=24)
- **输出维度**: 1536
- **精度**: float16
- **预训练**: 在大规模组织病理学数据上预训练

权重文件可以从以下来源获取：

- Hugging Face: `MahmoodLab/UNI2-h`
- 本地权重文件路径

## 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `--slide_path` | WSI文件路径 |
| `--job_dir` | 输出目录 |
| `--slide_encoder` | Slide encoder类型（未使用`--extract_features_only`时必需） |
| `--slide_encoder_weights_path` | Slide encoder权重路径（未使用`--extract_features_only`时必需） |

### WSI处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mag` | 20 | 提取放大倍数 (5/10/20/40) |
| `--patch_size` | 256 | Patch大小（像素） |
| `--overlap` | 0 | Patch重叠像素数 |
| `--custom_mpp_keys` | None | 自定义MPP键列表 |

### 分割参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--segmenter` | hest | 分割器类型 (hest/grandqc) |
| `--seg_weights_path` | None | 分割模型权重路径 |
| `--seg_conf_thresh` | 0.5 | 分割置信度阈值 |
| `--remove_holes` | False | 是否移除孔洞 |
| `--remove_artifacts` | False | 是否移除伪影 |
| `--remove_penmarks` | False | 是否移除笔迹标记 |

### Patch Encoder参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--patch_encoder` | uni_v2 | Patch encoder类型 |
| `--patch_encoder_weights_path` | None | Patch encoder权重路径 |
| `--batch_size` | 32 | 特征提取的batch size |

### Slide Encoder参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--slide_encoder` | gfy_abmil | Slide encoder类型（未使用`--extract_features_only`时必需） |
| `--slide_encoder_weights_path` | None | Slide encoder权重路径（未使用`--extract_features_only`时必需） |

### 特征提取模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--extract_features_only` | False | 仅提取patch特征，跳过slide encoder推理 |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | 0 | GPU设备索引 |
| `--save_slide_features` | False | 是否保存slide特征到文件 |

## 输出文件结构

运行完成后，输出目录结构如下：

```text
job_dir/
├── contours_geojson/          # 组织分割结果（GeoJSON格式）
│   └── {slide_name}.geojson
│
├── {mag}x_{patch_size}px_{overlap}px_overlap/
│   ├── patches/                # Patch坐标文件
│   │   └── {slide_name}_patches.h5
│   │
│   └── features_{patch_encoder}/  # Patch特征文件
│       └── {slide_name}.h5
│
└── (如果--save_slide_features，会有slide特征文件)
```

### H5文件格式

**Patch坐标文件** (`*_patches.h5`):

- `coords`: (N, 2) numpy数组，包含每个patch的(x, y)坐标
- `coords_attrs`: 属性字典（mag, patch_size等）

**Patch特征文件** (`features_*/{slide_name}.h5`):

- `coords`: (N, 2) 坐标数组
- `features`: (N, feature_dim) 特征矩阵
- `coords_attrs`: 坐标属性


## 常见问题

### Q1: 如何获取UNI2权重文件？

A: 可以从以下途径获取：

1. 从Hugging Face Hub下载：`MahmoodLab/UNI2-h`
2. 如果设置了`UNI_CKPT_PATH`环境变量，工具会自动使用该路径
3. 通过`--patch_encoder_weights_path`显式指定

### Q2: 提取的特征维度是多少？

A:

- UNI2: 1536维
- UNI v1: 1024维
- CONCH v1/v1.5: 768维
- CTransPath: 768维
- 其他模型请参考相应文档

### Q3: 如何处理内存不足的问题？

A:

- 减小`--batch_size`
- 使用较小的`--patch_size`
- 使用较低的`--mag`（如10x而非20x）

### Q4: 支持哪些WSI格式？

A: 支持OpenSlide支持的所有格式，包括：

- `.svs` (Aperio)
- `.ndpi` (Hamamatsu)
- `.vms`, `.vmu`, `.ndpi` (Hamamatsu)
- `.scn` (Leica)
- `.mrxs` (3DHistech)
- `.tiff`, `.tif`
- 以及其他OpenSlide支持的格式

## 技术架构

本工具基于TRIDENT框架构建，主要模块：

- `trident.wsi_objects`: WSI加载和处理
- `trident.segmentation_models`: 组织分割模型
- `trident.patch_encoder_models`: Patch特征编码器
- `trident.slide_encoder_models`: Slide级别编码器

更多技术细节请参考代码注释。
