# WSI推理工具使用指南

本工具提供了完整的端到端（E2E）WSI（全切片图像）处理流程，从原始slide文件到最终的推理结果。

## 目录

- [功能概述](#功能概述)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [详细流程说明](#详细流程说明)
- [提取UNI2特征](#提取uni2特征)
- [Attention Map 可解释性](#attention-map-可解释性)
- [参数说明](#参数说明)
- [输出文件结构](#输出文件结构)
- [常见问题](#常见问题)

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

### 保存Attention Map进行可视化

在推理时同时保存patch级别的attention权重，用于后续可视化：

```bash
python run_slide_encoder_inference.py \
    --slide_path /path/to/slide.svs \
    --job_dir /path/to/output \
    --mag 20 \
    --patch_size 256 \
    --patch_encoder uni_v2 \
    --patch_encoder_weights_path /path/to/uni2_weights.pt \
    --slide_encoder gfy_abmil \
    --slide_encoder_weights_path /path/to/slide_encoder_weights.pt \
    --save_attention_map \
    --attention_map_type both
```

输出结果会包含：
- `attention_maps/{slide_name}_attention.npz`: Attention权重和坐标数据
- `attention_maps/{slide_name}_attention_meta.json`: 统计信息
- `attention_maps/{slide_name}_heatmap.png`: 热力图可视化（当type包含heatmap）
- `attention_maps/{slide_name}_spatial.png`: 空间分布图（当type包含spatial）

### 从保存的Attention Map生成Heatmap

如果已有保存的attention map文件，可以直接生成heatmap可视化，无需重新计算：

```bash
python run_heatmap_visualization.py \
    --slide_path /path/to/slide.svs \
    --job_dir /path/to/output \
    --mag 20 \
    --patch_size 256 \
    --attention_from_file \
    --attention_map_path /path/to/output/attention_maps/slide_name_attention.npz \
    --heatmap_save_dir /path/to/heatmap_output
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

## Attention Map 可解释性

### 功能介绍

Attention map提供了模型决策过程的可视化，帮助理解：

- **哪些patches对预测贡献最大**：高attention权重的区域是模型关注的重点
- **模型的诊断依据追踪**：与病理学家的诊断逻辑是否一致
- **模型合理性验证**：注意力分布是否集中于诊断相关的病理特征
- **弱监督学习指导**：用于识别关键patches或标注指导

### 使用流程

#### 1. 推理时保存Attention Map

在`run_slide_encoder_inference.py`中添加`--save_attention_map`参数：

```bash
python run_slide_encoder_inference.py \
    --slide_path slide.svs \
    --job_dir output/ \
    --mag 20 \
    --patch_size 256 \
    --patch_encoder uni_v2 \
    --patch_encoder_weights_path encoder.pt \
    --slide_encoder gfy_abmil \
    --slide_encoder_weights_path model.pt \
    --save_attention_map \
    --attention_map_type both
```

**参数说明**：

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--save_attention_map` | - | 启用attention保存（布尔标志） |
| `--attention_map_type` | `heatmap`, `spatial`, `both` | 输出类型（默认heatmap） |

**输出文件**：

- `attention_maps/{slide}_attention.npz`: 原始权重数据（coords和weights）
- `attention_maps/{slide}_attention_meta.json`: 统计信息（min, max, mean, std等）
- `attention_maps/{slide}_heatmap.png`: 热力图可视化
- `attention_maps/{slide}_spatial.png`: 空间分布图

#### 2. 生成Heatmap可视化

**方式A**: 从保存的attention map生成（推荐）

```bash
python run_heatmap_visualization.py \
    --slide_path slide.svs \
    --job_dir output/ \
    --mag 20 \
    --patch_size 256 \
    --attention_from_file \
    --attention_map_path output/attention_maps/slide_name_attention.npz \
    --heatmap_save_dir output/heatmaps
```

优势：快速、高效，无需重新计算attention

**方式B**: 直接从slide encoder计算（原始方式）

```bash
python run_heatmap_visualization.py \
    --slide_path slide.svs \
    --job_dir output/ \
    --mag 20 \
    --patch_size 256 \
    --patch_encoder uni_v2 \
    --patch_encoder_weights_path encoder.pt \
    --slide_encoder gfy_abmil \
    --slide_encoder_weights_path model.pt \
    --heatmap_save_dir output/heatmaps
```

#### 3. Heatmap可视化参数

```bash
python run_heatmap_visualization.py \
    --slide_path slide.svs \
    --job_dir output/ \
    --attention_from_file \
    --attention_map_path output/attention_maps/slide_attention.npz \
    --vis_level 4 \                    # 可视化倍数
    --cmap coolwarm \                  # 颜色map (jet, viridis等)
    --alpha 0.5 \                      # 透明度 (0-1)
    --blank_canvas \                   # 使用空白背景而非原始图像
    --blur \                           # 应用高斯模糊
    --convert_to_percentiles \         # 转换为百分位数
    --binarize \                       # 二值化
    --binary_thresh 0.5                # 二值化阈值
```

### Python API 使用

加载和分析attention map数据：

```python
import numpy as np
import json

# 加载attention map
data = np.load('attention_maps/slide_attention.npz')
attention_weights = data['attention_weights']  # (N,)
coords = data['coords']  # (N, 2)

# 加载元数据
with open('attention_maps/slide_attention_meta.json', 'r') as f:
    meta = json.load(f)

# 获取top patches
top_k = 20
top_indices = np.argsort(attention_weights)[-top_k:][::-1]
top_coords = coords[top_indices]
top_weights = attention_weights[top_indices]

print(f"Total patches: {len(attention_weights)}")
print(f"Attention range: {attention_weights.min():.4f} - {attention_weights.max():.4f}")
print(f"Mean attention: {attention_weights.mean():.4f}")
print(f"\nTop {top_k} patches:")
for i, (coord, weight) in enumerate(zip(top_coords, top_weights), 1):
    print(f"  {i}. Coord: {coord}, Weight: {weight:.6f}")
```

### 支持的模型

#### Patch Encoder

所有支持的patch encoder都可以用于attention map提取：

- uni_v2, uni_v1
- conch_v1, conch_v15
- virchow, virchow2
- ctranspath, phikon, phikon_v2
- gigapath, resnet50等

#### Slide Encoder

**完全支持** attention权重返回：

- `gfy_abmil` (GFY-ABMIL) ✅ 推荐
- `abmil` (Attention-based MIL) ✅

**其他模型** 可能需要验证支持情况。

### 实际应用示例

#### 示例1: 肿瘤分类模型可解释性验证

```bash
# 推理并保存attention
python run_slide_encoder_inference.py \
    --slide_path tumor_slide.svs \
    --job_dir results/ \
    --slide_encoder gfy_abmil \
    --slide_encoder_weights_path tumor_model.pt \
    --save_attention_map \
    --attention_map_type both

# 生成高质量heatmap用于报告
python run_heatmap_visualization.py \
    --slide_path tumor_slide.svs \
    --job_dir results/ \
    --attention_from_file \
    --attention_map_path results/attention_maps/tumor_slide_attention.npz \
    --vis_level 4 \
    --cmap coolwarm \
    --alpha 0.6
```

**分析**：
1. 查看生成的heatmap，红色区域为模型关注的肿瘤区域
2. 与病理学家标注对比，验证模型诊断逻辑
3. 识别模型可能的偏差或失败案例

#### 示例2: 批量处理多个样本

```bash
import os
import subprocess

slides = ['slide1.svs', 'slide2.svs', 'slide3.svs']

for slide in slides:
    slide_name = os.path.splitext(slide)[0]
    
    # 推理并保存attention
    subprocess.run([
        'python', 'run_slide_encoder_inference.py',
        '--slide_path', f'slides/{slide}',
        '--job_dir', 'batch_output/',
        '--slide_encoder', 'gfy_abmil',
        '--slide_encoder_weights_path', 'model.pt',
        '--save_attention_map'
    ])
    
    # 生成heatmap
    subprocess.run([
        'python', 'run_heatmap_visualization.py',
        '--slide_path', f'slides/{slide}',
        '--job_dir', 'batch_output/',
        '--attention_from_file',
        '--attention_map_path', f'batch_output/attention_maps/{slide_name}_attention.npz',
        '--heatmap_save_dir', 'heatmaps/'
    ])
```

#### 示例3: 比较不同模型的attention分布

```python
import numpy as np
import json

# 比较两个模型的attention差异
model1_data = np.load('attention_maps/model1_attention.npz')
model2_data = np.load('attention_maps/model2_attention.npz')

attn1 = model1_data['attention_weights']
attn2 = model2_data['attention_weights']

# 计算相关性
correlation = np.corrcoef(attn1, attn2)[0, 1]
print(f"Attention correlation between models: {correlation:.4f}")

# 找到最不同的patches
diff = np.abs(attn1 - attn2)
different_indices = np.argsort(diff)[-20:]
print(f"Top 20 most different patches: {different_indices}")
```

### 文件输出结构

```text
output/
├── attention_maps/                      # Attention map结果
│   ├── slide1_attention.npz             # 原始权重数据
│   ├── slide1_attention_meta.json       # 统计信息
│   ├── slide1_heatmap.png               # 热力图
│   └── slide1_spatial.png               # 空间分布图
│
└── heatmaps/                            # Heatmap可视化结果
    ├── slide1_heatmap.png               # TRIDENT生成的高分辨率heatmap
    └── ...
```

### 常见问题

**Q: Attention权重的范围是多少？**

A: 通常在0-1之间（经过softmax归一化）。在可视化时会进一步扩展到0-255用于图像显示。

**Q: 为什么某些slides的attention分布很不均匀？**

A: 这通常表明模型对特定区域有强烈的聚焦。可能原因：
 - 该区域包含诊断相关的关键特征
 - 模型对该特征的识别能力强
 - 需要验证是否合理

**Q: 能否用attention权重进行区域级别的决策？**

A: 可以。高attention权重的patches对应诊断相关区域，可用于：
 - 关键区域自动定位
 - 病理特征识别
 - 弱监督学习

**Q: 如何选择合适的可视化参数？**

A: 
 - `vis_level`: 根据slide大小，建议4-5
 - `alpha`: 0.4-0.6用于overlay背景，1.0用于纯热力图
 - `cmap`: coolwarm/jet用于突出对比，viridis用于单调性
 - `blur`: 启用以平滑细节



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

### Attention Map参数（可选）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_attention_map` | False | 是否在推理时保存patch级别的attention map |
| `--attention_map_type` | heatmap | 输出类型: heatmap/spatial/both |

### Heatmap可视化参数（run_heatmap_visualization.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--attention_from_file` | False | 从保存的NPZ文件加载attention map（推荐） |
| `--attention_map_path` | None | 保存的attention map NPZ文件路径 |
| `--vis_level` | 4 | Heatmap可视化层级 |
| `--cmap` | coolwarm | Colormap类型(jet, viridis, cool warm等) |
| `--alpha` | 0.5 | Heatmap透明度 (0-1) |
| `--blank_canvas` | False | 使用纯白背景而非原始WSI |
| `--blur` | True | 应用高斯模糊 |
| `--convert_to_percentiles` | True | 转换为百分位数 |
| `--binarize` | False | 二值化输出 |
| `--binary_thresh` | 0.5 | 二值化阈值 (0-1) |

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
├── attention_maps/            # Attention map结果（当--save_attention_map时）
│   ├── {slide_name}_attention.npz        # 原始权重和坐标数据
│   ├── {slide_name}_attention_meta.json  # 统计信息
│   ├── {slide_name}_heatmap.png          # 热力图可视化
│   └── {slide_name}_spatial.png          # 空间分布图（可选）
│
└── heatmaps/                  # Heatmap可视化结果（run_heatmap_visualization.py输出）
    └── {slide_name}_heatmap.png
```

### H5文件格式

**Patch坐标文件** (`*_patches.h5`):

- `coords`: (N, 2) numpy数组，包含每个patch的(x, y)坐标
- `coords_attrs`: 属性字典（mag, patch_size等）

**Patch特征文件** (`features_*/{slide_name}.h5`):

- `coords`: (N, 2) 坐标数组
- `features`: (N, feature_dim) 特征矩阵
- `coords_attrs`: 坐标属性

### Attention Map文件格式

**NPZ文件** (`{slide_name}_attention.npz`):

- `attention_weights`: (N,) 权重数组，值在0-1之间（softmax后）
- `coords`: (N, 2) patch坐标数组

**JSON元数据** (`{slide_name}_attention_meta.json`):

```json
{
  "slide_name": "slide_name",
  "n_patches": 12345,
  "attention_stats": {
    "min": 0.0001,
    "max": 0.0432,
    "mean": 0.000812,
    "std": 0.001234
  },
  "patch_size": 256
}
```


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

### Q5: 如何生成和使用Attention Map？

A: 分为两步：

**第一步: 推理时保存attention map**

```bash
python run_slide_encoder_inference.py \
    --slide_path slide.svs \
    --job_dir output/ \
    --slide_encoder gfy_abmil \
    --slide_encoder_weights_path model.pt \
    --save_attention_map
```

**第二步: 生成heatmap可视化**

```bash
python run_heatmap_visualization.py \
    --slide_path slide.svs \
    --job_dir output/ \
    --attention_from_file \
    --attention_map_path output/attention_maps/slide_attention.npz
```

### Q6: Attention Map有什么用途？

A: 主要用途包括：

1. **模型可解释性**: 查看模型关注的区域
2. **诊断验证**: 对比公模型注意力分布与医学逻辑的一致性
3. **关键区域定位**: 自动识别诊断相关的patches
4. **弱监督学习**: 用于标注指导或区域级别决策
5. **模型改进**: 分析模型失败案例，指导后续优化

### Q7: 为什么生成heatmap时选择从文件加载而不是重新计算？

A: 从文件加载的优势：

1. **快速**: 无需重新运行slide encoder，可节省70-80%的时间
2. **高效**: 只需加载NPZ文件和生成可视化
3. **一致性**: 确保可视化与之前的推理结果完全相同
4. **灵活**: 可以用不同的可视化参数多次生成heatmap

推荐流程：
```
推理(一次) → 保存attention → 可视化(多次,不同参数)
```

### Q8: 如何比较不同模型的Attention分布？

A: 使用Python API对比：

```python
import numpy as np

# 加载两个模型的attention map
data1 = np.load('model1_attention.npz')
data2 = np.load('model2_attention.npz')

attn1 = data1['attention_weights']
attn2 = data2['attention_weights']

# 计算相关性
corr = np.corrcoef(attn1, attn2)[0, 1]
print(f"模型间Attention相关性: {corr:.4f}")

# 找出差异最大的patches
diff = np.abs(attn1 - attn2)
different_indices = np.argsort(diff)[-20:]  # Top 20
```

## 技术架构

本工具基于TRIDENT框架构建，主要模块：

- `trident.wsi_objects`: WSI加载和处理
- `trident.segmentation_models`: 组织分割模型
- `trident.patch_encoder_models`: Patch特征编码器
- `trident.slide_encoder_models`: Slide级别编码器

更多技术细节请参考代码注释。
