"""
使用 TRIDENT 进行 WSI heatmap 可视化的脚本

参考 CLAM 的 heatmap 可视化功能和 TRIDENT tutorial notebook 
(3-Training-a-WSI-Classification-Model-with-ABMIL-and-Heatmaps.ipynb)，
在 TRIDENT 框架中实现 attention heatmap 可视化。

本脚本使用 trident.Visualization.visualize_heatmap 函数（与 tutorial notebook 中使用的相同）。

示例用法:
```bash
python run_heatmap_visualization.py \
    --slide_path path/to/slide.svs \
    --job_dir output/ \
    --mag 20 \
    --patch_size 256 \
    --patch_encoder uni_v2 \
    --slide_encoder gfy_abmil \
    --slide_encoder_weights_path path/to/weights.pt \
    --heatmap_save_dir output/heatmaps
```

注意: 
- 本脚本使用 trident.Visualization.visualize_heatmap 函数，与 tutorial notebook 中的用法一致
- 模型需要支持返回 attention scores（如 ABMIL 类型的模型）
- 如果模型不支持 attention，将使用特征 norm 作为替代
"""

import argparse
import os
import torch
import numpy as np
import h5py
from pathlib import Path

from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory
from trident.slide_encoder_models.load import encoder_factory as slide_encoder_factory


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="使用 TRIDENT 进行 WSI heatmap 可视化（类似 CLAM）"
    )
    
    # 基本参数
    parser.add_argument("--gpu", type=int, default=0, help="使用的GPU索引")
    parser.add_argument("--slide_path", type=str, required=True, help="WSI文件路径")
    parser.add_argument("--job_dir", type=str, required=True, help="中间文件输出目录")
    parser.add_argument("--heatmap_save_dir", type=str, default=None, 
                        help="Heatmap保存目录（默认为job_dir/heatmaps）")

    
    # 分割参数
    parser.add_argument('--segmenter', type=str, default='hest', 
                        choices=['hest', 'grandqc'], 
                        help='组织分割器类型')
    parser.add_argument('--seg_conf_thresh', type=float, default=0.5, 
                        help='分割置信度阈值')
    parser.add_argument('--remove_holes', action='store_true', default=False, 
                        help='是否移除孔洞')
    
    # 坐标提取参数
    parser.add_argument("--mag", type=int, choices=[5, 10, 20, 40], default=20,
                        help="提取patches/features的放大倍数")
    parser.add_argument("--patch_size", type=int, default=256, 
                        help="Patch大小")
    parser.add_argument('--overlap', type=int, default=0, 
                        help='Patches之间的重叠像素数')
    
    # Patch encoder参数
    parser.add_argument('--patch_encoder', type=str, default='uni_v2', 
                        choices=['conch_v1', 'uni_v1', 'uni_v2', 'ctranspath', 'phikon', 
                                 'resnet50', 'gigapath', 'virchow', 'virchow2', 
                                 'hoptimus0', 'hoptimus1', 'phikon_v2', 'conch_v15', 
                                 'musk', 'hibou_l', 'kaiko-vits8', 'kaiko-vits16', 
                                 'kaiko-vitb8', 'kaiko-vitb16', 'kaiko-vitl14', 'lunit-vits8'],
                        help='使用的patch encoder')
    parser.add_argument('--patch_encoder_weights_path', type=str, default=None,
                        help='Patch encoder权重路径（可选）')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='特征提取的batch size')
    
    # Slide encoder参数
    parser.add_argument('--slide_encoder', type=str, required=True,
                        choices=['threads', 'titan', 'prism', 'gigapath', 'chief', 
                                 'madeleine', 'feather', 'gfy_abmil','gfy_transmil',
                                 'mean-virchow', 'mean-virchow2', 'mean-conch_v1', 
                                 'mean-conch_v15', 'mean-ctranspath', 'mean-gigapath', 
                                 'mean-resnet50', 'mean-hoptimus0', 'mean-phikon', 
                                 'mean-phikon_v2', 'mean-musk', 'mean-uni_v1', 'mean-uni_v2'],
                        help='使用的slide encoder（用于生成attention）')
    parser.add_argument('--slide_encoder_weights_path', type=str, required=True,
                        help='Slide encoder权重路径（必需）')
    
    # Heatmap可视化参数
    parser.add_argument('--vis_level', type=int, default=4,
                        help='可视化层级')
    parser.add_argument('--cmap', type=str, default='coolwarm',
                        help='Colormap名称（如jet, coolwarm, viridis等）')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Heatmap透明度（0-1）')
    parser.add_argument('--blank_canvas', action='store_true', default=False,
                        help='使用空白画布而非WSI作为背景')
    parser.add_argument('--blur', action='store_true', default=True,
                        help='对heatmap应用高斯模糊')
    parser.add_argument('--convert_to_percentiles', action='store_true', default=True,
                        help='将分数转换为百分位数')
    parser.add_argument('--binarize', action='store_true', default=False,
                        help='二值化heatmap')
    parser.add_argument('--binary_thresh', type=float, default=0.5,
                        help='二值化阈值（0-1）')
    
    # 其他参数
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                        help='用于存储MPP的自定义键')
    parser.add_argument('--save_attention_scores', action='store_true', default=False,
                        help='是否保存attention scores到文件')
    
    return parser.parse_args()


def load_slide_encoder_with_weights(slide_encoder_name, weights_path, device='cuda:0'):
    """
    加载slide encoder模型并加载自定义权重
    """
    slide_encoder = slide_encoder_factory(
        model_name=slide_encoder_name,
        pretrained=False,
    )
    
    if weights_path is not None:
        print(f"加载slide encoder权重: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        slide_encoder.model.load_state_dict(state_dict, strict=True)
        print("Slide encoder权重加载完成")
    else:
        raise ValueError("必须提供slide_encoder_weights_path参数")
    
    slide_encoder.to(device)
    slide_encoder.eval()
    
    return slide_encoder


def get_attention_scores_from_model(
    model,
    patch_features,
    coords,
    coords_attrs,
    device='cuda:0'
):
    """
    从slide encoder模型获取attention scores
    
    参考 tutorial notebook 中的用法，支持 return_raw_attention=True
    
    Args:
        model: Slide encoder模型（支持 return_raw_attention 参数）
        patch_features: Patch特征，numpy array, shape (N, D)
        coords: Patch坐标，numpy array, shape (N, 2)
        coords_attrs: 坐标属性字典
        device: 设备
    
    Returns:
        attention_scores: Attention scores, numpy array, shape (N,)
    """
    # 转换为tensor
    patch_features = torch.from_numpy(patch_features).float().to(device)
    patch_features = patch_features.unsqueeze(0)  # Add batch dimension
    
    coords_tensor = torch.from_numpy(coords).to(device)
    coords_tensor = coords_tensor.unsqueeze(0)  # Add batch dimension
    
    # 准备输入batch（按照 TRIDENT 的格式）
    batch = {
        'features': patch_features,
        'coords': coords_tensor,
        'attributes': coords_attrs
    }
    
    # 获取attention scores
    with torch.no_grad():
        # 方法1: 尝试使用 return_raw_attention=True（如 notebook 中的用法）
        try:
            with torch.autocast(device_type='cuda', enabled=(getattr(model, 'precision', torch.float32) != torch.float32)):
                result = model(batch, device=device, return_raw_attention=True)
                if len(result) == 2:
                    _, attention = result
                    attention_scores = attention.squeeze().cpu().numpy()
                else:
                    raise ValueError("模型返回格式不符合预期")
        except Exception as e:
            print(f"Error getting attention scores: {e}")
            raise e

    # 确保是 1D array
    if attention_scores.ndim > 1:
        attention_scores = attention_scores.squeeze()
    
    return attention_scores


def process_slide_and_generate_heatmap(args):
    """
    处理单个WSI并生成heatmap
    """
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"正在处理slide: {args.slide_path}")
    print(f"{'='*60}\n")
    
    # 设置heatmap保存目录
    if args.heatmap_save_dir is None:
        args.heatmap_save_dir = os.path.join(args.job_dir, 'heatmaps')
    os.makedirs(args.heatmap_save_dir, exist_ok=True)
    
    slide_name = os.path.splitext(os.path.basename(args.slide_path))[0]
    
    # 步骤1: 加载WSI
    seg_output_path = os.path.join(args.job_dir, 'contours_geojson', f'{slide_name}.geojson')
    tissue_seg_path = seg_output_path if os.path.exists(seg_output_path) else None
    
    slide = load_wsi(
        slide_path=args.slide_path, 
        lazy_init=False, 
        custom_mpp_keys=args.custom_mpp_keys,
        tissue_seg_path=tissue_seg_path
    )
    
    # 步骤2: 组织分割
    if not os.path.exists(seg_output_path):
        print("步骤1/5: 正在运行组织分割...")
        segmentation_model = segmentation_model_factory(
            model_name=args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        )
        segmentation_model.to(device)
        
        slide.segment_tissue(
            segmentation_model=segmentation_model,
            target_mag=segmentation_model.target_mag,
            job_dir=args.job_dir,
            device=device,
            holes_are_tissue=not args.remove_holes
        )
        print(f"组织分割完成。结果保存至 {seg_output_path}")
    else:
        print(f"步骤1/5: 组织分割结果已存在，跳过: {seg_output_path}")
    
    # 步骤3: 提取组织坐标
    save_coords = os.path.join(args.job_dir, f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap')
    coords_path = os.path.join(save_coords, 'patches', f'{slide.name}_patches.h5')
    
    if not os.path.exists(coords_path):
        print("步骤2/5: 正在提取组织坐标...")
        coords_path = slide.extract_tissue_coords(
            target_mag=args.mag,
            patch_size=args.patch_size,
            save_coords=save_coords,
            overlap=args.overlap
        )
        print(f"组织坐标提取完成。保存至 {coords_path}")
    else:
        print(f"步骤2/5: 组织坐标文件已存在，跳过: {coords_path}")
    
    # 步骤4: 提取patch特征
    features_dir = os.path.join(save_coords, f"features_{args.patch_encoder}")
    patch_features_h5_path = os.path.join(features_dir, f'{slide.name}.h5')
    
    if not os.path.exists(patch_features_h5_path):
        print("步骤3/5: 正在提取patch特征...")
        
        if args.patch_encoder_weights_path:
            patch_encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_weights_path)
        else:
            patch_encoder = encoder_factory(args.patch_encoder)
        patch_encoder.eval()
        patch_encoder.to(device)
        
        patch_features_path = slide.extract_patch_features(
            patch_encoder=patch_encoder,
            coords_path=coords_path,
            save_features=features_dir,
            device=device,
            batch_limit=args.batch_size
        )
        print(f"Patch特征提取完成。保存至 {patch_features_path}")
    else:
        print(f"步骤3/5: Patch特征文件已存在，跳过: {patch_features_h5_path}")
    
    # 步骤5: 加载slide encoder并获取attention scores
    print(f"步骤4/5: 正在加载slide encoder: {args.slide_encoder}")
    slide_encoder = load_slide_encoder_with_weights(
        slide_encoder_name=args.slide_encoder,
        weights_path=args.slide_encoder_weights_path,
        device=device
    )
    
    # 加载patch features
    with h5py.File(patch_features_h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_features = f['features'][:]
        coords_attrs = dict(f['coords'].attrs)
    
    print(f"获取attention scores (共{len(coords)}个patches)...")
    attention_scores = get_attention_scores_from_model(
        model=slide_encoder,
        patch_features=patch_features,
        coords=coords,
        coords_attrs=coords_attrs,
        device=device
    )
    
    # 保存attention scores（可选）
    if args.save_attention_scores:
        attention_save_path = os.path.join(args.heatmap_save_dir, f'{slide_name}_attention_scores.h5')
        with h5py.File(attention_save_path, 'w') as f:
            f.create_dataset('attention_scores', data=attention_scores)
            f.create_dataset('coords', data=coords)
            for key, val in coords_attrs.items():
                try:
                    f['coords'].attrs[key] = val
                except:
                    pass
        print(f"Attention scores已保存至: {attention_save_path}")
    
    # 步骤6: 生成heatmap可视化
    print(f"步骤5/5: 正在生成heatmap可视化...")
    
    # 使用 slide_name 作为文件名，以区分不同的 slide
    heatmap_filename = f"{slide_name}_heatmap.png"
    
    # 使用 output_dir 参数（符合 visualize_heatmap 的 API）
    heatmap_save_path = slide.visualize_attention_heatmap(
        attention_scores=attention_scores,
        coords=coords,
        patch_size_level0=coords_attrs['patch_size_level0'],
        output_dir=args.heatmap_save_dir,
        vis_level=args.vis_level,
        cmap=args.cmap,
        alpha=args.alpha,
        blank_canvas=args.blank_canvas,
        blur=args.blur,
        overlap=args.overlap,
        normalize=True,  # 默认使用 rank normalization
        convert_to_percentiles=args.convert_to_percentiles,
        binarize=args.binarize,
        thresh=args.binary_thresh,
        num_top_patches_to_save=-1,  # 默认不保存 top patches
        filename=heatmap_filename,  # 使用 slide_id 作为文件名
    )
    
    print(f"\n{'='*60}")
    print(f"Heatmap可视化完成！")
    print(f"保存路径: {heatmap_save_path}")
    print(f"{'='*60}\n")
    
    # 释放资源
    slide.release()
    
    return heatmap_save_path


def main():
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.job_dir, exist_ok=True)
    
    # 处理slide并生成heatmap
    heatmap_path = process_slide_and_generate_heatmap(args)
    
    return heatmap_path


if __name__ == "__main__":
    main()

