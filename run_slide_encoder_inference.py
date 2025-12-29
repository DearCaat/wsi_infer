"""
使用TRIDENT处理WSI并利用slide_encoder进行推理的脚本

示例用法:
```
python run_slide_encoder_inference.py \
    --slide_path output/wsis/394140.svs \
    --job_dir output/ \
    --mag 20 \
    --patch_size 256 \
    --seg_weights_path path/to/seg_weights.ckpt \
    --patch_encoder uni_v2 \
    --patch_encoder_weights_path path/to/patch_encoder_weights.pt \
    --slide_encoder threads \
    --slide_encoder_weights_path path/to/slide_encoder_weights.pt
```

"""
import argparse
import os
import torch
import numpy as np
import h5py

from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory
from trident.slide_encoder_models.load import encoder_factory as slide_encoder_factory


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="使用TRIDENT处理WSI并通过slide_encoder进行推理")
    
    # 基本参数
    parser.add_argument("--gpu", type=int, default=0, help="使用的GPU索引")
    parser.add_argument("--slide_path", type=str, required=True, help="WSI文件路径")
    parser.add_argument("--job_dir", type=str, required=True, help="输出目录")
    
    # 分割参数
    parser.add_argument('--segmenter', type=str, default='hest', 
                        choices=['hest', 'grandqc'], 
                        help='组织分割器类型')
    parser.add_argument('--seg_weights_path', type=str, default=None,
                        help='Segmentation模型权重路径（可选，如果提供则加载自定义权重）')
    parser.add_argument('--seg_conf_thresh', type=float, default=0.5, 
                        help='分割置信度阈值')
    parser.add_argument('--remove_holes', action='store_true', default=False, 
                        help='是否移除孔洞')
    parser.add_argument('--remove_artifacts', action='store_true', default=False, 
                        help='是否移除伪影')
    parser.add_argument('--remove_penmarks', action='store_true', default=False, 
                        help='是否移除笔迹标记')
    
    # 坐标提取参数
    parser.add_argument("--mag", type=int, choices=[5, 10, 20, 40], default=20,
                        help="提取patches/features的放大倍数")
    parser.add_argument("--patch_size", type=int, default=256, 
                        help="提取coords/features的patch大小")
    parser.add_argument('--overlap', type=int, default=0, 
                        help='patches之间的重叠像素数')
    
    # Patch encoder参数
    parser.add_argument('--patch_encoder', type=str, default='uni_v2', 
                        choices=['conch_v1', 'uni_v1', 'uni_v2', 'ctranspath', 'phikon', 
                                 'resnet50', 'gigapath', 'virchow', 'virchow2', 
                                 'hoptimus0', 'hoptimus1', 'phikon_v2', 'conch_v15', 'musk', 'hibou_l',
                                 'kaiko-vits8', 'kaiko-vits16', 'kaiko-vitb8', 'kaiko-vitb16',
                                 'kaiko-vitl14', 'lunit-vits8'],
                        help='使用的patch encoder')
    parser.add_argument('--patch_encoder_weights_path', type=str,
                        help='Patch encoder权重路径（可选，显式指定权重文件路径）')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='特征提取的batch size')
    
    # Slide encoder参数
    parser.add_argument('--slide_encoder', type=str, default='gfy_abmil', required=True,
                        choices=['threads', 'titan', 'prism', 'gigapath', 'chief', 'madeleine', 'feather',
                                 'mean-virchow', 'mean-virchow2', 'mean-conch_v1', 'mean-conch_v15', 
                                 'mean-ctranspath', 'mean-gigapath', 'mean-resnet50', 'mean-hoptimus0', 
                                 'mean-phikon', 'mean-phikon_v2', 'mean-musk', 'mean-uni_v1', 'mean-uni_v2',
                                 'gfy_abmil'],
                        help='使用的slide encoder')
    parser.add_argument('--slide_encoder_weights_path', type=str,
                        help='Slide encoder权重路径（可选，显式指定权重文件路径）')
    
    # 其他参数
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                        help='用于存储MPP（micron per pixel）的自定义键')
    parser.add_argument('--save_slide_features', action='store_true', default=False,
                        help='是否保存slide features到文件')
    
    return parser.parse_args()


def load_segmentation_model_with_weights(segmenter_name, weights_path=None, confidence_thresh=0.5, device='cuda:0'):
    """
    加载segmentation模型，可选择加载自定义权重
    
    Args:
        segmenter_name: segmentation模型名称
        weights_path: 自定义权重路径（可选）
        confidence_thresh: 置信度阈值
        device: 设备
    
    Returns:
        segmentation模型
    """
    # 加载segmentation模型
    segmentation_model = segmentation_model_factory(
        model_name=segmenter_name,
        confidence_thresh=confidence_thresh,
    )
    
    # 如果提供了自定义权重路径，加载权重
    if weights_path is not None:
        print(f"加载自定义segmentation权重: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        # 加载权重
        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 移除可能的前缀（如'model.'）
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # 移除aux相关的键（用于segmentation模型）
        state_dict = {k: v for k, v in state_dict.items() if 'aux' not in k}
        
        segmentation_model.model.load_state_dict(state_dict, strict=False)
        print("Segmentation权重加载完成")
    
    segmentation_model.to(device)
    segmentation_model.eval()
    
    return segmentation_model


def load_slide_encoder_with_weights(slide_encoder_name, weights_path, pretrained=False, device='cuda:0'):
    """
    加载slide encoder模型并加载自定义权重
    
    Args:
        slide_encoder_name: slide encoder名称
        weights_path: 权重路径（必需）
        pretrained: 是否加载预训练权重（通常为False，因为使用自定义权重）
        device: 设备
    
    Returns:
        slide encoder模型
    """
    # 加载slide encoder（不加载预训练权重，因为我们将加载自定义权重）
    slide_encoder = slide_encoder_factory(
        model_name=slide_encoder_name,
        pretrained=pretrained,
    )
    
    # 加载自定义权重
    if weights_path is not None:
        print(f"加载自定义权重: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        # 加载权重
        if weights_path.endswith('.pt') or weights_path.endswith('.pth'):
            # 尝试使用weights_only参数（PyTorch 2.0+），如果不支持则回退
            try:
                checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
            except TypeError:
                # 旧版本PyTorch不支持weights_only参数
                checkpoint = torch.load(weights_path, map_location='cpu')
            
            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 移除可能的前缀（如'model.'）
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            slide_encoder.model.load_state_dict(state_dict, strict=True)
            print("权重加载完成")
        else:
            raise ValueError(f"不支持的权重文件格式: {weights_path}")
    
    else:
        raise ValueError("必须提供slide_encoder_weights_path参数")
    
    slide_encoder.to(device)
    slide_encoder.eval()
    
    return slide_encoder


def process_slide_with_slide_encoder(args):
    """
    处理单个WSI：seg -> coords -> feat -> slide_encoder推理
    
    Args:
        args: 命令行参数
    
    Returns:
        slide features (numpy array)
    """
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    # 步骤1: 初始化WSI
    print(f"正在处理slide: {args.slide_path}")
    
    # 检查分割结果是否已存在
    slide_name = os.path.splitext(os.path.basename(args.slide_path))[0]
    seg_output_path = os.path.join(args.job_dir, 'contours_geojson', f'{slide_name}.geojson')
    tissue_seg_path = seg_output_path if os.path.exists(seg_output_path) else None
    
    slide = load_wsi(
        slide_path=args.slide_path, 
        lazy_init=False, 
        custom_mpp_keys=args.custom_mpp_keys,
        tissue_seg_path=tissue_seg_path
    )
    
    # 步骤2: 组织分割 (seg)
    if os.path.exists(seg_output_path):
        print(f"组织分割结果已存在，跳过分割步骤: {seg_output_path}")
    else:
        print("正在运行组织分割...")
        if args.seg_weights_path:
            segmentation_model = load_segmentation_model_with_weights(
                segmenter_name=args.segmenter,
                weights_path=args.seg_weights_path,
                confidence_thresh=args.seg_conf_thresh,
                device=device
            )
        else:
            segmentation_model = segmentation_model_factory(
                model_name=args.segmenter,
                confidence_thresh=args.seg_conf_thresh,
            )
            segmentation_model.to(device)
        if args.remove_artifacts or args.remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=args.remove_penmarks and not args.remove_artifacts
            )
        else:
            artifact_remover_model = None
        
        slide.segment_tissue(
            segmentation_model=segmentation_model,
            target_mag=segmentation_model.target_mag,
            job_dir=args.job_dir,
            device=device,
            holes_are_tissue=not args.remove_holes
        )
        # 额外移除伪影
        if artifact_remover_model is not None:
            slide.segment_tissue(
                segmentation_model=artifact_remover_model,
                target_mag=artifact_remover_model.target_mag,
                holes_are_tissue=False,
                job_dir=args.job_dir
            )
        print(f"组织分割完成。结果保存至 {os.path.join(args.job_dir, 'contours_geojson')}")
    
    # 步骤3: 提取组织坐标 (coords)
    save_coords = os.path.join(args.job_dir, f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap')
    coords_path = os.path.join(save_coords, 'patches', f'{slide.name}_patches.h5')
    
    if os.path.exists(coords_path):
        print(f"组织坐标文件已存在，跳过坐标提取步骤: {coords_path}")
    else:
        print("正在提取组织坐标...")
        coords_path = slide.extract_tissue_coords(
            target_mag=args.mag,
            patch_size=args.patch_size,
            save_coords=save_coords
        )
        print(f"组织坐标提取完成。保存至 {coords_path}")
    
    # 步骤4: 提取patch特征 (feat)
    features_dir = os.path.join(save_coords, f"features_{args.patch_encoder}")
    patch_features_h5_path = os.path.join(features_dir, f'{slide.name}.h5')
    
    if os.path.exists(patch_features_h5_path):
        print(f"Patch特征文件已存在，跳过特征提取步骤: {patch_features_h5_path}")
    else:
        print("正在从patches提取特征...")
        
        if args.patch_encoder_weights_path:
            if not os.path.exists(args.patch_encoder_weights_path):
                raise FileNotFoundError(f"Patch encoder权重文件不存在: {args.patch_encoder_weights_path}")
            print(f"加载patch encoder权重: {args.patch_encoder_weights_path}")
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
    
    # 步骤5: 加载slide encoder并加载权重
    print(f"正在加载slide encoder: {args.slide_encoder}")
    if not os.path.exists(args.slide_encoder_weights_path):
        raise FileNotFoundError(f"Slide encoder权重文件不存在: {args.slide_encoder_weights_path}")
    slide_encoder = load_slide_encoder_with_weights(
        slide_encoder_name=args.slide_encoder,
        weights_path=args.slide_encoder_weights_path,
        pretrained=False,  # 使用自定义权重，不使用预训练
        device=device
    )
    
    # 步骤6: 使用slide encoder处理patch features
    print("正在使用slide encoder完成下游任务...")
    
    # 加载patch features
    with h5py.File(patch_features_h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_features = f['features'][:]
        coords_attrs = dict(f['coords'].attrs)
    
    # 转换为tensor
    patch_features = torch.from_numpy(patch_features).float().to(device)
    patch_features = patch_features.unsqueeze(0)  # 添加batch维度
    
    coords = torch.from_numpy(coords).to(device)
    coords = coords.unsqueeze(0)  # 添加batch维度
    
    # 准备输入batch字典
    batch = {
        'features': patch_features,
        'coords': coords,
        'attributes': coords_attrs
    }
    
    # 完成下游任务
    with torch.no_grad():
        with torch.autocast(device_type='cuda', enabled=(slide_encoder.precision != torch.float32)):
            slide_features = slide_encoder(batch, device=device)
        slide_features = slide_features.float().cpu()
    
    if slide_features.ndim == 1:
        slide_features = slide_features.reshape(1, -1)
    prob = torch.softmax(slide_features, dim=-1).numpy()
    
    print(f"下游任务完成。结果: {prob}")
    return prob


def main():
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.job_dir, exist_ok=True)
    
    # 处理slide并获取结果
    prob = process_slide_with_slide_encoder(args)
    
    
    return prob


if __name__ == "__main__":
    main()

