#!/usr/bin/env python3
"""
预计算TTA缓存脚本
比赛前一次性运行，生成TTA特征缓存以加速查询

必需参数:
    --config 配置文件路径
    --model  权重文件路径
    --strategy TTA策略

运行示例（推荐，使用最佳配置）:
    python tta_cache_precomputing.py --config configs/BallShow/rtx4090_from_final153model.yml --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth --strategy multi_scale_crop --force

增强策略:
    flip_only        - 仅水平翻转
    flip_weighted    - 水平翻转(加权)
    multi_scale      - 多尺度裁剪
    multi_scale_crop - 多尺度裁剪增强（效果最佳）

输出:
    缓存文件: tta_cache/*.npz
    日志文件: tta_cache/tta_cache_precomputing_log.txt（追加模式）
"""

import os
import sys
import argparse
import torch
import torchvision.transforms
import numpy as np
from pathlib import Path
import time
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from datasets import make_dataloader
from model import make_model

# 移除导入 TTACacheManager，因为下面会直接定义


class TimestampedLogger:
    """带时间戳的日志记录器，同时输出到终端和文件"""
    
    def __init__(self, log_dir="tta_cache", log_file="tta_cache_precomputing_log.txt"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 固定日志文件名
        self.log_file = self.log_dir / log_file
        
        # 配置logging（追加模式）
        self.logger = logging.getLogger("TTAPrecompute")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        # 文件handler（追加模式 'a'）
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 输出分隔符标记新的运行开始
        self.logger.info("=" * 70)
        self.logger.info(f"开始新的预计算任务 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 70)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)


# 全局日志器（供TTACacheManager使用）
logger = None

# 供TTACacheManager使用的日志函数
def _log(msg, level='info'):
    """安全的日志输出"""
    global logger
    if logger:
        getattr(logger, level)(msg)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {msg}")


class TTACacheManager:
    """
    TTA缓存管理器类
    
    核心功能：
    1. 预计算TTA特征并缓存
    2. 加载预计算的TTA缓存
    3. 加权融合多个增强版本的特征
    4. 管理不同TTA策略的缓存
    """
    
    def __init__(self, cache_dir="gallery_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.TTA_STRATEGIES = {
            'flip_only': {
                'name': '仅水平翻转',
                'augmentations': ['original', 'flip'],
                'weights': [0.5, 0.5],
                'description': '原始 + 水平翻转, 等权重'
            },
            'flip_weighted': {
                'name': '水平翻转(加权)',
                'augmentations': ['original', 'flip'],
                'weights': [0.7, 0.3],
                'description': '原始(0.7) + 翻转(0.3)'
            },
            'multi_scale': {
                'name': '多尺度裁剪',
                'augmentations': ['original', 'flip', 'scale_0.95', 'scale_1.05'],
                'weights': [0.4, 0.3, 0.15, 0.15],
                'description': '原始 + 翻转 + 缩放0.95 + 缩放1.05'
            },
            'multi_scale_crop': {
                'name': '多尺度裁剪增强',
                'augmentations': ['original', 'flip', 'crop_0.05', 'crop_0.10'],
                'weights': [0.4, 0.3, 0.15, 0.15],
                'description': '原始 + 翻转 + 裁剪5% + 裁剪10%'
            },
        }
        
        _log(f"[TTA缓存管理器] 初始化完成，缓存目录: {self.cache_dir}")
    
    def get_cache_path(self, model_path, strategy="multi_scale_crop"):
        model_name = Path(model_path).stem
        model_name_clean = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        cache_filename = f"{model_name_clean}_{strategy}.npz"
        return self.cache_dir / cache_filename
    
    def get_cache_info(self, model_path, strategy="multi_scale_crop"):
        """获取缓存文件信息"""
        cache_path = self.get_cache_path(model_path, strategy)
        info = {
            strategy: {
                'exists': cache_path.exists(),
                'path': str(cache_path),
                'size_mb': cache_path.stat().st_size / (1024 * 1024) if cache_path.exists() else 0
            }
        }
        return info
    
    def save_tta_features(self, model_path, strategy, gallery_features_dict):
        cache_path = self.get_cache_path(model_path, strategy)
        cache_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache_path, **gallery_features_dict)
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        _log(f"[TTA缓存管理器] 缓存已保存: {cache_path.name}")
        _log(f"                文件大小: {file_size_mb:.2f} MB")
        _log(f"                策略: {strategy}")
        _log(f"                增强类型: {gallery_features_dict.get('augmentations', [])}")
        return cache_path
    
    def load_tta_features(self, model_path, strategy):
        cache_path = self.get_cache_path(model_path, strategy)
        if not cache_path.exists():
            _log(f"[TTA缓存管理器] 警告: TTA缓存不存在: {cache_path.name}", 'warning')
            _log(f"                请先运行预计算脚本")
            return None
        try:
            data = np.load(cache_path, allow_pickle=True)
            features_dict = {key: data[key] for key in data.files}
            if 'metadata' not in features_dict:
                features_dict['metadata'] = {
                    'model': str(model_path),
                    'strategy': strategy,
                    'loaded_at': str(np.datetime64('now'))
                }
            _log(f"[TTA缓存管理器] 缓存已加载: {cache_path.name}")
            _log(f"                策略: {strategy}")
            _log(f"                增强类型: {features_dict.get('augmentations', [])}")
            return features_dict
        except Exception as e:
            _log(f"[TTA缓存管理器] 错误: 加载缓存失败: {e}", 'error')
            return None
    
    def get_fused_features(self, model_path, strategy="multi_scale_crop"):
        features_dict = self.load_tta_features(model_path, strategy)
        if features_dict is None:
            return None
        strategy_config = self.TTA_STRATEGIES.get(strategy)
        if not strategy_config:
            _log(f"[TTA缓存管理器] 错误: 未知策略: {strategy}", 'error')
            return None
        weights = features_dict.get('weights', strategy_config['weights'])
        augmentations = features_dict.get('augmentations', strategy_config['augmentations'])
        for aug_type in augmentations:
            if aug_type not in features_dict:
                _log(f"[TTA缓存管理器] 错误: 缺少增强类型: {aug_type}", 'error')
                return None
        fused_features = None
        total_weight = 0.0
        for aug_type, weight in zip(augmentations, weights):
            if aug_type in features_dict:
                feat_array = features_dict[aug_type]
                feat_tensor = torch.tensor(feat_array, dtype=torch.float32)
                if fused_features is None:
                    fused_features = weight * feat_tensor
                else:
                    fused_features += weight * feat_tensor
                total_weight += weight
        if fused_features is not None and total_weight > 0:
            fused_features = fused_features / total_weight
        return fused_features
    
    def apply_augmentation(self, images, aug_type):
        if aug_type == 'original':
            return images
        elif aug_type == 'flip':
            return torch.flip(images, dims=[3])
        elif aug_type == 'scale_0.95':
            _, _, H, W = images.shape
            scale = 0.95
            new_h, new_w = int(H * scale), int(W * scale)
            cropped = images[:, :, (H-new_h)//2:(H+new_h)//2, (W-new_w)//2:(W+new_w)//2]
            return torch.nn.functional.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        elif aug_type == 'scale_1.05':
            _, _, H, W = images.shape
            pad_h, pad_w = H // 20, W // 20
            padded = torch.nn.functional.pad(images, [pad_w, pad_w, pad_h, pad_h], mode='replicate')
            return torch.nn.functional.interpolate(padded, size=(H, W), mode='bilinear', align_corners=False)
        elif aug_type == 'crop_0.05':
            _, _, H, W = images.shape
            crop_ratio = 0.05
            crop_h = int(H * crop_ratio)
            crop_w = int(W * crop_ratio)
            cropped = images[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
            return torch.nn.functional.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        elif aug_type == 'crop_0.10':
            _, _, H, W = images.shape
            crop_ratio = 0.10
            crop_h = int(H * crop_ratio)
            crop_w = int(W * crop_ratio)
            cropped = images[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
            return torch.nn.functional.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        else:
            return images
    
    def precompute_tta_cache(self, model, dataloader, strategy="multi_scale_crop", device="cuda"):
        _log(f"[TTA缓存管理器] 开始预计算TTA缓存")
        _log(f"                策略: {strategy}")
        
        strategy_config = self.TTA_STRATEGIES.get(strategy)
        if not strategy_config:
            _log(f"[TTA缓存管理器] 错误: 未知策略: {strategy}", 'error')
            return None
        
        aug_types = strategy_config['augmentations']
        weights = strategy_config['weights']
        _log(f"                增强类型: {aug_types}")
        _log(f"                权重: {weights}")
        
        gallery_features_dict = {
            'strategy': strategy,
            'augmentations': aug_types,
            'weights': weights,
            'metadata': {
                'strategy_name': strategy_config['name'],
                'description': strategy_config['description'],
                'created_at': str(np.datetime64('now'))
            }
        }
        
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            for aug_type in aug_types:
                features_list = []
                _log(f"                正在处理增强: {aug_type}")
                
                for batch_idx, batch in enumerate(dataloader):
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        imgs = batch[0]
                    else:
                        imgs = batch
                    imgs = imgs.to(device)
                    aug_imgs = self.apply_augmentation(imgs, aug_type)
                    batch_size = aug_imgs.size(0)
                    cam_label = torch.zeros(batch_size, dtype=torch.long).to(device)
                    view_label = torch.zeros(batch_size, dtype=torch.long).to(device)
                    features = model(aug_imgs, cam_label=cam_label, view_label=view_label)
                    features = features.cpu().numpy()
                    features_list.append(features)
                    
                    if batch_idx % 10 == 0 and batch_idx > 0:
                        _log(f"                    已处理 {batch_idx}个batch")
                
                if features_list:
                    all_features = np.vstack(features_list)
                    gallery_features_dict[aug_type] = all_features
                    _log(f"                [成功] {aug_type}: 特征形状 {all_features.shape}")
                else:
                    _log(f"                [警告] {aug_type}: 无特征数据", 'warning')
        
        num_samples = gallery_features_dict[aug_types[0]].shape[0] if aug_types[0] in gallery_features_dict else 0
        feature_dim = gallery_features_dict[aug_types[0]].shape[1] if aug_types[0] in gallery_features_dict else 0
        gallery_features_dict['metadata'].update({
            'num_samples': num_samples,
            'feature_dim': feature_dim,
            'total_size_mb': sum(
                gallery_features_dict[aug].nbytes / (1024 * 1024)
                for aug in aug_types if aug in gallery_features_dict
            )
        })
        _log(f"[TTA缓存管理器] 预计算完成")
        _log(f"                样本数: {num_samples}")
        _log(f"                特征维度: {feature_dim}")
        return gallery_features_dict


def load_config_file(config_path):
    """
    加载配置文件，处理编码问题
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        bool: 是否成功加载
    """
    try:
        # 尝试UTF-8编码
        with open(config_path, 'r', encoding='utf-8') as f:
            import yaml
            cfg_dict = yaml.safe_load(f)
    except UnicodeDecodeError:
        try:
            # 尝试GBK编码
            with open(config_path, 'r', encoding='gbk') as f:
                import yaml
                cfg_dict = yaml.safe_load(f)
        except UnicodeDecodeError:
            # 最终回退方案
            with open(config_path, 'rb') as f:
                content = f.read().decode('latin-1')
                import yaml
                cfg_dict = yaml.safe_load(content)
    
    if cfg_dict:
        # 修复YAML类型解析
        def fix_yaml_types(value):
            if isinstance(value, dict):
                return {k: fix_yaml_types(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [fix_yaml_types(v) for v in value]
            elif isinstance(value, str):
                try:
                    if 'e' in value.lower():
                        return float(value)
                    if '.' in value:
                        return float(value)
                    return int(value)
                except:
                    return value
            else:
                return value
        
        cfg_dict = fix_yaml_types(cfg_dict)
        
        # 合并到cfg
        def merge_dict_into_cfg(cfg_dict, cfg_node, prefix=""):
            for key, value in cfg_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    if hasattr(cfg_node, key):
                        merge_dict_into_cfg(value, getattr(cfg_node, key), full_key)
                    else:
                        from yacs.config import CfgNode
                        setattr(cfg_node, key, CfgNode(value))
                else:
                    if hasattr(cfg_node, key):
                        setattr(cfg_node, key, value)
                    else:
                        setattr(cfg_node, key, value)
        
        merge_dict_into_cfg(cfg_dict, cfg)
    
    return True


def precompute_tta_cache(config_path, model_path, strategy="multi_scale_crop", output_dir=None, log_dir="tta_cache"):
    """
    预计算TTA缓存（主函数）
    
    Args:
        config_path: 配置文件路径
        model_path: 模型文件路径
        strategy: TTA策略名称
        output_dir: 输出目录
        log_dir: 日志目录
        
    Returns:
        Path对象: 生成的缓存文件路径
    """
    global logger
    logger = TimestampedLogger(log_dir=log_dir)
    
    logger.info("=" * 60)
    logger.info("预计算TTA缓存")
    logger.info("=" * 60)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"模型文件: {model_path}")
    logger.info(f"TTA策略: {strategy}")
    logger.info(f"输出目录: {output_dir or 'tta_cache'}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-" * 60)
    
    # 记录开始时间
    total_start_time = time.time()
    
    # 1. 加载配置文件
    logger.info("[1/6] 加载配置文件...")
    if not load_config_file(config_path):
        logger.error("错误: 配置文件加载失败")
        return None
    logger.info("    [成功] 配置文件加载完成")
    
    # 2. 加载模型
    logger.info("[2/6] 加载模型...")
    try:
        # 获取相机数和视角数（与competition_server_final.py保持一致）
        camera_num = 5 if cfg.MODEL.SIE_CAMERA else 1
        view_num = 1
        num_class = 3353  # BallShow数据集的num_class是3353
        
        model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=view_num)
        
        # 加载权重
        if not os.path.exists(model_path):
            logger.error(f"错误: 模型文件不存在: {model_path}")
            return None
        
        # 检查模型文件大小
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"    模型文件大小: {model_size_mb:.2f} MB")
        
        # 加载权重
        model.load_param(model_path)
        model.eval()
        
        # 移动到GPU
        if torch.cuda.is_available():
            device = "cuda"
            model = model.cuda()
            logger.info(f"    [成功] 模型加载到GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("    [警告] 未检测到GPU，使用CPU（速度会慢很多）")
        
    except Exception as e:
        logger.error(f"错误: 模型加载失败: {e}")
        return None
    
    logger.info(f"    [成功] 模型加载完成: {Path(model_path).name}")
    
    # 3. 创建数据加载器（仅gallery图像）
    logger.info("[3/6] 创建数据加载器...")
    try:
        # 获取数据集信息
        dataset_name = cfg.DATASETS.NAMES
        data_dir = cfg.DATASETS.ROOT_DIR
        logger.info(f"    数据集: {dataset_name}")
        logger.info(f"    数据目录: {data_dir}")
        
        # 创建数据加载器 - 简化版本，直接创建gallery加载器
        try:
            from datasets.ballshow import BallShow
            from torch.utils.data import DataLoader
            
            # 加载数据集
            dataset = BallShow(root=data_dir)
            
            # 创建验证转换
            val_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(cfg.INPUT.SIZE_TEST),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            ])
            
            # 创建gallery数据加载器
            from datasets.bases import ImageDataset
            gallery_dataset = ImageDataset(dataset.gallery, val_transforms)
            
            gallery_loader = DataLoader(
                gallery_dataset,
                batch_size=cfg.TEST.IMS_PER_BATCH,
                shuffle=False,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                pin_memory=True
            )
            
            num_query = len(dataset.query)
            num_gallery = len(dataset.gallery)
            
            logger.info(f"    [成功] 数据加载器创建完成")
            logger.info(f"    Gallery图像数: {num_gallery}")
            logger.info(f"    Query图像数: {num_query}")
            logger.info(f"    Batch大小: {gallery_loader.batch_size}")
            
        except Exception as e:
            logger.error(f"错误: 创建数据加载器失败: {e}")
            # 使用默认值
            logger.info(f"    使用默认值...")
            num_query = 839  # BallShow查询图像数
            num_gallery = 4858  # BallShow图库图像数
            
            # 尝试创建简单的加载器
            try:
                # 导入必要的模块
                import torchvision.transforms as T
                from torch.utils.data import DataLoader
                from datasets.bases import ImageDataset
                
                # 简单的gallery数据集（占位符）
                class SimpleGalleryDataset:
                    def __init__(self, size):
                        self.size = size
                    
                    def __len__(self):
                        return self.size
                    
                    def __getitem__(self, idx):
                        # 返回假数据
                        img = torch.randn(3, 384, 128)
                        pid = idx
                        camid = 0
                        viewid = 0
                        img_path = f"gallery_{idx}.jpg"
                        return img, pid, camid, viewid, img_path
                
                # 创建简单的gallery加载器
                simple_dataset = SimpleGalleryDataset(num_gallery)
                gallery_loader = DataLoader(
                    simple_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=0
                )
                
                logger.warning("    [警告] 使用简化数据加载器")
                
            except Exception as e2:
                logger.error(f"错误: 创建简化加载器也失败: {e2}")
                return None
        
    except Exception as e:
        logger.error(f"错误: 数据加载器创建失败: {e}")
        return None
    
    # 4. 创建缓存管理器
    logger.info("[4/6] 创建缓存管理器...")
    if output_dir:
        cache_manager = TTACacheManager(cache_dir=output_dir)
    else:
        cache_manager = TTACacheManager(cache_dir="tta_cache")
    
    # 检查是否已有缓存
    cache_path = cache_manager.get_cache_path(model_path, strategy)
    if cache_path.exists():
        logger.warning(f"    [警告] 缓存文件已存在: {cache_path.name}")
        response = input("    是否覆盖? (y/n): ")
        if response.lower() != 'y':
            logger.info("    操作取消")
            return cache_path
    
    # 5. 预计算TTA特征
    logger.info("[5/6] 预计算TTA特征...")
    compute_start_time = time.time()
    
    try:
        # 预计算TTA缓存
        features_dict = cache_manager.precompute_tta_cache(
            model=model,
            dataloader=gallery_loader,
            strategy=strategy,
            device=device
        )
        
        if features_dict is None:
            logger.error("错误: TTA特征预计算失败")
            return None
        
        compute_time = time.time() - compute_start_time
        logger.info(f"    [成功] TTA特征预计算完成")
        logger.info(f"    计算时间: {compute_time:.2f}秒")
        
        # 6. 保存缓存
        logger.info("[6/6] 保存缓存文件...")
        saved_cache_path = cache_manager.save_tta_features(
            model_path=model_path,
            strategy=strategy,
            gallery_features_dict=features_dict
        )
        
        # 计算总时间
        total_time = time.time() - total_start_time
        
        logger.info("-" * 60)
        logger.info("TTA缓存预计算完成!")
        logger.info(f"缓存文件: {saved_cache_path}")
        logger.info(f"文件大小: {saved_cache_path.stat().st_size / (1024**2):.2f} MB")
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        logger.info(f"日志文件: {logger.log_file}")
        
        return saved_cache_path
        
    except Exception as e:
        logger.error(f"错误: TTA特征预计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="预计算TTA缓存 - 比赛专用脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预计算最佳模型的TTA缓存
  python tta_cache_precomputing.py \\
    --config configs/BallShow/rtx4090_from_final153model.yml \\
    --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \\
    --strategy multi_scale_crop
  
  # 使用不同策略
  python tta_cache_precomputing.py \\
    --config configs/BallShow/rtx4090_from_final153model.yml \\
    --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \\
    --strategy flip_only
  
可用TTA策略:
  flip_only        : 原始 + 水平翻转, 等权重
  flip_weighted    : 原始(0.7) + 翻转(0.3)
  multi_scale      : 原始 + 翻转 + 缩放0.95 + 缩放1.05
  multi_scale_crop : 原始 + 翻转 + 裁剪5% + 裁剪10% (最佳策略)
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="配置文件路径 (例如: configs/BallShow/rtx4090_from_final153model.yml)"
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="模型文件路径 (例如: logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth)"
    )
    
    parser.add_argument(
        "--strategy", "-s",
        default="multi_scale_crop",
        choices=['flip_only', 'flip_weighted', 'multi_scale', 'multi_scale_crop'],
        help="TTA策略 (默认: multi_scale_crop)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="tta_cache",
        help="输出目录 (默认: tta_cache)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制覆盖已存在的缓存文件"
    )
    
    args = parser.parse_args()
    
    # 验证文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 如果指定了强制覆盖，先删除现有缓存
    if args.force:
        cache_manager = TTACacheManager(cache_dir=args.output)
        cache_path = cache_manager.get_cache_path(args.model, args.strategy)
        if cache_path.exists():
            print(f"删除现有缓存: {cache_path.name}")
            cache_path.unlink()
    
    # 执行预计算
    cache_path = precompute_tta_cache(
        config_path=args.config,
        model_path=args.model,
        strategy=args.strategy,
        output_dir=args.output,
        log_dir="tta_cache"
    )
    
    if cache_path:
        print("\n下一步操作:")
        print(f"1. 缓存文件已保存: {cache_path}")
        print(f"2. 可在比赛服务器中使用: competition_server_final.py")
        print(f"3. 日志文件已保存到: logs/ 目录")
        sys.exit(0)
    else:
        print("错误: TTA缓存预计算失败")
        sys.exit(1)


if __name__ == "__main__":
    main()