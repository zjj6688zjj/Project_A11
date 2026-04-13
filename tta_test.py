#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) 测试脚本
在推理阶段对图片进行增强，提升预测稳定性

必需参数:
    --config_file 配置文件路径
    --weight      权重文件路径

运行示例（推荐，使用最佳配置）:
    python test_tta.py --config_file configs/BallShow/rtx4090_from_final153model.yml --weight logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth --strategy multi_scale_crop

增强策略:
    flip_only        - 仅水平翻转
    flip_weighted    - 水平翻转(加权)
    multi_scale      - 多尺度裁剪
    multi_scale_crop - 多尺度裁剪增强（效果最佳）
    all              - 测试所有策略
"""

import os
import torch
import torch.nn as nn
import argparse
import yaml
import logging
from datetime import datetime

# 添加项目根目录到路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.metrics import R1_mAP_eval


def setup_logger(name, output_dir):
    """设置日志，格式与test_log.txt一致"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复输出
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 自定义格式化：时间戳 + 模块名 + 级别 + 消息
    class TimestampedFormatter(logging.Formatter):
        def format(self, record):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            return f"{timestamp} {record.name} {record.levelname}: {record.getMessage()}"
    
    formatter = TimestampedFormatter()
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出到 output_dir/tta_test_log.txt
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'tta_test_log.txt')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_config_and_model(args):
    """加载配置和模型"""
    # 加载配置文件
    try:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
    except UnicodeDecodeError:
        try:
            with open(args.config_file, 'r', 'gbk') as f:
                cfg_dict = yaml.safe_load(f)
        except:
            with open(args.config_file, 'rb') as f:
                content = f.read().decode('latin-1')
                cfg_dict = yaml.safe_load(content)
    
    # 修复YAML类型解析问题
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
        return value
    
    cfg_dict = fix_yaml_types(cfg_dict)
    
    def merge_dict_into_cfg(cfg_dict, cfg_node, prefix=""):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                if hasattr(cfg_node, key):
                    merge_dict_into_cfg(value, getattr(cfg_node, key), f"{prefix}.{key}" if prefix else key)
                else:
                    from yacs.config import CfgNode
                    setattr(cfg_node, key, CfgNode(value))
            else:
                if hasattr(cfg_node, key):
                    setattr(cfg_node, key, value)
                else:
                    setattr(cfg_node, key, value)
    
    merge_dict_into_cfg(cfg_dict, cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    return cfg


# TTA策略配置
TTA_STRATEGIES = {
    'flip_only': {
        'name': '仅水平翻转',
        'augmentations': ['original', 'flip'],
        'weights': [0.5, 0.5],  # 加权融合
        'description': '原始 + 水平翻转, 等权重'
    },
    'flip_weighted': {
        'name': '水平翻转(加权)',
        'augmentations': ['original', 'flip'],
        'weights': [0.7, 0.3],  # 原始权重更高
        'description': '原始(0.7) + 翻转(0.3)'
    },
    'multi_scale': {
        'name': '多尺度裁剪',
        'augmentations': ['original', 'flip', 'scale_0.95', 'scale_1.05'],
        'weights': [0.4, 0.3, 0.15, 0.15],  # 加权融合
        'description': '原始 + 翻转 + 缩放0.95 + 缩放1.05'
    },
    'multi_scale_crop': {
        'name': '多尺度裁剪增强',
        'augmentations': ['original', 'flip', 'crop_0.05', 'crop_0.10'],
        'weights': [0.4, 0.3, 0.15, 0.15],
        'description': '原始 + 翻转 + 裁剪5% + 裁剪10%'
    },
}


def apply_augmentation(images, aug_type):
    """应用数据增强"""
    if aug_type == 'original':
        return images
    elif aug_type == 'flip':
        return torch.flip(images, dims=[3])
    elif aug_type == 'scale_0.95':
        _, _, H, W = images.shape
        scale = 0.95
        new_h, new_w = int(H * scale), int(W * scale)
        cropped = images[:, :, 
                        (H-new_h)//2:(H+new_h)//2,
                        (W-new_w)//2:(W+new_w)//2]
        return torch.nn.functional.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    elif aug_type == 'scale_1.05':
        _, _, H, W = images.shape
        # 先pad再裁剪中心
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


def extract_features_with_tta(model, images, camids, target_view, device, strategy_key='flip_only'):
    """
    使用TTA提取特征
    
    Args:
        model: 模型
        images: 图片张量 [B, C, H, W]
        camids: 相机ID
        target_view: 视角ID
        device: 设备
        strategy_key: TTA策略key
    
    Returns:
        features: 融合后的特征 [B, D]
    """
    model.eval()
    
    with torch.no_grad():
        images = images.to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)
        
        strategy = TTA_STRATEGIES[strategy_key]
        aug_list = strategy['augmentations']
        weights = strategy['weights']
        
        # 动态验证weights数量与aug_list匹配
        assert len(weights) == len(aug_list), \
            f"TTA策略'{strategy_key}': weights数量({len(weights)})与augmentations数量({len(aug_list)})不匹配"
        
        features_list = []
        
        for aug_type in aug_list:
            aug_images = apply_augmentation(images, aug_type)
            feat = model(aug_images, cam_label=camids, view_label=target_view)
            if isinstance(feat, tuple):
                feat = feat[0]
            features_list.append(feat)
        
        # 动态获取特征维度（从实际模型输出推断）
        feat_dim = features_list[0].shape[1]
        
        # 验证特征维度一致性
        feat_dims = [f.shape[1] for f in features_list]
        if len(set(feat_dims)) > 1:
            raise ValueError(f"TTA特征维度不一致: {feat_dims}")
        
        # L2归一化后加权平均
        features_list = [torch.nn.functional.normalize(f, p=2, dim=1) for f in features_list]
        weights_tensor = torch.tensor(weights, device=features_list[0].device).view(-1, 1, 1)  # [N, 1, 1]
        fused_features = torch.stack(features_list, dim=0)  # [N, B, D]
        fused_features = (fused_features * weights_tensor).sum(dim=0)  # [B, D]
        fused_features = torch.nn.functional.normalize(fused_features, p=2, dim=1)
    
    return fused_features


def do_inference_with_tta(cfg, model, val_loader, num_query, use_reranking=False, strategy_key='flip_only'):
    """使用TTA进行推理"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("transreid.tta")
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=use_reranking)
    evaluator.reset()
    
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for inference')
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    img_path_list = []
    strategy = TTA_STRATEGIES[strategy_key]
    logger.info(f"开始推理 - {strategy['name']}")
    
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        feat = extract_features_with_tta(model, img, camids, target_view, device, strategy_key=strategy_key)
        evaluator.update((feat, pid, camid))
        img_path_list.extend(imgpath)
        
        if (n_iter + 1) % 20 == 0:
            logger.info(f"  处理批次 {n_iter + 1}/{len(val_loader)}")
    
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    logger.info("=" * 60)
    logger.info(f"推理模式: {strategy['name']}")
    logger.info(f"详细策略: {strategy['description']}")
    logger.info("=" * 60)
    logger.info("mAP: {:.4f}%".format(mAP * 100))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}: {:.4f}%".format(r, cmc[r - 1] * 100))
    logger.info("=" * 60)
    
    return cmc, mAP, strategy['name']


def main():
    parser = argparse.ArgumentParser(description="TransReID TTA测试")
    parser.add_argument("--config_file", default="", help="配置文件路径", type=str)
    parser.add_argument("--weight", default="", help="权重文件路径", type=str)
    parser.add_argument("--reranking", action="store_true", help="使用re-ranking")
    parser.add_argument("--no_tta", action="store_true", help="不使用TTA（仅对比实验）")
    parser.add_argument("--strategy", default="all", help="TTA策略: all, flip_only, flip_weighted, multi_scale, multi_scale_crop")
    parser.add_argument("opts", help="修改配置选项", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # 先加载配置，获取输出目录
    cfg = load_config_and_model(args)
    
    # 设置权重路径
    if args.weight:
        cfg.defrost()
        cfg.TEST.WEIGHT = args.weight
        cfg.freeze()
    
    # 使用配置中的OUTPUT_DIR
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("transreid.tta", output_dir)
    
    logger.info("=" * 70)
    logger.info("Test-Time Augmentation (TTA) 测试脚本")
    logger.info("=" * 70)
    
    logger.info(f"配置文件: {args.config_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"日志文件: {os.path.join(output_dir, 'tta_test_log.txt')}")
    
    # 创建数据加载器
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    logger.info(f"数据集: query={num_query}, gallery={len(val_loader.dataset) - num_query}")
    logger.info(f"权重文件: {cfg.TEST.WEIGHT}")
    
    # 加载模型
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    # 直接传路径，让 load_param 内部处理
    model.load_param(cfg.TEST.WEIGHT)
    logger.info(f"模型加载完成: {cfg.TEST.WEIGHT}")
    
    # 确定设备并移动模型到GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 确定要测试的策略
    if args.strategy == "all":
        strategies_to_test = list(TTA_STRATEGIES.keys())
    else:
        strategies_to_test = [args.strategy]
    
    # 打印所有策略配置
    logger.info("\n【TTA可用策略】")
    logger.info("=" * 70)
    for key, strategy in TTA_STRATEGIES.items():
        logger.info(f"  {key}: {strategy['name']}")
        logger.info(f"    描述: {strategy['description']}")
    logger.info("=" * 70)
    
    # 存储所有结果
    all_results = {}
    
    # 测试不使用TTA的结果（作为基准）
    evaluator_baseline = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=args.reranking)
    evaluator_baseline.reset()
    model.eval()
    
    logger.info("\n" + "=" * 70)
    logger.info("【基准测试】不使用TTA")
    logger.info("=" * 70)
    logger.info("开始推理 - 基准(无TTA)")
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            if isinstance(feat, tuple):
                feat = feat[0]
            evaluator_baseline.update((feat, pid, camid))
        if (n_iter + 1) % 20 == 0:
            logger.info(f"  处理批次 {n_iter + 1}/{len(val_loader)}")
    
    cmc_baseline, mAP_baseline, distmat, pids, camids, qf, gf = evaluator_baseline.compute()
    logger.info("=" * 60)
    logger.info("推理模式: 基准(无TTA)")
    logger.info("=" * 60)
    logger.info("mAP: {:.4f}%".format(mAP_baseline * 100))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}: {:.4f}%".format(r, cmc_baseline[r - 1] * 100))
    logger.info("=" * 60)
    all_results['基准(无TTA)'] = {'cmc': cmc_baseline, 'mAP': mAP_baseline}
    
    # 测试各个TTA策略
    for strategy_key in strategies_to_test:
        logger.info("\n" + "=" * 70)
        logger.info(f"【TTA测试】{TTA_STRATEGIES[strategy_key]['name']}")
        logger.info("=" * 70)
        
        cmc, mAP, name = do_inference_with_tta(
            cfg, model, val_loader, num_query, 
            use_reranking=args.reranking, strategy_key=strategy_key
        )
        all_results[name] = {'cmc': cmc, 'mAP': mAP}
    
    # 对比结果
    logger.info("\n" + "=" * 70)
    logger.info("【所有结果对比】")
    logger.info("=" * 70)
    logger.info(f"{'策略':<20} {'mAP':<12} {'Rank-1':<12} {'Rank-5':<12} {'Rank-10':<12}")
    logger.info("-" * 70)
    
    best_mAP = 0
    best_rank1 = 0
    best_strategy = ""
    
    for name, result in all_results.items():
        cmc = result['cmc']
        mAP = result['mAP']
        logger.info(f"{name:<20} {mAP*100:.4f}%     {cmc[0]*100:.4f}%     {cmc[4]*100:.4f}%     {cmc[9]*100:.4f}%")
        
        # 找到最佳策略
        score = mAP * 100 + cmc[0] * 100  # 综合分数
        if score > best_mAP * 100 + best_rank1 * 100:
            best_mAP = mAP
            best_rank1 = cmc[0]
            best_strategy = name
    
    logger.info("=" * 70)
    
    # 与基准对比
    logger.info("\n【与基准对比】")
    logger.info("=" * 70)
    logger.info(f"{'策略':<20} {'mAP变化':<15} {'Rank-1变化':<15}")
    logger.info("-" * 50)
    baseline_mAP = all_results['基准(无TTA)']['mAP']
    baseline_rank1 = all_results['基准(无TTA)']['cmc'][0]
    
    for name, result in all_results.items():
        if name == '基准(无TTA)':
            continue
        cmc = result['cmc']
        mAP = result['mAP']
        mAP_diff = (mAP - baseline_mAP) * 100
        rank1_diff = (cmc[0] - baseline_rank1) * 100
        logger.info(f"{name:<20} {mAP_diff:+.4f}%         {rank1_diff:+.4f}%")
    logger.info("=" * 70)
    
    # 目标对比
    target_mAP = 91.5
    target_rank1 = 94.0
    
    logger.info(f"\n【最佳策略: {best_strategy}】")
    logger.info(f"mAP: {best_mAP*100:.4f}%, Rank-1: {best_rank1*100:.4f}%")
    logger.info(f"\n【与目标对比】")
    logger.info(f"目标 mAP: {target_mAP}%, 最佳结果: {best_mAP*100:.2f}%, 差距: {(best_mAP*100 - target_mAP):+.2f}%")
    logger.info(f"目标 Rank-1: {target_rank1}%, 最佳结果: {best_rank1*100:.2f}%, 差距: {(best_rank1*100 - target_rank1):+.2f}%")
    
    if best_mAP * 100 >= target_mAP and best_rank1 * 100 >= target_rank1:
        logger.info("\n🎉 恭喜！已达成双目标！")
    elif best_mAP * 100 >= target_mAP:
        logger.info("\n✅ mAP目标已达成！还需提升 Rank-1")
    elif best_rank1 * 100 >= target_rank1:
        logger.info("\n✅ Rank-1目标已达成！还需提升 mAP")
    else:
        logger.info("\n❌ 尚未达成目标，继续努力！")
    
    logger.info(f"\n日志文件: {os.path.join(output_dir, 'tta_test_log.txt')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
