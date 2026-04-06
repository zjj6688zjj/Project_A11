#!/usr/bin/env python3
"""
简单特征级融合脚本 v2
使用统一的融合配置，禁用SIE相机嵌入
"""

import os
import sys
import torch
import numpy as np
import yaml
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.defaults import _C as default_cfg
from datasets.make_dataloader import make_dataloader
from model.make_model import make_model
from utils.metrics import eval_func

def load_config(config_path):
    """加载并合并配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(config_path, 'r', encoding='gbk') as f:
            cfg_dict = yaml.safe_load(f)
    
    cfg = default_cfg.clone()
    cfg.defrost()
    
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
    cfg.freeze()
    
    return cfg

def load_model(model_path, config):
    """加载模型"""
    print(f"  加载权重: {model_path}")
    
    # 创建模型 - camera_num从配置中读取，默认5
    camera_num = getattr(config.MODEL, 'CAMERA_NUM', 5)
    model = make_model(config, num_class=3353, camera_num=camera_num, view_num=1)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 过滤不匹配的键
    model_state_dict = model.state_dict()
    
    matched_keys = []
    unmatched_keys = []
    
    for key in state_dict.keys():
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            matched_keys.append(key)
        else:
            unmatched_keys.append(key)
    
    print(f"    匹配的键: {len(matched_keys)}/{len(state_dict)}")
    if unmatched_keys:
        print(f"    不匹配的键: {len(unmatched_keys)} (将被忽略)")
        # 显示前几个不匹配的键
        for key in unmatched_keys[:3]:
            print(f"      - {key}")
    
    # 只加载匹配的权重
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in matched_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    return model

def extract_features(model, dataloader, device='cuda'):
    """提取特征"""
    model.to(device)
    model.eval()
    
    all_features = []
    all_pids = []
    all_camids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 处理不同格式的数据
            if len(batch) == 6:
                imgs, pids, camids, camids_batch, viewids, img_paths = batch
            elif len(batch) == 4:
                imgs, pids, camids, viewids = batch
            else:
                imgs = batch[0]
                pids = batch[1] if len(batch) > 1 else None
                camids = batch[2] if len(batch) > 2 else None
            
            imgs = imgs.to(device)
            
            # 前向传播
            try:
                outputs = model(imgs)
            except Exception as e:
                # 如果失败，尝试带cam_label
                batch_size = imgs.shape[0]
                camera_ids = torch.zeros(batch_size, dtype=torch.long).to(device)
                try:
                    outputs = model(imgs, cam_label=camera_ids)
                except:
                    outputs = model.base(imgs) if hasattr(model, 'base') else model(imgs)
            
            # 获取特征
            if isinstance(outputs, tuple):
                feat = outputs[0]
            else:
                feat = outputs
            
            all_features.append(feat.cpu())
            
            if pids is not None:
                all_pids.extend(pids.cpu().numpy() if isinstance(pids, torch.Tensor) else pids)
            if camids is not None:
                all_camids.extend(camids.cpu().numpy() if isinstance(camids, torch.Tensor) else camids)
            
            if (batch_idx + 1) % 4 == 0:
                print(f"    批次 {batch_idx+1}/{len(dataloader)}")
    
    features = torch.cat(all_features, dim=0)
    
    if not all_pids:
        all_pids = list(range(features.shape[0]))
    if not all_camids:
        all_camids = [0] * features.shape[0]
    
    return features, np.array(all_pids), np.array(all_camids)

def main():
    print("=" * 70)
    print("简单特征级融合 v2")
    print("使用统一的融合配置（禁用SIE相机嵌入）")
    print("=" * 70)
    
    start_time = time.time()
    
    # 融合专用配置
    FUSION_CONFIG = 'configs/ensemble_fusion.yml'
    
    # 融合模型配置 (只使用 divide_length=6 的兼容模型)
    # rtx4090_optimized (divide_length=4) 与其他模型不兼容，已排除
    model_configs = [
        ('logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth',      # 最佳Rank-1, divide_length=6
         'configs/BallShow/rtx4090_final.yml'),
        ('logs/BallShow_rtx4090_from_final212model/transformer_checkpoint_283.pth',  # 最佳mAP, divide_length=6
         'configs/BallShow/rtx4090_from_final212model.yml'),
        # ('logs/BallShow_rtx4090_optimized/transformer_checkpoint_171.pth',  # divide_length=4, 不兼容
        #  'configs/BallShow/rtx4090_optimized.yml')
    ]
    
    print(f"\n将融合 {len(model_configs)} 个模型")
    
    # 1. 加载融合专用配置创建数据加载器
    print("\n1. 加载配置和创建数据加载器")
    print(f"  融合配置: {FUSION_CONFIG}")
    
    fusion_cfg = load_config(FUSION_CONFIG)
    fusion_cfg.defrost()
    # 确保CAMERA_NUM正确
    fusion_cfg.MODEL.CAMERA_NUM = 5
    fusion_cfg.freeze()
    
    print("创建数据加载器...")
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(fusion_cfg)
    
    print(f"数据加载器创建完成:")
    print(f"  query数量: {num_query}")
    print(f"  总样本数: {len(val_loader.dataset)}")
    print(f"  相机数量: {camera_num}")
    
    # 2. 加载所有模型并提取特征
    print("\n2. 加载模型并提取特征")
    all_features = []
    all_pids = None
    all_camids = None
    
    for i, (model_path, config_path) in enumerate(model_configs):
        print(f"\n处理模型 {i+1}/{len(model_configs)}")
        print(f"  权重: {model_path}")
        print(f"  配置: {config_path}")
        
        # 加载原始配置（保持与训练时一致）
        model_cfg = load_config(config_path)
        
        # 只修改必要的参数，不改变JPM配置
        model_cfg.defrost()
        model_cfg.MODEL.CAMERA_NUM = 5
        # 不要强制修改SIE_CAMERA，保持与训练时一致
        model_cfg.freeze()
        
        # 加载模型
        model = load_model(model_path, model_cfg)
        
        # 提取特征
        print("  提取特征...")
        features, pids, camids = extract_features(model, val_loader)
        
        all_features.append(features)
        
        if all_pids is None:
            all_pids = pids
            all_camids = camids
        
        print(f"  特征形状: {features.shape}")
    
    # 3. 特征融合
    print("\n3. 特征融合")
    print(f"对 {len(all_features)} 个模型的特征进行平均融合")
    
    # 检查形状一致性
    shapes = [f.shape for f in all_features]
    if len(set(shapes)) > 1:
        print(f"警告: 特征形状不一致: {shapes}")
        min_dim = min(f.shape[1] for f in all_features)
        all_features = [f[:, :min_dim] for f in all_features]
        print(f"统一到最小维度: {min_dim}")
    
    # 平均融合
    fused_features = torch.stack(all_features).mean(dim=0)
    print(f"融合后特征形状: {fused_features.shape}")
    
    # 4. 评估
    print("\n4. 评估融合特征")
    
    qf = fused_features[:num_query]
    gf = fused_features[num_query:]
    q_pids = all_pids[:num_query]
    g_pids = all_pids[num_query:]
    q_camids = all_camids[:num_query]
    g_camids = all_camids[num_query:]
    
    # 计算余弦相似度
    qf_norm = torch.nn.functional.normalize(qf, p=2, dim=1)
    gf_norm = torch.nn.functional.normalize(gf, p=2, dim=1)
    distmat = 1 - torch.mm(qf_norm, gf_norm.t())
    distmat = distmat.numpy()
    
    # 计算指标
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    
    # 5. 输出结果
    print("\n" + "=" * 70)
    print("融合结果:")
    print("=" * 70)
    print(f"mAP: {mAP * 100:.2f}%")
    print(f"Rank-1: {cmc[0] * 100:.2f}%")
    print(f"Rank-5: {cmc[4] * 100:.2f}%")
    print(f"Rank-10: {cmc[9] * 100:.2f}%")
    
    # 与单模型比较
    print("\n" + "-" * 70)
    print("与单模型性能比较:")
    print("-" * 70)
    print(f"{'模型':<40} {'mAP':>10} {'Rank-1':>10}")
    print("-" * 70)
    print(f"{'rtx4090_final (epoch 153)':<40} {'91.3%':>10} {'94.2%':>10}")
    print(f"{'rtx4090_from_final212model (epoch 218)':<40} {'91.4%':>10} {'93.9%':>10}")
    print(f"{'rtx4090_optimized (epoch 171)':<40} {'91.3%':>10} {'93.9%':>10}")
    print("-" * 70)
    print(f"{'融合模型':<40} {mAP*100:>9.2f}% {cmc[0]*100:>9.2f}%")
    print("-" * 70)
    
    # 6. 保存结果
    print("\n6. 保存结果")
    
    output_dir = 'logs/ensemble_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存融合特征
    fused_features_path = os.path.join(output_dir, 'fused_features.npy')
    np.save(fused_features_path, fused_features.numpy())
    print(f"融合特征: {fused_features_path}")
    
    # 保存相似度矩阵
    distmat_path = os.path.join(output_dir, 'fused_distmat.npy')
    np.save(distmat_path, distmat)
    print(f"相似度矩阵: {distmat_path}")
    
    # 保存评估结果
    results_path = os.path.join(output_dir, 'fusion_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("特征融合结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"融合时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"融合模型数量: {len(model_configs)}\n\n")
        
        f.write("使用的模型:\n")
        for i, (model_path, config_path) in enumerate(model_configs):
            f.write(f"  {i+1}. {model_path}\n")
        
        f.write(f"\n评估结果:\n")
        f.write(f"  mAP: {mAP * 100:.2f}%\n")
        f.write(f"  Rank-1: {cmc[0] * 100:.2f}%\n")
        f.write(f"  Rank-5: {cmc[4] * 100:.2f}%\n")
        f.write(f"  Rank-10: {cmc[9] * 100:.2f}%\n")
        
        f.write(f"\nRank指标 (1-20):\n")
        for i in range(min(20, len(cmc))):
            f.write(f"  Rank-{i+1}: {cmc[i] * 100:.2f}%\n")
    
    print(f"评估结果: {results_path}")
    
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.1f} 秒")
    print("\n[完成] 特征级融合完成！")

if __name__ == '__main__':
    main()
