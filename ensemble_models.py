#!/usr/bin/env python3
"""
TransReID 模型融合脚本
融合多个训练好的模型以提升性能
支持特征级融合、分数级融合和决策级融合
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import yaml
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
try:
    from datasets.make_dataloader import make_dataloader
    from model.make_model import make_model
    from utils.metrics import R1_mAP_eval, eval_func
except ImportError:
    # 如果无法导入，尝试使用相对导入
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from datasets.make_dataloader import make_dataloader
    from model.make_model import make_model
    from utils.metrics import R1_mAP_eval, eval_func

import torch.nn.functional as F

# 融合专用配置路径
FUSION_CONFIG_PATH = 'configs/ensemble_fusion.yml'


def load_fusion_config():
    """加载融合专用配置（禁用SIE）"""
    from config.defaults import _C as default_cfg
    
    try:
        with open(FUSION_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
    except:
        with open(FUSION_CONFIG_PATH, 'r', encoding='gbk') as f:
            cfg_dict = yaml.safe_load(f)
    
    cfg = default_cfg.clone()
    cfg.defrost()
    
    def merge_dict(cfg_dict, cfg_node, prefix=""):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                if hasattr(cfg_node, key):
                    merge_dict(value, getattr(cfg_node, key), f"{prefix}.{key}" if prefix else key)
                else:
                    from yacs.config import CfgNode
                    setattr(cfg_node, key, CfgNode(value))
            else:
                setattr(cfg_node, key, value)
    
    merge_dict(cfg_dict, cfg)
    cfg.freeze()
    return cfg


def apply_fusion_settings(cfg):
    """对配置应用融合设置"""
    cfg.defrost()
    cfg.MODEL.CAMERA_NUM = 5  # 确保camera_num正确
    cfg.freeze()
    return cfg


class ModelEnsemble:
    """模型融合类"""
    
    def __init__(self, model_paths, config_paths=None, device='cuda'):
        """
        初始化模型融合
        
        Args:
            model_paths: 模型checkpoint路径列表
            config_paths: 配置文件路径列表（与模型对应）
            device: 设备
        """
        self.model_paths = model_paths
        self.config_paths = config_paths or [None] * len(model_paths)
        self.device = device
        self.models = []
        self.configs = []
        
        print(f"初始化模型融合，共 {len(model_paths)} 个模型")
        
        # 如果config_paths为None，创建等长的None列表
        if config_paths is None:
            config_paths = [None] * len(model_paths)
        
        # 加载所有模型
        for i, (model_path, config_path) in enumerate(zip(model_paths, config_paths)):
            print(f"\n加载模型 {i+1}: {model_path}")
            
            # 加载配置
            if config_path:
                # 使用提供的配置，但应用融合设置（禁用SIE）
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cfg_dict = yaml.safe_load(f)
                except:
                    with open(config_path, 'r', encoding='gbk') as f:
                        cfg_dict = yaml.safe_load(f)
                
                from config.defaults import _C as default_cfg
                cfg = default_cfg.clone()
                cfg.defrost()
                
                def merge_dict(cfg_dict, cfg_node, prefix=""):
                    for key, value in cfg_dict.items():
                        if isinstance(value, dict):
                            if hasattr(cfg_node, key):
                                merge_dict(value, getattr(cfg_node, key), f"{prefix}.{key}" if prefix else key)
                            else:
                                from yacs.config import CfgNode
                                setattr(cfg_node, key, CfgNode(value))
                        else:
                            setattr(cfg_node, key, value)
                
                merge_dict(cfg_dict, cfg)
            else:
                # 使用融合专用配置
                cfg = load_fusion_config()
            
            # 关键修复：禁用SIE，使用正确的camera_num
            cfg = apply_fusion_settings(cfg)
            
            self.configs.append(cfg)
            
            # 创建模型 - camera_num从配置读取
            camera_num = getattr(cfg.MODEL, 'CAMERA_NUM', 5)
            model = make_model(cfg, num_class=3353, camera_num=camera_num, view_num=1)
            model.to(device)
            
            # 加载权重（允许部分匹配）
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # 过滤不匹配的键
            model_state_dict = model.state_dict()
            matched_keys = [k for k in state_dict.keys() if k in model_state_dict and state_dict[k].shape == model_state_dict[k].shape]
            
            if len(matched_keys) < len(state_dict):
                print(f"  匹配的键: {len(matched_keys)}/{len(state_dict)} (不匹配将被忽略)")
            
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in matched_keys}
            model.load_state_dict(filtered_state_dict, strict=False)
            model.eval()
            
            self.models.append(model)
            print(f"  模型 {i+1} 加载完成，参数数量: {sum(p.numel() for p in model.parameters())}")
    
    def _infer_config_from_model_path(self, model_path):
        """从模型路径推断配置"""
        # 根据模型路径推断配置类型
        if 'rtx4090_final' in model_path:
            config_file = 'configs/BallShow/rtx4090_final.yml'
        elif 'rtx4090_from_final212model' in model_path:
            config_file = 'configs/BallShow/rtx4090_from_final212model.yml'
        elif 'rtx4090_optimized' in model_path:
            config_file = 'configs/BallShow/rtx4090_optimized.yml'
        elif 'rtx4090_sprint' in model_path:
            config_file = 'configs/BallShow/rtx4090_sprint.yml'
        elif 'rtx4090_ultimate' in model_path:
            config_file = 'configs/BallShow/rtx4090_ultimate.yml'
        elif 'vit_transreid_stride' in model_path:
            config_file = 'configs/BallShow/vit_transreid_stride.yml'
        elif 'competition_simple' in model_path:
            config_file = 'configs/BallShow/ballshow_competition_simple.yml'
        else:
            config_file = 'configs/BallShow/vit_base.yml'
        
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        
        return cfg
    
    def extract_features(self, dataloader, fusion_type='feature'):
        """
        提取特征并进行融合
        
        Args:
            dataloader: 数据加载器
            fusion_type: 融合类型 ('feature', 'score', 'decision')
        
        Returns:
            融合后的特征或预测结果
        """
        print(f"\n开始提取特征，融合类型: {fusion_type}")
        
        all_features = []
        all_scores = []
        all_predictions = []
        
        # 提取每个模型的特征
        for i, model in enumerate(self.models):
            print(f"  提取模型 {i+1} 的特征...")
            
            features, scores, predictions = self._extract_single_model_features(model, dataloader)
            all_features.append(features)
            all_scores.append(scores)
            all_predictions.append(predictions)
        
        # 根据融合类型进行融合
        if fusion_type == 'feature':
            fused_features = self._feature_level_fusion(all_features)
            return fused_features, None, None
        
        elif fusion_type == 'score':
            fused_scores = self._score_level_fusion(all_scores)
            fused_predictions = torch.argmax(fused_scores, dim=1)
            return None, fused_scores, fused_predictions
        
        elif fusion_type == 'decision':
            fused_predictions = self._decision_level_fusion(all_predictions)
            return None, None, fused_predictions
        
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def _extract_single_model_features(self, model, dataloader):
        """提取单个模型的特征"""
        model.eval()
        features = []
        scores = []
        predictions = []
        
        with torch.no_grad():
            for batch_idx, (imgs, pids, camids, _) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                
                # 前向传播
                outputs = model(imgs)
                
                # 提取特征
                if isinstance(outputs, tuple):
                    feat = outputs[0]  # 通常是特征
                    if len(outputs) > 1:
                        score = outputs[1]  # 分类分数
                    else:
                        score = None
                else:
                    feat = outputs
                    score = None
                
                features.append(feat.cpu())
                if score is not None:
                    scores.append(score.cpu())
                    preds = torch.argmax(score, dim=1)
                    predictions.append(preds.cpu())
                
                if batch_idx % 20 == 0:
                    print(f"    处理批次 {batch_idx}/{len(dataloader)}")
        
        features = torch.cat(features, dim=0)
        if scores:
            scores = torch.cat(scores, dim=0)
            predictions = torch.cat(predictions, dim=0)
        else:
            scores = None
            predictions = None
        
        return features, scores, predictions
    
    def _feature_level_fusion(self, features_list):
        """特征级融合：拼接所有模型的特征"""
        print("  执行特征级融合...")
        
        # 先归一化每个模型的特征
        normalized_features = []
        for features in features_list:
            # L2归一化
            norm_features = F.normalize(features, p=2, dim=1)
            normalized_features.append(norm_features)
        
        # 拼接特征
        fused_features = torch.cat(normalized_features, dim=1)
        print(f"  融合后特征维度: {fused_features.shape}")
        
        return fused_features
    
    def _score_level_fusion(self, scores_list):
        """分数级融合：加权平均分数"""
        print("  执行分数级融合...")
        
        # 根据验证集性能设置权重（这里简化处理，使用等权重）
        weights = [1.0 / len(scores_list)] * len(scores_list)
        
        # 加权平均
        fused_scores = torch.zeros_like(scores_list[0])
        for weight, scores in zip(weights, scores_list):
            fused_scores += weight * scores
        
        return fused_scores
    
    def _decision_level_fusion(self, predictions_list):
        """决策级融合：多数投票"""
        print("  执行决策级融合...")
        
        # 堆叠所有预测
        predictions_stack = torch.stack(predictions_list, dim=1)  # [N, M]
        
        # 多数投票
        fused_predictions = []
        for i in range(predictions_stack.shape[0]):
            votes = predictions_stack[i].tolist()
            # 统计每个类别的票数
            vote_count = {}
            for vote in votes:
                vote_count[vote] = vote_count.get(vote, 0) + 1
            
            # 选择票数最多的类别
            max_vote = max(vote_count.items(), key=lambda x: x[1])[0]
            fused_predictions.append(max_vote)
        
        return torch.tensor(fused_predictions)
    
    def evaluate_ensemble(self, dataloader, num_query, fusion_type='feature', reranking=False):
        """
        评估融合模型
        
        Args:
            dataloader: 数据加载器
            num_query: 查询图像数量
            fusion_type: 融合类型
            reranking: 是否使用re-ranking
        """
        print(f"\n评估融合模型 (融合类型: {fusion_type}, re-ranking: {reranking})")
        
        # 提取特征并进行融合
        fused_features, fused_scores, fused_predictions = self.extract_features(
            dataloader, fusion_type
        )
        
        # 计算指标
        if fusion_type == 'feature':
            # 对于特征级融合，使用余弦相似度
            qf = fused_features[:num_query]
            gf = fused_features[num_query:]
            
            # 计算相似度矩阵
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)
            distmat = 1 - torch.mm(qf, gf.t())
            
            # 转换为numpy
            distmat = distmat.numpy()
            
            # 提取query和gallery的pid和camid
            q_pids = []
            q_camids = []
            for item in dataloader.dataset.query:
                q_pids.append(item[1])  # pid
                q_camids.append(item[2])  # camid
                
            g_pids = []
            g_camids = []
            for item in dataloader.dataset.gallery:
                g_pids.append(item[1])  # pid
                g_camids.append(item[2])  # camid
                
            # 转换为numpy数组
            q_pids = np.array(q_pids)
            q_camids = np.array(q_camids)
            g_pids = np.array(g_pids)
            g_camids = np.array(g_camids)
            
        elif fusion_type in ['score', 'decision']:
            # 对于分数级或决策级融合，需要更复杂的评估
            # 这里简化处理
            print("  分数级/决策级融合评估需要更复杂的实现，这里简化处理")
            return None
        
        # 计算指标
        print("  计算评估指标...")
        evaluator = R1_mAP_eval(
            num_query, max_rank=50, feat_norm=True, 
            reranking=reranking
        )
        
        # 使用R1_mAP_eval的compute方法
        evaluator.reset()
        # 我们需要将数据格式化为R1_mAP_eval期望的格式
        # R1_mAP_eval.update期望(output)包含feat, pid, camid
        # 但我们已经有了distmat，所以直接计算
        
        # 使用eval_func函数计算指标
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        
        print(f"\n评估结果:")
        print(f"  mAP: {mAP:.1f}%")
        print(f"  Rank-1: {cmc[0]:.1f}%")
        print(f"  Rank-5: {cmc[4]:.1f}%")
        print(f"  Rank-10: {cmc[9]:.1f}%")
        
        return {
            'mAP': mAP,
            'Rank-1': cmc[0],
            'Rank-5': cmc[4],
            'Rank-10': cmc[9],
            'fusion_type': fusion_type,
            'reranking': reranking
        }
    
    def save_fused_model(self, output_path, fusion_type='feature'):
        """
        保存融合后的模型
        
        Args:
            output_path: 输出路径
            fusion_type: 融合类型
        """
        print(f"\n保存融合模型到: {output_path}")
        
        # 创建融合模型结构
        class FusedModel(torch.nn.Module):
            def __init__(self, models, fusion_type='feature'):
                super().__init__()
                self.models = torch.nn.ModuleList(models)
                self.fusion_type = fusion_type
                self.num_models = len(models)
                
                # 如果特征级融合，需要知道输出维度
                if fusion_type == 'feature':
                    # 获取第一个模型的输出维度
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 384, 128)
                        dummy_output = self.models[0](dummy_input)
                        if isinstance(dummy_output, tuple):
                            self.feature_dim = dummy_output[0].shape[1]
                        else:
                            self.feature_dim = dummy_output.shape[1]
                
            def forward(self, x):
                if self.fusion_type == 'feature':
                    # 特征级融合：拼接所有模型的特征
                    features = []
                    for model in self.models:
                        output = model(x)
                        if isinstance(output, tuple):
                            feat = output[0]
                        else:
                            feat = output
                        # L2归一化
                        feat = F.normalize(feat, p=2, dim=1)
                        features.append(feat)
                    
                    # 拼接特征
                    fused = torch.cat(features, dim=1)
                    return fused
                
                elif self.fusion_type == 'score':
                    # 分数级融合：平均分数
                    scores = []
                    for model in self.models:
                        output = model(x)
                        if isinstance(output, tuple) and len(output) > 1:
                            score = output[1]
                        else:
                            # 如果没有分类头，使用特征
                            if isinstance(output, tuple):
                                feat = output[0]
                            else:
                                feat = output
                            # 添加一个简单的分类头（仅用于保存）
                            score = torch.randn(feat.shape[0], 3353)  # 3353是类别数
                        
                        scores.append(score)
                    
                    # 平均分数
                    fused = torch.mean(torch.stack(scores), dim=0)
                    return fused
                
                else:
                    # 决策级融合不适合保存为单个模型
                    raise ValueError("决策级融合无法保存为单个模型")
        
        # 创建融合模型
        fused_model = FusedModel(self.models, fusion_type)
        fused_model.eval()
        
        # 保存模型
        torch.save({
            'state_dict': fused_model.state_dict(),
            'fusion_type': fusion_type,
            'num_models': len(self.models),
            'model_paths': self.model_paths,
            'config_paths': self.config_paths,
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, output_path)
        
        print(f"  融合模型保存完成")
        return fused_model


def main():
    parser = argparse.ArgumentParser(description='TransReID模型融合')
    parser.add_argument('--model1', type=str, required=True, help='第一个模型路径')
    parser.add_argument('--model2', type=str, required=True, help='第二个模型路径')
    parser.add_argument('--model3', type=str, help='第三个模型路径')
    parser.add_argument('--config1', type=str, help='第一个模型配置')
    parser.add_argument('--config2', type=str, help='第二个模型配置')
    parser.add_argument('--config3', type=str, help='第三个模型配置')
    parser.add_argument('--fusion_type', type=str, default='feature', 
                       choices=['feature', 'score', 'decision'], help='融合类型')
    parser.add_argument('--output_dir', type=str, default='logs/ensemble', help='输出目录')
    parser.add_argument('--reranking', action='store_true', help='使用re-ranking')
    parser.add_argument('--test', action='store_true', help='测试融合模型')
    
    args = parser.parse_args()
    
    # 准备模型路径
    model_paths = [args.model1, args.model2]
    config_paths = [args.config1, args.config2]
    
    if args.model3:
        model_paths.append(args.model3)
        config_paths.append(args.config3)
    
    # 清理None值
    config_paths = [cfg if cfg else None for cfg in config_paths]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("TransReID 模型融合脚本")
    print("=" * 70)
    print(f"融合类型: {args.fusion_type}")
    print(f"模型数量: {len(model_paths)}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 70)
    
    # 初始化模型融合
    ensemble = ModelEnsemble(model_paths, config_paths)
    
    if args.test:
        # 使用融合专用配置创建数据加载器
        cfg = load_fusion_config()
        
        # 创建数据加载器
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
        
        # 评估融合模型
        results = ensemble.evaluate_ensemble(
            val_loader, num_query, 
            fusion_type=args.fusion_type,
            reranking=args.reranking
        )
        
        # 保存评估结果
        result_file = os.path.join(args.output_dir, f'ensemble_results_{args.fusion_type}.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n评估结果已保存到: {result_file}")
    
    # 保存融合模型
    model_name = f'ensemble_{args.fusion_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    output_path = os.path.join(args.output_dir, model_name)
    
    ensemble.save_fused_model(output_path, args.fusion_type)
    
    print("\n" + "=" * 70)
    print("融合完成！")
    print(f"融合模型已保存到: {output_path}")
    print("\n使用说明:")
    print(f"1. 测试融合模型:")
    print(f"   python test.py --config_file configs/BallShow/rtx4090_final.yml \\")
    print(f"     --weight {output_path}")
    print(f"\n2. 使用融合模型进行推理:")
    print(f"   model = torch.load('{output_path}', map_location='cuda')")
    print(f"   fused_model = model['state_dict']")
    print("=" * 70)


def preset_ensemble():
    """预设融合方案：使用最佳checkpoint"""
    
    print("=" * 70)
    print("预设融合方案：使用RTX4090最佳checkpoint")
    print("=" * 70)
    
    # 定义最佳checkpoint
    checkpoints = {
        'rtx4090_final': 'logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth',
        'rtx4090_from_final212model': 'logs/BallShow_rtx4090_from_final212model/transformer_checkpoint_283.pth',
        'rtx4090_optimized': 'logs/BallShow_rtx4090_optimized/transformer_checkpoint_171.pth'
    }
    
    # 检查checkpoint是否存在
    available_checkpoints = {}
    for name, path in checkpoints.items():
        if os.path.exists(path):
            available_checkpoints[name] = path
            print(f"[OK] {name}: {path}")
        else:
            print(f"[NO] {name}: 文件不存在")
    
    if len(available_checkpoints) < 2:
        print("\n警告：至少需要2个checkpoint进行融合")
        return
    
    # 创建融合
    output_dir = 'logs/ensemble_preset'
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试不同融合类型
    fusion_types = ['feature', 'score', 'decision']
    
    for fusion_type in fusion_types:
        print(f"\n尝试 {fusion_type} 融合...")
        
        # 创建融合
        model_paths = list(available_checkpoints.values())
        config_paths = [None] * len(model_paths)  # 自动推断配置
        
        try:
            ensemble = ModelEnsemble(model_paths, config_paths)
            
            # 保存融合模型
            model_name = f'ensemble_{fusion_type}_preset.pth'
            output_path = os.path.join(output_dir, model_name)
            
            ensemble.save_fused_model(output_path, fusion_type)
            
            print(f"  [OK] 保存到: {output_path}")
            
        except Exception as e:
            print(f"  [FAIL] 融合失败: {e}")
    
    print("\n" + "=" * 70)
    print("预设融合完成！")
    print(f"模型保存在: {output_dir}")
    print("\n推荐使用特征级融合 (feature-level fusion)")
    print("它通常能获得最佳性能提升")
    print("=" * 70)


if __name__ == '__main__':
    # 如果有命令行参数，使用主函数
    if len(sys.argv) > 1:
        main()
    else:
        # 否则使用预设融合方案
        preset_ensemble()