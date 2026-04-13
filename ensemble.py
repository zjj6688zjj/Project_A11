#!/usr/bin/env python3
"""
TransReID 模型融合脚本（统一版）
================================
支持两种维度的融合：
1. 融合层级 (fusion_type): feature / score / decision
2. 融合策略 (fusion_method): average / max / weighted

特性：
- 支持任意数量模型输入（≥2）
- 每个模型特征独立缓存（ensemble_cache/ 目录）
- 缓存文件命名：{模型名}_features.npz
- 部分缓存缺失时仅重新提取缺失模型
- weighted 策略自动等分权重

默认模型（3个）：
  1. rtx4090_final_epoch153        (mAP: 91.3, Rank-1: 94.2)
  2. rtx4090_from_final212model_epoch283 (mAP: 91.4, Rank-1: 93.9)
  3. rtx4090_from_final153model_epoch177 (mAP: 91.3, Rank-1: 94.0)

使用方法:
# 无参数运行 - 使用预设最佳模型
python ensemble.py

# 指定融合层级
python ensemble.py --type feature

# 指定融合策略（默认 feature）
python ensemble.py --method weighted

# 同时指定（尝试所有组合或指定组合）
python ensemble.py --type all --method all

# 指定模型（≥2个）
python ensemble.py --models model1.pth model2.pth model3.pth

# 测试融合效果
python ensemble.py --test
"""

import os
import sys
import torch
import numpy as np
import yaml
import argparse
import time
import json
import hashlib
from datetime import datetime

# 缓存目录
CACHE_DIR = 'ensemble_cache'

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.defaults import _C as default_cfg
from datasets.make_dataloader import make_dataloader
from model.make_model import make_model
from utils.metrics import eval_func, R1_mAP_eval
import torch.nn.functional as F

# 全局日志记录器
class Logger:
    """同时输出到终端和文件的日志记录器（追加模式）"""
    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # 追加模式，不清空文件
    
    def log(self, *args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        timestamped_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(timestamped_msg, **kwargs)
        sys.stdout.flush()
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(timestamped_msg + '\n')
            except:
                pass

logger = Logger()


def ts():
    """返回带时间戳的日志前缀"""
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"


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
    
    def merge_dict_into_cfg(cfg_dict, cfg_node):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                if hasattr(cfg_node, key):
                    merge_dict_into_cfg(value, getattr(cfg_node, key))
                else:
                    from yacs.config import CfgNode
                    setattr(cfg_node, key, CfgNode(value))
            else:
                setattr(cfg_node, key, value)
    
    merge_dict_into_cfg(cfg_dict, cfg)
    cfg.freeze()
    
    return cfg


def load_model(model_path, config):
    """加载模型"""
    logger.log(f"  加载: {os.path.basename(model_path)}")
    
    camera_num = getattr(config.MODEL, 'CAMERA_NUM', 5)
    num_class = 3353
    
    model = make_model(config, num_class=num_class, camera_num=camera_num, view_num=1)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model_state_dict = model.state_dict()
    
    matched_keys = [k for k in state_dict.keys() 
                   if k in model_state_dict and state_dict[k].shape == model_state_dict[k].shape]
    
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in matched_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    return model


def apply_fusion_settings(cfg):
    """应用融合专用设置"""
    cfg.defrost()
    cfg.MODEL.CAMERA_NUM = 5
    cfg.freeze()
    return cfg


def load_fusion_config():
    """加载融合专用配置"""
    config_path = 'configs/ensemble_ver1.yml'
    if not os.path.exists(config_path):
        config_path = 'configs/BallShow/rtx4090_final.yml'
    
    return load_config(config_path)


class EnsembleModels:
    """模型融合类"""
    
    def __init__(self, model_paths, config_paths=None, model_names=None, device='cuda'):
        """
        初始化模型融合
        
        Args:
            model_paths: 模型checkpoint路径列表
            config_paths: 配置文件路径列表
            model_names: 模型名称列表（用于缓存文件名）
            device: 设备
        """
        self.model_paths = model_paths
        self.config_paths = config_paths or [None] * len(model_paths)
        self.model_names = model_names or []  # 如果没传，从路径提取
        self.device = device
        self.models = []
        self.configs = []
        self._cached_features = None  # 内存缓存
        self._cache_dir = CACHE_DIR   # 持久化缓存目录
        
        # 创建缓存目录
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
        
        logger.log(f"初始化模型融合，共 {len(model_paths)} 个模型")
        
        for i, (model_path, config_path) in enumerate(zip(model_paths, self.config_paths)):
            logger.log(f"\n加载模型 {i+1}: {os.path.basename(model_path)}")
            
            # 加载配置
            if config_path:
                cfg = load_config(config_path)
            else:
                cfg = load_fusion_config()
            
            cfg = apply_fusion_settings(cfg)
            self.configs.append(cfg)
            
            # 创建并加载模型
            camera_num = getattr(cfg.MODEL, 'CAMERA_NUM', 5)
            model = make_model(cfg, num_class=3353, camera_num=camera_num, view_num=1)
            model.to(device)
            
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            model_state_dict = model.state_dict()
            matched_keys = [k for k in state_dict.keys() 
                          if k in model_state_dict and state_dict[k].shape == model_state_dict[k].shape]
            
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in matched_keys}
            model.load_state_dict(filtered_state_dict, strict=False)
            model.eval()
            
            self.models.append(model)
            logger.log(f"  模型 {i+1} 加载完成，匹配权重: {len(matched_keys)}/{len(state_dict)}")
    
    def _get_cache_key(self):
        """生成缓存文件名（基于模型路径的hash）"""
        key_str = '_'.join([os.path.basename(p) for p in self.model_paths])
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    

    
    def _get_model_cache_key(self, model_index):
        """生成单个模型的缓存文件名"""
        model_name = self.model_names[model_index] if model_index < len(self.model_names) else f'model_{model_index}'
        return f"{model_name}_features.npz"
    
    def _load_cache(self):
        """尝试从本地加载缓存（部分缺失时只返回已存在的）"""
        existing = {}  # {model_index: data}
        missing = []
        
        for i in range(len(self.model_paths)):
            cache_file = os.path.join(self._cache_dir, self._get_model_cache_key(i))
            if os.path.exists(cache_file):
                data = np.load(cache_file, allow_pickle=True)
                existing[i] = data
            else:
                missing.append(i)
        
        # 如果全部缺失
        if not existing:
            return None
        
        # 部分缺失
        if missing:
            logger.log(f"  [缓存] 发现 {len(existing)} 个已有缓存，缺失 {missing}，将只重新提取缺失的模型")
        
        # 组装数据
        all_features, all_scores, all_predictions = [], [], []
        for i in range(len(self.model_paths)):
            if i in existing:
                data = existing[i]
                all_features.append(torch.from_numpy(data['features']))
                all_scores.append(torch.from_numpy(data['scores']) if data['scores'].size else None)
                all_predictions.append(torch.from_numpy(data['predictions']) if data['predictions'].size else None)
            else:
                all_features.append(None)
                all_scores.append(None)
                all_predictions.append(None)
        
        # 从第一个缓存加载 pids 和 camids
        first_data = existing[min(existing.keys())]
        all_pids = first_data['pids'].tolist()
        all_camids = first_data['camids'].tolist()
        
        logger.log(f"  [缓存] 成功加载 {len(existing)} 个模型的缓存！")
        return (all_features, all_scores, all_predictions, all_pids, all_camids, missing)
    
    def _save_cache(self, features_data):
        """保存特征到本地缓存（每个模型单独保存）"""
        all_features, all_scores, all_predictions, all_pids, all_camids = features_data
        
        for i in range(len(self.model_paths)):
            cache_file = os.path.join(self._cache_dir, self._get_model_cache_key(i))
            save_dict = {
                'features': all_features[i].numpy(),
                'scores': all_scores[i].numpy() if all_scores[i] is not None else np.array([]),
                'predictions': all_predictions[i].numpy() if all_predictions[i] is not None else np.array([]),
                'pids': np.array(all_pids),
                'camids': np.array(all_camids)
            }
            np.savez(cache_file, **save_dict)
            # 获取相对路径（相对于项目根目录）
            rel_path = os.path.relpath(cache_file, start=os.getcwd())
            logger.log(f"  [缓存] 模型 {i+1} 已保存: {rel_path}")
    
    def extract_all_features(self, dataloader, save_per_model=True):
        """
        提取所有模型的特征、分数和预测
        
        Args:
            save_per_model: 是否每个模型完成后立即保存缓存
            
        Returns:
            all_features: 特征列表
            all_scores: 分数列表
            all_predictions: 预测列表
            all_pids: PID列表
            all_camids: CameraID列表
        """
        all_features = []
        all_scores = []
        all_predictions = []
        all_pids = []
        all_camids = []
        
        for i, model in enumerate(self.models):
            logger.log(f"  提取模型 {i+1} 的特征...")
            
            model.eval()
            features = []
            scores = []
            predictions = []
            batch_pids = []
            batch_camids = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if len(batch) == 6:
                        imgs, pids, camids, camids_batch, viewids, img_paths = batch
                    elif len(batch) == 4:
                        imgs, pids, camids, viewids = batch
                    else:
                        imgs = batch[0]
                        pids = batch[1] if len(batch) > 1 else None
                        camids = batch[2] if len(batch) > 2 else None
                    
                    imgs = imgs.to(self.device)
                    
                    try:
                        outputs = model(imgs)
                    except:
                        camera_ids = torch.zeros(imgs.shape[0], dtype=torch.long).to(self.device)
                        outputs = model(imgs, cam_label=camera_ids)
                    
                    # 提取特征
                    if isinstance(outputs, tuple):
                        feat = outputs[0]
                        score = outputs[1] if len(outputs) > 1 else None
                    else:
                        feat = outputs
                        score = None
                    
                    features.append(feat.cpu())
                    
                    if score is not None:
                        scores.append(score.cpu())
                        predictions.append(score.argmax(dim=1).cpu())
                    
                    if pids is not None:
                        batch_pids.extend(pids.cpu().numpy() if isinstance(pids, torch.Tensor) else pids)
                    if camids is not None:
                        batch_camids.extend(camids.cpu().numpy() if isinstance(camids, torch.Tensor) else camids)
                    
                    if (batch_idx + 1) % 5 == 0:
                        logger.log(f"    批次 {batch_idx+1}/{len(dataloader)}")
            
            # 收集当前模型的结果
            current_features = torch.cat(features, dim=0)
            all_features.append(current_features)
            
            if scores:
                current_scores = torch.cat(scores, dim=0)
                all_scores.append(current_scores)
            else:
                all_scores.append(None)
                current_scores = None
                
            if predictions:
                current_predictions = torch.cat(predictions, dim=0)
                all_predictions.append(current_predictions)
            else:
                all_predictions.append(None)
                current_predictions = None
            
            if i == 0:
                all_pids = batch_pids
                all_camids = batch_camids
            
            # 每个模型完成后立即保存缓存
            if save_per_model:
                # 保存当前模型缓存
                cache_file = os.path.join(self._cache_dir, self._get_model_cache_key(i))
                save_dict = {
                    'features': current_features.numpy(),
                    'scores': current_scores.numpy() if current_scores is not None else np.array([]),
                    'predictions': current_predictions.numpy() if current_predictions is not None else np.array([]),
                    'pids': np.array(all_pids),
                    'camids': np.array(all_camids)
                }
                np.savez(cache_file, **save_dict)
                # 获取相对路径（相对于项目根目录）
                rel_path = os.path.relpath(cache_file, start=os.getcwd())
                logger.log(f"  [缓存] 模型 {i+1} 已保存: {rel_path}")
        
        return all_features, all_scores, all_predictions, all_pids, all_camids
    
    def extract_single_models(self, dataloader, model_indices):
        """提取指定模型的特征（用于补充缺失的缓存）"""
        all_features, all_scores, all_predictions = [], [], []
        batch_pids, batch_camids = [], []
        
        for i in model_indices:
            logger.log(f"  提取模型 {i+1} 的特征...")
            features, scores, predictions = [], [], []
            
            model = self.models[i]
            model.eval()
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if len(batch) == 6:
                        imgs, pids, camids, camids_batch, viewids, img_paths = batch
                    elif len(batch) == 4:
                        imgs, pids, camids, viewids = batch
                    else:
                        imgs = batch[0]
                        pids = batch[1] if len(batch) > 1 else None
                        camids = batch[2] if len(batch) > 2 else None
                    
                    imgs = imgs.to(self.device)
                    
                    try:
                        outputs = model(imgs)
                    except:
                        camera_ids = torch.zeros(imgs.shape[0], dtype=torch.long).to(self.device)
                        outputs = model(imgs, cam_label=camera_ids)
                    
                    if isinstance(outputs, tuple):
                        feat = outputs[0]
                        score = outputs[1] if len(outputs) > 1 else None
                    else:
                        feat = outputs
                        score = None
                    
                    features.append(feat.cpu())
                    
                    if score is not None:
                        scores.append(score.cpu())
                        predictions.append(score.argmax(dim=1).cpu())
                    
                    if pids is not None:
                        batch_pids.extend(pids.cpu().numpy() if isinstance(pids, torch.Tensor) else pids)
                    if camids is not None:
                        batch_camids.extend(camids.cpu().numpy() if isinstance(camids, torch.Tensor) else camids)
                    
                    if (batch_idx + 1) % 5 == 0:
                        logger.log(f"    批次 {batch_idx+1}/{len(dataloader)}")
            
            all_features.append(torch.cat(features, dim=0))
            if scores:
                all_scores.append(torch.cat(scores, dim=0))
            if predictions:
                all_predictions.append(torch.cat(predictions, dim=0))
            else:
                all_predictions.append(None)
        
        return all_features, all_scores, all_predictions
    
    def _save_single_cache(self, model_index, features, scores, predictions, pids, camids):
        """保存单个模型的缓存"""
        cache_file = os.path.join(self._cache_dir, self._get_model_cache_key(model_index))
        save_dict = {
            'features': features.numpy(),
            'scores': scores.numpy() if scores is not None else np.array([]),
            'predictions': predictions.numpy() if predictions is not None else np.array([]),
            'pids': np.array(pids),
            'camids': np.array(camids)
        }
        np.savez(cache_file, **save_dict)
        # 获取相对路径（相对于项目根目录）
        rel_path = os.path.relpath(cache_file, start=os.getcwd())
        logger.log(f"  [缓存] 模型 {model_index+1} 已保存: {rel_path}")
    
    def fuse_features(self, features_list, fusion_method='average'):
        """
        特征级融合
        
        Args:
            features_list: 特征列表
            fusion_method: 融合策略
        
        Returns:
            fused_features: 融合后的特征
        """
        logger.log(f"  执行特征级融合 (方法: {fusion_method})")
        
        # 先归一化每个模型的特征
        normalized_features = []
        for feat in features_list:
            norm_feat = F.normalize(feat, p=2, dim=1)
            normalized_features.append(norm_feat)
        
        if fusion_method == 'average':
            fused = torch.stack(normalized_features).mean(dim=0)
        elif fusion_method == 'max':
            fused = torch.stack(normalized_features).max(dim=0)[0]
        elif fusion_method == 'weighted':
            # 动态等分权重，支持任意数量模型
            n_models = len(normalized_features)
            weights = torch.ones(n_models, dtype=torch.float32) / n_models
            fused = torch.zeros_like(normalized_features[0])
            for i, feat in enumerate(normalized_features):
                fused += weights[i].item() * feat
        else:
            fused = torch.stack(normalized_features).mean(dim=0)
        
        fused = F.normalize(fused, p=2, dim=1)
        return fused
    
    def fuse_scores(self, scores_list, fusion_method='average'):
        """
        分数级融合
        
        Args:
            scores_list: 分数列表
            fusion_method: 融合策略
        
        Returns:
            fused_scores: 融合后的分数
        """
        logger.log(f"  执行分数级融合 (方法: {fusion_method})")
        
        if fusion_method == 'average':
            fused = torch.stack(scores_list).mean(dim=0)
        elif fusion_method == 'max':
            fused = torch.stack(scores_list).max(dim=0)[0]
        elif fusion_method == 'weighted':
            # 动态等分权重，支持任意数量模型
            n_models = len(scores_list)
            weights = torch.ones(n_models, dtype=torch.float32) / n_models
            fused = torch.zeros_like(scores_list[0])
            for i, score in enumerate(scores_list):
                fused += weights[i].item() * score
        else:
            fused = torch.stack(scores_list).mean(dim=0)
        
        return fused
    
    def fuse_decisions(self, predictions_list):
        """
        决策级融合（多数投票）
        
        Args:
            predictions_list: 预测列表
        
        Returns:
            fused_predictions: 融合后的预测
        """
        logger.log(f"  执行决策级融合 (多数投票)")
        
        # 堆叠所有预测
        predictions_stack = torch.stack([p for p in predictions_list if p is not None], dim=1)
        
        # 多数投票
        fused_predictions = []
        for i in range(predictions_stack.shape[0]):
            votes = predictions_stack[i].tolist()
            vote_count = {}
            for vote in votes:
                vote_count[vote] = vote_count.get(vote, 0) + 1
            max_vote = max(vote_count.items(), key=lambda x: x[1])[0]
            fused_predictions.append(max_vote)
        
        return torch.tensor(fused_predictions)
    
    def evaluate(self, dataloader, num_query, fusion_type='feature', fusion_method='average', reranking=False):
        """
        评估融合效果
        
        Args:
            dataloader: 数据加载器
            num_query: 查询图像数量
            fusion_type: 融合类型 (feature/score/decision)
            fusion_method: 融合策略 (average/max/weighted)
            reranking: 是否使用re-ranking
        
        Returns:
            results: 评估结果字典
        """
        logger.log(f"\n评估融合模型 (类型: {fusion_type}, 策略: {fusion_method})")
        
        # 提取所有模型的特征（优先使用缓存）
        if self._cached_features is None:
            # 先尝试加载本地缓存
            cached = self._load_cache()
            if cached is not None:
                # 解包缓存数据
                all_features, all_scores, all_predictions, all_pids, all_camids, missing = cached
                
                if missing:  # 有部分缓存，需要补充提取缺失的
                    logger.log(f"  [缓存] 补充提取缺失的模型: {[i+1 for i in missing]}")
                    new_features = self.extract_single_models(dataloader, missing)
                    # 合并
                    for i in missing:
                        idx = missing.index(i)
                        all_features[i] = new_features[0][idx]
                        all_scores[i] = new_features[1][idx]
                        all_predictions[i] = new_features[2][idx]
                    # 保存新提取的缓存
                    for i in missing:
                        self._save_single_cache(i, all_features[i], all_scores[i], all_predictions[i], all_pids, all_camids)
                    self._cached_features = (all_features, all_scores, all_predictions, all_pids, all_camids)
                else:
                    self._cached_features = (all_features, all_scores, all_predictions, all_pids, all_camids)
            else:
                # 本地没有缓存，从头提取
                logger.log("  [缓存] 本地无缓存，开始提取特征（首次运行，请稍候）...")
                # 使用逐模型保存模式
                self._cached_features = self.extract_all_features(dataloader, save_per_model=True)
                logger.log("  [缓存] 所有模型特征提取完成")
        
        all_features, all_scores, all_predictions, all_pids, all_camids = self._cached_features
        
        # 根据融合类型执行融合
        if fusion_type == 'feature':
            fused = self.fuse_features(all_features, fusion_method)
            return self._evaluate_features(fused, all_pids, all_camids, num_query, reranking)
        
        elif fusion_type == 'score':
            if not all_scores:
                logger.log("  警告: 模型未输出分类分数，无法执行分数级融合")
                return None
            
            fused_scores = self.fuse_scores(all_scores, fusion_method)
            # 分数级融合无法直接生成 distmat，返回分数用于评估
            return self._evaluate_scores(fused_scores, all_pids, all_camids, num_query)
        
        elif fusion_type == 'decision':
            if not any(all_predictions):
                logger.log("  警告: 模型未输出预测，无法执行决策级融合")
                return None
            
            fused_preds = self.fuse_decisions(all_predictions)
            return self._evaluate_decisions(fused_preds, all_pids, all_camids, num_query)
        
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def _evaluate_features(self, fused_features, all_pids, all_camids, num_query, reranking=False):
        """评估特征"""
        logger.log("  计算评估指标...")
        
        qf = fused_features[:num_query]
        gf = fused_features[num_query:]
        q_pids = np.array(all_pids[:num_query])
        g_pids = np.array(all_pids[num_query:])
        q_camids = np.array(all_camids[:num_query])
        g_camids = np.array(all_camids[num_query:])
        
        qf = qf.cpu() if isinstance(qf, torch.Tensor) else torch.from_numpy(qf)
        gf = gf.cpu() if isinstance(gf, torch.Tensor) else torch.from_numpy(gf)
        
        qf_norm = F.normalize(qf, p=2, dim=1)
        gf_norm = F.normalize(gf, p=2, dim=1)
        distmat = 1 - torch.mm(qf_norm, gf_norm.t())
        distmat = distmat.numpy()
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        
        return {
            'mAP': mAP * 100,
            'Rank-1': cmc[0] * 100,
            'Rank-5': cmc[4] * 100,
            'Rank-10': cmc[9] * 100,
            'CMC': cmc,
            'distmat': distmat,
            'features': fused_features
        }
    
    def _evaluate_scores(self, fused_scores, all_pids, all_camids, num_query):
        """评估分数（分数级融合）"""
        # 分数级融合只能通过准确率评估，无法生成 distmat
        q_pids = np.array(all_pids[:num_query])
        g_pids = np.array(all_pids[num_query:])
        
        q_scores = fused_scores[:num_query]
        g_scores = fused_scores[num_query:]
        
        # 使用余弦相似度计算 distmat（从分数重建）
        qf = q_scores
        gf = g_scores
        qf_norm = F.normalize(qf, p=2, dim=1)
        gf_norm = F.normalize(gf, p=2, dim=1)
        distmat = 1 - torch.mm(qf_norm, gf_norm.t()).numpy()
        
        q_camids = np.array(all_camids[:num_query])
        g_camids = np.array(all_camids[num_query:])
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        
        return {
            'mAP': mAP * 100,
            'Rank-1': cmc[0] * 100,
            'Rank-5': cmc[4] * 100,
            'Rank-10': cmc[9] * 100,
            'CMC': cmc,
            'distmat': distmat,
            'features': None  # 分数级融合不生成特征向量
        }
    
    def _evaluate_decisions(self, fused_predictions, all_pids, all_camids, num_query):
        """评估决策（决策级融合）"""
        # 决策级融合无法计算 mAP，只能计算准确率
        q_pids = np.array(all_pids[:num_query])
        g_pids = np.array(all_pids[num_query:])
        q_preds = fused_predictions[:num_query]
        
        # 计算 top-1 准确率（近似 Rank-1）
        correct = (q_preds.numpy() == q_pids).sum()
        rank1 = correct / num_query * 100
        
        return {
            'mAP': rank1,  # 决策级融合没有 mAP，用准确率代替
            'Rank-1': rank1,
            'Rank-5': rank1,  # 决策级融合没有 Rank-5
            'Rank-10': rank1,  # 决策级融合没有 Rank-10
            'CMC': None,
            'distmat': None,  # 决策级融合不生成 distmat
            'features': None,  # 决策级融合不生成特征向量
            'note': '决策级融合仅返回预测结果，无法生成 distmat'
        }
    
    def save_ensemble_model(self, output_path, fusion_type='feature', fusion_method='average', fused_features=None):
        """保存融合模型或融合后的特征向量"""
        logger.log(f"保存融合结果到: {output_path}")
        
        # 获取第一个模型的结构
        cfg = self.configs[0]
        camera_num = getattr(cfg.MODEL, 'CAMERA_NUM', 5)
        base_model = make_model(cfg, num_class=3353, camera_num=camera_num, view_num=1)
        
        # 加载第一个模型的权重
        checkpoint = torch.load(self.model_paths[0], map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        base_model.load_state_dict(state_dict, strict=False)
        
        save_dict = {
            'state_dict': base_model.state_dict(),
            'epoch': 0,
            'optimizer': None,
            'train_history': {'mAP': [], 'Rank-1': []},
            'best_mAP': 0,
            'best_Rank-1': 0,
            'is_ensemble': True,
            'ensemble_models': [{'path': p, 'config': c} for p, c in zip(self.model_paths, self.config_paths)],
            'fusion_type': fusion_type,
            'fusion_method': fusion_method,
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # 如果有融合后的特征向量，也保存进去
        if fused_features is not None:
            save_dict['fused_features'] = fused_features.cpu() if isinstance(fused_features, torch.Tensor) else fused_features
            logger.log(f"  已包含融合特征向量: shape={save_dict['fused_features'].shape}")
        
        torch.save(save_dict, output_path)
        file_size = os.path.getsize(output_path) / 1024 / 1024
        logger.log(f"  文件大小: {file_size:.2f} MB")


def main():
    global logger
    
    parser = argparse.ArgumentParser(description='TransReID 模型融合（统一版）')
    parser.add_argument('--models', nargs='+', help='模型文件路径列表')
    parser.add_argument('--configs', nargs='+', help='配置文件路径列表')
    parser.add_argument('--type', type=str, default='all', 
                       choices=['all', 'feature', 'score', 'decision'],
                       help='融合层级 (fusion_type): feature/score/decision')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'average', 'max', 'weighted'],
                       help='融合策略 (fusion_method): average/max/weighted')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--test', action='store_true', help='测试融合效果')
    parser.add_argument('--reranking', action='store_true', help='使用re-ranking')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 决定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    elif args.models:
        output_dir = 'logs/ensemble_custom'
    else:
        output_dir = 'logs/ensemble_preset'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化日志
    log_file = os.path.join(output_dir, 'ensemble_log.txt')
    logger = Logger(log_file)
    
    logger.log("=" * 70)
    logger.log("TransReID 模型融合脚本（统一版）")
    logger.log("=" * 70)
    
    # 默认最佳模型配置
    default_models = [
        {
            'path': 'logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth',
            'config': 'configs/BallShow/rtx4090_final.yml',
            'name': 'rtx4090_final_epoch153',
            'mAP': 91.3,
            'Rank-1': 94.2
        },
        {
            'path': 'logs/BallShow_rtx4090_from_final212model/transformer_checkpoint_283.pth',
            'config': 'configs/BallShow/rtx4090_from_final212model.yml',
            'name': 'rtx4090_from_final212model_epoch283',
            'mAP': 91.4,
            'Rank-1': 93.9
        },
        {
            'path': 'logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth',
            'config': 'configs/BallShow/rtx4090_from_final153model.yml',
            'name': 'rtx4090_from_final153model_epoch177',
            'mAP': 91.3,
            'Rank-1': 94.0
        },
    ]
    
    # 选择模型
    if args.models and len(args.models) >= 2:
        selected_models = []
        for path in args.models:
            if os.path.exists(path):
                config = 'configs/BallShow/rtx4090_final.yml'
                for dm in default_models:
                    if dm['path'] == path:
                        config = dm['config']
                        break
                selected_models.append({
                    'path': path,
                    'config': config,
                    'name': os.path.basename(path)
                })
            else:
                logger.log(f"  警告: 文件不存在 {path}")
        
        if len(selected_models) < 2:
            logger.log("错误: 至少需要2个有效模型")
            return
    else:
        selected_models = default_models
    
    # 检查可用模型
    logger.log(f"\n将融合 {len(selected_models)} 个模型:")
    available_models = []
    for model in selected_models:
        if os.path.exists(model['path']):
            info = f"{model['name']}"
            if model.get('mAP'):
                info += f" (mAP: {model['mAP']}%, R1: {model['Rank-1']}%)"
            logger.log(f"  ✓ {info}")
            available_models.append(model)
        else:
            logger.log(f"  ✗ {model['name']} - 文件不存在")
    
    if len(available_models) < 2:
        logger.log("错误: 可用模型不足2个")
        return
    
    selected_models = available_models
    
    # 准备模型路径、配置路径和模型名
    model_paths = [m['path'] for m in selected_models]
    config_paths = [m['config'] for m in selected_models]
    model_names = [m['name'] for m in selected_models]
    
    # 初始化融合模型
    logger.log("\n初始化融合模型...")
    ensemble = EnsembleModels(model_paths, config_paths, model_names)
    
    # 加载配置和数据
    logger.log("\n加载数据...")
    cfg = load_fusion_config()
    cfg = apply_fusion_settings(cfg)
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    logger.log(f"  Query: {num_query}, Gallery: {len(val_loader.dataset) - num_query}")
    
    # 决定要尝试的组合
    if args.type == 'all':
        fusion_types = ['feature', 'score', 'decision']
    else:
        fusion_types = [args.type]
    
    if args.method == 'all':
        fusion_methods = ['average', 'max', 'weighted']
    else:
        fusion_methods = [args.method]
    
    logger.log(f"\n将尝试 {len(fusion_types)} 种融合层级 × {len(fusion_methods)} 种融合策略 = {len(fusion_types) * len(fusion_methods)} 种组合")
    
    # 存储所有结果
    all_results = {}
    best_combo = None
    best_mAP = 0
    
    # 遍历所有组合
    for fusion_type in fusion_types:
        for fusion_method in fusion_methods:
            combo_name = f"{fusion_type}_{fusion_method}"
            logger.log(f"\n{'=' * 60}")
            logger.log(f"处理组合: {combo_name}")
            logger.log(f"{'=' * 60}")
            
            try:
                # 评估（默认执行，除非明确指定 --no-test）
                if not hasattr(args, 'no_test') or not args.no_test:
                    results = ensemble.evaluate(val_loader, num_query, fusion_type, fusion_method, args.reranking if hasattr(args, 'reranking') else False)
                    
                    if results:
                        all_results[combo_name] = results
                        
                        logger.log(f"\n结果:")
                        logger.log(f"  mAP: {results['mAP']:.2f}%")
                        logger.log(f"  Rank-1: {results['Rank-1']:.2f}%")
                        logger.log(f"  Rank-5: {results['Rank-5']:.2f}%")
                        logger.log(f"  Rank-10: {results['Rank-10']:.2f}%")
                        
                        # 与单模型比较
                        for m in selected_models:
                            if m.get('mAP'):
                                diff = results['mAP'] - m['mAP']
                                logger.log(f"  vs {m['name']}: {diff:+.2f}%")
                        
                        # 更新最佳
                        if results['mAP'] > best_mAP:
                            best_mAP = results['mAP']
                            best_combo = combo_name
                        
                        # 保存结果 txt
                        result_file = os.path.join(output_dir, f'{combo_name}_results.txt')
                        with open(result_file, 'w', encoding='utf-8') as f:
                            f.write("=" * 50 + "\n")
                            f.write(f"融合测试结果 - {combo_name}\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"融合层级: {fusion_type}\n")
                            f.write(f"融合策略: {fusion_method}\n")
                            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            f.write("-" * 50 + "\n")
                            f.write("融合模型:\n")
                            for m in selected_models:
                                f.write(f"  - {m['name']}: {m['path']}\n")
                            f.write("-" * 50 + "\n\n")
                            f.write("评估结果:\n")
                            f.write(f"  mAP:    {results['mAP']:.2f}%\n")
                            f.write(f"  Rank-1: {results['Rank-1']:.2f}%\n")
                            f.write(f"  Rank-5: {results['Rank-5']:.2f}%\n")
                            f.write(f"  Rank-10: {results['Rank-10']:.2f}%\n")
                            if results.get('note'):
                                f.write(f"\n注: {results['note']}\n")
                        logger.log(f"  结果已保存: {result_file}")
                        
                        # 保存 distmat
                        if results.get('distmat') is not None:
                            distmat_file = os.path.join(output_dir, f'{combo_name}_distmat.npy')
                            np.save(distmat_file, results['distmat'])
                            logger.log(f"  distmat已保存: {distmat_file}")
                        else:
                            logger.log(f"  此组合不支持保存 distmat")
                        
                        # 保存模型
                        model_file = os.path.join(output_dir, f'{combo_name}.pth')
                        # 传递融合后的特征向量
                        fused_feat = results.get('features')
                        ensemble.save_ensemble_model(model_file, fusion_type, fusion_method, fused_feat)
                        logger.log(f"  模型已保存: {model_file}")
                    else:
                        logger.log(f"  组合 {combo_name} 评估失败，跳过")
                
            except Exception as e:
                logger.log(f"  错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 输出汇总
    total_time = time.time() - start_time
    
    logger.log(f"\n{'=' * 70}")
    logger.log("所有组合处理完成!")
    logger.log(f"总耗时: {total_time/60:.1f} 分钟")
    logger.log(f"{'=' * 70}")
    
    if all_results:
        logger.log(f"\n结果汇总:")
        logger.log(f"{'-' * 70}")
        logger.log(f"{'组合':<25} {'mAP':<12} {'Rank-1':<12} {'Rank-5':<12} {'Rank-10':<12}")
        logger.log(f"{'-' * 70}")
        for combo, res in sorted(all_results.items(), key=lambda x: x[1]['mAP'], reverse=True):
            marker = " ★" if combo == best_combo else ""
            logger.log(f"{combo:<25} {res['mAP']:<11.2f}% {res['Rank-1']:<11.2f}% {res['Rank-5']:<11.2f}% {res['Rank-10']:<11.2f}%{marker}")
        logger.log(f"{'-' * 70}")
        
        if best_combo:
            logger.log(f"★ 最佳组合: {best_combo} (mAP: {best_mAP:.2f}%)")
    
    logger.log(f"\n输出目录: {output_dir}")
    logger.log(f"日志文件: {log_file}")


if __name__ == '__main__':
    main()
