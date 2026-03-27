"""
后端性能测试脚本(直接调用版 - 改进版)
使用真实图片特征，更准确地评估性能
"""

import time
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# 导入后端模块
import sys
sys.path.append('.')

from model.make_model import make_model
from config import cfg as default_cfg
from config.defaults import _C
from copy import deepcopy


def get_last_checkpoint_from_logs():
    """获取logs目录下每个子文件夹中最后一个pth文件"""
    logs_dir = Path("logs").resolve()
    if not logs_dir.exists():
        print(f"错误: logs目录不存在: {logs_dir}")
        raise FileNotFoundError("logs目录不存在")

    models = []
    
    print(f"扫描logs目录: {logs_dir}")
    
    # 获取所有子目录
    subdirs = []
    for item in sorted(logs_dir.iterdir()):
        item_path = logs_dir / item

        if item_path.is_dir():
            subdirs.append(item_path)
        else:
            print(f"    跳过非目录: {item}")
    
    if not subdirs:
        print("警告: logs目录下没有子目录")
        return models
    
    print(f"找到 {len(subdirs)} 个子目录")
    
    for i, config_path in enumerate(subdirs, 1):
        config_name = config_path.name
        
        print(f"  扫描目录 [{i}/{len(subdirs)}]: {config_name}")
        
        # 使用rglob递归查找所有.pth文件
        pth_files = list(config_path.rglob("*.pth"))
        
        if not pth_files:
            print(f"    未找到.pth文件")
            continue
        
        print(f"    找到 {len(pth_files)} 个.pth文件")
        
        max_epoch = 0
        latest_pth = None
        
        for pth_file in pth_files:
            # 提取epoch号
            filename = pth_file.name
            if 'checkpoint_' in filename:
                try:
                    # 匹配 transformer_checkpoint_10.pth 或 checkpoint_10.pth
                    epoch_str = filename.split('checkpoint_')[-1].split('.pth')[0]
                    epoch = int(epoch_str)
                    if epoch > max_epoch:
                        max_epoch = epoch
                        latest_pth = pth_file
                except Exception as e:
                    print(f"    解析文件名失败: {filename}, 错误: {e}")
                    continue
        
        if latest_pth:
            clean_name = config_name.replace("BallShow_", "")
            display_name = f"{clean_name} (Epoch {max_epoch})"
            
            models.append({
                "id": config_name,
                "name": display_name,
                "path": str(latest_pth),
                "epoch": max_epoch
            })
            print(f"    -> 添加模型: {display_name}")
        else:
            print(f"    未找到有效的checkpoint文件")
    
    if not models:
        print("警告: 未找到任何模型")
    else:
        print(f"找到 {len(models)} 个模型")
        
    models.sort(key=lambda x: x['epoch'], reverse=True)
    return models


def load_model_direct(model_path, device):
    """直接加载模型"""
    model_path = model_path.replace('\\', '/')

    state_dict = torch.load(model_path, map_location=device)

    # 推断模型配置
    has_sie = any('sie_embed' in k for k in state_dict.keys())
    has_jpm = any('jpm' in k for k in state_dict.keys())

    # 加载配置
    new_cfg = deepcopy(_C)
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_dir)
    config_file = f"configs/BallShow/{model_name.replace('BallShow_', '')}.yml"

    if model_name == "BallShow":
        config_file = "configs/BallShow/vit_transreid_stride.yml" if has_sie else "configs/BallShow/vit_base.yml"

    print(f"配置文件: {config_file}")

    if os.path.exists(config_file):
        new_cfg.merge_from_file(config_file)
    else:
        new_cfg.merge_from_file("configs/BallShow/vit_base.yml")

    if has_sie:
        new_cfg.MODEL.SIE_CAMERA = True
        new_cfg.MODEL.SIE_VIEW = True

    new_cfg.freeze()

    # 构建模型
    camera_num = 5 if new_cfg.MODEL.SIE_CAMERA else 1
    view_num = 1
    model = make_model(new_cfg, num_class=1000, camera_num=camera_num, view_num=view_num)

    # 移除分类器权重
    keys_to_remove = [k for k in state_dict.keys() if 'classifier' in k]
    for k in keys_to_remove:
        del state_dict[k]

    # 加载权重，允许部分加载（与 api_server.py 保持一致）
    # 注意：strict=False 允许缺失/多余的键，但不允许尺寸不匹配
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"  警告: 加载权重时出现尺寸不匹配: {e}")
        print(f"  将忽略尺寸不匹配的权重，使用随机初始化")
        # 移除尺寸不匹配的权重
        model_state_dict = model.state_dict()
        unmatched_keys = []
        for k in state_dict.keys():
            if k in model_state_dict and state_dict[k].shape != model_state_dict[k].shape:
                unmatched_keys.append(k)
        for k in unmatched_keys:
            print(f"    跳过不匹配的权重: {k} (checkpoint: {state_dict[k].shape} -> model: {model_state_dict[k].shape})")
            del state_dict[k]
        # 重新加载
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model, new_cfg


def preprocess_image(image, height=256, width=128):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def test_feature_extraction_direct():
    """测试特征提取性能(直接调用)"""
    print("\n" + "=" * 60)
    print("测试1: 特征提取性能(直接调用)")
    print("=" * 60)

    # 获取所有模型的最后一个checkpoint
    models = get_last_checkpoint_from_logs()
    if not models:
        print("未找到任何模型")
        return []

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 读取测试图片
    test_image_path = "data/BallShow/bounding_box_test/0002_c1s1_000001_00.jpg"
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return []

    image = Image.open(test_image_path).convert('RGB')

    # 对每个模型进行测试
    all_times = []
    for model_info in models:
        print(f"\n{'='*60}")
        print(f"测试模型: {model_info['name']}")
        print(f"{'='*60}")

        # 加载模型
        print(f"加载模型: {model_info['path']}")
        model, config = load_model_direct(model_info['path'], device)
        print("模型加载完成")

        # 获取输入尺寸
        if hasattr(config.INPUT, 'SIZE_TEST'):
            height = config.INPUT.SIZE_TEST[0]
            width = config.INPUT.SIZE_TEST[1]
        elif hasattr(config.INPUT, 'SIZE_TRAIN'):
            height = config.INPUT.SIZE_TRAIN[0]
            width = config.INPUT.SIZE_TRAIN[1]
        else:
            height, width = 256, 128

        print(f"输入尺寸: {height}x{width}")

        # 预热
        print("[预热] 预热GPU...")
        dummy = torch.randn(1, 3, height, width).to(device)
        cam_label = torch.tensor([0]).to(device) if config.MODEL.SIE_CAMERA else None
        view_label = torch.tensor([0]).to(device) if config.MODEL.SIE_VIEW else None
        _ = model(dummy, cam_label=cam_label, view_label=view_label)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print("预热完成")

        # 测试
        print("[正式测试] 连续20次特征提取...")
        times = []

        for i in range(20):
            # 预处理
            image_tensor = preprocess_image(image, height=height, width=width).to(device)

            # 提取特征并计时
            start_time = time.time()
            features = model(image_tensor, cam_label=cam_label, view_label=view_label)
            end_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
            print(f"  第{i+1}次: {elapsed_ms:.2f}ms")

        # 统计
        if times:
            print(f"\n[结果统计]")
            print(f"  平均时间: {np.mean(times):.2f}ms")
            print(f"  最短时间: {np.min(times):.2f}ms")
            print(f"  最长时间: {np.max(times):.2f}ms")
            print(f"  标准差:   {np.std(times):.2f}ms")

        all_times.extend(times)

    return all_times


def test_search_direct():
    """测试查询匹配性能(直接调用,使用真实特征)"""
    print("\n" + "=" * 50)
    print("测试2: 查询匹配性能(直接调用,CPU计算,使用真实特征)")
    print("=" * 50)

    # 获取所有模型的最后一个checkpoint
    models = get_last_checkpoint_from_logs()
    if not models:
        print("未找到任何模型")
        return []

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用第一个模型进行查询测试
    model_info = models[0]
    print(f"使用模型: {model_info['name']}")

    # 加载模型
    print(f"加载模型: {model_info['path']}")
    model, config = load_model_direct(model_info['path'], device)

    # 获取输入尺寸
    if hasattr(config.INPUT, 'SIZE_TEST'):
        height = config.INPUT.SIZE_TEST[0]
        width = config.INPUT.SIZE_TEST[1]
    elif hasattr(config.INPUT, 'SIZE_TRAIN'):
        height = config.INPUT.SIZE_TRAIN[0]
        width = config.INPUT.SIZE_TRAIN[1]
    else:
        height, width = 256, 128

    # 加载真实图片提取特征
    print("\n[准备] 加载真实图片并提取特征...")
    gallery_dir = "data/BallShow/bounding_box_test"

    if not os.path.exists(gallery_dir):
        print(f"  错误: Gallery目录不存在: {gallery_dir}")
        return []

    # 获取 gallery 图片
    gallery_images = sorted(Path(gallery_dir).glob("*.jpg"))

    if len(gallery_images) < 1000:
        print(f"  警告: 只找到 {len(gallery_images)} 张图片，将全部使用")
        num_images = len(gallery_images)
    else:
        print(f"  使用前 1000 张 gallery 图片")
        num_images = 1000

    # 提取特征
    gallery_features_list = []
    cam_label = torch.tensor([0]).to(device) if config.MODEL.SIE_CAMERA else None
    view_label = torch.tensor([0]).to(device) if config.MODEL.SIE_VIEW else None

    model.eval()

    for i, img_path in enumerate(gallery_images[:num_images]):
        try:
            img = Image.open(str(img_path)).convert('RGB')
            img_tensor = preprocess_image(img, height=height, width=width).to(device)

            with torch.no_grad():
                features = model(img_tensor, cam_label=cam_label, view_label=view_label)

                if features.dim() == 4:
                    features = features.mean(dim=[2, 3])
                elif features.dim() == 3:
                    features = features.mean(dim=1)
                elif features.dim() == 1:
                    features = features.unsqueeze(0)

                gallery_features_list.append(features.cpu().numpy())
        except Exception as e:
            continue

        if (i + 1) % 200 == 0:
            print(f"    已处理 {min(i + 1, num_images)}/{num_images} 张图片")

    if not gallery_features_list:
        print("  错误: 无法提取任何特征，使用随机数据")
        # 降级为随机数据
        gallery_features = np.random.randn(1000, 768).astype(np.float32)
        query_features = np.random.randn(1, 768).astype(np.float32)
    else:
        gallery_features = np.vstack(gallery_features_list)
        query_features = gallery_features[0:1]  # 使用第一张作为 query
        print(f"  [OK] Gallery 特征提取完成: shape={gallery_features.shape}")

    print(f"Query shape: {query_features.shape}")
    print(f"Gallery shape: {gallery_features.shape}")

    # 归一化
    query_normalized = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    gallery_normalized = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)

    # 对每个模型进行测试（查询匹配不依赖模型，只需测试一次）
    print("\n[预热] 预热...")
    similarities = query_normalized @ gallery_normalized.T
    similarities_flat = similarities.flatten()
    top_k = np.argpartition(similarities_flat, -10)[-10:]
    print("预热完成")

    # 测试
    print("\n[正式测试] 连续20次查询匹配(1000张gallery)...")
    times = []

    for i in range(20):
        start_time = time.time()

        # 计算相似度 (query: 1x768, gallery: 1000x768 -> result: 1x1000)
        similarities = query_normalized @ gallery_normalized.T  # shape: (1, 1000)

        # 获取top-10
        k = 10
        # similarities是1x1000,需要flatten成1000
        similarities_flat = similarities.flatten()  # shape: (1000,)
        top_k_indices = np.argpartition(similarities_flat, -k)[-k:]
        top_k_sorted = top_k_indices[np.argsort(-similarities_flat[top_k_indices])]
        top_k_values = similarities_flat[top_k_sorted]

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)

    # 统计
    if times:
        print(f"\n[结果统计]")
        print(f"  平均时间: {np.mean(times):.2f}ms")
        print(f"  最短时间: {np.min(times):.2f}ms")
        print(f"  最长时间: {np.max(times):.2f}ms")
        print(f"  标准差:   {np.std(times):.2f}ms")

    return times


def main():
    """主函数"""
    print("=" * 60)
    print("后端性能测试(直接调用版 - 使用真实特征)")
    print("=" * 60)

    # 测试1: 特征提取
    extract_times = test_feature_extraction_direct()

    # 测试2: 查询匹配
    search_times = test_search_direct()

    # 总结
    print("\n" + "=" * 50)
    print("[性能总结]")
    print("=" * 50)
    if extract_times:
        print(f"特征提取平均时间: {np.mean(extract_times):.2f}ms")
        print(f"特征提取最短时间: {np.min(extract_times):.2f}ms")
    if search_times:
        print(f"查询匹配平均时间(1000张gallery): {np.mean(search_times):.2f}ms")

    print("\n竞赛要求:")
    if extract_times:
        if np.mean(extract_times) <= 40:
            print(f"  特征提取 <= 40ms: [OK] 达标")
        else:
            print(f"  特征提取 <= 40ms: [X] 未达标 (平均{np.mean(extract_times):.2f}ms)")
    if search_times:
        if np.mean(search_times) <= 30:
            print(f"  查询匹配 <= 30ms: [OK] 达标")
        else:
            print(f"  查询匹配 <= 30ms: [X] 未达标 (平均{np.mean(search_times):.2f}ms)")


if __name__ == '__main__':
    main()
