"""
后端性能测试脚本(直接调用版)
不包含HTTP网络传输,测量纯后端性能
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
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs目录不存在")

    models = []
    for item in logs_dir.iterdir():
        if item.is_dir():
            # 查找该目录下所有的.pth文件
            pth_files = list(item.glob("*.pth"))
            if pth_files:
                # 按epoch号排序，取最后一个
                pth_files.sort(key=lambda x: int(x.stem.split('_')[-1]) if '_' in x.stem else 0)
                last_pth = pth_files[-1]
                models.append({
                    'name': item.name,
                    'path': str(last_pth)
                })

    return models


def load_model_direct(model_path, device):
    """直接加载模型(不通过API)"""
    model_path = model_path.replace('\\', '/')

    state_dict = torch.load(model_path, map_location=device)

    # 推断模型配置
    has_sie = any('sie_embed' in k for k in state_dict.keys())
    has_jpm = any('jpm' in k for k in state_dict.keys())

    # 推断位置编码形状来匹配输入配置
    pos_embed_shape = None
    for k in state_dict.keys():
        if 'pos_embed' in k:
            pos_embed_shape = state_dict[k].shape
            break

    # 加载配置
    new_cfg = deepcopy(_C)
    model_dir = model_path.replace('\\', '/')
    model_dir = model_dir[:model_dir.rfind('/')]
    model_name = model_dir[model_dir.rfind('/')+1:]

    # 根据文件夹名称推断配置文件（按最长匹配优先）
    config_file = None
    if model_name == 'BallShow_vit_transreid_stride_384':
        config_file = "configs/BallShow/vit_transreid_stride_384.yml"
    elif model_name == 'BallShow_vit_transreid_384':
        config_file = "configs/BallShow/vit_transreid_384.yml"
    elif model_name == 'BallShow_vit_transreid_stride':
        config_file = "configs/BallShow/vit_transreid_stride.yml"
    elif model_name == 'BallShow_vit_transreid':
        config_file = "configs/BallShow/vit_transreid.yml"
    elif model_name == 'BallShow_vit_base':
        config_file = "configs/BallShow/vit_base.yml"
    elif model_name == 'BallShow_vit_jpm':
        config_file = "configs/BallShow/vit_jpm.yml"
    elif model_name == 'BallShow_vit_sie':
        config_file = "configs/BallShow/vit_sie.yml"
    elif model_name == 'BallShow_transreid_advanced':
        config_file = "configs/BallShow/transreid_advanced.yml"
    elif model_name == 'BallShow_transreid_final':
        config_file = "configs/BallShow/transreid_final.yml"
    elif 'BallShow_vit_transreid' in model_name:
        config_file = "configs/BallShow/vit_transreid.yml"
    elif 'BallShow_vit_jpm' in model_name:
        config_file = "configs/BallShow/vit_jpm.yml"
    elif 'BallShow_vit_sie' in model_name:
        config_file = "configs/BallShow/vit_sie.yml"
    elif 'BallShow_vit_base' in model_name:
        config_file = "configs/BallShow/vit_base.yml"
    elif 'BallShow_transreid_advanced' in model_name:
        config_file = "configs/BallShow/transreid_advanced.yml"
    elif 'BallShow_transreid_final' in model_name:
        config_file = "configs/BallShow/transreid_final.yml"
    elif 'BallShow' in model_name:
        config_file = f"configs/BallShow/vit_transreid_stride.yml" if has_sie else "configs/BallShow/vit_base.yml"
    else:
        config_file = "configs/BallShow/vit_base.yml"

    print(f"配置文件: {config_file}")

    if not config_file.startswith('configs'):
        config_file = f"configs/BallShow/{config_file}"

    if 'BallShow' in config_file:
        if os.path.exists(config_file):
            new_cfg.merge_from_file(config_file)
        else:
            print(f"配置文件 {config_file} 不存在，使用默认配置")
            new_cfg.merge_from_file("configs/BallShow/vit_base.yml")
    else:
        new_cfg.merge_from_file(config_file)

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

    # 加载权重，允许部分加载
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  警告: 加载权重时出现问题: {e}")
        # 尝试忽略size mismatch
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  缺少的keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  未使用的keys: {len(unexpected_keys)}")

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


def extract_features_direct(model, image_tensor, device, cam_label=None, view_label=None):
    """直接提取特征"""
    with torch.no_grad():
        features = model(image_tensor, cam_label=cam_label, view_label=view_label)
        return features.cpu().numpy()


def test_feature_extraction_direct():
    """测试特征提取性能(直接调用)"""
    print("=" * 50)
    print("测试1: 特征提取性能(直接调用,无网络)")
    print("=" * 50)

    TEST_IMAGE_PATH = "data/BallShow/bounding_box_test/0002_c1s1_000001_00.jpg"

    # 获取所有模型的最后一个checkpoint
    models = get_last_checkpoint_from_logs()
    if not models:
        print("未找到任何模型")
        return []

    print(f"\n找到 {len(models)} 个模型:")
    for model_info in models:
        print(f"  - {model_info['name']}: {model_info['path']}")

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 读取测试图片
    image = Image.open(TEST_IMAGE_PATH).convert('RGB')

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
    """测试查询匹配性能(直接调用)"""
    print("\n" + "=" * 50)
    print("测试2: 查询匹配性能(直接调用,CPU计算)")
    print("=" * 50)

    # 获取所有模型的最后一个checkpoint
    models = get_last_checkpoint_from_logs()
    if not models:
        print("未找到任何模型")
        return []

    # 生成测试数据
    print("\n[准备] 生成测试数据...")
    # Query特征: 1张,768维
    query_features = np.random.randn(1, 768).astype(np.float32)
    # Gallery特征: 1000张,768维
    gallery_features = np.random.randn(1000, 768).astype(np.float32)

    print(f"Query shape: {query_features.shape}")
    print(f"Gallery shape: {gallery_features.shape}")

    # 归一化
    query_normalized = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_normalized = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)

    # 对每个模型进行测试（查询匹配不依赖模型，只需测试一次）
    print("\n[预热] 预热...")
    similarities = query_normalized @ gallery_normalized.T
    similarities_flat = similarities.flatten()
    top_k = np.argpartition(similarities_flat, -10)[-10:]
    print("预热完成")

    # 生成测试数据
    print("\n[准备] 生成测试数据...")
    # Query特征: 1张,768维
    query_features = np.random.randn(1, 768).astype(np.float32)
    # Gallery特征: 1000张,768维
    gallery_features = np.random.randn(1000, 768).astype(np.float32)

    print(f"Query shape: {query_features.shape}")
    print(f"Gallery shape: {gallery_features.shape}")

    # 归一化
    query_normalized = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_normalized = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)

    # 预热
    print("\n[预热] 预热...")
    similarities = query_normalized @ gallery_normalized.T
    similarities_flat = similarities.flatten()
    top_k = np.argpartition(similarities_flat, -10)[-10:]
    print("预热完成")

    # 测试
    print("\n[正式测试] 连续20次查询匹配(1000张gallery)...")
    print(f"将为 {len(models)} 个模型进行测试...")

    times = []
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
        print(f"  第{i+1}次: {elapsed_ms:.2f}ms")

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
    print("\n" + "=" * 50)
    print("后端性能测试(直接调用版)")
    print("=" * 50)
    print("注意: 此测试不包含HTTP网络传输开销")
    print("=" * 50)

    import os

    try:
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

        print(f"\n竞赛要求:")
        if extract_times:
            status = '[OK] 达标' if np.mean(extract_times) <= 40 else '[X] 未达标'
            print(f"  特征提取 <= 40ms: {status}")
        if search_times:
            status = '[OK] 达标' if np.mean(search_times) <= 30 else '[X] 未达标'
            print(f"  查询匹配 <= 30ms: {status}")

    except Exception as e:
        print(f"\n[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
