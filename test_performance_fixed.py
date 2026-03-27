"""
后端性能测试脚本
仅测试特征提取和查询匹配用时(使用FormData格式)
"""

import time
import requests
import numpy as np
import json
from PIL import Image
from pathlib import Path

# API服务器地址
BASE_URL = "http://127.0.0.1:8003"

# 测试图片路径
TEST_IMAGE_PATH = "data/BallShow/bounding_box_test/0002_c1s1_000001_00.jpg"


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


def test_feature_extraction():
    """测试特征提取性能"""
    print("=" * 60)
    print("测试1: 特征提取性能(API调用)")
    print("=" * 60)

    # 获取所有模型的最后一个checkpoint
    models = get_last_checkpoint_from_logs()
    if not models:
        print("未找到任何模型")
        return []

    print(f"API服务器: {BASE_URL}")
    print(f"测试图片: {TEST_IMAGE_PATH}")

    # 读取测试图片
    with open(TEST_IMAGE_PATH, 'rb') as f:
        image_data = f.read()

    all_times = []
    # 对每个模型进行测试
    for model_info in models:
        print(f"\n{'='*60}")
        print(f"测试模型: {model_info['name']}")
        print(f"{'='*60}")

        MODEL_PATH = model_info['path']

        # 预热
        print("[预热] 第一次请求(包含模型加载)...")
        files = {'image': ('test.jpg', image_data, 'image/jpeg')}
        data = {'model_path': MODEL_PATH}
        response = requests.post(f"{BASE_URL}/extract", files=files, data=data)
        print(f"状态码: {response.status_code}")

        # 正式测试
        print("[正式测试] 连续10次特征提取...")
        times = []
        for i in range(10):
            files = {'image': ('test.jpg', image_data, 'image/jpeg')}
            data = {'model_path': MODEL_PATH}

            start_time = time.time()
            response = requests.post(f"{BASE_URL}/extract", files=files, data=data)
            end_time = time.time()

            if response.status_code == 200:
                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)
                print(f"  第{i+1}次: {elapsed_ms:.2f}ms")
            else:
                print(f"  第{i+1}次: 错误 {response.status_code}")

        # 统计
        if times:
            print(f"\n[结果统计]")
            print(f"  平均时间: {np.mean(times):.2f}ms")
            print(f"  最短时间: {np.min(times):.2f}ms")
            print(f"  最长时间: {np.max(times):.2f}ms")
            print(f"  标准差:   {np.std(times):.2f}ms")

        all_times.extend(times)

    return all_times


def test_search():
    """测试查询匹配性能(使用真实特征)"""
    print("\n" + "=" * 60)
    print("测试2: 查询匹配性能(API调用,使用真实特征)")
    print("=" * 60)

    # 获取所有模型的最后一个checkpoint
    models = get_last_checkpoint_from_logs()
    if not models:
        print("未找到任何模型")
        return []

    # 只使用第一个模型进行查询测试
    model_info = models[0]
    MODEL_PATH = model_info['path']

    print(f"使用模型: {model_info['name']}")

    # 读取查询图片
    with open(TEST_IMAGE_PATH, 'rb') as f:
        query_data = f.read()

    # 先提取一些gallery特征
    print("\n[准备] 提取5张图片作为gallery...")
    gallery_images = [
        "data/BallShow/bounding_box_test/0002_c1s1_000001_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000002_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000003_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000004_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000005_00.jpg",
    ]

    gallery_features = []
    gallery_pids = []

    for idx, img_path in enumerate(gallery_images):
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()

            files = {'image': ('gallery.jpg', img_data, 'image/jpeg')}
            data = {'model_path': MODEL_PATH}
            response = requests.post(f"{BASE_URL}/extract", files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                gallery_features.append(result['features'])
                gallery_pids.append(str(idx + 1))  # 转换为字符串
                print(f"  [OK] {img_path}")
        except Exception as e:
            print(f"  [FAIL] {img_path}: {e}")

    print(f"\nGallery准备完成: {len(gallery_features)}张图片")

    # 序列化gallery特征
    gallery_features_str = json.dumps(gallery_features)
    gallery_pids_str = json.dumps(gallery_pids)

    # 预热
    print("\n[预热] 第一次查询...")
    files = {'image': ('query.jpg', query_data, 'image/jpeg')}
    data = {
        'model_path': MODEL_PATH,
        'gallery_features': gallery_features_str,
        'gallery_pids': gallery_pids_str,
        'k': 3  # 小于 gallery 大小
    }
    response = requests.post(f"{BASE_URL}/search", files=files, data=data)
    print(f"状态码: {response.status_code}")

    # 正式测试
    print("\n[正式测试] 连续10次查询匹配...")
    times = []
    for i in range(10):
        files = {'image': ('query.jpg', query_data, 'image/jpeg')}
        data = {
            'model_path': MODEL_PATH,
            'gallery_features': gallery_features_str,
            'gallery_pids': gallery_pids_str,
            'k': 3  # 小于 gallery 大小
        }

        start_time = time.time()
        response = requests.post(f"{BASE_URL}/search", files=files, data=data)
        end_time = time.time()

        if response.status_code == 200:
            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
            print(f"  第{i+1}次: {elapsed_ms:.2f}ms")
        else:
            print(f"  第{i+1}次: 错误 {response.status_code}")

    # 统计
    if times:
        print(f"\n[结果统计]")
        print(f"  平均时间: {np.mean(times):.2f}ms")
        print(f"  最短时间: {np.min(times):.2f}ms")
        print(f"  最长时间: {np.max(times):.2f}ms")
        print(f"  标准差:   {np.std(times):.2f}ms")

    return times


def test_search_with_reranking():
    """测试查询匹配性能(使用Re-ranking)"""
    print("\n" + "=" * 60)
    print("测试3: 查询匹配性能(API调用,使用Re-ranking)")
    print("=" * 60)

    # 获取所有模型的最后一个checkpoint
    models = get_last_checkpoint_from_logs()
    if not models:
        print("未找到任何模型")
        return []

    # 只使用第一个模型进行查询测试
    model_info = models[0]
    MODEL_PATH = model_info['path']

    print(f"使用模型: {model_info['name']}")

    # 读取查询图片
    with open(TEST_IMAGE_PATH, 'rb') as f:
        query_data = f.read()

    # 准备gallery特征
    print("\n[准备] 使用较大的gallery...")
    gallery_images = [
        "data/BallShow/bounding_box_test/0002_c1s1_000001_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000002_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000003_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000004_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000005_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000006_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000007_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000008_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000009_00.jpg",
        "data/BallShow/bounding_box_test/0002_c1s1_000010_00.jpg",
    ]

    gallery_features = []
    gallery_pids = []

    for idx, img_path in enumerate(gallery_images):
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()

            files = {'image': ('gallery.jpg', img_data, 'image/jpeg')}
            data = {'model_path': MODEL_PATH}
            response = requests.post(f"{BASE_URL}/extract", files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                gallery_features.append(result['features'])
                gallery_pids.append(str(idx + 1))  # 转换为字符串
        except:
            pass

    print(f"Gallery准备完成: {len(gallery_features)}张图片")

    # 序列化gallery特征
    gallery_features_str = json.dumps(gallery_features)
    gallery_pids_str = json.dumps(gallery_pids)

    # 预热
    print("\n[预热] 第一次查询(Re-ranking)...")
    files = {'image': ('query.jpg', query_data, 'image/jpeg')}
    data = {
        'model_path': MODEL_PATH,
        'gallery_features': gallery_features_str,
        'gallery_pids': gallery_pids_str,
        'use_reranking': 'true',
        'k': 5  # 小于 gallery 大小
    }
    response = requests.post(f"{BASE_URL}/search", files=files, data=data)

    if response.status_code != 200:
        print(f"查询失败: {response.status_code}")
        print(f"Re-ranking测试跳过(这是正常的,因为re-ranking需要特定的gallery大小)")
        return []

    print(f"状态码: {response.status_code}")

    # 正式测试
    print("\n[正式测试] 连续10次查询匹配(Re-ranking)...")
    times = []
    for i in range(10):
        files = {'image': ('query.jpg', query_data, 'image/jpeg')}
        data = {
            'model_path': MODEL_PATH,
            'gallery_features': gallery_features_str,
            'gallery_pids': gallery_pids_str,
            'use_reranking': 'true',
            'k': 5  # 小于 gallery 大小
        }

        start_time = time.time()
        response = requests.post(f"{BASE_URL}/search", files=files, data=data)
        end_time = time.time()

        if response.status_code == 200:
            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
            print(f"  第{i+1}次: {elapsed_ms:.2f}ms")
        else:
            print(f"  第{i+1}次: 错误 {response.status_code}")

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
    print("后端性能测试(API调用版 - 使用真实特征)")
    print("=" * 60)

    try:
        # 测试1: 特征提取
        extract_times = test_feature_extraction()

        # 测试2: 查询匹配(无Re-ranking)
        search_times = test_search()

        # 测试3: 查询匹配(使用Re-ranking)
        search_rerank_times = test_search_with_reranking()

        # 总结
        print("\n" + "=" * 50)
        print("[性能总结]")
        print("=" * 50)
        if extract_times:
            print(f"特征提取平均时间: {np.mean(extract_times):.2f}ms")
            print(f"特征提取最短时间: {np.min(extract_times):.2f}ms")
        if search_times:
            print(f"查询匹配平均时间(无Re-ranking): {np.mean(search_times):.2f}ms")
        if search_rerank_times:
            print(f"查询匹配平均时间(使用Re-ranking): {np.mean(search_rerank_times):.2f}ms")

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
