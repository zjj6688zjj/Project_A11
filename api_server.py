# ========== 全局UTF-8编码设置（修复Windows跨平台兼容）==========
import sys
import io
import os

# 强制UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

# 设置默认编码
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# ========== 原有导入 ==========
import json
import torch
import numpy as np
import glob
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from model.make_model import make_model
from config import cfg
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# 全局缓存
model_cache = {}
model_config_cache = {}
gallery_cache = {}
gallery_normalized_cache = {}  # 缓存归一化后的gallery特征
device = None

# Gallery特征缓存目录
GALLERY_CACHE_DIR = "gallery_cache"

# 初始化GPU设备
def init_device():
    """初始化并预热GPU"""
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        # 预热GPU
        dummy = torch.randn(1, 3, 256, 128).to(device)
        _ = dummy.sum()
        torch.cuda.synchronize()
        print(f"[OK] GPU initialized: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("[WARNING] CUDA not available, using CPU")

    return device

# 加载模型
def load_model(model_path):
    """加载模型并移到GPU"""
    model_path = model_path.replace('\\', '/')

    print(f"Loading model: {model_path}")

    # Linux训练的模型使用UTF-8，Windows训练的模型使用GBK
    # 统一使用encoding='utf-8'以兼容Linux训练的模型
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False, encoding='utf-8')
    except Exception as e:
        print(f"UTF-8 encoding load failed: {e}, trying without encoding...")
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e2:
            print(f"Standard load failed: {e2}, trying with pickle...")
            import pickle
            state_dict = torch.load(model_path, map_location=device, pickle_module=pickle)

    # 推断模型配置
    has_sie = any('sie_embed' in k for k in state_dict.keys())
    has_jpm = any('jpm' in k for k in state_dict.keys())
    
    # 打印 checkpoint 中的关键信息用于诊断
    print(f"  Checkpoint keys count: {len(state_dict.keys())}")
    pos_embed_shape = None
    for k, v in state_dict.items():
        if 'pos_embed' in k:
            pos_embed_shape = v.shape
            print(f"  Found {k}: shape={v.shape}")
    
    # 根据 pos_embed shape 推断原始训练配置
    # pos_embed shape: [1, num_patches+1, 768]
    # num_patches = h * w, 其中:
    #   h = (H - 16) / stride_y + 1
    #   w = (W - 16) / stride_x + 1
    inferred_stride = None
    if pos_embed_shape is not None:
        num_patches = pos_embed_shape[1] - 1  # 减去 cls token
        print(f"  Inferred num_patches (without cls): {num_patches}")
        # 尝试常见的尺寸组合
        # 256x128: (240/s+1) * (112/s+1) ≈ num_patches
        # 384x128: (368/s+1) * (112/s+1) ≈ num_patches
        for h in [256, 384]:
            for w in [128]:
                for stride in [16, 12, 8]:
                    num_y = (h - 16) // stride + 1
                    num_x = (w - 16) // stride + 1
                    if num_y * num_x == num_patches:
                        print(f"  => Inferred config: img_size=[{h},{w}], stride=[{stride},{stride}]")
                        inferred_stride = [stride, stride]
                        inferred_img_size = [h, w]

    # 加载配置
    from config.defaults import _C
    from copy import deepcopy

    new_cfg = deepcopy(_C)
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_dir)
    config_file = f"configs/BallShow/{model_name.replace('BallShow_', '')}.yml"

    if model_name == "BallShow":
        config_file = "configs/BallShow/vit_transreid_stride.yml" if has_sie else "configs/BallShow/vit_base.yml"

    def load_config_safe(config_path):
        """安全加载配置文件，修复Windows GBK编码问题"""
        # Monkey patch YACS merge_from_file to always use UTF-8
        from yacs import config as yacs_config
        original_merge_from_file = yacs_config.CfgNode.merge_from_file
        
        def patched_merge_from_file(self, cfg_filename):
            """强制使用UTF-8编码读取yaml文件"""
            with open(cfg_filename, "r", encoding="utf-8", errors="replace") as f:
                cfg = self.load_cfg(f)
            self.merge_from_other_cfg(cfg)
        
        # 应用patch
        yacs_config.CfgNode.merge_from_file = patched_merge_from_file
        
        try:
            new_cfg.merge_from_file(config_path)
        finally:
            # 恢复原始方法
            yacs_config.CfgNode.merge_from_file = original_merge_from_file
    
    if os.path.exists(config_file):
        load_config_safe(config_file)
    else:
        load_config_safe("configs/BallShow/vit_base.yml")

    # 应用从 checkpoint 推断的配置
    if inferred_stride is not None:
        print(f"  Applying inferred config: STRIDE_SIZE={inferred_stride}, SIZE_TRAIN={inferred_img_size}")
        new_cfg.MODEL.STRIDE_SIZE = inferred_stride
        new_cfg.INPUT.SIZE_TRAIN = inferred_img_size
        new_cfg.INPUT.SIZE_TEST = inferred_img_size

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

    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # 预热
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, new_cfg.INPUT.SIZE_TRAIN[0], new_cfg.INPUT.SIZE_TRAIN[1]).to(device)
        _ = model(dummy_input, cam_label=0, view_label=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[OK] Model loaded and warmed up on {device}")

    return model, new_cfg

# 预处理图像
def preprocess_image(image, height=256, width=128):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 提取特征(GPU版,返回numpy)
def extract_features(model, image, camera_id=0, view_id=0):
    """在GPU上提取特征,返回numpy数组"""
    import time
    start = time.time()

    with torch.no_grad():
        features = model(image, cam_label=camera_id, view_label=view_id)

        # 处理特征维度
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        elif features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 1:
            features = features.unsqueeze(0)

    # 立即转到CPU(numpy)避免GPU->CPU传输开销
    features_np = features.cpu().numpy()

    elapsed = (time.time() - start) * 1000
    print(f"  Feature extraction: {elapsed:.2f}ms on {features.device}")

    return features_np

# 计算相似度(CPU版,优化矩阵乘法)
def calculate_similarity_fast(query_features, gallery_features, k=10):
    """在CPU上快速计算相似度(使用numpy优化)"""
    import time
    start = time.time()

    # 归一化gallery特征(使用全局缓存)
    gallery_id = id(gallery_features)
    if gallery_id not in gallery_normalized_cache:
        gallery_norm = np.linalg.norm(gallery_features, axis=1, keepdims=True)
        gallery_normalized_cache[gallery_id] = gallery_features / (gallery_norm + 1e-8)

    gallery_normalized = gallery_normalized_cache[gallery_id]

    # 归一化query
    query_norm = np.linalg.norm(query_features)
    query_normalized = query_features / (query_norm + 1e-8)

    # 使用numpy的dot进行相似度计算
    similarities = np.dot(gallery_normalized, query_normalized.T).flatten()

    # 使用argpartition获取top-k(比全排序更快)
    top_k_indices = np.argpartition(similarities, -k)[-k:]

    # 对top-k结果排序
    top_k_sorted = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    top_k_values = similarities[top_k_sorted]

    elapsed = (time.time() - start) * 1000
    print(f"  Similarity calculation: {elapsed:.2f}ms on CPU")

    return top_k_values, top_k_sorted

# 扫描模型
def scan_models_from_logs(logs_dir="logs"):
    """扫描logs目录获取所有模型"""
    models = []

    if not os.path.exists(logs_dir):
        return models

    for config_name in os.listdir(logs_dir):
        config_path = os.path.join(logs_dir, config_name)

        if not os.path.isdir(config_path) or config_name.startswith('.'):
            continue

        pth_files = glob.glob(os.path.join(config_path, "*.pth"))

        if not pth_files:
            continue

        max_epoch = 0
        latest_pth = None

        for pth_file in pth_files:
            match = re.search(r'checkpoint_(\d+)\.pth$', os.path.basename(pth_file))
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_pth = pth_file

        if latest_pth:
            clean_name = config_name.replace("BallShow_", "")
            display_name = f"{clean_name} (Epoch {max_epoch})"

            models.append({
                "id": config_name,
                "name": display_name,
                "path": latest_pth,
                "epoch": max_epoch
            })

    models.sort(key=lambda x: x['epoch'], reverse=True)
    return models

# 预加载gallery特征(GPU提取,CPU存储)
def load_gallery_features(model, cfg, model_path):
    """在GPU上提取gallery特征,存储在CPU上"""
    gallery_dir = "data/BallShow/bounding_box_test"

    if not os.path.exists(gallery_dir):
        print(f"Warning: Gallery directory not found: {gallery_dir}")
        return None, []

    # 检查缓存文件
    os.makedirs(GALLERY_CACHE_DIR, exist_ok=True)
    cache_key = model_path.replace('\\', '/').replace('//', '/').replace('/', '_')
    cache_file = os.path.join(GALLERY_CACHE_DIR, f"{cache_key}.npz")

    if os.path.exists(cache_file):
        print(f"Loading gallery features from cache: {cache_file}")
        try:
            cache_data = np.load(cache_file, allow_pickle=True)
            gallery_features = cache_data['features']
            gallery_paths = cache_data['paths'].tolist() if 'paths' in cache_data else []
        except Exception:
            print(f"Cache load failed, regenerating...")
            os.unlink(cache_file)
            return None, []
        print(f"[OK] Gallery features loaded from cache: shape={gallery_features.shape}")
        return gallery_features, gallery_paths

    image_files = sorted(glob.glob(os.path.join(gallery_dir, "*.jpg")))

    if not image_files:
        print(f"Warning: No images found in {gallery_dir}")
        return None, []

    print(f"Loading {len(image_files)} gallery images...")

    height = cfg.INPUT.SIZE_TRAIN[0] if hasattr(cfg.INPUT, 'SIZE_TRAIN') else 256
    width = cfg.INPUT.SIZE_TRAIN[1] if hasattr(cfg.INPUT, 'SIZE_TRAIN') else 128

    batch_size = 64
    gallery_features_list = []
    gallery_paths = []

    model.eval()

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        valid_paths = []

        for img_path in batch_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess_image(img, height=height, width=width)
                batch_images.append(img_tensor)
                valid_paths.append(img_path)
            except Exception as e:
                continue

        if batch_images:
            batch_tensor = torch.cat(batch_images, dim=0).to(device)

            with torch.no_grad():
                features = model(batch_tensor, cam_label=0, view_label=0)

                # 处理特征维度
                print(f"    Raw features shape: {features.shape}, dim: {features.dim()}")
                if features.dim() == 4:
                    features = features.mean(dim=[2, 3])
                elif features.dim() == 3:
                    features = features.mean(dim=1)
                elif features.dim() == 1:
                    features = features.unsqueeze(0)
                print(f"    Processed features shape: {features.shape}")

            # 立即转到CPU存储
            gallery_features_list.append(features.cpu().numpy())
            gallery_paths.extend(valid_paths)

        # 优化: 每1000张显示一次进度
        if (i + batch_size) % 1000 == 0:
            print(f"  Processed {min(i + batch_size, len(image_files))}/{len(image_files)} images")

    if gallery_features_list:
        gallery_features = np.vstack(gallery_features_list)
        gallery_paths_array = np.array(gallery_paths, dtype='U')

        # 保存到缓存文件
        print(f"Saving gallery features to cache: {cache_file}")
        np.savez_compressed(cache_file, features=gallery_features, paths=gallery_paths_array)

        print(f"[OK] Gallery features loaded: shape={gallery_features.shape}")
        return gallery_features, gallery_paths
    else:
        return None, []

# API端点
@app.get("/api/models")
async def get_models():
    """获取模型列表"""
    try:
        models = scan_models_from_logs()
        return {
            'success': True,
            'models': models,
            'count': len(models)
        }
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract")
async def extract(
    model_path: str = Form(...),
    image: UploadFile = File(...)
):
    """提取图像特征(GPU版)"""
    import time
    total_start = time.time()

    try:
        # 统一路径格式(确保缓存一致性)
        model_path_normalized = model_path.replace('\\', '/').replace('//', '/')

        # 加载模型(仅首次)
        if model_path_normalized not in model_cache:
            print(f"\nLoading model: {model_path_normalized}")
            model, cfg = load_model(model_path_normalized)
            model_cache[model_path_normalized] = model
            model_config_cache[model_path_normalized] = cfg
        else:
            print(f"Using cached model: {model_path_normalized}")

        model = model_cache[model_path_normalized]
        cfg = model_config_cache[model_path_normalized]

        # 读取和预处理图像(主要耗时)
        read_start = time.time()
        image_data = await image.read()
        read_time = (time.time() - read_start) * 1000

        convert_start = time.time()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        convert_time = (time.time() - convert_start) * 1000

        preprocess_start = time.time()
        height = cfg.INPUT.SIZE_TRAIN[0] if hasattr(cfg.INPUT, 'SIZE_TRAIN') else 256
        width = cfg.INPUT.SIZE_TRAIN[1] if hasattr(cfg.INPUT, 'SIZE_TRAIN') else 128
        image_tensor = preprocess_image(image, height=height, width=width).to(device)
        preprocess_time = (time.time() - preprocess_start) * 1000

        # 提取特征(GPU -> CPU)
        feature_start = time.time()
        features = extract_features(model, image_tensor)
        feature_time = (time.time() - feature_start) * 1000

        # JSON序列化
        serialize_start = time.time()
        features_list = features.tolist()
        serialize_time = (time.time() - serialize_start) * 1000

        total_time = (time.time() - total_start) * 1000
        print(f"  Image read: {read_time:.2f}ms")
        print(f"  Image convert: {convert_time:.2f}ms")
        print(f"  Preprocess: {preprocess_time:.2f}ms")
        print(f"  Feature extract: {feature_time:.2f}ms")
        print(f"  Serialize: {serialize_time:.2f}ms")
        print(f"  Total extract time: {total_time:.2f}ms")

        return {
            'success': True,
            'features': features_list,
            'timing': {
                'total': round(total_time, 2),
                'feature_time': round(feature_time, 2),
                'preprocess_time': round(preprocess_time, 2),
                'read_convert_time': round(read_time + convert_time, 2),
                'serialize_time': round(serialize_time, 2)
            }
        }
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(
    image: UploadFile = File(None),
    model_path: str = Form(...),
    gallery_features: str = Form(None),
    gallery_pids: str = Form(None),
    use_reranking: str = Form('false'),
    k: int = Form(10),
    features: str = Form(None)  # 新增:直接传递query特征
):
    """搜索相似图像(混合策略: GPU提取特征,CPU计算相似度,支持re-ranking)"""
    import time
    total_start = time.time()

    try:
        # 统一路径格式(确保缓存一致性)
        model_path_normalized = model_path.replace('\\', '/').replace('//', '/')

        # 如果提供了gallery_features,直接使用;否则从磁盘加载
        if gallery_features:
            print(f"Using provided gallery features: {len(json.loads(gallery_features))} images")
            provided_gallery = np.array(json.loads(gallery_features), dtype=np.float32)
            provided_pids = json.loads(gallery_pids) if gallery_pids else list(range(len(provided_gallery)))
        else:
            provided_gallery = None

        # 处理query特征:优先使用直接传递的特征,否则从上传的图片提取
        extract_time = 0  # 初始化变量
        if features:
            # 直接使用传递的特征
            print(f"Using provided query features")
            query_features = np.array(json.loads(features), dtype=np.float32)
            if len(query_features.shape) == 1:
                query_features = query_features.reshape(1, -1)
            extract_time = 0  # 使用缓存特征时，提取时间为0
        elif image:
            image_data = await image.read()
            img = Image.open(BytesIO(image_data)).convert('RGB')

            # 加载模型
            if model_path_normalized not in model_cache:
                print(f"Loading model: {model_path_normalized}")
                model, cfg = load_model(model_path_normalized)
                model_cache[model_path_normalized] = model
                model_config_cache[model_path_normalized] = cfg
            else:
                print(f"Using cached model: {model_path_normalized}")
            
            model = model_cache[model_path_normalized]
            cfg = model_config_cache[model_path_normalized]
            
            # 提取特征
            height = cfg.INPUT.SIZE_TRAIN[0] if hasattr(cfg.INPUT, 'SIZE_TRAIN') else 256
            width = cfg.INPUT.SIZE_TRAIN[1] if hasattr(cfg.INPUT, 'SIZE_TRAIN') else 128
            image_tensor = preprocess_image(img, height=height, width=width).to(device)
            query_features = extract_features(model, image_tensor)
        else:
            raise ValueError("Either image or features must be provided")

        # 加载模型和gallery(仅首次,如果没有提供gallery_features)
        if provided_gallery is not None:
            # 使用提供的gallery特征
            gallery_features = provided_gallery
            gallery_paths = provided_pids
        else:
            # 从磁盘加载gallery
            if model_path_normalized not in gallery_cache:
                print(f"\nLoading gallery for model: {model_path_normalized}")

                if model_path_normalized not in model_cache:
                    print(f"Loading model: {model_path_normalized}")
                    model, cfg = load_model(model_path_normalized)
                    model_cache[model_path_normalized] = model
                    model_config_cache[model_path_normalized] = cfg
                else:
                    print(f"Using cached model: {model_path_normalized}")

                model = model_cache[model_path_normalized]
                cfg = model_config_cache[model_path_normalized]

                # 提取gallery特征(GPU -> CPU)
                gallery_features, gallery_paths = load_gallery_features(model, cfg, model_path_normalized)
                gallery_cache[model_path_normalized] = (gallery_features, gallery_paths)

            gallery_features, gallery_paths = gallery_cache[model_path_normalized]

        if gallery_features is None:
            return {
                'success': False,
                'message': 'Gallery not available'
            }

        # 检查是否使用re-ranking
        cfg = model_config_cache[model_path_normalized]
        config_use_reranking = cfg.TEST.RE_RANKING
        use_reranking = config_use_reranking or use_reranking.lower() == 'true'

        if use_reranking:
            print("  Using re-ranking...")
            from utils.reranking import re_ranking

            # 转换为torch tensor (GPU) - 确保是2D张量
            import torch
            # query_features: shape (768,) 或 (1, 768) 或 (1, 1, 3840) -> (1, 768)
            query_features_2d = query_features.reshape(1, -1)
            query_tensor = torch.from_numpy(query_features_2d).to(device)
            # gallery_features: shape (10, 768) 或 (10, 1, 3840) -> (10, 768)
            gallery_features_2d = gallery_features.reshape(gallery_features.shape[0], -1)
            gallery_tensor = torch.from_numpy(gallery_features_2d).to(device)

            print(f"  Query tensor shape: {query_tensor.shape}")
            print(f"  Gallery tensor shape: {gallery_tensor.shape}")

            # Re-ranking
            re_rank_start = time.time()
            re_rank_dist = re_ranking(query_tensor, gallery_tensor, k1=50, k2=15, lambda_value=0.3)
            re_rank_time = (time.time() - re_rank_start) * 1000

            print(f"  Re-ranking distance shape: {re_rank_dist.shape}")

            # 使用re-ranking后的距离(转换为相似度)
            similarities = 1 - re_rank_dist.flatten()
            print(f"  Re-ranking time: {re_rank_time:.2f}ms")

            # 获取top-k
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]
            top_k_values = similarities[top_k_sorted]
        else:
            # CPU上计算相似度
            top_k_values, top_k_indices = calculate_similarity_fast(query_features, gallery_features, k=k)
            top_k_sorted = top_k_indices

        # 准备结果
        results = []
        for i, idx in enumerate(top_k_sorted):
            results.append({
                'rank': i + 1,
                'id': os.path.basename(gallery_paths[idx]),
                'similarity': float(top_k_values[i]),
                'path': gallery_paths[idx]
            })

        total_time = (time.time() - total_start) * 1000
        rerank_info = "✓ Re-ranking enabled" if use_reranking else "✗ Re-ranking disabled"
        print(f"  Total search time: {total_time:.2f}ms ({rerank_info})")
        
        return {
            'success': True,
            'results': results,
            'reranking': use_reranking,
            'timing': {
                'total': round(total_time, 2),
                'extract_time': round(extract_time if 'extract_time' in locals() else 0, 2),
                'match_time': round(total_time - (extract_time if 'extract_time' in locals() else 0), 2)
            }
        }
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test():
    """测试服务器状态"""
    return {
        "status": "ok", 
        "message": "Server is running",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cached_models": len(model_cache),
        "loaded_galleries": len(gallery_cache)
    }

@app.get("/")
async def root():
    """主页"""
    from fastapi.responses import FileResponse
    return FileResponse("index.html")

if __name__ == '__main__':
    # 初始化设备
    init_device()

    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8003)
