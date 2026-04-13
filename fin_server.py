#!/usr/bin/env python3
"""
比赛专用API服务器
"""

import os
import sys
import time
import socket
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import cfg
from model import make_model
from tta_cache_precomputing import TTACacheManager

MODEL_CONFIG = "configs/BallShow/rtx4090_from_final153model.yml"
MODEL_PATH = "logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth"
BEST_STRATEGY = "multi_scale_crop"
TTA_CACHE_DIR = "tta_cache"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="data/BallShow"), name="static")

competition_model = None
competition_gallery_features = None
competition_gallery_names = []
model_metadata = {}
service_start_time = None


def find_available_port(start_port=8002, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result != 0:
                return port
        except Exception:
            continue
    return None


def load_competition_model():
    global competition_model, model_metadata

    print("[INFO] 加载模型...")

    try:
        if not os.path.exists(MODEL_CONFIG):
            raise FileNotFoundError(f"配置文件不存在: {MODEL_CONFIG}")

        def load_config_safe(config_path):
            from yacs import config as yacs_config
            original_merge_from_file = yacs_config.CfgNode.merge_from_file

            def patched_merge_from_file(self, cfg_filename):
                with open(cfg_filename, "r", encoding="utf-8", errors="replace") as f:
                    cfg = self.load_cfg(f)
                self.merge_from_other_cfg(cfg)

            yacs_config.CfgNode.merge_from_file = patched_merge_from_file

            try:
                cfg.merge_from_file(config_path)
            finally:
                yacs_config.CfgNode.merge_from_file = original_merge_from_file

        load_config_safe(MODEL_CONFIG)

        camera_num = 5 if cfg.MODEL.SIE_CAMERA else 1
        view_num = 1
        num_class = 3353

        model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=view_num)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

        model.load_param(MODEL_PATH)
        model.eval()

        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            model = model.to(device)
            model = model.half()
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}, FP16")
        else:
            device = torch.device("cpu")
            model = model.to(device)
            print("[WARNING] 未检测到GPU，使用CPU")

        model_metadata = {
            "model_name": Path(MODEL_PATH).name,
            "config_file": Path(MODEL_CONFIG).name,
            "device": str(device),
            "precision": "fp16" if device.type == "cuda" else "fp32",
            "loaded_at": datetime.now().isoformat(),
        }

        print(f"[OK] Model loaded: {model_metadata['model_name']}")
        competition_model = model
        return model

    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_tta_cache():
    global competition_gallery_features, competition_gallery_names

    print("[INFO] 加载TTA缓存...")

    gallery_dir = "data/BallShow/bounding_box_test"
    if os.path.exists(gallery_dir):
        competition_gallery_names = sorted([
            f for f in os.listdir(gallery_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"  Gallery: {len(competition_gallery_names)} images")
    else:
        print(f"[WARNING] Gallery目录不存在: {gallery_dir}")
        competition_gallery_names = []

    cache_manager = TTACacheManager(cache_dir=TTA_CACHE_DIR)
    cache_info = cache_manager.get_cache_info(MODEL_PATH, BEST_STRATEGY)

    if not cache_info.get(BEST_STRATEGY, {}).get('exists', False):
        print(f"[WARNING] TTA缓存不存在，请先运行预计算脚本")
        return None

    fused_features = cache_manager.get_fused_features(MODEL_PATH, BEST_STRATEGY)

    if fused_features is None:
        print("[ERROR] TTA缓存加载失败")
        return None

    if competition_model is not None:
        device = next(competition_model.parameters()).device
        fused_features = fused_features.to(device)

        if next(competition_model.parameters()).dtype == torch.float16:
            fused_features = fused_features.half()

    competition_gallery_features = fused_features

    print(f"[OK] TTA缓存: {BEST_STRATEGY}, shape={fused_features.shape}")

    return fused_features


def warmup_model():
    print("[INFO] 预热模型...")

    if competition_model is None:
        print("[WARNING] 模型未加载，跳过预热")
        return

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if next(competition_model.parameters()).dtype == torch.float16 else torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        if hasattr(cfg, 'INPUT') and hasattr(cfg.INPUT, 'SIZE_TEST'):
            img_height = cfg.INPUT.SIZE_TEST[0]
            img_width = cfg.INPUT.SIZE_TEST[1]
            input_size = (1, 3, img_height, img_width)
        else:
            input_size = (1, 3, 384, 128)

        dummy_input = torch.randn(input_size, device=device, dtype=dtype)

        with torch.no_grad():
            for _ in range(10):
                _ = competition_model(dummy_input, cam_label=0, view_label=0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        print(f"[OK] 模型预热完成")

    except Exception as e:
        print(f"[WARNING] 模型预热失败: {e}")


def preprocess_image(image_data: bytes):
    try:
        img = Image.open(io.BytesIO(image_data))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if hasattr(cfg, 'INPUT') and hasattr(cfg.INPUT, 'SIZE_TEST'):
            target_size = (cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0])
        else:
            target_size = (128, 384)

        img = img.resize(target_size, Image.BILINEAR)

        img_array = np.array(img).transpose(2, 0, 1)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)

        img_tensor = img_tensor / 255.0
        if hasattr(cfg, 'INPUT') and hasattr(cfg.INPUT, 'PIXEL_MEAN'):
            mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        if hasattr(cfg, 'INPUT') and hasattr(cfg.INPUT, 'PIXEL_STD'):
            std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)
        else:
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        img_tensor = img_tensor.unsqueeze(0)

        if competition_model is not None:
            device = next(competition_model.parameters()).device
            dtype = next(competition_model.parameters()).dtype
            img_tensor = img_tensor.to(device=device, dtype=dtype)

        return img_tensor

    except Exception as e:
        print(f"图像预处理失败: {e}")
        raise HTTPException(status_code=400, detail=f"图像处理失败: {str(e)}")


def extract_features_with_tta(image_tensor: torch.Tensor):
    if competition_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        cache_manager = TTACacheManager(cache_dir=TTA_CACHE_DIR)
        strategy_config = cache_manager.TTA_STRATEGIES.get(BEST_STRATEGY)

        if not strategy_config:
            raise HTTPException(status_code=400, detail=f"未知TTA策略: {BEST_STRATEGY}")

        aug_types = strategy_config['augmentations']
        weights = strategy_config['weights']

        features_list = []

        with torch.no_grad():
            for aug_type, weight in zip(aug_types, weights):
                aug_img = cache_manager.apply_augmentation(image_tensor, aug_type)
                features = competition_model(aug_img, cam_label=0, view_label=0)
                features_list.append(features * weight)

        if features_list:
            fused_features = torch.sum(torch.stack(features_list), dim=0)
        else:
            with torch.no_grad():
                fused_features = competition_model(image_tensor, cam_label=0, view_label=0)

        return fused_features

    except Exception as e:
        print(f"特征提取失败: {e}")
        raise HTTPException(status_code=500, detail=f"特征提取失败: {str(e)}")


@app.get("/")
async def root():
    try:
        with open("fin_frontend.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        uptime = time.time() - service_start_time if service_start_time else 0
        return {
            "service": "TransReID",
            "status": "running",
            "uptime_seconds": round(uptime, 2),
            "model": model_metadata.get("model_name", "unknown"),
            "endpoints": {
                "health": "/health",
                "extract": "/extract_features",
                "search": "/search",
                "competition": "/competition_search",
                "info": "/model_info"
            }
        }


def startup_event():
    global service_start_time

    print("=" * 50)
    print("[启动] TransReID比赛API服务器")
    print("=" * 50)

    service_start_time = time.time()

    model = load_competition_model()
    if model is None:
        print("[ERROR] 模型加载失败")
        return

    cache_features = load_tta_cache()
    if cache_features is None:
        print("[WARNING] TTA缓存未加载")

    warmup_model()

    print("-" * 50)
    print(f"[OK] Server ready at http://127.0.0.1:8002")
    print("=" * 50)


@app.on_event("startup")
async def lifespan_startup():
    startup_event()


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")


@app.get("/frontend")
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend page not found</h1>", status_code=404)


@app.get("/health")
async def health_check():
    gpu_info = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**2
        gpu_info = {
            "name": gpu_name,
            "memory_total_mb": round(gpu_memory_total, 1),
            "memory_used_mb": round(gpu_memory_used, 1),
            "memory_free_mb": round(gpu_memory_total - gpu_memory_used, 1)
        }

    cache_info = None
    if competition_gallery_features is not None:
        cache_info = {
            "shape": list(competition_gallery_features.shape),
            "dtype": str(competition_gallery_features.dtype),
            "device": str(competition_gallery_features.device),
            "strategy": BEST_STRATEGY
        }

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": competition_model is not None,
        "model_name": model_metadata.get("model_name", "unknown"),
        "config_file": model_metadata.get("config_file", "unknown"),
        "cache_loaded": competition_gallery_features is not None,
        "cache_info": cache_info,
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info
    }


@app.get("/model_info")
async def get_model_info():
    feature_dim = competition_gallery_features.shape[1] if competition_gallery_features is not None else 768

    return {
        "model": {
            "name": model_metadata.get("model_name", "unknown"),
            "feature_dim": feature_dim,
            "config_file": model_metadata.get("config_file", ""),
            "device": model_metadata.get("device", ""),
            "precision": model_metadata.get("precision", ""),
        },
        "tta_cache": {
            "strategy": BEST_STRATEGY,
            "loaded": competition_gallery_features is not None,
            "shape": competition_gallery_features.shape if competition_gallery_features is not None else None
        }
    }


@app.post("/extract_features")
async def extract_features(image: UploadFile = File(...)):
    start_time = time.time()

    image_data = await image.read()

    preprocess_time = time.time()
    img_tensor = preprocess_image(image_data)
    preprocess_ms = (time.time() - preprocess_time) * 1000

    extract_time = time.time()
    features = extract_features_with_tta(img_tensor)
    extract_ms = (time.time() - extract_time) * 1000

    features_np = features.cpu().numpy()

    total_ms = (time.time() - start_time) * 1000

    return {
        "features": features_np.tolist()[0],
        "time_ms": {
            "total": round(total_ms, 2),
            "preprocess": round(preprocess_ms, 2),
            "extraction": round(extract_ms, 2)
        },
        "feature_dim": features_np.shape[1],
        "strategy": BEST_STRATEGY,
        "augmentations": ['original', 'flip', 'crop_0.05', 'crop_0.10'],
        "weights": [0.4, 0.3, 0.15, 0.15]
    }


@app.post("/search")
async def search(image: UploadFile = File(...), k: int = Form(10)):
    print(f"[INFO] Search request, k={k}")
    start_time = time.time()

    if competition_gallery_features is None:
        raise HTTPException(status_code=500, detail="TTA缓存未加载，请先运行预计算脚本")

    extract_response = await extract_features(image)
    query_features = torch.tensor([extract_response["features"]],
                                 dtype=competition_gallery_features.dtype,
                                 device=competition_gallery_features.device)

    match_time = time.time()

    query_norm = torch.nn.functional.normalize(query_features, p=2, dim=1)
    gallery_norm = torch.nn.functional.normalize(competition_gallery_features, p=2, dim=1)

    similarity = torch.mm(query_norm, gallery_norm.t())

    _, indices = similarity.topk(k, dim=1)

    match_ms = (time.time() - match_time) * 1000

    indices_list = indices[0].cpu().numpy().tolist()
    similarity_list = similarity[0][indices[0]].cpu().numpy().tolist()

    total_ms = (time.time() - start_time) * 1000

    return {
        "matches": [
            {
                "rank": i + 1,
                "gallery_index": idx,
                "filename": competition_gallery_names[idx] if idx < len(competition_gallery_names) else f"image_{idx}.jpg",
                "similarity": round(sim, 4)
            }
            for i, (idx, sim) in enumerate(zip(indices_list, similarity_list))
        ],
        "time_ms": {
            "total": round(total_ms, 2),
            "feature_extraction": extract_response["time_ms"]["total"],
            "query_matching": round(match_ms, 2)
        },
        "strategy": BEST_STRATEGY,
        "cache_used": True,
        "k": k,
        "total_gallery": competition_gallery_features.shape[0]
    }


@app.post("/competition_search")
async def competition_search(image: UploadFile = File(...), k: int = Form(10)):
    start_time = time.time()

    search_result = await search(image, k)

    total_ms = (time.time() - start_time) * 1000
    matches_full = search_result["matches"]

    return {
        "competition": True,
        "model": Path(MODEL_PATH).stem,
        "strategy": BEST_STRATEGY,
        "matches": matches_full,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "k": k,
            "total_gallery": search_result["total_gallery"]
        }
    }


@app.post("/benchmark")
async def benchmark_test(image: UploadFile = File(...), iterations: int = Form(10)):
    print(f"[INFO] Benchmark: {iterations} iterations")

    image_data = await image.read()

    times = {
        "preprocess": [],
        "extraction": [],
        "matching": [],
        "total": []
    }

    for i in range(iterations):
        iter_start = time.time()

        preprocess_start = time.time()
        img_tensor = preprocess_image(image_data)
        times["preprocess"].append((time.time() - preprocess_start) * 1000)

        extract_start = time.time()
        features = extract_features_with_tta(img_tensor)
        times["extraction"].append((time.time() - extract_start) * 1000)

        if competition_gallery_features is not None:
            match_start = time.time()
            query_norm = torch.nn.functional.normalize(features, p=2, dim=1)
            gallery_norm = torch.nn.functional.normalize(competition_gallery_features, p=2, dim=1)
            similarity = torch.mm(query_norm, gallery_norm.t())
            _, _ = similarity.topk(10, dim=1)
            times["matching"].append((time.time() - match_start) * 1000)

        times["total"].append((time.time() - iter_start) * 1000)

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{iterations}")

    def get_stats(values):
        if not values:
            return None
        return {
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "avg": round(sum(values) / len(values), 2),
            "std": round(np.std(values) if len(values) > 1 else 0, 2)
        }

    return {
        "iterations": iterations,
        "results": {
            "preprocess_ms": get_stats(times["preprocess"]),
            "extraction_ms": get_stats(times["extraction"]),
            "matching_ms": get_stats(times["matching"]) if times["matching"] else None,
            "total_ms": get_stats(times["total"])
        },
        "status": "completed"
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("TransReID 比赛服务器")
    print("=" * 50)

    required_files = [
        MODEL_CONFIG,
        MODEL_PATH,
        "tta_cache_precomputing.py"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("[ERROR] 缺少必要文件:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)

    available_port = find_available_port(8002)

    if available_port is None:
        print("[ERROR] 端口 8002-8011 都被占用")
        sys.exit(1)

    if available_port != 8002:
        print(f"[INFO] 端口 8002 被占用，自动切换到: {available_port}")

    print(f"[INFO] Server: http://127.0.0.1:{available_port}")
    print("=" * 50)

    uvicorn.run(
        "fin_server:app",
        host="127.0.0.1",
        port=available_port,
        reload=False,
        workers=1
    )
