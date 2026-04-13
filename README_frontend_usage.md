# 🌐 TransReID比赛服务器前端使用指南

> 📅 更新时间: 2026-04-13

---

## 🚀 快速启动

### ▶️ 启动比赛服务器

```bash
python fin_server.py
```

服务器会自动检测可用端口（8002-8011）并启动。

### 🌐 访问地址

启动后访问: `http://localhost:8002` (端口可能因占用而变化，请查看控制台输出)

---

## 📱 前端功能说明

### 1️⃣ 界面布局

- **📊 左侧面板**: 服务器信息、缓存状态、模型状态
- **📤 上传区域**: 拖放或点击上传查询图片
- **⚙️ 控制区域**: 设置Top-K参数
- **📋 结果区域**: 显示检索结果和性能指标

### 2️⃣ 🔍 搜索功能

1. **📤 上传图片**: 拖放或点击选择查询图片
2. **⚙️ 设置参数**: Top-K结果数量（默认10）
3. **▶️ 开始搜索**: 点击"开搜！！！"按钮
4. **📋 查看结果**: 按相似度排序的匹配结果

### 3️⃣ ⚡ 性能指标

- **⏱️ 特征提取用时**: 目标 <=40ms
- **⏱️ 查询匹配用时**: 目标 <=30ms
- **📊 匹配结果数量**: 显示Top-K结果

### 4️⃣ 💾 服务器状态

- **💾 缓存**: TTA缓存是否已加载
- **🎲 TTA策略**: 当前使用的TTA策略
- **🧠 模型**: 当前加载的模型信息

---

## 🔌 API接口说明

### 🏆 比赛服务器接口 (fin_server.py - 端口8002)

| 端点 | 方法 | 说明 | 返回格式 |
|------|------|------|----------|
| `/` | GET | 前端页面 | HTML |
| `/health` | GET | 健康检查 | JSON |
| `/model_info` | GET | 模型信息 | JSON |
| `/extract_features` | POST | 特征提取 | JSON |
| `/search` | POST | 标准搜索 | JSON |
| `/competition_search` | POST | 比赛模式搜索 | JSON |
| `/benchmark` | POST | 性能基准测试 | JSON |

### 📁 静态文件服务

- `GET /static/{path}` - 访问数据目录中的图片

### 📖 接口详情

#### 1️⃣ Health Check

```bash
GET /health
```

返回示例：
```json
{
  "status": "healthy",
  "timestamp": "2026-04-13T10:30:00",
  "model_loaded": true,
  "model_name": "transformer_checkpoint_177.pth",
  "config_file": "rtx4090_from_final153model.yml",
  "cache_loaded": true,
  "gpu_available": true,
  "gpu_info": {
    "name": "NVIDIA RTX 4090",
    "memory_total_mb": 24576,
    "memory_used_mb": 2048,
    "memory_free_mb": 22528
  }
}
```

#### 2️⃣ Model Info

```bash
GET /model_info
```

返回示例：
```json
{
  "model": {
    "name": "transformer_checkpoint_177.pth",
    "feature_dim": 768,
    "config_file": "rtx4090_from_final153model.yml",
    "device": "cuda",
    "precision": "fp16"
  },
  "tta_cache": {
    "strategy": "multi_scale_crop",
    "loaded": true,
    "shape": [4858, 768]
  }
}
```

#### 3️⃣ Extract Features

```bash
POST /extract_features
Content-Type: multipart/form-data

image: <binary_image_data>
```

返回示例：
```json
{
  "features": [0.123, -0.456, ...],  // 768维特征向量
  "time_ms": {
    "total": 31.5,
    "preprocess": 2.1,
    "extraction": 29.4
  },
  "feature_dim": 768,
  "strategy": "multi_scale_crop",
  "augmentations": ["original", "flip", "crop_0.05", "crop_0.10"],
  "weights": [0.4, 0.3, 0.15, 0.15]
}
```

#### 4️⃣ Search

```bash
POST /search
Content-Type: multipart/form-data

image: <binary_image_data>
k: 10  // Top-K结果数量
```

返回示例：
```json
{
  "matches": [
    {
      "rank": 1,
      "gallery_index": 1234,
      "filename": "0001_c1s1_000001_00.jpg",
      "similarity": 0.9876
    },
    ...
  ],
  "time_ms": {
    "total": 32.1,
    "feature_extraction": 31.5,
    "query_matching": 0.6
  },
  "strategy": "multi_scale_crop",
  "cache_used": true,
  "k": 10,
  "total_gallery": 4858
}
```

#### 5️⃣ Competition Search

```bash
POST /competition_search
Content-Type: multipart/form-data

image: <binary_image_data>
k: 10
```

返回示例：
```json
{
  "competition": true,
  "model": "transformer_checkpoint_177",
  "strategy": "multi_scale_crop",
  "matches": [...],
  "metadata": {
    "timestamp": "2026-04-13T10:30:00",
    "k": 10,
    "total_gallery": 4858
  }
}
```

#### 6️⃣ Benchmark

```bash
POST /benchmark
Content-Type: multipart/form-data

image: <binary_image_data>
iterations: 10
```

返回示例：
```json
{
  "iterations": 10,
  "results": {
    "preprocess_ms": {"min": 1.8, "max": 2.5, "avg": 2.1, "std": 0.2},
    "extraction_ms": {"min": 28.5, "max": 32.1, "avg": 30.2, "std": 1.1},
    "matching_ms": {"min": 0.4, "max": 0.8, "avg": 0.6, "std": 0.1},
    "total_ms": {"min": 30.7, "max": 35.4, "avg": 32.9, "std": 1.3}
  },
  "status": "completed"
}
```

---

## ❓ 常见问题

### Q1: 访问 http://localhost:8002 显示JSON而不是网页

**⚠️ 原因**: 服务器启动失败或前端文件缺失

**✅ 解决**: 
- 检查控制台输出，确认服务器成功启动在哪个端口
- 确认 `fin_frontend.html` 文件存在于项目根目录

### Q2: 图片无法显示

**⚠️ 原因**: 静态文件服务路径不正确或图片不存在

**✅ 解决**: 
- 检查`data/BallShow/bounding_box_test`目录是否存在
- 确认图片文件存在且命名正确

### Q3: 搜索返回错误

**⚠️ 原因**: TTA缓存未预计算

**✅ 解决**: 先运行预计算脚本:

```bash
python tta_cache_precomputing.py \
  --config configs/BallShow/rtx4090_from_final153model.yml \
  --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy multi_scale_crop
```

### Q4: 端口被占用

**✅ 解决**: 服务器会自动切换到下一个可用端口（8003-8011），或修改 `fin_server.py` 中的 `start_port` 参数

### Q5: 特征提取时间过长

**⚠️ 原因**: TTA包含4次推理增强

**✅ 解决**: 
- 可切换到`flip_only`策略（仅2次推理）
- 检查GPU是否正常工作
- 确认使用了FP16半精度推理

---

## 📁 文件清单

### 📋 必需文件

```
fin_server.py                      # 🏆 比赛专用服务器
fin_frontend.html                  # 🌐 比赛前端页面
tta_cache_precomputing.py          # 💾 TTA缓存预计算脚本
configs/BallShow/rtx4090_from_final153model.yml    # ⚙️ 配置文件
logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth  # 🧠 模型权重
data/BallShow/                     # 📁 数据目录
tta_cache/                         # 💾 TTA缓存目录
```

### 📄 可选文件

```
README_frontend_usage.md           # 📖 本使用指南
```

---

## 📊 性能指标

### 🎯 精度指标

| 指标 | 数值 | 赛题要求 | 状态 |
|------|------|---------|------|
| mAP | 91.50% | >=91.5% | ✅ 达标 |
| Rank-1 | 94.40% | >=94.0% | ✅ 达标 |
| Rank-5 | 97.97% | - | - |
| Rank-10 | 99.17% | - | - |

### ⚡ 速度指标

| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| 特征提取 | ~31ms | <=40ms | ✅ 达标 |
| 查询匹配 | ~0.4ms | <=30ms | ✅ 达标 |
| 总响应时间 | <=33ms | - | ✅ 达标 |
| 特征维度 | 768维 | - | - |

---

## 🎲 TTA策略

### 📋 可用策略

| 策略 | 增强数量 | 速度 | 精度提升 |
|------|---------|------|---------|
| `flip_only` | 2次 | ⚡ 最快 | +0.1~0.3% |
| `flip_weighted` | 2次 | ⚡ 最快 | +0.1~0.3% |
| `multi_scale` | 4次 | 🚀 中等 | +0.2~0.5% |
| `multi_scale_crop` | 4次 | 🚀 中等 | +0.2~0.5% |

### ⚙️ 当前配置

比赛服务器默认使用 `multi_scale_crop` 策略：
- 📷 原始图像 (权重 0.4)
- 🔄 水平翻转 (权重 0.3)
- ✂️ 裁剪5% (权重 0.15)
- ✂️ 裁剪10% (权重 0.15)

---

## 🛠️ 技术规格

### 🎨 前端技术

- 纯HTML/CSS/JavaScript，无外部依赖
- 响应式设计，支持移动端
- 现代UI设计，深色主题
- 使用Tailwind CSS进行样式管理

### ⚙️ 服务器技术

- FastAPI + Uvicorn
- PyTorch半精度推理 (FP16)
- TTA缓存优化
- 静态文件服务
- 自动端口检测

---

## 📚 相关文档

- [README_main.md](README_main.md) - 📖 项目主文档
- [README_test_time_augmentation.md](README_test_time_augmentation.md) - 🎲 TTA完整指南
- [README_reranking.md](README_reranking.md) - 🔄 Re-ranking分析

---

*🏆 TransReID比赛系统 - 版本 2.0.0*

*📝 最后更新：2026-04-13*
