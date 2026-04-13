# 🧪 Test-Time Augmentation (TTA) 完整指南

> 📅 更新时间: 2026-04-13

---

## ⚠️ 重要警告

**🚨 TTA缓存和查询图片必须使用相同的增强策略和权重！！！**

如果底库使用 `multi_scale_crop` 策略（4种增强），查询时也必须使用同样的 `multi_scale_crop` 策略。

**❌ 两边增强策略必须完全一致，否则特征空间不匹配，匹配结果会严重下降！！！**

---

## 📖 概述

**🧪 Test-Time Augmentation (TTA)** 是一种在推理阶段对测试图像进行增强的技术。通过对同一张图像应用多种变换（如翻转、缩放、裁剪），分别提取特征后进行融合，可以提升预测的稳定性和准确性。

### ✅ TTA 的优势

| ✅ 优势 | 📝 说明 |
|:------|:------|
| **📈 提升精度** | 多次推理取平均，减少单次预测的方差 |
| **🚫 无需训练** | 只在测试时使用，不需要重新训练模型 |
| **🔧 兼容性高** | 适用于任何训练好的模型（ViT、ResNet等） |
| **⏱️ 开销可控** | 可选择不同的增强策略平衡效果和速度 |

---

## 🎯 TTA策略详解

### 📋 可用策略一览

| ⚙️ 策略 | 🎨 增强类型 | ⚖️ 权重 | 🔢 增强数量 | 📈 预期提升 | 🎯 推荐场景 |
|:------|:---------|:------|:--------:|:--------:|:---------|
| `flip_only` | 原始 + 水平翻转 | [0.5, 0.5] | 2 | +0.1~0.3% | ⏱️ 追求速度 |
| `flip_weighted` | 原始 + 水平翻转 | [0.7, 0.3] | 2 | +0.1~0.3% | ⚖️ 平衡时间与效果 |
| `multi_scale` | 原始 + 翻转 + 缩放0.95 + 缩放1.05 | [0.4, 0.3, 0.15, 0.15] | 4 | +0.2~0.5% | 📈 追求效果 |
| `multi_scale_crop` | 原始 + 翻转 + 裁剪5% + 裁剪10% | [0.4, 0.3, 0.15, 0.15] | 4 | +0.2~0.5% | ⭐ **最佳策略** |

### 1️⃣ flip_only（仅水平翻转）

```python
🧪 TTA_STRATEGY = {
    'augmentations': ['original', 'flip'],
    'weights': [0.5, 0.5],
}
```

**💡 原理**：水平翻转后左右对称性保持，特征基本不变，融合可减少噪声。

**⚡ 速度**：最快  
**📈 效果**：基础

### 2️⃣ flip_weighted（水平翻转加权）

```python
🧪 TTA_STRATEGY = {
    'augmentations': ['original', 'flip'],
    'weights': [0.7, 0.3],
}
```

**💡 原理**：原始图像权重更高，因为原始特征更可靠。

**⚡ 速度**：最快  
**📈 效果**：基础

### 3️⃣ multi_scale（多尺度）

```python
🧪 TTA_STRATEGY = {
    'augmentations': ['original', 'flip', 'scale_0.95', 'scale_1.05'],
    'weights': [0.4, 0.3, 0.15, 0.15],
}
```

**💡 原理**：不同尺度提取不同粒度的特征，大尺度保留整体，小尺度聚焦局部。

**⚡ 速度**：中等  
**📈 效果**：较好

### 4️⃣ multi_scale_crop（多尺度裁剪增强）⭐ 推荐

```python
🧪 TTA_STRATEGY = {
    'augmentations': ['original', 'flip', 'crop_0.05', 'crop_0.10'],
    'weights': [0.4, 0.3, 0.15, 0.15],
}
```

**💡 原理**：模拟不同距离拍摄的图像，裁剪中心区域强制模型关注主体。

**⚡ 速度**：中等  
**📈 效果**：最佳

---

## ⚙️ 实现原理

### 🧠 核心代码流程

```python
def extract_features_with_tta(model, images, ...):
    # 1️⃣ 对每种增强类型分别推理
    for aug_type in aug_list:
        aug_images = apply_augmentation(images, aug_type)
        feat = model(aug_images, ...)  # [B, D]
        features_list.append(feat)
    
    # 2️⃣ L2归一化
    features_list = [normalize(f, p=2, dim=1) for f in features_list]
    
    # 3️⃣ 加权融合
    weights_tensor = torch.tensor(weights).view(-1, 1, 1)  # [N, 1, 1]
    fused = (torch.stack(features_list, dim=0) * weights_tensor).sum(dim=0)
    
    # 4️⃣ 再次归一化
    fused = normalize(fused, p=2, dim=1)
    
    return fused  # [B, D]
```

### 📐 张量形状变化

```
原始特征:      [B, D]          # B=batch, D=特征维度(768)
增强后特征:    [N, B, D]        # N=增强数量
权重:          [N, 1, 1]        # 自动广播
加权融合:      [B, D]          # 输出
```

### 🎨 增强类型详解

| 🎨 类型 | ⚙️ 实现方式 | 📈 效果 |
|:------|:---------|:------|
| `original` | 不做任何变换 | 基准 |
| `flip` | `torch.flip(images, dims=[3])` | 水平翻转 |
| `scale_0.95` | 缩小5%后插值回原尺寸 | 聚焦中心 |
| `scale_1.05` | 扩大5%后裁剪中心 | 减少边缘 |
| `crop_0.05` | 裁剪5%边缘后插值 | 去除边界干扰 |
| `crop_0.10` | 裁剪10%边缘后插值 | 更极端的裁剪 |

---

## 🔄 TTA完整工作流程

### 🔄 四阶段示意图

```
📍 阶段1: 预计算时（一次性）
🐍 运行 tta_cache_precomputing.py

🖼️ 底库图片1 → 🎨 original → 🔍 提取特征 ──┐
         → 🔄 flip ─────→ 🔍 提取特征 ──┼──→ 💾 全部保存到 .npz 文件
         → ✂️ crop_0.05 ─→ 🔍 提取特征 ──┤
         → ✂️ crop_0.10 ─→ 🔍 提取特征 ──┘
                    ↓
📍 阶段2: 服务器启动时（一次性）
🐍 运行 fin_server.py

💾 从 .npz 加载 4 种增强的特征 → ⚖️ 加权融合 → 🔍 底库特征向量 (4858, 768)
                    ↓
📍 阶段3: 查询时（每次查询）
👤 用户发送查询图片

🖼️ 查询图片 → 🎨 original → 🧠 模型推理 ──┐
        → 🔄 flip ─────→ 🧠 模型推理 ──┼──→ ⚖️ 加权融合 → 🔍 查询特征向量
        → ✂️ crop_0.05 ─→ 🧠 模型推理 ──┤
        → ✂️ crop_0.10 ─→ 🧠 模型推理 ──┘
                    ↓
📍 阶段4: 匹配阶段

🔍 查询特征 (1, 768)  ──┐
                     ├──→ 📊 余弦相似度计算 → 📋 排序 → 🏆 返回 Top-K
🔍 底库特征 (4858, 768) ┘
```

### 💾 .npz 文件内容

`.npz` 文件中保存了**每种增强的原始特征**（不是融合后的）：

```
💾 tta_cache/*.npz 文件内容：
┌──────────────────────────────────────────────────┐
│  '🎨 original'   → 原始图片的特征 (4858, 768)        │
│  '🔄 flip'       → 翻转图片的特征 (4858, 768)        │
│  '✂️ crop_0.05'  → 裁剪5%图片的特征 (4858, 768)     │
│  '✂️ crop_0.10'  → 裁剪10%图片的特征 (4858, 768)    │
│  '⚖️ weights'    → [0.4, 0.3, 0.15, 0.15]           │
│  '📋 augmentations' → ['original', 'flip', ...]      │
│  '📄 metadata'   → 各种元数据                        │
└──────────────────────────────────────────────────┘
```

### ❓ 为什么查询时也要做TTA？

因为**底库和查询的特征空间必须一致**才能正确匹配：

| 📝 组件 | 🧪 TTA操作 | ✅ 是否需要 |
|:------|:--------|:--------:|
| 💾 底库 .npz 缓存 | 预计算特征并保存 | 一次性 |
| 🖼️ 查询图片 | 实时增强和推理 | 每次查询 |

**🔑 关键点**：
- ✅ 底库用 `multi_scale_crop` → `.npz` 里存的就是这种策略的特征
- ✅ 查询也必须用 `multi_scale_crop` → 才能与底库正确匹配
- ⚠️ 增强策略必须两边一致，否则特征空间不对齐

### ⏱️ 性能影响

| 🔢 增强数量 | ⏱️ 单次查询时间 | 📈 准确率影响 |
|:---------|:------------|:----------|
| 4种 (original+flip+crop_0.05+crop_0.10) | ~100ms | 最佳 |
| 2种 (original+flip) | ~50ms | 略有下降 |

---

## 💾 TTA缓存生成器

### tta_cache_precomputing.py（推荐）

**✅ 特点:**
- ✅ 支持完整的命令行参数
- ✅ 可自定义配置、模型路径、策略等
- ✅ 支持强制覆盖已有缓存
- ✅ 支持详细日志输出

**🚀 使用方式:**

```bash
🐍 python tta_cache_precomputing.py \
  --config configs/BallShow/rtx4090_from_final153model.yml \
  --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy multi_scale_crop
```

**🔄 强制覆盖已有缓存:**

```bash
🐍 python tta_cache_precomputing.py \
  --config configs/BallShow/rtx4090_from_final153model.yml \
  --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy multi_scale_crop \
  --force
```

### ⚙️ 参数说明

| ⚙️ 参数 | 🔤 简写 | ✅ 必需 | 📍 默认值 | 📝 说明 |
|:------|:------|:------|:--------|:------|
| `--config` | `-c` | ✅ 是 | - | ⚙️ 配置文件路径 |
| `--model` | `-m` | ✅ 是 | - | 🧠 模型权重文件路径 |
| `--strategy` | `-s` | ❌ 否 | multi_scale_crop | 🧪 TTA策略 |
| `--output` | `-o` | ❌ 否 | tta_cache | 📂 输出目录 |
| `--force` | `-f` | ❌ 否 | False | 🔄 强制覆盖已有缓存 |

### 📁 输出文件

缓存生成后保存在 `tta_cache/` 目录：

```
📂 tta_cache/
├── 💾 transformer_checkpoint_177_multi_scale_crop.npz  # 缓存文件
└── 📄 tta_cache_precomputing_log.txt                    # 日志文件
```

---

## 🧪 TTA测试脚本

### tta_test.py

**🎯 功能**：测试不同TTA策略的效果

**🚀 使用方式:**

```bash
# 📊 测试所有策略
🐍 python tta_test.py \
  --config_file configs/BallShow/rtx4090_from_final153model.yml \
  --weight logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy all

# 🎯 测试单个策略
🐍 python tta_test.py \
  --config_file configs/BallShow/rtx4090_from_final153model.yml \
  --weight logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy multi_scale_crop
```

### ⚙️ 参数说明

| ⚙️ 参数 | ✅ 必需 | 📍 默认值 | 📝 说明 |
|:------|:------|:--------|:------|
| `--config_file` | ✅ 是 | - | ⚙️ 配置文件路径 |
| `--weight` | ✅ 是 | - | 🧠 权重文件路径 |
| `--strategy` | ❌ 否 | all | 🧪 TTA策略: all/flip_only/flip_weighted/multi_scale/multi_scale_crop |
| `--reranking` | ❌ 否 | False | 🔄 使用re-ranking |
| `--no_tta` | ❌ 否 | False | 🚫 不使用TTA（仅对比实验） |

---

## 🏆 当前最佳配置

### fin_server.py 配置

```python
# 🥇 最佳模型配置
📄 MODEL_CONFIG = "configs/BallShow/rtx4090_from_final153model.yml"
💾 MODEL_PATH = "logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth"

# ⭐ 最佳TTA策略
🧪 BEST_STRATEGY = "multi_scale_crop"
```

### 📊 精度结果

| 📊 指标 | 📈 数值 | 🎯 赛题要求 | ✅ 状态 |
|:------|:----:|:--------:|:----:|
| **🎯 mAP** | 91.5019% | >=91.5% | ✅ 达标 |
| **🥇 Rank-1** | 94.3981% | >=94.0% | ✅ 达标 |
| **🥈 Rank-5** | 97.9738% | - | - |
| **🥉 Rank-10** | 99.1657% | - | - |

### ⚡ 性能基准

| 📊 指标 | 📈 数值 | 🎯 目标 | ✅ 状态 |
|:------|:----:|:----:|:----:|
| **⚡ 特征提取时间** | ~31ms | <=40ms | ✅ 达标 |
| **⚡ 查询匹配时间** | ~0.4ms | <=30ms | ✅ 达标 |
| **⏱️ 总响应时间** | <=33ms | - | ✅ 达标 |

---

## 🚀 快速开始

### 🔄 完整流程

```bash
# 1️⃣ 💾 生成TTA缓存
🐍 python tta_cache_precomputing.py \
  --config configs/BallShow/rtx4090_from_final153model.yml \
  --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy multi_scale_crop

# 2️⃣ 🧪 TTA测试
🐍 python tta_test.py \
  --config_file configs/BallShow/rtx4090_from_final153model.yml \
  --weight logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy all

# 3️⃣ 🚀 启动比赛服务器
🐍 python fin_server.py
```

### 🚀 启动比赛服务器

生成缓存后，启动服务器：

```bash
🐍 python fin_server.py
```

服务器会自动：
1. ✅ 检测TTA缓存目录
2. ✅ 加载预计算的Gallery特征
3. ✅ 启动API服务（端口8002）

---

## ❓ 常见问题

### ❓ Q1: 报错 "weights数量与augmentations数量不匹配"

**🔍 原因**：TTA策略配置中 `weights` 数量与 `augmentations` 数量不一致。

**💡 解决**：检查 `TTA_STRATEGIES` 字典中的配置，确保两者数量一致。

### ❓ Q2: 配置文件不存在

**🔍 检查**: `configs/BallShow/rtx4090_from_final153model.yml`

### ❓ Q3: 模型文件不存在

**🔍 检查**: `logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth`

### ❓ Q4: GPU内存不足

- 💡 尝试减少batch size
- 💡 或使用CPU模式（自动降级）

### ❓ Q5: 服务器启动失败

- 💡 检查TTA缓存是否已生成
- 💡 确认端口未被占用（默认8002）

---

## 🔧 适配不同模型

### 🤖 自动适配机制

脚本已内置**动态适配**功能：

1. ✅ **权重数量检查** - 自动验证 `weights` 数量与 `augmentations` 数量匹配
2. ✅ **特征维度检查** - 从实际模型输出动态获取特征维度
3. ✅ **兼容所有模型** - 无论 ViT、ResNet 或任何 TransReID 变体

```python
# 🤖 自动获取特征维度（无需手动指定）
feat_dim = features_list[0].shape[1]  # 运行时推断
```

### ✅ 适用于

| 🧠 模型类型 | ✅ 支持情况 |
|:---------|:--------:|
| ViT-Base | ✅ 完全支持 |
| ViT-Large | ✅ 完全支持 |
| ResNet50 | ✅ 完全支持 |
| 任意自定义模型 | ✅ 只需返回特征张量 |

---

## 📚 参考资料

- 📄 TransReID 论文: [TransReID: Transformer-Based Object Re-Identification](https://arxiv.org/abs/2104.03349)
- 📄 TTA 技术: [Test-Time Augmentation for Object Detection](https://arxiv.org/abs/1903.05842)
- 🔗 官方代码: https://github.com/damo-cv/TransReID

---

## 📚 相关文档

- [README_main.md](README_main.md) - 📖 项目主文档
- [README_frontend_usage.md](README_frontend_usage.md) - 🌐 前端使用指南
- [README_reranking.md](README_reranking.md) - 🔄 Re-ranking分析

---

*📝 最后更新：2026-04-13*
