# Test-Time Augmentation (TTA) 测试指南

## 什么是 Test-Time Augmentation？

**Test-Time Augmentation (TTA)** 是一种在推理阶段对测试图像进行增强的技术。通过对同一张图像应用多种变换（如翻转、缩放、裁剪），分别提取特征后进行融合，可以提升预测的稳定性和准确性。

### TTA 的优势

| 优势 | 说明 |
|:-----|:-----|
| **提升精度** | 多次推理取平均，减少单次预测的方差 |
| **无需训练** | 只在测试时使用，不需要重新训练模型 |
| **兼容性高** | 适用于任何训练好的模型（ViT、ResNet等） |
| **开销可控** | 可选择不同的增强策略平衡效果和速度 |

---

## 快速开始

### 基础命令

```bash
python test_tta.py --config_file configs/BallShow/rtx4090_final.yml \
                   --weight logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth \
                   --strategy all
```

### 参数说明

| 参数 | 说明 | 示例 |
|:-----|:-----|:-----|
| `--config_file` | 配置文件路径 | `configs/BallShow/rtx4090_final.yml` |
| `--weight` | 模型权重文件路径 | `logs/.../transformer_checkpoint_153.pth` |
| `--strategy` | TTA策略选择 | `all` / `flip_only` / `flip_weighted` / `multi_scale` / `multi_scale_crop` |
| `--reranking` | 可选，启用re-ranking后处理 | 添加 `--reranking` |

### 可用策略

| 策略 | 说明 | 增强数量 | 预期提升 |
|:-----|:-----|:--------:|:--------:|
| `flip_only` | 原始 + 水平翻转，等权重 | 2 | +0.1~0.3% |
| `flip_weighted` | 原始(0.7) + 翻转(0.3) | 2 | +0.1~0.3% |
| `multi_scale` | 原始 + 翻转 + 缩放0.95 + 缩放1.05 | 4 | +0.2~0.5% |
| `multi_scale_crop` | 原始 + 翻转 + 裁剪5% + 裁剪10% | 4 | +0.2~0.5% |
| `all` | 测试所有策略 | - | - |

---

## 增强策略详解

### 1. flip_only（仅水平翻转）

```python
'flip_only': {
    'augmentations': ['original', 'flip'],
    'weights': [0.5, 0.5]
}
```

**原理**：水平翻转后左右对称性保持，特征基本不变，融合可减少噪声。

**速度**：⭐⭐⭐⭐⭐ 最快
**效果**：⭐⭐⭐ 基础

### 2. flip_weighted（水平翻转加权）

```python
'flip_weighted': {
    'augmentations': ['original', 'flip'],
    'weights': [0.7, 0.3]
}
```

**原理**：原始图像权重更高，因为原始特征更可靠。

**速度**：⭐⭐⭐⭐⭐ 最快
**效果**：⭐⭐⭐ 基础

### 3. multi_scale（多尺度）

```python
'multi_scale': {
    'augmentations': ['original', 'flip', 'scale_0.95', 'scale_1.05'],
    'weights': [0.4, 0.3, 0.15, 0.15]
}
```

**原理**：不同尺度提取不同粒度的特征，大尺度保留整体，小尺度聚焦局部。

**速度**：⭐⭐⭐ 中等
**效果**：⭐⭐⭐⭐ 较好

### 4. multi_scale_crop（多尺度裁剪）

```python
'multi_scale_crop': {
    'augmentations': ['original', 'flip', 'crop_0.05', 'crop_0.10'],
    'weights': [0.4, 0.3, 0.15, 0.15]
}
```

**原理**：模拟不同距离拍摄的图像，裁剪中心区域强制模型关注主体。

**速度**：⭐⭐⭐ 中等
**效果**：⭐⭐⭐⭐ 较好

---

## 实现原理

### 核心代码流程

```python
def extract_features_with_tta(model, images, ...):
    # 1. 对每种增强类型分别推理
    for aug_type in aug_list:
        aug_images = apply_augmentation(images, aug_type)
        feat = model(aug_images, ...)  # [B, D]
        features_list.append(feat)
    
    # 2. L2归一化
    features_list = [normalize(f, p=2, dim=1) for f in features_list]
    
    # 3. 加权融合
    weights_tensor = torch.tensor(weights).view(-1, 1, 1)  # [N, 1, 1]
    fused = (torch.stack(features_list, dim=0) * weights_tensor).sum(dim=0)
    
    # 4. 再次归一化
    fused = normalize(fused, p=2, dim=1)
    
    return fused  # [B, D]
```

### 张量形状变化

```
原始特征:      [B, D]          # B=batch, D=特征维度(512/768/...)
增强后特征:    [N, B, D]        # N=增强数量
权重:          [N, 1, 1]        # 自动广播
加权融合:      [B, D]          # 输出
```

---

## 增强类型详解

| 类型 | 实现方式 | 效果 |
|:-----|:---------|:-----|
| `original` | 不做任何变换 | 基准 |
| `flip` | `torch.flip(images, dims=[3])` | 水平翻转 |
| `scale_0.95` | 缩小5%后插值回原尺寸 | 聚焦中心 |
| `scale_1.05` | 扩大5%后裁剪中心 | 减少边缘 |
| `crop_0.05` | 裁剪5%边缘后插值 | 去除边界干扰 |
| `crop_0.10` | 裁剪10%边缘后插值 | 更极端的裁剪 |

---

## 适配不同模型

### 自动适配机制

脚本已内置**动态适配**功能：

1. **权重数量检查** - 自动验证 `weights` 数量与 `augmentations` 数量匹配
2. **特征维度检查** - 从实际模型输出动态获取特征维度
3. **兼容所有模型** - 无论 ViT、ResNet 或任何 TransReID 变体

```python
# 自动获取特征维度（无需手动指定）
feat_dim = features_list[0].shape[1]  # 运行时推断
```

### 适用于

| 模型类型 | 支持情况 |
|:---------|:---------|
| ViT-Base | ✅ 完全支持 |
| ViT-Large | ✅ 完全支持 |
| ResNet50 | ✅ 完全支持 |
| 任意自定义模型 | ✅ 只需返回特征张量 |

---

## 批量测试

### 测试所有 RTX4090 模型

```bash
# Windows
run_tta_tests.bat

# Linux/Mac
bash run_tta_tests.sh
```

### 汇总结果

```bash
python summarize_tta_results.py
```

---

## 输出说明

### 日志格式

```
2026-04-06 13:47:46,640 transreid.tta INFO: ======================================================================
2026-04-06 13:47:46,640 transreid.tta INFO: Test-Time Augmentation (TTA) 测试脚本
...
2026-04-06 13:55:02,827 transreid.tta INFO: mAP: 91.0832%
2026-04-06 13:55:02,827 transreid.tta INFO: CMC curve, Rank-1  : 94.1597%
```

### 输出文件

- **日志文件**: `logs/{模型名称}/tta_test_log.txt`
- **结果文件**: `logs/{模型名称}/tta_results.txt`

---

## 性能对比示例

| 策略 | mAP | Rank-1 | Rank-5 | 耗时倍率 |
|:-----|:----:|:------:|:------:|:--------:|
| 基准（无TTA） | 91.08% | 94.16% | 97.97% | 1.0x |
| flip_only | +0.15% | +0.10% | +0.05% | 2.0x |
| flip_weighted | +0.12% | +0.08% | +0.03% | 2.0x |
| multi_scale | +0.25% | +0.18% | +0.10% | 4.0x |
| multi_scale_crop | +0.22% | +0.15% | +0.08% | 4.0x |

*注：具体数值因模型和数据集而异*

---

## 注意事项

### 1. 训练时未使用 reranking

项目训练时**未启用** reranking，因此：
- 测试时**建议不使用** `--reranking` 标志
- 除非训练配置中启用了 reranking

### 2. 特征维度

脚本会自动适配，但确保模型输出的第一维是特征维度：

```python
# 正确
feat = model(images)  # [B, D]

# 如果返回tuple，取第一个
if isinstance(feat, tuple):
    feat = feat[0]
```

### 3. 批量大小

TTA 会将 batch 中的每张图像做 N 次增强（N=增强策略数量），显存占用略增，建议：
- 测试 batch size: 64-128（根据显存调整）

### 4. 权重归一化

TTA策略中的 `weights` 会自动归一化处理，无需手动保证和为1。

---

## 常见问题

### Q1: 报错 "weights数量与augmentations数量不匹配"

**原因**：TTA策略配置中 `weights` 数量与 `augmentations` 数量不一致。

**解决**：检查 `TTA_STRATEGIES` 字典中的配置，确保两者数量一致。

### Q2: 报错 "TTA特征维度不一致"

**原因**：模型对不同增强类型输出的特征维度不同。

**解决**：检查模型forward实现，确保不同增强输入输出维度一致。

### Q3: TTA 效果不如预期

**原因**：
1. 数据集本身已进行过强数据增强
2. 图像分辨率已足够高
3. 增强策略不适合当前数据集

**建议**：
- 尝试 `multi_scale` 策略
- 调整 `weights` 权重分配

### Q4: 如何选择策略？

| 场景 | 推荐策略 |
|:-----|:---------|
| 追求速度 | `flip_only` |
| 追求效果 | `multi_scale` |
| 平衡时间与效果 | `flip_weighted` |
| 全部测试 | `all` |

---

## 参考资料

- TransReID 论文: [TransReID: Transformer-Based Object Re-Identification](https://arxiv.org/abs/2104.03349)
- TTA 技术: [Test-Time Augmentation for Object Detection](https://arxiv.org/abs/1903.05842)
- 官方代码: https://github.com/damo-cv/TransReID