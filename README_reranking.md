# Re-ranking 测试分析

## 重要发现

> ⚠️ **训练时 Re-ranking 配置无效！**

### 代码证据 (`processor/processor.py`)

```python
# 训练时 (第41行) - 没有 reranking 参数！
evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

# 测试时 (第146行) - 有 reranking 参数
evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=use_reranking)
```

**结论**：配置文件中的 `RE_RANKING: True` 在训练时是**完全无效的**！训练代码根本没有读取这个参数！

---

## Re-ranking 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **K1** | 25 | k-reciprocal 编码的近邻数 |
| **K2** | 6 | k-互惠近邻数 |
| **LAMBDA** | 0.95 | Jaccard相似度权重 |
| **FEAT_NORM** | yes | L2归一化 |

---

## 测试命令

### 启用 reranking
```bash
python test.py --reranking --config_file configs/BallShow/rtx4090_final.yml TEST.WEIGHT logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth
```

### 不启用 reranking（推荐）
```bash
python test.py --config_file configs/BallShow/rtx4090_final.yml TEST.WEIGHT logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth
```

---

## 测试结果对比

| 指标 | test.py (reranking) | 训练时评估 | 差异 |
|------|---------------------|-----------|------|
| **mAP** | 86.2% | 91.3% | ⬇️ -5.1% |
| **Rank-1** | 87.2% | 94.2% | ⬇️ -7.0% |
| **Rank-5** | 95.1% | 98.0% | ⬇️ -2.9% |
| **Rank-10** | 97.6% | 99.0% | ⬇️ -1.4% |

**解释**：测试时用了 reranking，训练时没用 reranking，所以结果不同！

---

## RTX4090系列配置中的 Re-ranking 设置

| 配置文件 | RE_RANKING | 训练时是否生效 | 实际效果 |
|---------|-----------|--------------|---------|
| rtx4090_final.yml | True | ❌ **无效** | 训练不用，测试才用 |
| rtx4090_optimized.yml | True | ❌ **无效** | 训练不用，测试才用 |
| rtx4090_ultimate.yml | True | ❌ **无效** | 训练不用，测试才用 |
| rtx4090_sprint.yml | True | ❌ **无效** | 训练不用，测试才用 |
| rtx4090_guaranteed.yml | True | ❌ **无效** | 训练不用，测试才用 |
| rtx4090_from_final212model.yml | True | ❌ **无效** | 训练不用，测试才用 |
| rtx4090_from_final153model.yml | True | ❌ **无效** | 训练不用，测试才用 |

---

## Vit-TransReID系列配置

| 配置文件 | RE_RANKING | mAP | Rank-1 |
|---------|-----------|-----|--------|
| vit_transreid_stride.yml | false | 90.7% | 93.0% |
| vit_base.yml | false | 88.8% | 92.5% |
| vit_sie.yml | false | 88.6% | 92.1% |

---

## 所有模型检查点 Reranking 对比测试结果

### 详细对比表

| 模型 | Epoch | 无 Reranking | 有 Reranking | ΔmAP | ΔRank-1 | 推荐 |
|------|-------|--------------|--------------|------|---------|------|
| **vit_base** | 120 | mAP 88.8%, R1 92.4% | mAP 80.0%, R1 83.1% | **-8.8%** | -9.3% | ❌ |
| **vit_sie** | 60 | mAP 88.6%, R1 92.1% | mAP 79.8%, R1 81.9% | **-8.8%** | -10.2% | ❌ |
| **vit_transreid_stride** | 65 | mAP 90.7%, R1 93.0% | mAP 86.1%, R1 86.9% | **-4.6%** | -6.1% | ❌ |
| **rtx4090_final** | 153 | 训练评估: 91.3%, 94.2% | mAP 86.2%, R1 87.2% | **-5.1%** | -7.0% | ❌ |
| **transreid_final** | 10 | - | mAP 76.6%, R1 80.5% | - | - | - |
| **competition_best** | 35 | - | mAP 86.1%, R1 87.4% | - | - | - |
| **competition_simple** | 40 | - | mAP 86.2%, R1 87.4% | - | - | - |
| **competition_simple** | 45 | mAP 89.9%, R1 93.2% | - | - | - | ✅ |
| **competition_384_optimized** | - | mAP 88.6%, R1 92.6% | - | - | - | ✅ |

### 结论汇总

| 指标 | 无 Reranking | 有 Reranking | 差异 |
|------|--------------|--------------|------|
| **最佳 mAP** | 91.3% (rtx4090_final) | 86.2% (rtx4090_final) | -5.1% |
| **最佳 Rank-1** | 94.2% (rtx4090_final) | 87.4% (competition_simple) | -6.8% |
| **平均 ΔmAP** | - | - | **-6.8%** |

> ⚠️ **所有测试模型均显示 reranking 导致性能下降 4.6%~8.8%**，无任何例外！

---

## 结论与建议

### 最终提交建议

> ❌ **强烈建议：最终提交时不要启用 reranking！**

| 场景 | 是否启用 reranking | 原因 |
|------|-------------------|------|
| **最终比赛提交** | ❌ **不推荐** | 训练评估 91.3%/94.2% > 测试 reranking 86.2%/87.2% |
| 快速验证模型 | ✅ 可用 | 用于快速对比，但结果仅供参考 |
| 学术论文评估 | ❌ **不推荐** | 应使用训练时一致的评估方式 |

**结论**：reranking 对本项目**有负面效果**，会导致 mAP 下降约 5%，Rank-1 下降约 7%。

---

### 结论

1. **训练评估没有 reranking** → 训练结果 91.3%/94.2% 是**没有 reranking** 的真实性能
2. **reranking 对球秀项目效果不佳** → 使用 reranking 后 mAP 反而下降 5%！
3. **配置无效** → `RE_RANKING: True` 在训练代码中被忽略

### 建议

1. ❌ **不要在测试时使用 --reranking** → 结果不可靠，且性能反而下降
2. ✅ **最终提交使用训练时保存的最佳检查点** → 不要再单独测试
3. ✅ **以训练时的评估结果为准** → 这是模型的真实性能

### 为什么 reranking 反而降低性能？

可能的解释：
1. **reranking 适用于一般场景**，但对篮球运动员 ReID 的特定特征分布不适用
2. **数据集特点不同**：篮球运动员的姿态变化、遮挡情况可能与 reranking 假设不符
3. **k-reciprocal 编码假设**：reranking 假设相似的查询应该有相似的近邻，这在篮球场景中可能不成立

---

## 技术细节

### Re-ranking 算法原理

k-reciprocal reranking 是一种后处理技术，通过考虑样本之间的互惠最近邻关系来改进检索结果。

**核心思想**：
1. 对于每个样本，找到其 k-reciprocal 近邻（互相都是对方最近邻的样本）
2. 使用 Jaccard 距离修正余弦相似度
3. 最终相似度 = λ × 余弦相似度 + (1-λ) × Jaccard 距离

**参考论文**：
```
Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding. CVPR 2017.
```

---

*最后更新：2026-04-06*
