# 基于复杂运动场景的篮球持球人身份重识别算法开发

## 文档目录

本项目文档已拆分为多个文件：

| 文档 | 说明 |
:|------|------|
| [README_main.md](README_main.md) | 项目主文档（赛题背景、结果汇总、配置对比、优化指南） |
| [README_ensembling_models.md](README_ensembling_models.md) | 模型融合指南 |
| [README_reranking.md](README_reranking.md) | Re-ranking 测试分析 |
| [README_learning_rate_scheduler.md](README_learning_rate_scheduler.md) | 学习率调度器配置说明 |
| [README_test_time_augmentation.md](README_test_time_augmentation.md) | TTA 测试分析与策略推荐 |

---

## 快速链接

### 当前最佳结果

| 评估指标 | 当前最佳结果 | 赛题达标要求 | 差距 |
:|:--------:|:-----------:|:-----------:|:----:|
| **mAP** | **91.4%** | >=91.5% | 差0.1% |
| **Rank-1** | **94.2%** | >=94.0% | [OK] 已达标 |

### 快速开始

```bash
# 测试最佳模型
python test.py --config_file configs/BallShow/rtx4090_final.yml TEST.WEIGHT logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth

# 使用TTA测试（测试时增强，推荐）
python test_tta.py --config_file configs/BallShow/rtx4090_final.yml --weight logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth --strategy all

# 训练新模型
python train.py --config_file configs/BallShow/rtx4090_sprint.yml

# 从检查点继续微调
python train.py --config_file configs/BallShow/rtx4090_from_final212model.yml --resume logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth
```

---

## 重要发现

### TTA (Test-Time Augmentation) 测试

TTA 是一种在推理阶段对图片进行增强的技术，可以提升预测稳定性。

详见 [README_test_time_augmentation.md](README_test_time_augmentation.md)

### Re-ranking 配置无效 [WARNING]

训练时配置文件中的 `RE_RANKING: True` 是**无效的**！训练代码不使用这个参数。

详见 [README_reranking.md](README_reranking.md)

### 模型融合存在问题 [WARNING]

直接使用 `final_simple_fusion.py` 融合后性能大幅下降，需要确保模型配置一致。

详见 [README_ensembling_models.md](README_ensembling_models.md)

---

## 项目结构

```
TransReID-master/
├── README_main.md                    # 主文档
├── README_ensembling_models.md        # 模型融合指南
├── README_reranking.md              # Re-ranking 分析
├── README_learning_rate_scheduler.md # 学习率调度器说明
├── README_test_time_augmentation.md  # TTA 测试分析
├── configs/BallShow/                 # 所有配置文件
├── logs/                            # 训练日志和检查点
├── data/                            # 数据集
├── ensemble_models.py                # 模型融合脚本
├── final_simple_fusion.py           # 简单融合脚本
├── test_tta.py                      # TTA 测试脚本
├── run_tta_tests.py                 # 批量 TTA 测试
├── summarize_tta_results.py         # TTA 结果汇总
├── train.py / test.py               # 训练和测试脚本
└── solver/                          # 优化器和调度器
```
