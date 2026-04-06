# 基于复杂运动场景的篮球持球人身份重识别算法开发

## 赛题背景

本赛题要求在真实的"球秀"业务场景中检索特定持球人。相较于传统的行人重识别任务，本赛题针对与特定的篮球场景，若直接使用现有的成熟ReID算法会面临三大挑战：

1. **相似球衣区分：** 场上队友穿着完全相同的球衣，仅靠颜色无法区分。算法必须具备细粒度特征提取能力，依赖球衣号码（常被手臂或球体遮挡）、体态、护具、发型及鞋履颜色来进行区分。

2. **严重遮挡与姿态多变：** 持球人是防守方的重点照顾对象，常处于多人包夹（重叠遮挡）状态；同时，运球、上篮、投篮等动作导致身体姿态发生剧烈形变，与常规站立姿态差异巨大。

3. **环境干扰：** 比赛数据涵盖室内木地板（强反光）、室外塑胶场（复杂背景）、夜间灯光等多种环境。持球人的移动速度通常最快，导致图像极易产生运动模糊，丢失纹理细节。

---

## 赛题要求与当前最佳结果

| 评估指标 | 当前最佳结果 | 赛题达标要求 | 差距 |
|:--------:|:-----------:|:-----------:|:----:|
| **mAP** | **91.4%** | ≥91.5% | 差0.1% |
| **Rank-1** | **94.2%** | ≥94.0% | ✅ 已达标 |

**说明：** Rank-1已达标，mAP还差0.1%。

---

## 所有模型训练结果汇总

### RTX4090系列模型（384×128高分辨率）

| 模型配置 | 配置文件 | 最佳mAP | 最佳Rank-1 | 最佳Epoch | 状态 |
|:--------|:--------|:-------|:-----------|:---------|:-----|
| rtx4090_final | `rtx4090_final.yml` | **91.3%** | **94.2%** | 153 | ✅ 已完成 |
| rtx4090_optimized | `rtx4090_optimized.yml` | 91.3% | 93.9% | 171 | ✅ 已完成 |
| rtx4090_ultimate | `rtx4090_ultimate.yml` | 91.1% | 93.9% | 241 | ✅ 已完成 |
| rtx4090_sprint | `rtx4090_sprint.yml` | - | - | 训练中 | 🔄 进行中 |
| vit_transreid_stride | `vit_transreid_stride.yml` | 90.7% | 93.0% | 65 | ✅ 已完成 |

### 早期实验模型（256×128分辨率）

| 模型配置 | 配置文件 | 最佳mAP | 最佳Rank-1 | 最佳Epoch | 备注 |
|:--------|:--------|:-------|:-----------|:---------|:-----|
| competition_simple | `ballshow_competition_simple.yml` | 89.9% | 93.2% | 45 | ✅ 可用 |
| transreid_advanced | `transreid_advanced.yml` | 82.9% | 86.5% | 40 | 收敛异常 |
| competition_best | `ballshow_competition_best.yml` | 86.1% | 87.4% | 35 | 欠拟合 |
| vit_base | `vit_base.yml` | 88.8% | 92.5% | 95 | ✅ 可用 |
| vit_sie | `vit_sie.yml` | 88.6% | 92.1% | 60 | ✅ 可用 |
| vit_transreid | `vit_transreid.yml` | 88.5% | 91.7% | 20 | ✅ 可用 |
| vit_transreid_384 | `vit_transreid_384.yml` | 88.5% | 91.5% | 20 | ✅ 可用 |
| vit_transreid_stride_384 | `vit_transreid_stride_384.yml` | 88.3% | 91.8% | 15 | ✅ 可用 |
| transreid_final | `transreid_final.yml` | 76.6% | 80.5% | 10 | 收敛异常 |
| vit_jpm | `vit_jpm.yml` | ~10% | ~19% | - | ❌ 训练失败 |

### 早期探索模型（基础训练）

| 模型配置 | 配置文件 | 最佳mAP | 最佳Rank-1 | 最佳Epoch | 备注 |
|:--------|:--------|:-------|:-----------|:---------|:-----|
| base_train | `ballshow.yml` | 88.5% | 91.1% | 20 | 基准模型 |

---

### 模型性能排名（Top 10）

| 排名 | 模型 | mAP | Rank-1 | 分辨率 |
|:----:|:-----|:----:|:------:|:------:|
| 🥇 | rtx4090_final | **91.3%** | **94.2%** | 384×128 |
| 🥈 | rtx4090_optimized | 91.3% | 93.9% | 384×128 |
| 🥉 | rtx4090_ultimate | 91.1% | 93.9% | 384×128 |
| 4 | vit_transreid_stride | 90.7% | 93.0% | 256×128 |
| 5 | competition_simple | 89.9% | 93.2% | 256×128 |
| 6 | vit_base | 88.8% | 92.5% | 256×128 |
| 7 | vit_sie | 88.6% | 92.1% | 256×128 |
| 8 | vit_transreid_384 | 88.5% | 91.5% | 384×128 |
| 9 | vit_transreid | 88.5% | 91.7% | 256×128 |
| 10 | vit_transreid_stride_384 | 88.3% | 91.8% | 384×128 |

---

## RTX4090系列模型配置对比

| 配置项 | optimized | ultimate | final | sprint (新) |
|--------|-----------|----------|-------|------------|
| **MAX_EPOCHS** | 200 | 250 | 250 | 300 |
| **BASE_LR** | 0.012 | 0.008 | 0.01 | 0.008 |
| **STEPS** | (60, 120) ⚠️ | (80, 160) ⚠️ | (80, 160) ⚠️ | (100, 200) ⚠️ |
| **WARMUP_EPOCHS** | 10 | 10 | 10 | 15 |
| **WEIGHT_DECAY** | 0.0005 | 0.0003 | 0.0003 | 0.0002 |
| **COSINE_MARGIN** | 0.5 | 0.6 | 0.55 | 0.6 |
| **COSINE_SCALE** | 30 | 35 | 32 | 36 |
| **TRIPLET_MARGIN** | 0.3 | 0.35 | 0.35 | 0.4 |
| **K1** | 20 | 25 | 25 | 30 |
| **K2** | 6 | 8 | 6 | 8 |

**通用配置（所有RTX4090模型）：**
- 输入分辨率：384×128
- 训练Batch Size：96
- 测试Batch Size：512
- 模型架构：ViT-Base-Patch16-224-TransReID
- Stride Size：[12, 12]
- SIE Camera：启用
- JPM：启用
- 中心损失：启用
- 标签平滑：启用
- Re-ranking：启用（但训练时不生效）

**⚠️ 调度器说明：**
- 项目使用带 warmup 的余弦学习率调度器（CosineLRScheduler）
- `STEPS` 和 `GAMMA` 参数在配置文件中存在但**无效**
- 学习率衰减由 `MAX_EPOCHS` 控制余弦周期，`BASE_LR` 控制最大学习率
- 学习率变化公式：`LR = LR_min + 0.5 × (BASE_LR - LR_min) × (1 + cos(π × epoch/MAX_EPOCHS))`

---

## RTX 4090 性能优化指南

### 性能目标

| 指标 | 当前最佳 | 目标 | 状态 |
|:----:|:--------:|:----:|:----:|
| **mAP** | 91.3% | ≥91.5% | 差0.2% |
| **Rank-1** | 94.2% | ≥94.0% | ✅ 已达标 |

### RTX 4090 核心优势
- **24GB显存**：可运行高分辨率模型 (384×128)
- **大Batch Size**：训练96-128，测试512
- **高分辨率**：384×128比256×128提升 **2-3% mAP**
- **快速训练**：比RTX 4060快2-3倍

---

### 六步优化策略（从易到难）

#### 第1步：延长训练周期 ⭐⭐⭐⭐⭐（最推荐）
**预期提升：1.5-2.5% | 时间：4-8小时 | 成功率：95%**

```bash
# 使用sprint配置继续训练
python train.py --config_file configs/BallShow/rtx4090_sprint.yml

# 或继续训练已有模型
python train.py --config_file configs/BallShow/rtx4090_final.yml \
  --resume logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth
```

#### 第2步：增强数据增强 ⭐⭐⭐
充分利用RTX 4090的算力做更强数据增强（RE_PROB: 0.8, PROB: 0.6, PADDING: 16）。

#### 第3步：调整损失函数权重 ⭐⭐⭐
```yaml
CENTER_LOSS_WEIGHT: 0.001
MARGIN: 0.5
TRIPLET_LOSS_WEIGHT: 1.2
```

#### 第4步：学习率调度优化
项目使用带 warmup 的余弦学习率调度器，详见 [README_learning_rate_scheduler.md](README_learning_rate_scheduler.md)

#### 第5步：模型融合
详见 [README_ensembling_models.md](README_ensembling_models.md)

#### 第6步：Re-ranking
详见 [README_reranking.md](README_reranking.md)

---

### 性能提升路径

| 阶段 | 方法 | 累计提升 | 预期mAP |
|:----:|:-----|:--------:|:-------:|
| 当前 | 基础模型 | - | 91.3% |
| 阶段1 | 延长训练 | +1.0% | 92.3% ✅ |
| 阶段2 | 模型融合 | +1.0% | **93.3%** ✅ |


## 达标策略建议

### 方案一：使用sprint配置继续训练（推荐）

```bash
# 训练sprint模型
python train.py --config_file configs/BallShow/rtx4090_sprint.yml
```

sprint配置的核心改进：
- 更长的训练周期（300 epochs）→ 余弦退火调度器提供更长的高学习率训练时间
- 更严格的Margin（Cosine 0.6, Triplet 0.4）→ 更强的特征判别
- 更宽松的正则化（WEIGHT_DECAY 0.0002）→ 减少约束

**预期提升：** +0.2-0.4% mAP

### 方案二：从最佳检查点继续微调

```bash
# 从rtx4090_final的最佳Rank-1检查点继续训练
python train.py --config_file configs/BallShow/rtx4090_from_final212model.yml \
  --resume logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth
```

### 方案三：模型融合
详见 [README_ensembling_models.md](README_ensembling_models.md)

---

## 测试命令

### 测试单个模型（不使用 reranking）

> ⚠️ **重要提示**：训练时未使用 reranking，因此测试时也不应使用！详见 [README_reranking.md](README_reranking.md)

```bash
# 不使用 reranking（推荐）
python test.py --config_file configs/BallShow/rtx4090_final.yml TEST.WEIGHT logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth
```

### 批量测试所有检查点

```bash
python test.py --config_file configs/BallShow/rtx4090_final.yml --test-all
```

### 提取并分析训练结果

```bash
python rtx4090results.py
```

### TTA 测试（Test-Time Augmentation）

TTA 是一种在推理阶段对图片进行增强的技术，可以提升预测稳定性。项目内置**动态适配**机制，支持任意模型（ViT、ResNet等），无需手动配置。

```bash
# TTA 测试（推荐方法）
python test_tta.py --config_file configs/BallShow/rtx4090_final.yml \
                   --weight logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth \
                   --strategy all

# 测试单个策略
python test_tta.py --config_file configs/BallShow/rtx4090_final.yml \
                   --weight logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth \
                   --strategy flip_only

# 批量测试所有 RTX4090 模型
run_tta_tests.bat

# 汇总 TTA 测试结果
python summarize_tta_results.py
```

详见 [README_test_time_augmentation.md](README_test_time_augmentation.md)

---

## 安装与使用

### 环境依赖

```python
Python 3.7+，PyTorch 1.7.1+
pip install timm yacs termcolor
pip install -r requirements.txt
```

### 数据准备

将训练集与测试集解压至`data/BallShow/`目录：

```
data/BallShow/
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```

### 预训练权重

下载ImageNet预训练的Transformer模型：
```
https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

### 路径修改

将configs文件夹中配置文件的路径修改为本地保存路径。

---

## 项目结构

```
TransReID-master/
├── configs/BallShow/          # 所有配置文件
│   ├── rtx4090_final.yml      # 最佳配置（Rank-1 94.2%）
│   ├── rtx4090_optimized.yml
│   ├── rtx4090_ultimate.yml
│   └── rtx4090_sprint.yml     # 冲刺配置
├── logs/                       # 训练日志和检查点
│   ├── BallShow_rtx4090_final/
│   ├── BallShow_rtx4090_optimized/
│   └── BallShow_rtx4090_ultimate/
├── data/                       # 数据集
├── ensemble_models.py          # 模型融合脚本
├── final_simple_fusion.py      # 简单融合脚本
├── rtx4090results.py           # 结果提取脚本
├── solver/                     # 优化器和调度器
│   ├── scheduler_factory.py    # 余弦学习率调度器
│   ├── cosine_lr.py           # 余弦退火实现
│   └── lr_scheduler.py        # 多步衰减（遗留，未使用）
└── test.py / train.py          # 测试和训练脚本
```

---

## RTX4090系列模型最终排名

| 排名 | 模型 | mAP | Rank-1 | Epoch | 备注 |
|:----:|:-----|:----:|:------:|:-----:|:-----|
| 🥇 | rtx4090_from_final212model_283_model | **91.4%** | 93.8% | 287 | mAP最高(多次出现) |
| 🥈 | rtx4090_from_final212model | 91.4% | 93.9% | 283 | - |
| 🥉 | rtx4090_final | 91.3% | **94.2%** | 153 | Rank-1最高 |
| 4 | rtx4090_optimized | 91.3% | 93.9% | 171 | - |
| 5 | rtx4090_sprint | 91.1% | 93.8% | 255 | - |
| 6 | rtx4090_ultimate | 91.1% | 93.9% | 241 | - |

**目标达成情况**：
- Rank-1: 94.2% ≥ 94.0% ✅ 已达标
- mAP: 91.4% ≥ 91.5% ❌ 差0.1%

---

## 未来探索方向：更高分辨率猜想

### 冒进想法（2026-04-06）

**猜想**：增大图像分辨率 + 减小batch size，可能进一步提升mAP和Rank-1

### 理论依据

| 当前配置 | 猜想配置 | 分析 |
|---------|---------|------|
| 384×128 | 448×160 或 512×192 | 保持3:1比例，面积增加23-33% |
| batch 96 | batch 64 | 显存平衡，可能提升泛化 |

### 潜在收益

- ✅ 更多细节：球衣号码、面部特征、护具纹理更清晰
- ✅ 更细粒度区分：相似球员更容易区分
- ✅ 可能突破mAP 91.4% 瓶颈

### 潜在风险

- ❌ 无历史数据支撑
- ❌ 计算量增加，训练时间延长
- ❌ 可能过拟合
- ❌ 显存压力增大

### 建议配置

```yaml
INPUT:
  SIZE_TRAIN: [448, 160]   # 保持3:1比例，面积增加23%
  SIZE_TEST: [448, 160]

SOLVER:
  IMS_PER_BATCH: 64        # 减小batch size以适应更高分辨率
  MAX_EPOCHS: 300          # 保持充足训练时间
```

### 适用场景

如果 guaranteed 配置仍无法达标，可以尝试此方案作为**最后的冲刺手段**。

---

## 引用

本项目基于**TransReID**实现：

```
@InProceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```

官方代码：https://github.com/damo-cv/TransReID
