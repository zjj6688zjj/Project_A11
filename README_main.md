# 🏀 基于复杂运动场景的篮球持球人身份重识别算法开发

> 📅 更新时间: 2026-04-13

## 🎯 赛题背景

本赛题要求在真实的"球秀"业务场景中检索特定持球人。相较于传统的行人重识别任务，本赛题针对特定的篮球场景，若直接使用现有的成熟ReID算法会面临三大挑战：

1. **👕 相似球衣区分**：场上队友穿着完全相同的球衣，仅靠颜色无法区分。算法必须具备细粒度特征提取能力，依赖球衣号码（常被手臂或球体遮挡）、体态、护具、发型及鞋履颜色来进行区分。

2. **🚫 严重遮挡与姿态多变**：持球人是防守方的重点照顾对象，常处于多人包夹（重叠遮挡）状态；同时，运球、上篮、投篮等动作导致身体姿态发生剧烈形变，与常规站立姿态差异巨大。

3. **🌙 环境干扰**：比赛数据涵盖室内木地板（强反光）、室外塑胶场（复杂背景）、夜间灯光等多种环境。持球人的移动速度通常最快，导致图像极易产生运动模糊，丢失纹理细节。

---

## 🏆 赛题要求与当前最佳结果

| 📊 评估指标 | 🏆 当前最佳结果 | 🎯 赛题达标要求 | ✅ 状态 |
|:----------:|:-------------:|:-------------:|:------:|
| **mAP** | **91.50%** | >=91.5% | ✅ 达标 |
| **Rank-1** | **94.40%** | >=94.0% | ✅ 达标 |

**💡 说明：** 使用TTA (Test-Time Augmentation) 后两项指标均已达标！

**📅 最新TTA测试结果（2026-04-11）：**
- 🥇 **模型**：`rtx4090_from_final153model` (epoch 177)
- 🧪 **TTA策略**：`multi_scale_crop`
- 📊 **完整指标**：mAP: 91.5019%, Rank-1: 94.3981%, Rank-5: 97.9738%, Rank-10: 99.1657%

---

## 📊 所有模型训练结果汇总

### 🎮 RTX4090系列模型（384×128高分辨率）

| 🏷️ 模型配置 | 📄 配置文件 | 🎯 最佳mAP | 🥇 最佳Rank-1 | 🔢 最佳Epoch | ✅ 状态 |
|:----------|:-----------|:---------|:-----------|:---------|:------|
| rtx4090_final | `rtx4090_final.yml` | 91.3% | 94.2% | 153 | ✅ 已完成 |
| rtx4090_from_final153model | `rtx4090_from_final153model.yml` | 91.3% | 94.0% | 177 | ✅ 已完成 |
| rtx4090_from_final212model | `rtx4090_from_final212model.yml` | 91.4% | 93.9% | 283 | ✅ 已完成 |
| rtx4090_optimized | `rtx4090_optimized.yml` | 91.3% | 93.9% | 171 | ✅ 已完成 |
| rtx4090_ultimate | `rtx4090_ultimate.yml` | 91.1% | 93.9% | 241 | ✅ 已完成 |
| rtx4090_sprint | `rtx4090_sprint.yml` | - | - | - | ✅ 已完成 |
| vit_transreid_stride | `vit_transreid_stride.yml` | 90.7% | 93.0% | 65 | ✅ 已完成 |

### 📚 早期实验模型（256×128分辨率）

| 🏷️ 模型配置 | 📄 配置文件 | 🎯 最佳mAP | 🥇 最佳Rank-1 | 🔢 最佳Epoch | 📝 备注 |
|:----------|:-----------|:---------|:-----------|:---------|:------|
| competition_simple | `ballshow_competition_simple.yml` | 89.9% | 93.2% | 45 | 📚 早期实验 |
| vit_base | `vit_base.yml` | 88.8% | 92.5% | 95 | 📚 早期实验 |
| vit_sie | `vit_sie.yml` | 88.6% | 92.1% | 60 | 📚 早期实验 |
| vit_transreid | `vit_transreid.yml` | 88.5% | 91.7% | 20 | 📚 早期实验 |
| vit_transreid_384 | `vit_transreid_384.yml` | 88.5% | 91.5% | 20 | 📚 早期实验 |
| vit_transreid_stride_384 | `vit_transreid_stride_384.yml` | 88.3% | 91.8% | 15 | 📚 早期实验 |
| transreid_advanced | `transreid_advanced.yml` | 82.9% | 86.5% | 40 | ⚠️ 收敛异常 |
| competition_best | `ballshow_competition_best.yml` | 86.1% | 87.4% | 35 | ⚠️ 欠拟合 |
| transreid_final | `transreid_final.yml` | 76.6% | 80.5% | 10 | ⚠️ 收敛异常 |
| vit_jpm | `vit_jpm.yml` | ~10% | ~19% | - | ❌ 训练失败 |

---

### 🏆 模型性能排名（Top 10）

| 🏅 排名 | 🏷️ 模型 | 🎯 mAP | 🥇 Rank-1 | 📐 分辨率 | 📝 备注 |
|:----:|:-----|:----:|:------:|:------:|:------|
| 🥇 1 | rtx4090_from_final153model + TTA | **91.50%** | **94.40%** | 384×128 | 🏆 TTA冠军 |
| 🥈 2 | rtx4090_from_final212model | 91.4% | 93.9% | 384×128 | 🥇 最佳单模型 |
| 🥉 3 | rtx4090_final | 91.3% | 94.2% | 384×128 | 🥇 Rank-1最高 |
| 4 | rtx4090_optimized | 91.3% | 93.9% | 384×128 | - |
| 5 | rtx4090_ultimate | 91.1% | 93.9% | 384×128 | - |
| 6 | vit_transreid_stride | 90.7% | 93.0% | 256×128 | - |
| 7 | competition_simple | 89.9% | 93.2% | 256×128 | - |
| 8 | vit_base | 88.8% | 92.5% | 256×128 | - |
| 9 | vit_sie | 88.6% | 92.1% | 256×128 | - |
| 10 | vit_transreid_384 | 88.5% | 91.5% | 384×128 | - |

---

## ⚙️ RTX4090系列模型配置对比

| ⚙️ 配置项 | optimized | ultimate | final | from_final153model | sprint |
|:--------|:----------|:---------|:------|:-------------------|:-------|
| **MAX_EPOCHS** | 200 | 250 | 250 | 250 | 300 |
| **BASE_LR** | 0.012 | 0.008 | 0.01 | 0.0015 | 0.008 |
| **WARMUP_EPOCHS** | 10 | 10 | 10 | 2 | 15 |
| **WEIGHT_DECAY** | 0.0005 | 0.0003 | 0.0003 | 0.00015 | 0.0002 |
| **COSINE_MARGIN** | 0.5 | 0.6 | 0.55 | 0.55 | 0.6 |
| **COSINE_SCALE** | 30 | 35 | 32 | 32 | 36 |
| **TRIPLET_MARGIN** | 0.3 | 0.35 | 0.35 | 0.35 | 0.4 |
| **K1** | 20 | 25 | 25 | 25 | 30 |
| **K2** | 6 | 8 | 6 | 6 | 8 |

**⚙️ 通用配置（所有RTX4090模型）：**
- 📐 输入分辨率：384×128
- 📦 训练Batch Size：96
- 📦 测试Batch Size：512
- 🧠 模型架构：ViT-Base-Patch16-224-TransReID
- 📏 Stride Size：[12, 12]
- 📷 SIE Camera：启用
- 🔀 JPM：启用（DEVIDE_LENGTH=6）
- 🎯 中心损失：启用
- 🏷️ 标签平滑：启用
- 🔄 Re-ranking：禁用（训练时配置无效，且会降低性能）

**📈 调度器说明：**
- 项目使用带 warmup 的余弦学习率调度器（CosineLRScheduler）
- `STEPS` 和 `GAMMA` 参数在配置文件中存在但**无效**
- 学习率衰减由 `MAX_EPOCHS` 控制余弦周期，`BASE_LR` 控制最大学习率

**⚠️ 【重要】调度器使用绝对epoch计数：**
- 断点续训时，调度器使用**绝对epoch**而非相对epoch进行计算
- 从检查点继续训练时，如果 `MAX_EPOCHS` 设置过小，学习率可能过低
- 💡 解决方案：调整 `MAX_EPOCHS` 为绝对epoch总数

---

## ✅ 达标策略

### ✅ 已达标

通过TTA测试，当前最佳模型 `rtx4090_from_final153model` 已达到赛题要求：
- **🎯 mAP**: 91.50% (要求 >=91.5%) ✅ 达标
- **🥇 Rank-1**: 94.40% (要求 >=94.0%) ✅ 达标

### 🚀 性能优化指南

#### 第1️⃣ 步：TTA测试（最推荐）
**📈 预期提升：0.2-0.5% | ⏱️ 时间：几分钟 | ✅ 成功率：99%**

```bash
# 💾 生成TTA缓存
python tta_cache_precomputing.py \
  --config configs/BallShow/rtx4090_from_final153model.yml \
  --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy multi_scale_crop

# 🧪 TTA测试
python tta_test.py \
  --config_file configs/BallShow/rtx4090_from_final153model.yml \
  --weight logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy all
```

#### 第2️⃣ 步：断点继续微调
**📈 预期提升：0.1-0.3% | ⏱️ 时间：2-4小时 | ✅ 成功率：90%**

```bash
# 🔄 继续训练已有模型
python train.py \
  --config_file configs/BallShow/rtx4090_from_final153model.yml \
  --resume logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth
```

#### 第3️⃣ 步：启动比赛服务器
```bash
# 🚀 启动比赛专用服务器
python fin_server.py

# 🌐 访问 http://localhost:8002 使用前端界面
```

---

## 🧪 测试命令

### 🧪 测试单个模型（不使用 reranking）

> ⚠️ **重要提示**：训练时未使用 reranking，因此测试时也不应使用！

```bash
# 🚫 不使用 reranking（推荐）
python test.py \
  --config_file configs/BallShow/rtx4090_from_final153model.yml \
  TEST.WEIGHT logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth
```

### 📊 批量测试所有检查点

```bash
python test.py --config_file configs/BallShow/rtx4090_from_final153model.yml --test-all
```

### 🧪 TTA 测试

TTA 是一种在推理阶段对图片进行增强的技术，可以提升预测稳定性。

```bash
# 💾 生成TTA缓存
python tta_cache_precomputing.py \
  --config configs/BallShow/rtx4090_from_final153model.yml \
  --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy multi_scale_crop

# 🧪 TTA 测试（推荐方法）
python tta_test.py \
  --config_file configs/BallShow/rtx4090_from_final153model.yml \
  --weight logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth \
  --strategy all
```

📖 详见 [README_test_time_augmentation.md](README_test_time_augmentation.md)

---

## 🌐 比赛服务器

### 🚀 启动服务器

```bash
# 🚀 启动比赛专用服务器（端口8002）
python fin_server.py

# 🔄 服务器会自动检测可用端口（8002-8011）
```

### 🌐 访问前端

启动后访问: `http://localhost:8002` (端口可能因占用而变化)

### 🔌 API接口

| 🔌 端点 | 📝 方法 | 📝 说明 |
|:------|:------|:------|
| `/` | GET | 🌐 前端页面 |
| `/health` | GET | 💓 健康检查 |
| `/model_info` | GET | 🧠 模型信息 |
| `/extract_features` | POST | 🔍 特征提取 |
| `/search` | POST | 🔍 标准搜索 |
| `/competition_search` | POST | 🏆 比赛模式搜索 |
| `/benchmark` | POST | 📊 性能基准测试 |

📖 详见 [README_frontend_usage.md](README_frontend_usage.md)

---

## ⚙️ 安装与使用

### 🖥️ 环境依赖

```bash
🐍 Python 3.7+，🔥 PyTorch 1.7.1+
pip install timm yacs termcolor
pip install -r requirements.txt
```

### 📂 数据准备

将训练集与测试集解压至`data/BallShow/`目录：

```
📂 data/BallShow/
    ├── 📂 bounding_box_train/    # 训练集
    ├── 📂 bounding_box_test/     # 测试集/Gallery
    └── 📂 query/                 # 查询集
```

### 🏋️ 预训练权重

下载ImageNet预训练的Transformer模型：
```
🔗 https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

---

## 📁 项目结构

```
📂 项目根目录/
├── 📄 README.md                          # 项目入口文档
├── 📄 README_main.md                     # 本文档（主文档）
├── 📄 README_frontend_usage.md           # 前端使用指南
├── 📄 README_test_time_augmentation.md   # TTA完整指南
├── 📄 README_ensembling_models.md        # 模型融合指南
├── 📄 README_reranking.md                # Re-ranking分析
├── 📄 README_learning_rate_scheduler.md  # 学习率调度器说明
├── 🐍 fin_server.py                      # 比赛专用API服务器
├── 🌐 fin_frontend.html                  # 比赛前端页面
├── 🐍 tta_cache_precomputing.py          # TTA缓存预计算脚本
├── 🐍 tta_test.py                        # TTA测试脚本
├── 🐍 ensemble.py                        # 模型融合脚本
├── ⚙️ configs/BallShow/                  # 所有配置文件
├── 📂 logs/                              # 训练日志和检查点
├── 📂 data/                              # 数据集
├── 🐍 train.py / test.py                 # 训练和测试脚本
├── ⚙️ solver/                            # 优化器和调度器
├── ⚙️ processor/                         # 训练和推理处理器
├── 🧠 model/                             # 模型架构
├── 📊 datasets/                          # 数据加载器
└── 📉 loss/                              # 损失函数
```

---

## 📚 引用

本项目基于**TransReID**实现：

```bibtex
@InProceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```

🔗 官方代码：https://github.com/damo-cv/TransReID

---

*📝 最后更新：2026-04-13*
