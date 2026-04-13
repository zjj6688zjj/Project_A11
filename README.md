# 🏀 基于复杂运动场景的篮球持球人身份重识别算法开发

> 📅 更新时间: 2026-04-13

## 📚 文档目录

本项目文档已拆分为多个文件：

| 📄 文档 | 📝 说明 |
|:------:|:--------|
| [README_main.md](README_main.md) | 📖 项目主文档（赛题背景、结果汇总、配置对比、优化指南） |
| [README_ensembling_models.md](README_ensembling_models.md) | 🔗 模型融合指南 |
| [README_reranking.md](README_reranking.md) | 🔄 Re-ranking 测试分析 |
| [README_learning_rate_scheduler.md](README_learning_rate_scheduler.md) | 📈 学习率调度器配置说明 |
| [README_test_time_augmentation.md](README_test_time_augmentation.md) | 🧪 TTA 完整指南（策略、缓存生成、性能指标） |
| [README_frontend_usage.md](README_frontend_usage.md) | 🌐 前端使用指南 |
| [README_project_analysis.md](README_project_analysis.md) | 📊 项目文件分析报告 |
| [README_emoji_guide.md](README_emoji_guide.md) | 😊 Emoji符号指南 |

---

## 🚀 快速链接

### 🎯 当前最佳结果

| 📊 评估指标 | 🏆 当前最佳结果 | 🎯 赛题达标要求 | ✅ 状态 |
|:----------:|:-------------:|:-------------:|:------:|
| **mAP** | **91.50%** | >=91.5% | ✅ 达标 |
| **Rank-1** | **94.40%** | >=94.0% | ✅ 达标 |

**💡 说明**：
- 🚫 无TTA时，最佳单模型mAP为91.4%，距离目标91.5%差0.1%
- ✅ 使用TTA后，`rtx4090_from_final153model`模型达到mAP 91.50%，已达标

**📅 最新测试结果（2026-04-13）：**
- 🥇 **最佳单模型**：`rtx4090_from_final212model` (epoch 283)
- 📊 **测试结果**：mAP: 91.4%, Rank-1: 93.7%, Rank-5: 98.1%, Rank-10: 98.9%
- ⚡ **性能指标**：特征提取 ~31ms，查询匹配 ~0.4ms

**📅 历史最佳TTA结果（2026-04-11）：**
- 🥇 **模型**：`rtx4090_from_final153model` (epoch 177)
- 🧪 **TTA策略**：`multi_scale_crop`
- 📊 **TTA融合结果**：mAP: 91.50%, Rank-1: 94.40%

**💡 说明：**
- 🚫 无TTA时最佳模型为`rtx4090_from_final212model`，mAP 91.4%，Rank-1 93.7%
- ✅ 使用TTA时最佳模型为`rtx4090_from_final153model`，mAP 91.50%, Rank-1 94.40%均已达标

### ⚡ 快速开始

```bash
# 🧪 测试最佳单模型（无TTA）
python test.py --config_file configs/BallShow/rtx4090_from_final212model.yml --test-all

# ✅ 测试最佳TTA模型（推荐）
python test.py --config_file configs/BallShow/rtx4090_from_final153model.yml --test-all

# 🔄 批量Re-ranking对比测试
python autoTestForTop3_rtx4090_yml_withAndWithoutRerank.py

# 🧪 使用TTA测试（测试时增强）
python tta_test.py --config_file configs/BallShow/rtx4090_from_final153model.yml --weight logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth --strategy multi_scale_crop

# 🚀 启动比赛服务器
python fin_server.py

# 💾 生成TTA缓存
python tta_cache_precomputing.py --config configs/BallShow/rtx4090_from_final153model.yml --model logs/BallShow_rtx4090_from_final153model/transformer_checkpoint_177.pth --strategy multi_scale_crop
```

---

## 🔍 重要发现

### 🧪 TTA (Test-Time Augmentation) 测试

TTA 是一种在推理阶段对图片进行增强的技术，可以提升预测稳定性。

📖 详见 [README_test_time_augmentation.md](README_test_time_augmentation.md)

### 🔄 Re-ranking 对本项目无效 ⚠️

训练时配置文件中的 `RE_RANKING: True` 是**无效的**！更关键的是，**Re-ranking 会降低本项目性能**。

**📊 测试结论（5组对比，2026-04-13）：**
- 🚫 无Re-ranking平均：mAP 91.3%
- 🔄 有Re-ranking平均：mAP 86.6%
- 📉 **性能下降：mAP -4.7%, Rank-1 -6.6%**

**⚠️ 建议：最终提交时禁用 Re-ranking！**

📖 详见 [README_reranking.md](README_reranking.md)

### 🔗 模型融合存在问题 ⚠️

直接使用最优的几个模型进行融合后性能大幅下降，因为融合的模型的训练配置不一致。

📖 详见 [README_ensembling_models.md](README_ensembling_models.md)

### 📈 学习率调度器使用绝对epoch ⚠️

断点续训时，学习率调度器使用**绝对epoch**而非相对epoch进行计算，可能导致学习率过低。从检查点继续训练时需要特别注意 `MAX_EPOCHS` 的设置。

📖 详见 [README_learning_rate_scheduler.md](README_learning_rate_scheduler.md)

---

## 📁 项目结构

```
📂 项目根目录/
├── 📄 README.md                          # 本文档（项目入口）
├── 📄 README_main.md                     # 主文档
├── 📄 README_ensembling_models.md        # 模型融合指南
├── 📄 README_reranking.md                # Re-ranking 分析
├── 📄 README_learning_rate_scheduler.md   # 学习率调度器说明
├── 📄 README_test_time_augmentation.md     # TTA 完整指南
├── 📄 README_frontend_usage.md            # 前端使用指南
├── 📄 README_project_analysis.md          # 项目文件分析
├── 📄 README_emoji_guide.md              # Emoji符号指南
├── 🐍 fin_server.py                      # 比赛专用API服务器（端口8002）
├── 🌐 fin_frontend.html                  # 比赛前端页面
├── 🐍 tta_cache_precomputing.py          # TTA缓存预计算脚本
├── 🐍 tta_test.py                        # TTA测试脚本
├── 🐍 ensemble.py                         # 模型融合脚本
├── 🐍 train.py / test.py                  # 训练和测试脚本
├── 🐍 autoTestForTop3_rtx4090_yml_withAndWithoutRerank.py  # 批量Re-ranking对比测试
├── 🐍 rtx4090results.py                   # RTX 4090结果汇总分析
├── 🐍 diagnoseAndTestOfCpuAndGpu.py      # GPU/CPU诊断脚本
├── 🐍 yml_pth_correspondence.py          # 配置文件与模型对应关系脚本
├── ⚙️ config/                            # 默认配置
├── ⚙️ configs/BallShow/                  # 所有配置文件
├── 📂 logs/                              # 训练日志和检查点
├── 📂 data/                              # 数据集
├── 📂 tta_cache/                         # TTA缓存目录
├── 📂 ensemble_cache/                    # 模型融合缓存
├── ⚙️ solver/                            # 优化器和调度器
├── ⚙️ processor/                         # 训练和推理处理器
├── 🧠 model/                             # 模型架构
├── 📊 datasets/                          # 数据加载器
├── 📉 loss/                              # 损失函数
├── 🛠️ utils/                             # 工具函数和评估指标
└── 🏋️ weights/                           # 预训练权重
```

---

## ⚙️ 安装与依赖

### 🖥️ 环境要求

- 🐍 Python 3.7+
- 🔥 PyTorch 1.7.1+
- 🎮 CUDA 11.0+ (推荐)

### 📦 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖包：**
- `torch`, `torchvision` - 深度学习框架 🔥
- `timm` - PyTorch图像模型库 🖼️
- `yacs` - 配置管理 ⚙️
- `fastapi`, `uvicorn` - Web服务器 🌐
- `Pillow`, `opencv-python` - 图像处理 🖼️
- `numpy` - 数值计算 🔢

### 📂 数据准备

将训练集与测试集解压至`data/BallShow/`目录：

```
📂 data/BallShow/
    ├── 📂 bounding_box_train/    # 训练集
    ├── 📂 bounding_box_test/     # 测试集/Gallery
    └── 📂 query/                 # 查询集
```

### 🏋️ 预训练权重

下载ImageNet预训练的ViT模型：
```bash
# 自动下载或手动下载到 weights/ 目录
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

---

## 🎯 核心功能

### 1️⃣ 模型训练

```bash
# 🏋️ 从头训练
python train.py --config_file configs/BallShow/rtx4090_final.yml

# 🔄 从检查点继续训练
python train.py --config_file configs/BallShow/rtx4090_from_final153model.yml --resume logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth
```

### 2️⃣ 模型测试

```bash
# 🧪 测试单个模型（推荐：不使用reranking）
python test.py --config_file configs/BallShow/rtx4090_final.yml TEST.WEIGHT logs/BallShow_rtx4090_final/transformer_checkpoint_153.pth

# 📊 批量测试所有检查点
python test.py --config_file configs/BallShow/rtx4090_final.yml --test-all
```

### 3️⃣ TTA测试

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

### 4️⃣ 启动比赛服务器

```bash
# 🚀 启动服务器（自动检测端口8002-8011）
python fin_server.py

# 🌐 访问前端
# http://localhost:8002
```

### 5️⃣ 模型融合

```bash
# 🔗 使用预设模型进行融合
python ensemble.py

# ⚙️ 指定融合类型和策略
python ensemble.py --type feature --method average

# 📊 测试所有组合
python ensemble.py --type all --method all
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
