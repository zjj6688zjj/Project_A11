# 📈 学习率调度器配置说明

> 📅 更新时间: 2026-04-13

## ⚠️ 重要声明

> **📌 项目只支持一种调度器：带 warmup 的余弦学习率调度器（CosineLRScheduler）**

---

## 🐍 代码实现

- **📄 主要文件**：`solver/scheduler_factory.py`
- **⚙️ 调度器创建**：`create_scheduler(cfg, optimizer)` 函数
- **📈 调度器类型**：余弦退火（CosineLRScheduler）

---

## ✅ 有效参数

以下参数控制余弦退火调度器：

| ⚙️ 参数 | 📝 说明 | 📍 默认值 | 🎯 作用 |
|:------|:------|:--------|:------|
| `MAX_EPOCHS` | 总训练周期 | 250 | 控制余弦周期长度 |
| `BASE_LR` | 最大学习率 | 0.01 | 学习率上限 |
| `WARMUP_EPOCHS` | 预热周期 | 5 | 学习率线性增加到BASE_LR |
| `WARMUP_METHOD` | 预热方法 | 'linear' | 线性/常数预热 |
| `WARMUP_FACTOR` | 预热因子 | 0.01 | 初始学习率比例 |

---

## ❌ 无效参数（遗留配置）

> **⚠️ 以下参数在配置文件中存在但无效**：

| ⚙️ 参数 | 📝 说明 | 📝 备注 |
|:------|:------|:------|
| `STEPS` | 多步衰减节点 | 项目不使用多步衰减调度器 |
| `GAMMA` | 衰减因子 | 项目不使用多步衰减调度器 |

---

## 📐 学习率变化公式

学习率按余弦退火曲线变化：

```
📈 LR = LR_min + 0.5 * (BASE_LR - LR_min) * (1 + cos(pi * epoch/MAX_EPOCHS))
```

其中 `LR_min = 0.002 * BASE_LR`

---

## 📊 可视化

```
📈 学习率变化曲线 (MAX_EPOCHS=250, BASE_LR=0.01, WARMUP_EPOCHS=10):

LR
0.01 ┤                    /-----------------------------\
     |                  /                                 \
0.005┤                /                                     \
     |              /                                         \
0.002┤------------/                                             \
     |                                                            \
     └──────────────────────────────────────────────────────────────
     0    10   50   100  150  200  250               epoch
           ↑ warmup
```

---

## 💡 配置建议

### 🎯 场景1：从头训练新模型

```yaml
⚙️ SOLVER:
  BASE_LR: 0.01              # 标准学习率
  MAX_EPOCHS: 250-300        # 充足训练时间
  WARMUP_EPOCHS: 10-15       # 稳定预热
  WARMUP_METHOD: 'linear'
```

### 🎯 场景2：从检查点继续微调

```yaml
⚙️ SOLVER:
  BASE_LR: 0.002-0.003       # 较小学习率，避免破坏已收敛模型
  MAX_EPOCHS: 300            # 继续训练
  WARMUP_EPOCHS: 5           # 短预热，模型已预热过
  WARMUP_METHOD: 'linear'
```

### 🎯 场景3：精细微调（距离目标0.1%）

```yaml
⚙️ SOLVER:
  BASE_LR: 0.001             # 更小的学习率
  MAX_EPOCHS: 300-400        # 更长训练时间
  WARMUP_EPOCHS: 3           # 极短预热
```

---

## 📊 RTX4090系列配置的学习率设置

| ⚙️ 配置 | 📈 BASE_LR | 🔢 MAX_EPOCHS | 🔥 WARMUP | 📝 训练方式 |
|:------|:---------|:------------|:--------|:---------|
| rtx4090_final | 0.01 | 250 | 10 | 🏋️ 从头训练 |
| rtx4090_optimized | 0.012 | 200 | 10 | 🏋️ 从头训练 |
| rtx4090_ultimate | 0.008 | 250 | 10 | 🏋️ 从头训练 |
| rtx4090_sprint | 0.008 | 300 | 15 | 🏋️ 从头训练 |
| rtx4090_from_final212model | 0.003 | 300 | 5 | 🔄 微调 |
| rtx4090_from_final153model | 0.0015 | 250 | 2 | 🔄 微调 |

---

## 🐍 调度器相关代码

### scheduler_factory.py

```python
def create_scheduler(cfg, optimizer):
    """⚙️ 创建学习率调度器"""
    scheduler = CosineLRScheduler(
        optimizer,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
        num_epochs=cfg.SOLVER.MAX_EPOCHS,
        base_lr=cfg.SOLVER.BASE_LR,
        alpha=0.002,  # 最小学习率比例
    )
    return scheduler
```

### cosine_lr.py

```python
class CosineLRScheduler:
    def __init__(self, optimizer, warmup_epochs, warmup_method, 
                 num_epochs, base_lr, alpha=0.002):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        self.min_lr = base_lr * alpha
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 🔥 Warmup: 线性增加到 base_lr
            lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            # 📈 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
```

---

## ❓ 常见问题

### ❓ Q1: 为什么配置了 STEPS 和 GAMMA 但没效果？

**💡 答案**：项目使用 CosineLRScheduler，这些参数被忽略。如果需要多步衰减，需要修改调度器实现。

### ❓ Q2: 如何判断模型是否收敛？

**💡 答案**：观察训练日志中的 mAP 和 Rank-1 指标：
- 📈 指标持续上升 → 继续训练
- ➡️ 指标停滞超过 20-30 epochs → 可以停止或调整学习率
- 📉 指标开始下降 → 过拟合，立即停止

### ❓ Q3: 从检查点继续训练时学习率如何设置？

**💡 答案**：一般设置为原始 BASE_LR 的 1/3 ~ 1/5：
- 📉 原始 0.01 → 继续训练用 0.002-0.003
- 📉 原始 0.008 → 继续训练用 0.001-0.002

---

## ⚠️ 【关键发现】调度器使用绝对epoch计数（2026-04-09）

### 🔍 问题背景

在断点续训场景中，发现学习率调度器使用**绝对epoch**而非相对epoch进行计算，这导致从检查点继续训练时学习率可能过低。

### 🐍 代码实现分析

#### 1️⃣ 调度器创建 (`solver/scheduler_factory.py`)

```python
lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,  # 设置为 MAX_EPOCHS
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,      # 使用epoch计数
        # ... 其他参数
    )
```

#### 2️⃣ 调度器调用 (`processor/processor.py`)

```python
for epoch in range(start_epoch, epochs + 1):
    scheduler.step(epoch)  # 传入绝对epoch值
```

#### 3️⃣ 调度器计算 (`cosine_lr.py`)

```python
else:
    i = t // self.t_initial  # t是传入的epoch
    t_i = self.t_initial
    t_curr = t - (self.t_initial * i)  # 当前周期内的位置
```

### 📉 实际影响

#### 📝 场景：从epoch 177继续训练

```yaml
⚙️ # 配置示例
MAX_EPOCHS: 180
BASE_LR: 0.0012
```

**📊 调度器计算**：
- 📥 传入 `epoch=177`, `t_initial=180`
- 📊 进度 = 177/180 ≈ 98%
- 📉 学习率 ≈ 0.0000036（非常低）

**❌ 导致问题**：学习率过低，缺乏优化动力突破瓶颈。

### 💡 解决方案

#### 💡 方案1：调整MAX_EPOCHS（推荐）

```yaml
⚙️ # 将MAX_EPOCHS设置为从0开始的总epoch数
MAX_EPOCHS: 357  # 绝对epoch总数（177 + 180）
```

📊 计算效果：
- 📊 进度 = 177/357 ≈ 50%
- 📈 学习率 ≈ 中等水平

#### 💡 方案2：修改调度器代码

修改 `processor.py`：
```python
# 📝 原来的：
scheduler.step(epoch)

# ✅ 修改为相对epoch：
scheduler.step(epoch - start_epoch + 1)
```

#### 💡 方案3：修改scheduler_factory.py

```python
def create_scheduler(cfg, optimizer, start_epoch=0):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    # 🔄 如果从检查点继续，调整起始点
    if start_epoch > 0:
        last_epoch = start_epoch - 1
    else:
        last_epoch = -1
    
    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            # ... 其他参数
            last_epoch=last_epoch  # 设置起始epoch
        )
```

### ✅ 验证方法

检查训练日志中的学习率：
```
# 📄 日志示例
Epoch[177] Iteration[40/232] Loss: 1.396, Acc: 0.999, Base Lr: 2.96e-04
```

📐 学习率计算公式验证：
```python
📈 LR = LR_min + 0.5 * (BASE_LR - LR_min) * (1 + cos(pi * epoch/MAX_EPOCHS))
```

### 💡 配置建议

1. **🏋️ 从头训练**：`MAX_EPOCHS` 设置为实际训练周期
2. **🔄 断点续训**：
   - ✅ 方法A：调整 `MAX_EPOCHS` 为绝对epoch总数
   - ✅ 方法B：修改代码使用相对epoch
   - ✅ 方法C：适度提高 `BASE_LR` 补偿低学习率

### 📊 历史案例分析

#### 📊 rtx4090_from_final153model 训练日志
- ⚙️ 配置：`MAX_EPOCHS: 250`, `BASE_LR: 0.0015`
- 📉 epoch 177时学习率：2.96e-04 (0.000296)
- ✅ 计算验证：`cos(pi * 177 / 250) ≈ -0.607` → 匹配日志

---

## 📚 相关文档

- [README_main.md](README_main.md) - 📖 项目主文档
- [README_test_time_augmentation.md](README_test_time_augmentation.md) - 🧪 TTA完整指南

---

*📝 最后更新：2026-04-13*
