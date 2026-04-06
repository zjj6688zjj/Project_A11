# 学习率调度器配置说明

## 重要声明

> ⚠️ **项目只支持一种调度器：带 warmup 的余弦学习率调度器（CosineLRScheduler）**

---

## 代码实现

- **主要文件**：`solver/scheduler_factory.py`
- **调度器创建**：`create_scheduler(cfg, optimizer)` 函数
- **调度器类型**：余弦退火（CosineLRScheduler）

---

## 有效参数

以下参数控制余弦退火调度器：

| 参数 | 说明 | 默认值 | 作用 |
|------|------|--------|------|
| `MAX_EPOCHS` | 总训练周期 | 250 | 控制余弦周期长度 |
| `BASE_LR` | 最大学习率 | 0.01 | 学习率上限 |
| `WARMUP_EPOCHS` | 预热周期 | 5 | 学习率线性增加到BASE_LR |
| `WARMUP_METHOD` | 预热方法 | 'linear' | 线性/常数预热 |
| `WARMUP_FACTOR` | 预热因子 | 0.01 | 初始学习率比例 |

---

## 无效参数（遗留配置）

> ⚠️ 以下参数在配置文件中存在但**无效**：

| 参数 | 说明 | 备注 |
|------|------|------|
| `STEPS` | 多步衰减节点 | 项目不使用多步衰减调度器 |
| `GAMMA` | 衰减因子 | 项目不使用多步衰减调度器 |

---

## 学习率变化公式

学习率按余弦退火曲线变化：

```
LR = LR_min + 0.5 × (BASE_LR - LR_min) × (1 + cos(π × epoch/MAX_EPOCHS))
```

其中 `LR_min = 0.002 × BASE_LR`

---

## 可视化

```
学习率变化曲线 (MAX_EPOCHS=250, BASE_LR=0.01, WARMUP_EPOCHS=10):

LR
0.01 ┤╮                    ╭─────────────────────────────╮
     │ ╰──╮          ╭───╯                             │
0.002 ┤    ╰────────╯                                 │
     └──────────────────────────────────────────────────┘
     0    10   50   100  150  200  250               epoch
           ↑ warmup
```

---

## 配置建议

### 场景1：从头训练新模型

```yaml
SOLVER:
  BASE_LR: 0.01              # 标准学习率
  MAX_EPOCHS: 250-300        # 充足训练时间
  WARMUP_EPOCHS: 10-15       # 稳定预热
  WARMUP_METHOD: 'linear'
```

### 场景2：从检查点继续微调

```yaml
SOLVER:
  BASE_LR: 0.002-0.003       # 较小学习率，避免破坏已收敛模型
  MAX_EPOCHS: 300            # 继续训练
  WARMUP_EPOCHS: 5           # 短预热，模型已预热过
  WARMUP_METHOD: 'linear'
```

### 场景3：精细微调（距离目标0.1%）

```yaml
SOLVER:
  BASE_LR: 0.001             # 更小的学习率
  MAX_EPOCHS: 300-400        # 更长训练时间
  WARMUP_EPOCHS: 3           # 极短预热
```

---

## RTX4090系列配置的学习率设置

| 配置 | BASE_LR | MAX_EPOCHS | WARMUP | 训练方式 |
|------|---------|------------|--------|---------|
| rtx4090_final | 0.01 | 250 | 10 | 从头训练 |
| rtx4090_optimized | 0.012 | 200 | 10 | 从头训练 |
| rtx4090_ultimate | 0.008 | 250 | 10 | 从头训练 |
| rtx4090_sprint | 0.008 | 300 | 15 | 从头训练 |
| rtx4090_from_final212model | 0.003 | 300 | 5 | 微调 |
| rtx4090_from_final153model | 0.002 | 300 | 5 | 微调 |

---

## 调度器相关代码

### scheduler_factory.py

```python
def create_scheduler(cfg, optimizer):
    """创建学习率调度器"""
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
            # Warmup: 线性增加到 base_lr
            lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
```

---

## 常见问题

### Q1: 为什么配置了 STEPS 和 GAMMA 但没效果？

A: 项目使用 CosineLRScheduler，这些参数被忽略。如果需要多步衰减，需要修改调度器实现。

### Q2: 如何判断模型是否收敛？

A: 观察训练日志中的 mAP 和 Rank-1 指标：
- 指标持续上升 → 继续训练
- 指标停滞超过 20-30 epochs → 可以停止或调整学习率
- 指标开始下降 → 过拟合，立即停止

### Q3: 从检查点继续训练时学习率如何设置？

A: 一般设置为原始 BASE_LR 的 1/3 ~ 1/5：
- 原始 0.01 → 继续训练用 0.002-0.003
- 原始 0.008 → 继续训练用 0.001-0.002

---

*最后更新：2026-04-06*
