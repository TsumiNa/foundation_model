# 📚 AutoEncoder 优化功能文档索引

## 🎯 快速导航

根据你的需求，选择合适的文档：

### 我想...

#### 📖 理解核心算法原理
→ **[README_OPTIMIZATION_CORE.md](README_OPTIMIZATION_CORE.md)**
- 算法流程图
- 核心代码片段（3行核心）
- 数学公式
- 常见问题解答

→ **[OPTIMIZATION_ALGORITHM_CORE.md](OPTIMIZATION_ALGORITHM_CORE.md)**
- 完整的代码实现（逐行注释）
- 技术细节深入解释
- 数值计算示例
- 与其他技术的对比

#### 💻 看可运行的代码示例
→ **[examples/minimal_latent_optimization.py](examples/minimal_latent_optimization.py)**
- 独立的最小化实现
- 梯度计算演示
- 可视化功能
- 运行方式：`python examples/minimal_latent_optimization.py`

#### 📝 了解所有的改进和新功能
→ **[SUMMARY.md](SUMMARY.md)**
- 完整的修复和改进总结
- 使用场景对比
- 设计思想说明

→ **[OPTIMIZATION_IMPROVEMENTS.md](OPTIMIZATION_IMPROVEMENTS.md)**
- 新增功能详解（perturbation, multi-restart）
- 使用场景和最佳实践
- 实际应用建议

#### 🐛 查看修复的 Bug
→ **[AUTOENCODER_FIXES.md](AUTOENCODER_FIXES.md)**
- AutoEncoderHead 的 bug 详情
- 修复前后对比
- 数据流说明

#### 🧪 运行实际测试
→ **[notebooks/verify_autoencoder_optimization.ipynb](notebooks/verify_autoencoder_optimization.ipynb)**
- 基础功能验证
- 使用真实/合成数据
- 端到端测试流程

→ **[notebooks/advanced_optimization_demo.ipynb](notebooks/advanced_optimization_demo.ipynb)**
- 高级优化策略演示
- 多种初始化方法对比
- 性能分析和可视化

---

## 📂 文档结构

```
foundation_model/
│
├── 📘 快速开始
│   ├── README_OPTIMIZATION_CORE.md          ⭐ 推荐先看这个
│   └── SUMMARY.md                           完整总结
│
├── 📗 详细文档
│   ├── OPTIMIZATION_ALGORITHM_CORE.md       算法详解（带注释代码）
│   ├── OPTIMIZATION_IMPROVEMENTS.md         改进说明
│   └── AUTOENCODER_FIXES.md                 Bug 修复记录
│
├── 💻 代码示例
│   └── examples/
│       └── minimal_latent_optimization.py   最小化可运行示例
│
├── 📓 Jupyter Notebooks
│   └── notebooks/
│       ├── verify_autoencoder_optimization.ipynb      基础验证
│       └── advanced_optimization_demo.ipynb           高级演示
│
└── 🔧 源代码
    └── src/foundation_model/
        ├── models/
        │   ├── flexible_multi_task_model.py           optimize_latent 实现
        │   └── task_head/
        │       └── autoencoder.py                      AutoEncoderHead 实现
        └── data/
            └── dataset.py                              AutoEncoder 数据处理
```

---

## 🎓 学习路径

### 路径 1: 快速理解（15分钟）

1. **[README_OPTIMIZATION_CORE.md](README_OPTIMIZATION_CORE.md)** - 5分钟
   - 看核心代码片段（3行核心）
   - 看流程图
   - 看数学表述

2. **[SUMMARY.md](SUMMARY.md)** - 10分钟
   - 了解改进点
   - 看使用示例
   - 了解应用场景

### 路径 2: 深入学习（1小时）

1. **[README_OPTIMIZATION_CORE.md](README_OPTIMIZATION_CORE.md)** - 10分钟
2. **[OPTIMIZATION_ALGORITHM_CORE.md](OPTIMIZATION_ALGORITHM_CORE.md)** - 30分钟
   - 逐行阅读代码注释
   - 理解技术细节
3. **运行示例代码** - 20分钟
   ```bash
   python examples/minimal_latent_optimization.py
   ```

### 路径 3: 实践应用（2小时）

1. **阅读文档** - 30分钟
   - [README_OPTIMIZATION_CORE.md](README_OPTIMIZATION_CORE.md)
   - [OPTIMIZATION_IMPROVEMENTS.md](OPTIMIZATION_IMPROVEMENTS.md)

2. **运行 Notebooks** - 1小时
   ```bash
   jupyter notebook notebooks/verify_autoencoder_optimization.ipynb
   jupyter notebook notebooks/advanced_optimization_demo.ipynb
   ```

3. **应用到实际项目** - 30分钟
   - 加载训练好的模型
   - 调用 `model.optimize_latent()`
   - 分析结果

---

## 🔑 核心概念速查

### 算法原理

```python
# 核心思想（伪代码）
latent = 初始化_latent()
for step in range(优化步数):
    property = 任务头(latent)
    loss = -property  # 最大化
    梯度 = 自动微分(loss, latent)
    latent = latent - 学习率 * 梯度
descriptor = 解码器(latent)
```

### 关键 API

```python
result = model.optimize_latent(
    task_name="density",        # 要优化的任务
    initial_input=None,         # 初始输入（可选）
    mode="max",                 # "max" 或 "min"
    steps=200,                  # 优化步数
    lr=0.1,                     # 学习率
    ae_task_name="reconstruction",  # AutoEncoder 任务
    num_restarts=10,            # 重启次数（避免局部最优）
    perturbation_std=0.2,       # 扰动标准差
)
```

### 返回值

```python
{
    'optimized_latent': torch.Tensor,     # 优化后的 latent (1, latent_dim)
    'optimized_score': torch.Tensor,      # 最终分数 (1, 1)
    'reconstructed_input': torch.Tensor,  # 重构的 descriptor (1, input_dim)
    'history': list[float],               # 优化历史
    'initial_score': float,               # 初始分数
    'all_restarts': list[dict],           # 所有重启的结果（如果 num_restarts > 1）
}
```

---

## ❓ 常见问题快速解答

### Q1: 核心算法是什么？
**A**: 固定模型参数，用梯度下降直接优化 latent representation，使目标任务输出达到极值。

📄 详见：[README_OPTIMIZATION_CORE.md - 核心算法](README_OPTIMIZATION_CORE.md#核心算法一句话总结)

### Q2: 为什么不需要 initial_input？
**A**: 可以从随机 latent 开始优化，不一定需要从真实输入编码。这是改进后的功能。

📄 详见：[OPTIMIZATION_IMPROVEMENTS.md - 灵活初始化](OPTIMIZATION_IMPROVEMENTS.md#1-灵活的初始化策略)

### Q3: 如何避免局部最优？
**A**: 使用 multi-restart（多重启动）和 perturbation（扰动）机制。

📄 详见：[OPTIMIZATION_IMPROVEMENTS.md - Multi-Restart](OPTIMIZATION_IMPROVEMENTS.md#3-多重启动multi-restart避免局部最优)

### Q4: 核心代码在哪里？
**A**:
- 实现：`src/foundation_model/models/flexible_multi_task_model.py:1486-1724`
- 示例：`examples/minimal_latent_optimization.py`

📄 详见：[OPTIMIZATION_ALGORITHM_CORE.md - 核心代码](OPTIMIZATION_ALGORITHM_CORE.md#part-2-优化循环核心算法)

### Q5: AutoEncoderHead 有什么 Bug？
**A**: 缺少 `_predict_impl` 方法实现，已修复。

📄 详见：[AUTOENCODER_FIXES.md](AUTOENCODER_FIXES.md)

---

## 🎯 使用场景导航

### 场景 1: 材料发现（从零开始搜索）
```python
result = model.optimize_latent(
    task_name="thermal_conductivity",
    initial_input=None,       # 随机搜索
    mode="max",
    num_restarts=100,         # 大量尝试
    steps=500,
    ae_task_name="reconstruction"
)
```
📄 详见：[SUMMARY.md - 场景1](SUMMARY.md#场景-1-材料发现从零开始)

### 场景 2: 改进已知材料
```python
result = model.optimize_latent(
    task_name="strength",
    initial_input=current_best,  # 从最好的材料开始
    perturbation_std=0.1,        # 小幅改动
    num_restarts=10,
    mode="max",
    ae_task_name="reconstruction"
)
```
📄 详见：[SUMMARY.md - 场景2](SUMMARY.md#场景-2-改进已知材料)

### 场景 3: 全局搜索（生产环境）
```python
result = model.optimize_latent(
    task_name="density",
    initial_input=None,
    mode="max",
    steps=500,
    num_restarts=50,           # 充分探索
    perturbation_std=0.2,
    ae_task_name="reconstruction"
)
```
📄 详见：[OPTIMIZATION_IMPROVEMENTS.md - 场景3](OPTIMIZATION_IMPROVEMENTS.md#场景-3-全局搜索多重启动)

---

## 📞 获取帮助

### 如果你想...

- **理解算法原理** → 阅读 [OPTIMIZATION_ALGORITHM_CORE.md](OPTIMIZATION_ALGORITHM_CORE.md)
- **快速上手使用** → 阅读 [README_OPTIMIZATION_CORE.md](README_OPTIMIZATION_CORE.md)
- **看代码示例** → 运行 [examples/minimal_latent_optimization.py](examples/minimal_latent_optimization.py)
- **测试实际功能** → 运行 Notebooks
- **了解所有改进** → 阅读 [SUMMARY.md](SUMMARY.md)

---

## 🚀 快速开始（3步）

```bash
# 1. 查看核心算法说明
cat README_OPTIMIZATION_CORE.md

# 2. 运行最小化示例
python examples/minimal_latent_optimization.py

# 3. 在你的项目中使用
result = model.optimize_latent(
    task_name="your_task",
    mode="max",
    num_restarts=10
)
```

---

**文档齐全，开始探索吧！** 🎉
