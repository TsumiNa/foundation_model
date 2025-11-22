# Latent Optimization Method Improvements

## 概述

根据你的建议，我重新设计了 `optimize_latent` 方法，使其更加灵活和robust。

## 主要改进

### 1. 灵活的初始化策略

**之前的问题**：
- 必须提供 `initial_input` 参数
- 用户可能随便给一个值导致优化效果不佳
- 没有探索不同起点的机制

**改进后**：
```python
# ✅ 支持随机初始化（从标准正态分布采样 latent）
result = model.optimize_latent(
    task_name="density",
    initial_input=None,  # 不需要提供初始值
    mode="max"
)

# ✅ 支持从已知好的点开始
result = model.optimize_latent(
    task_name="density",
    initial_input=good_sample,  # 提供一个已知的样本
    mode="max"
)
```

### 2. Perturbation 机制避免局部极值

**你的建议**："比如设计成可以添加一个perturbation好了，总比随便取一个初值导致计算出错好"

**实现**：
```python
# 在好的起点附近添加扰动探索
result = model.optimize_latent(
    task_name="density",
    initial_input=good_sample,
    perturbation_std=0.1,  # 添加标准差为0.1的高斯噪声
    mode="max"
)
```

这样可以：
- 从一个已知好的点开始
- 通过添加噪声探索附近区域
- 避免总是收敛到同一个局部极值

### 3. 多重启动（Multi-Restart）避免局部最优

**核心改进**：
```python
# 尝试多个不同的起点，返回最好的结果
result = model.optimize_latent(
    task_name="density",
    initial_input=None,  # 或者提供一个基准点
    mode="max",
    num_restarts=10,  # 尝试10个不同的起点
    perturbation_std=0.2,  # 每次重启添加不同的扰动
    ae_task_name="reconstruction"
)

# 查看最佳结果
print(f"Best score: {result['optimized_score'].item():.4f}")
print(f"From restart: {result['all_restarts'][0]['restart_idx']}")

# 查看所有尝试的结果
for r in result['all_restarts']:
    print(f"Restart {r['restart_idx']}: {r['optimized_score'].item():.4f}")
```

## 新增参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_input` | `Tensor \| None` | `None` | 初始输入，`None` 时随机初始化 |
| `num_restarts` | `int` | `1` | 重启次数，>1 时返回最佳结果 |
| `perturbation_std` | `float` | `0.0` | 添加到初始 latent 的高斯噪声标准差 |
| `latent_dim` | `int \| None` | `None` | Latent 维度（随机初始化时需要） |

## 新增返回值

除了原有的返回值，新增：
- `initial_score`: 起始点的任务值
- `all_restarts`: 所有重启的详细结果（当 `num_restarts > 1` 时）

## 使用场景对比

### 场景 1: 快速探索（单次运行）

```python
# 从随机点开始，快速找到一个极值
result = model.optimize_latent(
    task_name="density",
    initial_input=None,  # 随机初始化
    mode="max",
    steps=200,
    ae_task_name="reconstruction"
)
```

### 场景 2: 在已知好点附近精细搜索

```python
# 假设你已经有一个 density 较高的样本
high_density_sample = train_features[best_idx:best_idx+1]

# 在这个点附近搜索更好的结果
result = model.optimize_latent(
    task_name="density",
    initial_input=high_density_sample,
    perturbation_std=0.05,  # 小扰动，精细搜索
    num_restarts=5,  # 尝试5个不同方向
    mode="max",
    ae_task_name="reconstruction"
)
```

### 场景 3: 全局搜索（多重启动）

```python
# 不确定起点，尝试多个随机起点找全局最优
result = model.optimize_latent(
    task_name="density",
    initial_input=None,  # 每次重启都随机初始化
    mode="max",
    steps=300,
    num_restarts=20,  # 20个不同起点
    ae_task_name="reconstruction"
)

# 分析所有重启的结果
scores = [r['optimized_score'].item() for r in result['all_restarts']]
print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
print(f"Best from restart {result['all_restarts'][scores.index(max(scores))]['restart_idx']}")
```

### 场景 4: 混合策略（已知起点 + 多重启动）

```python
# 从一个好点开始，但添加较大扰动探索更广泛的区域
result = model.optimize_latent(
    task_name="density",
    initial_input=good_sample,
    perturbation_std=0.3,  # 较大扰动
    num_restarts=10,  # 多次重启
    mode="max",
    ae_task_name="reconstruction"
)
```

## 实际应用建议

### 1. 材料设计（寻找极端性质）

```python
# 寻找最高密度的材料
result_max = model.optimize_latent(
    task_name="density",
    initial_input=None,
    mode="max",
    num_restarts=50,  # 充分探索
    steps=500,
    ae_task_name="reconstruction"
)

# 寻找最低密度的材料
result_min = model.optimize_latent(
    task_name="density",
    initial_input=None,
    mode="min",
    num_restarts=50,
    steps=500,
    ae_task_name="reconstruction"
)

# 重构并分析材料描述符
high_density_descriptor = result_max['reconstructed_input']
low_density_descriptor = result_min['reconstructed_input']
```

### 2. 改进已知材料

```python
# 从现有材料出发，寻找性能更好的变体
existing_material = features[100:101]  # 某个已知材料

improved = model.optimize_latent(
    task_name="thermal_conductivity",
    initial_input=existing_material,
    perturbation_std=0.1,  # 小幅改动
    num_restarts=10,
    mode="max",
    ae_task_name="reconstruction"
)

# 比较改进前后
print(f"Original: {improved['initial_score']:.4f}")
print(f"Improved: {improved['optimized_score'].item():.4f}")
print(f"Gain: {improved['optimized_score'].item() - improved['initial_score']:.4f}")
```

### 3. 多目标优化（手动实现）

```python
# 优化多个性质，手动权衡
results = []

for weight_density in [0, 0.3, 0.5, 0.7, 1.0]:
    # 这里需要修改模型支持加权多任务，或分别优化后筛选
    # 示例：分别优化两个任务后根据 Pareto 前沿筛选
    pass
```

## 技术细节

### 为什么支持 `initial_input=None`？

1. **灵活性**：不是所有用户都有好的起点
2. **探索性**：随机初始化可以探索 latent space 的不同区域
3. **避免偏差**：不依赖特定的训练样本

### Perturbation 的作用

- **避免确定性**：同一个起点每次运行可能得到不同结果
- **局部搜索**：在已知好点附近探索
- **鲁棒性**：测试优化结果对初始条件的敏感性

### Multi-Restart 的优势

- **全局最优**：增加找到全局最优解的概率
- **统计分析**：可以分析优化landscape的特性
- **置信度**：多次重启得到相似结果说明解更可靠

## 向后兼容性

原有代码仍然可以正常工作：

```python
# 旧代码（仍然支持）
result = model.optimize_latent(
    task_name="density",
    initial_input=my_input,  # 必须提供
    mode="max",
    ae_task_name="reconstruction"
)
```

## 性能考虑

- `num_restarts=1`（默认）：与旧版本性能相同
- `num_restarts=N`：运行时间约为单次的 N 倍
- 建议：先用少量重启（5-10）测试，再决定是否增加

## 总结

你的建议非常有价值！新的设计：

✅ **更灵活**：支持随机初始化，不强制要求初始输入
✅ **更robust**：通过 perturbation 避免总是困在同一个局部极值
✅ **更可靠**：multi-restart 机制提高找到全局最优的概率
✅ **更实用**：符合实际材料设计的工作流程
✅ **向后兼容**：不影响现有代码

这个设计比原来的"必须提供initial_input"要合理得多！
