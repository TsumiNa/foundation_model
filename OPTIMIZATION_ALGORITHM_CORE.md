# 自动微分优化算法核心实现详解

## 🎯 核心算法：Latent Space Gradient Optimization

### 数学原理

**问题定义**：
```
给定训练好的模型: x → Encoder(x) = z → TaskHead(z) = y

优化目标: 找到 z* 使得 y = TaskHead(z) 达到极值

重构: x* = Decoder(z*)
```

**优化算法**：
```python
for step in range(max_steps):
    # 1. 前向传播
    y = TaskHead(z)

    # 2. 计算损失（最大化时取负）
    loss = -y  # for maximization

    # 3. 反向传播（只更新 z，模型参数固定）
    loss.backward()

    # 4. 优化器更新 z
    optimizer.step(z)
```

---

## 💻 核心代码实现（带详细注释）

### Part 1: 初始化优化变量

```python
# ============================================================================
# 步骤 1: 初始化 latent representation
# ============================================================================

# 方式 1: 从输入编码得到初始 latent
if initial_input is not None:
    with torch.no_grad():
        _, initial_latent = self.encoder(initial_input)
        # initial_latent.shape: (1, latent_dim)

# 方式 2: 从随机向量开始
else:
    initial_latent = torch.randn(1, latent_dim, device=device)
    # 从标准正态分布 N(0,1) 采样

# 可选: 添加扰动探索不同起点
if perturbation_std > 0:
    noise = torch.randn_like(initial_latent) * perturbation_std
    initial_latent = initial_latent + noise
    # 添加高斯噪声: z_perturbed ~ N(z_init, σ²I)

# ============================================================================
# 步骤 2: 创建可优化的 latent（关键步骤！）
# ============================================================================

# 从计算图中分离并设置 requires_grad=True
latent = initial_latent.detach().clone().requires_grad_(True)
#        ^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#        从模型计算图分离          使其成为优化变量
#
# 这样 latent 不受模型参数影响，只能通过梯度下降更新

# ============================================================================
# 步骤 3: 创建优化器（只优化 latent，不优化模型参数）
# ============================================================================

optimizer = optim.Adam([latent], lr=lr)
#                      ^^^^^^^^
#                      只传入 latent 作为优化参数
#                      模型参数（encoder, task_head）不会被更新
```

---

### Part 2: 优化循环（核心算法）

```python
# ============================================================================
# 优化循环: 通过自动微分更新 latent
# ============================================================================

sign = 1.0 if mode == "max" else -1.0
# 最大化: 优化 -loss (梯度上升)
# 最小化: 优化 +loss (梯度下降)

for step in range(steps):
    # ------------------------------------------------------------------------
    # 步骤 1: 清零梯度
    # ------------------------------------------------------------------------
    optimizer.zero_grad()

    # ------------------------------------------------------------------------
    # 步骤 2: 前向传播（只通过 task head）
    # ------------------------------------------------------------------------
    pred = self.task_heads[task_name](latent)
    #      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #      latent → task_head → prediction
    #
    #      关键: 只使用 task_head，不使用 encoder
    #      因为我们直接优化 latent，不需要从输入编码

    # pred.shape: (1, output_dim)  例如: (1, 1) for scalar property

    # ------------------------------------------------------------------------
    # 步骤 3: 计算损失
    # ------------------------------------------------------------------------
    loss = -sign * pred.sum()
    #      ^^^^^^^^^^^^^^^^^^^^
    #      最大化: loss = -pred  (梯度下降 = 梯度上升负值)
    #      最小化: loss = +pred  (标准梯度下降)

    # ------------------------------------------------------------------------
    # 步骤 4: 反向传播（计算 ∂loss/∂latent）
    # ------------------------------------------------------------------------
    loss.backward()
    #
    # PyTorch 自动计算:
    #   ∂loss/∂latent = ∂loss/∂pred · ∂pred/∂latent
    #
    # 梯度存储在: latent.grad

    # ------------------------------------------------------------------------
    # 步骤 5: 更新 latent（梯度下降步）
    # ------------------------------------------------------------------------
    optimizer.step()
    #
    # Adam 更新规则:
    #   m_t = β₁ · m_{t-1} + (1-β₁) · ∇loss        # 一阶矩估计
    #   v_t = β₂ · v_{t-1} + (1-β₂) · (∇loss)²    # 二阶矩估计
    #   latent_new = latent_old - α · m_t / (√v_t + ε)
    #
    # 其中:
    #   α = learning rate
    #   β₁ = 0.9 (默认)
    #   β₂ = 0.999 (默认)

    # ------------------------------------------------------------------------
    # 步骤 6: 记录优化历史（可选）
    # ------------------------------------------------------------------------
    with torch.no_grad():
        score = pred.item()
        history.append(score)
```

---

### Part 3: 重构优化后的 descriptor

```python
# ============================================================================
# 使用 AutoEncoder 重构 descriptor
# ============================================================================

with torch.no_grad():
    # ------------------------------------------------------------------------
    # 获取最终优化的 latent
    # ------------------------------------------------------------------------
    optimized_latent = latent.detach()
    # shape: (1, latent_dim)

    # ------------------------------------------------------------------------
    # 计算最终任务分数
    # ------------------------------------------------------------------------
    optimized_score = self.task_heads[task_name](optimized_latent)
    # shape: (1, 1)

    # ------------------------------------------------------------------------
    # 通过 AutoEncoder decoder 重构输入
    # ------------------------------------------------------------------------
    if ae_task_name is not None:
        reconstructed_input = self.task_heads[ae_task_name](optimized_latent)
        #                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                     z* → Decoder → x*
        #
        #                     AutoEncoderHead 就是 decoder
        #
        # reconstructed_input.shape: (1, input_dim)
        #
        # 这就是优化后的材料描述符！
```

---

## 🔍 关键技术细节

### 1. 为什么用 `detach().clone()`？

```python
latent = initial_latent.detach().clone().requires_grad_(True)
```

**原因**：
- `detach()`: 从原始计算图分离，避免反向传播到 encoder
- `clone()`: 创建副本，避免修改原始数据
- `requires_grad_(True)`: 使其成为优化变量

**如果不 detach()**：
```python
# ❌ 错误做法
latent = initial_latent.requires_grad_(True)

# 问题: backward() 会尝试更新 encoder 的参数
# 导致:
# 1. 优化目标混乱（既优化 latent 又优化 encoder）
# 2. 可能破坏训练好的 encoder
```

### 2. 为什么只传 `[latent]` 给优化器？

```python
optimizer = optim.Adam([latent], lr=lr)
```

**原因**：
- 只有 `latent` 需要被优化
- 模型参数（encoder, task_head）保持固定
- 这是 **变量优化** 而非 **参数优化**

**对比训练时的优化器**：
```python
# 训练时: 优化模型参数
optimizer = optim.Adam(model.parameters(), lr=lr)

# 优化 latent 时: 只优化 latent 变量
optimizer = optim.Adam([latent], lr=lr)
```

### 3. 为什么用 `loss = -sign * pred.sum()`？

```python
sign = 1.0 if mode == "max" else -1.0
loss = -sign * pred.sum()
```

**数学解释**：

| 目标 | sign | loss | 梯度下降效果 |
|------|------|------|--------------|
| 最大化 y | +1 | -y | min(-y) = max(y) ✓ |
| 最小化 y | -1 | +y | min(+y) = min(y) ✓ |

**为什么不直接用梯度上升**？
```python
# 也可以这样实现
if mode == "max":
    # 梯度上升
    pred = self.task_heads[task_name](latent)
    (-pred).backward()  # 负梯度方向
    optimizer.step()
else:
    # 梯度下降
    pred = self.task_heads[task_name](latent)
    pred.backward()
    optimizer.step()

# 但用 loss = -sign * pred 更简洁统一
```

### 4. 计算图示例

```
初始化阶段:
    initial_input → Encoder → initial_latent
                    ^^^^^^^^
                    requires_grad=False (固定参数)
                                ↓
                           detach().clone()
                                ↓
                            latent (requires_grad=True)

优化阶段 (每一步):
    latent → TaskHead → pred → loss
    ^^^^^^   ^^^^^^^^^
    可优化     固定参数

    backward():
    latent ← ∂loss/∂latent
    ^^^^^
    更新这个

重构阶段:
    optimized_latent → AutoEncoderHead → reconstructed_input
                       ^^^^^^^^^^^^^^^^
                       就是 decoder
```

---

## 📊 与标准优化问题的对比

### 标准神经网络训练
```python
# 优化目标: 模型参数 θ
# 固定: 输入数据 x, 标签 y

for epoch in range(epochs):
    pred = model(x; θ)  # θ 是参数
    loss = criterion(pred, y)
    loss.backward()  # 计算 ∂loss/∂θ
    optimizer.step()  # 更新 θ
```

### Latent 优化（本算法）
```python
# 优化目标: latent z
# 固定: 模型参数 θ

for step in range(steps):
    pred = task_head(z; θ)  # θ 固定
    loss = -pred  # 最大化 pred
    loss.backward()  # 计算 ∂loss/∂z
    optimizer.step()  # 更新 z
```

**关键区别**：
- 训练: 优化参数，固定数据
- Latent优化: 优化数据表示，固定参数

---

## 🎨 可视化理解

### 优化轨迹
```
Latent Space (2D visualization):

初始点 z₀ •
           ↘
            • z₁
             ↘
              • z₂    ← 梯度方向
               ↘
                • z₃
                 ↘
                  • z* (最优点)

Property value:
f(z₀) = 1.2
f(z₁) = 1.5
f(z₂) = 1.8
f(z₃) = 2.1
f(z*) = 2.3  ← 最大值
```

### 多重启动效果
```
Latent Space with multiple restarts:

Restart 1: •────→ • (local max, score=2.1)
Restart 2: •──────────→ • (global max, score=2.8) ✓
Restart 3: •───→ • (local max, score=1.9)
Restart 4: •────────→ • (local max, score=2.3)
Restart 5: •───→ • (local max, score=2.0)

选择: Restart 2 的结果
```

---

## 🔬 完整代码流程总结

```python
def optimize_latent(self, task_name, initial_input, mode, ...):

    # ========================================
    # Phase 1: 初始化
    # ========================================

    # 1.1 获取或生成初始 latent
    if initial_input is not None:
        _, initial_latent = self.encoder(initial_input)
    else:
        initial_latent = torch.randn(1, latent_dim)

    # 1.2 添加扰动（可选）
    if perturbation_std > 0:
        initial_latent += torch.randn_like(initial_latent) * perturbation_std

    # 1.3 创建可优化变量
    latent = initial_latent.detach().clone().requires_grad_(True)

    # 1.4 创建优化器
    optimizer = optim.Adam([latent], lr=lr)

    # ========================================
    # Phase 2: 优化循环
    # ========================================

    for step in range(steps):
        # 2.1 清零梯度
        optimizer.zero_grad()

        # 2.2 前向传播
        pred = self.task_heads[task_name](latent)

        # 2.3 计算损失
        loss = -sign * pred.sum()

        # 2.4 反向传播
        loss.backward()  # 计算 ∂loss/∂latent

        # 2.5 更新 latent
        optimizer.step()  # latent ← latent - lr * ∇loss

    # ========================================
    # Phase 3: 重构
    # ========================================

    with torch.no_grad():
        # 3.1 获取优化后的 latent
        optimized_latent = latent.detach()

        # 3.2 计算最终分数
        optimized_score = self.task_heads[task_name](optimized_latent)

        # 3.3 重构 descriptor
        if ae_task_name:
            reconstructed_input = self.task_heads[ae_task_name](optimized_latent)
            # AutoEncoderHead: latent → decoder → input_space

    return {
        'optimized_latent': optimized_latent,
        'optimized_score': optimized_score,
        'reconstructed_input': reconstructed_input,
    }
```

---

## 🧮 数值例子

假设我们优化密度（density）：

```python
# 初始状态
initial_input: [0.1, 0.3, -0.2, ...]  # 190维
     ↓ encoder
initial_latent: [0.5, -0.1, 0.3, ...]  # 128维
     ↓ task_head
initial_density: 1.23 g/cm³

# 优化过程（每一步）
Step 1: latent = [0.5, -0.1, 0.3, ...] → density = 1.23
        ∇density/∇latent = [0.02, 0.05, -0.01, ...]
        latent ← latent + lr * ∇ = [0.51, -0.05, 0.29, ...]

Step 2: latent = [0.51, -0.05, 0.29, ...] → density = 1.28
        ∇density/∇latent = [0.03, 0.04, -0.02, ...]
        latent ← [0.525, 0.0, 0.27, ...]

...

Step 200: latent = [0.8, 0.2, 0.1, ...] → density = 2.15 g/cm³

# 重构
optimized_latent: [0.8, 0.2, 0.1, ...]  # 128维
     ↓ autoencoder decoder
reconstructed_input: [0.2, 0.5, 0.1, ...]  # 190维
     ↓ 这就是新材料的描述符！

# 验证
reconstructed_input → encoder → latent' ≈ optimized_latent
latent' → task_head → density' ≈ 2.15 g/cm³ ✓
```

---

## 💡 关键洞察

1. **自动微分的力量**
   - 不需要手动推导 ∂property/∂latent
   - PyTorch 自动计算所有梯度
   - 支持任意复杂的神经网络

2. **变量优化 vs 参数优化**
   - 训练: 固定数据，优化参数
   - Latent优化: 固定参数，优化表示

3. **为什么需要 AutoEncoder**
   - Latent 空间是抽象的，无法直接解释
   - AutoEncoder 将其映射回可解释的 descriptor 空间
   - Descriptor 可以指导实际材料合成

4. **多重启动的重要性**
   - Latent 空间可能有多个局部最优
   - 多起点探索增加找到全局最优的概率
   - 类似于遗传算法的种群思想

---

## 📚 相关技术

这个算法与以下技术相关：

1. **对抗样本生成（Adversarial Examples）**
   - 也是优化输入来欺骗模型
   - 区别: 我们是寻找有意义的极值，不是欺骗

2. **DeepDream / Neural Style Transfer**
   - 也是优化输入来最大化某些激活
   - 区别: 我们优化 latent，然后重构

3. **Bayesian Optimization**
   - 也用于黑盒优化
   - 区别: 我们利用了梯度信息（更高效）

4. **GAN 中的 latent space manipulation**
   - 也是在 latent space 中寻找特定属性
   - 区别: 我们用梯度下降，GAN用采样

---

这就是核心算法！简单但强大 🚀
