"""
最小化的 Latent 优化示例 - 核心算法演示

这个脚本展示了通过自动微分优化 latent space 的核心原理，
不依赖完整的 FlexibleMultiTaskModel。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class SimpleTaskHead(nn.Module):
    """简单的任务头，用于演示"""

    def __init__(self, latent_dim=10, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class SimpleDecoder(nn.Module):
    """简单的解码器（AutoEncoder 的 decoder 部分）"""

    def __init__(self, latent_dim=10, input_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, z):
        return self.net(z)


def optimize_latent_minimal(
    task_head: nn.Module,
    decoder: nn.Module = None,
    initial_latent: torch.Tensor = None,
    latent_dim: int = 10,
    mode: str = "max",
    steps: int = 200,
    lr: float = 0.1,
):
    """
    通过自动微分优化 latent representation 的最小实现。

    核心算法的精简版本，用于教学和理解。

    Parameters
    ----------
    task_head : nn.Module
        任务头网络 (固定参数)
    decoder : nn.Module, optional
        解码器网络 (固定参数)
    initial_latent : torch.Tensor, optional
        初始 latent，如果为 None 则随机初始化
    latent_dim : int
        Latent 维度
    mode : str
        "max" 或 "min"
    steps : int
        优化步数
    lr : float
        学习率

    Returns
    -------
    dict
        包含优化结果的字典
    """

    # ========================================================================
    # 步骤 1: 初始化要优化的 latent 变量
    # ========================================================================

    if initial_latent is None:
        # 从标准正态分布随机采样
        initial_latent = torch.randn(1, latent_dim)
        print(f"Random initialization: latent ~ N(0, 1)")
    else:
        print(f"Using provided initial latent")

    # 关键: 从计算图分离并设置 requires_grad=True
    latent = initial_latent.detach().clone().requires_grad_(True)
    #        ^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #        分离计算图               设为优化变量

    print(f"Initial latent shape: {latent.shape}")
    print(f"Initial latent requires_grad: {latent.requires_grad}\n")

    # ========================================================================
    # 步骤 2: 创建优化器（只优化 latent，不优化模型参数）
    # ========================================================================

    optimizer = optim.Adam([latent], lr=lr)
    #                      ^^^^^^^^
    #                      只传入 latent，不传模型参数

    print(f"Optimizer: Adam, lr={lr}")
    print(f"Optimizing variables: latent only")
    print(f"Model parameters: frozen\n")

    # ========================================================================
    # 步骤 3: 优化循环
    # ========================================================================

    sign = 1.0 if mode == "max" else -1.0
    history = []

    # 固定模型为评估模式
    task_head.eval()
    if decoder is not None:
        decoder.eval()

    print(f"Starting optimization ({mode}imization)...")
    print("-" * 50)

    for step in range(steps):
        # --------------------------------------------------------------------
        # 3.1 清零梯度
        # --------------------------------------------------------------------
        optimizer.zero_grad()

        # --------------------------------------------------------------------
        # 3.2 前向传播: latent → task_head → prediction
        # --------------------------------------------------------------------
        pred = task_head(latent)
        #      ^^^^^^^^^^^^^^^^^^^^^
        #      只通过任务头，不通过编码器
        #      因为我们直接优化 latent

        # --------------------------------------------------------------------
        # 3.3 计算损失
        # --------------------------------------------------------------------
        # 最大化: loss = -pred (使得梯度下降 = 梯度上升)
        # 最小化: loss = +pred (标准梯度下降)
        loss = -sign * pred.sum()

        # --------------------------------------------------------------------
        # 3.4 反向传播: 计算 ∂loss/∂latent
        # --------------------------------------------------------------------
        loss.backward()
        #
        # PyTorch 自动计算:
        #   gradients = ∂loss/∂latent
        #
        # 存储在: latent.grad

        # --------------------------------------------------------------------
        # 3.5 梯度下降更新 latent
        # --------------------------------------------------------------------
        optimizer.step()
        #
        # 更新规则:
        #   latent_new = latent_old - lr * ∂loss/∂latent

        # --------------------------------------------------------------------
        # 3.6 记录历史
        # --------------------------------------------------------------------
        with torch.no_grad():
            score = pred.item()
            history.append(score)

            if step % 50 == 0 or step == steps - 1:
                print(f"Step {step:3d}: score = {score:.6f}, "
                      f"latent_norm = {latent.norm().item():.4f}")

    print("-" * 50)

    # ========================================================================
    # 步骤 4: 获取优化结果
    # ========================================================================

    with torch.no_grad():
        optimized_latent = latent.detach()
        optimized_score = task_head(optimized_latent)

        # 如果有解码器，重构回输入空间
        reconstructed = None
        if decoder is not None:
            reconstructed = decoder(optimized_latent)
            print(f"\nReconstructed input shape: {reconstructed.shape}")

    print(f"\nOptimization complete!")
    print(f"Initial score: {history[0]:.6f}")
    print(f"Final score: {optimized_score.item():.6f}")
    print(f"Improvement: {optimized_score.item() - history[0]:+.6f}")

    return {
        "optimized_latent": optimized_latent,
        "optimized_score": optimized_score,
        "reconstructed": reconstructed,
        "history": history,
    }


def visualize_optimization(result):
    """可视化优化过程"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 优化曲线
    axes[0].plot(result["history"], linewidth=2)
    axes[0].axhline(result["history"][0], color='gray',
                    linestyle='--', label='Initial')
    axes[0].axhline(result["optimized_score"].item(), color='red',
                    linestyle='--', label='Final')
    axes[0].set_xlabel("Optimization Step")
    axes[0].set_ylabel("Property Score")
    axes[0].set_title("Optimization Progress")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Latent 可视化（前10维）
    latent_vals = result["optimized_latent"].numpy().flatten()[:10]
    axes[1].bar(range(len(latent_vals)), latent_vals, alpha=0.7)
    axes[1].set_xlabel("Latent Dimension")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Optimized Latent (first 10 dims)")
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("optimization_result.png", dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: optimization_result.png")
    plt.show()


def demo_gradient_computation():
    """演示梯度计算的细节"""

    print("\n" + "=" * 70)
    print("演示: 自动微分计算 ∂property/∂latent")
    print("=" * 70 + "\n")

    # 创建简单网络
    latent_dim = 3
    net = nn.Sequential(
        nn.Linear(latent_dim, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )

    # 创建 latent（需要梯度）
    latent = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    print(f"Initial latent: {latent}")
    print(f"Network: {net}\n")

    # 前向传播
    output = net(latent)
    print(f"Forward pass: latent → network → output")
    print(f"Output: {output.item():.6f}\n")

    # 反向传播
    loss = -output  # 最大化 output
    loss.backward()

    print(f"Backward pass: computing ∂(-output)/∂latent")
    print(f"Gradient: {latent.grad}")
    print(f"\n解释: 这个梯度告诉我们如何调整 latent 来增大 output")
    print(f"      如果 grad[i] > 0, 增加 latent[i] 会增大 output")
    print(f"      如果 grad[i] < 0, 减少 latent[i] 会增大 output")


def main():
    """主函数：运行完整示例"""

    print("=" * 70)
    print("最小化 Latent 优化演示 - 核心算法")
    print("=" * 70)

    # 设置随机种子
    torch.manual_seed(42)

    # 创建模型
    latent_dim = 10
    input_dim = 50

    task_head = SimpleTaskHead(latent_dim=latent_dim, output_dim=1)
    decoder = SimpleDecoder(latent_dim=latent_dim, input_dim=input_dim)

    # 随机初始化模型参数（模拟训练好的模型）
    print("\nInitializing models...")
    print(f"Task Head: {latent_dim}D latent → property (1D)")
    print(f"Decoder: {latent_dim}D latent → {input_dim}D input")

    # 运行优化
    print("\n" + "=" * 70)
    print("Running optimization...")
    print("=" * 70 + "\n")

    result = optimize_latent_minimal(
        task_head=task_head,
        decoder=decoder,
        initial_latent=None,  # 随机初始化
        latent_dim=latent_dim,
        mode="max",
        steps=200,
        lr=0.1,
    )

    # 可视化
    visualize_optimization(result)

    # 梯度计算演示
    demo_gradient_computation()

    # 多重启动演示
    print("\n" + "=" * 70)
    print("演示: 多重启动避免局部最优")
    print("=" * 70 + "\n")

    best_score = -float('inf')
    best_result = None

    for i in range(5):
        print(f"\n--- Restart {i+1}/5 ---")
        result = optimize_latent_minimal(
            task_head=task_head,
            decoder=decoder,
            initial_latent=None,  # 每次随机初始化
            latent_dim=latent_dim,
            mode="max",
            steps=100,
            lr=0.1,
        )

        if result["optimized_score"].item() > best_score:
            best_score = result["optimized_score"].item()
            best_result = result

    print("\n" + "=" * 70)
    print(f"Best result across 5 restarts: {best_score:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
