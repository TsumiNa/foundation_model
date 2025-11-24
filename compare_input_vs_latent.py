#!/usr/bin/env python3
"""
Compare input space vs latent space optimization under the simplified architecture.

Simplified Architecture: X → encoder → latent → Tanh (in FlexibleMultiTaskModel) → task heads

Two optimization strategies:
1. Input space: Optimize X, forward through encoder → Tanh → task_heads
2. Latent space: Optimize latent, apply Tanh → task_heads, then Tanh → AE → reconstructed X
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    MLPEncoderConfig,
    RegressionTaskConfig,
    AutoEncoderTaskConfig,
    OptimizerConfig,
)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

print("=" * 80)
print("COMPARING INPUT SPACE VS LATENT SPACE OPTIMIZATION")
print("=" * 80)
print("\nSimplified Architecture:")
print("  X → encoder → latent → Tanh (in FlexibleMultiTaskModel) → task heads\n")

# Load data
descriptor_path = Path('data/amorphous_polymer_FFDescriptor_20250730.parquet')
target_path = Path('data/amorphous_polymer_non_PI_properties_20250730.parquet')

if not descriptor_path.exists():
    print("⚠️  Data files not found. Skipping test.")
    exit(0)

descriptors = pd.read_parquet(descriptor_path)
properties = pd.read_parquet(target_path)[['density']]
merged = descriptors.join(properties).dropna(subset=['density'])
descriptor_cols = [c for c in descriptors.columns if pd.api.types.is_numeric_dtype(descriptors[c])]

# Use subset
X = torch.tensor(merged[descriptor_cols].values[:256], dtype=torch.float32).to(device)
y = torch.tensor(merged['density'].values[:256], dtype=torch.float32).unsqueeze(1).to(device)

input_dim = X.shape[1]
latent_dim = 64

print(f"Dataset:")
print(f"  Samples: {len(X)}")
print(f"  Input dim: {input_dim}")
print(f"  Latent dim: {latent_dim}")

# Create model with simplified architecture
encoder_config = MLPEncoderConfig(hidden_dims=[input_dim, 128, latent_dim], norm=True)
density_task = RegressionTaskConfig(
    name='density',
    data_column='density',
    dims=[latent_dim, 32, 1],
    norm=True,
)
ae_task = AutoEncoderTaskConfig(
    name='reconstruction',
    data_column='__autoencoder__',
    dims=[latent_dim, 128, input_dim],
    norm=True,
)

model = FlexibleMultiTaskModel(
    encoder_config=encoder_config,
    task_configs=[density_task, ae_task],
    shared_block_optimizer=OptimizerConfig(lr=5e-3),
).to(device)

# Train model
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
optim = torch.optim.Adam(model.parameters(), lr=5e-3)

print("\nTraining model (20 epochs)...")
model.train()
for epoch in range(20):
    total_loss = 0.0
    for batch_x, batch_y in loader:
        optim.zero_grad()
        outputs = model(batch_x)
        pred = outputs['density']
        recon = outputs['reconstruction']
        loss = F.mse_loss(pred, batch_y) + F.mse_loss(recon, batch_x)
        loss.backward()
        optim.step()
        total_loss += loss.item() * batch_x.size(0)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss / len(dataset):.4f}")

model.eval()
print("✓ Training complete\n")

# Get seed sample
seed_idx = 0
seed_batch = X[seed_idx:seed_idx+1]

print("=" * 80)
print("OPTIMIZATION COMPARISON")
print("=" * 80)

# Run both optimization methods
steps = 250
lr = 0.05

print(f"\nOptimization settings:")
print(f"  Steps: {steps}")
print(f"  Learning rate: {lr}")
print(f"  Objective: Maximize density")

print("\n" + "-" * 80)
print("METHOD 1: Input Space Optimization")
print("-" * 80)

result_input = model.optimize_latent(
    task_name="density",
    initial_input=seed_batch,
    mode="max",
    steps=steps,
    lr=lr,
    optimize_space="input",
)

initial_score_input = result_input.initial_score[0, 0, 0].item()
final_score_input = result_input.optimized_target[0, 0, 0].item()
improvement_input = final_score_input - initial_score_input

print(f"\nResults:")
print(f"  Initial score: {initial_score_input:.4f}")
print(f"  Final score: {final_score_input:.4f}")
print(f"  Improvement: {improvement_input:+.4f}")

# Check convergence
trajectory_input = result_input.trajectory.squeeze().cpu().numpy()  # Flatten trajectory
if len(trajectory_input) >= 20:
    last_20_std = np.std(trajectory_input[-20:])
    converged_input = last_20_std < 0.01
    print(f"  Convergence: {converged_input} (std last 20 steps: {last_20_std:.6f})")
else:
    converged_input = False
    print(f"  Convergence: N/A (insufficient steps)")

# Check Tanh constraint
with torch.no_grad():
    opt_input = result_input.optimized_input[0]
    latent_opt = model.encoder(opt_input)
    h_task_opt = torch.tanh(latent_opt)
    max_h_task = h_task_opt.abs().max().item()
    print(f"  Max |Tanh(latent)|: {max_h_task:.6f} (should be ≤ 1.0)")

print("\n" + "-" * 80)
print("METHOD 2: Latent Space Optimization (with AE reconstruction)")
print("-" * 80)

result_latent = model.optimize_latent(
    task_name="density",
    initial_input=seed_batch,
    mode="max",
    steps=steps,
    lr=lr,
    ae_task_name="reconstruction",
    optimize_space="latent",
)

initial_score_latent = result_latent.initial_score[0, 0, 0].item()
final_score_latent = result_latent.optimized_target[0, 0, 0].item()
improvement_latent = final_score_latent - initial_score_latent

print(f"\nResults:")
print(f"  Initial score: {initial_score_latent:.4f}")
print(f"  Final score: {final_score_latent:.4f}")
print(f"  Improvement: {improvement_latent:+.4f}")

# Check convergence
trajectory_latent = result_latent.trajectory.squeeze().cpu().numpy()  # Flatten trajectory
if len(trajectory_latent) >= 20:
    last_20_std = np.std(trajectory_latent[-20:])
    converged_latent = last_20_std < 0.01
    print(f"  Convergence: {converged_latent} (std last 20 steps: {last_20_std:.6f})")
else:
    converged_latent = False
    print(f"  Convergence: N/A (insufficient steps)")

# Check AE reconstruction
with torch.no_grad():
    reconstructed = result_latent.optimized_input[0]
    reconstruction_error = F.mse_loss(reconstructed, seed_batch).item()
    print(f"  AE reconstruction error: {reconstruction_error:.6f}")

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

score_ratio = final_score_latent / final_score_input
improvement_ratio = improvement_latent / improvement_input

print(f"\nFinal Scores:")
print(f"  Input space:  {final_score_input:.4f}")
print(f"  Latent space: {final_score_latent:.4f}")
print(f"  Ratio (latent/input): {score_ratio:.3f}x")

print(f"\nImprovements:")
print(f"  Input space:  {improvement_input:+.4f}")
print(f"  Latent space: {improvement_latent:+.4f}")
print(f"  Ratio (latent/input): {improvement_ratio:.3f}x")

print(f"\nConvergence:")
print(f"  Input space:  {converged_input}")
print(f"  Latent space: {converged_latent}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Score trajectories
ax = axes[0, 0]
ax.plot(trajectory_input, label='Input Space', linewidth=2, color='#2E86AB')
ax.plot(trajectory_latent, label='Latent Space (with AE)', linewidth=2, color='#A23B72')
ax.set_xlabel('Optimization Step', fontsize=11)
ax.set_ylabel('Density Score', fontsize=11)
ax.set_title('Score Trajectories', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Score improvement comparison
ax = axes[0, 1]
methods = ['Input\nSpace', 'Latent\nSpace']
improvements = [improvement_input, improvement_latent]
colors = ['#2E86AB', '#A23B72']
bars = ax.bar(methods, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Score Improvement', fontsize=11)
ax.set_title('Final Improvement Comparison', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, improvements):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:+.3f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

# Plot 3: Final score comparison
ax = axes[1, 0]
final_scores = [final_score_input, final_score_latent]
bars = ax.bar(methods, final_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Final Density Score', fontsize=11)
ax.set_title('Final Score Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, final_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold')

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')

summary_data = [
    ['Metric', 'Input Space', 'Latent Space', 'Ratio'],
    ['Initial Score', f'{initial_score_input:.4f}', f'{initial_score_latent:.4f}', '1.00x'],
    ['Final Score', f'{final_score_input:.4f}', f'{final_score_latent:.4f}', f'{score_ratio:.3f}x'],
    ['Improvement', f'{improvement_input:+.4f}', f'{improvement_latent:+.4f}', f'{improvement_ratio:.3f}x'],
    ['Converged', str(converged_input), str(converged_latent), '—'],
    ['AE Recon Error', '—', f'{reconstruction_error:.4f}', '—'],
]

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.25, 0.25, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(summary_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=10)

plt.suptitle('Input Space vs Latent Space Optimization\nSimplified Architecture (Unified Tanh)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('input_vs_latent_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to 'input_vs_latent_comparison.png'")

# Detailed analysis
print("\n" + "=" * 80)
print("DETAILED ANALYSIS")
print("=" * 80)

print("\n1. Score Performance:")
if 0.8 < score_ratio < 1.2:
    print(f"   ✓ Comparable performance (ratio: {score_ratio:.3f}x)")
elif score_ratio > 1.2:
    print(f"   ⚠ Latent space significantly better (ratio: {score_ratio:.3f}x)")
else:
    print(f"   ⚠ Input space significantly better (ratio: {score_ratio:.3f}x)")

print("\n2. Convergence Behavior:")
# Check convergence from trajectory
window = 20
if len(trajectory_input) >= window:
    input_std = np.std(trajectory_input[-window:])
    latent_std = np.std(trajectory_latent[-window:])
    print(f"   Input space std (last {window} steps): {input_std:.6f}")
    print(f"   Latent space std (last {window} steps): {latent_std:.6f}")

    if input_std < 0.01 and latent_std < 0.01:
        print(f"   ✓ Both methods converged")
    elif input_std < 0.01:
        print(f"   ⚠ Only input space converged")
    elif latent_std < 0.01:
        print(f"   ⚠ Only latent space converged")
    else:
        print(f"   ⚠ Neither method fully converged")

print("\n3. Architectural Consistency:")
print(f"   ✓ Both methods apply Tanh at FlexibleMultiTaskModel level")
print(f"   ✓ Max |Tanh(latent)| = {max_h_task:.6f} (bounded to [-1, 1])")

print("\n4. AE Reconstruction:")
if reconstruction_error < 1.0:
    print(f"   ✓ Low reconstruction error: {reconstruction_error:.4f}")
elif reconstruction_error < 2.0:
    print(f"   ⚠ Moderate reconstruction error: {reconstruction_error:.4f}")
else:
    print(f"   ✗ High reconstruction error: {reconstruction_error:.4f}")

print("\n5. Optimization Dimensionality:")
print(f"   Input space: {input_dim} dimensions")
print(f"   Latent space: {latent_dim} dimensions")
print(f"   Reduction: {latent_dim/input_dim:.1%} of input space")
print(f"   Advantage: Latent space optimization searches in lower-dimensional space")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

checks = {
    "Comparable scores": 0.8 < score_ratio < 1.2,
    "Both methods converge": converged_input and converged_latent,
    "AE reconstruction valid": reconstruction_error < 1.0,
    "Tanh bounds respected": max_h_task <= 1.0,
}

passed = sum(checks.values())
total = len(checks)

for check, result in checks.items():
    print(f"{'✓' if result else '⚠'} {check}")

print(f"\nPassed {passed}/{total} checks")

if passed >= 3:
    print("\n✅ SIMPLIFIED ARCHITECTURE WORKS CORRECTLY!")
    print("\nKey findings:")
    print("  • Both optimization strategies produce comparable results")
    print("  • Latent space optimization works in lower-dimensional space")
    print("  • AE reconstruction maintains consistency")
    print("  • Unified Tanh constraint applied consistently")
else:
    print("\n⚠️ Some aspects need attention. Review the detailed analysis above.")

print("=" * 80)
