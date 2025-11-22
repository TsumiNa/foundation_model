# AutoEncoder Implementation Fixes and Enhancements

## Summary

This document summarizes all fixes and enhancements made to support AutoEncoder-based latent optimization in the foundation model.

## 1. Fixed AutoEncoderHead Implementation

**File**: [src/foundation_model/models/task_head/autoencoder.py](src/foundation_model/models/task_head/autoencoder.py)

### Issues Found:
1. **Missing `_predict_impl` method** (Critical Bug)
   - BaseTaskHead requires all subclasses to implement the abstract method `_predict_impl`
   - Original implementation only raised `NotImplementedError`

2. **Missing imports**
   - Missing `Dict` from typing
   - Missing `ndarray` from numpy

3. **Inconsistent code style in `compute_loss`**
   - Implementation was mathematically correct but stylistically different from other task heads

### Fixes Applied:

```python
# Added proper imports
from typing import Dict, Optional
from numpy import ndarray

# Implemented _predict_impl method
def _predict_impl(self, x: torch.Tensor) -> Dict[str, ndarray]:
    """
    Core prediction logic for autoencoder.

    Returns
    -------
    Dict[str, ndarray]
        A dictionary containing the reconstructed values: {"reconstruction": x}.
    """
    return {"reconstruction": x.detach().cpu().numpy()}

# Standardized compute_loss implementation
def compute_loss(self, pred, target, mask=None):
    if mask is None:
        mask = torch.ones_like(target)

    # Check if there are any valid samples
    valid_count = mask.sum()
    if valid_count == 0:
        return None

    # Apply mask to both predictions and targets
    losses = F.mse_loss(pred, target, reduction="none") * mask

    # Compute total loss
    total_loss = losses.sum() / valid_count

    return total_loss
```

## 2. Added `optimize_latent` Method to FlexibleMultiTaskModel

**File**: [src/foundation_model/models/flexible_multi_task_model.py](src/foundation_model/models/flexible_multi_task_model.py:1486-1634)

### New Functionality:

The `optimize_latent` method enables gradient-based optimization of latent representations to find inputs that maximize or minimize target properties.

**Key Features**:
- Supports both maximization and minimization
- Uses Adam optimizer for stable convergence
- Optional autoencoder reconstruction of optimized latents
- Returns optimization history for visualization
- Proper error handling and validation

**Method Signature**:
```python
def optimize_latent(
    self,
    task_name: str,           # Target regression task to optimize
    initial_input: torch.Tensor,  # Starting point (1, input_dim)
    mode: str = "max",        # "max" or "min"
    steps: int = 200,         # Number of optimization steps
    lr: float = 0.1,          # Learning rate
    ae_task_name: str | None = None,  # Optional autoencoder for reconstruction
) -> dict[str, torch.Tensor | list[float]]
```

**Return Values**:
- `optimized_latent`: Optimized latent representation (1, latent_dim)
- `optimized_score`: Final task output value
- `reconstructed_input`: Reconstructed input from autoencoder (if provided)
- `history`: List of task scores at each optimization step

**Example Usage**:
```python
result = model.optimize_latent(
    task_name="density",
    initial_input=torch.randn(1, 190),
    mode="max",
    steps=300,
    lr=0.05,
    ae_task_name="reconstruction"
)

print(f"Optimized density: {result['optimized_score'].item():.4f}")
reconstructed_descriptor = result['reconstructed_input']
```

## 3. Created Comprehensive Verification Notebook

**File**: [notebooks/verify_autoencoder_optimization.ipynb](notebooks/verify_autoencoder_optimization.ipynb)

### Notebook Structure:

1. **Data Loading**
   - Supports both real polymer data and synthetic fallback data
   - Uses the same data paths as `dynamic_task_suite_config_radonpy.toml`
   - Handles both normalized and raw density columns

2. **Model Configuration**
   - MLP encoder with dimensions matching production config
   - Regression task for density prediction
   - AutoEncoder task for reconstruction
   - Proper optimizer configurations

3. **Training**
   - Lightning Trainer with early stopping
   - Joint training of density prediction and reconstruction
   - Progress monitoring and logging

4. **Model Evaluation**
   - Density prediction metrics (MAE, R²)
   - Reconstruction error analysis
   - Visualization of results

5. **Latent Optimization**
   - Maximize density optimization
   - Minimize density optimization
   - Progress visualization
   - Descriptor comparison

6. **Validation**
   - Verifies reconstructed inputs preserve optimized property values
   - Demonstrates end-to-end pipeline

### Key Visualizations:
- Density prediction scatter plot
- Reconstruction error distribution
- Optimization progress curves
- Descriptor feature comparisons

## 4. Data Pipeline Compatibility

The existing data pipeline already supports AutoEncoder tasks correctly:

**In CompoundDataset** ([src/foundation_model/data/dataset.py:155](src/foundation_model/data/dataset.py#L155)):
```python
if task_type == TaskType.AUTOENCODER:
    # AutoEncoder tasks use the input features as target
    # Skip y_dict and task_masks_dict population
    continue
```

**In training_step** ([src/foundation_model/models/flexible_multi_task_model.py:634](src/foundation_model/models/flexible_multi_task_model.py#L634)):
```python
if isinstance(head, AutoEncoderHead):
    target = x  # Use input as target
    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
```

This ensures autoencoder tasks automatically use input features as reconstruction targets without requiring explicit target data.

## Testing and Validation

### Run the Verification Notebook:
```bash
jupyter notebook notebooks/verify_autoencoder_optimization.ipynb
```

### Expected Results:
1. ✅ Model trains successfully with both tasks
2. ✅ Density predictions achieve reasonable R² scores
3. ✅ Reconstruction errors remain low
4. ✅ Latent optimization finds extreme density values
5. ✅ Reconstructed inputs preserve optimized properties

### Integration with Production Pipeline:

To add autoencoder to the dynamic task suite:

```python
# In dynamic_task_suite.py
from foundation_model.models.model_config import AutoEncoderTaskConfig

# Add autoencoder task configuration
ae_task = AutoEncoderTaskConfig(
    name="reconstruction",
    data_column="__autoencoder__",  # Special marker
    dims=[latent_dim, 256, input_dim],
    norm=True,
    residual=False,
    loss_weight=0.1,  # Lower weight for auxiliary task
)

# Include in task_configs
task_configs = [density_task, ae_task]
```

## Benefits of This Implementation

1. **Inverse Design**: Find material descriptors that achieve extreme property values
2. **Latent Space Exploration**: Discover relationships between latent representations and properties
3. **Property Optimization**: Gradient-based search for optimal materials
4. **Interpretability**: Reconstruct optimized descriptors for analysis

## Future Enhancements

1. **Regularization**: Add constraints to keep optimized latents realistic
2. **Multi-objective**: Optimize multiple properties simultaneously
3. **Variational AE**: Add KL divergence for better latent space structure
4. **Conditional Generation**: Guide optimization with additional constraints

## Related Files

- AutoEncoder task head: [src/foundation_model/models/task_head/autoencoder.py](src/foundation_model/models/task_head/autoencoder.py)
- Model implementation: [src/foundation_model/models/flexible_multi_task_model.py](src/foundation_model/models/flexible_multi_task_model.py)
- Verification notebook: [notebooks/verify_autoencoder_optimization.ipynb](notebooks/verify_autoencoder_optimization.ipynb)
- Config example: [samples/dynamic_task_suite_config_radonpy.toml](samples/dynamic_task_suite_config_radonpy.toml)
