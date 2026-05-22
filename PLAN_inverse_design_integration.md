# Plan: Built-in AutoEncoder Head Support

## Background

`FlexibleMultiTaskModel.optimize_latent()` is already implemented. The latent-space exploration
workflow (post-training inverse design) is an **independent system** out of scope here.

This plan covers only the training-time AE support changes.

---

## Confirmed Findings

| Question | Answer |
|----------|--------|
| `LinearBlock(output_active=None)` supported? | **Yes** — `if self.output_active:` already handles `None` → linear pass-through |
| AE task name | `"__reconstruction__"` (hardcoded everywhere) |
| `ae_task_name` parameter on `optimize_latent` | **Remove** — hardcode `"__reconstruction__"` |
| `AutoEncoderTaskConfig` | **Remove from public API**; replace with private `_AEConfig` |
| AE `loss_weight` | Fixed `1.0` |
| `autoencoder_nonnegative=True` activation | `Softplus`; `False` → linear (`output_active=None`) |

---

## Design Summary

Two new parameters on `FlexibleMultiTaskModel`:

```python
enable_autoencoder: bool = False
autoencoder_nonnegative: bool = False
```

When `enable_autoencoder=True`, the model auto-creates an AE head mirroring the encoder dims.
No user-facing config class is exposed. The AE head is registered under `"__reconstruction__"`
in the internal `task_heads` dict so the existing training loop handles its loss automatically.

---

## Dim Derivation — Mirror of Encoder

```
MLPEncoderConfig.hidden_dims = [input_dim, h1, …, latent_dim]
  → AE dims = reversed(hidden_dims)   e.g. [latent_dim, …, h1, input_dim]

TransformerEncoderConfig: has .latent_dim and .input_dim
  → AE dims = [latent_dim, input_dim]   (single linear projection)
```

Both encoder config classes already expose `.input_dim` and `.latent_dim`.

---

## Implementation Steps

### Step 1 — Remove `AutoEncoderTaskConfig`; add private `_AEConfig`

**File**: `src/foundation_model/models/model_config.py`

- Delete `AutoEncoderTaskConfig`.
- Remove it from `TaskConfigType` union.
- Add a private (non-exported) dataclass `_AEConfig` with only what the training loop needs:

```python
@dataclass
class _AEConfig:
    """Internal config for the auto-created reconstruction head. Not part of public API."""
    name: str = "__reconstruction__"
    type: TaskType = TaskType.AUTOENCODER
    dims: List[int] = field(default_factory=list)   # populated by model at init
    nonnegative: bool = False
    norm: bool = True
    residual: bool = False
    loss_weight: float = 1.0
    enabled: bool = True
    data_column: str = "__autoencoder__"            # existing DataModule sentinel
```

`TaskType.AUTOENCODER` enum value is **kept** — DataModule and Dataset still use it to skip
external data loading for AE tasks.

`TaskConfigType` becomes:

```python
TaskConfigType = RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig
```

---

### Step 2 — Update `AutoEncoderHead`

**File**: `src/foundation_model/models/task_head/autoencoder.py`

- Replace `AutoEncoderTaskConfig` import with `_AEConfig`.
- Replace hardcoded `Sigmoid` with:

```python
output_act = torch.nn.Softplus() if config.nonnegative else None
self.net = LinearBlock(
    [d_in] + head_internal_dims[:-1],
    normalization=config.norm,
    residual=config.residual,
    dim_output_layer=head_internal_dims[-1],
    output_active=output_act,
)
```

`output_active=None` is already handled by `LinearBlock` (linear pass-through confirmed).

---

### Step 3 — Update `FlexibleMultiTaskModel`

**File**: `src/foundation_model/models/flexible_multi_task_model.py`

#### 3a — New `__init__` parameters

```python
def __init__(
    self,
    task_configs: Sequence[RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig],
    *,
    encoder_config: ...,
    freeze_shared_encoder: bool = False,
    shared_block_optimizer: OptimizerConfig | None = None,
    enable_learnable_loss_balancer: bool = False,
    allow_all_missing_in_batch: bool = True,
    enable_autoencoder: bool = False,          # NEW
    autoencoder_nonnegative: bool = False,     # NEW
):
```

`AutoEncoderTaskConfig` is removed from the `task_configs` type hint and validation.

#### 3b — Auto-create AE config in `__init__` (after encoder init, before `_init_task_heads`)

```python
self._ae_config: _AEConfig | None = None
if enable_autoencoder:
    dims = self._derive_ae_dims(self.encoder_config)
    self._ae_config = _AEConfig(dims=dims, nonnegative=autoencoder_nonnegative)
    # Append to internal task_configs so training loop and DataModule see it
    self.task_configs.append(self._ae_config)
    self.task_configs_map[self._ae_config.name] = self._ae_config
```

#### 3c — Static helper `_derive_ae_dims`

```python
@staticmethod
def _derive_ae_dims(encoder_config: BaseEncoderConfig) -> list[int]:
    if isinstance(encoder_config, MLPEncoderConfig):
        return list(reversed(encoder_config.hidden_dims))
    # TransformerEncoderConfig
    return [encoder_config.latent_dim, encoder_config.input_dim]
```

#### 3d — Remove `ae_task_name` from `optimize_latent`

In `optimize_latent`, replace every reference to `ae_task_name` with the constant
`"__reconstruction__"`. Update the validation block:

```python
# optimize_space == "latent"
AE_TASK_NAME = "__reconstruction__"
if AE_TASK_NAME not in self.task_heads:
    raise ValueError(
        "optimize_space='latent' requires enable_autoencoder=True on this model."
    )
```

Remove the `ae_task_name` parameter from the method signature and all docstring references.

#### 3e — Remove `AutoEncoderTaskConfig` from all type-hints and assertions

Grep targets in `flexible_multi_task_model.py`:
- `__init__` task_configs type annotation (line 112)
- `add_task` type annotation (line 404)
- `isinstance(..., AutoEncoderTaskConfig)` assertions (lines 424, 458)
- Import line 43

Replace `isinstance(config_item, AutoEncoderTaskConfig)` with
`isinstance(config_item, _AEConfig)` or `config_item.type == TaskType.AUTOENCODER`.

---

### Step 4 — Update DataModule and Dataset

**Files**: `datamodule.py`, `dataset.py`

- Remove `AutoEncoderTaskConfig` import; the `TaskType.AUTOENCODER` check is sufficient.
- `TaskConfig` type alias in `datamodule.py` drops `AutoEncoderTaskConfig`:

```python
TaskConfig = RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig
```

No logic changes — the existing `cfg.type != TaskType.AUTOENCODER` guards already work with
`_AEConfig` because `_AEConfig.type = TaskType.AUTOENCODER`.

---

### Step 5 — Tests

**`flexible_multi_task_model_test.py`** — new group `TestAutoEncoder`:

| Test | Checks |
|------|--------|
| `test_enable_autoencoder_mlp` | AE head created; dims = reversed `hidden_dims`; forward runs; AE loss in training metrics |
| `test_enable_autoencoder_transformer` | dims = `[latent_dim, input_dim]` |
| `test_nonnegative_output` | `autoencoder_nonnegative=True` → all output values ≥ 0 |
| `test_linear_output` | `autoencoder_nonnegative=False` → output can be negative |
| `test_no_autoencoder_default` | `enable_autoencoder=False` (default) → `"__reconstruction__"` not in `task_heads` |
| `test_optimize_latent_requires_ae` | `optimize_space="latent"` without AE → `ValueError` |
| `test_optimize_latent_with_ae` | `optimize_space="latent"` with `enable_autoencoder=True` → runs correctly |

**`task_head/autoencoder_test.py`** (new):

| Test | Checks |
|------|--------|
| `test_softplus_output` | `nonnegative=True` → output ≥ 0 for arbitrary input |
| `test_linear_output` | `nonnegative=False` → output can be negative |

---

## Files Touched

| File | Change |
|------|--------|
| `models/model_config.py` | Remove `AutoEncoderTaskConfig`; add private `_AEConfig`; update `TaskConfigType` |
| `models/task_head/autoencoder.py` | Swap import to `_AEConfig`; replace `Sigmoid` with `nonnegative`-driven activation |
| `models/flexible_multi_task_model.py` | Add `enable_autoencoder` + `autoencoder_nonnegative`; `_derive_ae_dims`; remove `ae_task_name` from `optimize_latent`; clean up `AutoEncoderTaskConfig` references |
| `data/datamodule.py` | Remove `AutoEncoderTaskConfig` import; update `TaskConfig` alias |
| `data/dataset.py` | Remove `AutoEncoderTaskConfig` import if present |
| `models/flexible_multi_task_model_test.py` | Add `TestAutoEncoder` group |
| `models/task_head/autoencoder_test.py` | New test file |
