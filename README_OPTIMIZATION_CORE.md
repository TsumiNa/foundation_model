# optimize_latent usage and core algorithm

> Scope: `FlexibleMultiTaskModel.optimize_latent`. Supports single-task extrema/targets or multi-target regression via `task_targets`. `initial_input` (sample or batch) is required; `initial_input=None` random latent is no longer supported.

## Feature overview

- Explicit start: optimize `initial_input`, with optional `perturbation_std` and `num_restarts`.
- Target matching: `target_value` minimizes MSE to a desired value; otherwise `mode="max"/"min"` searches extremes.
- Multi-target: `task_targets={task: target}` minimizes mean MSE across multiple regression heads.
- Multi-restart: perturb starts, keep best result, and expose `all_restarts`.
- Optional reconstruction: provide AE task name to decode back to descriptor space.

## Quickstart

```python
seed_batch = train_features[:1]  # or random sample matching descriptor dim
result = model.optimize_latent(
    task_name="density",
    initial_input=seed_batch,
    mode="max",              # or "min"
    target_value=None,       # fill with float/tensor to match a value
    steps=200,
    lr=0.05,
    num_restarts=10,
    perturbation_std=0.1,
    ae_task_name="reconstruction",
)
print(result["optimized_target"], result["optimized_input"].shape)
```

### Match a specific value

```python
result = model.optimize_latent(
    task_name="density",
    initial_input=seed_batch,
    target_value=5.0,
    steps=300,
    num_restarts=5,
    perturbation_std=0.05,
)
```

### Multi-target joint matching

```python
result = model.optimize_latent(
    task_name="density",  # kept for backward compatibility
    initial_input=seed_batch,
    task_targets={
        "density": 5.0,
        "thermal_conductivity": 1.2,
    },
    steps=400,
    num_restarts=5,
    perturbation_std=0.1,
)
print(result["optimized_target"])
```

## Parameter reference

| Parameter | Purpose |
| --- | --- |
| `initial_input` | Required starting point (batch allowed); must match encoder input dim |
| `mode` | `"max"/"min"` when neither `target_value` nor `task_targets` is provided |
| `target_value` | Single-task target via MSE; overrides `mode` |
| `task_targets` | Mapping `{task_name: target}` for multi-task joint MSE |
| `num_restarts` | Restart count; returns best and `all_restarts` |
| `perturbation_std` | Gaussian noise per restart on the input |
| `ae_task_name` | Optional AE head name to reconstruct input |
| `steps` / `lr` | Optimization iterations and learning rate |
| `return_details` | If True, also return initial scores and full trajectories |

## Return fields (what they mean)

- Default dict:
  - `optimized_input` `(B, R, D)`: optimized inputs for each restart; `B` batch, `R` restarts, `D` input_dim (decoded if AE provided).
  - `optimized_target` `(B, R, T)`: final per-task outputs for each restart; `T` = number of regression targets (1 for single-task).
- With `return_details=True`:
  - First element (results): `{"optimized_input": (B,R,D), "optimized_score": (B,R,T)}`
    - `optimized_score`: final per-task outputs per restart (same as `optimized_target`).
  - Second element (details): `{"initial_score": (B,T), "trajectory": (B,R,steps,T)}`
    - `initial_score`: per-task outputs at the provided start point (no restarts dimension).
    - `trajectory`: per-task outputs at every step and restart for each batch sample.

## Core algorithm (optimize input directly)

1. Treat `initial_input` as the variable; optionally add noise; set `requires_grad_(True)`.
2. Forward: `input -> encoder -> task_head(s)` to get scores.
3. Loss: use MSE for `target_value`/`task_targets`; otherwise `-score` or `+score` per `mode`.
4. Backprop and Adam update only on the input variable.
5. For restarts, select by best score or lowest MSE.
6. If AE is provided, decode `optimized_latent` to get `optimized_input`.

### Minimal core snippet

```python
optim_input = start_input.detach().clone().requires_grad_(True)
optimizer = torch.optim.Adam([optim_input], lr=lr)
for _ in range(steps):
    optimizer.zero_grad()
    _, h = model.encoder(optim_input)
    pred = model.task_heads[task_name](h)
    loss = (pred - target_tensor).pow(2).mean() if target_tensor is not None else -sign * pred.mean()
    loss.backward()
    optimizer.step()
```

## Notes and limits

- Multi-target supports regression heads via joint MSE; mixed non-regression heads are not covered—customize loss if needed.
- Input feature dim must match encoder input.
- `num_restarts` improves search at linear cost; large `perturbation_std` may wander off-distribution.
