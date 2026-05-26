"""Comprehensive evaluation of A (max_elements) + B (fixed_amounts) + C (min_nonzero_weight).

Exercises each feature in isolation and combined, on the inverse-design fine-tuned model.
Verifies that:
  1. Each constraint enforces its contract (≤ K non-zero; fixed values held; floor respected).
  2. Combinations compose cleanly (the simplex invariant is preserved everywhere).
  3. The annealing_scale knob still works on top of the full feature stack.

Reports per-config: achieved targets (FE, Mag), QC, non-zero count, row-0 recipe.
"""
from __future__ import annotations

import time
import tomllib
from pathlib import Path

import torch
from lightning import seed_everything

from foundation_model.scripts.continual_rehearsal_demo import (
    QC_CLASSES,
    ContinualRehearsalConfig,
    ContinualRehearsalRunner,
)
from foundation_model.scripts.eval_inverse_methods import _qc_prob, _seed_weights_from_compositions
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "samples/continual_rehearsal_demo_config_inverse_baseline.toml"
CKPT = REPO / "artifacts/inverse_design_run/finetune/final_model.pt"
SCENARIO = "scenario1_fe_down_magnetic_up"


def _build():
    raw = tomllib.loads(CFG_PATH.read_text(encoding="utf-8"))
    scenarios = raw.pop("inverse_scenarios", [])
    sc = next(s for s in scenarios if s["name"] == SCENARIO)
    raw["inverse_reg_tasks"] = sc["reg_tasks"]
    raw["inverse_reg_targets"] = sc["reg_targets"]
    config = ContinualRehearsalConfig(**raw)
    seed_everything(config.random_seed, workers=True)
    runner = ContinualRehearsalRunner(config)
    model = runner._build_full_model()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    kernel = runner._kmd.kernel_torch(device=device, dtype=dtype)
    return runner, model, kernel, device


def _summarise(label: str, res, *, expected=None, floor=None):
    """Print one config's results + verify the user-facing contract."""
    w = res.optimized_weights
    B = w.shape[0]
    nz_mask = w > 1e-6
    nz_per_row = nz_mask.sum(dim=-1).tolist()
    row_sums = w.sum(dim=-1)
    targets = res.optimized_target.cpu().numpy()
    print(f"\n  [{label}]")
    print(f"    elapsed: {res.optimized_weights.shape[0]} rows; nz per row: {nz_per_row}")
    print(f"    row sums (should be 1.0): min={row_sums.min():.5f}, max={row_sums.max():.5f}")
    print(f"    FE  = {targets[:, 0].mean():+.3f} ± {targets[:, 0].std():.3f} (target -2.0)")
    print(f"    Mag = {targets[:, 1].mean():+.3f} ± {targets[:, 1].std():.3f} (target +2.0)")
    # Row 0 recipe.
    w0 = w[0].cpu().numpy()
    top = sorted(((float(w0[i]), DEFAULT_ELEMENTS[i]) for i in range(len(w0)) if float(w0[i]) > 1e-4), reverse=True)
    print(f"    row 0 recipe: " + ", ".join(f"{s}={v:.3f}" for v, s in top))
    # Contract checks.
    if expected is not None:
        for sym, want in expected.items():
            idx = DEFAULT_ELEMENTS.index(sym)
            got = w[:, idx].tolist()
            ok = all(abs(g - want) < 1e-4 for g in got)
            mark = "✓" if ok else "✗"
            print(f"    [{mark}] {sym} pinned at {want}: got {[round(g, 4) for g in got]}")
    if floor is not None:
        # Every non-zero unlocked element should be ≥ floor.
        violated = (w > 1e-6) & (w < floor - 1e-5)
        n_violations = int(violated.sum().item())
        smallest_nz = w[w > 1e-6].min().item() if (w > 1e-6).any() else 0.0
        mark = "✓" if n_violations == 0 else f"✗ ({n_violations} positions)"
        print(f"    [{mark}] every non-zero ≥ {floor}: smallest non-zero = {smallest_nz:.4f}")


def main() -> None:
    print(f"[loading] {CKPT}")
    runner, model, kernel, device = _build()
    def _qc_fn(x): return _qc_prob(model, x)
    seeds = runner._select_seeds(model, device, _qc_fn)[:8]
    w_seed = _seed_weights_from_compositions(seeds, n_components=kernel.shape[0])
    common = dict(
        task_targets={"formation_energy": -2.0, "magnetization": 2.0},
        class_targets={"material_type": QC_CLASSES},
        class_target_weight=5.0,
        initial_weights=w_seed,
        seed_blend=0.95,
        steps=300,
        lr=0.05,
    )

    print("\n=== individual features ===")
    torch.manual_seed(0)
    _summarise("baseline (no constraint)", model.optimize_composition(kernel, **common))

    torch.manual_seed(0)
    _summarise("A only — max_elements=3", model.optimize_composition(kernel, max_elements=3, **common))

    torch.manual_seed(0)
    _summarise(
        "B only — fixed Au=0.65, Ga=0.20",
        model.optimize_composition(kernel, fixed_amounts={"Au": 0.65, "Ga": 0.20}, **common),
        expected={"Au": 0.65, "Ga": 0.20},
    )

    torch.manual_seed(0)
    _summarise(
        "C only — floor=0.10",
        model.optimize_composition(kernel, min_nonzero_weight=0.10, **common),
        floor=0.10,
    )

    print("\n=== pairwise combinations ===")
    torch.manual_seed(0)
    _summarise(
        "A + B — K=4, fixed Au=0.65 Ga=0.20",
        model.optimize_composition(
            kernel, max_elements=4, fixed_amounts={"Au": 0.65, "Ga": 0.20}, **common
        ),
        expected={"Au": 0.65, "Ga": 0.20},
    )

    torch.manual_seed(0)
    _summarise(
        "A + C — K=5, floor=0.10",
        model.optimize_composition(kernel, max_elements=5, min_nonzero_weight=0.10, **common),
        floor=0.10,
    )

    torch.manual_seed(0)
    _summarise(
        "B + C — fixed Au=0.30 Ga=0.20, floor=0.10",
        model.optimize_composition(
            kernel, fixed_amounts={"Au": 0.30, "Ga": 0.20}, min_nonzero_weight=0.10, **common
        ),
        expected={"Au": 0.30, "Ga": 0.20},
        floor=0.10,
    )

    print("\n=== full stack A+B+C ===")
    torch.manual_seed(0)
    _summarise(
        "A + B + C — K=4, fixed Au=0.30 Ga=0.20, floor=0.10  (default annealing)",
        model.optimize_composition(
            kernel, max_elements=4,
            fixed_amounts={"Au": 0.30, "Ga": 0.20},
            min_nonzero_weight=0.10,
            **common,
        ),
        expected={"Au": 0.30, "Ga": 0.20},
        floor=0.10,
    )

    torch.manual_seed(0)
    _summarise(
        "A + B + C — same + annealing_scale=0.8 (more exploration)",
        model.optimize_composition(
            kernel, max_elements=4,
            fixed_amounts={"Au": 0.30, "Ga": 0.20},
            min_nonzero_weight=0.10,
            annealing_scale=0.8,
            **common,
        ),
        expected={"Au": 0.30, "Ga": 0.20},
        floor=0.10,
    )

    torch.manual_seed(0)
    _summarise(
        "A + B + C — same + advanced schedule (warm-up then linear)",
        model.optimize_composition(
            kernel, max_elements=4,
            fixed_amounts={"Au": 0.30, "Ga": 0.20},
            min_nonzero_weight=0.10,
            annealing_scale=0.5,
            annealing_schedule={
                "step": [0.2, 0.7, 1.0],
                "tau": [0.9, 0.5, 0.0],
                "annealing_func": ["constant", "linear", "linear"],
            },
            **common,
        ),
        expected={"Au": 0.30, "Ga": 0.20},
        floor=0.10,
    )


if __name__ == "__main__":
    main()
