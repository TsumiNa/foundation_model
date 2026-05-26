"""Real-model smoke test for max_elements.

Loads the inverse-design fine-tuned model (the same checkpoint paper_inverse_3scenarios uses),
constructs the right KMD kernel via ContinualRehearsalRunner, and compares:

1. Baseline (no max_elements) — existing behaviour.
2. max_elements=3 / 5 with geometric annealing — should give exactly K-element recipes.
3. max_elements=3 with constant-τ — control showing annealing matters.

Reports: non-zero counts, achieved regression targets, QC probability, recipe row 0.
"""

from __future__ import annotations

import tomllib
import time
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
    """Mirror the loading dance from paper_inverse_comparison.run()."""
    raw = tomllib.loads(CFG_PATH.read_text(encoding="utf-8"))
    # Inject scenario 1's overrides (formation_energy down, magnetization up).
    scenarios = raw.pop("inverse_scenarios", [])
    sc = next(s for s in scenarios if s["name"] == SCENARIO)
    raw["inverse_reg_tasks"] = sc["reg_tasks"]
    raw["inverse_reg_targets"] = sc["reg_targets"]
    # Drop fields ContinualRehearsalConfig doesn't accept (inverse_seed_explicit_append OK).
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
    return runner, model, kernel, config, device


def _report(label, res, B):
    w = res.optimized_weights
    nz = (w > 1e-4).sum(dim=-1)
    targets = res.optimized_target.cpu().numpy()
    print(f"  [{label}]")
    print(f"    non-zero(>1e-4): {nz.tolist()}  mean={nz.float().mean():.2f}")
    print(f"    formation_energy = {targets[:, 0].mean():+.3f} ± {targets[:, 0].std():.3f} (target -2.0)")
    print(f"    magnetization    = {targets[:, 1].mean():+.3f} ± {targets[:, 1].std():.3f} (target +2.0)")
    # Row 0 recipe
    w0 = w[0].detach().cpu()
    top = sorted(((float(w0[i]), DEFAULT_ELEMENTS[i]) for i in range(len(w0)) if float(w0[i]) > 1e-4), reverse=True)
    print(f"    row 0 recipe ({len(top)} elements): " + ", ".join(f"{s}={v:.3f}" for v, s in top))


def main() -> None:
    print(f"[loading] checkpoint={CKPT}")
    runner, model, kernel, config, device = _build()
    print(f"[loaded] kernel shape={kernel.shape}, encoder input_dim={getattr(model.encoder, 'input_dim', None)}")

    # Seed compositions (paper script's top-QC selection)
    def _qc_fn(x):
        return _qc_prob(model, x)

    seeds = runner._select_seeds(model, device, _qc_fn)[:8]  # only 8 for speed
    print(f"[seeds] {seeds}")
    n = kernel.shape[0]
    w_seed = _seed_weights_from_compositions(seeds, n_components=n)

    targets = {"formation_energy": -2.0, "magnetization": 2.0}
    common = dict(
        task_targets=targets,
        class_targets={"material_type": QC_CLASSES},
        class_target_weight=5.0,
        initial_weights=w_seed,
        seed_blend=0.95,
        steps=300,
        lr=0.05,
    )

    print("\n[run 1] baseline (no max_elements)")
    torch.manual_seed(0)
    t0 = time.perf_counter()
    res0 = model.optimize_composition(kernel, **common)
    el0 = time.perf_counter() - t0
    print(f"  elapsed {el0:.2f}s")
    _report("baseline", res0, len(seeds))

    for K in (3, 5):
        torch.manual_seed(0)
        print(f"\n[run] max_elements={K} (default annealing_scale=0.5)")
        t0 = time.perf_counter()
        res = model.optimize_composition(
            kernel,
            max_elements=K,
            record_weights_trajectory=True,
            **common,
        )
        el = time.perf_counter() - t0
        print(f"  elapsed {el:.2f}s  (overhead vs baseline: {el - el0:+.2f}s)")
        _report(f"K={K}", res, len(seeds))
        # annealing visualisation
        traj = res.weights_trajectory
        nz_t = (traj > 1e-3).sum(dim=-1).float().mean(dim=-1)
        chk = [0, 30, 100, 200, 290, 299]
        print("    annealing nz over trajectory: " + ", ".join(f"step{s}={nz_t[s]:.1f}" for s in chk if s < len(nz_t)))

    torch.manual_seed(0)
    print("\n[run] max_elements=3, annealing_scale=0.0 (no exploration, τ_start=1)")
    res_c = model.optimize_composition(kernel, max_elements=3, annealing_scale=0.0, **common)
    _report("K=3 scale=0.0", res_c, len(seeds))

    torch.manual_seed(0)
    print("\n[run] max_elements=3, annealing_scale=1.0 (max exploration, τ_start=25)")
    res_h = model.optimize_composition(kernel, max_elements=3, annealing_scale=1.0, **common)
    _report("K=3 scale=1.0", res_h, len(seeds))

    torch.manual_seed(0)
    print("\n[run] max_elements=3, advanced dict (linear warm-up, then geometric tail)")
    res_d = model.optimize_composition(
        kernel,
        max_elements=3,
        annealing_scale=0.5,
        annealing_schedule={
            "step": [0.3, 0.6],
            "scale": [0.9, 0.5],
            "annealing_func": ["linear", "linear"],
        },
        **common,
    )
    _report("K=3 dict (warm-up→linear→tail)", res_d, len(seeds))


if __name__ == "__main__":
    main()
