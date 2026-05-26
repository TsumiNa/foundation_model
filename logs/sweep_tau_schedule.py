"""Systematic sweep over topk_tau_start × topk_schedule × K × target-set.

Goal: find a sweet spot for the annealing knobs by comparing achieved targets vs the
unconstrained baseline and vs the "constant τ" controls (which approximate post-hoc
projection at high τ and hard-from-start at low τ).

Output:
  logs/sweep_tau_schedule_results.json  — all per-config results
  stdout: pivot tables for quick reading

We test:
  - **target_set**: '2T' (FE + mag, 3 objectives with QC) vs '1T' (mag only, 2 objectives with QC)
    Rationale: 3 targets may be over-constrained; user asked to retry without FE if needed.
  - **K**: 3 (the user's main case) and 5 (a looser constraint for comparison)
  - **τ_start**: 1.0, 2.0, 5.0, 10.0, 20.0
  - **schedule**: geometric, linear, cosine
  - Controls per (target_set, K):
    * no-constraint baseline
    * constant τ=1.0  (effectively post-hoc projection)
    * constant τ=0.01 (effectively hard-from-start)

For each config we report:
  - achieved target means/stds across the n_starts=8 batch
  - QC probability (P(material_type ∈ QC_CLASSES)) post-optimisation
  - the most common K-element recipe (intersection of top-K per row)

Run from repo root:
  uv run python logs/sweep_tau_schedule.py
"""

from __future__ import annotations

import itertools
import json
import time
import tomllib
from collections import Counter
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
OUT_JSON = REPO / "logs/sweep_tau_schedule_results.json"

# Per-objective targets used in the scenario; 2T uses both, 1T drops FE.
TARGETS_FULL = {"formation_energy": -2.0, "magnetization": 2.0}


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


def _qc_prob_mean(model, x_desc: torch.Tensor) -> float:
    """Mean over batch of P(class ∈ QC_CLASSES)."""
    p = _qc_prob(model, x_desc)  # (B,) per-row QC probability
    return float(p.mean())


def _tau_to_scale(tau: float) -> float:
    """Convert a raw τ to the normalised ``annealing_scale`` (inverse of ``τ = 25**scale``).

    Clamped to [0, 1]: τ values outside [1, 25] saturate at the endpoints.
    """
    import math

    if tau <= 1.0:
        return 0.0
    if tau >= 25.0:
        return 1.0
    return math.log(tau) / math.log(25.0)


def _evaluate(model, kernel, w_seed, *, max_elements, tau_start, tau_end, schedule, targets, seed=0, steps=300):
    """Adapter from the old (τ_start, τ_end, schedule) triple to the new annealing API.

    - schedule="constant": single-segment dict that holds the start scale for the full run.
    - schedule="geometric" with default τ_end=0.01: just set ``annealing_scale`` (default geometric
      from ``25**scale`` down to 0.01).
    - schedule="linear"/"cosine": single-segment dict that interpolates scale_start → 0 (i.e.
      τ → 1.0) with the named func; the default tail from step=1.0 → ... is absent (we cover the
      full optimisation). Note the linear/cosine sweep cannot reach τ=0.01 inside the loop
      without the geometric tail — the final hard projection still gives K-hot output.
    """
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    kwargs = dict(
        task_targets=targets,
        class_targets={"material_type": QC_CLASSES},
        class_target_weight=5.0,
        initial_weights=w_seed,
        seed_blend=0.95,
        steps=steps,
        lr=0.05,
    )
    if max_elements is not None:
        kwargs["max_elements"] = max_elements
        scale_start = _tau_to_scale(tau_start)
        if schedule == "geometric":
            # Default schedule; just supply annealing_scale (covers τ_start → 0.01 geometrically).
            kwargs["annealing_scale"] = scale_start
        elif schedule == "constant":
            kwargs["annealing_scale"] = scale_start
            kwargs["annealing_schedule"] = {
                "step": [1.0],
                "scale": [scale_start],
                "annealing_func": ["constant"],
            }
        elif schedule in ("linear", "cosine"):
            # Single-segment schedule interpolating in scale space from scale_start down to 0
            # (τ from 25**scale_start down to 1.0). The hard projection at the end still
            # cleans up to K-hot regardless of where the in-loop τ settles.
            kwargs["annealing_scale"] = scale_start
            kwargs["annealing_schedule"] = {
                "step": [1.0],
                "scale": [0.0],
                "annealing_func": [schedule],
            }
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
    res = model.optimize_composition(kernel, **kwargs)
    elapsed = time.perf_counter() - t0
    w = res.optimized_weights
    nz = (w > 1e-6).sum(dim=-1).tolist()
    targets_arr = res.optimized_target.cpu().numpy()
    achieved = {
        t: {"mean": float(targets_arr[:, j].mean()), "std": float(targets_arr[:, j].std())}
        for j, t in enumerate(targets.keys())
    }
    # QC probability after decode
    qc_after = _qc_prob_mean(model, res.optimized_descriptor)
    # Per-row top-K recipe → element frequency across batch
    elem_counter: Counter[str] = Counter()
    K_used = max_elements if max_elements is not None else 5
    for b in range(w.shape[0]):
        top_idx = w[b].argsort(descending=True)[:K_used]
        for i in top_idx:
            if float(w[b, i]) > 1e-4:
                elem_counter[DEFAULT_ELEMENTS[int(i)]] += 1
    return {
        "elapsed_s": round(elapsed, 3),
        "nz_per_row": nz,
        "achieved": achieved,
        "qc_after": qc_after,
        "top_elements": elem_counter.most_common(8),
    }


def main() -> None:
    print(f"[loading] {CKPT}")
    runner, model, kernel, device = _build()

    def _qc_fn(x):
        return _qc_prob(model, x)

    seeds = runner._select_seeds(model, device, _qc_fn)[:8]
    print(f"[seeds] {seeds}")
    w_seed = _seed_weights_from_compositions(seeds, n_components=kernel.shape[0])

    target_sets = {
        "2T": TARGETS_FULL,  # FE + mag
        "1T": {"magnetization": 2.0},  # mag only (drop FE)
    }
    Ks = [3, 5]
    tau_starts = [1.0, 2.0, 5.0, 10.0, 20.0]
    schedules = ["geometric", "linear", "cosine"]
    tau_end = 0.01

    results: list[dict] = []
    n_total = 0
    for tset_name in target_sets:
        # Baseline (no constraint)
        n_total += 1
        n_total += len(Ks) * 2  # constant τ=1.0 and τ=0.01 per K
        n_total += len(Ks) * len(tau_starts) * len(schedules)
    print(f"[plan] {n_total} configs total")

    counter = 0
    for tset_name, tgt in target_sets.items():
        # No-constraint baseline
        counter += 1
        print(f"\n[{counter}/{n_total}] {tset_name}  baseline (no max_elements)")
        out = _evaluate(
            model, kernel, w_seed, max_elements=None, tau_start=None, tau_end=None, schedule=None, targets=tgt
        )
        results.append({"target_set": tset_name, "K": None, "tau_start": None, "schedule": "baseline", **out})

        for K in Ks:
            # Controls: constant τ=1.0 (≈post-hoc) and τ=0.01 (hard from start)
            for ctrl_name, ctrl_tau in [("const_t1.0", 1.0), ("const_t0.01", 0.01)]:
                counter += 1
                print(f"[{counter}/{n_total}] {tset_name}  K={K}  {ctrl_name}")
                out = _evaluate(
                    model,
                    kernel,
                    w_seed,
                    max_elements=K,
                    tau_start=ctrl_tau,
                    tau_end=ctrl_tau,
                    schedule="constant",
                    targets=tgt,
                )
                results.append({"target_set": tset_name, "K": K, "tau_start": ctrl_tau, "schedule": ctrl_name, **out})

            # The sweep
            for tau_start, sched in itertools.product(tau_starts, schedules):
                counter += 1
                print(f"[{counter}/{n_total}] {tset_name}  K={K}  τ_start={tau_start}  sched={sched}")
                out = _evaluate(
                    model,
                    kernel,
                    w_seed,
                    max_elements=K,
                    tau_start=tau_start,
                    tau_end=tau_end,
                    schedule=sched,
                    targets=tgt,
                )
                results.append({"target_set": tset_name, "K": K, "tau_start": tau_start, "schedule": sched, **out})

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {OUT_JSON}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Pivot the JSON into a per-target-set, per-K markdown table."""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for tset in ("2T", "1T"):
        subset = [r for r in results if r["target_set"] == tset]
        print(f"\n### Target set: {tset}")
        tgt_keys = list(subset[0]["achieved"].keys()) if subset else []
        header = f"{'config':<35} {'nz_mean':>8} {'QC':>6}  " + "  ".join(f"{t[:10]:>10}" for t in tgt_keys)
        print(header)
        print("-" * len(header))

        # Sort: baseline → controls → sweep (by K, then tau_start, then schedule)
        def _key(r):
            sched_order = {"baseline": 0, "const_t1.0": 1, "const_t0.01": 2, "geometric": 3, "linear": 4, "cosine": 5}
            K = r["K"] if r["K"] is not None else 0
            tau = r["tau_start"] if r["tau_start"] is not None else 0.0
            return (K, tau, sched_order.get(r["schedule"], 99))

        for r in sorted(subset, key=_key):
            tag = "baseline" if r["schedule"] == "baseline" else f"K{r['K']} {r['schedule']:<10} τ0={r['tau_start']:<4}"
            nz_mean = sum(r["nz_per_row"]) / len(r["nz_per_row"])
            row = f"{tag:<35} {nz_mean:>8.2f} {r['qc_after']:>6.3f}  "
            row += "  ".join(f"{r['achieved'][t]['mean']:>+5.2f}±{r['achieved'][t]['std']:.2f}" for t in tgt_keys)
            print(row)


if __name__ == "__main__":
    main()
