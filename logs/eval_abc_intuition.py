"""Behavioral sweep + intuition check for max_elements / fixed_amounts / min_nonzero_weight.

For each feature alone (1-D sweep) and a few combined cases (pairwise + full stack), this
script:

1. Runs the optimisation on the inverse-design fine-tuned model.
2. Records achieved targets (FE, Mag, klat), QC probability, non-zero count, sample recipe.
3. Verifies the user-facing contract (e.g., ``∀ row: ``Au == pinned``; ``nz ≤ K``;
   ``every non-zero ≥ floor``).
4. Compares the observed trend against a pre-stated intuition and flags mismatches.

Output:
  * ``logs/eval_abc_intuition.json`` — every config's numbers.
  * ``logs/eval_abc_intuition.png`` — 1-D sweeps for A / B / C + combinations.
  * stdout — markdown summary with PASS/FAIL on each intuition.

Two scenarios are exercised: scenario1 (FE↓, Mag↑) and scenario3 (FE↓, klat↑). The 2-target
scenario1 lets us isolate the magnetization channel; scenario3 swaps in klat as a more
"reachable" objective.
"""
from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path
from typing import Any

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
OUT_JSON = REPO / "logs/eval_abc_intuition.json"


def _build(scenario_name: str):
    raw = tomllib.loads(CFG_PATH.read_text(encoding="utf-8"))
    scenarios = raw.pop("inverse_scenarios", [])
    sc = next(s for s in scenarios if s["name"] == scenario_name)
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
    def _qc_fn(x): return _qc_prob(model, x)
    seeds = runner._select_seeds(model, device, _qc_fn)[:8]
    w_seed = _seed_weights_from_compositions(seeds, n_components=kernel.shape[0])
    return model, kernel, w_seed, dict(sc), seeds


def _run(model, kernel, w_seed, reg_targets, **kwargs) -> dict:
    common = dict(
        task_targets=reg_targets,
        class_targets={"material_type": QC_CLASSES},
        class_target_weight=5.0,
        initial_weights=w_seed,
        seed_blend=0.95,
        steps=300,
        lr=0.05,
    )
    common.update(kwargs)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    res = model.optimize_composition(kernel, **common)
    elapsed = time.perf_counter() - t0
    w = res.optimized_weights
    targets = res.optimized_target.cpu().numpy()
    achieved = {t: {"mean": float(targets[:, j].mean()), "std": float(targets[:, j].std())}
                for j, t in enumerate(reg_targets)}
    qc = float(_qc_prob(model, res.optimized_descriptor).mean())
    nz = (w > 1e-6).sum(dim=-1)
    # Row-0 top 5 elements for inspection.
    top = sorted(
        ((float(w[0, i]), DEFAULT_ELEMENTS[i]) for i in range(kernel.shape[0]) if float(w[0, i]) > 1e-4),
        reverse=True,
    )[:5]
    return {
        "elapsed_s": round(elapsed, 3),
        "achieved": achieved,
        "qc": qc,
        "nz_mean": float(nz.float().mean()),
        "nz_min": int(nz.min()),
        "nz_max": int(nz.max()),
        "smallest_nonzero": float(w[w > 1e-6].min()) if (w > 1e-6).any() else 0.0,
        "row0_recipe": [(s, round(v, 4)) for v, s in top],
        # For contract checks:
        "_au_col": [float(w[b, DEFAULT_ELEMENTS.index("Au")]) for b in range(w.shape[0])],
        "_ga_col": [float(w[b, DEFAULT_ELEMENTS.index("Ga")]) for b in range(w.shape[0])],
    }


def main() -> None:
    print("=" * 80)
    print("Behavioral evaluation: A (max_elements) + B (fixed_amounts) + C (min_nonzero_weight)")
    print("=" * 80)

    all_results: dict[str, list[dict]] = {}

    for scenario_name in ["scenario1_fe_down_magnetic_up", "scenario3_fe_down_klat_up"]:
        print(f"\n\n### {scenario_name}")
        model, kernel, w_seed, sc, _ = _build(scenario_name)
        reg_targets = dict(zip(sc["reg_tasks"], sc["reg_targets"]))
        bucket: list[dict] = []

        # === A sweep ===
        print("\n[A] max_elements sweep")
        for K in (None, 2, 3, 4, 5, 6, 8, 10):
            out = _run(model, kernel, w_seed, reg_targets, **({} if K is None else {"max_elements": K}))
            out["experiment"] = "A"
            out["K"] = K
            out["floor"] = 0.0
            out["n_fixed"] = 0
            bucket.append(out)

        # === B sweep (Ga fixed at 0.20, vary Au) ===
        print("[B] fixed_amounts sweep (Ga=0.20, Au varies)")
        for au in (0.0, 0.30, 0.45, 0.65, 0.75):
            if au == 0.0:
                fa = {"Ga": 0.20}      # Au omitted → only Ga pinned
            else:
                fa = {"Au": au, "Ga": 0.20}
            out = _run(model, kernel, w_seed, reg_targets, fixed_amounts=fa)
            out["experiment"] = "B"
            out["K"] = None
            out["au_fixed"] = au
            out["n_fixed"] = len(fa)
            bucket.append(out)

        # === C sweep ===
        print("[C] min_nonzero_weight sweep")
        for floor in (0.0, 0.05, 0.10, 0.15, 0.20):
            kw = {} if floor == 0.0 else {"min_nonzero_weight": floor}
            out = _run(model, kernel, w_seed, reg_targets, **kw)
            out["experiment"] = "C"
            out["K"] = None
            out["floor"] = floor
            out["n_fixed"] = 0
            bucket.append(out)

        # === A+B (K varies, Au+Ga fixed) ===
        print("[A+B] K varies, fixed Au=0.65, Ga=0.20")
        for K in (3, 4, 5, 6, 8):
            out = _run(model, kernel, w_seed, reg_targets,
                       max_elements=K, fixed_amounts={"Au": 0.65, "Ga": 0.20})
            out["experiment"] = "A+B"
            out["K"] = K
            out["n_fixed"] = 2
            bucket.append(out)

        # === A+C (K=5, floor varies) ===
        print("[A+C] K=5, floor varies")
        for floor in (0.0, 0.05, 0.10, 0.15, 0.20):
            kw = {"max_elements": 5}
            if floor > 0:
                kw["min_nonzero_weight"] = floor
            out = _run(model, kernel, w_seed, reg_targets, **kw)
            out["experiment"] = "A+C"
            out["K"] = 5
            out["floor"] = floor
            bucket.append(out)

        # === B+C (fix Au=0.30 Ga=0.20, floor varies) ===
        print("[B+C] fixed Au=0.30, Ga=0.20, floor varies")
        for floor in (0.0, 0.05, 0.10, 0.15):
            kw = {"fixed_amounts": {"Au": 0.30, "Ga": 0.20}}
            if floor > 0:
                kw["min_nonzero_weight"] = floor
            out = _run(model, kernel, w_seed, reg_targets, **kw)
            out["experiment"] = "B+C"
            out["K"] = None
            out["floor"] = floor
            bucket.append(out)

        # === A+B+C full stack at 3 annealing settings ===
        print("[A+B+C] K=4, Au=0.30 Ga=0.20, floor=0.10, annealing varies")
        for scale_label, scale in [("scale=0.3", 0.3), ("scale=0.5 default", 0.5), ("scale=0.8", 0.8)]:
            out = _run(model, kernel, w_seed, reg_targets,
                       max_elements=4,
                       fixed_amounts={"Au": 0.30, "Ga": 0.20},
                       min_nonzero_weight=0.10,
                       annealing_scale=scale)
            out["experiment"] = "A+B+C"
            out["annealing_scale_label"] = scale_label
            out["K"] = 4
            out["n_fixed"] = 2
            out["floor"] = 0.10
            bucket.append(out)

        all_results[scenario_name] = bucket

    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"\n[saved] {OUT_JSON}")

    # === Intuition checks ===
    _print_intuition_checks(all_results)


def _print_intuition_checks(all_results: dict[str, list[dict]]) -> None:
    """Each intuition: a one-line description + observed values + PASS/FAIL.

    A failure here doesn't mean the implementation is broken — it means the model's loss
    landscape doesn't satisfy the assumed monotone relationship. Useful regardless.
    """
    print("\n" + "=" * 100)
    print("INTUITION CHECKS")
    print("=" * 100)

    for scenario_name, bucket in all_results.items():
        print(f"\n## {scenario_name}")
        sc_short = "FE/Mag" if "magnetic" in scenario_name else "FE/klat"
        reg_keys = list(bucket[0]["achieved"].keys())

        def filter_exp(name: str) -> list[dict]:
            return [r for r in bucket if r["experiment"] == name]

        # Check 1: A — nz exactly == K (or ≤ K).
        a_rows = filter_exp("A")
        constrained = [r for r in a_rows if r["K"] is not None]
        nz_eq_K = all(r["nz_max"] <= r["K"] for r in constrained)
        print(f"\n  [A] nz_max ≤ K for every K-constrained config: "
              f"{'PASS' if nz_eq_K else 'FAIL'}")
        for r in a_rows:
            tag = f"K={r['K']}" if r["K"] is not None else "baseline"
            t_str = "  ".join(f"{k}={r['achieved'][k]['mean']:+.2f}" for k in reg_keys)
            print(f"      {tag:<10}  nz∈[{r['nz_min']}, {r['nz_max']}]  QC={r['qc']:.2f}  {t_str}")

        # Check 2: A — primary target (first reg) improves as K grows.
        fe_seq = [r["achieved"][reg_keys[0]]["mean"] for r in a_rows if r["K"] is not None]
        Ks = [r["K"] for r in a_rows if r["K"] is not None]
        # Trend: lower-is-better for FE (target is -2.0). Compare smallest K to largest K.
        fe_improves_with_K = fe_seq[0] >= fe_seq[-1]
        print(f"  [A] {reg_keys[0]} decreases (improves toward target) as K grows (K={Ks[0]}→{Ks[-1]}): "
              f"{fe_seq[0]:+.2f} → {fe_seq[-1]:+.2f}  "
              f"{'PASS' if fe_improves_with_K else 'FAIL'}")

        # Check 3: B — Au and Ga pinned exactly across the batch.
        b_rows = filter_exp("B")
        all_pinned_ok = True
        for r in b_rows:
            if r["au_fixed"] > 0:
                if not all(abs(v - r["au_fixed"]) < 1e-4 for v in r["_au_col"]):
                    all_pinned_ok = False
            if not all(abs(v - 0.20) < 1e-4 for v in r["_ga_col"]):
                all_pinned_ok = False
        print(f"\n  [B] fixed Au/Ga held exactly across all batch rows: "
              f"{'PASS' if all_pinned_ok else 'FAIL'}")
        for r in b_rows:
            au_lbl = f"Au={r['au_fixed']:.2f}" if r["au_fixed"] > 0 else "Au not fixed"
            t_str = "  ".join(f"{k}={r['achieved'][k]['mean']:+.2f}" for k in reg_keys)
            print(f"      {au_lbl:<14}  nz∈[{r['nz_min']}, {r['nz_max']}]  QC={r['qc']:.2f}  {t_str}")

        # Check 4: B — as Au grows from 0.30 to 0.80, less free mass → primary target worsens
        b_pinned_rows = [r for r in b_rows if r["au_fixed"] > 0]
        fe_seq_b = [r["achieved"][reg_keys[0]]["mean"] for r in b_pinned_rows]
        fe_worsens_with_au = fe_seq_b[0] <= fe_seq_b[-1]
        print(f"  [B] {reg_keys[0]} worsens as Au pin grows (Au=0.30→0.75): "
              f"{fe_seq_b[0]:+.2f} → {fe_seq_b[-1]:+.2f}  "
              f"{'PASS' if fe_worsens_with_au else 'FAIL'}")

        # Check 5: C — every non-zero ≥ floor; smallest_nonzero ≥ floor (within tol).
        c_rows = filter_exp("C")
        floor_held = all(r["smallest_nonzero"] >= r["floor"] - 1e-5 or r["floor"] == 0 for r in c_rows)
        print(f"\n  [C] smallest non-zero ≥ floor for every floored config: "
              f"{'PASS' if floor_held else 'FAIL'}")
        for r in c_rows:
            t_str = "  ".join(f"{k}={r['achieved'][k]['mean']:+.2f}" for k in reg_keys)
            print(f"      floor={r['floor']:.2f}  nz∈[{r['nz_min']}, {r['nz_max']}]  "
                  f"min_nz={r['smallest_nonzero']:.3f}  QC={r['qc']:.2f}  {t_str}")

        # Check 6: C — nz decreases monotonically as floor grows.
        nz_seq = [r["nz_mean"] for r in c_rows]
        nz_monotone = all(nz_seq[i] >= nz_seq[i+1] for i in range(len(nz_seq) - 1))
        print(f"  [C] nz_mean decreases monotonically with floor: "
              f"{[round(n, 1) for n in nz_seq]}  "
              f"{'PASS' if nz_monotone else 'FAIL'}")

        # Check 7: A+B — nz exactly = K, Au+Ga held.
        ab_rows = filter_exp("A+B")
        ab_nz_ok = all(r["nz_max"] <= r["K"] for r in ab_rows)
        ab_pin_ok = all(
            all(abs(v - 0.65) < 1e-4 for v in r["_au_col"])
            and all(abs(v - 0.20) < 1e-4 for v in r["_ga_col"])
            for r in ab_rows
        )
        print(f"\n  [A+B] nz ≤ K AND Au/Ga held exactly: "
              f"{'PASS' if (ab_nz_ok and ab_pin_ok) else 'FAIL'}")
        for r in ab_rows:
            t_str = "  ".join(f"{k}={r['achieved'][k]['mean']:+.2f}" for k in reg_keys)
            print(f"      K={r['K']:<2}  nz∈[{r['nz_min']}, {r['nz_max']}]  QC={r['qc']:.2f}  {t_str}")

        # Check 8: A+B — K=3 (only 1 free slot) is worse than K=8 (6 free slots) on primary target.
        fe_K3 = next(r["achieved"][reg_keys[0]]["mean"] for r in ab_rows if r["K"] == 3)
        fe_K8 = next(r["achieved"][reg_keys[0]]["mean"] for r in ab_rows if r["K"] == 8)
        print(f"  [A+B] {reg_keys[0]} at K=3 ≥ K=8 (less freedom is worse): "
              f"{fe_K3:+.2f} ≥ {fe_K8:+.2f}  "
              f"{'PASS' if fe_K3 >= fe_K8 else 'FAIL'}")

        # Check 9: A+C — nz_mean ≤ K=5, decreases as floor grows.
        ac_rows = filter_exp("A+C")
        ac_nz_le_K = all(r["nz_max"] <= 5 for r in ac_rows)
        ac_nz_drop = all(ac_rows[i]["nz_mean"] >= ac_rows[i+1]["nz_mean"] for i in range(len(ac_rows) - 1))
        print(f"\n  [A+C] nz ≤ K=5 AND nz_mean decreases with floor: "
              f"{'PASS' if (ac_nz_le_K and ac_nz_drop) else 'FAIL'}")
        for r in ac_rows:
            t_str = "  ".join(f"{k}={r['achieved'][k]['mean']:+.2f}" for k in reg_keys)
            print(f"      K=5, floor={r['floor']:.2f}  nz∈[{r['nz_min']}, {r['nz_max']}]  "
                  f"min_nz={r['smallest_nonzero']:.3f}  QC={r['qc']:.2f}  {t_str}")

        # Check 10: B+C — Au=0.30, Ga=0.20 held; floor respected.
        bc_rows = filter_exp("B+C")
        bc_pin_ok = all(
            all(abs(v - 0.30) < 1e-4 for v in r["_au_col"])
            and all(abs(v - 0.20) < 1e-4 for v in r["_ga_col"])
            for r in bc_rows
        )
        bc_floor_ok = all(r["smallest_nonzero"] >= r["floor"] - 1e-5 or r["floor"] == 0 for r in bc_rows)
        print(f"\n  [B+C] fixed values held AND floor respected: "
              f"{'PASS' if (bc_pin_ok and bc_floor_ok) else 'FAIL'}")
        for r in bc_rows:
            t_str = "  ".join(f"{k}={r['achieved'][k]['mean']:+.2f}" for k in reg_keys)
            print(f"      fix Au=.30 Ga=.20, floor={r['floor']:.2f}  "
                  f"nz∈[{r['nz_min']}, {r['nz_max']}]  min_nz={r['smallest_nonzero']:.3f}  "
                  f"QC={r['qc']:.2f}  {t_str}")

        # Check 11: A+B+C — all three contracts hold simultaneously.
        abc_rows = filter_exp("A+B+C")
        all_contracts_ok = all(
            r["nz_max"] <= r["K"]
            and all(abs(v - 0.30) < 1e-4 for v in r["_au_col"])
            and all(abs(v - 0.20) < 1e-4 for v in r["_ga_col"])
            and r["smallest_nonzero"] >= r["floor"] - 1e-5
            for r in abc_rows
        )
        print(f"\n  [A+B+C] all three contracts hold simultaneously: "
              f"{'PASS' if all_contracts_ok else 'FAIL'}")
        for r in abc_rows:
            t_str = "  ".join(f"{k}={r['achieved'][k]['mean']:+.2f}" for k in reg_keys)
            print(f"      {r['annealing_scale_label']:<20}  nz∈[{r['nz_min']}, {r['nz_max']}]  "
                  f"min_nz={r['smallest_nonzero']:.3f}  QC={r['qc']:.2f}  {t_str}")


if __name__ == "__main__":
    main()
