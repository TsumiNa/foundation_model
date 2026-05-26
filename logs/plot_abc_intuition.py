"""Multi-panel sweep plot from eval_abc_intuition.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
JSON_PATH = REPO / "logs/eval_abc_intuition.json"
OUT_PNG = REPO / "logs/eval_abc_intuition.png"

SCENARIOS = ("scenario1_fe_down_magnetic_up", "scenario3_fe_down_klat_up")
PRETTY = {
    "scenario1_fe_down_magnetic_up": "Scenario 1 — FE↓, Mag↑",
    "scenario3_fe_down_klat_up": "Scenario 3 — FE↓, klat↑",
}


def main() -> None:
    data = json.loads(JSON_PATH.read_text())

    # Two rows (scenarios) × five columns (one per experiment: A, B, C, A+B / A+C / B+C / A+B+C)
    # We'll do: row=scenario, columns = A | B | C | A+C | B+C
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), squeeze=False)

    for r, scen in enumerate(SCENARIOS):
        bucket = data[scen]
        reg_keys = list(bucket[0]["achieved"].keys())
        primary = reg_keys[0]   # "formation_energy"
        secondary = reg_keys[1]  # mag or klat

        # --- Column 0: A sweep (K vs targets, QC, nz) ---
        a_rows = [r for r in bucket if r["experiment"] == "A" and r["K"] is not None]
        Ks = [r["K"] for r in a_rows]
        ax = axes[r, 0]
        ax.plot(Ks, [r["achieved"][primary]["mean"] for r in a_rows], "o-", label=primary, color="#2563EB")
        ax.plot(Ks, [r["achieved"][secondary]["mean"] for r in a_rows], "s-", label=secondary, color="#E0762A")
        ax.plot(Ks, [r["qc"] for r in a_rows], "^-", label="QC", color="#55A868")
        # baseline reference lines
        base = next(r for r in bucket if r["experiment"] == "A" and r["K"] is None)
        for v, c, ls in [(base["achieved"][primary]["mean"], "#2563EB", "--"),
                          (base["achieved"][secondary]["mean"], "#E0762A", "--"),
                          (base["qc"], "#55A868", "--")]:
            ax.axhline(v, color=c, linestyle=ls, alpha=0.4, linewidth=0.8)
        ax.set_xlabel("max_elements (K)")
        ax.set_title(f"{PRETTY[scen]}\nA — vary K")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # --- Column 1: B sweep (Au pin vs targets, QC) ---
        b_rows = [r for r in bucket if r["experiment"] == "B"]
        aus = [r["au_fixed"] for r in b_rows]
        ax = axes[r, 1]
        ax.plot(aus, [r["achieved"][primary]["mean"] for r in b_rows], "o-", label=primary, color="#2563EB")
        ax.plot(aus, [r["achieved"][secondary]["mean"] for r in b_rows], "s-", label=secondary, color="#E0762A")
        ax.plot(aus, [r["qc"] for r in b_rows], "^-", label="QC", color="#55A868")
        ax.set_xlabel("fixed Au amount (Ga=0.20)")
        ax.set_title("B — vary Au pin")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # --- Column 2: C sweep (floor vs targets, nz_mean) ---
        c_rows = [r for r in bucket if r["experiment"] == "C"]
        floors = [r["floor"] for r in c_rows]
        ax = axes[r, 2]
        ax.plot(floors, [r["achieved"][primary]["mean"] for r in c_rows], "o-", label=primary, color="#2563EB")
        ax.plot(floors, [r["achieved"][secondary]["mean"] for r in c_rows], "s-", label=secondary, color="#E0762A")
        ax.plot(floors, [r["qc"] for r in c_rows], "^-", label="QC", color="#55A868")
        ax_nz = ax.twinx()
        ax_nz.plot(floors, [r["nz_mean"] for r in c_rows], "d:", label="nz_mean", color="#888")
        ax_nz.set_ylabel("mean nz", color="#888")
        ax_nz.set_yscale("symlog")
        ax.set_xlabel("min_nonzero_weight (floor)")
        ax.set_title("C — vary floor")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # --- Column 3: A+C (K=5, floor sweep) ---
        ac_rows = [r for r in bucket if r["experiment"] == "A+C"]
        floors = [r["floor"] for r in ac_rows]
        ax = axes[r, 3]
        ax.plot(floors, [r["achieved"][primary]["mean"] for r in ac_rows], "o-", label=primary, color="#2563EB")
        ax.plot(floors, [r["achieved"][secondary]["mean"] for r in ac_rows], "s-", label=secondary, color="#E0762A")
        ax.plot(floors, [r["qc"] for r in ac_rows], "^-", label="QC", color="#55A868")
        ax_nz = ax.twinx()
        ax_nz.plot(floors, [r["nz_mean"] for r in ac_rows], "d:", color="#888")
        ax_nz.set_ylabel("mean nz", color="#888")
        ax.set_xlabel("floor (K=5 fixed)")
        ax.set_title("A+C — K=5, vary floor")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # --- Column 4: B+C (fix Au=0.30 Ga=0.20, floor sweep) ---
        bc_rows = [r for r in bucket if r["experiment"] == "B+C"]
        floors = [r["floor"] for r in bc_rows]
        ax = axes[r, 4]
        ax.plot(floors, [r["achieved"][primary]["mean"] for r in bc_rows], "o-", label=primary, color="#2563EB")
        ax.plot(floors, [r["achieved"][secondary]["mean"] for r in bc_rows], "s-", label=secondary, color="#E0762A")
        ax.plot(floors, [r["qc"] for r in bc_rows], "^-", label="QC", color="#55A868")
        ax_nz = ax.twinx()
        ax_nz.plot(floors, [r["nz_mean"] for r in bc_rows], "d:", color="#888")
        ax_nz.set_ylabel("mean nz", color="#888")
        ax.set_xlabel("floor (fix Au=0.30 Ga=0.20)")
        ax.set_title("B+C — fix + vary floor")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Behavioural sweeps for A (max_elements) · B (fixed_amounts) · C (min_nonzero_weight)\n"
        "dashed horizontal lines in A panels = unconstrained baseline; lower FE / higher Mag/klat / higher QC is better",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
