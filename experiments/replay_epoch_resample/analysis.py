# Compare per-epoch replay resampling (local MPS, steps 1-10) against the rikyu
# frozen-subset baselines over the same 10-step prefix.
#
# Local runs have no metrics_table.csv (stopped after step 10 by design) — rebuild the
# long table from the per-step <task>_metrics.json dumps, which carry the same numbers.
#
# Usage: .venv/bin/python experiments/replay_epoch_resample/analysis.py
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAX_STEP = 10

LOCAL = {
    "n200-epoch": ROOT / "artifacts/replay_sweep_epoch/replay_n200_epoch",
    "n500-epoch": ROOT / "artifacts/replay_sweep_epoch/replay_n500_epoch",
}
RIKYU = {
    "n200-step": ROOT / "artifacts/replay_sweep/replay_n200_rikyu",
    "n500-step": ROOT / "artifacts/replay_sweep/replay_n500_rikyu",
    "n1000-step": ROOT / "artifacts/replay_sweep/replay_n1000_rikyu",
}


def local_table(run_dir: Path) -> pd.DataFrame:
    rows = []
    for step_dir in sorted(run_dir.glob("training/step*_*")):
        step = int(step_dir.name[4:6])
        if step > MAX_STEP:
            continue
        for mfile in step_dir.glob("*_metrics.json"):
            task = mfile.name[: -len("_metrics.json")]
            m = json.loads(mfile.read_text())
            rows.append({"step": step, "task": task, "r2": m.get("r2", m.get("primary"))})
    return pd.DataFrame(rows)


def rikyu_table(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "training/metrics_table.csv")
    return df[df.step <= MAX_STEP][["step", "task", "r2"]]


tables = {name: local_table(p) for name, p in LOCAL.items()} | {name: rikyu_table(p) for name, p in RIKYU.items()}

order = (
    tables["n200-epoch"].groupby("task")["step"].min().sort_values().index.tolist()
)  # introduction order recovered from first appearance

# --- summary: mean R2 over seen tasks at each step, and at step 10 ---
print(f"{'run':<12} " + " ".join(f"s{s:02d}" for s in range(1, MAX_STEP + 1)) + "   mean@10")
summary = {}
for name, df in tables.items():
    per_step = df.groupby("step")["r2"].mean()
    final = df[df.step == MAX_STEP].set_index("task")["r2"]
    summary[name] = final
    cells = " ".join(f"{per_step.get(s, float('nan')):.2f}" for s in range(1, MAX_STEP + 1))
    print(f"{name:<12} {cells}   {final.mean():.4f}")

print("\nPer-task R2 at step 10:")
cmp = pd.DataFrame(summary).loc[[t for t in order if t in summary["n200-epoch"].index]]
print(cmp.round(3).to_string())

# --- figure: per-task retention trajectories ---
styles = {
    "n200-epoch": dict(color="tab:red", lw=2),
    "n500-epoch": dict(color="tab:orange", lw=2),
    "n200-step": dict(color="tab:blue", lw=1.2, ls="--"),
    "n500-step": dict(color="tab:cyan", lw=1.2, ls="--"),
    "n1000-step": dict(color="tab:green", lw=1.2, ls=":"),
}
fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex=True)
for ax, task in zip(axes.ravel(), order):
    for name, df in tables.items():
        t = df[df.task == task].sort_values("step")
        ax.plot(t.step, t.r2, label=name, **styles[name])
    ax.set_title(task, fontsize=9)
    ax.set_ylim(min(-0.1, cmp.loc[task].min() - 0.05), 1.02)
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7)
fig.suptitle("Replay retention, steps 1-10: per-epoch resampling (solid) vs frozen subsets (dashed)")
fig.supxlabel("pretraining step")
fig.supylabel("test R²")
fig.tight_layout()
out = ROOT / "artifacts/replay_sweep_epoch/epoch_vs_step_retention.png"
fig.savefig(out, dpi=150)
print(f"\nfigure: {out}")
