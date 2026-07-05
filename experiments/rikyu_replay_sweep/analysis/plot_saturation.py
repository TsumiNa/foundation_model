#!/usr/bin/env python3
"""Plot replay data size vs final dense-task retention, with a saturation fit + n estimate."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
OUT = str(Path(__file__).resolve().parent / "replay_saturation.png")

DENSE = ["density","efermi","final_energy","total_magnetization","volume","formation_energy"]
NTR = 23678  # ~ dense task train labels

def fnum(x):
    try: return float(x)
    except: return None

def load(path):
    rows = list(csv.DictReader(open(path)))
    mx = max(int(r["step"]) for r in rows)
    final = {r["task"]: fnum(r["primary"]) for r in rows if int(r["step"]) == mx}
    intro = {r["task"]: fnum(r["primary"]) for r in rows if r["new_task"] == r["task"]}
    return final, intro

def dmean(d):
    v = [d[t] for t in DENSE if d.get(t) is not None]
    return sum(v)/len(v)

RUNS = {"n100":("count",100),"n200":("count",200),"n500":("count",500),"n1000":("count",1000),
        "n1500":("count",1500),"n2000":("count",2000),"n2500":("count",2500),
        "base0p05":("frac",0.05),"0p10":("frac",0.10),"0p15":("frac",0.15),"0p20":("frac",0.20)}
fixed_x, fixed_y, frac_x, frac_y, ceils = [], [], [], [], []
labels = {}
for tag,(kind,val) in RUNS.items():
    p = RES/f"mt_{tag}.csv"
    if not p.exists(): continue
    final, intro = load(p)
    x = val if kind=="count" else val*NTR
    y = dmean(final)
    ceils.append(dmean(intro))
    if kind=="count": fixed_x.append(x); fixed_y.append(y)
    else: frac_x.append(x); frac_y.append(y)
    labels[x] = (tag, kind, val, y)

CEIL = float(np.mean(ceils))   # no-forgetting (at-intro) dense ceiling

# fit gap = C - y = g0/(1 + x/k)  ->  1/gap linear in x
X = np.array(fixed_x+frac_x); Y = np.array(fixed_y+frac_y)
gap = CEIL - Y
A = np.vstack([np.ones(len(X)), X]).T
a,b = np.linalg.lstsq(A, 1.0/gap, rcond=None)[0]
g0, k = 1.0/a, a/b
def yfit(x): return CEIL - g0/(1+x/k)
def n_for(frac): return k*(1/(1-frac)-1)
n90, n95 = n_for(0.90), n_for(0.95)

# ---- plot ----
INK="#1a1a2e"; MUTED="#6b7280"; GRID="#e5e7eb"
BLUE="#0077BB"; ORANGE="#EE7733"; CURVE="#334155"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":MUTED})
fig, ax = plt.subplots(figsize=(9,5.6), dpi=150)

xs = np.logspace(np.log10(80), np.log10(NTR), 300)
ax.plot(xs, yfit(xs), color=CURVE, lw=2, zorder=2,
        label=f"saturation fit:  gap = {g0:.2f}/(1+n/{k:.0f})")
ax.axhline(CEIL, color=MUTED, ls="--", lw=1.4, zorder=1)
ax.text(NTR*0.98, CEIL+0.004, f"no-forgetting ceiling ≈ {CEIL:.2f}  (full replay / joint)",
        ha="right", va="bottom", color=MUTED, fontsize=9.5)

# saturation guides
for xn,fr in [(n90,90),(n95,95)]:
    ax.axvline(xn, color=GRID, lw=1.2, ls=":", zorder=0)
    ax.text(xn, 0.462, f"{fr}% sat\nn≈{xn:,.0f}", ha="center", va="bottom", fontsize=9, color=MUTED)

ax.scatter(fixed_x, fixed_y, s=90, marker="o", color=BLUE, zorder=3,
           edgecolor="white", linewidth=1.2, label="fixed count  (n per task)")
ax.scatter(frac_x, frac_y, s=95, marker="s", color=ORANGE, zorder=3,
           edgecolor="white", linewidth=1.2, label="fraction  (→ samples/dense-task)")
for x,(tag,kind,val,y) in labels.items():
    lab = f"{val:g}" if kind=="count" else f"{val:g}×"
    dy = 10 if kind == "count" else -17   # counts label above, fractions below -> no collision
    ax.annotate(lab, (x,y), textcoords="offset points", xytext=(0,dy),
                ha="center", fontsize=9, color=(BLUE if kind=="count" else ORANGE))

ax.set_xscale("log")
ax.set_xlim(80, NTR*1.05)
ax.set_ylim(0.45, 0.78)
ax.set_xlabel("replay samples per dense qc task  (log scale)", color=INK)
ax.set_ylabel("final dense-task mean R²  (retention)", color=INK)
ax.set_title("Replay data size vs retention — continual-rehearsal (24 tasks, GB200)", color=INK, fontsize=12.5)
ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
ax.tick_params(colors=MUTED)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
fig.text(0.012,0.012,f"{len(X)} completed runs. Fit extrapolation beyond n≈{max(X):,.0f} is uncertain.",
         fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
print(f"ceiling={CEIL:.3f}  g0={g0:.3f}  k={k:.0f}")
print(f"90% sat n≈{n90:,.0f}   95% sat n≈{n95:,.0f}")
print(f"knee (half-gap) at n≈k={k:.0f}")
