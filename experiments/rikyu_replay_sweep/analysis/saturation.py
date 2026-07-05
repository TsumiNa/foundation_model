#!/usr/bin/env python3
"""Replay data size vs final primary metric, + saturation-n estimate. Uses completed runs only."""
import csv
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"

ORDER = ["density","efermi","final_energy","total_magnetization","volume","dielectric_total",
         "dielectric_ionic","dielectric_electronic","magnetization","curie","neel","kp",
         "magnetic_susceptibility","zt","power_factor","thermal_conductivity","electrical_resistivity",
         "dos_density","seebeck","formation_energy","magnetic_moment","tc","klat","material_type"]
REG_LIKE = [t for t in ORDER if t != "material_type"]
DENSE = ["density","efermi","final_energy","total_magnetization","volume","formation_energy"]  # ~23678 labels

N_TRAIN = {"density":23678,"efermi":23668,"final_energy":23678,"total_magnetization":23678,"volume":23678,"formation_energy":23180}

def fnum(x):
    try: return float(x)
    except: return None

def load(path):
    rows = list(csv.DictReader(open(path)))
    mx = max(int(r["step"]) for r in rows)
    final = {r["task"]: fnum(r["primary"]) for r in rows if int(r["step"]) == mx}
    intro = {}
    for r in rows:
        if r["new_task"] == r["task"]:
            intro[r["task"]] = fnum(r["primary"])
    return final, intro

def mean(vs):
    vs = [v for v in vs if v is not None]
    return sum(vs)/len(vs) if vs else None

RUNS = {
    "base0p05": ("frac", 0.05),
    "0p10": ("frac", 0.10),
    "0p15": ("frac", 0.15),
    "0p20": ("frac", 0.20),
    "n100": ("count", 100),
    "n200": ("count", 200),
    "n500": ("count", 500),
    "n1000": ("count", 1000),
    "n1500": ("count", 1500),
    "n2000": ("count", 2000),
    "n2500": ("count", 2500),
}
data = {}
for tag,(kind,val) in RUNS.items():
    p = RES / f"mt_{tag}.csv"
    if not p.exists(): continue
    final, intro = load(p)
    data[tag] = dict(kind=kind, val=val, final=final, intro=intro)

# ceiling = at-intro mean reg/kr R2 (averaged across available runs)
intro_means = [mean([d["intro"].get(t) for t in REG_LIKE]) for d in data.values()]
CEIL = float(np.mean(intro_means))
print(f"at-intro mean reg/kr R2 (no-forgetting ceiling) ~ {CEIL:.3f}")
print()
print(f"{'run':9} {'kind':6} {'val':>6} | {'final meanR2':>12} {'dense meanR2':>12}")
for tag,d in data.items():
    fm = mean([d["final"].get(t) for t in REG_LIKE])
    dm = mean([d["final"].get(t) for t in DENSE])
    print(f"{tag:9} {d['kind']:6} {d['val']:>6} | {fm:12.3f} {dm:12.3f}")

# --- fixed-count series: mean reg/kr R2 vs n ---
fixed = [(d["val"], mean([d["final"].get(t) for t in REG_LIKE])) for d in data.values() if d["kind"]=="count"]
fixed.sort()
print("\nfixed-count series (n, mean reg/kr R2):", [(n, round(y,3)) for n,y in fixed])

# --- dense-task unified curve: x = replay samples per dense task, y = dense mean R2 ---
pts = []
for tag,d in data.items():
    dm = mean([d["final"].get(t) for t in DENSE])
    if d["kind"]=="count":
        x = d["val"]                        # each dense task gets exactly val samples
    else:
        x = d["val"]*np.mean([N_TRAIN[t] for t in DENSE])  # frac * ~23678
    pts.append((x, dm, tag))
pts.sort()
print("\ndense-task curve (replay/task, dense meanR2, run):")
for x,y,tag in pts:
    print(f"  {x:8.0f}  {y:.3f}  {tag}")

# --- saturation fit on dense curve: y = C - (C-y0)/(1 + x/k)  (Michaelis-Menten-like gap decay) ---
X = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts])
Cd = float(np.mean([mean([d["intro"].get(t) for t in DENSE]) for d in data.values()]))
print(f"\ndense at-intro ceiling ~ {Cd:.3f}")
# fit gap = C - y = g0 / (1 + x/k)  ->  1/gap = (1/g0) + (1/(g0*k)) * x  (linear in x)
gap = Cd - Y
good = gap > 1e-3
inv = 1.0/gap[good]
A = np.vstack([np.ones(good.sum()), X[good]]).T
coef, *_ = np.linalg.lstsq(A, inv, rcond=None)
a, b = coef              # 1/gap = a + b*x  -> g0 = 1/a, k = a/b
g0 = 1.0/a; k = a/b
print(f"fit: gap(x) = {g0:.3f} / (1 + x/{k:.0f})   (ceiling {Cd:.3f})")
for frac_sat in (0.90, 0.95, 0.99):
    # y reaches frac_sat of the way from y(0)=Cd-g0 to Cd  => gap = (1-frac_sat)*g0
    # g0/(1+x/k) = (1-frac_sat)*g0 -> x = k*(1/(1-frac_sat) - 1)
    xn = k*(1/(1-frac_sat) - 1)
    print(f"  {int(frac_sat*100)}% saturation (dense): replay/task n ~ {xn:6.0f}  (final dense R2 ~ {Cd-(1-frac_sat)*g0:.3f})")
