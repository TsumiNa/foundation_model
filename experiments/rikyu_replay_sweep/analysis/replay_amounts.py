#!/usr/bin/env python3
"""Per-task replay sample counts for each sweep setting, from measured n_valid_train."""

# n_valid_train per task (train-split valid labels), extracted from the run log.
N = {
    "density":23678,"efermi":23668,"final_energy":23678,"total_magnetization":23678,"volume":23678,
    "dielectric_total":3124,"dielectric_ionic":3124,"dielectric_electronic":3124,
    "magnetization":1160,"curie":6272,"neel":3466,"kp":3875,
    "magnetic_susceptibility":58,"zt":3445,"power_factor":3638,"thermal_conductivity":4272,
    "electrical_resistivity":5051,"dos_density":7009,"seebeck":8072,
    "formation_energy":23180,"magnetic_moment":851,"tc":7207,"klat":3863,"material_type":33556,
}
ORDER = ["density","efermi","final_energy","total_magnetization","volume","dielectric_total",
         "dielectric_ionic","dielectric_electronic","magnetization","curie","neel","kp",
         "magnetic_susceptibility","zt","power_factor","thermal_conductivity","electrical_resistivity",
         "dos_density","seebeck","formation_energy","magnetic_moment","tc","klat","material_type"]

FRACS = [0.05, 0.10, 0.15, 0.20]
COUNTS = [100, 200, 500, 1000]

def frac_amt(n, f):
    return round(f * n)

def count_amt(n, c):
    return min(c, n)  # clamp: min(1.0, c/n) * n

hdr = f"{'task':22} {'n_train':>7} | " + " ".join(f"{f:>5}" for f in FRACS) + " | " + " ".join(f"{c:>5}" for c in COUNTS)
print(hdr); print("-"*len(hdr))
for t in ORDER:
    n = N[t]
    fr = " ".join(f"{frac_amt(n,f):>5}" for f in FRACS)
    ct = ""
    for c in COUNTS:
        a = count_amt(n, c)
        mark = "*" if c > n else " "   # * = clamped to full (count exceeds available)
        ct += f"{a:>5}{mark}"[:6].rjust(6) if False else f"{a:>4}{mark} "
    print(f"{t:22} {n:>7} | {fr} | {ct}")
print("-"*len(hdr))
tot = sum(N[t] for t in ORDER)
print(f"{'TOTAL (all tasks)':22} {tot:>7} | " +
      " ".join(f"{round(f*tot):>5}" for f in FRACS) + " |  (fixed-count totals depend on clamping)")
print("\n'*' after a fixed-count value = clamped to 100% (requested count exceeds the task's n_train).")
print("For a step's replay, sum over the ALREADY-LEARNED tasks (the new task always uses n_train in full).")
