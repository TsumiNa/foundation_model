#!/usr/bin/env python3
"""Build a self-contained interactive HTML viewer for the inverse-design candidate compositions.

Usage: python build_trajectory_viewer.py [MIRROR_DIR] [REPLAY_N]
       (defaults: <repo>/artifacts/task_scaling, 1000)

Panels under mode / order / k / target-scenario / path dropdowns + a step slider:
  1. heatmap — elements × the 20 candidates, shade = weight at the current step (final-step
     diversity stats in the header); click a column to select a candidate.
  2. progress curves — the selected candidate's per-target progress (0 = seed, 1 = target).
  3. composition bars — the selected candidate at the current step.
  4. seeds → optimized — all 20 rows: the seed composition and what it became at the final step
     (top-4 elements each); click a row to select.

Rendering is HiDPI-aware (devicePixelRatio-scaled canvases — crisp on retina displays); element
weights are embedded as permille integers to keep the file small. Emits viewer_n<REPLAY_N>.html.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

HERE = Path(__file__).resolve().parent
MIRROR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "artifacts/task_scaling"
REPLAY_N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TAG = "" if REPLAY_N == 1000 else f"_n{REPLAY_N}"
SCENARIOS = ["fe_down_total_up", "fe_down_ionic_up", "fe_down_electronic_up"]
PATHS = ["latent_default", "comp_k4"]
LINE_FRAMES = 21
HM_FRAMES = 12
MAX_ROWS = 12
TARGET_VALUES = {"formation_energy": -1.0, "dielectric_total": 1.0, "dielectric_ionic": 1.0, "dielectric_electronic": 1.0}
TRUTH_COLS = {
    "formation_energy": "Formation energy per atom (normalized)",
    "dielectric_total": "Dielectric total (normalized)",
    "dielectric_ionic": "Dielectric ionic (normalized)",
    "dielectric_electronic": "Dielectric electronic (normalized)",
}
_qc = pd.read_parquet(HERE.parents[2] / "data/qc_ac_te_mp_dos_reformat_20260515.pd.parquet",
                      columns=["composition", *TRUTH_COLS.values()]).set_index("composition")


def truths(comp: str, tasks: list[str]) -> list[float | None]:
    if comp not in _qc.index:
        return [None] * len(tasks)
    row = _qc.loc[comp]
    out = []
    for t_ in tasks:
        v = row[TRUTH_COLS[t_]]
        out.append(None if pd.isna(v) else round(float(v), 2))
    return out


def top_pairs(w: np.ndarray, top: int = 4) -> list[list[int]]:
    """Top elements of one composition as [element_index, permille] pairs."""
    idx = np.argsort(w)[::-1][:top]
    return [[int(i), int(round(float(w[int(i)]) * 1000))] for i in idx if w[int(i)] > 1e-3]


def build_entry(targets: np.ndarray, weights: np.ndarray, labels: list[str], seed_comps: list[str]) -> dict:
    steps, n_seeds, _ = targets.shape
    tasks = [la.split("→")[0].split("~")[0] for la in labels]
    tvals = np.array([TARGET_VALUES[t] for t in tasks])
    line_ids = sorted(set(np.linspace(0, steps - 1, LINE_FRAMES).astype(int).tolist()))
    hm_ids = sorted(set(np.linspace(0, steps - 1, HM_FRAMES).astype(int).tolist()))

    y0 = targets[0]
    denom = tvals[None, :] - y0
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    prog = np.transpose((targets[np.array(line_ids)] - y0[None]) / denom[None], (1, 0, 2))

    best = int(np.argmin(np.abs(targets[-1] - tvals[None, :]).sum(axis=1)))

    have_w = weights.shape[0] == steps
    if have_w:
        final = weights[-1]
        union: set[int] = set()
        for b in range(n_seeds):
            union.update(int(i) for i in np.argsort(final[b])[::-1][:4] if final[b, int(i)] > 1e-3)
        rows = sorted(union, key=lambda i: -float(final[:, i].mean()))[:MAX_ROWS]
        hm = np.round(weights[np.array(hm_ids)][:, :, rows] * 1000).astype(int)  # permille
        systems = {tuple(sorted(np.argsort(final[b])[::-1][:3].tolist())) for b in range(n_seeds)}
        l1 = float(np.mean([np.abs(final[a] - final[c]).sum() for a, c in combinations(range(n_seeds), 2)])) if n_seeds > 1 else 0.0
        div = {"systems": len(systems), "l1": round(l1, 2)}
        so = []
        for b in range(n_seeds):
            comp = seed_comps[b] if b < len(seed_comps) else ""
            so.append(
                {
                    "s": top_pairs(weights[0, b]),
                    "f": top_pairs(final[b]),
                    "tv": truths(comp, tasks),  # seed ground truth per target task (None = no label)
                    "sp": [round(float(x), 2) for x in targets[0, b]],  # predicted at the seed
                    "fp": [round(float(x), 2) for x in targets[-1, b]],  # predicted at the final step
                }
            )
        hm_rows = rows
        hm_list = hm.tolist()
    else:
        hm_rows, hm_list, div, so = [], [], {"systems": 0, "l1": 0.0}, []

    return {
        "tasks": tasks,
        "line_steps": [int(s) for s in line_ids],
        "prog": np.round(prog, 2).tolist(),
        "hm_steps": [int(s) for s in hm_ids],
        "rows": hm_rows,  # element indices; symbols resolved via the global ELEMS table
        "hm": hm_list,  # (frame, seed, row) permille ints
        "so": so,  # per seed: seed composition + final optimised composition (top-4 pairs)
        "best": best,
        "div": div,
    }


data: dict = {}
n_loaded = 0
for mode in ("ws", "ft"):
    for ord_id in range(3):
        for k in range(1, 22):
            for scen in SCENARIOS:
                for path in PATHS:
                    p = MIRROR / f"{mode}{TAG}_o{ord_id}/k{k:02d}/inverse3/{scen}/trajectories/{path}.npz"
                    if not p.exists():
                        continue
                    z = np.load(p, allow_pickle=False)
                    seeds_json = p.parents[2] / "seeds.json"
                    seed_comps = json.loads(seeds_json.read_text())["seeds"] if seeds_json.exists() else []
                    entry = build_entry(z["targets"], z["weights"], [str(x) for x in z["labels"]], seed_comps)
                    data.setdefault(mode, {}).setdefault(str(ord_id), {}).setdefault(str(k), {}).setdefault(scen, {})[path] = entry
                    n_loaded += 1
print(f"loaded {n_loaded} trajectory files")

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Inverse candidates — replay n=%%N%%</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;color:#1a1a2e}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
select,button{font-size:14px;padding:2px 6px}
#slider{width:300px}
#divline{color:#374151;font-size:13px;margin:4px 0 6px 0}
.panels{display:flex;gap:10px;flex-wrap:wrap}
#solist{font:12px ui-monospace,Menlo,monospace;line-height:1.75;max-height:420px;overflow-y:auto;
  border:1px solid #e5e7eb;border-radius:6px;padding:8px 10px;min-width:430px}
#solist .row{cursor:pointer;white-space:nowrap}
#solist .row:hover{background:#f3f4f6}
#solist .sel{background:#fdeaea}
#tip{position:fixed;background:#1a1a2e;color:#fff;padding:3px 7px;border-radius:4px;font-size:12px;pointer-events:none;display:none}
.legend{color:#6b7280;font-size:12.5px;margin-top:6px;max-width:1120px}
</style></head><body>
<h3>Inverse-design candidates — composition diversity across targets, paths and k (replay n=%%N%%)</h3>
<div class="controls">
  mode <select id="mode"></select> order <select id="ord"></select> k <select id="k"></select>
  target <select id="scen"></select> path <select id="path"></select>
  step <input type="range" id="slider" min="0" value="0"> <span id="stepLabel"></span>
  <button id="play">▶ play</button>
</div>
<div id="divline"></div>
<canvas id="hm"></canvas>
<div class="panels">
  <canvas id="line"></canvas>
  <canvas id="bar"></canvas>
  <div id="solist"></div>
</div>
<div class="legend"><b>heatmap</b>: elements × 20 candidates, shade = weight at the current step (▲ = best final; click a column
to select) · <b>curves</b>: the selected candidate's per-target progress (0 = seed, 1 = target, dashed) · <b>bars</b>: its
composition at the current step · <b>list</b>: every candidate's seed composition → final optimised composition (top-4 elements,
numbers = fraction; click to select). Latent-path compositions are KMD-decoded from the optimised latent; composition-path
ones are the recipe itself.</div>
<div id="tip"></div>
<script>
const DATA = %%DATA%%;
const ELEMS = %%ELEMS%%;
const COLORS = ["#2563EB","#55A868","#E67E22","#9467bd"];
const DPR = window.devicePixelRatio || 1;
const sel = id => document.getElementById(id);
const cnv = {hm:sel("hm"), line:sel("line"), bar:sel("bar")};
const ctx = {};
function setup(c, w, h){ c.style.width=w+"px"; c.style.height=h+"px"; c.width=Math.round(w*DPR); c.height=Math.round(h*DPR);
  const x=c.getContext("2d"); x.setTransform(DPR,0,0,DPR,0,0); return x; }
ctx.hm = setup(cnv.hm, 1100, 320); ctx.line = setup(cnv.line, 560, 420); ctx.bar = setup(cnv.bar, 340, 420);
const HMW=1100, HMH=320, LNW=560, LNH=420, BRW=340, BRH=420;
const tip = sel("tip");
let seed = 0, hmCells = [];
const fmt = pairs => pairs.map(([e,pm])=>ELEMS[e]+(pm/1000).toFixed(2)).join(" ") || "-";
function fill(el, opts, keep){ const v=el.value; el.innerHTML = opts.map(o=>`<option>${o}</option>`).join("");
  if(keep && opts.includes(v)) el.value = v; }
function keys(o){ return Object.keys(o); }
function cur(){ const m=sel("mode").value,o=sel("ord").value,k=sel("k").value,s=sel("scen").value,p=sel("path").value;
  return ((((DATA[m]||{})[o]||{})[k]||{})[s]||{})[p] || null; }
function refreshMenus(resetSeed){
  fill(sel("mode"), keys(DATA), true);
  const m=sel("mode").value; fill(sel("ord"), keys(DATA[m]), true);
  const o=sel("ord").value; fill(sel("k"), keys(DATA[m][o]).sort((a,b)=>+a-+b), true);
  const k=sel("k").value; fill(sel("scen"), keys(DATA[m][o][k]), true);
  const s=sel("scen").value; fill(sel("path"), keys(DATA[m][o][k][s]), true);
  const d=cur(); if(d && resetSeed) seed = d.best;
}
function hmFrame(d, step){ return d.hm_steps.reduce((acc,s,i)=> s<=step? i: acc, 0); }
function draw(){
  const d=cur(); if(!d) return;
  const F=d.line_steps.length, f=Math.min(+sel("slider").value,F-1);
  sel("slider").max=F-1;
  const step=d.line_steps[f], total=d.line_steps[F-1]+1;
  sel("stepLabel").textContent=`step ${step+1}/${total}`;
  sel("divline").textContent=`final-step diversity of the 20 candidates: ${d.div.systems} distinct top-3 element systems · mean pairwise L1 = ${d.div.l1}  (0 = identical recipes, 2 = no shared elements)`;

  // ===== heatmap =====
  const H=ctx.hm, rows=d.rows.length, B=d.prog.length;
  H.clearRect(0,0,HMW,HMH); hmCells=[];
  if(rows){
    const bf=hmFrame(d, step), grid=d.hm[bf];
    const L=50,T=28, cw=(HMW-L-16)/B, chh=(HMH-T-22)/rows;
    let maxw=1; grid.forEach(r=>r.forEach(v=>{if(v>maxw)maxw=v;}));
    H.font="11px sans-serif";
    for(let r=0;r<rows;r++){
      H.fillStyle="#374151"; H.fillText(ELEMS[d.rows[r]], 14, T+r*chh+chh*0.65);
      for(let b=0;b<B;b++){
        const v=grid[b][r]/maxw;
        H.fillStyle=`rgba(37,99,235,${Math.min(1,v).toFixed(2)})`;
        H.fillRect(L+b*cw+1, T+r*chh+1, cw-2, chh-2);
        hmCells.push({x:L+b*cw,y:T+r*chh,w:cw,h:chh,b,elem:ELEMS[d.rows[r]],v:grid[b][r]/1000});
      }
    }
    for(let b=0;b<B;b++){
      H.fillStyle = b===seed? "#C44E52" : "#6b7280";
      H.fillText((b===d.best?"▲":"")+b, L+b*cw+cw/2-8, 17);
      if(b===seed){H.strokeStyle="#C44E52";H.lineWidth=1.5;H.strokeRect(L+b*cw,T,cw,rows*chh);}
    }
    H.fillStyle="#6b7280"; H.fillText("candidate (▲ = best final) — click a column to inspect", L, HMH-6);
  }

  // ===== progress curves =====
  const X2=ctx.line, P=d.prog[seed];
  let vals=[]; P.forEach(row=>row.forEach(v=>vals.push(v))); vals.push(0,1.05);
  const y0=Math.min(...vals)-0.05, y1=Math.max(...vals)+0.05;
  const L=50,R=12,T=20,Bm=42, W=LNW-L-R, Hh=LNH-T-Bm;
  const px=i=>L+d.line_steps[i]/(total-1)*W, py=v=>T+Hh-(v-y0)/(y1-y0)*Hh;
  X2.clearRect(0,0,LNW,LNH);
  X2.strokeStyle="#e5e7eb"; X2.fillStyle="#6b7280"; X2.font="11px sans-serif";
  for(let i=0;i<=5;i++){const vy=y0+(y1-y0)*i/5;
    X2.beginPath();X2.moveTo(L,py(vy));X2.lineTo(L+W,py(vy));X2.stroke();
    X2.fillText(vy.toFixed(1),14,py(vy)+4);}
  X2.strokeStyle="#666";X2.setLineDash([5,4]);
  X2.beginPath();X2.moveTo(L,py(1));X2.lineTo(L+W,py(1));X2.stroke();X2.setLineDash([]);
  d.tasks.forEach((t,c)=>{
    X2.strokeStyle=COLORS[c%COLORS.length];X2.lineWidth=2;X2.beginPath();
    P.forEach((row,i)=>{ i? X2.lineTo(px(i),py(row[c])) : X2.moveTo(px(i),py(row[c])); });
    X2.stroke();
    X2.fillStyle=COLORS[c%COLORS.length];X2.fillRect(L+8,T+4+c*15,10,3);
    X2.fillText(t,L+22,T+9+c*15);
  });
  X2.strokeStyle="#444";X2.lineWidth=1.4;
  X2.beginPath();X2.moveTo(px(f),T);X2.lineTo(px(f),T+Hh);X2.stroke();
  X2.fillStyle="#6b7280";
  X2.fillText(`candidate ${seed}${seed===d.best?" (best)":""}`, L+W/2-40, 12);
  X2.fillText("Optimisation step", L+W/2-46, LNH-8);
  for(let i=0;i<=4;i++){const s=Math.round((total-1)*i/4);
    X2.fillText(String(s), L+s/(total-1)*W-8, T+Hh+16);}

  // ===== bars =====
  const X3=ctx.bar;
  X3.clearRect(0,0,BRW,BRH);
  if(rows){
    const bf=hmFrame(d, step), w=d.rows.map((_,r)=>d.hm[bf][seed][r]/1000);
    X3.fillStyle="#1a1a2e"; X3.font="bold 12px sans-serif";
    X3.fillText(`candidate ${seed} @ step ${d.hm_steps[bf]+1}`, 60, 16);
    const bh=Math.min(22,(BRH-64)/rows), maxw=Math.max(0.4,...w);
    d.rows.forEach((ei,i)=>{
      const y=30+i*(bh+5);
      X3.fillStyle="#6b7280"; X3.font="12px sans-serif"; X3.fillText(ELEMS[ei], 14, y+bh*0.7);
      X3.fillStyle="#2563EB"; X3.fillRect(44,y,(w[i]||0)/maxw*(BRW-104),bh);
      X3.fillStyle="#374151"; X3.fillText((w[i]||0).toFixed(2), 48+(w[i]||0)/maxw*(BRW-104), y+bh*0.7);
    });
  }

  // ===== seeds -> optimized list =====
  const so = d.so || [];
  const vv = v => v==null? "–" : v.toFixed(2);
  const pair = (arr)=> d.tasks.map((t,i)=>`${t.replace("dielectric_","diel_").replace("formation_energy","FE")} ${vv(arr?arr[i]:null)}`).join(", ");
  sel("solist").innerHTML = "<b>seed composition → final optimised</b> <span style='color:#6b7280'>(hover a row for true/predicted values)</span><br>" + so.map((r,b)=>
    `<div class="row ${b===seed?'sel':''}" data-b="${b}" title="seed TRUE: ${pair(r.tv)}\nseed predicted: ${pair(r.sp)}\nfinal predicted: ${pair(r.fp)}">${b===d.best?"▲":"&nbsp;"}${String(b).padStart(2)} ${fmt(r.s)} → ${fmt(r.f)}</div>`
  ).join("");
  document.querySelectorAll("#solist .row").forEach(el=>el.addEventListener("click",()=>{seed=+el.dataset.b;draw();}));
}
["mode","ord","k","scen","path"].forEach(id=>sel(id).addEventListener("change",()=>{refreshMenus(id==="mode"||id==="ord");
  const d=cur(); if(d && seed>=d.prog.length) seed=d.best; draw();}));
sel("slider").addEventListener("input",draw);
let timer=null;
sel("play").addEventListener("click",()=>{
  if(timer){clearInterval(timer);timer=null;sel("play").textContent="▶ play";return;}
  sel("play").textContent="⏸ pause";
  timer=setInterval(()=>{const s=sel("slider");s.value=(+s.value+1)%(+s.max+1);draw();},160);
});
cnv.hm.addEventListener("click",e=>{
  const r=cnv.hm.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const c=hmCells.find(c=>mx>=c.x&&mx<c.x+c.w&&my>=c.y&&my<c.y+c.h);
  if(c){seed=c.b;draw();}
});
cnv.hm.addEventListener("mousemove",e=>{
  const r=cnv.hm.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const c=hmCells.find(c=>mx>=c.x&&mx<c.x+c.w&&my>=c.y&&my<c.y+c.h);
  if(c){tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";
    tip.textContent=`candidate ${c.b} · ${c.elem} = ${c.v.toFixed(3)}`;}
  else tip.style.display="none";
});
refreshMenus(true); draw();
</script></body></html>
"""

out = HERE / f"viewer_n{REPLAY_N}.html"
out.write_text(
    html.replace("%%DATA%%", json.dumps(data, separators=(",", ":")))
    .replace("%%ELEMS%%", json.dumps(list(DEFAULT_ELEMENTS), separators=(",", ":")))
    .replace("%%N%%", str(REPLAY_N))
)
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
