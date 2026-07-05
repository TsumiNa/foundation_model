#!/usr/bin/env python3
"""Build a self-contained interactive HTML viewer for the inverse-design candidate compositions.

Usage: python build_trajectory_viewer.py [MIRROR_DIR] [REPLAY_N]
       (defaults: <repo>/artifacts/task_scaling, 1000)

Focus: the DIVERSITY of the designed compositions across optimisation paths and pretraining
checkpoints k. Three linked panels under mode/order/k/target-scenario/path dropdowns + a step slider:

  1. heatmap — elements (rows) × the 20 candidates (columns), cell shade = element weight at the
     current step. Flipping k or path re-renders instantly, so ensemble differences are directly
     visible; the header shows diversity stats of the FINAL step (distinct top-3 element systems
     among the 20 candidates + their mean pairwise L1 distance). Click a column to inspect it.
  2. progress curves — per-target progress of the selected candidate (0 = seed, 1 = target).
  3. composition bars — the selected candidate at the current step (same rows as the heatmap).

Everything is embedded; open in any browser. Emits viewer_n<REPLAY_N>.html.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

HERE = Path(__file__).resolve().parent
MIRROR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "artifacts/task_scaling"
REPLAY_N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TAG = "" if REPLAY_N == 1000 else f"_n{REPLAY_N}"
SCENARIOS = ["fe_down_total_up", "fe_down_ionic_up", "fe_down_electronic_up"]
PATHS = ["latent_default", "comp_k4"]
LINE_FRAMES = 21
HM_FRAMES = 12
MAX_ROWS = 12  # heatmap element rows (union of the candidates' final top-4s, by mean weight)
TARGET_VALUES = {"formation_energy": -1.0, "dielectric_total": 1.0, "dielectric_ionic": 1.0, "dielectric_electronic": 1.0}


def build_entry(targets: np.ndarray, weights: np.ndarray, labels: list[str]) -> dict:
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
        final = weights[-1]  # (B, 94)
        # Common heatmap rows: union of every candidate's final top-4, ranked by mean final weight.
        union: set[int] = set()
        for b in range(n_seeds):
            union.update(int(i) for i in np.argsort(final[b])[::-1][:4] if final[b, int(i)] > 1e-3)
        rows = sorted(union, key=lambda i: -float(final[:, i].mean()))[:MAX_ROWS]
        hm = np.round(weights[np.array(hm_ids)][:, :, rows], 3)  # (F, B, R)
        systems = {tuple(sorted(np.argsort(final[b])[::-1][:3].tolist())) for b in range(n_seeds)}
        l1 = float(np.mean([np.abs(final[a] - final[c]).sum() for a, c in combinations(range(n_seeds), 2)])) if n_seeds > 1 else 0.0
        div = {"systems": len(systems), "l1": round(l1, 2)}
        hm_elems = [DEFAULT_ELEMENTS[i] for i in rows]
        hm_list = hm.tolist()
    else:
        hm_elems, hm_list, div = [], [], {"systems": 0, "l1": 0.0}

    return {
        "tasks": tasks,
        "line_steps": [int(s) for s in line_ids],
        "prog": np.round(prog, 3).tolist(),
        "hm_steps": [int(s) for s in hm_ids],
        "hm_elems": hm_elems,
        "hm": hm_list,
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
                    entry = build_entry(z["targets"], z["weights"], [str(x) for x in z["labels"]])
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
.panels{display:flex;gap:8px}
#tip{position:fixed;background:#1a1a2e;color:#fff;padding:3px 7px;border-radius:4px;font-size:12px;pointer-events:none;display:none}
.legend{color:#6b7280;font-size:12.5px;margin-top:6px;max-width:1100px}
</style></head><body>
<h3>Inverse-design candidates — composition diversity across targets, paths and k (replay n=%%N%%)</h3>
<div class="controls">
  mode <select id="mode"></select> order <select id="ord"></select> k <select id="k"></select>
  target <select id="scen"></select> path <select id="path"></select>
  step <input type="range" id="slider" min="0" value="0"> <span id="stepLabel"></span>
  <button id="play">▶ play</button>
</div>
<div id="divline"></div>
<canvas id="hm" width="1090" height="330"></canvas>
<div class="panels"><canvas id="line" width="640" height="400"></canvas><canvas id="bar" width="440" height="400"></canvas></div>
<div class="legend"><b>top</b>: elements × the 20 candidates, shade = weight at the current step — flip k / path to compare the
ensembles; ▲ marks the best-final candidate, click a column to inspect it below · <b>bottom left</b>: the selected candidate's
per-target progress (0 = seed, 1 = target, dashed) with the current-step marker · <b>bottom right</b>: its composition at the
current step. Latent-path compositions are KMD-decoded from the optimised latent; composition-path ones are the recipe itself.</div>
<div id="tip"></div>
<script>
const DATA = %%DATA%%;
const COLORS = ["#2563EB","#55A868","#E67E22","#9467bd"];
const sel = id => document.getElementById(id);
const cnv = {hm:sel("hm"), line:sel("line"), bar:sel("bar")};
const ctx = {hm:cnv.hm.getContext("2d"), line:cnv.line.getContext("2d"), bar:cnv.bar.getContext("2d")};
const tip = sel("tip");
let seed = 0, hmCells = [];
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
  sel("divline").textContent=`final-step diversity of the 20 candidates: ${d.div.systems} distinct top-3 element systems · mean pairwise L1 = ${d.div.l1}`;

  // ===== heatmap =====
  const H=ctx.hm, rows=d.hm_elems.length, B=d.prog.length;
  H.clearRect(0,0,cnv.hm.width,cnv.hm.height);
  hmCells=[];
  if(rows){
    const bf=hmFrame(d, step), grid=d.hm[bf];
    const L=52,T=30, cw=(cnv.hm.width-L-20)/B, ch=(cnv.hm.height-T-24)/rows;
    let maxw=0.001; grid.forEach(r=>r.forEach(v=>{if(v>maxw)maxw=v;}));
    H.font="11px sans-serif";
    for(let r=0;r<rows;r++){
      H.fillStyle="#374151"; H.fillText(d.hm_elems[r], 14, T+r*ch+ch*0.65);
      for(let b=0;b<B;b++){
        const v=grid[b][r]/maxw;
        H.fillStyle=`rgba(37,99,235,${Math.min(1,v).toFixed(2)})`;
        H.fillRect(L+b*cw+1, T+r*ch+1, cw-2, ch-2);
        hmCells.push({x:L+b*cw,y:T+r*ch,w:cw,h:ch,b,elem:d.hm_elems[r],v:grid[b][r]});
      }
    }
    for(let b=0;b<B;b++){
      H.fillStyle = b===seed? "#C44E52" : "#6b7280";
      H.fillText((b===d.best?"▲":"")+b, L+b*cw+cw/2-8, 18);
      if(b===seed){H.strokeStyle="#C44E52";H.lineWidth=1.5;H.strokeRect(L+b*cw,T,cw,rows*(cnv.hm.height-T-24)/rows);}
    }
    H.fillStyle="#6b7280"; H.fillText("candidate (▲ = best final) — click a column to inspect", L, cnv.hm.height-6);
  }

  // ===== progress curves =====
  const X2=ctx.line, P=d.prog[seed];
  let vals=[]; P.forEach(row=>row.forEach(v=>vals.push(v))); vals.push(0,1.05);
  const y0=Math.min(...vals)-0.05, y1=Math.max(...vals)+0.05;
  const L=52,R=14,T=20,Bm=44, W=cnv.line.width-L-R, Hh=cnv.line.height-T-Bm;
  const px=i=>L+d.line_steps[i]/(total-1)*W, py=v=>T+Hh-(v-y0)/(y1-y0)*Hh;
  X2.clearRect(0,0,cnv.line.width,cnv.line.height);
  X2.strokeStyle="#e5e7eb"; X2.fillStyle="#6b7280"; X2.font="11px sans-serif";
  for(let i=0;i<=5;i++){const vy=y0+(y1-y0)*i/5;
    X2.beginPath();X2.moveTo(L,py(vy));X2.lineTo(L+W,py(vy));X2.stroke();
    X2.fillText(vy.toFixed(1),16,py(vy)+4);}
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
  X2.fillText(`candidate ${seed}${seed===d.best?" (best)":""} — progress`, L+W/2-70, 12);
  X2.fillText("Optimisation step", L+W/2-46, cnv.line.height-8);
  for(let i=0;i<=4;i++){const s=Math.round((total-1)*i/4);
    X2.fillText(String(s), L+s/(total-1)*W-8, T+Hh+16);}

  // ===== composition bars for the selected candidate =====
  const X3=ctx.bar;
  X3.clearRect(0,0,cnv.bar.width,cnv.bar.height);
  if(rows){
    const bf=hmFrame(d, step), col=d.hm[bf].map(r=>r), w=d.hm[bf][0].map((_,r)=>d.hm[bf][seed][r]);
    X3.fillStyle="#1a1a2e"; X3.font="bold 12px sans-serif";
    X3.fillText(`Composition — candidate ${seed} (step ${d.hm_steps[bf]+1})`, 90, 16);
    const bh=Math.min(24,(cnv.bar.height-64)/rows), maxw=Math.max(0.4,...w);
    d.hm_elems.forEach((e,i)=>{
      const y=30+i*(bh+5);
      X3.fillStyle="#6b7280"; X3.font="12px sans-serif"; X3.fillText(e, 18, y+bh*0.7);
      X3.fillStyle="#2563EB"; X3.fillRect(50,y,(w[i]||0)/maxw*(cnv.bar.width-104),bh);
      X3.fillStyle="#374151"; X3.fillText((w[i]||0).toFixed(2), 54+(w[i]||0)/maxw*(cnv.bar.width-104), y+bh*0.7);
    });
    X3.fillStyle="#6b7280"; X3.fillText("weight", cnv.bar.width/2-18, cnv.bar.height-8);
  } else {
    X3.fillStyle="#6b7280"; X3.fillText("no per-step weights recorded for this run", 60, 60);
  }
}
["mode","ord","k","scen","path"].forEach(id=>sel(id).addEventListener("change",()=>{refreshMenus(id!=="mode"&&id!=="ord"?false:true);
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
    tip.textContent=`candidate ${c.b} · ${c.elem} = ${c.v}`;}
  else tip.style.display="none";
});
refreshMenus(true); draw();
</script></body></html>
"""

out = HERE / f"viewer_n{REPLAY_N}.html"
out.write_text(html.replace("%%DATA%%", json.dumps(data, separators=(",", ":"))).replace("%%N%%", str(REPLAY_N)))
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
