#!/usr/bin/env python3
"""Build a self-contained interactive HTML viewer for the inverse-design trajectories.

Usage: python build_trajectory_viewer.py [MIRROR_DIR] [REPLAY_N]
       (defaults: <repo>/artifacts/task_scaling, 1000)

Reads every {ws,ft}_o{ord}/k{KK}/inverse/fe_down_diel_up/trajectories/<path>.npz from the mirror,
subsamples the optimisation to ~31 frames, and emits viewer_n<REPLAY_N>.html with dropdowns
(mode / task order / k / path) + a step slider. Each frame shows the 20 candidates in objective
space — x = predicted formation_energy (target −1), y = mean predicted dielectric channel
(target +1) — with hover showing the candidate's current composition (top-4 elements). The red
star marks the joint target. Everything is embedded; no server or network needed.
"""

import json
import sys
from pathlib import Path

import numpy as np

from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

HERE = Path(__file__).resolve().parent
MIRROR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "artifacts/task_scaling"
REPLAY_N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TAG = "" if REPLAY_N == 1000 else f"_n{REPLAY_N}"
SCENARIO = "fe_down_diel_up"
PATHS = ["latent_default", "comp_k4_lowdiv"]
N_FRAMES = 31


def comp_string(w: np.ndarray, top: int = 4) -> str:
    idx = np.argsort(w)[::-1][:top]
    parts = [f"{DEFAULT_ELEMENTS[int(i)]}{w[int(i)]:.2f}" for i in idx if w[int(i)] > 1e-3]
    return " ".join(parts) if parts else "-"


data: dict = {}
n_loaded = 0
for mode in ("ws", "ft"):
    for ord_id in range(3):
        for k in range(1, 22):
            for path in PATHS:
                npz_path = MIRROR / f"{mode}{TAG}_o{ord_id}/k{k:02d}/inverse/{SCENARIO}/trajectories/{path}.npz"
                if not npz_path.exists():
                    continue
                z = np.load(npz_path, allow_pickle=False)
                targets, weights = z["targets"], z["weights"]  # (S,B,4), (S,B,94)
                labels = [str(x) for x in z["labels"]]
                fe_idx = next(i for i, la in enumerate(labels) if la.startswith("formation_energy"))
                diel_idx = [i for i, la in enumerate(labels) if la.startswith("dielectric")]
                steps = targets.shape[0]
                frame_ids = sorted(set(np.linspace(0, steps - 1, N_FRAMES).astype(int).tolist()))
                frames, comps = [], []
                for s in frame_ids:
                    x = targets[s, :, fe_idx]
                    y = targets[s, :, diel_idx].mean(axis=1)
                    frames.append([[round(float(a), 3), round(float(b), 3)] for a, b in zip(x, y)])
                    if weights.shape[0] == steps:
                        comps.append([comp_string(weights[s, b]) for b in range(weights.shape[1])])
                    else:
                        comps.append(["-"] * targets.shape[1])
                data.setdefault(mode, {}).setdefault(str(ord_id), {}).setdefault(str(k), {})[path] = {
                    "steps": [int(s) for s in frame_ids],
                    "frames": frames,
                    "comps": comps,
                }
                n_loaded += 1

print(f"loaded {n_loaded} trajectory files")

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Inverse trajectories — replay n=%%N%%</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;color:#1a1a2e}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
select,button{font-size:14px;padding:2px 6px}
#slider{width:340px}
#tip{position:fixed;background:#1a1a2e;color:#fff;padding:4px 8px;border-radius:4px;font-size:12px;pointer-events:none;display:none}
.legend{color:#6b7280;font-size:12.5px;margin-top:6px}
</style></head><body>
<h3>Inverse-design candidate trajectories — task-scaling, replay n=%%N%% (scenario fe_down_diel_up)</h3>
<div class="controls">
  mode <select id="mode"></select>
  order <select id="ord"></select>
  k <select id="k"></select>
  path <select id="path"></select>
  step <input type="range" id="slider" min="0" value="0">
  <span id="stepLabel"></span>
  <button id="play">▶ play</button>
</div>
<canvas id="cv" width="860" height="600"></canvas>
<div class="legend">x = predicted formation_energy (target −1) · y = mean predicted dielectric channel (target +1) ·
★ = joint target · each dot = one of the 20 seed candidates · hover a dot for its current composition (top-4 elements)</div>
<div id="tip"></div>
<script>
const DATA = %%DATA%%;
const cv = document.getElementById("cv"), ctx = cv.getContext("2d"), tip = document.getElementById("tip");
const sel = id => document.getElementById(id);
let pts = [];
function fill(el, opts){ el.innerHTML = opts.map(o => `<option>${o}</option>`).join(""); }
function keys(o){ return Object.keys(o).sort((a,b)=>(+a||a) > (+b||b) ? 1 : -1); }
function cur(){ const m=sel("mode").value, o=sel("ord").value, k=sel("k").value, p=sel("path").value;
  return ((DATA[m]||{})[o]||{})[k] ? DATA[m][o][k][p] : null; }
function refreshMenus(){
  fill(sel("mode"), keys(DATA));
  const m = sel("mode").value; fill(sel("ord"), keys(DATA[m]));
  const o = sel("ord").value; fill(sel("k"), keys(DATA[m][o]).sort((a,b)=>+a-+b));
  const k = sel("k").value; fill(sel("path"), keys(DATA[m][o][k]));
}
function draw(){
  const d = cur(); if(!d){ ctx.clearRect(0,0,cv.width,cv.height); return; }
  const f = Math.min(+sel("slider").value, d.frames.length-1);
  sel("slider").max = d.frames.length-1;
  sel("stepLabel").textContent = `optimisation step ${d.steps[f]}`;
  // global bounds over ALL frames of this selection (stable axes while sliding)
  let xs=[], ys=[];
  d.frames.flat().forEach(([x,y])=>{xs.push(x);ys.push(y);});
  xs.push(-1); ys.push(1);
  const pad=0.15, x0=Math.min(...xs)-pad, x1=Math.max(...xs)+pad, y0=Math.min(...ys)-pad, y1=Math.max(...ys)+pad;
  const L=60,R=20,T=16,B=46, W=cv.width-L-R, H=cv.height-T-B;
  const X=v=>L+(v-x0)/(x1-x0)*W, Y=v=>T+H-(v-y0)/(y1-y0)*H;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.strokeStyle="#e5e7eb"; ctx.fillStyle="#6b7280"; ctx.font="11px sans-serif";
  for(let i=0;i<=6;i++){
    const vx=x0+(x1-x0)*i/6, vy=y0+(y1-y0)*i/6;
    ctx.beginPath(); ctx.moveTo(X(vx),T); ctx.lineTo(X(vx),T+H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(L,Y(vy)); ctx.lineTo(L+W,Y(vy)); ctx.stroke();
    ctx.fillText(vx.toFixed(2), X(vx)-12, T+H+16);
    ctx.fillText(vy.toFixed(2), 8, Y(vy)+4);
  }
  ctx.fillText("predicted formation_energy", L+W/2-70, cv.height-8);
  ctx.save(); ctx.translate(14,T+H/2+60); ctx.rotate(-Math.PI/2); ctx.fillText("mean predicted dielectric", 0,0); ctx.restore();
  // target star
  ctx.fillStyle="#C44E52"; ctx.font="20px sans-serif"; ctx.fillText("★", X(-1)-7, Y(1)+7);
  // faint step-0 ghosts
  ctx.fillStyle="rgba(107,114,128,0.30)";
  d.frames[0].forEach(([x,y])=>{ctx.beginPath();ctx.arc(X(x),Y(y),3,0,7);ctx.fill();});
  // current frame
  pts=[];
  ctx.fillStyle="#0077BB";
  d.frames[f].forEach(([x,y],b)=>{ctx.beginPath();ctx.arc(X(x),Y(y),4.5,0,7);ctx.fill();
    pts.push({px:X(x),py:Y(y),x,y,comp:d.comps[f][b],b});});
}
["mode","ord","k","path"].forEach((id,i)=>sel(id).addEventListener("change",()=>{
  if(id==="mode"||id==="ord"||id==="k"){ const m=sel("mode").value;
    if(id==="mode") fill(sel("ord"), keys(DATA[m]));
    const o=sel("ord").value;
    if(id!=="k") fill(sel("k"), keys(DATA[m][o]).sort((a,b)=>+a-+b));
    const k=sel("k").value; fill(sel("path"), keys(DATA[m][o][k]));
  }
  draw();
}));
sel("slider").addEventListener("input", draw);
let timer=null;
sel("play").addEventListener("click",()=>{
  if(timer){clearInterval(timer);timer=null;sel("play").textContent="▶ play";return;}
  sel("play").textContent="⏸ pause";
  timer=setInterval(()=>{const s=sel("slider");s.value=(+s.value+1)%(+s.max+1);draw();},150);
});
cv.addEventListener("mousemove",e=>{
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  let best=null,bd=144;
  pts.forEach(p=>{const d2=(p.px-mx)**2+(p.py-my)**2; if(d2<bd){bd=d2;best=p;}});
  if(best){tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";
    tip.textContent=`seed ${best.b}: ${best.comp}  (FE ${best.x}, diel ${best.y})`;}
  else tip.style.display="none";
});
refreshMenus(); draw();
</script></body></html>
"""

out = HERE / f"viewer_n{REPLAY_N}.html"
out.write_text(html.replace("%%DATA%%", json.dumps(data, separators=(",", ":"))).replace("%%N%%", str(REPLAY_N)))
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
