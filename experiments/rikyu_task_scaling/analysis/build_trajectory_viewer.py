#!/usr/bin/env python3
"""Build a self-contained interactive HTML viewer for the inverse-design candidate compositions.

Usage: python build_trajectory_viewer.py [MIRROR_DIR] [REPLAY_N]
       (defaults: <repo>/artifacts/task_scaling, 1000)

Layout: two columns.
  LEFT  — elements × 20-candidates heatmap on top (animated over the optimisation); below it the
          selected candidate's per-target progress curves (3/4 width) and its current composition
          as weight-sorted bars (1/4 width).
  RIGHT — every candidate's seed composition → final optimised composition (click to select;
          hover for true/predicted values), with usage + experiment notes underneath.

Animation runs at one frame per 5 optimisation steps; a numeric box jumps straight to any step.
The data payload is gzip-compressed and base64-embedded (decoded with the browser's native
DecompressionStream), so the file stays small despite the 5-step frame rate. Canvases are
devicePixelRatio-scaled (crisp on retina). Emits viewer_n<REPLAY_N>.html.
"""

import base64
import gzip
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
FRAME_STEP = 5
MAX_ROWS = 12
TARGET_VALUES = {"formation_energy": -1.0, "dielectric_total": 1.0, "dielectric_ionic": 1.0, "dielectric_electronic": 1.0}
TRUTH_COLS = {
    "formation_energy": "Formation energy per atom (normalized)",
    "dielectric_total": "Dielectric total (normalized)",
    "dielectric_ionic": "Dielectric ionic (normalized)",
    "dielectric_electronic": "Dielectric electronic (normalized)",
}
_qc = pd.read_parquet(
    HERE.parents[2] / "data/qc_ac_te_mp_dos_reformat_20260515.pd.parquet", columns=["composition", *TRUTH_COLS.values()]
).set_index("composition")


def truths(comp: str, tasks: list[str]) -> list[float | None]:
    if comp not in _qc.index:
        return [None] * len(tasks)
    row = _qc.loc[comp]
    return [None if pd.isna(row[TRUTH_COLS[t]]) else round(float(row[TRUTH_COLS[t]]), 2) for t in tasks]


def top_pairs(w: np.ndarray, top: int = 4) -> list[list[int]]:
    idx = np.argsort(w)[::-1][:top]
    return [[int(i), int(round(float(w[int(i)]) * 1000))] for i in idx if w[int(i)] > 1e-3]


def build_entry(targets: np.ndarray, weights: np.ndarray, labels: list[str], seed_comps: list[str]) -> dict:
    steps, n_seeds, _ = targets.shape
    tasks = [la.split("→")[0].split("~")[0] for la in labels]
    tvals = np.array([TARGET_VALUES[t] for t in tasks])
    frame_ids = sorted(set(list(range(0, steps, FRAME_STEP)) + [steps - 1]))

    y0 = targets[0]
    denom = tvals[None, :] - y0
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    prog = np.transpose((targets[np.array(frame_ids)] - y0[None]) / denom[None], (1, 0, 2))  # (B,F,T)

    best = int(np.argmin(np.abs(targets[-1] - tvals[None, :]).sum(axis=1)))

    have_w = weights.shape[0] == steps
    if have_w:
        final = weights[-1]
        union: set[int] = set()
        for b in range(n_seeds):
            union.update(int(i) for i in np.argsort(final[b])[::-1][:4] if final[b, int(i)] > 1e-3)
        rows = sorted(union, key=lambda i: -float(final[:, i].mean()))[:MAX_ROWS]
        hm = np.round(weights[np.array(frame_ids)][:, :, rows] * 1000).astype(int)  # (F,B,R) permille
        systems = {tuple(sorted(np.argsort(final[b])[::-1][:3].tolist())) for b in range(n_seeds)}
        l1 = (
            float(np.mean([np.abs(final[a] - final[c]).sum() for a, c in combinations(range(n_seeds), 2)]))
            if n_seeds > 1
            else 0.0
        )
        div = {"systems": len(systems), "l1": round(l1, 2)}
        so = []
        for b in range(n_seeds):
            comp = seed_comps[b] if b < len(seed_comps) else ""
            so.append(
                {
                    "s": top_pairs(weights[0, b]),
                    "f": top_pairs(final[b]),
                    "tv": truths(comp, tasks),
                    "sp": [round(float(x), 2) for x in targets[0, b]],
                    "fp": [round(float(x), 2) for x in targets[-1, b]],
                }
            )
        hm_rows, hm_list = rows, hm.tolist()
    else:
        hm_rows, hm_list, div, so = [], [], {"systems": 0, "l1": 0.0}, []

    return {
        "tasks": tasks,
        "steps": [int(s) for s in frame_ids],  # shared frame grid for curves + heatmap + bars
        "prog": np.round(prog, 2).tolist(),
        "rows": hm_rows,
        "hm": hm_list,
        "so": so,
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

payload = {"data": data, "elems": list(DEFAULT_ELEMENTS)}
b64 = base64.b64encode(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), 6)).decode()

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Inverse candidates — replay n=%%N%%</title>
<style>
body{font-family:system-ui,sans-serif;margin:14px;color:#1a1a2e}
h3{margin:0 0 8px 0}
.controls{display:flex;gap:11px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
select,button,input[type=number]{font-size:13.5px;padding:2px 6px}
#slider{width:230px}
#jump{width:64px}
#divline{color:#374151;font-size:12.5px;margin:2px 0 8px 0}
.cols{display:flex;gap:14px;align-items:flex-start}
.left{flex:0 0 66%;max-width:66%}
.lrow{display:flex;gap:8px;margin-top:6px}
.right{flex:1 1 auto;min-width:0}
#solist{font:11.5px ui-monospace,Menlo,monospace;line-height:1.7;max-height:392px;overflow-y:auto;
  border:1px solid #e5e7eb;border-radius:6px;padding:7px 9px}
#solist .row{cursor:pointer;white-space:nowrap}
#solist .row:hover{background:#f3f4f6}
#solist .sel{background:#fdeaea}
#info{border:1px solid #e5e7eb;border-radius:6px;padding:8px 11px;margin-top:10px;color:#374151;font-size:12.5px;line-height:1.55}
#info b{color:#1a1a2e}
#tip{position:fixed;background:#1a1a2e;color:#fff;padding:3px 7px;border-radius:4px;font-size:12px;pointer-events:none;display:none;white-space:pre}
#loading{color:#6b7280;font-size:14px;margin:30px}
</style></head><body>
<h3>Inverse-design candidates — replay n=%%N%%</h3>
<div id="loading">decompressing data…</div>
<div id="app" style="display:none">
<div class="controls">
  mode <select id="mode"></select> order <select id="ord"></select> k <select id="k"></select>
  target <select id="scen"></select> path <select id="path"></select>
  <input type="range" id="slider" min="0" value="0">
  <span id="stepLabel"></span>
  step <input type="number" id="jump" min="1" step="1"> <button id="go">jump</button>
  <button id="play">▶ play</button>
  <button id="dlcur" title="Download the current selection as JSON">⬇ selection</button>
  <button id="dlall" title="Download the whole dataset as JSON">⬇ all</button>
</div>
<div id="divline"></div>
<div class="cols">
  <div class="left">
    <canvas id="hm"></canvas>
    <div class="lrow"><canvas id="line"></canvas><canvas id="bar"></canvas></div>
  </div>
  <div class="right">
    <div id="solist"></div>
    <div id="info">
      <b>How to use</b><br>
      · Dropdowns: mode (ws = full-model warm-start / ft = frozen-encoder finetune) × task order × k (number of
        pretraining tasks) × target scenario × optimisation path<br>
      · Slider or "step + jump" moves to any optimisation step (animation: 1 frame per 5 steps); ▶ play animates<br>
      · Click a heatmap column or a list row to select a candidate; hover heatmap cells for weights,
        hover list rows for values<br>
      · List-row tooltip: seed TRUE values (dataset labels, – = unlabeled) / seed predicted / final predicted<br>
      · ⬇ buttons download the raw data as JSON (current selection or the whole dataset)<br><br>
      <b>Experiment</b><br>
      · Objective: formation_energy → −1σ AND the selected dielectric task → +1σ (dielectric weight 2.0), z-scored units<br>
      · Seeds: weighted_random — 20 test compositions sampled by rank of TRUE dielectric_total, element systems deduplicated<br>
      · Paths: latent = latent-space optimisation with AE alignment, KMD-decoded; comp_k4 = direct composition
        optimisation (≤4 elements, diversity 0.5)<br>
      · 300 optimisation steps; progress axis: 0 = seed level, 1 = target reached (dashed line)
    </div>
  </div>
</div>
</div>
<div id="tip"></div>
<script>
const B64 = "%%B64%%";
async function decode(){
  const bin = Uint8Array.from(atob(B64), c=>c.charCodeAt(0));
  const ds = new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip")));
  return JSON.parse(await ds.text());
}
let DATA=null, ELEMS=null;
const COLORS=["#2563EB","#55A868","#E67E22","#9467bd"];
const DPR=window.devicePixelRatio||1;
const sel=id=>document.getElementById(id);
let HMW=760,HMH=300,LNW=560,LNH=390,BRW=192,BRH=390;
let ctx={}, tip=null, seed=0, hmCells=[];
function sizeCanvases(){
  const lw = document.querySelector(".left").clientWidth;
  HMW = lw; LNW = Math.floor(lw*0.75)-8; BRW = lw-LNW-8;
  ctx.hm=setup(sel("hm"),HMW,HMH); ctx.line=setup(sel("line"),LNW,LNH); ctx.bar=setup(sel("bar"),BRW,BRH);
}
function setup(c,w,h){c.style.width=w+"px";c.style.height=h+"px";c.width=Math.round(w*DPR);c.height=Math.round(h*DPR);
  const x=c.getContext("2d");x.setTransform(DPR,0,0,DPR,0,0);return x;}
const fmt=p=>p.map(([e,pm])=>ELEMS[e]+(pm/1000).toFixed(2)).join(" ")||"-";
function fill(el,opts,keep){const v=el.value;el.innerHTML=opts.map(o=>`<option>${o}</option>`).join("");
  if(keep&&opts.includes(v))el.value=v;}
const keys=o=>Object.keys(o);
function cur(){const m=sel("mode").value,o=sel("ord").value,k=sel("k").value,s=sel("scen").value,p=sel("path").value;
  return ((((DATA[m]||{})[o]||{})[k]||{})[s]||{})[p]||null;}
function refreshMenus(resetSeed){
  fill(sel("mode"),keys(DATA),true);
  const m=sel("mode").value;fill(sel("ord"),keys(DATA[m]),true);
  const o=sel("ord").value;fill(sel("k"),keys(DATA[m][o]).sort((a,b)=>+a-+b),true);
  const k=sel("k").value;fill(sel("scen"),keys(DATA[m][o][k]),true);
  const s=sel("scen").value;fill(sel("path"),keys(DATA[m][o][k][s]),true);
  const d=cur();if(d&&resetSeed)seed=d.best;
}
function draw(){
  const d=cur();if(!d)return;
  const F=d.steps.length,f=Math.min(+sel("slider").value,F-1);
  sel("slider").max=F-1;
  const step=d.steps[f],total=d.steps[F-1]+1;
  sel("stepLabel").textContent=`step ${step+1}/${total}`;
  sel("jump").max=total;
  sel("divline").textContent=`final-step diversity of the 20 candidates: ${d.div.systems} distinct top-3 element systems · mean pairwise L1 = ${d.div.l1}  (0 = identical, 2 = no shared elements)`;

  // ===== heatmap (left top) =====
  const H=ctx.hm,rows=d.rows.length,B=d.prog.length;
  H.clearRect(0,0,HMW,HMH);hmCells=[];
  if(rows){
    const grid=d.hm[f];
    const L=44,T=26,cw=(HMW-L-12)/B,chh=(HMH-T-20)/rows;
    let maxw=1;grid.forEach(r=>r.forEach(v=>{if(v>maxw)maxw=v;}));
    H.font="10.5px sans-serif";
    for(let r=0;r<rows;r++){
      H.fillStyle="#374151";H.fillText(ELEMS[d.rows[r]],12,T+r*chh+chh*0.65);
      for(let b=0;b<B;b++){
        H.fillStyle=`rgba(37,99,235,${Math.min(1,grid[b][r]/maxw).toFixed(2)})`;
        H.fillRect(L+b*cw+1,T+r*chh+1,cw-2,chh-2);
        hmCells.push({x:L+b*cw,y:T+r*chh,w:cw,h:chh,b,elem:ELEMS[d.rows[r]],v:grid[b][r]/1000});
      }
    }
    for(let b=0;b<B;b++){
      H.fillStyle=b===seed?"#C44E52":"#6b7280";
      H.fillText((b===d.best?"▲":"")+b,L+b*cw+cw/2-8,16);
      if(b===seed){H.strokeStyle="#C44E52";H.lineWidth=1.5;H.strokeRect(L+b*cw,T,cw,rows*chh);}
    }
    H.fillStyle="#6b7280";H.fillText("candidate (▲ = best final; click a column to inspect)",L,HMH-5);
  }

  // ===== progress curves (left bottom, 3/4) =====
  const X2=ctx.line,P=d.prog[seed];
  let vals=[];P.forEach(row=>row.forEach(v=>vals.push(v)));vals.push(0,1.05);
  const y0=Math.min(...vals)-0.05,y1=Math.max(...vals)+0.05;
  const L=46,R=10,T=20,Bm=40,W=LNW-L-R,Hh=LNH-T-Bm;
  const px=i=>L+d.steps[i]/(total-1)*W,py=v=>T+Hh-(v-y0)/(y1-y0)*Hh;
  X2.clearRect(0,0,LNW,LNH);
  X2.strokeStyle="#e5e7eb";X2.fillStyle="#6b7280";X2.font="10.5px sans-serif";
  for(let i=0;i<=5;i++){const vy=y0+(y1-y0)*i/5;
    X2.beginPath();X2.moveTo(L,py(vy));X2.lineTo(L+W,py(vy));X2.stroke();
    X2.fillText(vy.toFixed(1),12,py(vy)+4);}
  X2.strokeStyle="#666";X2.setLineDash([5,4]);
  X2.beginPath();X2.moveTo(L,py(1));X2.lineTo(L+W,py(1));X2.stroke();X2.setLineDash([]);
  d.tasks.forEach((t,c)=>{
    X2.strokeStyle=COLORS[c%COLORS.length];X2.lineWidth=2;X2.beginPath();
    P.forEach((row,i)=>{i?X2.lineTo(px(i),py(row[c])):X2.moveTo(px(i),py(row[c]));});
    X2.stroke();
    X2.fillStyle=COLORS[c%COLORS.length];X2.fillRect(L+8,T+4+c*14,10,3);
    X2.fillText(t,L+22,T+9+c*14);
  });
  X2.strokeStyle="#444";X2.lineWidth=1.4;
  X2.beginPath();X2.moveTo(px(f),T);X2.lineTo(px(f),T+Hh);X2.stroke();
  X2.fillStyle="#6b7280";
  X2.fillText(`candidate ${seed}${seed===d.best?" (best)":""} — progress (0 = seed, 1 = target)`,L+W/2-120,12);
  X2.fillText("Optimisation step",L+W/2-42,LNH-6);
  for(let i=0;i<=4;i++){const s0=Math.round((total-1)*i/4);
    X2.fillText(String(s0),L+s0/(total-1)*W-8,T+Hh+15);}

  // ===== composition bars (left bottom, 1/4) — sorted by weight =====
  const X3=ctx.bar;
  X3.clearRect(0,0,BRW,BRH);
  if(rows){
    const pairs=d.rows.map((ei,r)=>[ei,d.hm[f][seed][r]/1000]).filter(p=>p[1]>0.001)
                      .sort((a,b)=>b[1]-a[1]);
    X3.fillStyle="#1a1a2e";X3.font="bold 11px sans-serif";
    X3.fillText(`candidate ${seed}`,8,14);
    X3.font="10.5px sans-serif";X3.fillStyle="#6b7280";
    X3.fillText(`composition @ step ${step+1}`,8,28);
    const bh=Math.min(22,(BRH-70)/Math.max(pairs.length,1)),maxw=Math.max(0.4,...pairs.map(p=>p[1]));
    pairs.forEach(([ei,v],i)=>{
      const y=40+i*(bh+5);
      X3.fillStyle="#6b7280";X3.font="11px sans-serif";X3.fillText(ELEMS[ei],6,y+bh*0.7);
      X3.fillStyle="#2563EB";X3.fillRect(30,y,v/maxw*(BRW-84),bh);
      X3.fillStyle="#374151";X3.fillText(v.toFixed(2),34+v/maxw*(BRW-84),y+bh*0.7);
    });
  }

  // ===== seeds -> optimized (right) =====
  const so=d.so||[];
  const vv=v=>v==null?"–":v.toFixed(2);
  const pair=arr=>d.tasks.map((t,i)=>`${t.replace("dielectric_","diel_").replace("formation_energy","FE")} ${vv(arr?arr[i]:null)}`).join(", ");
  sel("solist").innerHTML="<b>seed composition → final optimised</b> <span style='color:#6b7280'>(hover for values, click to select)</span><br>"+so.map((r,b)=>
    `<div class="row ${b===seed?'sel':''}" data-b="${b}" title="seed TRUE: ${pair(r.tv)}\\nseed predicted: ${pair(r.sp)}\\nfinal predicted: ${pair(r.fp)}">${b===d.best?"▲":"&nbsp;"}${String(b).padStart(2)} ${fmt(r.s)} → ${fmt(r.f)}</div>`
  ).join("");
  document.querySelectorAll("#solist .row").forEach(el=>el.addEventListener("click",()=>{seed=+el.dataset.b;draw();}));
}
function jumpToStep(raw){
  const d=cur();if(!d)return;
  const target=Math.max(1,Math.min(+raw||1,d.steps[d.steps.length-1]+1))-1;
  let bi=0,bd=1e9;
  d.steps.forEach((s,i)=>{const dd=Math.abs(s-target);if(dd<bd){bd=dd;bi=i;}});
  sel("slider").value=bi;draw();
}
async function main(){
  const p=await decode();DATA=p.data;ELEMS=p.elems;
  document.getElementById("loading").style.display="none";
  document.getElementById("app").style.display="block";
  sizeCanvases();
  window.addEventListener("resize",()=>{sizeCanvases();draw();});
  tip=sel("tip");
  ["mode","ord","k","scen","path"].forEach(id=>sel(id).addEventListener("change",()=>{
    refreshMenus(id==="mode"||id==="ord");
    const d=cur();if(d&&seed>=d.prog.length)seed=d.best;draw();}));
  sel("slider").addEventListener("input",draw);
  function dl(obj,name){
    const a=document.createElement("a");
    a.href=URL.createObjectURL(new Blob([JSON.stringify(obj,null,1)],{type:"application/json"}));
    a.download=name;a.click();URL.revokeObjectURL(a.href);
  }
  sel("dlcur").addEventListener("click",()=>{
    const d=cur();if(!d)return;
    const m=sel("mode").value,o=sel("ord").value,k=sel("k").value,s=sel("scen").value,pp=sel("path").value;
    dl({selection:{mode:m,order:+o,k:+k,scenario:s,path:pp,replay_n:%%N%%},
        element_table:ELEMS,notes:"prog: (candidate,frame,task) progress 0=seed 1=target; hm: (frame,candidate,row) element weight in permille; rows: element indices; so: per-candidate seed/final top-4 [elem,permille] + tv/sp/fp = seed-true/seed-pred/final-pred per task",
        ...d}, `inverse_n%%N%%_${m}_o${o}_k${k}_${s}_${pp}.json`);
  });
  sel("dlall").addEventListener("click",()=>dl({replay_n:%%N%%,element_table:ELEMS,data:DATA},"inverse_n%%N%%_all.json"));
  sel("go").addEventListener("click",()=>jumpToStep(sel("jump").value));
  sel("jump").addEventListener("keydown",e=>{if(e.key==="Enter")jumpToStep(sel("jump").value);});
  let timer=null;
  sel("play").addEventListener("click",()=>{
    if(timer){clearInterval(timer);timer=null;sel("play").textContent="▶ play";return;}
    sel("play").textContent="⏸ pause";
    timer=setInterval(()=>{const s=sel("slider");s.value=(+s.value+1)%(+s.max+1);draw();},120);
  });
  const hm=sel("hm");
  hm.addEventListener("click",e=>{
    const r=hm.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const c=hmCells.find(c=>mx>=c.x&&mx<c.x+c.w&&my>=c.y&&my<c.y+c.h);
    if(c){seed=c.b;draw();}
  });
  hm.addEventListener("mousemove",e=>{
    const r=hm.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const c=hmCells.find(c=>mx>=c.x&&mx<c.x+c.w&&my>=c.y&&my<c.y+c.h);
    if(c){tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";
      tip.textContent=`candidate ${c.b} · ${c.elem} = ${c.v.toFixed(3)}`;}
    else tip.style.display="none";
  });
  refreshMenus(true);draw();
}
if(typeof DecompressionStream==="undefined"){
  document.getElementById("loading").textContent="This viewer needs a browser with native DecompressionStream (Chrome 80+/Safari 16.4+/Firefox 113+).";
}else{main();}
</script></body></html>
"""

out = HERE / f"viewer_n{REPLAY_N}.html"
out.write_text(html.replace("%%B64%%", b64).replace("%%N%%", str(REPLAY_N)))
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, payload gz {len(b64) * 3 / 4 / 1e6:.1f} MB)")
