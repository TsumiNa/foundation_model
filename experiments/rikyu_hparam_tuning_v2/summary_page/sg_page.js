// Figures for the space-group investigation page. Data: the SG constant prepended by gen_sg_page.py.
(function(){
const C={warm:"var(--warm)",xfer:"var(--xfer)",frz:"var(--frz)",alone:"var(--alone)",ink:"var(--ink)",muted:"var(--muted)",rule:"var(--rule)",soft:"var(--rule-soft)"};
const NS="http://www.w3.org/2000/svg";
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;p.appendChild(e);return e;}
function fig(id,fn){const s=document.getElementById(id);if(!s)return;try{fn(s);}catch(e){console.error("fig "+id,e);el(s,"text",{x:20,y:30,class:"lab"},"figure failed: "+e.message);}}
function grid(s,x0,x1,y0,y1,ticks,fmt,axis){ // horizontal gridlines with labels at y positions
  ticks.forEach(t=>{const y=y1-(t-axis[0])/(axis[1]-axis[0])*(y1-y0);el(s,"line",{x1:x0,x2:x1,y1:y,y2:y,stroke:C.soft});el(s,"text",{x:x0-8,y:y+4,"text-anchor":"end",class:"axis",fill:C.muted},fmt(t));});}
const f3=v=>v.toFixed(3), pct=v=>(v*100).toFixed(1)+"%";

// 1 ladder: from the pipeline replica to the paper's number, one factor at a time
fig("fig-ladder",s=>{
  const rows=SG.ladder, W=960, x0=420, x1=900, top=46, h=34, gap=12;
  const H=top+rows.length*(h+gap)+40; s.setAttribute("height",H);
  const xs=v=>x0+v*(x1-x0);
  [0,.1,.2,.3,.4,.5,.6,.7].forEach(t=>{el(s,"line",{x1:xs(t),x2:xs(t),y1:top-8,y2:H-30,stroke:C.soft});el(s,"text",{x:xs(t),y:H-12,"text-anchor":"middle",class:"axis",fill:C.muted},(t*100).toFixed(0)+"%");});
  el(s,"text",{x:x0,y:22,class:"name",fill:C.muted},"top-1 accuracy on the test split (151 classes)");
  rows.forEach((r,i)=>{const y=top+i*(h+gap);const col=r.kind==="pipe"?C.frz:(r.kind==="ref"?C.alone:C.warm);
    el(s,"text",{x:x0-12,y:y+h/2+5,"text-anchor":"end",class:"name",fill:C.ink},r.label);
    el(s,"rect",{x:x0,y:y+4,width:Math.max(0,xs(r.acc)-x0),height:h-8,rx:3,fill:col,opacity:r.kind==="ref"?.45:.9});
    if(r.sd!=null){el(s,"line",{x1:xs(r.acc-r.sd),x2:xs(r.acc+r.sd),y1:y+h/2,y2:y+h/2,stroke:C.ink,"stroke-width":1.5});}
    el(s,"text",{x:xs(r.acc)+8,y:y+h/2+5,class:"val",fill:C.ink},pct(r.acc)+(r.note?"  "+r.note:""));});
  const yp=top-8, xp=xs(SG.paper_top1);
  el(s,"line",{x1:xp,x2:xp,y1:yp,y2:H-30,stroke:C.xfer,"stroke-dasharray":"5 4","stroke-width":1.5});
  el(s,"text",{x:xp+6,y:H-40,class:"lab",fill:C.xfer},"paper: 60.2% (213 classes)");
});

// 2 top-k recall curves
fig("fig-topk",s=>{
  const W=960,H=360,x0=80,x1=900,y0=62,y1=308; s.setAttribute("height",H);
  const ks=[1,5,10,30,40], xpos=k=>x0+(Math.log(k)/Math.log(40))*(x1-x0), ys=v=>y1-(v-0.2)/(0.8)*(y1-y0);
  grid(s,x0,x1,y0,y1,[.2,.4,.6,.8,1],v=>(v*100).toFixed(0)+"%",[.2,1]);
  ks.forEach(k=>{el(s,"line",{x1:xpos(k),x2:xpos(k),y1:y0,y2:y1,stroke:C.soft});el(s,"text",{x:xpos(k),y:y1+20,"text-anchor":"middle",class:"axis",fill:C.muted},"top-"+k);});
  el(s,"text",{x:(x0+x1)/2,y:H-6,"text-anchor":"middle",class:"name",fill:C.muted},"k · recall = share of test rows whose true space group is among the k most probable predictions");
  SG.topk.forEach(a=>{const pts=a.k.map((k,i)=>[xpos(k),ys(a.rec[i])]);
    el(s,"path",{d:pts.map((p,i)=>(i?"L":"M")+p[0]+","+p[1]).join(" "),fill:"none",stroke:a.color,"stroke-width":2.2,"stroke-dasharray":a.dash||"none"});
    pts.forEach(p=>el(s,"circle",{cx:p[0],cy:p[1],r:4.5,fill:a.color,stroke:"var(--surface)","stroke-width":1.5}));
  });
  SG.topk.forEach((a,i)=>{const lx=x0+(i%3)*270, ly=y0-30+Math.floor(i/3)*18; el(s,"line",{x1:lx,x2:lx+22,y1:ly,y2:ly,stroke:a.color,"stroke-width":2.2,"stroke-dasharray":a.dash||"none"});el(s,"text",{x:lx+28,y:ly+4,class:"axis",fill:C.ink},a.label);});
});

// 3 paired effects per factor
fig("fig-effects",s=>{
  const rows=SG.effects, x0=430,x1=900,top=40,h=30; const H=top+rows.length*h+40; s.setAttribute("height",H);
  const lo=-0.15,hi=0.40, xs=v=>x0+(v-lo)/(hi-lo)*(x1-x0);
  [-.1,0,.1,.2,.3,.4].forEach(t=>{el(s,"line",{x1:xs(t),x2:xs(t),y1:top-10,y2:H-28,stroke:t===0?C.ink:C.soft,"stroke-width":t===0?1.2:1});el(s,"text",{x:xs(t),y:H-10,"text-anchor":"middle",class:"axis",fill:C.muted},(t>0?"+":"")+(t*100).toFixed(0)+" pt");});
  el(s,"text",{x:x0,y:20,class:"name",fill:C.muted},"Δ top-1 accuracy · one dot per matched pair · bar = mean");
  rows.forEach((r,i)=>{const y=top+i*h+h/2;
    el(s,"text",{x:x0-12,y:y+5,"text-anchor":"end",class:"name",fill:C.ink},r.label);
    r.deltas.forEach(d=>el(s,"circle",{cx:xs(d),cy:y,r:6,fill:r.color,opacity:.85,stroke:"var(--surface)","stroke-width":1.2}));
    const m=r.deltas.reduce((a,b)=>a+b,0)/r.deltas.length; el(s,"line",{x1:xs(m),x2:xs(m),y1:y-11,y2:y+11,stroke:C.ink,"stroke-width":2.5});
    el(s,"text",{x:xs(Math.max(...r.deltas))+12,y:y+5,class:"val",fill:C.muted},(m>0?"+":"")+(m*100).toFixed(1)+" pt");});
});

// 4 recall of the twelve largest groups under the three pipeline arms
fig("fig-recall",s=>{
  const g=SG.recall, W=960,H=360,x0=70,x1=930,y0=40,y1=290; s.setAttribute("height",H);
  const n=g.groups.length, bw=(x1-x0)/n, ys=v=>y1-v*(y1-y0);
  grid(s,x0,x1,y0,y1,[0,.25,.5,.75,1],v=>(v*100).toFixed(0)+"%",[0,1]);
  el(s,"text",{x:x0,y:22,class:"name",fill:C.muted},"recall per space group · the twelve largest groups, seed 2025 · label under the bars = rows in the test split");
  const arms=[["balanced_kmd",C.alone,"balanced weights, KMD (as run)"],["plain_kmd",C.frz,"unweighted, KMD"],["plain_classic",C.warm,"unweighted, XenonPy classic"]];
  g.groups.forEach((grp,i)=>{const cx=x0+i*bw+bw/2, w=bw*0.24;
    arms.forEach((a,j)=>{const v=g.rec[a[0]][i];el(s,"rect",{x:cx+(j-1.5)*w+w*0.1,y:ys(v),width:w*0.8,height:y1-ys(v),fill:a[1],rx:2});});
    el(s,"text",{x:cx,y:y1+18,"text-anchor":"middle",class:"name",fill:C.ink},grp);
    el(s,"text",{x:cx,y:y1+34,"text-anchor":"middle",class:"axis",fill:C.muted},g.n_test[i]);});
  arms.forEach((a,j)=>{el(s,"rect",{x:x0+j*290,y:H-18,width:12,height:12,fill:a[1],rx:2});el(s,"text",{x:x0+j*290+18,y:H-7,class:"axis",fill:C.ink},a[2]);});
});

// 5 atoms per cell by space group
fig("fig-atoms",s=>{
  const a=SG.atoms, W=960,H=345,x0=70,x1=930,y0=40,y1=260; s.setAttribute("height",H);
  const bins=a.labels, n=bins.length, bw=(x1-x0)/n, groups=Object.keys(a.hist), cols=[C.warm,C.frz,C.xfer,C.alone];
  const mx=Math.max(...groups.map(g=>Math.max(...a.hist[g])))*1.05, ys=v=>y1-v/mx*(y1-y0);
  grid(s,x0,x1,y0,y1,[0,.2,.4,.6].filter(v=>v<mx),v=>(v*100).toFixed(0)+"%",[0,mx]);
  el(s,"text",{x:x0,y:22,class:"name",fill:C.muted},"share of a space group's training rows, by atoms in the cell of the composition string the model receives");
  bins.forEach((b,i)=>{const cx=x0+i*bw+bw/2, w=bw*0.2;
    groups.forEach((g,j)=>{const v=a.hist[g][i];el(s,"rect",{x:cx+(j-2)*w+w*0.1,y:ys(v),width:w*0.8,height:y1-ys(v),fill:cols[j],rx:2});});
    el(s,"text",{x:cx,y:y1+18,"text-anchor":"middle",class:"axis",fill:C.ink},b);});
  el(s,"text",{x:(x0+x1)/2,y:y1+38,"text-anchor":"middle",class:"name",fill:C.muted},"atoms per cell");
  groups.forEach((g,j)=>{el(s,"rect",{x:x0+j*200,y:H-18,width:12,height:12,fill:cols[j],rx:2});el(s,"text",{x:x0+j*200+18,y:H-7,class:"axis",fill:C.ink},g+" (n = "+a.n[g]+")");});
});

// 6 the balanced weights against class size
fig("fig-weights",s=>{
  const w=SG.weights, W=960,H=300,x0=80,x1=900,y0=30,y1=250; s.setAttribute("height",H);
  const lx=v=>Math.log10(v), xs=v=>x0+(lx(v)-1)/(lx(3500)-1)*(x1-x0), ys=v=>y1-(lx(v)+1.3)/(lx(25)+1.3)*(y1-y0);
  [10,30,100,300,1000,3000].forEach(t=>{el(s,"line",{x1:xs(t),x2:xs(t),y1:y0,y2:y1,stroke:C.soft});el(s,"text",{x:xs(t),y:y1+18,"text-anchor":"middle",class:"axis",fill:C.muted},t.toLocaleString());});
  [0.1,1,10].forEach(t=>{el(s,"line",{x1:x0,x2:x1,y1:ys(t),y2:ys(t),stroke:C.soft});el(s,"text",{x:x0-8,y:ys(t)+4,"text-anchor":"end",class:"axis",fill:C.muted},"×"+t);});
  el(s,"line",{x1:x0,x2:x1,y1:ys(1),y2:ys(1),stroke:C.ink,"stroke-dasharray":"4 4"});
  el(s,"text",{x:x0,y:20,class:"name",fill:C.muted},"weight of one row of each class in the loss (sklearn balanced) against the class size · dashed = unweighted");
  el(s,"text",{x:(x0+x1)/2,y:H-8,"text-anchor":"middle",class:"name",fill:C.muted},"rows in the class (training split, log scale)");
  w.forEach(p=>el(s,"circle",{cx:xs(p[0]),cy:ys(p[1]),r:4,fill:C.xfer,opacity:.75}));
  const big=w.slice().sort((a,b)=>b[0]-a[0])[0], small=w.slice().sort((a,b)=>a[0]-b[0])[0];
  el(s,"text",{x:xs(big[0])-8,y:ys(big[1])-10,"text-anchor":"end",class:"lab",fill:C.ink},"Fm-3m: ×"+big[1].toFixed(2));
  el(s,"text",{x:xs(small[0])+10,y:ys(small[1])+4,class:"lab",fill:C.ink},"a 10-row group: ×"+small[1].toFixed(1));
});
})();
