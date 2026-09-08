// Figures for the "two ceilings" page. D is injected by gen_labels_page.py from summary/mp_labels_page_data.json.
(function(){
const NS="http://www.w3.org/2000/svg";
const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const med=a=>{const s=a.slice().sort((x,y)=>x-y);const n=s.length;return n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2;};
const txt=(svg,x,y,s,cls,attrs)=>{const t=el("text",Object.assign({x,y,class:cls||"axis"},attrs||{}));t.textContent=s;svg.appendChild(t);return t;};
const size=(svg,W,H)=>{svg.setAttribute("width",W);svg.setAttribute("height",H);};

function draw(){
  const C={warm:css("--warm"),xfer:css("--xfer"),frz:css("--frz"),alone:css("--alone")};
  const cRule=css("--rule"),cSoft=css("--rule-soft"),cInk=css("--ink"),cMuted=css("--muted"),cSurf=css("--surface");

  // ---- fig-ceil: the six MP single-task ceilings ----
  try{(function(){const svg=document.getElementById("fig-ceil");if(!svg)return;svg.textContent="";
    const rows=D.ceilings; const W=960,rowH=40,m={t:36,r:120,b:40,l:190}; const H=m.t+rows.length*rowH+m.b; size(svg,W,H); const iw=W-m.l-m.r;
    const X=v=>m.l+v*iw;
    for(let v=0;v<=1.001;v+=0.2){svg.appendChild(el("line",{x1:X(v),x2:X(v),y1:m.t-8,y2:m.t+rows.length*rowH,stroke:v===0?cRule:cSoft}));txt(svg,X(v),m.t-14,v.toFixed(1),"axis",{"text-anchor":"middle"});}
    rows.forEach(([task,n,r2,sd],i)=>{const y=m.t+i*rowH+8;const bad=(task==="final_energy"||task==="volume");const col=bad?C.xfer:C.warm;
      svg.appendChild(el("rect",{x:X(0),y,width:X(r2)-X(0),height:rowH-16,rx:3,fill:col,opacity:bad?0.9:0.75}));
      svg.appendChild(el("line",{x1:X(r2-sd),x2:X(r2+sd),y1:y+(rowH-16)/2,y2:y+(rowH-16)/2,stroke:cInk,"stroke-width":1.5}));
      txt(svg,m.l-12,y+(rowH-16)/2+5,task.replace(/_/g," "),"name",{"text-anchor":"end",fill:cInk});
      txt(svg,X(r2)+10,y+(rowH-16)/2+5,`${r2.toFixed(4)} ± ${sd.toFixed(4)}`,"val",{fill:col});});
    txt(svg,m.l+iw/2,H-8,"single-task R², KMD descriptor, 5 seeds, 2026-05-15 labels","axis",{"text-anchor":"middle"});
  })();}catch(e){console.error("fig-ceil failed",e);}

  // ---- fig-probe: six known materials, energy per atom under each scheme ----
  try{(function(){const svg=document.getElementById("fig-probe");if(!svg)return;svg.textContent="";
    const rows=D.probe; const W=960,rowH=46,m={t:40,r:30,b:44,l:150}; const H=m.t+rows.length*rowH+m.b; size(svg,W,H); const iw=W-m.l-m.r;
    const lo=-90,hi=0; const X=v=>m.l+(v-lo)/(hi-lo)*iw;
    for(let v=-90;v<=0.01;v+=10){svg.appendChild(el("line",{x1:X(v),x2:X(v),y1:m.t-8,y2:m.t+rows.length*rowH,stroke:v===0?cRule:cSoft}));txt(svg,X(v),m.t-14,v,"axis",{"text-anchor":"middle"});}
    rows.forEach(([label,f,gga,r2s,mixed,old,neu],i)=>{const y=m.t+i*rowH+rowH/2;
      txt(svg,m.l-12,y+5,label,"name",{"text-anchor":"end",fill:cInk});
      svg.appendChild(el("line",{x1:X(gga),x2:X(r2s),y1:y,y2:y,stroke:cMuted,opacity:.4}));
      svg.appendChild(el("rect",{x:X(old)-7,y:y-7,width:14,height:14,fill:C.xfer,opacity:.9}));           // dataset 2026-05-15 (copied the mixed scheme)
      svg.appendChild(el("circle",{cx:X(r2s),cy:y,r:6,fill:cSurf,stroke:C.frz,"stroke-width":2}));      // r2SCAN
      svg.appendChild(el("circle",{cx:X(gga),cy:y,r:6.5,fill:C.warm}));                                  // GGA / GGA+U
      svg.appendChild(el("circle",{cx:X(neu),cy:y+0,r:10,fill:"none",stroke:C.warm,"stroke-width":1.5,"stroke-dasharray":"3 2"})); // rebuilt dataset
      txt(svg,X(gga)+14,y-8,gga.toFixed(2),"val",{fill:C.warm}); txt(svg,X(old)-12,y+18,old.toFixed(2),"val",{fill:C.xfer,"text-anchor":"end"});});
    txt(svg,m.l+iw/2,H-8,"energy per atom (eV)","axis",{"text-anchor":"middle"});
  })();}catch(e){console.error("fig-probe failed",e);}

  // ---- fig-mixed: mixed-scheme minus GGA energy over every material ----
  try{(function(){const svg=document.getElementById("fig-mixed");if(!svg)return;svg.textContent="";
    const h=D.local.mixed_minus_gga.hist; const labels=["−90..−60","−60..−40","−40..−30","−30..−20","−20..−10","−10..−5","−5..−1","−1..0","= 0"];
    const W=960,H=300,m={t:40,r:30,b:60,l:70}; size(svg,W,H); const iw=W-m.l-m.r,ih=H-m.t-m.b;
    const mx=Math.log10(Math.max(...h.counts)+1); const Y=c=>m.t+ih-Math.log10(c+1)/mx*ih; const bw=iw/h.counts.length;
    [1,10,100,1000,10000].forEach(v=>{if(Math.log10(v+1)<=mx){svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:cSoft}));txt(svg,m.l-10,Y(v)+4,v.toLocaleString(),"axis",{"text-anchor":"end"});}});
    h.counts.forEach((c,i)=>{const x=m.l+i*bw+6;const last=i===h.counts.length-1;svg.appendChild(el("rect",{x,y:Y(c),width:bw-12,height:m.t+ih-Y(c),rx:3,fill:last?C.warm:C.xfer,opacity:.85}));
      txt(svg,x+(bw-12)/2,Y(c)-6,c.toLocaleString(),"val",{"text-anchor":"middle",fill:last?C.warm:C.xfer}); txt(svg,x+(bw-12)/2,m.t+ih+18,labels[i],"axis",{"text-anchor":"middle"});});
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,stroke:cRule}));
    txt(svg,m.l+iw/2,H-12,"mixed-scheme energy − GGA/GGA+U energy (eV per atom), 33,159 materials with both · log count","axis",{"text-anchor":"middle"});
  })();}catch(e){console.error("fig-mixed failed",e);}

  // ---- fig-fehist: the label before and after ----
  try{(function(){const svg=document.getElementById("fig-fehist");if(!svg)return;svg.textContent="";
    const a=D.local.fe_old_hist,b=D.local.fe_new_hist; const W=960,H=300,m={t:40,r:30,b:56,l:70}; size(svg,W,H); const iw=W-m.l-m.r,ih=H-m.t-m.b;
    const n=a.counts.length; const mx=Math.log10(Math.max(...a.counts,...b.counts)+1); const Y=c=>m.t+ih-Math.log10(c+1)/mx*ih; const bw=iw/n;
    [1,10,100,1000,10000].forEach(v=>{if(Math.log10(v+1)<=mx){svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:cSoft}));txt(svg,m.l-10,Y(v)+4,v.toLocaleString(),"axis",{"text-anchor":"end"});}});
    for(let i=0;i<n;i++){const x=m.l+i*bw;
      svg.appendChild(el("rect",{x:x+3,y:Y(a.counts[i]),width:bw/2-4,height:m.t+ih-Y(a.counts[i]),fill:C.xfer,opacity:.85,rx:2}));
      svg.appendChild(el("rect",{x:x+bw/2+1,y:Y(b.counts[i]),width:bw/2-4,height:m.t+ih-Y(b.counts[i]),fill:C.warm,opacity:.85,rx:2}));
      if(i%3===0)txt(svg,x,m.t+ih+18,a.edges[i],"axis",{"text-anchor":"middle"});}
    txt(svg,m.l+iw,m.t+ih+18,"0","axis",{"text-anchor":"middle"});
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,stroke:cRule}));
    txt(svg,m.l+iw/2,H-12,"Final energy per atom (eV), 5-eV bins · log count · 33,166 rows with both labels","axis",{"text-anchor":"middle"});
  })();}catch(e){console.error("fig-fehist failed",e);}

  // ---- strips: every run as a dot, the median as a bar ----
  function strips(svg,groups,title,ylabel,opts){svg.textContent=""; opts=opts||{};
    const W=960,H=opts.H||300,m={t:52,r:20,b:70,l:70}; size(svg,W,H); const iw=W-m.l-m.r,ih=H-m.t-m.b;
    const all=groups.flatMap(g=>g.vals); let lo=opts.lo!=null?opts.lo:Math.min(...all),hi=opts.hi!=null?opts.hi:Math.max(...all); const pad=(hi-lo)*.12||.02; if(opts.lo==null)lo-=pad; if(opts.hi==null)hi+=pad;
    const Y=v=>m.t+ih-(Math.max(v,lo)-lo)/(hi-lo)*ih;
    for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4;svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:cSoft}));txt(svg,m.l-10,Y(v)+4,v.toFixed(3),"axis",{"text-anchor":"end"});}
    const colW=iw/groups.length;
    groups.forEach((g,i)=>{const cx=m.l+colW*i+colW/2;
      g.vals.forEach((v,q)=>{const jit=((q*37)%11-5)*2.4;const below=v<lo; svg.appendChild(el("circle",{cx:cx+jit,cy:Y(v),r:5,fill:g.col,opacity:below?.35:.75}));});
      const mv=med(g.vals); svg.appendChild(el("line",{x1:cx-30,x2:cx+30,y1:Y(mv),y2:Y(mv),stroke:g.col,"stroke-width":3}));
      txt(svg,cx,m.t+ih+22,g.name,"lab",{"text-anchor":"middle",fill:g.col}); txt(svg,cx,m.t+ih+40,`n = ${g.vals.length}`,"axis",{"text-anchor":"middle"}); txt(svg,cx,m.t+ih+56,`median ${mv.toFixed(4)}`,"axis",{"text-anchor":"middle"});
      if(g.sub)txt(svg,cx,m.t-16,g.sub,"val",{"text-anchor":"middle",fill:g.col});});
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,stroke:cRule}));
    txt(svg,m.l,m.t-34,title,"axis");}
  try{const s=document.getElementById("fig-fe-seeds");if(s)strips(s,[{name:"2026-05-15 label",vals:D.runs.final_energy_old_kmd,col:C.xfer},{name:"GGA/GGA+U label",vals:D.runs.final_energy_new_kmd,col:C.warm}],"final_energy · R² of each run · KMD, same recipe, 5 seeds each","R²");}catch(e){console.error("fig-fe-seeds failed",e);}
  try{const s=document.getElementById("fig-vol-seeds");if(s)strips(s,[{name:"KMD",vals:D.runs.volume_old_kmd,col:C.alone},{name:"XenonPy classic",vals:D.runs.volume_xenonpy_classic,col:C.warm},{name:"XenonPy, no sum block",vals:D.runs.volume_xenonpy_nosum,col:C.frz}],"volume · R² of each run · single task, 5 seeds per arm","R²",{lo:0.55,hi:1.02});}catch(e){console.error("fig-vol-seeds failed",e);}
  try{const s=document.getElementById("fig-other-seeds");if(s)strips(s,[{name:"KMD",vals:D.runs.final_energy_old_kmd,col:C.alone},{name:"classic",vals:D.runs.final_energy_xenonpy_classic,col:C.warm,sub:"final_energy (2026-05-15 label)"},{name:"no sum",vals:D.runs.final_energy_xenonpy_nosum,col:C.frz},{name:"KMD",vals:D.runs.dos_density_old_kmd,col:C.alone},{name:"classic",vals:D.runs.dos_density_xenonpy_classic,col:C.warm,sub:"dos_density"},{name:"no sum",vals:D.runs.dos_density_xenonpy_nosum,col:C.frz}],"R² of each run · the descriptor barely matters for per-atom and intensive labels","R²");}catch(e){console.error("fig-other-seeds failed",e);}

  // ---- fig-curves: loss curves, three volume runs ----
  try{(function(){const svg=document.getElementById("fig-curves");if(!svg)return;svg.textContent="";
    const runs=[["KMD (R² 0.61)",D.curves.volume_kmd_s2025,C.alone],["XenonPy classic (0.997)",D.curves.volume_xenonpy_classic_s2025,C.warm]];
    const W=960,H=330,m={t:40,r:20,b:50,l:66}; size(svg,W,H); const pw=(W-m.l-m.r-40)/2, ih=H-m.t-m.b;
    [["train",0,"training loss (raw), log scale"],["val",pw+40,"validation loss (raw), log scale"]].forEach(([key,ox,title])=>{const x0=m.l+ox;
      const pts=runs.flatMap(r=>r[1]?r[1][key]:[]); const ys=pts.map(p=>p[1]).filter(v=>v>0); const lo=Math.log10(Math.min(...ys)),hi=Math.log10(Math.max(...ys)); const xmax=Math.max(...pts.map(p=>p[0]));
      const X=e=>x0+e/xmax*pw, Y=v=>m.t+ih-(Math.log10(v)-lo)/(hi-lo)*ih;
      for(let p=Math.ceil(lo);p<=Math.floor(hi);p++){const v=Math.pow(10,p);svg.appendChild(el("line",{x1:x0,x2:x0+pw,y1:Y(v),y2:Y(v),stroke:cSoft}));txt(svg,x0-8,Y(v)+4,v>=1?v:v.toString(),"axis",{"text-anchor":"end"});}
      runs.forEach(([name,c,col])=>{if(!c)return;const d=c[key].filter(p=>p[1]>0).map((p,i)=>`${i?"L":"M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");svg.appendChild(el("path",{d,fill:"none",stroke:col,"stroke-width":2.2}));});
      [0,50,100,150].forEach(e=>{if(e<=xmax){txt(svg,X(e),m.t+ih+18,e,"axis",{"text-anchor":"middle"});}});
      svg.appendChild(el("line",{x1:x0,x2:x0+pw,y1:m.t+ih,y2:m.t+ih,stroke:cRule})); txt(svg,x0,m.t-16,title,"axis"); txt(svg,x0+pw/2,H-8,"epoch","axis",{"text-anchor":"middle"});});
  })();}catch(e){console.error("fig-curves failed",e);}

  // ---- fig-fecurves: validation loss before and after the label fix ----
  try{(function(){const svg=document.getElementById("fig-fecurves");if(!svg)return;svg.textContent="";
    const runs=[["2026-05-15 label (R² 0.78)",D.curves.final_energy_old_kmd_s2025,C.xfer],["GGA/GGA+U label (R² 0.999)",D.curves.final_energy_new_kmd_s2025,C.warm]];
    const W=960,H=300,m={t:40,r:20,b:50,l:66}; size(svg,W,H); const iw=W-m.l-m.r, ih=H-m.t-m.b;
    const pts=runs.flatMap(r=>r[1]?r[1].val:[]); const ys=pts.map(p=>p[1]).filter(v=>v>0); const lo=Math.log10(Math.min(...ys)),hi=Math.log10(Math.max(...ys)); const xmax=Math.max(...pts.map(p=>p[0]));
    const X=e=>m.l+e/xmax*iw, Y=v=>m.t+ih-(Math.log10(v)-lo)/(hi-lo)*ih;
    for(let p=Math.ceil(lo);p<=Math.floor(hi);p++){const v=Math.pow(10,p);svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:cSoft}));txt(svg,m.l-8,Y(v)+4,v>=1?v:v.toString(),"axis",{"text-anchor":"end"});}
    runs.forEach(([name,c,col])=>{if(!c)return;const d=c.val.filter(p=>p[1]>0).map((p,i)=>`${i?"L":"M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");svg.appendChild(el("path",{d,fill:"none",stroke:col,"stroke-width":2.2}));});
    [0,25,50,75,100,125].forEach(e=>{if(e<=xmax)txt(svg,X(e),m.t+ih+18,e,"axis",{"text-anchor":"middle"});});
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,stroke:cRule})); txt(svg,m.l,m.t-16,"final_energy · validation loss (raw) of seed 2025, log scale · same recipe, only the label differs","axis"); txt(svg,m.l+iw/2,H-8,"epoch","axis",{"text-anchor":"middle"});
  })();}catch(e){console.error("fig-fecurves failed",e);}

  // ---- fig-atoms: atoms per cell ----
  try{(function(){const svg=document.getElementById("fig-atoms");if(!svg)return;svg.textContent="";
    const h=D.local.atoms_per_cell.hist_log2; const W=960,H=280,m={t:36,r:30,b:56,l:70}; size(svg,W,H); const iw=W-m.l-m.r,ih=H-m.t-m.b;
    const mx=Math.log10(Math.max(...h.counts)+1); const Y=c=>m.t+ih-Math.log10(c+1)/mx*ih; const bw=iw/h.counts.length;
    [1,10,100,1000,10000].forEach(v=>{if(Math.log10(v+1)<=mx){svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:cSoft}));txt(svg,m.l-10,Y(v)+4,v.toLocaleString(),"axis",{"text-anchor":"end"});}});
    h.counts.forEach((c,i)=>{const x=m.l+i*bw+6;svg.appendChild(el("rect",{x,y:Y(c),width:bw-12,height:m.t+ih-Y(c),rx:3,fill:C.frz,opacity:.8}));txt(svg,x+(bw-12)/2,Y(c)-6,c.toLocaleString(),"val",{"text-anchor":"middle",fill:C.frz});txt(svg,x+(bw-12)/2,m.t+ih+18,`${h.edges[i]}–${h.edges[i+1]-1}`,"axis",{"text-anchor":"middle"});});
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,stroke:cRule}));
    txt(svg,m.l+iw/2,H-12,"atoms per cell of the dataset's composition string, 33,829 MP rows · log count","axis",{"text-anchor":"middle"});
  })();}catch(e){console.error("fig-atoms failed",e);}

  // ---- fig-cols: non-null rows per column before and after the rebuild ----
  try{(function(){const svg=document.getElementById("fig-cols");if(!svg)return;svg.textContent="";
    const rows=D.local.columns; const W=1000,rowH=24,m={t:34,r:130,b:36,l:340}; const H=m.t+rows.length*rowH+m.b; size(svg,W,H); const iw=W-m.l-m.r; const X=v=>m.l+v/33829*iw;
    [0,10000,20000,30000].forEach(v=>{svg.appendChild(el("line",{x1:X(v),x2:X(v),y1:m.t-8,y2:m.t+rows.length*rowH,stroke:v===0?cRule:cSoft}));txt(svg,X(v),m.t-14,v.toLocaleString(),"axis",{"text-anchor":"middle"});});
    rows.forEach((r,i)=>{const y=m.t+i*rowH+3;
      svg.appendChild(el("rect",{x:X(0),y,width:X(r.before)-X(0),height:8,fill:C.xfer,opacity:.7}));
      svg.appendChild(el("rect",{x:X(0),y:y+9,width:X(r.after)-X(0),height:8,fill:C.warm,opacity:.85}));
      txt(svg,m.l-10,y+13,r.col,"name",{"text-anchor":"end",fill:cInk}); txt(svg,X(Math.max(r.before,r.after))+8,y+13,`${r.before.toLocaleString()} → ${r.after.toLocaleString()}`,"val",{fill:cInk});});
    txt(svg,m.l+iw/2,H-6,"non-null MP rows per column · 2026-05-15 (ochre, above) vs 2026-09-08 (teal, below)","axis",{"text-anchor":"middle"});
  })();}catch(e){console.error("fig-cols failed",e);}

  // ---- fig-scatter: observation vs prediction, one seed per regression task ----
  try{(function(){const svg=document.getElementById("fig-scatter");if(!svg||!D.baselines)return;svg.textContent="";
    const tasks=D.baselines.per_task.filter(r=>r.kind==="regression"&&r.scatter); const cols=4,cell=240,pad=14,foot=44; const rowsN=Math.ceil(tasks.length/cols);
    const W=cols*cell,H=rowsN*(cell+foot-24); size(svg,W,H);
    tasks.forEach((r,i)=>{const cx=(i%cols)*cell+pad,cy=Math.floor(i/cols)*(cell+foot-24)+pad,w=cell-2*pad,h=cell-2*pad-24;
      const t=r.scatter.true,p=r.scatter.pred; const lo=Math.min(...t,...p),hi=Math.max(...t,...p); const X=v=>cx+(v-lo)/(hi-lo)*w, Y=v=>cy+h-(v-lo)/(hi-lo)*h;
      svg.appendChild(el("rect",{x:cx,y:cy,width:w,height:h,fill:"none",stroke:cRule}));
      svg.appendChild(el("line",{x1:X(lo),y1:Y(lo),x2:X(hi),y2:Y(hi),stroke:cMuted,"stroke-dasharray":"4 3",opacity:.7}));
      const col=r.group==="updated"?C.warm:C.frz;
      for(let k=0;k<t.length;k++)svg.appendChild(el("circle",{cx:X(p[k]),cy:Y(t[k]),r:1.6,fill:col,opacity:.45}));
      txt(svg,cx,cy+h+16,r.task.replace(/_/g," "),"lab",{fill:cInk}); txt(svg,cx,cy+h+34,`R² ${r.r2.mean.toFixed(3)} · MAE ${r.mae.mean.toFixed(3)} · n ${r.scatter.n.toLocaleString()}`,"val",{fill:col});});
  })();}catch(e){console.error("fig-scatter failed",e);}

  // ---- fig-cm: confusion matrices of the three classification tasks ----
  try{(function(){const svg=document.getElementById("fig-cm");if(!svg||!D.baselines)return;svg.textContent="";
    const tasks=D.baselines.per_task.filter(r=>r.kind==="classification"&&r.confusion); const W=960,H=330; size(svg,W,H);
    const panel=W/tasks.length;
    tasks.forEach((r,i)=>{const M=r.confusion.matrix,k=M.length,cell=Math.min(50,(panel-120)/k),x0=i*panel+70,y0=84;
      txt(svg,x0,y0-58,r.task.replace(/_/g," "),"lab",{fill:cInk}); txt(svg,x0,y0-40,`macro-F1 ${r.macro_f1.mean.toFixed(3)} · accuracy ${r.accuracy.mean.toFixed(3)}`,"val",{fill:cMuted});
      r.classes.forEach((c,jx)=>{txt(svg,x0+cell*jx+cell/2,y0-8,c,"axis",{"text-anchor":"middle"});txt(svg,x0-8,y0+cell*jx+cell/2+4,c,"axis",{"text-anchor":"end"});});
      txt(svg,x0+cell*k/2,y0-20,"predicted →","axis",{"text-anchor":"middle",fill:cMuted});
      M.forEach((row,a)=>{const tot=row.reduce((s,v)=>s+v,0);row.forEach((v,b)=>{const f=tot?v/tot:0;
        svg.appendChild(el("rect",{x:x0+cell*b+1,y:y0+cell*a+1,width:cell-2,height:cell-2,rx:2,fill:a===b?C.warm:C.xfer,opacity:f>0?0.08+0.5*Math.sqrt(f):0.03}));
        if(v){txt(svg,x0+cell*b+cell/2,y0+cell*a+cell/2+1,(f*100).toFixed(0)+"%","axis",{"text-anchor":"middle",fill:cInk});txt(svg,x0+cell*b+cell/2,y0+cell*a+cell/2+14,v.toLocaleString(),"axis",{"text-anchor":"middle","font-size":"10",fill:cMuted});}});
        txt(svg,x0+cell*k+8,y0+cell*a+cell/2+4,tot.toLocaleString(),"axis",{fill:cMuted});});
      if(i===0)txt(svg,x0,y0+cell*k+24,"rows = true class · % of the row · count · right: row total","axis",{fill:cMuted});});
  })();}catch(e){console.error("fig-cm failed",e);}
}
draw();
document.querySelectorAll("svg").forEach(s=>{const w=+s.getAttribute("width"),h=+s.getAttribute("height");if(w&&h&&!s.getAttribute("viewBox")){s.setAttribute("viewBox",`0 0 ${w} ${h}`);s.style.width="100%";s.style.maxWidth=Math.round(w*1.4)+"px";s.style.height="auto";}});
const mq=window.matchMedia("(prefers-color-scheme: dark)"); mq.addEventListener&&mq.addEventListener("change",draw);
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
})();
