// language switch: one attribute on the root, remembered per viewer
(function(){{const root=document.documentElement,btns=document.querySelectorAll(".langbar button");
function set(l){{root.dataset.lang=l;btns.forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.lang===l)));try{{localStorage.setItem("lang",l);}}catch(e){{}}}}
btns.forEach(b=>b.addEventListener("click",()=>set(b.dataset.lang)));
let l="en";try{{l=localStorage.getItem("lang")||"en";}}catch(e){{}} set(l);}})();
