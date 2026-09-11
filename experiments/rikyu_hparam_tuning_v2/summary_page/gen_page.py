# -*- coding: utf-8 -*-
"""Generate the trilingual transferability summary — summary_page/transferability_summary.html

English is the source text; the Chinese and Japanese renderings follow it sentence by sentence so the
three can be checked against each other. Figures are drawn by page.js from one data layer (the
constants at the top of that file plus summary/position_runs.json); only prose, headings, captions,
ledger labels and table headers switch language. Figure-internal labels stay English.

    python summary_page/gen_page.py        # then publish summary_page/transferability_summary.html
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent          # summary_page/
EXP = HERE.parent                                # the experiment directory
JS = (HERE / "page.js").read_text(encoding="utf-8")
# per-run scores of the four example tasks at every position of the transfer stage (RIKYU dump)
_pr = json.load(open(EXP / "summary" / "position_runs.json"))
POSRUNS_JS = "const POSRUNS=" + json.dumps({t: [[r["pos"], None if r["at_train"] is None else round(r["at_train"], 5),
                                                None if r["at_end"] is None else round(r["at_end"], 5)] for r in v]
                                            for t, v in _pr.items()}, separators=(",", ":")) + ";\n"
STYLE = (HERE / "_shared_style.html").read_text(encoding="utf-8")
OUT = HERE / "transferability_summary.html"   # results/ is gitignored; this one travels in git
ft=json.load(open(f"{EXP}/summary/ft.json")); pos=json.load(open(f"{EXP}/summary/position.json"))
ta=json.load(open(f"{EXP}/summary/transfer_adopted.json")); mx=json.load(open(f"{EXP}/summary/matched_xfer.json"))
mxd={r["task"]:r for r in mx["per_task"] if "transfer" in r}
lb={r["task"]:r for r in json.load(open(f"{EXP}/summary/long_budget.json"))["per_task"]}

def T(en,zh,ja): return f'<span class="l-en">{en}</span><span class="l-zh">{zh}</span><span class="l-ja">{ja}</span>'

# ---------- ledgers ----------
def f4(x): return "-" if x is None else f"{x:.4f}"
def pcv(x): return "-" if x is None else f"{x:+.1f}%"
def two(se): return "-" if not se else f"{2*se:.4f}"
V={"better":("better","更好","改善"),"worse":("worse","更差","悪化"),"unresolved":("unresolved","不可分","判定不能"),
   "better (negligible)":("better (negligible)","更好(可忽略)","改善(無視できる)"),"worse (negligible)":("worse (negligible)","更差(可忽略)","悪化(無視できる)")}
def verd(d):
    if not d: return "-"
    if not d["separated"]: k="unresolved"
    else: k=("better" if d["delta"]>0 else "worse")+("" if d["practically_significant"] else " (negligible)")
    return T(*V[k])
def ledger(rows):
    body="".join(f"<tr><td>{T(*c)}</td><td>{v}</td><td>{n}</td><td>{u}</td><td>{ver}</td><td>{T(*src)}</td></tr>" for c,v,n,u,ver,src in rows)
    hdr=f"<tr><th>{T('Claim','断言','主張')}</th><th>{T('Value','数值','値')}</th><th>n</th><th>2×SE</th><th>{T('Verdict','判定','判定')}</th><th>{T('Source','来源','出典')}</th></tr>"
    return f'<div class="tablebox ledger"><table><thead>{hdr}</thead><tbody>{body}</tbody></table></div>'
def EVof(t):
    r=next(x for x in ft["per_task"] if x["task"]==t); p=next(x for x in pos["per_task"] if x["task"]==t); xr=mxd[t]
    return r,p,xr
def arm_row(name3,mean,rel,se,sep,n,base,src3):
    d={"delta":mean-base,"separated":sep,"practically_significant":abs(mean-base)>=0.01}
    return (name3, f"{f4(mean)} ({pcv(rel)})", n, two(se), verd(d), src3)
L_XFER=("xfer (step 24, replay) vs alone","xfer(第 24 步,带 replay)vs 单独训练","xfer(第24ステップ、リプレイあり)vs 単独学習")
L_FRZ=("frozen vs alone","冻结编码器 vs 单独训练","エンコーダ凍結 vs 単独学習")
L_WARM=("warm-start vs alone","warm-start vs 单独训练","ウォームスタート vs 単独学習")
L_WF=("warm-start vs frozen","warm-start vs 冻结","ウォームスタート vs 凍結")
L_SH=("gradient share at step 24","第 24 步的梯度份额","第24ステップでの勾配占有率")
L_EP=("median epochs: xfer / alone","中位 epoch 数:xfer / 单独","エポック数の中央値:xfer / 単独")
L_EARLY=("early slots 1–8 vs alone","早期位置 1–8 vs 单独","序盤(1–8位置)vs 単独")
L_LATE=("late slots 17–24 vs alone","晚期位置 17–24 vs 单独","終盤(17–24位置)vs 単独")
SRC_MX=("matched_xfer.json, 10 orderings, shared rows","matched_xfer.json,10 组顺序,共同测试行","matched_xfer.json、10通りの順序、共通テスト行")
SRC_FT=("ft.json, stage_ft","ft.json,stage_ft","ft.json、stage_ft")
SRC_POS=("position.json","position.json","position.json")
SRC_CSV=("metrics.csv","metrics.csv","metrics.csv")
SRC_SHARE=("N_train / (N_train + Σ replay)","N_train / (N_train + Σ replay)","N_train / (N_train + Σ replay)")

def band_row(label3,d):
    return (label3, pcv(d["relative_pct"]), d["n"], two(d["se_of_difference"]), verd(d), SRC_POS)

r,p,xr=EVof("zt"); z=next(x for x in ta["per_task"] if x["task"]=="zt")
zt_l=ledger([
 (("6-task probe: multi-task vs alone","6 任务探针:多任务 vs 单独训练","6タスク探針:マルチタスク vs 単独学習"), f"{f4(z['multi_task_r2'])} vs {f4(z['single_task_r2'])} ({pcv(z['relative_pct'])})","25 vs 5",two(z["se_of_difference"]),T(*V["better" if z["separated"] else "unresolved"]),("transfer_adopted.json","transfer_adopted.json","transfer_adopted.json")),
 arm_row(L_XFER,xr["multi_task"],xr["relative_pct"],xr.get("se_of_difference"),xr["separated"],10,r["single_task"],SRC_MX),
 arm_row(L_FRZ,r["ftz"]["mean"],r["ftz_vs_single"]["relative_pct"],r["ftz_vs_single"]["se_of_difference"],r["ftz_vs_single"]["separated"],r["ftz"]["n"],r["single_task"],SRC_FT),
 arm_row(L_WARM,r["ftf"]["mean"],r["ftf_vs_single"]["relative_pct"],r["ftf_vs_single"]["se_of_difference"],r["ftf_vs_single"]["separated"],r["ftf"]["n"],r["single_task"],SRC_FT),
 (L_WF,pcv(r["ftf_vs_ftz"]["relative_pct"]),"10 vs 10",two(r["ftf_vs_ftz"]["se_of_difference"]),verd(r["ftf_vs_ftz"]),SRC_FT),
 (L_SH,"4.2%","-","-","-",SRC_SHARE),(L_EP,"58 / 72","10 / 5","-","-",SRC_CSV)])
r,p,xr=EVof("material_type")
mt_l=ledger([
 arm_row(L_XFER,xr["multi_task"],xr["relative_pct"],xr.get("se_of_difference"),xr["separated"],10,r["single_task"],("matched_xfer.json, macro-F1","matched_xfer.json,macro-F1","matched_xfer.json、macro-F1")),
 arm_row(L_FRZ,r["ftz"]["mean"],r["ftz_vs_single"]["relative_pct"],r["ftz_vs_single"]["se_of_difference"],r["ftz_vs_single"]["separated"],r["ftz"]["n"],r["single_task"],("ft.json, macro-F1","ft.json,macro-F1","ft.json、macro-F1")),
 arm_row(L_WARM,r["ftf"]["mean"],r["ftf_vs_single"]["relative_pct"],r["ftf_vs_single"]["se_of_difference"],r["ftf_vs_single"]["separated"],r["ftf"]["n"],r["single_task"],("ft.json, macro-F1","ft.json,macro-F1","ft.json、macro-F1")),
 (L_WF,pcv(r["ftf_vs_ftz"]["relative_pct"]),"10 vs 10",two(r["ftf_vs_ftz"]["se_of_difference"]),verd(r["ftf_vs_ftz"]),SRC_FT),
 band_row(L_EARLY,p["early_1_8"]),band_row(L_LATE,p["late_17_24"]),
 (("rare-class precision, alone → xfer","稀有类 precision,单独 → xfer","希少クラスのprecision、単独→xfer"),"IAC 30.2→53.5 · IQC 51.1→62.2","5 vs 3 runs","-","-",("matched confusion matrices","匹配测试集的混淆矩阵","共通テスト行の混同行列")),
 (("“others” misfiled as rare, alone → xfer","others 误判为稀有类,单独 → xfer","othersの希少クラスへの誤分類、単独→xfer"),"1.16% → 0.49% (84 → 36 rows/run)","5 vs 3 runs","-","-",("matched confusion matrices","匹配测试集的混淆矩阵","共通テスト行の混同行列")),
 (L_SH,"32.2%","-","-","-",("largest of any task","24 个任务中最大","全タスク中で最大"))])
r,p,xr=EVof("magnetic_moment")
mm_l=ledger([
 arm_row(L_XFER,xr["multi_task"],xr["relative_pct"],xr.get("se_of_difference"),xr["separated"],10,r["single_task"],SRC_MX),
 arm_row(L_FRZ,r["ftz"]["mean"],r["ftz_vs_single"]["relative_pct"],r["ftz_vs_single"]["se_of_difference"],r["ftz_vs_single"]["separated"],r["ftz"]["n"],r["single_task"],SRC_FT),
 arm_row(L_WARM,r["ftf"]["mean"],r["ftf_vs_single"]["relative_pct"],r["ftf_vs_single"]["se_of_difference"],r["ftf_vs_single"]["separated"],r["ftf"]["n"],r["single_task"],SRC_FT),
 (("warm-start vs xfer","warm-start vs xfer","ウォームスタート vs xfer"),pcv(r["ftf_vs_xfer"]["relative_pct"]),"10 vs 10",two(r["ftf_vs_xfer"]["se_of_difference"]),verd(r["ftf_vs_xfer"]),("ft.json (one-sided SE)","ft.json(单侧 SE)","ft.json(片側SE)")),
 band_row(L_EARLY,p["early_1_8"]),band_row(L_LATE,p["late_17_24"]),
 (L_SH,"1.1%","-","-","-",("851 rows vs 79,113 replay rows","851 行 vs 79,113 行 replay","851行 vs リプレイ79,113行")),(L_EP,"56 / 142","10 / 5","-","-",SRC_CSV)])
r,p,xr=EVof("final_energy")
fe_l=ledger([
 arm_row(L_XFER,xr["multi_task"],xr["relative_pct"],xr.get("se_of_difference"),xr["separated"],10,r["single_task"],SRC_MX),
 arm_row(L_FRZ,r["ftz"]["mean"],r["ftz_vs_single"]["relative_pct"],r["ftz_vs_single"]["se_of_difference"],r["ftz_vs_single"]["separated"],r["ftz"]["n"],r["single_task"],SRC_FT),
 arm_row(L_WARM,r["ftf"]["mean"],r["ftf_vs_single"]["relative_pct"],r["ftf_vs_single"]["se_of_difference"],r["ftf_vs_single"]["separated"],r["ftf"]["n"],r["single_task"],SRC_FT),
 (L_WF,pcv(r["ftf_vs_ftz"]["relative_pct"]),"10 vs 10",two(r["ftf_vs_ftz"]["se_of_difference"]),verd(r["ftf_vs_ftz"]),SRC_FT),
 band_row(L_EARLY,p["early_1_8"]),band_row(L_LATE,p["late_17_24"]),
 (L_SH,"24.4%","-","-","-",("not a small task","不是小任务","小さなタスクではない")),
 (("descriptor: formula_to_composition('Fe2O3') == ('Fe4O6')","描述符:formula_to_composition('Fe2O3') == ('Fe4O6')","記述子:formula_to_composition('Fe2O3') == ('Fe4O6')"),"True","-","-","-",("verified in code","代码层面验证","コード上で検証")),
 (("corr(Volume, atoms per cell)","corr(体积, 原胞原子数)","corr(体積, 単位胞の原子数)"),"+0.868 (75.3% of variance)","33,829","-","-",("qc dataset","qc 数据集","qcデータセット")),
 (("reduced formulas that repeat","重复出现的约化化学式","重複する既約化学式"),"7 of 33,822 (0.02%)","-","-","-",("qc dataset — not label noise","qc 数据集——不是标签噪声","qcデータセット — ラベルノイズではない")),
 (("500 epochs, no early stop: warm-start vs alone, same budget","500 epoch、无早停:warm-start vs 单独,同预算","500エポック・早期終了なし:ウォームスタート vs 単独、同予算"),f"{lb['final_energy']['warm_500']['mean']:.4f} vs {lb['final_energy']['alone_500']['mean']:.4f} ({pcv(lb['final_energy']['warm_500_vs_alone_500']['relative_pct'])})","10 vs 5",two(lb['final_energy']['warm_500_vs_alone_500']['se_of_difference']),verd(lb['final_energy']['warm_500_vs_alone_500']),("long_budget.json: stage_ft ftfl, stage_single stL","long_budget.json:stage_ft ftfl、stage_single stL","long_budget.json:stage_ft ftfl、stage_single stL")),
 (("epoch of lowest validation loss, warm-start / alone","验证损失最低的 epoch,warm-start / 单独","検証損失が最小のエポック、ウォームスタート / 単独"),f"{lb['final_energy']['warm_500_curve']['best_epoch']:.0f} / {lb['final_energy']['alone_500_curve']['best_epoch']:.0f} (medians)","10 / 5","-","-",("per-epoch logs of the 500-epoch runs","500 epoch 运行的逐 epoch 日志","500エポック実行のエポックごとのログ")),
 (("train / validation loss at epoch 500, warm-start vs alone","第 500 epoch 的训练 / 验证损失,warm-start vs 单独","第500エポックの学習 / 検証損失、ウォームスタート vs 単独"),"0.011 / 0.294 vs 0.015 / 0.223","10 vs 5","-","-",("metrics.csv, medians","metrics.csv 中位数","metrics.csv の中央値"))])

# ---------- legends shared by several figures ----------
LEG_STRIP=f'<div class="legend"><span>● {T("one run (a seed, or an ordering)","一个运行(一个 seed 或一组顺序)","1実行(1シードまたは1順序)")}</span><span>▬ {T("median of the arm","该臂的中位数","アームの中央値")}</span><span>┄ {T("median of alone","单独训练的中位数","単独の中央値")}</span></div>'
LEG_POS=f'<div class="legend"><span>● {T("one run, at the position where this task was trained","一个运行,位于该任务被训练的位置","1実行、このタスクが学習された位置")}</span><span>▬ {T("median at that position","该位置的中位数","その位置の中央値")}</span><span>┄ {T("median of alone; shaded = range of its 5 seeds","单独中位数;阴影 = 5 个 seed 的范围","単独の中央値;網掛け=5シードの範囲")}</span><span>{T("tinted columns = early (1–8) and late (17–24) bands, % as in the ledger","浅色区 = 早期(1–8)与晚期(17–24)区段,% 同证据表","色付き帯=序盤(1–8)と終盤(17–24)、%は証拠表と同じ")}</span></div>'
KEY_DOTS=f'<span>● {T("one run","一个运行","1実行")} · ▬ {T("median","中位数","中央値")}</span>'
KEY_FILL=f'<span><i class="sw" style="background:var(--ink)"></i> {T("filled = separated and |Δ| ≥ 0.01","实心 = 分离且 |Δ| ≥ 0.01","塗りつぶし=分離かつ|Δ| ≥ 0.01")}</span><span><i class="sw" style="background:var(--surface);border:2px solid var(--ink)"></i> {T("hollow = unresolved or negligible","空心 = 不可分或可忽略","中抜き=判定不能または無視できる")}</span>'
HOWTO=f'''<div class="callout">{T(
"<b>How to read the numbers.</b> Every comparison is made task by task, never pooled. “vs alone” is the arm's mean score minus the alone mean, as a percentage of the alone mean. SE is the standard error of that difference; it combines the spread of both arms (√(SE<sub>arm</sub>² + SE<sub>alone</sub>²)), and 2×SE is roughly a 95% interval around the difference. A difference is <em>separated</em> when it is larger than 2×SE, and counts as <em>better</em> or <em>worse</em> only if it is also at least 0.01 in absolute score, the practical threshold; anything else is <em>unresolved</em>. n is 10 orderings for xfer, frozen and warm-start, and 5 seeds for alone.",
"<b>数字怎么读。</b>所有比较都按任务逐一进行,从不合并。“vs 单独”是该臂的平均分减去单独训练的平均分,再除以单独的平均分,以百分比表示。SE 是这一差值的标准误,合并了两臂各自的离散(√(SE<sub>臂</sub>² + SE<sub>单独</sub>²));2×SE 大致对应差值的 95% 区间。差值大于 2×SE 称为<em>分离</em>;分离且绝对差值不小于 0.01(实用门槛)才算<em>更好</em>或<em>更差</em>;其余为<em>不可分</em>。n:xfer、冻结、warm-start 各为 10 组顺序,单独为 5 个 seed。",
"<b>数値の読み方。</b>比較はすべてタスクごとに行い、プールしない。「vs 単独」はそのアームの平均スコアから単独学習の平均を引き、単独の平均で割った百分率。SEはその差の標準誤差で、両アームの散らばりを合成したもの(√(SE<sub>アーム</sub>² + SE<sub>単独</sub>²))。2×SEは差のおおよそ95%区間に当たる。差が2×SEより大きければ<em>分離</em>とし、さらに絶対差が0.01(実用閾値)以上のときだけ<em>改善</em>または<em>悪化</em>と数える。それ以外は<em>判定不能</em>。nはxfer・凍結・ウォームスタートが10通りの順序、単独が5シード。")}</div>'''

CNT=ft.get("counts",{})
def cell_u(key):
    c=CNT.get(key); n_u=sum(1 for r in ft["per_task"] if r.get("ftfu"))
    if not c or n_u==0:
        return f'<div class="run">{T("running · n=3 · fresh head","进行中 · n=3 · 新建 head","実行中 · n=3 · 新規ヘッド")}</div>'
    s=f"{c['better']} / {c['worse']} / {c['unresolved']}"
    return f'<div class="done">{T("done · n=3 · fresh head · "+s,"完成 · n=3 · 新建 head · "+s,"完了 · n=3 · 新規ヘッド · "+s)}</div>'

# ---------- page ----------
def P(en,zh,ja,cls=""): return f'<p class="{cls}">{T(en,zh,ja)}</p>' if cls else f"<p>{T(en,zh,ja)}</p>"
def CAP(en,zh,ja): return f"<figcaption>{T(en,zh,ja)}</figcaption>"
def FH(en,zh,ja): return f'<div class="fighead"><span class="fignum"></span><span class="figtitle">{T(en,zh,ja)}</span></div>'
def TH(en,zh,ja): return f'<div class="tabhead"><span class="tabnum"></span><span class="figtitle">{T(en,zh,ja)}</span></div>'


out=[]
out.append('''<title>Transferability, Four Ways</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=Public+Sans:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&family=Noto+Sans+JP:wght@400;500;700&family=Noto+Serif+SC:wght@600&family=Noto+Serif+JP:wght@600&display=swap">
'''+STYLE+'''
<div class="langbar" role="group" aria-label="Language"><button data-lang="en" aria-pressed="true">EN</button><button data-lang="zh">中</button><button data-lang="ja">日</button></div>
<div class="wrap">''')

# ---- header ----
out.append(f'''<header>
<p class="eyebrow">{T("Continual multi-task pretraining · 24 tasks · transferability","连续多任务预训练 · 24 任务 · 迁移性","継続的マルチタスク事前学習 · 24タスク · 転移性")}</p>
<h1>{T("Transferability, four ways in","迁移性:四种接入方式","転移性:4通りのつなぎ方")}</h1>
<p class="standfirst">{T(
"The first transfer measurement said 18 of 24 tasks are worse off with a shared encoder. That measured one particular way of attaching a task, and of the four compared here it turned out to be the weakest. Measured three more ways, the shared encoder is roughly a wash against training alone, with real wins on four tasks and real losses on three. This page explains the four setups, shows the full result, walks through one task of each kind, and closes the question of whether the encoder needs to have seen the task at all.",
"最初的迁移测量说 24 个任务里有 18 个在共享编码器下变差。它测的只是一种特定的接入方式,而在本页对比的四种里,它的效果最差。再用三种方式测量后,共享编码器相对于单独训练大体持平:四个任务确实获益,三个任务确实受损。本页先解释四种设置,再给出完整结果,逐一分析每类的一个代表任务,最后回答编码器是否需要见过该任务这个问题。",
"最初の転移測定では、24タスク中18が共有エンコーダで悪化するとされた。だがそれは特定のつなぎ方だけを測っており、本ページで比較した4通りの中では最も効果が低かった。さらに3通りで測ると、共有エンコーダは単独学習とほぼ互角で、4タスクは実際に改善し、3タスクは実際に悪化する。本ページでは4つの設定を説明し、全結果を示し、各種類の代表タスクを1つずつ詳しく見て、エンコーダがそのタスクを見ている必要があるのかという問いに答える。")}</p>
</header>''')

# ---- 1 setups ----
out.append(f'''<section class="section">
<div class="col"><p class="kicker">{T("1 · The setups","1 · 四种设置","1 · 設定")}</p><h2>{T('What "transfer" means here, four ways','这里的"迁移"指什么:四种方式','ここで言う「転移」とは:4通り')}</h2>
{P("Every arm starts from the same place: an encoder trained continuously on the other 23 tasks with hybrid replay. They differ in <em>how</em> the 24th task, X, is then attached to it. <span class='num'>xfer</span> is the campaign's shorthand for the first of these; the other names are ours.",
   "每个臂的起点相同:一个在其余 23 个任务上以混合 replay 连续训练出的编码器。差别只在于第 24 个任务 X <em>如何</em>接上去。<span class='num'>xfer</span> 是本轮 campaign 对第一种方式的简写;其余名称是我们自己起的。",
   "すべてのアームの出発点は同じで、他の23タスクをハイブリッドリプレイで継続学習したエンコーダである。違いは24番目のタスクXを<em>どう</em>つなぐかだけだ。<span class='num'>xfer</span>は最初の方式に対するキャンペーンでの略記で、他の名前は本ページでの呼称である。")}</div>
<figure>{FH("The six ways of attaching task X, drawn as branches off the 23-task encoder","接入任务 X 的六种方式,画成从 23 任务编码器分出的分支","タスクXをつなぐ6通りの方法を、23タスクエンコーダからの分岐として描いたもの")}<div class="figbox"><svg id="fig0" width="960" height="250" role="img" aria-label="Schematic of the four transfer arms branching from a 23-task encoder."></svg></div>
{CAP("Solid paths are measured (n = 10 orderings per task). The dashed pair — fine-tuning an encoder that has never seen X, with a fresh head — is measured at n = 3 orderings per task.",
     "实线路径已测量(每任务 10 组顺序)。虚线的两条——在从未见过 X 的编码器上、用新建的 head 微调——每任务测了 3 组顺序。",
     "実線の経路は測定済み(タスクごとに10通りの順序)。破線の2本 — Xを一度も見ていないエンコーダに新規ヘッドで微調整 — はタスクごとに3通りの順序で測定。")}</figure>
{TH("The six arms: what trains, whether replay is on, what early stopping watches, and what each asks","六个臂:训练什么、是否有 replay、早停监控什么、各回答什么问题","6つのアーム:何を学習するか、リプレイの有無、早期終了の監視対象、何を問うか")}<div class="tablebox setup"><table>
<thead><tr><th>{T("Arm","臂","アーム")}</th><th>{T("What trains","训练什么","何を学習するか")}</th><th>{T("Replay","Replay","リプレイ")}</th><th>{T("Early stopping watches","早停监控","早期終了の監視対象")}</th><th>{T("What it asks","回答什么问题","何を問うか")}</th></tr></thead>
<tbody>
<tr><td class="c-alone">alone</td><td>{T("A fresh model on X only, 5 seeds.","只在 X 上从零训练的模型,5 个 seed。","Xだけで新規に学習したモデル、5シード。")}</td><td>{T("none","无","なし")}</td><td>{T("X's own loss","X 自身的损失","X自身の損失")}</td><td>{T('The reference. Everything below is "vs alone".','参照基线。下文一切"vs 单独"都相对它。','参照基準。以下の「vs 単独」はすべてこれに対する値。')}</td></tr>
<tr><td class="c-xfer">xfer</td><td>{T("X added as step 24 of the continual sequence: encoder + fresh head, jointly.","X 作为连续序列的第 24 步加入:编码器 + 新建 head 联合训练。","Xを継続系列の第24ステップとして追加:エンコーダと新規ヘッドを同時学習。")}</td><td>{T("<b>on</b> — every epoch draws max(1500, 0.3N) rows from each of the 23 old tasks","<b>开</b>——每个 epoch 从 23 个旧任务各抽 max(1500, 0.3N) 行","<b>あり</b> — 各エポックで旧23タスクそれぞれから max(1500, 0.3N) 行を抽出")}</td><td>{T("the <b>sum</b> over all 24 tasks","24 个任务的<b>总和</b>","24タスクの<b>合計</b>")}</td><td>{T("What does X get when it arrives last in continual pretraining?","X 作为末位加入连续预训练时得到什么?","継続事前学習の最後に来たXは何を得るか?")}</td></tr>
<tr><td class="c-frz">frozen</td><td>{T("From the xfer checkpoint: X's head only. Encoder bit-frozen.","从 xfer 检查点出发:只训 X 的 head,编码器逐位冻结。","xferのチェックポイントから:Xのヘッドのみ学習。エンコーダはビット単位で凍結。")}</td><td>{T("none","无","なし")}</td><td>{T("X's own loss","X 自身的损失","X自身の損失")}</td><td>{T("Is the shared representation, as is, good for X?","共享表示原样拿来,对 X 够不够好?","共有表現はそのままでXに有効か?")}</td></tr>
<tr><td class="c-warm">warm-start</td><td>{T("From the xfer checkpoint: encoder + X's head.","从 xfer 检查点出发:编码器 + X 的 head 一起训。","xferのチェックポイントから:エンコーダとXのヘッドを学習。")}</td><td>{T("none","无","なし")}</td><td>{T("X's own loss","X 自身的损失","X自身の損失")}</td><td>{T("The model library's intended use.","模型库预想的用法。","モデルライブラリの想定用途。")}</td></tr>
<tr><td class="c-frz">frozen, unseen</td><td>{T("From a 23-task encoder that never saw X: fresh head only.","从从未见过 X 的 23 任务编码器出发:只训新建 head。","Xを一度も見ていない23タスクのエンコーダから:新規ヘッドのみ学習。")}</td><td>{T("none","无","なし")}</td><td>{T("X's own loss","X 自身的损失","X自身の損失")}</td><td>{T("Same question, on a genuinely new task.","同一问题,但针对真正的新任务。","同じ問いを、真に新しいタスクで。")}</td></tr>
<tr><td class="c-warm">warm-start, unseen</td><td>{T("From the same 23-task encoder: encoder + fresh head.","从同一 23 任务编码器出发:编码器 + 新建 head。","同じ23タスクのエンコーダから:エンコーダと新規ヘッド。")}</td><td>{T("none","无","なし")}</td><td>{T("X's own loss","X 自身的损失","X自身の損失")}</td><td>{T("The library's use on a genuinely new task.","模型库在真正新任务上的用法。","真に新しいタスクでのライブラリの用途。")}</td></tr>
</tbody></table></div>
<div class="col" style="margin-top:18px">{P("The two details that end up mattering are in the third and fourth columns. In xfer, X shares every epoch with roughly 79,000 replayed rows from the other tasks, and the stopping rule is the total loss — which the 23 already-converged tasks dominate. Both frozen and warm-start remove replay and let X's own loss decide when it is done.",
"最终起决定作用的两个细节在第三、四列。在 xfer 里,X 每个 epoch 都和其他任务约 79,000 行 replay 同台,而停止规则看的是总损失——由 23 个已收敛的任务主导。冻结与 warm-start 两臂都去掉了 replay,让 X 自身的损失决定何时结束。",
"最終的に効いてくる2つの細部は第3・第4列にある。xferではXは各エポックで他タスクの約79,000行のリプレイと同居し、停止規則は合計損失 — すでに収束した23タスクが支配する — である。凍結とウォームスタートはいずれもリプレイを外し、X自身の損失で終了を決める。")}</div>
</section>''')

# ---- 2 result ----
out.append(f'''<section class="section">
<div class="col"><p class="kicker">{T("2 · The result","2 · 结果","2 · 結果")}</p><h2>{T("One column nearly all red, one nearly all grey — separated by a replay switch","一列几乎全红,一列几乎全灰——中间只差一个 replay 开关","一列はほぼ全て赤、一列はほぼ全て灰 — 違いはリプレイのスイッチ一つ")}</h2></div>
<div class="stats">
<div class="stat"><p class="label">{T("xfer vs alone","xfer vs 单独","xfer vs 単独")}</p><p class="value c-xfer" id="s-xfer"></p><p class="note">{T("better / worse / unresolved, 24 tasks","更好 / 更差 / 不可分,24 个任务","改善 / 悪化 / 判定不能、24タスク")}</p></div>
<div class="stat"><p class="label">{T("frozen vs alone","冻结 vs 单独","凍結 vs 単独")}</p><p class="value c-frz" id="s-frz"></p><p class="note">{T("better / worse / unresolved","更好 / 更差 / 不可分","改善 / 悪化 / 判定不能")}</p></div>
<div class="stat"><p class="label">{T("warm-start vs alone","warm-start vs 单独","ウォームスタート vs 単独")}</p><p class="value c-warm" id="s-warm"></p><p class="note">{T("better / worse / unresolved","更好 / 更差 / 不可分","改善 / 悪化 / 判定不能")}</p></div>
<div class="stat"><p class="label">{T("warm-start vs xfer","warm-start vs xfer","ウォームスタート vs xfer")}</p><p class="value c-warm" id="s-rec"></p><p class="note">{T("tasks that recover once replay is removed","去掉 replay 后回升的任务数","リプレイを外すと回復するタスク数")}</p></div>
</div>
{HOWTO}
<figure>{FH("Score change against training alone, task by task, for xfer, frozen and warm-start","每个任务相对单独训练的分数变化:xfer、冻结、warm-start","タスクごとの単独学習に対するスコア変化:xfer、凍結、ウォームスタート")}<div class="figbox">
<div class="legend"><span><i class="sw" style="background:var(--xfer)"></i> {T("xfer (replay, placed last)","xfer(带 replay,排末位)","xfer(リプレイあり、最後に配置)")}</span><span><i class="sw" style="background:var(--frz)"></i> {T("frozen","冻结","凍結")}</span><span><i class="sw" style="background:var(--warm)"></i> {T("warm-start","warm-start","ウォームスタート")}</span>{KEY_FILL}<span style="color:var(--muted)">{T("connector runs xfer → warm-start · x-axis clipped at ±30%","连线从 xfer 指向 warm-start · 横轴截断于 ±30%","連結線は xfer → ウォームスタート · 横軸は±30%で切り詰め")}</span></div>
<svg id="fig1" width="960" height="720" role="img" aria-label="Per-task relative change against training alone for the three arms, sorted by warm-start."></svg>
</div>{CAP("Each row is one task, sorted by its warm-start result. The connector shows how far removing replay moves it. Filled markers are separated from zero at 2×SE (both arms) and past the 0.01 practical threshold; hollow markers are not. magnetic_susceptibility (58 rows) sits off-scale to the left in every arm and is excluded from the counts.",
"每行一个任务,按 warm-start 结果排序。连线显示去掉 replay 后移动了多远。实心标记 = 在 2×SE(计两臂)下与零分离且超过 0.01 实用门槛;空心 = 否。magnetic_susceptibility(58 行)在每个臂都超出左边界,不计入计数。",
"各行が1タスクで、ウォームスタートの結果順に並ぶ。連結線はリプレイ除去でどれだけ動いたかを示す。塗りつぶしは2×SE(両アーム)で0から分離し、かつ実用閾値0.01を超えるもの、中抜きはそれ以外。magnetic_susceptibility(58行)はどのアームでも左に振り切れており、集計から除外。")}</figure>
{TH("Full result — every task under every arm, with its verdict against training alone","完整结果——每个任务在每个臂下的值,及相对单独训练的判定","全結果 — 全タスク・全アームの値と、単独学習に対する判定")}<div class="tablebox"><table id="bigtable">
<thead><tr><th rowspan="2">{T("Task","任务","タスク")}</th><th rowspan="2">N</th><th rowspan="2">{T("Alone","单独","単独")}</th><th colspan="2" class="grp c-xfer">xfer</th><th colspan="2" class="grp c-frz">{T("frozen","冻结","凍結")}</th><th colspan="2" class="grp c-warm">warm-start</th><th rowspan="2">{T("warm − frozen","warm − 冻结","warm − 凍結")}</th><th colspan="2" class="grp c-frz">{T("frozen, unseen","冻结,未见过","凍結、未知")}</th><th colspan="2" class="grp c-warm">{T("warm-start, unseen","warm-start,未见过","ウォームスタート、未知")}</th></tr>
<tr><th>{T("value","值","値")}</th><th>{T("vs alone","vs 单独","vs 単独")}</th><th>{T("value","值","値")}</th><th>{T("vs alone","vs 单独","vs 単独")}</th><th>{T("value","值","値")}</th><th>{T("vs alone","vs 单独","vs 単独")}</th><th>{T("value","值","値")}</th><th>{T("vs alone","vs 单独","vs 単独")}</th><th>{T("value","值","値")}</th><th>{T("vs alone","vs 单独","vs 単独")}</th></tr></thead>
<tbody></tbody></table></div>
<div class="col"><p class="tabnote">{T("* separated at 2×SE and |Δ| ≥ 0.01 &nbsp;·&nbsp; · separated but below the threshold &nbsp;·&nbsp; material_type is scored on macro-F1, all others on R². seebeck and power_factor rows use the 400-epoch reruns in every arm (the 150-epoch cap truncated them).",
"* 在 2×SE 下分离且 |Δ| ≥ 0.01 · · 分离但低于门槛 · material_type 用 macro-F1,其余用 R²。seebeck 与 power_factor 两行在每个臂都用 400 epoch 的补跑(150 上限截断了它们)。",
"* 2×SEで分離かつ|Δ| ≥ 0.01 · · 分離するが閾値未満 · material_typeはmacro-F1、他はR²。seebeckとpower_factorの行は全アームで400エポックの再実行を使用(150エポックの上限で打ち切られていたため)。")}</p></div>
</section>''')

# ---- 3 why ----
out.append(f'''<section class="section">
<div class="col"><p class="kicker">{T("3 · Why","3 · 原因","3 · なぜか")}</p><h2>{T("At step 24 the new task barely owns its own gradient","第 24 步里,新任务几乎不拥有自己的梯度","第24ステップで新タスクは自分の勾配をほとんど持てない")}</h2>
{P("In xfer, X's rows and the replayed rows share every epoch. The replay total is roughly fixed — about 79,000 rows — so X's share of the gradient is set by its own size:",
   "在 xfer 里,X 的样本和 replay 样本共享每一个 epoch。replay 总量大致固定——约 79,000 行——所以 X 的梯度份额由它自身的大小决定:",
   "xferではXの行とリプレイ行が各エポックを共有する。リプレイ総量はほぼ一定 — 約79,000行 — なので、Xの勾配占有率は自身のサイズで決まる:")}</div>
{TH("The share of the step-24 gradient that the new task owns, for four tasks","四个任务在第 24 步各自占有的梯度份额","4タスクが第24ステップの勾配に占める割合")}<div class="tablebox" style="max-width:560px"><table><thead><tr><th>{T("Task","任务","タスク")}</th><th>{T("X's rows","X 的行数","Xの行数")}</th><th>{T("Replay rows","Replay 行数","リプレイ行数")}</th><th>{T("X's share","X 的份额","Xの占有率")}</th></tr></thead><tbody id="sharetable"></tbody></table></div>
<div class="col" style="margin-top:16px">
{P("And the stopping rule compounds it: early stopping watches the sum over all 24 tasks, which the 23 replayed tasks — already converged — dominate. It fires when <em>they</em> stop improving. So the last task gets a median of about 60 epochs where, trained alone, it takes 90 to 150. Two thirds of the tasks still had their own loss falling when xfer stopped.",
   "停止规则又加重了这一点:早停监控 24 个任务的总和,而这由 23 个已收敛的 replay 任务主导。它在<em>它们</em>停止改进时触发。于是末位任务只得到中位约 60 个 epoch,而单独训练要 90–150 个。xfer 停止时,三分之二的任务自身损失仍在下降。",
   "停止規則がこれを増幅する。早期終了は24タスクの合計を監視し、それはすでに収束した23のリプレイタスクが支配する。<em>それら</em>の改善が止まった時点で発火する。結果、最後のタスクは中央値で約60エポックしか得られない — 単独なら90〜150エポック必要なのに。xferが停止した時点で、3分の2のタスクは自身の損失がまだ下がり続けていた。")}
{P('That is a mechanism for "the smaller the task, the larger the loss" — the correlation the first measurement found — that says nothing about whether the encoder’s representation suits the task. The frozen and warm-start arms are the test: same checkpoint, same task, replay off, X’s own loss deciding. <b>19 tasks recover, none get worse.</b> The dilution is worth keeping as a result in its own right: whatever the encoder offers, the new task’s share of the training data decides whether it shows.',
   '这就是"任务越小、损失越大"——最初测量发现的那条相关性——的机制,而它与编码器表示是否适合该任务毫无关系。冻结和 warm-start 两臂正是检验:同一检查点、同一任务、关掉 replay、由 X 自身损失决定。<b>19 个任务回升,0 个变差。</b>稀释本身也值得作为结论记下:无论编码器能提供什么,新任务在训练数据中所占的份额决定了它能否显现。',
   'これが「タスクが小さいほど損失が大きい」 — 最初の測定が見つけた相関 — の機構であり、エンコーダの表現がそのタスクに適するかどうかとは無関係である。凍結とウォームスタートがその検証だ:同じチェックポイント、同じタスク、リプレイなし、X自身の損失で判断。<b>19タスクが回復し、悪化はゼロ。</b>希釈自体も結論として残す価値がある:エンコーダが何を提供しようと、新タスクが学習データに占める割合がそれが現れるかどうかを決める。')}</div>
</section>''')

# ---- 4 examples ----
out.append(f'''<section class="section">
<div class="col"><p class="kicker">{T("4 · One task of each kind","4 · 每类一个代表任务","4 · 各種類の代表タスク")}</p><h2>{T('What "better", "unresolved" and "worse" each look like up close — and the exception','"更好"、"不可分"、"更差"各自的近景——以及例外','「改善」「判定不能」「悪化」それぞれの近景 — そして例外')}</h2>
{P("Four tasks, chosen as the representative of each verdict. Every one comes with three things: a <b>ledger</b> of every number the prose relies on, with its sample size, uncertainty and source; the <b>run-level distribution</b> of all four arms (every dot is one run — 5 seeds for alone, 10 orderings otherwise; the bar is the median, the dashed line the alone median carried across); and a <b>task-specific figure</b> that carries the argument peculiar to that task. In the ledgers, 2×SE is twice the standard error of that row's difference, as defined in section 2; under each arm of a strip, “epochs” is the median number of training epochs the arm ran before early stopping.",
   "四个任务,各为一种判定的代表。每个都配三样东西:正文依赖的每个数字的<b>证据表</b>(含样本量、不确定度、来源);四个臂的<b>逐运行分布</b>(每点一个运行——单独 5 个 seed,其余 10 组顺序;粗线为中位数,虚线为贯穿的单独中位数);以及承载该任务独有论证的<b>专属图</b>。证据表中的 2×SE 是该行差值标准误的两倍,定义见第 2 节;分布图每个臂下方的 “epochs” 是该臂早停前训练 epoch 数的中位数。",
   "各判定の代表として4タスクを選んだ。それぞれに3点を添える:本文が依拠する全数値の<b>証拠表</b>(標本数・不確かさ・出典付き)、4アームの<b>実行単位の分布</b>(各点が1実行 — 単独は5シード、他は10通りの順序。太線は中央値、破線は単独の中央値を横に引いたもの)、そのタスク固有の論点を担う<b>専用の図</b>。証拠表の2×SEはその行の差の標準誤差の2倍で、定義は第2節のとおり。分布図の各アームの下の「epochs」は早期終了までに学習したエポック数の中央値。")}</div>
<div class="rep">''')

# zt
out.append(f'''<div><h3><span class="c-warm">{T("Better","更好","改善")}</span> — {T("zt: a real gain that replay had hidden","zt:被 replay 掩盖的真实增益","zt:リプレイに隠されていた本物の改善")}</h3>
<div class="col">{P("zt was one of the six-task probe's original winners. Under xfer it fell to unresolved and was written off. Remove replay and it returns as a resolvable gain in both arms, and the two arms agree to within 0.2%: the shared representation already holds what zt needs, and unfreezing adds nothing.",
"zt 是 6 任务探针最初的赢家之一。在 xfer 下它退成不可分,被当作没有效果。去掉 replay,它在两个臂都以可分辨的增益回归,且两臂相差不到 0.2%:共享表示已经包含 zt 需要的东西,解冻不再增加什么。",
"ztは6タスク探針での当初の勝者の1つだった。xferでは判定不能に落ち、効果なしと片付けられた。リプレイを外すと両アームで判別可能な改善として戻り、両アームの差は0.2%以内:共有表現がztの必要とするものをすでに含んでおり、凍結解除は何も加えない。")}</div>
{TH("zt — every number the text relies on, with n, 2×SE, verdict and source","zt——正文依赖的每个数字,含 n、2×SE、判定与来源","zt — 本文が依拠する全数値、n・2×SE・判定・出典付き")}{zt_l}
<figure>{FH("zt — the score of every run, grouped by arm","zt——每个运行的分数,按臂分组","zt — 実行ごとのスコア、アーム別")}<div class="figbox">{LEG_STRIP}<svg class="repfig" data-task="zt" width="960" height="250" role="img" aria-label="zt: per-run scores for the four arms."></svg></div>
{CAP("Run-level distribution. xfer's median sits above alone but its spread straddles it; both no-replay arms sit clear.","逐运行分布。xfer 的中位数高于单独,但离散范围跨过它;两个无 replay 臂完全在其上方。","実行単位の分布。xferの中央値は単独より上だが散らばりがそれを跨ぐ。リプレイなしの両アームは明確に上にある。")}</figure>
<figure>{FH("zt — the same gain measured four ways, each with its 2×SE interval","zt——同一增益的四种测量,各带 2×SE 区间","zt — 同じ改善の4通りの測定、それぞれ2×SE区間付き")}<div class="figbox"><svg id="fig-zt-arc" width="960" height="230" role="img" aria-label="zt measured four ways, each with its 2SE interval."></svg></div>
{CAP("The same task measured four ways, as change against training alone with 2×SE intervals (both arms' uncertainty). The probe and the two no-replay arms agree; only xfer — the arm with replay — sits back at the zero line. The probe was measuring the no-replay case all along.",
"同一任务的四次测量,以相对单独训练的变化表示,带 2×SE 区间(计两臂不确定度)。探针与两个无 replay 臂一致;只有带 replay 的 xfer 退回零线。探针一直测的就是无 replay 的情形。",
"同じタスクを4通りで測り、単独学習に対する変化として2×SE区間(両アームの不確かさ)付きで示す。探針とリプレイなしの2アームは一致し、リプレイありのxferだけがゼロ線に戻る。探針は最初からリプレイなしの状況を測っていたのだ。")}</figure></div>''')

# material_type
out.append(f'''<div><h3><span class="c-warm">{T("Better, and different in kind","更好,但性质不同","改善、ただし性質が異なる")}</span> — {T("material_type: the one task that wants the encoder left alone","material_type:唯一希望编码器保持不动的任务","material_type:エンコーダに触れてほしくない唯一のタスク")}</h3>
<div class="col">{P("The campaign's most conspicuous result, the only task that was a winner even under xfer, and the only classification task — five classes, 99.25% of rows in one of them. Its gain runs opposite to the extensive properties in every way. Where zt's frozen and warm-start arms agree, material_type's diverge: frozen beats warm-start, the largest &quot;unfreezing hurts&quot; on the board. It owned 32% of the gradient at step 24, the most of any task, so dilution barely touched it.",
"本轮最显眼的结果,唯一在 xfer 下也获益的任务,也是唯一的分类任务——五类,99.25% 的行属于其中一类。它的增益在每个方面都与广延量相反。zt 的冻结与 warm-start 一致,material_type 的则分道扬镳:冻结胜过 warm-start,是全板最大的&quot;解冻有害&quot;。它在第 24 步占 32% 的梯度,为所有任务之最,稀释几乎没碰到它。",
"キャンペーンで最も目立つ結果であり、xferでも勝者だった唯一のタスク、そして唯一の分類タスク — 5クラス、行の99.25%が1クラスに集中。その改善はあらゆる点で示量性物性と逆向きだ。ztの凍結とウォームスタートが一致するのに対し、material_typeでは分岐する:凍結がウォームスタートに勝ち、全体で最大の「凍結解除が有害」となる。第24ステップで勾配の32%を占め、全タスク中最大なので、希釈はほとんど届かなかった。")}</div>
{TH("material_type — every number the text relies on, with n, 2×SE, verdict and source","material_type——正文依赖的每个数字,含 n、2×SE、判定与来源","material_type — 本文が依拠する全数値、n・2×SE・判定・出典付き")}{mt_l}
<figure>{FH("material_type — macro-F1 of every run, grouped by arm","material_type——每个运行的 macro-F1,按臂分组","material_type — 実行ごとの macro-F1、アーム別")}<div class="figbox">{LEG_STRIP}<svg class="repfig" data-task="material_type" width="960" height="250" role="img" aria-label="material_type: per-run macro-F1 for the four arms."></svg></div>
{CAP("Run-level distribution, macro-F1. Note the order: frozen highest, then warm-start, then xfer, then alone — the only task where freezing wins.","逐运行分布,macro-F1。注意顺序:冻结最高,其次 warm-start,再次 xfer,最后单独——唯一冻结获胜的任务。","実行単位の分布、macro-F1。順序に注目:凍結が最高、次にウォームスタート、xfer、単独 — 凍結が勝つ唯一のタスク。")}</figure>
<figure>{FH("material_type — score by position in the training sequence, at its own step and after the sequence ends","material_type——按训练序列位置的分数,在自己那一步与序列结束时","material_type — 学習系列での位置別スコア、自身のステップ時点と系列終了時点")}<div class="figbox"><div class="legend"><span><i class="sw sq" style="background:var(--warm)"></i> {T("at its own step","在自己那一步","自身のステップ時点")}</span><span><i class="sw sq" style="background:var(--xfer)"></i> {T("after the sequence ends","序列结束时","系列終了時点")}</span><span><i class="sw sq" style="background:var(--alone);opacity:.6"></i> {T("alone (5 seeds)","单独(5 seed)","単独(5シード)")}</span>{KEY_DOTS}</div>
<svg id="fig-mt-pos" width="980" height="340" role="img" aria-label="material_type macro-F1 by position band, at its own step and at sequence end, against alone."></svg></div>
{CAP("Position bands from the transfer stage, each run scored on its own test rows: every dot is one run, the bar the median; % labels compare each median with the alone median (0.5761). This is the only task whose score rises the <em>later</em> it appears — the opposite of every regression task — and every band's median, early or late, sits above the best of the 5 alone seeds (0.6035).",
"迁移阶段的位置分段,各运行用自身测试行评分:每点一个运行,粗线为中位数;% 为各中位数相对单独中位数(0.5761)。它是唯一越<em>晚</em>出现分数越高的任务——与所有回归任务相反——而且无论早晚,每一段的中位数都高过 5 个单独 seed 里最好的那个(0.6035)。",
"転移段階の位置帯、各実行は自身のテスト行で評価:各点が1実行、太線が中央値。%は各中央値の単独中央値(0.5761)に対する値。<em>後</em>に出るほどスコアが上がる唯一のタスク — すべての回帰タスクと逆 — であり、早晩を問わずどの帯の中央値も単独5シードの最良値(0.6035)より上にある。")}</figure>
<figure>{FH("material_type — recall and precision of each class, alone vs xfer","material_type——每个类别的 recall 与 precision,单独 vs xfer","material_type — クラスごとの recall と precision、単独 vs xfer")}<div class="figbox"><div class="legend"><span><i class="sw sq" style="background:var(--xfer)"></i> {T("alone","单独","単独")}</span><span><i class="sw sq" style="background:var(--warm)"></i> {T("xfer (multi-task)","xfer(多任务)","xfer(マルチタスク)")}</span></div>
<svg id="fig-mt-prf" width="880" height="290" role="img" aria-label="Per-class recall and precision, alone vs xfer, on matched rows."></svg></div>
{CAP("Where the gain comes from, decomposed on the same 7,354 test rows (5 alone seeds vs 3 xfer orderings). Recall on the four rare classes does not move — IQC even drops — while their precision roughly doubles. The model finds no quasicrystal it was missing; it stops misfiling ordinary material as one. Decomposed for the xfer arm; the frozen arm's larger gain has not been decomposed and is assumed, not shown, to share the mechanism.",
"增益的来源,在同一 7,354 行测试集上分解(5 个单独 seed vs 3 组 xfer 顺序)。四个稀有类的 recall 不动——IQC 反而下降——而它们的 precision 大约翻倍。模型没有找到原本漏掉的准晶,而是不再把普通材料误判成准晶。这是对 xfer 臂的分解;冻结臂更大的增益未做分解,只是假定同一机制,未经展示。",
"改善の出所を同じ7,354テスト行(単独5シード vs xfer 3順序)で分解。希少4クラスのrecallは動かず — IQCはむしろ低下 — precisionはほぼ倍増。モデルは見逃していた準結晶を見つけたのではなく、普通の材料を準結晶と誤分類しなくなった。これはxferアームの分解であり、凍結アームのより大きな改善は分解しておらず、同じ機構だと仮定しているに過ぎない。")}</figure>
<figure>{FH("material_type — confusion matrices, alone vs xfer","material_type——混淆矩阵,单独 vs xfer","material_type — 混同行列、単独 vs xfer")}<div class="figbox"><svg id="fig-mt-cm" width="980" height="400" role="img" aria-label="Row-normalised confusion matrices, alone vs xfer."></svg></div>
{CAP('Rows are true classes, columns predicted; upper number = % of the row, lower = rows per run. The dashed row is "others", the 7,298-row majority: its leak into the four rare classes falls from 1.16% to 0.49%. Because the rare classes hold 1–28 rows each, removing ~48 false positives a run is what doubles their precision. This is a representation of "what ordinary material looks like" — which is what 23 property-regression tasks teach — and fine-tuning the encoder on material_type’s own majority-dominated loss degrades it. So the recipe inverts for this task: <b>keep the encoder, train the head.</b> Whether that holds for other imbalanced classification tasks is untested; there is only one on the board.',
'行为真实类,列为预测类;上方数字 = 占该行的 %,下方 = 每运行行数。虚线框为 "others",7,298 行的多数类:它泄漏进四个稀有类的比例从 1.16% 降到 0.49%。稀有类各只有 1–28 行,每运行少约 48 个误报就足以让 precision 翻倍。这是一种"普通材料长什么样"的表示——正是 23 个性质回归任务教给编码器的——而让编码器按 material_type 自己多数类主导的损失微调会破坏它。所以对这个任务,配方反过来:<b>保留编码器,只训 head。</b>这一点对其他不均衡分类任务是否成立尚未检验;板上只有这一个。',
'行が真のクラス、列が予測。上の数字は行内の%、下は実行あたり行数。破線枠は"others"、7,298行の多数派クラス:希少4クラスへの漏れが1.16%から0.49%に減る。希少クラスは各1〜28行しかないので、実行あたり約48件の偽陽性を除くだけでprecisionが倍になる。これは「普通の材料とはどんなものか」の表現 — 23の物性回帰タスクがエンコーダに教えるもの — であり、material_type自身の多数派支配の損失でエンコーダを微調整するとそれが損なわれる。ゆえにこのタスクではレシピが逆転する:<b>エンコーダは保持し、ヘッドだけ学習。</b>他の不均衡分類タスクでも成り立つかは未検証で、板上には1つしかない。')}</figure></div>''')

# magnetic_moment
out.append(f'''<div><h3><span class="c-alone">{T("Unresolved","不可分","判定不能")}</span> — {T("magnetic_moment: the dilution poster child","magnetic_moment:稀释效应的典型","magnetic_moment:希釈の典型例")}</h3>
<div class="col">{P('851 training rows, the second-smallest real task. In xfer it owned 1.1% of every epoch and was cut off at a median 56 epochs against 142 alone; it read as clear negative transfer. Warm-start brings it to indistinguishable from alone. The "negative transfer" came from the training setup, not from the task; what remains is a task the encoder neither helps nor hurts.',
'851 行训练数据,第二小的实质任务。在 xfer 里它每个 epoch 只占 1.1%,并在中位 56 个 epoch 时被掐断,而单独训练要 142 个;它读起来是明确的负迁移。warm-start 把它带回与单独训练不可分。"负迁移"来自训练设置,而不是任务本身;剩下的是一个编码器既不帮也不害的任务。',
'訓練行851、実質的に2番目に小さいタスク。xferでは各エポックの1.1%しか占めず、単独なら142エポックのところ中央値56エポックで打ち切られ、明確な負の転移に見えた。ウォームスタートで単独と区別できない水準に戻る。「負の転移」は学習設定によるものでタスク自体のものではなく、残るのはエンコーダが助けも害もしないタスクである。')}</div>
{TH("magnetic_moment — every number the text relies on, with n, 2×SE, verdict and source","magnetic_moment——正文依赖的每个数字,含 n、2×SE、判定与来源","magnetic_moment — 本文が依拠する全数値、n・2×SE・判定・出典付き")}{mm_l}
<figure>{FH("magnetic_moment — the score of every run, grouped by arm","magnetic_moment——每个运行的分数,按臂分组","magnetic_moment — 実行ごとのスコア、アーム別")}<div class="figbox">{LEG_STRIP}<svg class="repfig" data-task="magnetic_moment" width="960" height="250" role="img" aria-label="magnetic_moment: per-run scores for the four arms."></svg></div>
{CAP("Run-level distribution. xfer and frozen sit below alone; warm-start's median lands on the alone line with its spread across it — unresolved, not &quot;worse&quot;.","逐运行分布。xfer 与冻结低于单独;warm-start 的中位数落在单独线上、离散跨过它——是不可分,不是&quot;更差&quot;。","実行単位の分布。xferと凍結は単独より下。ウォームスタートの中央値は単独の線上にあり散らばりがそれを跨ぐ — 「悪化」ではなく判定不能。")}</figure>
<figure>{FH("magnetic_moment — score at each of the 24 positions in the transfer stage","magnetic_moment——迁移阶段 24 个位置上各自的分数","magnetic_moment — 転移段階の24位置それぞれでのスコア")}<div class="figbox">{LEG_POS}<svg class="posfig" data-task="magnetic_moment" width="980" height="330" role="img" aria-label="magnetic_moment score at each sequence position, xfer at its own step."></svg></div>
{CAP("The transfer stage's position curve: the task's score at the step where it was trained, at each of the 24 positions — every dot one run, the bar the median at that position — against the alone median (dashed) and the range of the 5 alone seeds (shaded). Unhurt at early positions, hurt only late: the more tasks precede it — the more replay it competes with at its own step — the worse it does. That is the dilution signature, and the reason the −6.3% at position 24 belongs to the training setup rather than to the task.",
"迁移阶段的位置曲线:该任务在被训练那一步的分数,24 个位置逐个显示——每点一个运行,粗线为该位置的中位数——对照单独训练的中位数(虚线)和 5 个 seed 的范围(阴影)。早期位置不吃亏,晚期才受损:前面的任务越多——在自己那一步要对抗的 replay 越多——表现越差。这就是稀释的指纹,也是第 24 位那个 −6.3% 属于训练设置而非任务本身的原因。",
"転移段階の位置曲線:学習されたステップでのスコアを24位置それぞれに示す — 各点が1実行、太線がその位置の中央値 — 単独の中央値(破線)と5シードの範囲(網掛け)と比較。序盤の位置では損なわれず、終盤でのみ悪化:先行タスクが多いほど — 自身のステップで競合するリプレイが多いほど — 悪くなる。これが希釈の特徴であり、24位置での−6.3%がタスクではなく学習設定の性質である理由だ。")}</figure></div>''')

# final_energy
out.append(f'''<div><h3><span class="c-xfer">{T("Worse","更差","悪化")}</span> — {T("final_energy: two mechanisms, each owning a share","final_energy:两个机制,各占一份","final_energy:2つの機構がそれぞれ一部を担う")}</h3>
<div class="col">{P("The largest loss of the campaign, and not a dilution story: with 23,678 rows final_energy owned 24.4% of the gradient at step 24, yet lost heavily. Removing replay recovers most of it and unfreezing the encoder is worth the biggest unfreezing gain on the board — but it stays resolvably worse than alone. A budget check rules out undertraining: with early stopping off and 500 epochs for both arms, warm-start stays 10.0% below training alone. Its validation loss bottoms out at epoch 143 and drifts up, while its training loss ends lower than the single-task model’s (0.011 vs 0.015): it fits the training set better and generalises worse.",
"本轮最大的损失,而且不是稀释的故事:final_energy 有 23,678 行,在第 24 步占 24.4% 的梯度,却仍大幅受损。去掉 replay 恢复了大部分,解冻编码器带来全板最大的解冻收益——但它仍可分辨地差于单独训练。预算检验排除了训练不足:两臂都关掉早停跑满 500 epoch 后,warm-start 仍比单独训练低 10.0%。它的验证损失在第 143 epoch 触底后回升,而训练损失最终比单任务模型更低(0.011 vs 0.015):拟合训练集更好,泛化更差。",
"キャンペーン最大の損失であり、希釈の話ではない:23,678行のfinal_energyは第24ステップで勾配の24.4%を占めながら大きく損なわれた。リプレイ除去で大半が回復し、凍結解除は全体で最大の凍結解除効果をもたらす — それでも単独より判別可能なほど悪い。予算検証で学習不足は否定された:両アームとも早期終了を切って500エポック回しても、ウォームスタートは単独学習より10.0%低いまま。検証損失は第143エポックで底を打って上昇に転じ、学習損失は単独モデルより低く終わる(0.011 vs 0.015):訓練データへの適合は良く、汎化は悪い。")}</div>
{TH("final_energy — every number the text relies on, with n, 2×SE, verdict and source","final_energy——正文依赖的每个数字,含 n、2×SE、判定与来源","final_energy — 本文が依拠する全数値、n・2×SE・判定・出典付き")}{fe_l}
<figure>{FH("final_energy — the score of every run, grouped by arm","final_energy——每个运行的分数,按臂分组","final_energy — 実行ごとのスコア、アーム別")}<div class="figbox">{LEG_STRIP}<svg class="repfig" data-task="final_energy" width="960" height="250" role="img" aria-label="final_energy: per-run scores for the four arms."></svg></div>
{CAP("Run-level distribution. Every arm sits below alone; warm-start closes most of the gap but not all of it, with no overlap of the alone spread.","逐运行分布。每个臂都低于单独;warm-start 收窄了大部分差距但没有全部,与单独的离散范围没有重叠。","実行単位の分布。全アームが単独より下。ウォームスタートは差の大半を埋めるが全部ではなく、単独の散らばりとは重ならない。")}</figure>
<figure>{FH("final_energy — score at each of the 24 positions in the transfer stage","final_energy——迁移阶段 24 个位置上各自的分数","final_energy — 転移段階の24位置それぞれでのスコア")}<div class="figbox">{LEG_POS}<svg class="posfig" data-task="final_energy" width="980" height="330" role="img" aria-label="final_energy score at each sequence position, xfer at its own step."></svg></div>
{CAP("Position curve. Unlike magnetic_moment it is hurt at <em>every</em> position — −13.7% even in the first eight slots, where replay is light — and gets worse the later it sits. A task with 24% of the gradient is not being drowned; something about the shared representation itself costs it.",
"位置曲线。与 magnetic_moment 不同,它在<em>每个</em>位置都受损——前八个位置 replay 很轻,也已 −13.7%——而且越靠后越差。占 24% 梯度的任务不是被淹没了;是共享表示本身有什么东西让它付出代价。",
"位置曲線。magnetic_momentと異なり<em>すべての</em>位置で損なわれる — リプレイが軽い最初の8位置でも−13.7% — そして後ろほど悪化。勾配の24%を持つタスクが溺れているのではなく、共有表現そのものの何かが代償を課している。")}</figure>
<figure>{FH("Warm-start vs alone for all tasks, coloured by the kind of quantity","所有任务的 warm-start vs 单独,按物理量类型着色","全タスクのウォームスタート vs 単独、物理量の種類で色分け")}<div class="figbox"><div class="legend"><span><i class="sw" style="background:var(--xfer)"></i> {T("extensive property","广延量","示量性物性")}</span><span><i class="sw" style="background:var(--frz)"></i> {T("intensive property","强度量","示強性物性")}</span><span><i class="sw" style="background:var(--warm)"></i> {T("classification","分类","分類")}</span><span style="color:var(--muted)">{T("filled = separated at 2×SE and |Δ| ≥ 0.01","实心 = 2×SE 分离且 |Δ| ≥ 0.01","塗りつぶし = 2×SEで分離かつ|Δ| ≥ 0.01")}</span></div>
<svg id="fig-kind" width="960" height="560" role="img" aria-label="Warm-start change against alone for 23 tasks, coloured by whether the property is extensive, intensive or a classification."></svg></div>
{CAP("Warm-start vs alone across the board, coloured by the kind of quantity. The three extensive properties — those that scale with the size of the cell — are the three largest resolvable losses. Intensive properties cluster around zero; the one classification task is the largest gain. magnetic_susceptibility (58 rows) is omitted as off-scale.",
"全板的 warm-start vs 单独,按物理量类型着色。三个广延量——随原胞大小变化的量——正是三个最大的可分辨损失。强度量聚在零附近;唯一的分类任务是最大的增益。magnetic_susceptibility(58 行)超出量程,未画出。",
"全タスクのウォームスタート vs 単独を物理量の種類で色分け。3つの示量性物性 — 単位胞のサイズに比例する量 — が判別可能な損失の上位3つである。示強性物性はゼロ付近に集まり、唯一の分類タスクが最大の改善。magnetic_susceptibility(58行)は範囲外のため省略。")}</figure>
{TH("Three formulas that are one and the same input to the descriptor, at different cell sizes","三个对描述符而言完全相同的输入,原胞大小却不同","記述子にとっては同一の入力である3つの化学式、単位胞サイズは異なる")}<div class="tablebox" style="max-width:640px"><table><thead><tr><th>{T("Reduced formula","约化化学式","既約化学式")}</th><th>{T("Atoms per cell","原胞原子数","単位胞の原子数")}</th><th>{T("Volume","体积","体積")}</th></tr></thead>
<tbody><tr><td>AgSO<sub>4</sub></td><td>12 / 48</td><td>162.0 / 616.1</td></tr><tr><td>U(PO<sub>3</sub>)<sub>4</sub></td><td>34 / 136</td><td>450.1 / 1929.9</td></tr><tr><td>Ba(FeAs)<sub>2</sub></td><td>5 / 10</td><td>98.0 / 216.5</td></tr></tbody></table></div>
<div class="col" style="margin-top:14px">{P("Why extensive properties: the descriptor returns atomic fractions, so Fe<sub>2</sub>O<sub>3</sub> and Fe<sub>4</sub>O<sub>6</sub> are identical to it — verified in code — and the three formulas above are the same input with a 4× difference in volume. Cell scale is not in the model's input at all. A single-task model recovers it from within-dataset composition-to-size correlations (corr(Volume, atoms) = +0.868); a shared encoder, pulled toward what 24 tasks have in common, keeps the scale-free part and loses that correlation. Replay explains the first two thirds of final_energy's loss; scale-blindness the rest. volume and dos_density follow the same pattern. This is the one part of the picture the replay mechanism does not explain, and the next experiment on the list.",
"为什么是广延量:描述符返回原子分数,所以 Fe<sub>2</sub>O<sub>3</sub> 和 Fe<sub>4</sub>O<sub>6</sub> 对它完全相同——代码层面已验证——上表三个化学式是同一输入、体积相差 4 倍。原胞尺度根本不在模型输入里。单任务模型能从数据集内的组成–尺寸相关性把它推回来(corr(体积, 原子数) = +0.868);共享编码器被拉向 24 个任务的共性,保留了尺度无关的部分,丢掉了那条相关性。replay 解释了 final_energy 损失的前三分之二,尺度盲解释其余。volume 和 dos_density 遵循同一模式。这是 replay 机制解释不了的那一部分,也是清单上的下一个实验。",
"なぜ示量性物性か:記述子は原子分率を返すので、Fe<sub>2</sub>O<sub>3</sub>とFe<sub>4</sub>O<sub>6</sub>は同一 — コード上で検証済み — であり、上表の3つの化学式は同じ入力で体積が4倍違う。単位胞のスケールはモデル入力に一切含まれない。単独モデルはデータセット内の組成–サイズ相関(corr(体積, 原子数) = +0.868)からそれを復元するが、24タスクの共通部分へ引き寄せられた共有エンコーダはスケール非依存の部分だけを残し、その相関を失う。final_energyの損失の最初の3分の2はリプレイが、残りはスケール盲が説明する。volumeとdos_densityも同じパターン。これはリプレイ機構が説明できない唯一の部分であり、リストの次の実験である。")}</div>
</div></div></section>''')

# ---- 5 summary + plan ----
out.append(f'''<section class="section">
<div class="col"><p class="kicker">{T("5 · Where this leaves us","5 · 现在的位置","5 · 現在地")}</p><h2>{T("Conclusions so far","目前的结论","現時点の結論")}</h2></div>
<div class="col">
{P('<b>xfer — adding the task last, with replay on — is the weakest of the four ways to attach a task.</b> "18 of 24 worse" holds for that setup only. Starting from the same encoder with replay off, 19 tasks recover and none get worse.',
   '<b>xfer(带 replay、排末位加入)是四种接入方式里效果最差的一种。</b>"24 个中 18 个更差"只对这一种设置成立。从同一编码器出发、关掉 replay,19 个任务回升、0 个变差。',
   '<b>xfer(リプレイありで最後に追加)は4通りのつなぎ方の中で最も効果が低い。</b>「24中18が悪化」はこの設定にだけ当てはまる。同じエンコーダからリプレイなしで始めると、19タスクが回復し悪化はゼロ。',"callout")}
{P('<b>Warm-starting from the 24-task encoder is break-even against training alone</b> — 4 better, 4 worse, 14 unresolved. The gains are real (material_type +21.9%, zt +6.7%, magnetization +4.9%, dielectric_total +3.4%) and so are the losses (final_energy −9.1%, volume −6.9%, dos_density −4.2%).',
   '<b>从 24 任务编码器 warm-start 与单独训练大体持平</b>——4 好、4 差、14 不可分。增益是真的(material_type +21.9%、zt +6.7%、magnetization +4.9%、dielectric_total +3.4%),损失也是真的(final_energy −9.1%、volume −6.9%、dos_density −4.2%)。',
   '<b>24タスクエンコーダからのウォームスタートは単独学習と互角</b> — 改善4、悪化4、判定不能14。改善は本物(material_type +21.9%、zt +6.7%、magnetization +4.9%、dielectric_total +3.4%)で、損失も本物(final_energy −9.1%、volume −6.9%、dos_density −4.2%)。',"callout")}
{P('<b>Unfreezing the encoder helps 13 tasks and hurts 2</b>, and helps most exactly where the frozen representation was weakest. The frozen encoder is not a drop-in feature extractor. material_type is the one exception: it wants the shared representation left alone.',
   '<b>解冻编码器对 13 个任务有帮助、2 个有害</b>,而且帮得最多的正是冻结表示最弱的地方。冻结编码器不是即插即用的特征提取器。material_type 是唯一的例外:它希望共享表示保持不动。',
   '<b>凍結解除は13タスクを改善し2タスクを悪化させる</b>。最も効くのは凍結表現が最も弱かった所だ。凍結エンコーダはそのまま使える特徴抽出器ではない。material_typeが唯一の例外で、共有表現に触れてほしくない。',"callout")}
{P('<b>The original puzzle is answered.</b> material_type was the only xfer winner because it is the largest task (32% of the gradient at step 24, the least diluted) and because what it needs — a good model of ordinary materials — is what the other 23 tasks teach the encoder. The other 23 were not unsuited to transfer; they were drowned.',
   '<b>最初的谜题有了答案。</b>material_type 是唯一的 xfer 赢家,因为它是最大的任务(第 24 步占 32% 梯度,稀释最轻),也因为它需要的——对普通材料的良好刻画——正是其他 23 个任务教给编码器的。其余 23 个并非不适合迁移,它们是被淹没了。',
   '<b>当初の謎は解けた。</b>material_typeが唯一のxfer勝者だったのは、最大のタスク(第24ステップで勾配の32%、最も希釈されにくい)であり、必要とするもの — 普通の材料の良いモデル — が他の23タスクがエンコーダに教えるものだったからだ。他の23は転移に不向きだったのではなく、溺れていた。',"callout")}
{P('<b>The dilution is itself a finding, not only an artefact.</b> Some properties do gain from the shared encoder, but how much of the training signal the new task owns decides whether any of that gain is visible: with 1% of the gradient at step 24, a task loses however good the representation is. The volume of the task’s own training data remains the first-order factor; transfer is second-order.',
   '<b>稀释本身就是一个重要发现,不只是干扰。</b>有些性质确实从共享编码器获益,但新任务在训练信号里占多大份额,决定了这种获益能否显现:在第 24 步只占 1% 梯度的任务,无论表示多好都会受损。任务自身的训练数据量仍是第一位的因素,迁移是第二位的。',
   '<b>希釈そのものが重要な知見であり、単なる副作用ではない。</b>一部の物性は確かに共有エンコーダから利益を得るが、新タスクが学習信号のどれだけを占めるかがその利益が見えるかどうかを決める:第24ステップで勾配の1%しか持たないタスクは、表現がどれほど良くても悪化する。タスク自身の学習データ量が依然として第一の要因であり、転移は第二の要因である。',"callout")}
</div>
<div class="col" style="margin-top:28px"><h3>{T("The four combinations, and what each answered","四种组合,以及各自回答了什么","4つの組み合わせと、それぞれが答えたこと")}</h3>
{P("Fixing the pretraining and varying how X is attached gives a 2×2. All four cells are done. The two unseen cells re-ran the first 23 steps of three transfer orderings with X dropped, same seed, to reproduce the step-23 encoder the pruner discarded — an encoder that has never seen X — and fine-tuned it with a fresh head.",
   "固定预训练、改变 X 的接入方式,得到一个 2×2。四格都已完成。两个未见过的格子把三个迁移顺序的前 23 步去掉 X 后同 seed 重跑,复现被裁剪器删掉的第 23 步编码器——一个从未见过 X 的编码器——再用新建的 head 微调。",
   "事前学習を固定しXのつなぎ方を変えると2×2になる。4マスすべて完了。未知の2マスは、3つの転移順序の最初の23ステップをXを除いて同じシードで再実行し、プルーナが捨てた第23ステップのエンコーダ — Xを一度も見ていないエンコーダ — を再現して、新規ヘッドで微調整した。")}</div>
{TH("The 2×2 — whether the encoder saw X, by frozen or warm-start: better / worse / unresolved against alone","2×2——编码器是否见过 X,乘以冻结或 warm-start:相对单独的 好 / 差 / 不可分","2×2 — エンコーダがXを見たか × 凍結かウォームスタートか:単独に対する 改善 / 悪化 / 判定不能")}<div class="matrix">
<div class="h"></div><div class="h">{T("frozen encoder","冻结编码器","エンコーダ凍結")}</div><div class="h">{T("encoder + head","编码器 + head","エンコーダ + ヘッド")}</div>
<div class="h">{T("encoder saw X once","编码器见过 X 一次","エンコーダはXを一度見た")}</div><div class="done">{T("done · n=10 · 4 / 12 / 7","完成 · n=10 · 4 / 12 / 7","完了 · n=10 · 4 / 12 / 7")}</div><div class="done">{T("done · n=10 · 4 / 4 / 14","完成 · n=10 · 4 / 4 / 14","完了 · n=10 · 4 / 4 / 14")}</div>
<div class="h">{T("encoder never saw X","编码器从未见过 X","エンコーダはXを見ていない")}</div>{cell_u("ftzu_vs_single")}{cell_u("ftfu_vs_single")}
</div>
<div class="col">{P('<b>Answered.</b> With the encoder unfrozen, the single exposure at step 24 makes no measurable difference: warm-start from an encoder that never saw X is 2 better / 4 worse / 15 unresolved against training alone, and against the seen-once warm-start it is 2 better / 2 worse / 19 unresolved over 23 tasks. Frozen, the exposure does help — 4 tasks better, none worse — because a head trained at step 24 is better than a fresh one on a fixed representation. material_type gains +21.5% from an encoder that never saw it, so its gain is the 23-task representation, not exposure.',
   '<b>已回答。</b>编码器解冻时,第 24 步那一次接触没有可测量的影响:从未见过 X 的编码器 warm-start 对比单独训练是 2 好 / 4 差 / 15 不可分,对比见过一次的 warm-start 在 23 个任务上是 2 好 / 2 差 / 19 不可分。冻结时那次接触确实有用——4 个任务更好、0 个更差——因为在固定的表示上,第 24 步训出的 head 好于新建的 head。material_type 从一个从未见过它的编码器上获得 +21.5%,所以它的增益来自 23 任务的表示,不是来自接触。',
   '<b>答えが出た。</b>エンコーダを解凍する場合、第24ステップでの一度の接触に測定可能な違いはない:Xを一度も見ていないエンコーダからのウォームスタートは単独学習に対して 2 改善 / 4 悪化 / 15 判定不能、既知のウォームスタートに対しては23タスクで 2 改善 / 2 悪化 / 19 判定不能。凍結時には接触が効く — 4 タスク改善、悪化ゼロ — 固定表現の上では第24ステップで学習したヘッドが新規ヘッドより良いからだ。material_type は一度も見ていないエンコーダから +21.5% を得るので、その改善は23タスクの表現によるもので、接触によるものではない。', "callout")}
{P('<b>The recipe simplifies.</b> For a new task, pretrain on the existing tasks, then warm-start fine-tune the new one: the continual step with replay adds nothing the fine-tune does not recover, and costs the most. Warm-start is the transfer method carried into the next phase, where pretraining length and order are varied under it.',
   '<b>配方简化了。</b>对新任务:在已有任务上预训练,再对新任务 warm-start 微调;带 replay 的连续预训练步没有带来微调补不回来的东西,却最耗时。warm-start 作为迁移方式进入下一阶段,在它之下再改变预训练的长度和顺序。',
   '<b>レシピは単純化される。</b>新タスクには、既存タスクで事前学習し、新タスクをウォームスタート微調整する。リプレイ付きの継続ステップは微調整で取り戻せないものを何も加えず、最も高くつく。ウォームスタートを転移方式として次段階へ持ち越し、その下で事前学習の長さと順序を変える。', "callout")}
<h3 style="margin-top:22px">{T("Next","下一步","次のステップ")}</h3>
<ol class="plan">
<li>{T("<b>Move to the 2026-09-11 dataset first.</b> final_energy and volume — two of the four warm-start losers — turned out to be a label and a descriptor problem (see the companion page); their rows above are void, and the baselines for every relabelled and added task already exist on the new data.","<b>先切换到 2026-09-11 数据集。</b>warm-start 四个受损任务里的 final_energy 和 volume 已证明是标签和描述符的问题(见配套页面);上表中它们的行作废,新数据上所有标签更新和新增任务的基线已就绪。","<b>まず 2026-09-11 データセットへ移る。</b>ウォームスタートで悪化した4タスクのうち final_energy と volume はラベルと記述子の問題だった(併設ページ参照)。上表のその行は無効で、新データでのラベル更新・新規タスクの基準はすでにある。")}</li>
<li>{T("<b>Fix warm-start as the method, vary the pretraining</b> — length (encoders truncated at 4, 8, 12, 16, 23 tasks) and order (the 10 random orderings). This is the transferability-vs-pretraining question the first measurement tried to answer, redone under a transfer method that does not drown the task.","<b>固定 warm-start,改变预训练</b>——长度(在 4、8、12、16、23 个任务处截断的编码器)和顺序(10 组随机顺序)。这是最初测量想回答的&quot;迁移性 vs 预训练&quot;问题,在一个不会淹没任务的迁移方式下重做。","<b>ウォームスタートを固定し、事前学習を変える</b> — 長さ(4、8、12、16、23タスクで打ち切ったエンコーダ)と順序(10通りのランダム順序)。最初の測定が答えようとした「転移性 vs 事前学習」の問いを、タスクを溺れさせない転移方式でやり直す。")}</li>
<li>{T("<b>Decide the descriptor policy.</b> The companion page shows KMD is scale-blind: volume trains to R² 0.997 with the XenonPy classic descriptor and 0.979 as volume per atom with KMD, against 0.62 as cell volume. Per-atom targets or a descriptor that carries cell scale — pick one before the phase-B runs.","<b>决定描述符策略。</b>配套页面表明 KMD 对晶胞尺度不敏感:volume 用 XenonPy classic 描述符可训到 R² 0.997,用 KMD 以每原子体积训到 0.979,而以晶胞体积只有 0.62。每原子目标或带尺度信息的描述符——在 phase-B 之前二选一。","<b>記述子の方針を決める。</b>併設ページが示す通り KMD は格子スケールに盲目:volume は XenonPy classic 記述子で R² 0.997、KMD で原子あたり体積なら 0.979、セル体積のままでは 0.62。原子あたりの目標か、スケールを持つ記述子か — phase-B の前に決める。")}</li>
<li>{T("<b>A task outside the 24, at low data volume</b> — the library's real use case, still unmeasured.","<b>24 个之外的任务、低数据量</b>——模型库真正的用途,仍未测量。","<b>24タスク外のタスク、低データ量で</b> — ライブラリの本来の用途、まだ未測定。")}</li>
</ol></div>
<div class="col caveats" style="margin-top:28px"><h3>{T("What to hold loosely","需要保留态度的几点","留保すべき点")}</h3><ul>
<li>{T('<b>"Seen once" is two things at once</b> — the encoder met X under replay, and X’s head was trained there. The unseen arms use a fresh head; a seen / unseen difference bundles both.','<b>"见过一次"同时是两件事</b>——编码器在 replay 下接触过 X,X 的 head 也是在那时训的。未见过的两臂用新建 head;见过/未见过的差值把两者捆在一起。','<b>「一度見た」は2つのことを同時に含む</b> — エンコーダがリプレイ下でXに接し、Xのヘッドもそこで学習された。未知アームは新規ヘッドを使うので、既知/未知の差は両方を束ねている。')}</li>
<li>{T("<b>Every arm reports last-epoch weights</b>, not best-epoch (checkpointing is off). Identical across arms, so comparisons stand, but every absolute value is 24 epochs past its best.","<b>每个臂报告的都是最后一个 epoch 的权重</b>,不是最优 epoch(检查点关闭)。各臂一致,比较成立,但每个绝对值都是“最优之后又跑了 24 个 epoch”的状态。","<b>全アームが最終エポックの重み</b>を報告し、最良エポックではない(チェックポイントは無効)。アーム間で同一なので比較は成り立つが、絶対値はどれも最良から24エポック後の値。")}</li>
<li>{T("<b>Baselines and both fine-tune arms were audited for convergence.</b> seebeck and power_factor hit the 150-epoch cap everywhere and were rerun at 400 in every arm; the ceilings moved by +0.0006 and +0.0071 and no verdict changed.","<b>基线和两个微调臂都做了收敛审计。</b>seebeck 与 power_factor 在所有臂都撞到 150 epoch 上限,已在每个臂以 400 重跑;天花板移动 +0.0006 和 +0.0071,没有任何判定改变。","<b>基準と両微調整アームは収束を監査済み。</b>seebeckとpower_factorは全アームで150エポックの上限に達し、全アームで400で再実行。天井は+0.0006と+0.0071動き、判定は一つも変わらなかった。")}</li>
<li>{T("<b>The unseen arms are n=3</b> against n=10 elsewhere. Read their unresolved column generously.","<b>未见过的两臂是 n=3</b>,其余是 n=10。它们的&quot;不可分&quot;一列要宽松地读。","<b>未知アームはn=3</b>で、他はn=10。その判定不能の列は寛大に読むこと。")}</li>
</ul></div>
</section>
<footer><p>{T("Transfer stage: 24 tasks × 10 shuffled orderings, task under test last, hybrid replay max(1500, 0.3N). Fine-tune stage: 480 runs + 40 reruns, fm finetune, 150-epoch cap with patience 24. Single-task baselines: 5 seeds.","迁移阶段:24 任务 × 10 组打乱顺序,待测任务排末位,混合 replay max(1500, 0.3N)。微调阶段:480 个运行 + 40 个补跑,fm finetune,150 epoch 上限、patience 24。单任务基线:5 seed。","転移段階:24タスク × 10通りのシャッフル順序、被検タスクを最後に、ハイブリッドリプレイ max(1500, 0.3N)。微調整段階:480実行 + 40再実行、fm finetune、150エポック上限・patience 24。単独基準:5シード。")}</p>
<p>{T("2×SE: twice the standard error of the difference between the two arms compared. Metric: macro-F1 for material_type, R² otherwise. Figure labels stay in English in every language. The unseen arms: stage_xu (72 pretraining runs) → ftzu / ftfu (144 fine-tunes), n = 3 orderings per task.","2×SE:所比较两臂差值标准误的两倍。指标:material_type 用 macro-F1,其余用 R²。图内标签在各语言下均保留英文。未见过的两臂:stage_xu(72 个预训练运行)→ ftzu / ftfu(144 个微调),每任务 3 组顺序。","2×SE:比較する2アームの差の標準誤差の2倍。指標:material_typeはmacro-F1、他はR²。図中のラベルはどの言語でも英語のまま。未知アーム:stage_xu(72の事前学習実行)→ ftzu / ftfu(144の微調整)、タスクごとに3順序。")}</p></footer>
</div>
<script>
// language switch: one attribute on the root, remembered per viewer
(function(){{const root=document.documentElement,btns=document.querySelectorAll(".langbar button");
function set(l){{root.dataset.lang=l;btns.forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.lang===l)));try{{localStorage.setItem("lang",l);}}catch(e){{}}}}
btns.forEach(b=>b.addEventListener("click",()=>set(b.dataset.lang)));
let l="en";try{{l=localStorage.getItem("lang")||"en";}}catch(e){{}} set(l);}})();
</script>
<script>''')
out.append(POSRUNS_JS); out.append(JS)
out.append("</script>\n")
OUT.write_text("".join(out), encoding="utf-8")
print(f"  wrote {OUT} ({len(''.join(out))//1024} KB)")
