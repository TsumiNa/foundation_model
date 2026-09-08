# -*- coding: utf-8 -*-
"""Generate "Two ceilings, two causes" — summary_page/mp_labels_summary.html

The page records why final_energy (R² 0.77) and volume (0.62) sat far below every other Materials
Project task, with the evidence for each cause, what it voids in the transfer conclusions, the standard
the rebuilt dataset follows, and the update brief for the 2026-09-12 dataset. English is the source
text; Chinese and Japanese follow it sentence by sentence behind the switch. Figures are drawn by
labels_page.js from summary/mp_labels_page_data.json; the stylesheet and language switch are shared
with the transferability page (extracted into summary_page/_shared_*.html/js).

    python summary_page/gen_labels_page.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
D = json.load(open(EXP / "summary" / "mp_labels_page_data.json"))
STYLE = (HERE / "_shared_style.html").read_text(encoding="utf-8")
LANGBAR = (HERE / "_shared_langbar.html").read_text(encoding="utf-8")
LANGJS = (HERE / "_shared_lang.js").read_text(encoding="utf-8")
JS = (HERE / "labels_page.js").read_text(encoding="utf-8")
OUT = HERE / "mp_labels_summary.html"


def T(en, zh, ja):
    return f'<span class="l-en">{en}</span><span class="l-zh">{zh}</span><span class="l-ja">{ja}</span>'


def P(en, zh, ja, cls=""):
    return f'<p class="{cls}">{T(en, zh, ja)}</p>' if cls else f"<p>{T(en, zh, ja)}</p>"


def CAP(en, zh, ja):
    return f"<figcaption>{T(en, zh, ja)}</figcaption>"


def H2(kick, en, zh, ja):
    return f'<div class="col"><p class="kicker">{kick}</p><h2>{T(en, zh, ja)}</h2></div>'


def table(head, rows, cls="tablebox", style=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="{cls}"{(" style=" + chr(34) + style + chr(34)) if style else ""}><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


R = D["runs"]; L = D["local"]; C6 = D["sample600"]; K = D["consistency"]; DS = D["descriptor"]; MG = L["mixed_minus_gga"]
def m(v): return sum(v) / len(v)
def sd(v): mu = m(v); return (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5
fe_old, fe_new = R["final_energy_old_kmd"], R["final_energy_new_kmd"]
vol_kmd, vol_c, vol_n = R["volume_old_kmd"], R["volume_xenonpy_classic"], R["volume_xenonpy_nosum"]
pos = lambda v: f'<td class="pos">{v}</td>'
neg = lambda v: f'<td class="neg">{v}</td>'

out = []
out.append("<title>Two Ceilings, Two Causes</title>\n" + STYLE + "\n<style>.tablebox.wrap td{white-space:normal;font-family:var(--f-body);text-align:left;vertical-align:top;color:var(--ink-2)}.tablebox.wrap td:first-child{color:var(--ink)}</style>\n" + LANGBAR + '<div class="wrap">')

# ---------------- header ----------------
out.append(f'''<header>
<p class="eyebrow">{T("Continual multi-task pretraining · Materials Project labels · findings and data update, 2026-09-12","连续多任务预训练 · Materials Project 标签 · 发现与数据更新,2026-09-12","継続的マルチタスク事前学習 · Materials Project ラベル · 知見とデータ更新、2026-09-12")}</p>
<h1>{T("Two easy tasks that would not train","两个本该容易、却训不好的任务","本来容易なのに学習できなかった2つのタスク")}</h1>
<p class="standfirst">{T(
"Final energy and volume are properties that earlier models predicted with correlation ≈ 0.99. On this dataset, with the KMD descriptor and the campaign's recipe, they trained to R² 0.77 and 0.62 and were read as limits of the model or of transfer. The question was why. The answer is two different causes: final_energy's label mixes two DFT energy references, and volume's label depends on a cell size the descriptor cannot see. With the label fixed and nothing else changed, final_energy reaches 0.999; with cell scale in the input, volume reaches 0.997. Because the first cause was misaligned data, the Materials Project part of the dataset was audited and rebuilt on one level of theory, and the additional MP properties that audit made available were added. This page gives the evidence, what it voids, the standard the data now follows, and the update.",
"Final energy 和 volume 是早先的模型能预测到相关系数 ≈ 0.99 的性质。在这个数据集上,用 KMD 描述符和本轮配方,它们只训到 R² 0.77 和 0.62,并被解读为模型或迁移的极限。问题是为什么。答案是两个不同的原因:final_energy 的标签混合了两种 DFT 能量参考;volume 的标签依赖描述符看不见的原胞大小。只修标签、其余不动,final_energy 到 0.999;输入里有原胞尺度,volume 到 0.997。由于第一个原因是数据对不齐,数据集的 Materials Project 部分被全面核查并按单一理论水平重建,核查过程中发现可用的其他 MP 属性也一并补入。本页给出证据、由此作废的结论、数据现在遵循的标准,以及更新内容。",
"Final energy と volume は、以前のモデルが相関 ≈ 0.99 で予測できていた物性である。このデータセットでは、KMD 記述子とキャンペーンのレシピで R² 0.77 と 0.62 にしか達せず、モデルや転移の限界と読まれていた。問いは「なぜか」。答えは2つの異なる原因:final_energy のラベルは2種類の DFT エネルギー基準を混ぜており、volume のラベルは記述子に見えない単位胞の大きさに依存する。ラベルを直すだけで final_energy は 0.999 に、入力に単位胞スケールがあれば volume は 0.997 に達する。第一の原因がデータの不整合だったため、データセットの Materials Project 部分を全面的に監査し単一の理論レベルで再構築し、監査で利用可能になった他の MP 物性も追加した。本ページは証拠、無効になる結論、データが今後従う基準、そして更新内容を示す。")}</p>
</header>''')

# ---------------- 1 symptom ----------------
rows = [[t.replace("_", " "), f"{n:,}", f"{r:.4f}", f"{s:.4f}"] for t, n, r, s in D["ceilings"]]
out.append(f'''<section class="section">{H2("1 · The symptom", "Six MP tasks, one recipe, two outliers", "六个 MP 任务、同一配方、两个异常值", "MP の6タスク、同一レシピ、2つの外れ値")}
<div class="col">{P("Every task in the campaign is trained alone as its baseline: the same descriptor (KMD), the same adopted hyper-parameters, five seeds, early stopping. The Materials Project tasks with about 23,000 training rows split cleanly: four train to R² 0.91–0.99, two do not — and those two are properties that earlier models, on XenonPy composition descriptors, had predicted with correlation ≈ 0.99. Neither a model nor a data-volume problem explains that; the cause had to be in what the model was given.",
"campaign 里每个任务都先单独训练作为基线:同一描述符(KMD)、同一套采用的超参数、5 个 seed、早停。训练行数约 23,000 的 Materials Project 任务分成两群:四个训到 R² 0.91–0.99,两个不行——而这两个正是早先用 XenonPy 组成描述符的模型能预测到相关系数 ≈ 0.99 的性质。模型或数据量都解释不了,原因只能在喂给模型的东西里。",
"キャンペーンの各タスクはまず単独で学習し基準とする:同じ記述子(KMD)、同じ採用ハイパーパラメータ、5シード、早期終了。訓練行数約23,000の Materials Project タスクは明確に二分される:4つは R² 0.91–0.99 まで学習でき、2つはできない — しかもその2つは、以前 XenonPy 組成記述子のモデルが相関 ≈ 0.99 で予測できていた物性である。モデルでもデータ量でも説明がつかず、原因はモデルに与えたものの中にあるはずだった。")}</div>
<figure><div class="figbox"><svg id="fig-ceil" width="960" height="320" role="img" aria-label="Single-task R² of the six MP tasks"></svg></div>
{CAP("Single-task R², mean ± sd over five seeds, on the 2026-05-15 labels. total_magnetization (0.72) is per formula unit and shares volume's dependence on cell size; it is not analysed further here.",
"单任务 R²,5 个 seed 的均值 ± sd,2026-05-15 标签。total_magnetization(0.72)是每化学式单元的量,和 volume 一样依赖原胞大小;本页不再展开。",
"単独学習の R²、5シードの平均 ± sd、2026-05-15 ラベル。total_magnetization(0.72)は化学式単位あたりの量で、volume と同じく単位胞の大きさに依存する。本ページでは扱わない。")}</figure>
{table([T("task","任务","タスク"), "N train", "R²", "sd"], rows, style="max-width:520px")}
</section>''')

# ---------------- 2 final_energy ----------------
probe_rows = []
for label, f, gga, r2s, mixed, old, neu in D["probe"]:
    probe_rows.append([label, f"{gga:.3f}", f"{r2s:.3f}", f"{mixed:.3f}", f'<span class="c-xfer">{old:.4f}</span>', f'<span class="c-warm">{neu:.3f}</span>'])
out.append(f'''<section class="section">{H2("2 · final_energy", "The label mixed two DFT energy references", "标签混合了两种 DFT 能量参考", "ラベルが2つの DFT エネルギー基準を混ぜていた")}
<div class="col">{P("The dataset's <em>Final energy per atom</em> runs from −86.4 to −0.04 eV. A DFT energy per atom on the Materials Project's GGA scale does not go below about −14 eV, and 18.8% of the rows sit below that. Spot checks against known materials show values far off the GGA scale for some elements and on it for others:",
"数据集里的 <em>Final energy per atom</em> 从 −86.4 到 −0.04 eV。Materials Project 的 GGA 口径下,每原子 DFT 能量不会低于约 −14 eV,而 18.8% 的行低于此值。用已知材料抽查,有的元素远离 GGA 口径,有的又在口径上:",
"データセットの <em>Final energy per atom</em> は −86.4 から −0.04 eV に広がる。Materials Project の GGA スケールでは原子あたり DFT エネルギーは約 −14 eV を下回らないのに、行の 18.8% がそれより低い。既知物質での抜き取り検査では、GGA スケールから大きく外れる元素もあれば、スケール上にある元素もある:")}</div>
<figure><div class="figbox"><div class="legend"><span><i class="sw" style="background:var(--warm)"></i> {T("GGA / GGA+U","GGA / GGA+U","GGA / GGA+U")}</span><span><i class="sw" style="background:var(--surface);border:2px solid var(--frz)"></i> r2SCAN</span><span><i class="sw sq" style="background:var(--xfer)"></i> {T("dataset 2026-05-15","数据集 2026-05-15","データセット 2026-05-15")}</span><span><i class="sw" style="background:none;border:1.5px dashed var(--warm)"></i> {T("dataset 2026-09-12","数据集 2026-09-12","データセット 2026-09-12")}</span></div>
<svg id="fig-probe" width="960" height="360" role="img" aria-label="Energy per atom of six materials under each scheme"></svg></div>
{CAP("Six materials queried from the Materials Project API on 2026-09-12. The old dataset value (square) sits on the r2SCAN value wherever MP has an r2SCAN calculation, and on the GGA value where it does not (NaCl). The rebuilt dataset (dashed ring) sits on GGA / GGA+U for all of them.",
"2026-09-12 从 Materials Project API 查询的六种材料。旧数据集的值(方块)在 MP 有 r2SCAN 计算的地方落在 r2SCAN 值上,没有的地方落在 GGA 值上(NaCl)。重建后的数据集(虚线圈)全部落在 GGA / GGA+U 上。",
"2026-09-12 に Materials Project API から取得した6物質。旧データセットの値(四角)は MP に r2SCAN 計算がある物質では r2SCAN 値に、ない物質(NaCl)では GGA 値に一致する。再構築データセット(破線の輪)は全て GGA / GGA+U 上にある。")}</figure>
{table([T("material","材料","物質"), T("GGA / GGA+U","GGA / GGA+U","GGA / GGA+U"), "r2SCAN", T("MP summary (mixed)","MP summary(混合)","MP summary(混合)"), T("dataset 2026-05-15","数据集 2026-05-15","データセット 2026-05-15"), T("dataset 2026-09-12","数据集 2026-09-12","データセット 2026-09-12")], probe_rows, style="max-width:900px")}
<div class="col" style="margin-top:18px">{P("The Materials Project's <em>summary.energy_per_atom</em> is its <em>mixed</em> thermodynamic scheme (thermo_type GGA_GGA+U_R2SCAN): where an r2SCAN calculation exists it reports the r2SCAN total energy, whose zero differs from the GGA one by an element-dependent amount. The 2025-04-10 export copied that field faithfully — 89.7% of a random 600 rows are still bit-identical to today's API value — so this is not a collection bug. Over all 33,159 materials that carry both schemes:",
"Materials Project 的 <em>summary.energy_per_atom</em> 是它的<em>混合</em>热力学方案(thermo_type GGA_GGA+U_R2SCAN):凡是有 r2SCAN 计算的条目就报 r2SCAN 总能,而它的零点与 GGA 相差一个随元素变化的量。2025-04-10 的导出忠实复制了这个字段——随机 600 行里 89.7% 至今与 API 逐位相同——所以不是采集错误。在同时有两种口径的 33,159 个材料上:",
"Materials Project の <em>summary.energy_per_atom</em> は<em>混合</em>熱力学スキーム(thermo_type GGA_GGA+U_R2SCAN)である:r2SCAN 計算がある項目では r2SCAN 全エネルギーを返し、その零点は GGA と元素依存の量だけ異なる。2025-04-10 のエクスポートはこの欄を忠実に写した — ランダム600行の 89.7% は今日の API 値とビット単位で同一 — つまり収集バグではない。両スキームを持つ 33,159 物質全体では:")}</div>
<figure><div class="figbox"><svg id="fig-mixed" width="960" height="300" role="img" aria-label="Mixed-scheme minus GGA energy over all materials"></svg></div>
{CAP(f"Mixed-scheme energy minus GGA / GGA+U energy per material. {MG['equal_share']*100:.1f}% are equal (no r2SCAN calculation, or GGA preferred); {MG['below_5eV_share']*100:.1f}% sit more than 5 eV per atom below, up to −72 eV. One column, two energy zeros.",
f"每个材料的混合方案能量减去 GGA / GGA+U 能量。{MG['equal_share']*100:.1f}% 相等(没有 r2SCAN 计算,或以 GGA 为准);{MG['below_5eV_share']*100:.1f}% 低出 5 eV/atom 以上,最多 −72 eV。一列数据,两个能量零点。",
f"物質ごとの混合スキームエネルギー − GGA / GGA+U エネルギー。{MG['equal_share']*100:.1f}% は等しく(r2SCAN 計算なし、または GGA 優先)、{MG['below_5eV_share']*100:.1f}% は 5 eV/atom 以上低く、最大 −72 eV。1つの列に2つのエネルギー零点。")}</figure>
<div class="col"><h3>{T("Two checks that rule out the innocent readings","排除两种无害解读的检验","無害な解釈を排除する2つの検証")}</h3></div>
{table([T("check","检验","検証"), T("result","结果","結果"), T("reading","解读","解釈")], [
 [T("Is it a total energy per cell? corr(E, atoms per cell)","是不是每原胞总能?corr(E, 原胞原子数)","単位胞あたりの全エネルギーか? corr(E, 原子数)"), f"{K['corr_E_N']:+.3f}; Pt has 1 atom per cell and reads −51.5", T("No — it does not scale with the cell","否——不随原胞大小变化","否 — 単位胞の大きさに比例しない")],
 [T("Is it a per-atom energy on one reference? E − Σ xᵢ·E(element) vs the dataset's formation energy","是不是单一参考下的每原子能量?E − Σ xᵢ·E(单质) 对比数据集的 formation energy","単一基準の原子あたりエネルギーか? E − Σ xᵢ·E(単体) と formation energy の比較"), f"corr {K['corr_ref_formation']:.2f}; median |diff| {K['median_abs_diff']:.1f} eV, 90th pct {K['p90_abs_diff']:.1f} eV", T("No — a consistent reference would reproduce the formation energy to ~0.1 eV","否——参考一致时应能重现 formation energy 到 ~0.1 eV","否 — 基準が一貫していれば formation energy を ~0.1 eV で再現するはず")],
 [T("Does composition carry it? gradient boosting on XenonPy features, pipeline split / random split","组成能否表达它?XenonPy 特征上的梯度提升,pipeline 划分 / 随机划分","組成で表せるか? XenonPy 特徴での勾配ブースティング、pipeline 分割 / ランダム分割"), f"R² {D['gbm']['pipeline_split']:.3f} / {D['gbm']['random_split']:.3f}; rows above −14 eV only: {D['gbm']['above_minus14_only']:.3f}", T("No — not the split, not the tail alone; the element-dependent zero dominates the variance","否——不是划分、也不只是尾部;随元素变化的零点主导了方差","否 — 分割でも裾だけでもない。元素依存の零点が分散を支配する")]], cls="tablebox wrap", style="max-width:1000px")}
<div class="col" style="margin-top:18px"><h3>{T("The fix, and the proof","修正与验证","修正と検証")}</h3>
{P(f"The GGA / GGA+U scheme (thermo_type GGA_GGA+U) covers {C6['gga_coverage']*100:.1f}% of the entries and runs from −14.2 to −0.01 eV — the scale the earlier final-energy models had been trained on. The dataset was rebuilt on it. Trained alone with the unchanged recipe and descriptor, final_energy goes from R² {m(fe_old):.4f} ± {sd(fe_old):.4f} to <b>{m(fe_new):.4f} ± {sd(fe_new):.4f}</b>.",
f"GGA / GGA+U 口径(thermo_type GGA_GGA+U)覆盖 {C6['gga_coverage']*100:.1f}% 的条目,范围 −14.2 到 −0.01 eV——正是早先 final energy 模型用的口径。数据集据此重建。配方和描述符不变,单独训练的 final_energy 从 R² {m(fe_old):.4f} ± {sd(fe_old):.4f} 升到 <b>{m(fe_new):.4f} ± {sd(fe_new):.4f}</b>。",
f"GGA / GGA+U スキーム(thermo_type GGA_GGA+U)は項目の {C6['gga_coverage']*100:.1f}% をカバーし、−14.2 から −0.01 eV の範囲 — 以前の final energy モデルが学習していたスケールである。データセットはこれで再構築した。レシピと記述子を変えずに単独学習すると、final_energy は R² {m(fe_old):.4f} ± {sd(fe_old):.4f} から <b>{m(fe_new):.4f} ± {sd(fe_new):.4f}</b> になる。")}</div>
<figure><div class="figbox"><div class="legend"><span><i class="sw sq" style="background:var(--xfer)"></i> {T("2026-05-15 label","2026-05-15 标签","2026-05-15 ラベル")}</span><span><i class="sw sq" style="background:var(--warm)"></i> {T("GGA / GGA+U label","GGA / GGA+U 标签","GGA / GGA+U ラベル")}</span></div>
<svg id="fig-fehist" width="960" height="300" role="img" aria-label="Distribution of the final-energy label before and after"></svg></div>
{CAP("The label before and after, same 33,166 rows. The long tail below −14 eV is gone; 27.4% of the rows changed by more than 0.05 eV.",
"修正前后的标签,同一批 33,166 行。−14 eV 以下的长尾消失;27.4% 的行变化超过 0.05 eV。",
"修正前後のラベル、同じ 33,166 行。−14 eV 以下の長い裾が消え、行の 27.4% が 0.05 eV 以上変化した。")}</figure>
<figure><div class="figbox"><div class="legend"><span>● {T("one seed","一个 seed","1シード")} · ▬ {T("median","中位数","中央値")}</span></div><svg id="fig-fe-seeds" width="960" height="300" role="img" aria-label="final_energy R² per seed before and after"></svg></div>
{CAP("Five seeds each. Nothing but the label differs between the two arms.","各 5 个 seed。两臂之间只有标签不同。","各5シード。2つのアームの違いはラベルだけ。")}</figure>
<figure><div class="figbox"><div class="legend"><span><i class="sw" style="background:var(--xfer)"></i> {T("2026-05-15 label (R² 0.78)","2026-05-15 标签(R² 0.78)","2026-05-15 ラベル(R² 0.78)")}</span><span><i class="sw" style="background:var(--warm)"></i> {T("GGA / GGA+U label (R² 0.999)","GGA / GGA+U 标签(R² 0.999)","GGA / GGA+U ラベル(R² 0.999)")}</span></div><svg id="fig-fecurves" width="960" height="300" role="img" aria-label="Validation loss curves before and after"></svg></div>
{CAP("Validation loss of one seed under each label. On the old label the loss floors at 0.22 within 100 epochs — the element-dependent offset is simply not learnable from composition aggregates; on the GGA label it keeps falling by two orders of magnitude.",
"两种标签下同一 seed 的验证损失。旧标签下损失在 100 epoch 内停在 0.22——随元素变化的偏移根本无法从组成聚合特征学到;GGA 标签下损失继续下降两个数量级。",
"各ラベルでの同一シードの検証損失。旧ラベルでは 100 エポック以内に 0.22 で頭打ち — 元素依存のずれは組成の集約特徴からは学べない。GGA ラベルでは2桁下がり続ける。")}</figure>
</section>''')

# ---------------- 3 volume ----------------
ridge = D["ridge"]
desc_rows = []
for t in ("volume", "final_energy", "dos_density"):
    d = DS[t]
    c = f"{d['classic']:.4f} ({d['classic_vs']:+.1f}%{'*' if d['classic_sep'] else ''})"
    n = f"{d['nosum']:.4f} ({d['nosum_vs']:+.1f}%{'*' if d['nosum_sep'] else ''})"
    desc_rows.append([t.replace("_", " "), f"{d['kmd']:.4f}", c, n])
out.append(f'''<section class="section">{H2("3 · volume", "The label depends on a cell size the descriptor cannot see", "标签依赖描述符看不见的原胞大小", "ラベルは記述子に見えない単位胞の大きさに依存する")}
<div class="col">{P(f"<em>Volume (normalized)</em> is the dataset's <em>volume_scaler</em> — StandardScaler followed by a Yeo-Johnson transform with λ ≈ 0 — applied to the <em>cell</em> volume, so it is essentially log(cell volume): its correlation with log V is {L['corr_volnorm_logV']:.3f}, with V itself {L['corr_volnorm_V']:.3f}. And the cell is whatever structure Materials Project stored: a median of {L['atoms_per_cell']['quantiles']['0.5']:.0f} atoms, 128 at the 99th percentile, thousands at the extreme. corr(volume, atoms per cell) = {L['corr_volume_atoms']:.3f}: most of the label is the atom count.",
f"<em>Volume (normalized)</em> 来自数据集的 <em>volume_scaler</em>——StandardScaler 之后接 λ ≈ 0 的 Yeo-Johnson 变换——作用在<em>原胞</em>体积上,所以它本质上是 log(原胞体积):与 log V 的相关 {L['corr_volnorm_logV']:.3f},与 V 本身 {L['corr_volnorm_V']:.3f}。而原胞就是 Materials Project 存的那个结构:原子数中位 {L['atoms_per_cell']['quantiles']['0.5']:.0f},99% 分位 128,极端到数千。corr(体积, 原胞原子数) = {L['corr_volume_atoms']:.3f}:标签的大头是原子数。",
f"<em>Volume (normalized)</em> はデータセットの <em>volume_scaler</em> — StandardScaler の後に λ ≈ 0 の Yeo-Johnson 変換 — を<em>単位胞</em>体積に適用したもので、本質的に log(単位胞体積) である:log V との相関 {L['corr_volnorm_logV']:.3f}、V 自体とは {L['corr_volnorm_V']:.3f}。そして単位胞は Materials Project が保存した構造そのもの:原子数の中央値 {L['atoms_per_cell']['quantiles']['0.5']:.0f}、99パーセンタイル 128、極端には数千。corr(体積, 原子数) = {L['corr_volume_atoms']:.3f}:ラベルの大半は原子数である。")}</div>
<figure><div class="figbox"><svg id="fig-atoms" width="960" height="280" role="img" aria-label="Atoms per cell distribution"></svg></div>
{CAP("Atoms per cell of the composition string the model receives. KMD turns that string into atomic fractions before anything else happens, so Fe₂O₃ and Fe₄O₆ are the same input — verified in code — and the atom count never reaches the model.",
"模型收到的组成串的原胞原子数。KMD 先把这个串变成原子分数,之后才有别的操作,所以 Fe₂O₃ 和 Fe₄O₆ 是同一个输入——已在代码中验证——原子数从未到达模型。",
"モデルが受け取る組成文字列の単位胞原子数。KMD はまずその文字列を原子分率に変換するので、Fe₂O₃ と Fe₄O₆ は同一の入力 — コードで検証済み — であり、原子数はモデルに届かない。")}</figure>
<div class="col"><h3>{T("The contrast experiment: XenonPy classic, with and without its scale-bearing block","对比实验:XenonPy classic,带与不带承载尺度的块","対比実験:XenonPy classic、スケールを担うブロックの有無")}</h3>
{P("XenonPy's classic composition descriptor — the one the data notebooks compute, weighted sum / average / variance / max / min over 58 element properties, then StandardScaler and Yeo-Johnson — has one block that uses the raw cell amounts: the weighted sum. Dropping it leaves a scale-free descriptor like KMD. A linear ridge fit on the true target already tells the story, and the single-task network confirms it:",
"XenonPy 的 classic 组成描述符——数据 notebook 算的那一个:58 个元素性质上的 weighted sum / average / variance / max / min,再做 StandardScaler 和 Yeo-Johnson——只有一个块使用原胞的原始原子数:weighted sum。去掉它就是像 KMD 一样尺度无关的描述符。在真实目标上做线性岭回归已能说明问题,单任务网络进一步印证:",
"XenonPy の classic 組成記述子 — データノートブックが計算するもので、58 の元素物性に対する weighted sum / average / variance / max / min の後に StandardScaler と Yeo-Johnson — で単位胞の生の原子数を使うのは weighted sum の1ブロックだけ。それを外すと KMD と同じスケール非依存の記述子になる。実際の目標に対する線形リッジ回帰がすでに答えを示し、単独学習ネットワークがそれを裏付ける:")}</div>
{table([T("target","目标","目標"), "KMD", T("XenonPy classic","XenonPy classic","XenonPy classic"), T("without the sum block","去掉 sum 块","sum ブロックなし")], [
 ["Volume (normalized) — ridge, test R²", f"{ridge['Volume (normalized)']['KMD']:.3f}", f'<span class="c-warm"><b>{ridge["Volume (normalized)"]["classic"]:.3f}</b></span>', f"{ridge['Volume (normalized)']['nosum']:.3f}"],
 ["Final energy per atom (normalized), 2026-05-15 label — ridge, test R²", f"{ridge['Final energy per atom (normalized)']['KMD']:.3f}", f"{ridge['Final energy per atom (normalized)']['classic']:.3f}", f"{ridge['Final energy per atom (normalized)']['nosum']:.3f}"]], style="max-width:900px")}
<div style="height:14px"></div>
{table([T("task · single-task network, 5 seeds","任务 · 单任务网络,5 seed","タスク · 単独学習ネットワーク、5シード"), "KMD", T("XenonPy classic (vs KMD)","XenonPy classic(vs KMD)","XenonPy classic(vs KMD)"), T("without the sum block (vs KMD)","去掉 sum 块(vs KMD)","sum ブロックなし(vs KMD)")], desc_rows, style="max-width:900px")}
<figure style="margin-top:14px"><div class="figbox"><div class="legend"><span>● {T("one seed","一个 seed","1シード")} · ▬ {T("median","中位数","中央値")}</span></div><svg id="fig-vol-seeds" width="960" height="300" role="img" aria-label="volume R² per seed by descriptor"></svg></div>
{CAP("volume, single task. With cell scale in the input, R² 0.997 on every seed; without it, 0.59–0.62 whichever scale-free descriptor is used.",
"volume,单任务。输入里有原胞尺度时每个 seed 都是 R² 0.997;没有时无论哪种尺度无关描述符都是 0.59–0.62。",
"volume、単独学習。入力に単位胞スケールがあれば全シードで R² 0.997、なければどのスケール非依存記述子でも 0.59–0.62。")}</figure>
<figure><div class="figbox"><div class="legend"><span>● {T("one seed","一个 seed","1シード")} · ▬ {T("median","中位数","中央値")}</span></div><svg id="fig-other-seeds" width="960" height="300" role="img" aria-label="final_energy and dos_density R² per seed by descriptor"></svg></div>
{CAP("The same three descriptors on final_energy (old label) and dos_density: per-atom and intensive labels do not care about the sum block, and the descriptor family lands within a few percent of KMD. The descriptor was volume's problem, not final_energy's.",
"同样三种描述符用在 final_energy(旧标签)和 dos_density 上:每原子量和强度量不在乎 sum 块,描述符家族与 KMD 相差不过几个百分点。描述符是 volume 的问题,不是 final_energy 的。",
"同じ3つの記述子を final_energy(旧ラベル)と dos_density に適用:原子あたり・示強性のラベルは sum ブロックに影響されず、記述子ファミリーは KMD と数%以内。記述子は volume の問題であって final_energy の問題ではない。")}</figure>
<figure><div class="figbox"><div class="legend"><span><i class="sw" style="background:var(--alone)"></i> KMD</span><span><i class="sw" style="background:var(--warm)"></i> {T("XenonPy classic","XenonPy classic","XenonPy classic")}</span></div>
<svg id="fig-curves" width="960" height="330" role="img" aria-label="Training and validation loss of three volume runs"></svg></div>
{CAP("Seed 2025 of each. With KMD the validation loss floors at 0.4 within 40 epochs: the atom count the label needs is not in the input. With the scale-bearing descriptor both losses keep falling for the full 150 epochs.",
"各取 seed 2025。KMD 下验证损失 40 epoch 内停在 0.4:标签需要的原子数不在输入里。带尺度的描述符下两条损失在整整 150 epoch 里持续下降。",
"各シード 2025。KMD では検証損失が 40 エポック以内に 0.4 で頭打ち:ラベルが必要とする原子数が入力にない。スケールを持つ記述子では両損失が 150 エポックの間下がり続ける。")}</figure>
</section>''')

# ---------------- 4 what it voids ----------------
void_rows = [[t.replace("_", " "), neg(f"{a:+.1f}%"), neg(f"{b:+.1f}%"), neg(f"{c:+.1f}%"), T("void — label / descriptor artefact; re-measure on the 2026-09-12 dataset", "作废——标签 / 描述符假象;在 2026-09-12 数据集上重测", "無効 — ラベル / 記述子の人工物。2026-09-12 データセットで再測定")] for t, a, b, c in D["transfer_void"]]
keep_rows = [[t.replace("_", " "), f"{a:+.1f}%", f"{b:+.1f}%", pos(f"{c:+.1f}%"), T("stands — labels unchanged", "成立——标签未变", "有効 — ラベル不変")] for t, a, b, c in D["transfer_keep"]]
def strip_td(rows):  # cells built with pos()/neg() already carry <td>; wrap plain ones
    out_rows = []
    for r in rows:
        out_rows.append("".join(c if c.startswith("<td") else f"<td>{c}</td>" for c in r))
    return out_rows
def table_raw(head, rows, style=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tablebox wrap"{(" style=" + chr(34) + style + chr(34)) if style else ""}><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'
out.append(f'''<section class="section">{H2("4 · Consequences", "What this voids, and what stands", "哪些作废,哪些成立", "何が無効になり、何が残るか")}
<div class="col">{P("Every stage of the campaign — single-task baselines, the transfer stage, the warm-start stage, the running unseen-encoder stage — used the 2026-05-15 labels. The transfer verdicts on the two tasks whose labels were the problem are void; the mechanism found through the other tasks is untouched.",
"campaign 的每个阶段——单任务基线、迁移阶段、warm-start 阶段、正在跑的未见过编码器阶段——都用的是 2026-05-15 标签。标签有问题的两个任务上的迁移判定作废;通过其他任务发现的机制不受影响。",
"キャンペーンの全段階 — 単独基準、転移段階、ウォームスタート段階、実行中の未知エンコーダ段階 — は 2026-05-15 ラベルを使った。ラベルが問題だった2タスクの転移判定は無効。他のタスクで見つけた機構は影響を受けない。")}</div>
{table_raw([T("task","任务","タスク"), T("xfer vs alone","xfer vs 单独","xfer vs 単独"), T("frozen vs alone","冻结 vs 单独","凍結 vs 単独"), T("warm-start vs alone","warm-start vs 单独","ウォームスタート vs 単独"), T("status","状态","状態")], strip_td(void_rows + keep_rows), style="max-width:1000px")}
<div class="col" style="margin-top:16px">
{P("<b>Void.</b> final_energy's −9.1% under warm-start, its position curve and its 500-epoch budget check were measured against a label no model can learn; volume's −6.9% was measured with a descriptor that cannot see the label. Neither says anything about transfer.",
"<b>作废。</b>final_energy 在 warm-start 下的 −9.1%、它的位置曲线和 500 epoch 预算检验,都是对着一个任何模型都学不会的标签测的;volume 的 −6.9% 是用一个看不见标签的描述符测的。两者都说明不了迁移。",
"<b>無効。</b>final_energy のウォームスタート −9.1%、位置曲線、500 エポック予算検証は、どのモデルにも学べないラベルに対して測定された。volume の −6.9% はラベルが見えない記述子で測定された。どちらも転移については何も語らない。", "callout")}
{P("<b>Stands.</b> Replay dilution at the last step (19 tasks recover once replay is removed, none get worse), material_type's gain and its preference for a frozen encoder, and the gains on zt, magnetization and dielectric_total: their labels did not change.",
"<b>成立。</b>末位的 replay 稀释(去掉 replay 后 19 个任务回升、0 个变差)、material_type 的增益及其对冻结编码器的偏好、zt / magnetization / dielectric_total 的增益:它们的标签没有变。",
"<b>有効。</b>最終ステップでのリプレイ希釈(リプレイを外すと19タスクが回復し悪化はゼロ)、material_type の改善と凍結エンコーダへの選好、zt / magnetization / dielectric_total の改善:これらのラベルは変わっていない。", "callout")}
{P("<b>Also reopened.</b> The extensive-vs-intensive reading of the warm-start losses rested on final_energy and volume; dos_density (−4.2%) is the only extensive-property loss left, and it is intensive per energy point. The remaining question about the shared encoder's generalisation is now carried by the unseen-encoder arms alone.",
"<b>同时重开。</b>对 warm-start 损失的“广延量 vs 强度量”解读建立在 final_energy 和 volume 上;剩下的只有 dos_density(−4.2%),而它按能量点算是强度量。关于共享编码器泛化的遗留问题,现在只能由未见过编码器的两臂来回答。",
"<b>再び開かれた点。</b>ウォームスタート損失の「示量性 vs 示強性」という読みは final_energy と volume に依っていた。残るのは dos_density(−4.2%)だけで、これはエネルギー点ごとには示強性である。共有エンコーダの汎化に関する残された問いは、今や未知エンコーダのアームだけが担う。", "callout")}
</div></section>''')

# ---------------- 5 the standard ----------------
rules = [
 ("One level of theory per column family.", "Energies from the thermo endpoint with thermo_type GGA_GGA+U; structure-derived and magnetic quantities from the GGA / GGA+U entry's own task; electronic-structure values only where MP's origin task is GGA or GGA+U; elastic, dielectric and piezoelectric values from their GGA workflows. Never the summary's mixed fields.",
  "每个列族只用一种理论水平。", "能量取 thermo 端点 thermo_type GGA_GGA+U;结构类和磁性量取 GGA / GGA+U entry 自己的任务;能带类只在 MP 的来源任务是 GGA / GGA+U 时保留;弹性、介电、压电取各自的 GGA 流程。绝不用 summary 的混合字段。",
  "列ファミリーごとに理論レベルは1つ。", "エネルギーは thermo エンドポイントの thermo_type GGA_GGA+U から。構造由来と磁性の量は GGA / GGA+U エントリ自身のタスクから。電子構造の値は MP の由来タスクが GGA / GGA+U の場合のみ。弾性・誘電・圧電はそれぞれの GGA ワークフローから。summary の混合欄は決して使わない。"),
 ("Per-cell quantities live on the dataset's cell.", "Volume and total magnetisation from another calculation's cell are rescaled by the atom-count ratio after checking the reduced formula matches; intensive forms (volume per atom, magnetisation per volume) are stored alongside.",
  "每原胞的量必须落在数据集的原胞上。", "来自其他计算原胞的体积和总磁矩,先核对约化化学式一致,再按原子数比例换算;同时存储强度量形式(每原子体积、每体积磁化)。",
  "単位胞あたりの量はデータセットの単位胞上に置く。", "他の計算の単位胞から来た体積と全磁化は、既約化学式の一致を確認した上で原子数比で換算する。示強性の形(原子あたり体積、体積あたり磁化)も併せて保存する。"),
 ("Provenance travels with the value.", "The new-format MP id, the origin task's run type, and the 31 legacy ids MP no longer resolves are recorded; the build script and its inputs are kept with the data.",
  "来源随数值一起保存。", "记录新格式 MP id、来源任务的 run type、以及 MP 已不再解析的 31 个旧 id;生成脚本及其输入与数据放在一起。",
  "来歴は値と共に保存する。", "新形式の MP id、由来タスクの run type、MP がもう解決しない31の旧 id を記録する。生成スクリプトとその入力はデータと共に保管する。"),
 ("Descriptors and preprocessing come from the data notebooks, never re-implemented.", "XenonPy classic with StandardScaler + Yeo-Johnson as calculate_compositional_desc.ipynb produces it; a variant that is not in a notebook is proposed, not run.",
  "描述符和预处理只从数据 notebook 取,绝不重实现。", "按 calculate_compositional_desc.ipynb 的做法:XenonPy classic 加 StandardScaler + Yeo-Johnson;notebook 里没有的变体只能提议,不能直接跑。",
  "記述子と前処理はデータノートブックから取り、再実装しない。", "calculate_compositional_desc.ipynb が生成する通り、XenonPy classic に StandardScaler + Yeo-Johnson。ノートブックにない変種は提案するだけで、実行しない。"),
 ("A label that depends on cell scale needs a scale-aware descriptor or an intensive form.", "volume, total magnetisation per cell and anything per formula unit are only meaningful to a model whose input carries the atom count; otherwise train the intensive column.",
  "依赖原胞尺度的标签需要能感知尺度的描述符,或改用强度量形式。", "volume、每原胞总磁矩以及任何每化学式单元的量,只对输入里带原子数的模型有意义;否则训练强度量那一列。",
  "単位胞スケールに依存するラベルには、スケールを認識する記述子か示強性の形が必要。", "volume、単位胞あたり全磁化、化学式単位あたりの量は、入力に原子数を持つモデルにしか意味がない。そうでなければ示強性の列を学習する。"),
 ("Four gates before a label is trained on.", "Spot-check against known references (Si, Pt, Fe, NaCl); for energies, E − Σ xᵢ·E(element) must reproduce the formation energy; a ridge or gradient-boosting baseline on the notebook descriptors; and a per-column change report against the previous version.",
  "标签投入训练前的四道门。", "对已知参考值抽查(Si、Pt、Fe、NaCl);能量类要求 E − Σ xᵢ·E(单质) 能重现 formation energy;在 notebook 描述符上做岭回归或梯度提升基线;与上一版本逐列出变更报告。",
  "ラベルを学習に使う前の4つの関門。", "既知の基準値(Si、Pt、Fe、NaCl)での抜き取り検査。エネルギーは E − Σ xᵢ·E(単体) が formation energy を再現すること。ノートブック記述子でのリッジ回帰または勾配ブースティング基準。前版との列ごとの変更報告。"),
 ("Versioned files, never overwritten.", "A new date in the file name, a new preprocessing object, and a CHANGES note; results are never compared across dataset versions for a column whose label changed.",
  "文件带版本,绝不覆盖。", "文件名带新日期、新的 preprocessing 对象、一份 CHANGES 说明;标签变过的列,结果绝不跨数据集版本比较。",
  "ファイルはバージョン付きで、上書きしない。", "ファイル名に新しい日付、新しい前処理オブジェクト、CHANGES ノート。ラベルが変わった列については、データセット版をまたいで結果を比較しない。"),
]
items = "".join(f"<li><b>{T(a, c, e)}</b> {T(b, d, f)}</li>" for a, b, c, d, e, f in rules)
out.append(f'''<section class="section">{H2("5 · The standard", "How Materials Project data enters the dataset from now on", "从今往后 Materials Project 数据如何进入数据集", "今後 Materials Project データをデータセットに入れる方法")}
<div class="col plan"><ol class="plan">{items}</ol></div></section>''')


# ---------------- 7 the added properties, explained ----------------
PROPS = [
 # (task name, what it is, how Materials Project computes it, unit, our label form)
 ("band_gap", ("Band gap", "带隙", "バンドギャップ"),
  ("The energy an electron must gain to jump from the highest filled electronic state to the lowest empty one. Zero for metals; a few eV for insulators; the number that decides whether a material conducts, absorbs visible light, or is transparent.",
   "电子从最高占据态跳到最低空态所需的能量。金属为零,绝缘体为几 eV;它决定材料导电、吸收可见光还是透明。",
   "電子が最高占有状態から最低空状態へ跳ぶのに必要なエネルギー。金属ではゼロ、絶縁体では数 eV。導電するか、可視光を吸収するか、透明かを決める量。"),
  ("From the DFT band structure along high-symmetry lines (GGA / GGA+U). GGA underestimates gaps, typically by 30–50%, consistently across materials.",
   "由沿高对称线的 DFT 能带结构得到(GGA / GGA+U)。GGA 系统性低估带隙,通常低 30–50%,但材料之间一致。",
   "高対称線に沿った DFT バンド構造から(GGA / GGA+U)。GGA はギャップを一貫して 30–50% 程度過小評価する。"),
  "eV", ("regression on the normalised value", "对归一化值做回归", "正規化値の回帰")),
 ("cbm / vbm", ("Conduction-band minimum / valence-band maximum", "导带底 / 价带顶", "伝導帯下端 / 価電子帯上端"),
  ("The two band edges themselves: the lowest empty level (CBM) and the highest filled level (VBM). Their difference is the band gap; their absolute positions govern how electrons move across an interface, e.g. in a solar cell or a battery electrode.",
   "两个能带边缘本身:最低空能级(CBM)和最高占据能级(VBM)。二者之差是带隙;它们的绝对位置决定电子如何跨越界面,例如太阳能电池或电池电极。",
   "2つのバンド端そのもの:最低空準位(CBM)と最高占有準位(VBM)。差がバンドギャップ、絶対位置は太陽電池や電池電極などの界面での電子の移動を決める。"),
  ("Eigenvalues of the same GGA band-structure calculation, on the calculation's own energy reference; defined only for materials with a gap (about half of the entries).",
   "同一 GGA 能带计算的本征值,以该计算自身的能量零点为参考;只对有带隙的材料定义(约一半条目)。",
   "同じ GGA バンド構造計算の固有値、その計算自身のエネルギー基準。ギャップのある物質(項目の約半数)にのみ定義。"),
  "eV", ("regression", "回归", "回帰")),
 ("is_metal · is_gap_direct", ("Metal or not · direct or indirect gap", "是否金属 · 直接 / 间接带隙", "金属か否か · 直接 / 間接ギャップ"),
  ("Two yes/no facts read off the band structure: whether the gap is zero (a metal), and — for non-metals — whether the band edges sit at the same crystal momentum (a direct gap, which absorbs and emits light efficiently, as in LEDs) or not (indirect, as in silicon).",
   "从能带结构读出的两个是/否事实:带隙是否为零(金属);对非金属,能带边缘是否位于同一晶体动量(直接带隙,发光吸光高效,如 LED 材料)或不在(间接带隙,如硅)。",
   "バンド構造から読める2つの yes/no:ギャップがゼロか(金属)、非金属ではバンド端が同じ結晶運動量にあるか(直接ギャップ、LED のように発光・吸光が効率的)否か(間接、シリコンなど)。"),
  ("Derived from the GGA band structure; is_gap_direct is meaningful only for non-metals.", "由 GGA 能带结构导出;is_gap_direct 只对非金属有意义。", "GGA バンド構造から導出。is_gap_direct は非金属でのみ意味を持つ。"),
  "—", ("two-class classification", "二分类", "2クラス分類")),
 ("density_atomic", ("Volume per atom", "每原子体积", "原子あたり体積"),
  ("The average space one atom occupies in the crystal: cell volume divided by the number of atoms in the cell. Unlike the cell volume it does not depend on how large a cell the database happened to store, so a composition-only model can learn it.",
   "晶体中平均每个原子占据的空间:原胞体积除以原胞原子数。与原胞体积不同,它不依赖数据库恰好存了多大的原胞,所以只看组成的模型也能学。",
   "結晶中で1原子が平均して占める空間:単位胞体積を原子数で割ったもの。単位胞体積と違い、データベースがどの大きさの単位胞を保存したかに依存しないので、組成だけのモデルでも学習できる。"),
  ("From the GGA-relaxed structure of each material.", "来自每个材料的 GGA 弛豫结构。", "各物質の GGA 緩和構造から。"),
  "Å³ / atom", ("regression", "回归", "回帰")),
 ("magnetization_per_volume · magnetization_per_fu", ("Magnetisation per volume · per formula unit", "每体积 / 每化学式单元磁化强度", "体積あたり / 化学式単位あたり磁化"),
  ("The net magnetic moment of the crystal in its calculated ground state, expressed per unit volume (an intensive quantity, comparable across materials) or per formula unit (per Fe₂O₃, per NaCl…). The unit μB, the Bohr magneton, is roughly the moment of one electron's spin.",
   "晶体在计算基态下的净磁矩,按单位体积表示(强度量,可跨材料比较)或按化学式单元表示(每个 Fe₂O₃、每个 NaCl…)。单位 μB(玻尔磁子)大致是一个电子自旋的磁矩。",
   "計算された基底状態での結晶の正味磁気モーメントを、単位体積あたり(示強性、物質間で比較可能)または化学式単位あたり(Fe₂O₃ ひとつ、NaCl ひとつ…)で表したもの。単位 μB(ボーア磁子)はおよそ電子スピン1個のモーメント。"),
  ("Total magnetisation from the OUTCAR of the GGA / GGA+U calculation (collinear spins), divided by the GGA cell volume or by the number of formula units in the dataset's cell.",
   "取 GGA / GGA+U 计算 OUTCAR 中的总磁化(共线自旋),除以 GGA 原胞体积或数据集原胞里的化学式单元数。",
   "GGA / GGA+U 計算の OUTCAR の全磁化(共線スピン)を、GGA 単位胞体積またはデータセット単位胞中の化学式単位数で割る。"),
  "μB / Å³ · μB / f.u.", ("regression", "回归", "回帰")),
 ("magnetic_ordering", ("Magnetic ordering", "磁序", "磁気秩序"),
  ("How the atomic moments line up in the calculated ground state: non-magnetic (no moments), ferromagnetic (all parallel, a net moment — the fridge-magnet case), ferrimagnetic (anti-parallel but unequal, a smaller net moment), antiferromagnetic (anti-parallel and cancelling, no net moment).",
   "计算基态下原子磁矩的排列方式:非磁(无磁矩)、铁磁(全部平行,有净磁矩——冰箱贴那种)、亚铁磁(反平行但不等量,净磁矩较小)、反铁磁(反平行且抵消,无净磁矩)。",
   "計算基底状態での原子磁気モーメントの並び方:非磁性(モーメントなし)、強磁性(全て平行で正味モーメントあり — 冷蔵庫マグネットの場合)、フェリ磁性(反平行だが不等で小さな正味モーメント)、反強磁性(反平行で打ち消し合い正味モーメントなし)。"),
  ("pymatgen's collinear magnetic-structure analyser applied to the per-site moments of the GGA / GGA+U calculation.",
   "把 pymatgen 的共线磁结构分析器作用于 GGA / GGA+U 计算的逐位点磁矩。",
   "GGA / GGA+U 計算のサイトごとのモーメントに pymatgen の共線磁気構造解析器を適用。"),
  "—", ("four-class classification (NM 67%, FM 23%, FiM 8%, AFM 2%)", "四分类(NM 67%、FM 23%、FiM 8%、AFM 2%)", "4クラス分類(NM 67%、FM 23%、FiM 8%、AFM 2%)")),
 ("reaction_energy", ("Equilibrium reaction energy", "平衡反应能", "平衡反応エネルギー"),
  ("How much energy the compound would release or absorb if it decomposed into the most competitive neighbouring phases — the thermodynamic safety margin of a stable compound. Zero means it sits exactly on the boundary; more negative means more firmly stable.",
   "化合物分解为最具竞争力的相邻相时释放或吸收的能量——稳定化合物的热力学安全余量。零表示恰在边界上,越负越稳定。",
   "化合物が最も競合する隣接相へ分解したときに放出・吸収されるエネルギー — 安定化合物の熱力学的な余裕。ゼロはちょうど境界、負が大きいほど安定。"),
  ("From the GGA / GGA+U convex hull of the chemical system, per atom.", "由该化学体系的 GGA / GGA+U 凸包得到,每原子。", "その化学系の GGA / GGA+U 凸包から、原子あたり。"),
  "eV / atom", ("regression", "回归", "回帰")),
 ("bulk_modulus · shear_modulus", ("Bulk modulus · shear modulus", "体模量 · 剪切模量", "体積弾性率 · 剪断弾性率"),
  ("Two stiffnesses: the bulk modulus is the pressure needed to squeeze the material to a slightly smaller volume; the shear modulus is the stress needed to change its shape without changing its volume. Diamond is stiff (K ≈ 440 GPa), rubber is not.",
   "两种刚度:体模量是把材料压缩到略小体积所需的压强;剪切模量是改变形状而不改变体积所需的应力。金刚石很硬(K ≈ 440 GPa),橡胶不是。",
   "2つの剛性:体積弾性率は物質をわずかに小さい体積へ圧縮するのに必要な圧力、剪断弾性率は体積を変えずに形を変えるのに必要な応力。ダイヤモンドは硬く(K ≈ 440 GPa)、ゴムは柔らかい。"),
  ("Voigt–Reuss–Hill averages of the elastic tensor, computed by deforming the GGA structure in several directions and fitting the stress response.",
   "弹性张量的 Voigt–Reuss–Hill 平均,通过在几个方向上使 GGA 结构变形并拟合应力响应算得。",
   "弾性テンソルの Voigt–Reuss–Hill 平均。GGA 構造を複数方向に変形させ応力応答を当てはめて計算。"),
  "GPa", ("regression", "回归", "回帰")),
 ("poisson_ratio · universal_anisotropy", ("Poisson ratio · universal anisotropy", "泊松比 · 通用各向异性", "ポアソン比 · 普遍異方性"),
  ("Poisson's ratio: how much a material narrows when stretched (about 0.3 for most solids, 0.5 for something incompressible like rubber). The universal anisotropy index: how strongly the stiffness depends on direction — 0 for a material that responds the same way in every direction, larger when it does not.",
   "泊松比:材料被拉伸时横向收窄的程度(多数固体约 0.3,橡胶等不可压缩材料 0.5)。通用各向异性指数:刚度随方向变化的程度——各方向响应相同为 0,越依赖方向越大。",
   "ポアソン比:引き伸ばしたとき横方向にどれだけ細くなるか(多くの固体で約 0.3、ゴムなど非圧縮性で 0.5)。普遍異方性指数:剛性がどれだけ方向に依存するか — 全方向で同じなら 0、依存が強いほど大きい。"),
  ("Both derived from the same elastic tensor (Voigt and Reuss bounds).", "两者都由同一弹性张量导出(Voigt 与 Reuss 上下界)。", "どちらも同じ弾性テンソル(Voigt と Reuss の限界)から導出。"),
  "—", ("regression", "回归", "回帰")),
 ("refractive_index", ("Refractive index", "折射率", "屈折率"),
  ("How much light slows down inside the material (glass ≈ 1.5, diamond ≈ 2.4). It is the square root of the electronic part of the dielectric constant, so it is a direct optical property.",
   "光在材料内部减速的程度(玻璃 ≈ 1.5,金刚石 ≈ 2.4)。它是介电常数电子部分的平方根,是直接的光学性质。",
   "物質内部で光がどれだけ遅くなるか(ガラス ≈ 1.5、ダイヤモンド ≈ 2.4)。誘電率の電子部分の平方根で、直接的な光学物性。"),
  ("From the GGA dielectric tensor computed with density-functional perturbation theory (DFPT).", "由密度泛函微扰理论(DFPT)计算的 GGA 介电张量得到。", "密度汎関数摂動理論(DFPT)で計算した GGA 誘電テンソルから。"),
  "—", ("regression", "回归", "回帰")),
 ("piezoelectric_max", ("Piezoelectric maximum", "压电最大值", "圧電最大値"),
  ("How much electric charge appears on the crystal's faces when it is squeezed — the effect behind quartz clocks and ultrasound transducers. Only crystals without a centre of symmetry can be piezoelectric, so few entries have it.",
   "晶体被挤压时表面出现多少电荷——石英钟和超声换能器背后的效应。只有没有对称中心的晶体才有压电性,所以条目很少。",
   "結晶を押したとき表面にどれだけ電荷が現れるか — 水晶時計や超音波振動子の原理。対称中心のない結晶だけが圧電性を持つので、該当項目は少ない。"),
  ("The largest component of the DFPT piezoelectric tensor.", "DFPT 压电张量的最大分量。", "DFPT 圧電テンソルの最大成分。"),
  "C / m²", ("regression (196 test rows)", "回归(196 个测试行)", "回帰(テスト行 196)")),
 ("space_group", ("Space group", "空间群", "空間群"),
  ("The symmetry of the crystal: which rotations, mirrors and translations map the structure onto itself. There are 230 possible space groups; the label is a symbol such as Fm-3m (rock salt), Pnma or P2₁/c. Nature is very uneven here — a few groups hold thousands of compounds, many hold a handful.",
   "晶体的对称性:哪些旋转、镜面和平移能把结构映射回自身。共有 230 种可能的空间群;标签是 Fm-3m(岩盐型)、Pnma、P2₁/c 这样的符号。自然界在这里极不均匀——少数空间群包含上千种化合物,很多只有几种。",
   "結晶の対称性:どの回転・鏡映・並進が構造を自分自身に写すか。可能な空間群は 230 種で、ラベルは Fm-3m(岩塩型)、Pnma、P2₁/c のような記号。自然はここで非常に不均一 — 少数の群が数千の化合物を含み、多くの群はわずか数個。"),
  ("Symmetry detection on the relaxed structure (spglib, as stored by Materials Project). Groups with at least 10 entries and at least one row in both the train and test split are kept; rarer groups are treated as missing, like any other task's missing labels.",
   "对弛豫结构做对称性识别(spglib,Materials Project 存储)。保留至少 10 个条目且在训练集和测试集里各至少 1 行的空间群;更稀有的按缺失处理,和其他任务的缺失标签一样。",
   "緩和構造に対する対称性検出(spglib、Materials Project が保存)。10 項目以上かつ訓練・テスト両分割に1行以上ある群を保持し、それより稀な群は他タスクの欠損ラベルと同様に欠損扱い。"),
  "—", ("151-class classification, very uneven", "151 类分类,极不均衡", "151クラス分類、極めて不均一")),
]
def props_section():
    head=[T("task","任务","タスク"), T("what it is","是什么","何か"), T("how Materials Project computes it","Materials Project 怎么算","Materials Project の計算方法"), T("unit","单位","単位"), T("our label","我们的标签","本研究のラベル")]
    rows=[[f"<b>{T(*nm)}</b><br><span style='color:var(--muted);font-size:12.5px'>{tk}</span>", T(*what), T(*how), unit, T(*form)] for tk, nm, what, how, unit, form in PROPS]
    return f'''<section class="section">{H2("7 · The added properties", "What the new tasks measure, in plain terms", "新任务测的是什么,用大白话说", "新タスクが測るもの、平易に")}
<div class="col">{P("For readers outside materials science: each property below is a single number (or a class) that Materials Project computed for a crystal with density-functional theory (DFT), a quantum-mechanical approximation to how electrons arrange themselves. The model receives only the chemical composition and must predict these numbers. Every regression label is standardised before training (StandardScaler followed by a Yeo-Johnson transform), so the R² and MAE reported elsewhere on this page are on that normalised scale.",
"给材料领域之外的读者:下面每个性质都是 Materials Project 用密度泛函理论(DFT,一种描述电子如何排布的量子力学近似)为一种晶体算出的一个数(或一个类别)。模型只拿到化学组成,要预测这些数。所有回归标签训练前都做了标准化(StandardScaler 再接 Yeo-Johnson 变换),所以本页其他地方报告的 R² 与 MAE 都在这个归一化尺度上。",
"材料科学以外の読者へ:以下の各物性は、Materials Project が結晶に対して密度汎関数理論(DFT、電子の配置を扱う量子力学的近似)で計算した1つの数値(または分類)である。モデルは化学組成だけを受け取り、これらを予測しなければならない。回帰ラベルはすべて学習前に標準化される(StandardScaler の後に Yeo-Johnson 変換)ので、本ページの R² と MAE はその正規化スケール上の値である。")}</div>
{table(head, rows, cls="tablebox wrap")}
</section>'''

# ---------------- 8 baselines on the rebuilt dataset (only when scored) ----------------
BASE = EXP / "summary" / "baselines_mp2026.json"
def baselines_section():
    if not BASE.exists():
        return ""
    B = json.load(open(BASE)); D["baselines"] = B
    rows = []
    for r in B["per_task"]:
        if r.get("n_seeds", 0) == 0:
            continue
        grp = T("relabelled", "标签更新", "ラベル更新") if r["group"] == "updated" else T("added", "新增", "新規")
        if r["kind"] == "regression":
            old = r.get("old_ceiling"); oldv = f"{old['mean']:.4f}" if old else "-"
            delta = (f"{(r['r2']['mean'] - old['mean']) / old['mean'] * 100:+.1f}%" if old else "-")
            rows.append([r["task"].replace("_", " "), grp, "R²", f"{r['n_test']:,}", f"{r['r2']['mean']:.4f} ± {r['r2']['sd']:.4f}", f"{r['mae']['mean']:.4f}", oldv, delta])
        else:
            rows.append([r["task"].replace("_", " ") + (f" ({len(r['classes'])} classes)" if r["task"] == "space_group" else ""), grp, "macro-F1", f"{r['n_test']:,}", f"{r['macro_f1']['mean']:.4f} ± {r['macro_f1']['sd']:.4f}", f"acc {r['accuracy']['mean']:.4f}", "-", "-"])
    return f'''<section class="section">{H2("8 · Baselines on the rebuilt dataset", "Every relabelled and added task, trained alone", "所有标签更新和新增的任务,单独训练", "ラベル更新・新規の全タスク、単独学習")}
<div class="col">{P("The same recipe as every other baseline in the campaign: KMD, the adopted values, five seeds, early stopping, last-epoch weights, the dataset's own split. Regression tasks report R² and MAE on the normalised scale; classification tasks macro-F1 and accuracy. Where a task existed on the 2026-05-15 labels, its old ceiling and the change are shown beside it.",
"与 campaign 里其他基线完全相同的配方:KMD、采用参数、5 个 seed、早停、最后 epoch 权重、数据集自带划分。回归任务报告归一化尺度上的 R² 与 MAE,分类任务报告 macro-F1 与准确率。在 2026-05-15 标签下已存在的任务,旁边给出旧上限和变化。",
"キャンペーンの他の基準と同一のレシピ:KMD、採用値、5シード、早期終了、最終エポックの重み、データセット自身の分割。回帰タスクは正規化スケールでの R² と MAE、分類タスクは macro-F1 と正解率を報告。2026-05-15 ラベルで存在したタスクには旧天井と変化を併記。")}</div>
{table([T("task","任务","タスク"), T("group","类别","区分"), T("metric","指标","指標"), T("test rows","测试行数","テスト行数"), T("mean ± sd, 5 seeds","均值 ± sd,5 seed","平均 ± sd、5シード"), "MAE", T("2026-05-15 ceiling","2026-05-15 上限","2026-05-15 天井"), "Δ"], rows, cls="tablebox")}
<div class="col" style="margin-top:16px">
{P("<b>Relabelled tasks.</b> final_energy goes from 0.77 to 0.999; formation_energy, density and the dielectric tasks move within noise or slightly up (efermi +2.2%, dielectric_electronic +6.0%) — the GGA-family labels are at least as learnable as the old ones. volume stays at 0.61: the label is now the GGA cell volume, but KMD still cannot see the cell, while its per-atom form (density_atomic) trains to 0.979. With KMD, train the intensive column.",
"<b>标签更新的任务。</b>final_energy 从 0.77 到 0.999;formation_energy、density 和介电任务在噪声内或略升(efermi +2.2%、dielectric_electronic +6.0%)——GGA 族标签至少和旧标签一样可学。volume 仍是 0.61:标签现在是 GGA 原胞体积,但 KMD 还是看不见原胞,而它的每原子形式(density_atomic)训到 0.979。用 KMD 时,训练强度量那一列。",
"<b>ラベル更新タスク。</b>final_energy は 0.77 から 0.999 へ。formation_energy、density、誘電タスクはノイズ内かわずかに上昇(efermi +2.2%、dielectric_electronic +6.0%)— GGA 系ラベルは旧ラベルと同等以上に学習可能。volume は 0.61 のまま:ラベルは GGA 単位胞体積になったが KMD にはやはり単位胞が見えない。一方、原子あたりの形(density_atomic)は 0.979 まで学習できる。KMD を使うなら示強性の列を学習する。", "callout")}
{P("<b>Magnetism is the next real limit.</b> Total magnetisation lands at 0.72–0.74 whether per cell, per formula unit or per volume, and magnetic ordering reaches macro-F1 0.56 (accuracy 0.75, dominated by the non-magnetic class): from composition alone the model knows which elements carry moments but not how they order.",
"<b>磁性是下一个真正的极限。</b>总磁矩无论按原胞、按化学式单元还是按体积,都停在 0.72–0.74;磁序的 macro-F1 只有 0.56(准确率 0.75,由非磁类主导):只凭组成,模型知道哪些元素带磁矩,却不知道它们如何排列。",
"<b>磁性が次の本当の限界。</b>全磁化は単位胞あたり・化学式単位あたり・体積あたりのいずれでも 0.72–0.74 に留まり、磁気秩序は macro-F1 0.56(正解率 0.75、非磁性クラスが支配)。組成だけからは、どの元素がモーメントを持つかは分かっても、どう秩序化するかは分からない。", "callout")}
{P("<b>Added tasks.</b> Band gap 0.88, CBM 0.88, VBM 0.92, is-metal macro-F1 0.93, bulk modulus 0.93, refractive index 0.89 and shear modulus 0.79 are solid new tasks. Poisson ratio (0.31), universal anisotropy (0.35) and reaction energy (0.34) carry little composition signal, and piezoelectric maximum (196 test rows) none at all; they should enter the multi-task set only as deliberately hard or low-data tasks, or not at all. Space group (151 classes) reaches macro-F1 0.20 and accuracy 0.24: composition alone identifies the simple high-symmetry families (Fm-3m, I4/mmm, Cmcm, P-62m are recognised 30–60% of the time) and almost never the low-symmetry groups such as Pnma, P2₁/c or C2/c, which are decided by structure, not stoichiometry.",
"<b>新增任务。</b>Band gap 0.88、CBM 0.88、VBM 0.92、is-metal macro-F1 0.93、bulk modulus 0.93、折射率 0.89、shear modulus 0.79 是扎实的新任务。泊松比(0.31)、通用各向异性(0.35)和反应能(0.34)组成信号很弱,压电最大值(196 个测试行)完全学不到;它们只应作为刻意设置的困难 / 小数据任务进入多任务集,或者不进。空间群(151 类)macro-F1 0.20、准确率 0.24:只凭组成能认出简单的高对称家族(Fm-3m、I4/mmm、Cmcm、P-62m 有 30–60% 被识别),而 Pnma、P2₁/c、C2/c 这类低对称群几乎认不出——它们由结构决定,不由化学计量决定。",
"<b>新規タスク。</b>バンドギャップ 0.88、CBM 0.88、VBM 0.92、is-metal macro-F1 0.93、体積弾性率 0.93、屈折率 0.89、剪断弾性率 0.79 は堅実な新規タスク。ポアソン比(0.31)、普遍異方性(0.35)、反応エネルギー(0.34)は組成の信号が弱く、圧電最大値(テスト行 196)は全く学習できない。これらは意図的な難タスク・少データタスクとしてのみマルチタスク集合に入れるか、入れない。空間群(151クラス)は macro-F1 0.20、正解率 0.24:組成だけで分かるのは単純な高対称族(Fm-3m、I4/mmm、Cmcm、P-62m は 30–60% 認識)で、Pnma、P2₁/c、C2/c のような低対称群はほぼ認識できない — 構造が決めるもので、化学量論では決まらない。", "callout")}
</div>
<figure style="margin-top:14px"><div class="figbox"><div class="legend"><span><i class="sw" style="background:var(--warm)"></i> {T("relabelled task","标签更新的任务","ラベル更新タスク")}</span><span><i class="sw" style="background:var(--frz)"></i> {T("added task","新增任务","新規タスク")}</span><span style="color:var(--muted)">{T("2,000 test rows sampled per task · seed 2025 · normalised scale · dashed = y = x","每任务抽样 2,000 个测试行 · seed 2025 · 归一化尺度 · 虚线为 y = x","タスクごとにテスト行 2,000 を抽出 · シード 2025 · 正規化スケール · 破線は y = x")}</span></div>
<svg id="fig-scatter" width="960" height="1440" role="img" aria-label="Observation versus prediction for every regression task"></svg></div>
{CAP("Observation (vertical) against prediction (horizontal) for one seed of every regression task; the R² is the five-seed mean.","每个回归任务一个 seed 的观测值(纵轴)对预测值(横轴);R² 为 5 个 seed 的均值。","各回帰タスク1シードの観測値(縦軸)対予測値(横軸)。R² は5シードの平均。")}</figure>
<figure><div class="figbox"><svg id="fig-cm" width="960" height="300" role="img" aria-label="Confusion matrices of the classification tasks"></svg></div>
{CAP("Row-normalised confusion matrices of the three small classification tasks, seed 2025; macro-F1 and accuracy are five-seed means.","三个小分类任务的行归一化混淆矩阵,seed 2025;macro-F1 与准确率为 5 个 seed 的均值。","3つの小規模分類タスクの行正規化混同行列、シード 2025。macro-F1 と正解率は5シードの平均。")}</figure>
<figure><div class="figbox"><svg id="fig-sg" width="960" height="620" role="img" aria-label="Space-group confusion over the twelve most common groups"></svg></div>
{CAP("Space group, seed 2025: the twelve most common groups shown individually, the other 139 folded into one row and column. Rows are true groups, cells are % of the row.","空间群,seed 2025:最常见的 12 个群单独显示,其余 139 个并入一行一列。行为真实群,格内为该行的 %。","空間群、シード 2025:最も多い12群を個別に、残り139群を1行1列にまとめて表示。行が真の群、セルは行内の%。")}</figure>
<figure><div class="figbox"><svg id="fig-sg-f1" width="960" height="330" role="img" aria-label="Per-class F1 against class size for the space-group task"></svg></div>
{CAP("Per-group F1 (seed 2025) against the number of entries in the group, log scale. Groups with hundreds of examples are learned; groups with tens mostly are not — the class imbalance is the physics of the crystal world, not a data defect, and it sets what macro-F1 can reach.","每个空间群的 F1(seed 2025)对该群条目数,对数坐标。有几百个样例的群学得到,只有几十个的大多学不到——类别不均衡是晶体世界的物理事实,不是数据缺陷,它决定了 macro-F1 能到多高。","各空間群の F1(シード 2025)対その群の項目数、対数軸。数百例ある群は学習でき、数十例の群はほとんど学習できない — クラスの不均衡は結晶世界の物理的事実であってデータの欠陥ではなく、macro-F1 の上限を決める。")}</figure>
</section>'''

# ---------------- 6 update brief ----------------
V = L["validation"]
val_rows = []
for col in ("Volume", "Density"):
    v = V[col]; val_rows.append([col, T("new / old","新 / 旧","新 / 旧"), f"{v['q01']:.3f}", f"{v['med']:.3f}", f"{v['q99']:.3f}", "-", f"{v['n']:,}"])
for col in ("Final energy per atom", "Formation energy per atom", "Band gap", "Efermi", "Total magnetization"):
    v = V[col]; val_rows.append([col, T("new − old","新 − 旧","新 − 旧"), f"{v['q01']:+.3f}", f"{v['med']:+.3f}", f"{v['q99']:+.3f}", f"{v['changed_share']*100:.1f}%", f"{v['n']:,}"])
newcols = [
 ("Final energy per atom", "GGA / GGA+U thermo energy", "GGA / GGA+U thermo 能量", "GGA / GGA+U thermo エネルギー", "33,166"),
 ("Formation energy per atom", "GGA / GGA+U thermo scheme", "GGA / GGA+U thermo 方案", "GGA / GGA+U thermo スキーム", "33,166"),
 ("Energy above hull · Equilibrium reaction energy per atom", "new, GGA / GGA+U", "新增,GGA / GGA+U", "新規、GGA / GGA+U", "33,166 · 30,689"),
 ("Volume · Density · Density atomic", "GGA entry structure; volume rescaled to the dataset's cell; density atomic is per atom", "GGA entry 的结构;体积换算到数据集原胞;density atomic 为每原子", "GGA エントリの構造。体積はデータセット単位胞に換算。density atomic は原子あたり", "33,166"),
 ("Total magnetization · per formula unit · per volume", "GGA task OUTCAR, |μB|; per cell rescaled like volume", "GGA 任务 OUTCAR,|μB|;每原胞按体积同样换算", "GGA タスクの OUTCAR、|μB|。単位胞あたりは体積と同様に換算", "33,166"),
 ("Magnetic ordering (label) · Number of magnetic sites", "pymatgen analyser on the GGA task's site moments: NM 22,257 · FM 7,544 · FiM 2,649 · AFM 716", "pymatgen 分析器作用于 GGA 任务的位点磁矩:NM 22,257 · FM 7,544 · FiM 2,649 · AFM 716", "GGA タスクのサイト磁気モーメントに pymatgen 分析器:NM 22,257 · FM 7,544 · FiM 2,649 · AFM 716", "33,166 · 33,144"),
 ("Band gap · Efermi · Is metal · Is gap direct", "summary values, kept only where the electronic-structure origin is GGA / GGA+U (2,983 r2SCAN + 56 SCAN origins blanked)", "summary 值,只在能带来源为 GGA / GGA+U 时保留(2,983 个 r2SCAN、56 个 SCAN 来源置空)", "summary 値、電子構造の由来が GGA / GGA+U の場合のみ保持(r2SCAN 由来 2,983、SCAN 由来 56 を空に)", "30,759 · 30,741"),
 ("CBM · VBM", "new, same gate", "新增,同样门控", "新規、同じ条件", "16,857"),
 ("Bulk modulus · Shear modulus · Poisson ratio · Universal anisotropy", "new, GGA deformation workflow, physical bounds applied", "新增,GGA 形变流程,加物理范围过滤", "新規、GGA 変形ワークフロー、物理範囲で制限", "6,551 · 6,466 · 6,499 · 6,124"),
 ("Dielectric total / ionic / electronic · Refractive index", "GGA DFPT; 9 out-of-bound rows dropped", "GGA DFPT;9 行超出范围被去除", "GGA DFPT。範囲外の9行を除外", "4,501 · 4,500"),
 ("Piezoelectric max", "new, GGA DFPT", "新增,GGA DFPT", "新規、GGA DFPT", "1,362"),
 ("MP id", "new-format Materials Project id (legacy mp-<n> stays the index; 31 ids no longer resolve)", "新格式 Materials Project id(旧 mp-<n> 仍为索引;31 个 id 已不可解析)", "新形式の Materials Project id(旧 mp-<n> は索引のまま。31 個は解決不能)", "33,798"),
 ("Space group (task label)", "dense encoding of the 151 space groups with ≥ 10 entries and a row in both train and test; rarer groups and non-MP rows are missing, like any other task", "151 个空间群(≥ 10 条且训练集、测试集各有一行)的密集编码;更稀有的群和非 MP 行按缺失处理,与其他任务一致", "10 項目以上かつ訓練・テスト両方に行がある 151 空間群の密な符号化。稀な群と非 MP 行は他タスク同様に欠損", "33,524"),
]
newcol_rows = [[c, T(a, b, d), n] for c, a, b, d, n in newcols]
out.append(f'''<section class="section">{H2("6 · Update brief", "Dataset 2026-09-12: what changed", "数据集 2026-09-12:变了什么", "データセット 2026-09-12:何が変わったか")}
<div class="col">{P("The first cause was misaligned data, not a modelling choice, so the Materials Project part of the dataset was audited column by column against the API and rebuilt on one level of theory: energies from the GGA / GGA+U thermodynamic scheme, structure and magnetism from the GGA-family calculation of each material, electronic-structure values only where MP's own record says the source was GGA-family. The audit also showed which further MP properties are available on that same footing; those were added.",
"第一个原因是数据对不齐,不是建模选择,所以数据集的 Materials Project 部分逐列对照 API 做了核查,并按单一理论水平重建:能量取 GGA / GGA+U 热力学方案;结构和磁性取每个材料的 GGA 族计算;能带类只在 MP 自己的记录表明来源是 GGA 族时保留。核查同时显示了在同一口径下还有哪些 MP 属性可用,这些也一并补入。",
"第一の原因はデータの不整合であってモデリング上の選択ではない。そこでデータセットの Materials Project 部分を列ごとに API と照合して監査し、単一の理論レベルで再構築した:エネルギーは GGA / GGA+U 熱力学スキームから、構造と磁性は各物質の GGA 系計算から、電子構造の値は MP 自身の記録が GGA 系由来と示す場合のみ。監査は同じ基準で利用できる他の MP 物性も明らかにし、それらを追加した。")}</div>
<div class="stats">
<div class="stat"><p class="label">{T("rows","行数","行数")}</p><p class="value">{L['shape_new'][0]:,}</p><p class="note">{T("unchanged; qa- and starry- rows byte-identical","不变;qa- 与 starry- 行逐字节相同","不変。qa- と starry- 行はバイト単位で同一")}</p></div>
<div class="stat"><p class="label">{T("columns","列数","列数")}</p><p class="value">{L['shape_old'][1]} → {L['shape_new'][1]}</p><p class="note">{T("18 new properties with their normalised columns and labels, and the space-group task label","18 个新属性及其归一化列和标签,加空间群任务标签","18の新規物性とその正規化列・ラベル、および空間群タスクラベル")}</p></div>
<div class="stat"><p class="label">{T("MP rows on GGA / GGA+U","GGA / GGA+U 口径的 MP 行","GGA / GGA+U 基準の MP 行")}</p><p class="value">33,166</p><p class="note">{T("of 33,829; 663 have no GGA-family calculation and are blank","共 33,829;663 个没有 GGA 族计算,置空","33,829 中。663 は GGA 系計算がなく空欄")}</p></div>
<div class="stat"><p class="label">{T("final_energy alone","final_energy 单独训练","final_energy 単独学習")}</p><p class="value c-warm">{m(fe_new):.4f}</p><p class="note">{T(f"was {m(fe_old):.4f} on the old label","旧标签下为 " + f"{m(fe_old):.4f}","旧ラベルでは " + f"{m(fe_old):.4f}")}</p></div>
</div>
<div class="col">{P("Files: <code>data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet</code>, <code>data/preprocessing_objects_20260912.pkl.z</code>, <code>data/qc_ac_te_mp_dos_reformat_20260912_CHANGES.md</code>; built by <code>data/data/scripts/rebuild_mp_gga_20260912.py</code> from API pulls made on 2026-09-12. The 2026-05-15 files are untouched.",
"文件:<code>data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet</code>、<code>data/preprocessing_objects_20260912.pkl.z</code>、<code>data/qc_ac_te_mp_dos_reformat_20260912_CHANGES.md</code>;由 <code>data/data/scripts/rebuild_mp_gga_20260912.py</code> 从 2026-09-12 的 API 拉取生成。2026-05-15 的文件未动。",
"ファイル:<code>data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet</code>、<code>data/preprocessing_objects_20260912.pkl.z</code>、<code>data/qc_ac_te_mp_dos_reformat_20260912_CHANGES.md</code>。<code>data/data/scripts/rebuild_mp_gga_20260912.py</code> が 2026-09-12 の API 取得から生成。2026-05-15 のファイルは未変更。")}</div>
{table([T("column(s)","列","列"), T("source in the rebuilt dataset","重建数据集中的来源","再構築データセットでの出典"), T("MP rows","MP 行数","MP 行数")], newcol_rows, cls="tablebox wrap")}
<figure style="margin-top:14px"><div class="figbox"><svg id="fig-cols" width="960" height="660" role="img" aria-label="Non-null rows per column before and after"></svg></div>
{CAP("Coverage per column. Columns that existed lose the 663 materials without a GGA-family calculation and, for the electronic structure, the 3,039 whose MP origin is not GGA-family; the new columns cover what their GGA workflows cover.",
"逐列覆盖率。原有列失去 663 个没有 GGA 族计算的材料,能带类再失去 3,039 个 MP 来源不是 GGA 族的;新列覆盖各自 GGA 流程覆盖的范围。",
"列ごとの網羅率。既存の列は GGA 系計算のない 663 物質を失い、電子構造はさらに MP の由来が GGA 系でない 3,039 を失う。新規列はそれぞれの GGA ワークフローが対象とする範囲を持つ。")}</figure>
<div class="col"><h3>{T("Validation against the 2026-05-15 dataset","对 2026-05-15 数据集的校验","2026-05-15 データセットとの検証")}</h3></div>
{table([T("column","列","列"), T("comparison","比较","比較"), "1%", T("median","中位数","中央値"), "99%", T("|Δ| > 0.05","|Δ| > 0.05","|Δ| > 0.05"), T("rows","行数","行数")], val_rows, style="max-width:1000px")}
<div class="col" style="margin-top:16px">
{P("Volume moves by +1.4% at the median — GGA lattices are slightly larger than r2SCAN ones — and Density inversely. Final energy changes only where the old value was r2SCAN. Band gap and Efermi differences are Materials Project's own updates to its electronic-structure builder since 2025-04, now consistently GGA-family. Total magnetisation changes where the cell or the functional changed.",
"Volume 中位变化 +1.4%——GGA 晶格略大于 r2SCAN——Density 反向。Final energy 只在旧值是 r2SCAN 的地方变化。Band gap 和 Efermi 的差异是 Materials Project 自 2025-04 以来对能带构建的更新,现在统一为 GGA 族。总磁矩在原胞或泛函变化的地方变化。",
"Volume は中央値で +1.4% 動く — GGA 格子は r2SCAN よりわずかに大きい — Density はその逆。Final energy は旧値が r2SCAN だった箇所でのみ変わる。Band gap と Efermi の差は Materials Project 側の 2025-04 以降の電子構造ビルダー更新で、今は一貫して GGA 系。全磁化は単位胞または汎関数が変わった箇所で変わる。")}
{P("Spot checks on the rebuilt labels: Fe (bcc) −8.470 eV/atom, 2.18 μB per atom, FM; Si −5.425 eV/atom, band gap 0.610 eV; Pt −6.071 eV/atom; Fe₂O₃ formation energy −1.707 eV/atom, AFM — all on the GGA / GGA+U reference values.",
"重建标签的抽查:Fe(bcc)−8.470 eV/atom、每原子 2.18 μB、FM;Si −5.425 eV/atom、带隙 0.610 eV;Pt −6.071 eV/atom;Fe₂O₃ 生成能 −1.707 eV/atom、AFM——全部落在 GGA / GGA+U 参考值上。",
"再構築ラベルの抜き取り検査:Fe(bcc)−8.470 eV/atom、原子あたり 2.18 μB、FM。Si −5.425 eV/atom、バンドギャップ 0.610 eV。Pt −6.071 eV/atom。Fe₂O₃ 生成エネルギー −1.707 eV/atom、AFM — すべて GGA / GGA+U の基準値上にある。")}
<h3 style="margin-top:22px">{T("Next","下一步","次のステップ")}</h3>
<ol class="plan">
<li>{T("<b>Choose the multi-task set on the new data.</b> Section 8 gives every relabelled and added task its baseline; band gap, per-atom volume, magnetisation per volume, magnetic ordering, is-metal, bulk and shear modulus and refractive index earn a place; Poisson ratio, universal anisotropy, reaction energy and piezoelectric maximum do not.","<b>在新数据上确定多任务集。</b>第 8 节给出了每个标签更新和新增任务的基线;band gap、每原子体积、每体积磁化、磁序、is-metal、体模量与剪切模量、折射率有资格进入;泊松比、通用各向异性、反应能、压电最大值不进。","<b>新データでのマルチタスク集合を決める。</b>第8節がラベル更新・新規の全タスクに基準を与えた。バンドギャップ、原子あたり体積、体積あたり磁化、磁気秩序、is-metal、体積・剪断弾性率、屈折率は採用に値する。ポアソン比、普遍異方性、反応エネルギー、圧電最大値は採用しない。")}</li>
<li>{T("<b>Re-run the transfer stages on the 2026-09-12 dataset</b> with that set once the unseen-encoder stage on the old data has closed the 2×2; volume enters through its per-atom form, or with a scale-aware descriptor.","<b>在 2026-09-12 数据集上重跑迁移阶段</b>,待旧数据的未见过编码器阶段把 2×2 收尾后进行;volume 以每原子形式进入,或改用能感知尺度的描述符。","<b>2026-09-12 データセットで転移段階を再実行</b>する。旧データでの未知エンコーダ段階が 2×2 を完成させた後に。volume は原子あたりの形で、あるいはスケール認識記述子で入れる。")}</li>
<li>{T("<b>Choose the descriptor policy</b>: KMD stays invertible (needed for inverse design) but is scale-blind; XenonPy classic sees scale but is not invertible. Either keep both and train per-atom labels with KMD, or carry the atom count as a side input.","<b>决定描述符策略</b>:KMD 可逆(逆向设计需要)但尺度盲;XenonPy classic 看得见尺度但不可逆。要么两者都保留、用 KMD 训练每原子标签,要么把原子数作为旁路输入。","<b>記述子の方針を決める</b>:KMD は可逆(逆設計に必要)だがスケールに盲目。XenonPy classic はスケールが見えるが可逆ではない。両方を保持して KMD で原子あたりラベルを学習するか、原子数を副入力として持たせるか。")}</li>
</ol></div>
</section>
{props_section()}
{baselines_section()}
<footer><p>{T("Data: summary/mp_labels_page_data.json (per-seed results, curves, API comparisons, dataset statistics); runs stA_*, stM_final_energy_*, stXc_* and stXn_* on RIKYU; MP API queries of 2026-09-12. Figure labels stay in English in every language.",
"数据:summary/mp_labels_page_data.json(逐 seed 结果、曲线、API 比对、数据集统计);RIKYU 上的 stA_*、stM_final_energy_*、stXc_*、stXn_*;2026-09-12 的 MP API 查询。图内标签在各语言下均保留英文。",
"データ:summary/mp_labels_page_data.json(シードごとの結果、曲線、API 比較、データセット統計)。RIKYU 上の stA_*、stM_final_energy_*、stXc_*、stXn_*。2026-09-12 の MP API 照会。図中のラベルはどの言語でも英語のまま。")}</p></footer>
</div>
<script>
{LANGJS}
</script>
<script>
const D = {json.dumps(D, separators=(",", ":"))};
{JS}
</script>
''')

OUT.write_text("".join(out), encoding="utf-8")
print(f"  wrote {OUT} ({len(''.join(out)) // 1024} KB)")
