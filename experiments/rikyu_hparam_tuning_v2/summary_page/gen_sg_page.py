# -*- coding: utf-8 -*-
"""Generate "Space group: 0.24 here, 0.60 in the paper" — summary_page/space_group_summary.html

The investigation of why the pipeline's single-task space-group classifier scores top-1 0.24 while
the ShotgunCSP classifier (Liu et al., npj Comput. Mater. 2024, Fig. 3) scores 0.60 on the same
population. English is the source text; Chinese and Japanese follow it behind the switch. Figures
are drawn by sg_page.js from the SG constant built here out of:

  summary/space_group_study_table.json   the local factorial (16 arms × 3 seeds), aggregated
  summary/space_group_study.json         per-run top-k recall
  summary/space_group_confirm.json       the pipeline confirmation on RIKYU (stN / stW / stX, 5 seeds)
  summary/space_group_perclass.json      per-class recall of the three pipeline arms, seed 2025
  summary/space_group_atoms.json         atoms-per-cell evidence
  summary/space_group_classes.json       the 151 classes and their counts

    python summary_page/gen_sg_page.py
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
S = EXP / "summary"
STYLE = (HERE / "_shared_style.html").read_text(encoding="utf-8")
LANGBAR = (HERE / "_shared_langbar.html").read_text(encoding="utf-8")
LANGJS = (HERE / "_shared_lang.js").read_text(encoding="utf-8")
JS = (HERE / "sg_page.js").read_text(encoding="utf-8")
OUT = HERE / "space_group_summary.html"

tab = {a["arm"]: a for a in json.load(open(S / "space_group_study_table.json"))["arms"]}
runs = json.load(open(S / "space_group_study.json"))["runs"]
conf = json.load(open(S / "space_group_confirm.json"))["arms"]
perc = json.load(open(S / "space_group_perclass.json"))
atoms = json.load(open(S / "space_group_atoms.json"))
classes = json.load(open(S / "space_group_classes.json"))
PAPER = {"top1": 0.6022, "top10": 0.8535, "top30": 0.9261, "top40": 0.9402, "sd1": 0.0087}


def T(en, zh, ja):
    return f'<span class="l-en">{en}</span><span class="l-zh">{zh}</span><span class="l-ja">{ja}</span>'


def P(en, zh, ja, cls=""):
    return f'<p class="{cls}">{T(en, zh, ja)}</p>' if cls else f"<p>{T(en, zh, ja)}</p>"


def CAP(en, zh, ja):
    return f"<figcaption>{T(en, zh, ja)}</figcaption>"


def FH(en, zh, ja):
    return f'<div class="fighead"><span class="fignum"></span><span class="figtitle">{T(en, zh, ja)}</span></div>'


def TH(en, zh, ja):
    return f'<div class="tabhead"><span class="tabnum"></span><span class="figtitle">{T(en, zh, ja)}</span></div>'


def H2(kick, en, zh, ja):
    return f'<div class="col"><p class="kicker">{kick}</p><h2>{T(en, zh, ja)}</h2></div>'


def table(head, rows, cls="tablebox", style=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="{cls}"{(" style=" + chr(34) + style + chr(34)) if style else ""}><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def last(arm, k="accuracy"):
    return tab[arm]["last"][k]["mean"]


def lsd(arm, k="accuracy"):
    return tab[arm]["last"][k]["sd"]


def conf_stats(tag):
    rs = conf[tag]["runs"]
    acc = [r["accuracy"] for r in rs]; f1 = [r["macro_f1"] for r in rs]
    return {"n": len(rs), "acc": st.fmean(acc), "acc_sd": st.stdev(acc), "f1": st.fmean(f1), "f1_sd": st.stdev(f1),
            "epochs": st.fmean([r["epochs"] for r in rs]), "best": st.fmean([r["best_epoch"] for r in rs]),
            "vl": (rs[0]["val_loss_first"], rs[0]["val_loss_min"], rs[0]["val_loss_last"])}


CN, CW, CX = conf_stats("balanced_kmd"), conf_stats("plain_kmd"), conf_stats("plain_classic")
pc = lambda v: f"{v * 100:.1f}%"
f4 = lambda v: f"{v:.4f}"

# ---------------------------------------------------------------- data for the figures
ladder = [
    {"label": "replica: KMD, balanced weights, pipeline recipe", "acc": last("fm_replica"), "sd": lsd("fm_replica"), "kind": "step"},
    {"label": "+ unweighted cross-entropy", "acc": last("fm_plain"), "sd": lsd("fm_plain"), "kind": "step"},
    {"label": "+ the paper's FC-NN", "acc": last("paper_model_fm_recipe"), "sd": lsd("paper_model_fm_recipe"), "kind": "step"},
    {"label": "+ XenonPy classic descriptor (= paper setup)", "acc": last("paper_full_classic"), "sd": lsd("paper_full_classic"), "kind": "step"},
    {"label": "RIKYU pipeline, as run (stN)", "acc": CN["acc"], "sd": CN["acc_sd"], "kind": "pipe"},
    {"label": "RIKYU pipeline, class_weights = none (stW)", "acc": CW["acc"], "sd": CW["acc_sd"], "kind": "pipe"},
    {"label": "RIKYU pipeline, none + XenonPy classic (stX)", "acc": CX["acc"], "sd": CX["acc_sd"], "kind": "pipe"},
]


def topk_mean(arm):
    rs = [r for r in runs if r["arm"] == arm]
    return [st.fmean(r["last"][f"top{k}"] for r in rs) for k in (1, 5, 10, 30)]


topk = [
    {"label": "replica (as run)", "k": [1, 5, 10, 30], "rec": topk_mean("fm_replica"), "color": "var(--alone)"},
    {"label": "unweighted, KMD", "k": [1, 5, 10, 30], "rec": topk_mean("fm_plain"), "color": "var(--frz)"},
    {"label": "paper net, KMD", "k": [1, 5, 10, 30], "rec": topk_mean("paper_full_kmd"), "color": "var(--frz)", "dash": "5 4"},
    {"label": "paper net, XenonPy classic", "k": [1, 5, 10, 30], "rec": topk_mean("paper_full_classic"), "color": "var(--warm)"},
    {"label": "paper, Fig. 3 (213 classes)", "k": [1, 10, 30, 40], "rec": [PAPER["top1"], PAPER["top10"], PAPER["top30"], PAPER["top40"]], "color": "var(--xfer)", "dash": "2 3"},
]
D = lambda a, b: last(a) - last(b)
effects = [
    {"label": "loss: unweighted − balanced", "color": "var(--xfer)", "deltas": [D("fm_plain", "fm_replica"), D("fm_plain_classic", "fm_replica_classic"), D("paper_model_fm_recipe", "paper_model_fm_recipe_bal"), D("paper_full_kmd", "paper_full_kmd_bal")]},
    {"label": "descriptor: XenonPy classic − KMD", "color": "var(--warm)", "deltas": [D("fm_plain_classic", "fm_plain"), D("paper_full_classic", "paper_full_kmd"), D("fm_replica_classic", "fm_replica")]},
    {"label": "descriptor: classic w/o sum block − KMD", "color": "var(--warm)", "deltas": [D("paper_full_nosum", "paper_full_kmd"), D("paper_full_nosum_rsplit", "paper_full_kmd_rsplit")]},
    {"label": "model: paper net − pipeline shape (unweighted)", "color": "var(--frz)", "deltas": [D("paper_model_fm_recipe", "fm_plain_scaled"), D("paper_full_kmd", "fm_model_paper_recipe")]},
    {"label": "model: paper net − pipeline shape (balanced)", "color": "var(--frz)", "deltas": [D("paper_model_fm_recipe_bal", "fm_replica_scaled")]},
    {"label": "recipe: paper − pipeline optimiser", "color": "var(--alone)", "deltas": [D("fm_model_paper_recipe", "fm_plain_scaled"), D("paper_full_kmd", "paper_model_fm_recipe")]},
    {"label": "input scaling: shotgun chain − none", "color": "var(--alone)", "deltas": [D("fm_plain_scaled", "fm_plain"), D("fm_replica_scaled", "fm_replica")]},
    {"label": "split: random 80/10/10 − dataset split", "color": "var(--alone)", "deltas": [D("paper_full_kmd_rsplit", "paper_full_kmd"), D("paper_full_nosum_rsplit", "paper_full_nosum"), D("fm_replica_rsplit", "fm_replica")]},
    {"label": "weights: best-val-loss − last epoch", "color": "var(--alone)", "deltas": [tab[a]["best"]["accuracy"]["mean"] - tab[a]["last"]["accuracy"]["mean"] for a in ("fm_replica", "fm_plain", "paper_full_kmd", "paper_full_classic", "fm_plain_classic")]},
]
eff = {e["label"]: st.fmean(e["deltas"]) for e in effects}

groups12 = sorted(classes["counts"].items(), key=lambda kv: -kv[1])[:12]
recall = {"groups": [g for g, _ in groups12], "n_test": [perc["arms"]["balanced_kmd"]["per_class"][g]["n_test"] for g, _ in groups12],
          "rec": {arm: [perc["arms"][arm]["per_class"][g]["recall"] for g, _ in groups12] for arm in ("balanced_kmd", "plain_kmd", "plain_classic")}}
bins = atoms["atoms_hist_bins"]
labels = [f"{bins[i]}–{bins[i + 1] - 1}" if bins[i + 1] - 1 > bins[i] else f"{bins[i]}" for i in range(len(bins) - 1)]
labels[-1] = f"≥ {bins[-2]}"
hist = {g: [v / sum(vs) for v in vs] for g, vs in atoms["atoms_hist"].items()}
atoms_fig = {"labels": labels, "hist": hist, "n": {g: atoms["atoms_by_group"][g]["n"] for g in hist}}
cnts = sorted(classes["counts"].values())
tot = sum(cnts)
weights = [[c, tot / (151 * c)] for c in cnts]
w_min, w_max = min(w for _, w in weights), max(w for _, w in weights)
big3 = sum(cnts[-3:]) / tot
big3_mass = sum(c * (tot / (151 * c)) for c in cnts[-3:]) / sum(c * (tot / (151 * c)) for c in cnts)
small = [c for c in cnts if c < 30]
small_rows = sum(small) / tot
small_mass = sum(c * (tot / (151 * c)) for c in small) / sum(c * (tot / (151 * c)) for c in cnts)
SG = {"paper_top1": PAPER["top1"], "ladder": ladder, "topk": topk, "effects": effects, "recall": recall, "atoms": atoms_fig, "weights": weights}

# ---------------------------------------------------------------- arms table
ARM_LABEL = {
    "fm_replica": "pipeline replica", "fm_plain": "replica, unweighted CE", "fm_plain_scaled": "… + shotgun scaling", "fm_model_paper_recipe": "pipeline shape, paper optimiser",
    "paper_model_fm_recipe_bal": "paper net, pipeline optimiser, balanced", "paper_model_fm_recipe": "paper net, pipeline optimiser", "paper_full_kmd": "paper net + paper optimiser, KMD",
    "paper_full_kmd_bal": "… with balanced weights", "paper_full_classic": "paper setup, XenonPy classic (290)", "paper_full_nosum": "paper setup, classic without sum block (232)",
    "fm_replica_classic": "pipeline replica, XenonPy classic", "fm_plain_classic": "replica unweighted, XenonPy classic", "fm_replica_scaled": "replica + shotgun scaling",
    "paper_full_kmd_rsplit": "paper setup, KMD, random split", "paper_full_nosum_rsplit": "paper setup, no-sum, random split", "fm_replica_rsplit": "pipeline replica, random split",
}
arm_rows = []
for arm, a in tab.items():
    f = a["factors"]
    la, bb = a["last"], a["best"]
    cell = lambda m, k: f"{m[k]['mean']:.3f}"
    acc = f"<b>{la['accuracy']['mean']:.4f}</b> ± {la['accuracy']['sd']:.3f}"
    arm_rows.append([f"<b>{ARM_LABEL[arm]}</b><br><span style='color:var(--muted);font-size:12.5px'>{arm}</span>",
                     f["descriptor"], f["scaling"], f["model"], f["loss"], f["recipe"], f.get("split", "dataset"),
                     f"{a['epochs'][0]:.0f} / {a['best_epoch'][0]:.0f}", acc, cell(la, "macro_f1"), cell(la, "top5"),
                     cell(bb, "accuracy"), cell(bb, "top10"), cell(bb, "top30")])

confirm_rows = [
    [T("as run — balanced weights, KMD (stN)", "原样——balanced 权重,KMD(stN)", "実行時のまま — balanced 重み、KMD(stN)"), "5", f"<b>{CN['acc']:.4f}</b> ± {CN['acc_sd']:.4f}", f"{CN['f1']:.4f} ± {CN['f1_sd']:.4f}", f"{CN['epochs']:.0f} / {CN['best']:.0f}", f"{CN['vl'][0]} → {CN['vl'][1]} → {CN['vl'][2]}"],
    [T("class_weights = none, KMD (stW)", "class_weights = none,KMD(stW)", "class_weights = none、KMD(stW)"), "5", f"<b>{CW['acc']:.4f}</b> ± {CW['acc_sd']:.4f}", f"{CW['f1']:.4f} ± {CW['f1_sd']:.4f}", f"{CW['epochs']:.0f} / {CW['best']:.0f}", f"{CW['vl'][0]} → {CW['vl'][1]} → {CW['vl'][2]}"],
    [T("class_weights = none, XenonPy classic (stX)", "class_weights = none,XenonPy classic(stX)", "class_weights = none、XenonPy classic(stX)"), "5", f"<b>{CX['acc']:.4f}</b> ± {CX['acc_sd']:.4f}", f"{CX['f1']:.4f} ± {CX['f1_sd']:.4f}", f"{CX['epochs']:.0f} / {CX['best']:.0f}", f"{CX['vl'][0]} → {CX['vl'][1]} → {CX['vl'][2]}"],
]
rec = perc["arms"]
r_big = lambda arm, g: rec[arm]["per_class"][g]["recall"]

out = []
out.append("<title>Space Group Gap</title>\n" + STYLE + "\n<style>.tablebox.prose td{white-space:normal;font-family:var(--f-body);text-align:left;vertical-align:top;color:var(--ink-2)}.tablebox.prose td:first-child{color:var(--ink)}</style>\n" + LANGBAR + '<div class="wrap">')

out.append(f'''<header>
<p class="eyebrow">{T("Continual multi-task pretraining · space-group task · investigation, 2026-09-11","连续多任务预训练 · 空间群任务 · 调查,2026-09-11","継続的マルチタスク事前学習 · 空間群タスク · 調査、2026-09-11")}</p>
<h1>{T("Space group: 0.24 here, 0.60 in the paper","空间群:这里 0.24,论文里 0.60","空間群:ここでは 0.24、論文では 0.60")}</h1>
<p class="standfirst">{T(
f"The single-task space-group baseline on the 2026-09-11 dataset scored top-1 accuracy {CN['acc']:.3f} over 151 groups. The ShotgunCSP classifier (Liu et al., npj Comput. Mater. 2024, Fig. 3) — a fully connected network on XenonPy's 290 compositional features, trained on the same population of stable Materials Project entries — reaches 60.2% over 213 groups. The question was whether the descriptor, the model or the training is to blame. The answer is two causes and one minor one, measured on the same rows, label and split: the pipeline's <b>class-balanced loss</b> costs about 21 points on a 151-class head; the <b>KMD descriptor</b> cannot see the number of atoms in the cell, which the space group constrains, and that costs about 11 more; the paper's wider network adds 4. Optimiser, input scaling, split and checkpoint choice change nothing. With all three switched the paper's number is reproduced on our data ({last('paper_full_classic'):.3f}), and the first two are confirmed inside the pipeline on RIKYU ({CN['acc']:.3f} → {CW['acc']:.3f} → {CX['acc']:.3f}).",
f"2026-09-11 数据集上的单任务空间群基线在 151 个空间群上 top-1 准确率只有 {CN['acc']:.3f}。ShotgunCSP 的分类器(Liu et al., npj Comput. Mater. 2024, 图 3)——XenonPy 290 维组成特征上的全连接网络,训练数据是同一批 Materials Project 稳定条目——在 213 个空间群上达到 60.2%。问题是:该怪描述符、模型结构还是训练?答案是两个主因加一个次因,全部在同一批行、同一标签、同一划分上测得:pipeline 的<b>类平衡损失</b>在 151 类的 head 上损失约 21 个百分点;<b>KMD 描述符</b>看不见原胞里的原子数,而空间群恰恰约束着它,再损失约 11 个百分点;论文那个更宽的网络再贡献 4 个。优化器、输入缩放、划分方式和权重选取都没有影响。三项都切换后,论文的数字在我们的数据上得以复现({last('paper_full_classic'):.3f});前两项已在 RIKYU 的 pipeline 内部确认({CN['acc']:.3f} → {CW['acc']:.3f} → {CX['acc']:.3f})。",
f"2026-09-11 データセットでの単独学習の空間群ベースラインは、151 群に対して top-1 正解率 {CN['acc']:.3f} だった。ShotgunCSP の分類器(Liu et al., npj Comput. Mater. 2024, 図 3)— XenonPy の 290 次元組成特徴に対する全結合ネットワークで、同じ Materials Project 安定エントリ集団で学習 — は 213 群に対して 60.2% に達する。問いは、記述子・モデル・学習のどれが原因か。答えは主因2つと副因1つで、すべて同じ行・同じラベル・同じ分割で測定した:パイプラインの<b>クラス平衡損失</b>は 151 クラスのヘッドで約 21 ポイントを失わせる。<b>KMD 記述子</b>は単位胞の原子数が見えず、空間群はまさにそれを制約するため、さらに約 11 ポイント。論文のより広いネットワークが 4 ポイントを加える。最適化器・入力スケーリング・分割・重みの選び方は何も変えない。3つを切り替えると論文の数値が我々のデータで再現され({last('paper_full_classic'):.3f})、前2つは RIKYU のパイプライン内部で確認済み({CN['acc']:.3f} → {CW['acc']:.3f} → {CX['acc']:.3f})。")}</p>
</header>''')

# ---------------- 1 the two setups ----------------
setup_rows = [
    [T("population", "数据总体", "母集団"), T("33,040 stable Materials Project entries (2022 dump), 120 benchmark crystals removed", "33,040 个 Materials Project 稳定条目(2022 快照),去掉 120 个 benchmark 晶体", "Materials Project の安定エントリ 33,040 件(2022 ダンプ)、ベンチマーク結晶 120 件を除外"), T("33,524 Materials Project rows of the 2026-09-11 dataset with a kept space-group label (same stable-only population, 2025 dump)", "2026-09-11 数据集里带保留空间群标签的 33,524 个 Materials Project 行(同样只含稳定条目,2025 快照)", "2026-09-11 データセットのうち空間群ラベルを持つ Materials Project 行 33,524 件(同じ安定エントリのみの母集団、2025 ダンプ)")],
    [T("classes", "类别数", "クラス数"), T("213 — every group present", "213——库中出现的全部空间群", "213 — 存在する全群"), T("151 — groups with ≥ 10 rows and a row in both train and test; rarer groups are missing labels (0.8% of rows)", "151——至少 10 行且训练集、测试集各有一行的空间群;更稀有的按缺失处理(0.8% 的行)", "151 — 10 行以上かつ訓練・テスト両方に行がある群。より稀な群は欠損扱い(行の 0.8%)")],
    [T("split", "划分", "分割"), T("random 4 : 1, repeated 100 times", "随机 4 : 1,重复 100 次", "ランダム 4 : 1、100 回反復"), T("the dataset's own split: 23,487 / 5,006 / 5,031", "数据集自带划分:23,487 / 5,006 / 5,031", "データセット自身の分割:23,487 / 5,006 / 5,031")],
    [T("descriptor", "描述符", "記述子"), T("XenonPy classic, 290 columns: weighted sum / average / variance / max / min over 58 element properties", "XenonPy classic,290 列:58 个元素性质的 weighted sum / average / variance / max / min", "XenonPy classic、290 列:58 元素物性の weighted sum / average / variance / max / min"), T("KMD, 464 columns: kernel means of the same 58 properties over the <em>atomic fractions</em> (n_grids = 8) — scale-free by construction", "KMD,464 列:同样 58 个性质在<em>原子分数</em>上的核均值(n_grids = 8)——按定义与原胞大小无关", "KMD、464 列:同じ 58 物性の<em>原子分率</em>上のカーネル平均(n_grids = 8)— 定義上スケール非依存")],
    [T("model", "模型", "モデル"), T("FC-NN, 2–4 layers, widths decaying ×0.8–0.95, dropout 0–0.2, Optuna-tuned (200 trials, 5-fold CV); the 2026 refit: 4 × Linear→Dropout(0.23)→GELU, widths 345 → 296 → 255 → 219", "全连接网络,2–4 层,宽度按 ×0.8–0.95 递减,dropout 0–0.2,Optuna 调参(200 次,5 折 CV);2026 年重训版:4 × Linear→Dropout(0.23)→GELU,宽度 345 → 296 → 255 → 219", "全結合ネットワーク、2–4 層、幅は ×0.8–0.95 で減衰、dropout 0–0.2、Optuna 調整(200 試行、5 分割 CV)。2026 年再学習版:4 × Linear→Dropout(0.23)→GELU、幅 345 → 296 → 255 → 219"), T("shared encoder 464 → 256 → 384 (BatchNorm + LeakyReLU) and a head 384 → 64 → 151; the adopted stage_single shape", "共享编码器 464 → 256 → 384(BatchNorm + LeakyReLU)加 head 384 → 64 → 151;即 stage_single 采用的形状", "共有エンコーダ 464 → 256 → 384(BatchNorm + LeakyReLU)とヘッド 384 → 64 → 151。stage_single で採用した形")],
    [T("loss", "损失", "損失"), T("plain cross-entropy", "普通交叉熵", "通常の交差エントロピー"), T(f"cross-entropy weighted by the inverse class frequency, w<sub>c</sub> = N / (151 · N<sub>c</sub>) — the same formula as scikit-learn's “balanced” convention, computed by the catalog and passed to PyTorch's cross-entropy; applied to every classification head, unconditionally; here ×{w_min:.3f} for Fm-3m up to ×{w_max:.1f} for a 10-row group", f"按类频率倒数加权的交叉熵,w<sub>c</sub> = N / (151 · N<sub>c</sub>)——与 scikit-learn “balanced” 约定同一公式,由 catalog 计算后交给 PyTorch 的交叉熵;对每个分类 head 无条件启用;这里 Fm-3m 为 ×{w_min:.3f},10 行的群为 ×{w_max:.1f}", f"クラス頻度の逆数で重み付けした交差エントロピー、w<sub>c</sub> = N / (151 · N<sub>c</sub>) — scikit-learn の “balanced” 規約と同じ式で、カタログが計算して PyTorch の交差エントロピーに渡す。全分類ヘッドに無条件で適用。ここでは Fm-3m が ×{w_min:.3f}、10 行の群が ×{w_max:.1f}")],
    [T("training", "训练", "学習"), T("Adam, early stopping, batch 1,024; hyper-parameters from the search", "Adam,早停,batch 1,024;超参数来自搜索", "Adam、早期終了、バッチ 1,024。ハイパーパラメータは探索から"), T("AdamW (encoder lr 2e-3, head 5e-3), ReduceLROnPlateau on the train loss, early stopping on the validation loss (patience 24), batch 256, last-epoch weights", "AdamW(编码器 lr 2e-3,head 5e-3),ReduceLROnPlateau 看训练损失,早停看验证损失(patience 24),batch 256,取最后一个 epoch 的权重", "AdamW(エンコーダ lr 2e-3、ヘッド 5e-3)、学習損失で ReduceLROnPlateau、検証損失で早期終了(patience 24)、バッチ 256、最終エポックの重み")],
    [T("reported", "报告的数字", "報告値"), T("top-1 recall 60.22 ± 0.87% (top-10 85.35%, top-30 92.61%)", "top-1 recall 60.22 ± 0.87%(top-10 85.35%,top-30 92.61%)", "top-1 recall 60.22 ± 0.87%(top-10 85.35%、top-30 92.61%)"), T(f"accuracy {CN['acc']:.4f} ± {CN['acc_sd']:.4f}, macro-F1 {CN['f1']:.4f} (5 seeds)", f"准确率 {CN['acc']:.4f} ± {CN['acc_sd']:.4f},macro-F1 {CN['f1']:.4f}(5 seed)", f"正解率 {CN['acc']:.4f} ± {CN['acc_sd']:.4f}、macro-F1 {CN['f1']:.4f}(5 シード)")],
]
out.append(f'''<section class="section">{H2("1 · The two setups", "What the paper did, and what the pipeline does", "论文怎么做的,pipeline 怎么做的", "論文の設定と、パイプラインの設定")}
<div class="col">{P("The two numbers are not measured the same way, so the first step was to list every difference. Six of them could matter: the class set, the split, the descriptor, the network, the loss, and the optimiser with its stopping and weight-selection rules. The population is the same — stable Materials Project entries — so the data itself is not on the list.",
"两个数字的测法不同,所以第一步是把所有差异列出来。可能起作用的有六项:类别集合、划分、描述符、网络、损失、以及优化器连同它的停止和取权重规则。数据总体相同——都是 Materials Project 的稳定条目——所以数据本身不在清单上。",
"2つの数値は同じ方法で測られていないので、まず全ての違いを列挙した。効きうるのは6つ:クラス集合、分割、記述子、ネットワーク、損失、そして停止と重み選択の規則を含む最適化器。母集団は同じ — Materials Project の安定エントリ — なので、データ自体はリストに入らない。")}</div>
{TH("Paper versus pipeline, factor by factor", "论文 vs pipeline,逐项对照", "論文とパイプライン、要因ごとの対照")}
{table([T("factor","因素","要因"), T("ShotgunCSP paper (2024) and its 2026 refit","ShotgunCSP 论文(2024)及 2026 年重训","ShotgunCSP 論文(2024)と 2026 年再学習"), T("this pipeline, stage_single","本 pipeline,stage_single","本パイプライン、stage_single")], setup_rows, cls="tablebox prose")}
</section>''')

# ---------------- 2 method ----------------
out.append(f'''<section class="section">{H2("2 · Method", "One factor at a time, on the same rows", "同一批数据上,每次只换一个因素", "同じ行の上で、一度に一要因ずつ")}
<div class="col">{P(f"A local harness trains every arm on the same 23,487 training rows, the same 151-class label and the same validation and test splits the pipeline used. The pipeline arm is built from the project's own encoder and classification-head classes with the adopted shape, loss and optimiser, and it reproduces the RIKYU run to the third decimal (top-1 {last('fm_replica'):.3f} ± {lsd('fm_replica'):.3f} against {CN['acc']:.3f} ± {CN['acc_sd']:.3f}, stopping at the same epoch for the same reason). The paper arm is the network and optimiser persisted with the 2026 refit of the classifier, applied through the same harness. Between them, sixteen arms switch one factor at a time: the descriptor (KMD; XenonPy classic; classic without its weighted-sum block), input scaling, the network, the loss weighting, the optimiser recipe and the split. Three seeds per arm; every run is scored with its last-epoch weights, as the pipeline reports, and again with the weights of the lowest validation loss. Two arms were then re-run inside the real pipeline on RIKYU, five seeds each, through a new per-task switch <code>class_weights = &quot;none&quot;</code>.",
f"一个本地训练框架把每个臂都放在 pipeline 用过的同一批 23,487 个训练行、同一 151 类标签、同一验证集和测试集上训练。pipeline 臂直接用项目自己的编码器和分类 head 类,按采用的形状、损失和优化器搭建,复现 RIKYU 的运行到小数点后第三位(top-1 {last('fm_replica'):.3f} ± {lsd('fm_replica'):.3f},对比 {CN['acc']:.3f} ± {CN['acc_sd']:.3f},且在同一 epoch 因同样原因停止)。论文臂是 2026 年重训分类器时保存下来的网络和优化器,通过同一框架运行。两者之间,十六个臂每次只切换一个因素:描述符(KMD;XenonPy classic;去掉 weighted-sum 块的 classic)、输入缩放、网络、损失权重、优化器配方和划分。每臂 3 个 seed;每次运行都用最后一个 epoch 的权重评分(与 pipeline 报告方式一致),再用验证损失最低那个 epoch 的权重评分一次。随后两个臂通过新增的任务级开关 <code>class_weights = &quot;none&quot;</code> 在 RIKYU 的真实 pipeline 里重跑,各 5 个 seed。",
f"ローカルのハーネスは、パイプラインが使ったのと同じ 23,487 訓練行、同じ 151 クラスのラベル、同じ検証・テスト分割で全アームを学習する。パイプラインのアームはプロジェクト自身のエンコーダと分類ヘッドのクラスから、採用した形・損失・最適化器で組み立て、RIKYU の実行を小数第3位まで再現する(top-1 {last('fm_replica'):.3f} ± {lsd('fm_replica'):.3f} 対 {CN['acc']:.3f} ± {CN['acc_sd']:.3f}、同じエポックで同じ理由で停止)。論文のアームは分類器の 2026 年再学習で保存されたネットワークと最適化器を、同じハーネスで動かしたもの。その間で 16 のアームが一度に一要因ずつ切り替える:記述子(KMD、XenonPy classic、weighted-sum ブロックを除いた classic)、入力スケーリング、ネットワーク、損失の重み付け、最適化器のレシピ、分割。各アーム 3 シード。全実行を最終エポックの重みで評価し(パイプラインの報告方法)、さらに検証損失最小のエポックの重みでも評価する。その後、2つのアームを新しいタスク単位のスイッチ <code>class_weights = &quot;none&quot;</code> で RIKYU の実際のパイプライン内で各 5 シード再実行した。")}</div>
</section>''')

# ---------------- 3 result ----------------
out.append(f'''<section class="section">{H2("3 · Result", "Three switches take the replica from 0.25 to the paper's 0.61", "三个开关把复现版从 0.25 带到论文的 0.61", "3つのスイッチで再現版は 0.25 から論文の 0.61 へ")}
{FH("From the pipeline replica to the paper's number, one switch at a time — and the same two switches inside the pipeline on RIKYU", "从 pipeline 复现版到论文的数字,每次一个开关——以及同样两个开关在 RIKYU pipeline 内部的结果", "パイプライン再現版から論文の数値へ、一度に1スイッチ — そして同じ2スイッチを RIKYU のパイプライン内部で")}
<figure><div class="figbox"><svg id="fig-ladder" width="960" height="330" role="img" aria-label="Top-1 accuracy ladder from the pipeline replica to the paper setup"></svg></div>
{CAP(f"Top of the chart: the local harness, three seeds, last-epoch weights, the same rows and split throughout. Removing the class weights adds {(last('fm_plain')-last('fm_replica'))*100:.0f} points at the pipeline's shape (27 on average over the four matched pairs), the paper's network 4 more, the XenonPy classic descriptor another {eff['descriptor: XenonPy classic − KMD']*100:.0f}; the end point, {last('paper_full_classic'):.3f}, is the paper's 60.2% measured on 151 instead of 213 groups. Bottom: the real pipeline on RIKYU, five seeds each — as run, with the weights switched off, and with the XenonPy descriptor as well.",
f"上半部分:本地框架,3 个 seed,最后 epoch 的权重,全程同一批行和划分。在 pipeline 的形状下去掉类权重加 {(last('fm_plain')-last('fm_replica'))*100:.0f} 个百分点(四对配对臂平均 27 个),换成论文的网络再加 4 个,换成 XenonPy classic 描述符再加 {eff['descriptor: XenonPy classic − KMD']*100:.0f} 个;终点 {last('paper_full_classic'):.3f} 就是论文的 60.2%,只是在 151 个而非 213 个空间群上测得。下半部分:RIKYU 上的真实 pipeline,各 5 个 seed——原样、关掉权重、再换 XenonPy 描述符。",
f"上段:ローカルハーネス、3 シード、最終エポックの重み、終始同じ行と分割。パイプラインの形でクラス重みを外すと {(last('fm_plain')-last('fm_replica'))*100:.0f} ポイント(4つの対の平均では 27)、論文のネットワークでさらに 4、XenonPy classic 記述子でさらに {eff['descriptor: XenonPy classic − KMD']*100:.0f} ポイント。終点 {last('paper_full_classic'):.3f} は、213 群ではなく 151 群で測った論文の 60.2% である。下段:RIKYU 上の実際のパイプライン、各 5 シード — 実行時のまま、重みを切ったもの、さらに XenonPy 記述子にしたもの。")}</figure>
{FH("Top-k recall of four arms, with the paper's Figure 3 points", "四个臂的 top-k recall,附论文图 3 的数据点", "4アームの top-k recall と、論文の図 3 の点")}
<figure><div class="figbox"><svg id="fig-topk" width="960" height="330" role="img" aria-label="Top-k recall curves"></svg></div>
{CAP(f"Recall at k = 1, 5, 10, 30 (mean of three seeds, last-epoch weights); the paper reports k = 1, 10, 30, 40. The replica trails at every k, not only at the top: the balanced weights spread probability over rare groups instead of concentrating it. The paper setup on our data sits on the paper's curve within a point (top-10 {topk_mean('paper_full_classic')[2]*100:.1f}% vs 85.4%, top-30 {topk_mean('paper_full_classic')[3]*100:.1f}% vs 92.6%).",
f"k = 1、5、10、30 处的 recall(3 个 seed 均值,最后 epoch 的权重);论文报告的是 k = 1、10、30、40。复现版在每个 k 上都落后,不只是 top-1:balanced 权重把概率摊到稀有群上,而不是集中起来。论文设置在我们数据上的曲线与论文曲线相差不到一个百分点(top-10 {topk_mean('paper_full_classic')[2]*100:.1f}% vs 85.4%,top-30 {topk_mean('paper_full_classic')[3]*100:.1f}% vs 92.6%)。",
f"k = 1、5、10、30 での recall(3 シード平均、最終エポックの重み)。論文は k = 1、10、30、40 を報告。再現版はどの k でも遅れており、top-1 だけではない:balanced 重みは確率を集中させず稀な群に散らす。我々のデータでの論文設定は論文の曲線に1ポイント以内で乗る(top-10 {topk_mean('paper_full_classic')[2]*100:.1f}% 対 85.4%、top-30 {topk_mean('paper_full_classic')[3]*100:.1f}% 対 92.6%)。")}</figure>
{FH("The effect of each factor, as matched pairs of arms", "每个因素的效应,按配对的臂计算", "各要因の効果、アームの対で")}
<figure><div class="figbox"><svg id="fig-effects" width="960" height="360" role="img" aria-label="Paired effects of each factor on top-1 accuracy"></svg></div>
{CAP("Each dot is one pair of arms that differ in that factor only; the bar is the mean. The loss weighting and the descriptor are the two large effects; the network is small; the optimiser recipe, input scaling and split are within a point or two; and taking the lowest-validation-loss epoch instead of the last one <em>lowers</em> accuracy in every arm — under plain cross-entropy the validation loss rises with over-confidence while accuracy is still improving, and under balanced weights the loss minimum comes at epoch 6, when the large groups are not learned yet.",
"每个点是只在该因素上不同的一对臂;粗线为均值。损失权重和描述符是两个大效应;网络效应小;优化器配方、输入缩放和划分都在一两个百分点之内;而用验证损失最低的 epoch 代替最后一个 epoch 在每个臂上都<em>降低</em>准确率——普通交叉熵下验证损失因过度自信而上升、准确率却仍在提高;balanced 权重下损失最低点在第 6 个 epoch,那时大群还没学会。",
"各点はその要因だけが異なるアームの対、太線は平均。損失の重み付けと記述子が2つの大きな効果。ネットワークは小さい。最適化器のレシピ、入力スケーリング、分割は1〜2ポイント以内。検証損失最小のエポックを最終エポックの代わりに使うとどのアームでも正解率が<em>下がる</em> — 通常の交差エントロピーでは過信により検証損失が上がる一方で正解率はまだ改善しており、balanced 重みでは損失最小が第 6 エポックで、大きな群がまだ学習されていない。")}</figure>
{TH("All sixteen arms: last-epoch and best-validation-loss weights, mean of three seeds", "全部十六个臂:最后 epoch 与验证损失最低 epoch 的权重,3 个 seed 均值", "全 16 アーム:最終エポックと検証損失最小エポックの重み、3 シード平均")}
{table([T("arm","臂","アーム"), T("descriptor","描述符","記述子"), T("scaling","缩放","スケーリング"), T("model","模型","モデル"), T("loss","损失","損失"), T("recipe","配方","レシピ"), T("split","划分","分割"), T("epochs / best","epoch 数 / 最优","エポック / 最良"), T("LAST top-1","LAST top-1","LAST top-1"), "macro-F1", "top-5", T("BEST top-1","BEST top-1","BEST top-1"), "top-10", "top-30"], arm_rows, cls="tablebox")}
<div class="col"><p class="tabnote">{T("LAST = the weights after the last epoch (early stopping on the validation loss, patience 24 or 30, cap 150) — what the pipeline reports. BEST = the weights of the epoch with the lowest validation loss. Model “fm” = the pipeline’s encoder + head at the adopted shape; “paper” = the 2026 refit’s 4-layer GELU network. Recipe “fm” = AdamW + ReduceLROnPlateau, batch 256; “paper” = Adam, gradient clipping, batch 1,024. Scaling “shotgun” = min-max → Yeo-Johnson → standard, fitted on the train split.",
"LAST = 最后一个 epoch 之后的权重(按验证损失早停,patience 24 或 30,上限 150)——即 pipeline 报告的数字。BEST = 验证损失最低那个 epoch 的权重。模型 “fm” = pipeline 的编码器 + head,采用形状;“paper” = 2026 重训的 4 层 GELU 网络。配方 “fm” = AdamW + ReduceLROnPlateau,batch 256;“paper” = Adam、梯度裁剪、batch 1,024。缩放 “shotgun” = min-max → Yeo-Johnson → standard,在训练集上拟合。",
"LAST = 最終エポック後の重み(検証損失で早期終了、patience 24 または 30、上限 150)— パイプラインが報告する値。BEST = 検証損失最小エポックの重み。モデル “fm” = 採用形状のパイプラインのエンコーダ + ヘッド、“paper” = 2026 再学習の 4 層 GELU ネットワーク。レシピ “fm” = AdamW + ReduceLROnPlateau、バッチ 256、“paper” = Adam、勾配クリップ、バッチ 1,024。スケーリング “shotgun” = min-max → Yeo-Johnson → standard、訓練分割で当てはめ。")}</p></div>
</section>''')

# ---------------- 4 why ----------------
out.append(f'''<section class="section">{H2("4 · Why", "What each of the two causes does to the model", "两个原因各自对模型做了什么", "2つの原因がそれぞれモデルに何をするか")}
<div class="col"><h3>{T("The balanced loss on 151 classes", "151 类上的类平衡损失", "151 クラスでのクラス平衡損失")}</h3>
{P(f"The catalog gives every classification head inverse-frequency class weights — the formula scikit-learn calls “balanced”, here computed in the catalog and handed to PyTorch's cross-entropy: a row of class c weighs N / (151 · N<sub>c</sub>). With classes from 3,311 rows down to 10, that is ×{w_min:.3f} for Fm-3m and ×{w_max:.1f} for a ten-row group — a {w_max/w_min:.0f}-fold range. The three largest groups hold {big3*100:.0f}% of the rows but {big3_mass*100:.0f}% of the loss; the {len(small)} groups with fewer than 30 rows hold {small_rows*100:.0f}% of the rows and {small_mass*100:.0f}% of the loss. The head is asked to get 39 groups of a dozen examples right at the price of the ones that make up most of the test set.",
f"catalog 给每个分类 head 都配上按类频率倒数的权重——即 scikit-learn 称为 “balanced” 的那个公式,这里由 catalog 算出后交给 PyTorch 的交叉熵:类 c 的一行权重为 N / (151 · N<sub>c</sub>)。类的大小从 3,311 行到 10 行,于是 Fm-3m 是 ×{w_min:.3f},十行的群是 ×{w_max:.1f}——相差 {w_max/w_min:.0f} 倍。最大的三个群占 {big3*100:.0f}% 的行,却只占 {big3_mass*100:.0f}% 的损失;{len(small)} 个不到 30 行的群占 {small_rows*100:.0f}% 的行,却占 {small_mass*100:.0f}% 的损失。head 被要求把 39 个只有十来个样例的群学对,代价是测试集里占大头的那些群。",
f"カタログは全分類ヘッドにクラス頻度の逆数の重みを与える — scikit-learn が “balanced” と呼ぶ式で、ここではカタログが計算して PyTorch の交差エントロピーに渡す:クラス c の1行の重みは N / (151 · N<sub>c</sub>)。クラスサイズは 3,311 行から 10 行まであるので、Fm-3m は ×{w_min:.3f}、10 行の群は ×{w_max:.1f} — {w_max/w_min:.0f} 倍の幅。最大3群は行の {big3*100:.0f}% を占めるが損失の {big3_mass*100:.0f}% しか占めず、30 行未満の {len(small)} 群は行の {small_rows*100:.0f}% で損失の {small_mass*100:.0f}% を占める。ヘッドは十数例しかない 39 群を当てるよう求められ、その代償はテスト集合の大半を占める群である。")}</div>
{FH("The balanced class weights against class size", "balanced 类权重与类大小的关系", "balanced クラス重みとクラスサイズ")}
<figure><div class="figbox"><svg id="fig-weights" width="960" height="300" role="img" aria-label="Class weight versus class size"></svg></div>
{CAP("One dot per space group. A row of the largest group counts for a fifteenth of an unweighted row; a row of a ten-row group for twenty-two unweighted rows.", "每点一个空间群。最大群的一行只抵普通一行的十五分之一;十行群的一行抵二十二行。", "各点が1空間群。最大群の1行は重みなしの1行の 15 分の1、10 行の群の1行は 22 行分に相当する。")}</figure>
{FH("Recall of the twelve largest groups: as run, weights off, weights off with the XenonPy descriptor", "最大的十二个群的 recall:原样、关掉权重、关掉权重并换 XenonPy 描述符", "最大 12 群の recall:実行時のまま、重みなし、重みなし + XenonPy 記述子")}
<figure><div class="figbox"><svg id="fig-recall" width="960" height="360" role="img" aria-label="Per-class recall of the largest groups under the three pipeline arms"></svg></div>
{CAP(f"Pipeline runs on RIKYU, seed 2025, test split. As run, the head recalls {r_big('balanced_kmd','Fm-3m')*100:.0f}% of Fm-3m, {r_big('balanced_kmd','Pnma')*100:.0f}% of Pnma and none of P2<sub>1</sub>/c or C2/c — the three largest low-symmetry groups are simply not predicted. With the weights off the same head recalls {r_big('plain_kmd','Fm-3m')*100:.0f}%, {r_big('plain_kmd','Pnma')*100:.0f}%, {r_big('plain_kmd','P2_1/c')*100:.0f}% and {r_big('plain_kmd','C2/c')*100:.0f}%. Macro-F1 — the metric the weights are meant to protect — does not improve either: {CN['f1']:.3f} with weights, {CW['f1']:.3f} without; on 151 classes the rare groups are not learnable from a dozen compositions, and the weights only sacrifice the common ones.",
f"RIKYU 上的 pipeline 运行,seed 2025,测试集。原样时 head 只召回 {r_big('balanced_kmd','Fm-3m')*100:.0f}% 的 Fm-3m、{r_big('balanced_kmd','Pnma')*100:.0f}% 的 Pnma,P2<sub>1</sub>/c 和 C2/c 一个都没有——三个最大的低对称群根本不被预测。关掉权重后同一个 head 召回 {r_big('plain_kmd','Fm-3m')*100:.0f}%、{r_big('plain_kmd','Pnma')*100:.0f}%、{r_big('plain_kmd','P2_1/c')*100:.0f}% 和 {r_big('plain_kmd','C2/c')*100:.0f}%。权重本想保护的 macro-F1 也没有变好:带权重 {CN['f1']:.3f},不带 {CW['f1']:.3f};在 151 类上,稀有群靠十几个组成学不会,权重只是牺牲了常见群。",
f"RIKYU 上のパイプライン実行、シード 2025、テスト分割。実行時のままではヘッドは Fm-3m の {r_big('balanced_kmd','Fm-3m')*100:.0f}%、Pnma の {r_big('balanced_kmd','Pnma')*100:.0f}% しか再現せず、P2<sub>1</sub>/c と C2/c はゼロ — 最大の低対称3群はそもそも予測されない。重みを切ると同じヘッドが {r_big('plain_kmd','Fm-3m')*100:.0f}%、{r_big('plain_kmd','Pnma')*100:.0f}%、{r_big('plain_kmd','P2_1/c')*100:.0f}%、{r_big('plain_kmd','C2/c')*100:.0f}% を再現する。重みが守るはずの macro-F1 も改善しない:重みあり {CN['f1']:.3f}、なし {CW['f1']:.3f}。151 クラスでは稀な群は十数の組成からは学べず、重みは頻出群を犠牲にするだけである。")}</figure>
<div class="col">{P(f"The weights also break the stopping rule. The validation loss the pipeline watches is the same weighted loss, and under it the minimum comes at epoch {CN['best']:.0f} — before the common groups are learned — after which it climbs from {CN['vl'][1]} to {CN['vl'][2]} while accuracy keeps rising; early stopping fires at epoch {CN['epochs']:.0f} with the training loss still at 1.6. Without weights the loss bottoms at epoch {CW['best']:.0f}, the run lasts {CW['epochs']:.0f} epochs, and the training loss reaches 1.2. This is why selecting the lowest-validation-loss epoch would make the weighted run worse still (0.14): the loss minimum is not the accuracy optimum.",
f"权重还破坏了停止规则。pipeline 监控的验证损失就是同一个加权损失,在它之下最低点出现在第 {CN['best']:.0f} 个 epoch——常见群还没学会——之后从 {CN['vl'][1]} 升到 {CN['vl'][2]},而准确率仍在上升;早停在第 {CN['epochs']:.0f} 个 epoch 触发,此时训练损失还停在 1.6。不带权重时损失在第 {CW['best']:.0f} 个 epoch 触底,运行持续 {CW['epochs']:.0f} 个 epoch,训练损失到 1.2。这也是为什么改取验证损失最低的 epoch 会让加权运行更差(0.14):损失最低点不是准确率最优点。",
f"重みは停止規則も壊す。パイプラインが監視する検証損失は同じ重み付き損失で、その最小は第 {CN['best']:.0f} エポック — 頻出群がまだ学習されていない時点 — に来て、その後 {CN['vl'][1]} から {CN['vl'][2]} へ上がる一方で正解率は上がり続ける。早期終了は第 {CN['epochs']:.0f} エポックで発火し、学習損失はまだ 1.6 のまま。重みなしでは損失は第 {CW['best']:.0f} エポックで底を打ち、実行は {CW['epochs']:.0f} エポック続き、学習損失は 1.2 に達する。検証損失最小のエポックを選ぶと重み付き実行がさらに悪くなる(0.14)のはこのためで、損失最小は正解率最適ではない。")}
<h3 style="margin-top:22px">{T("The descriptor cannot see the cell", "描述符看不见原胞", "記述子には単位胞が見えない")}</h3>
{P(f"The XenonPy classic table beats KMD by {eff['descriptor: XenonPy classic − KMD']*100:.0f} points in every setting, but the same table without its weighted-sum block — the only block that uses the raw atom counts — beats KMD by {eff['descriptor: classic w/o sum block − KMD']*100:.1f}. The gain is the sum block, i.e. the size of the cell. That is the same finding as for volume on the companion page, and it has a physical reason here: a space group fixes the Wyckoff multiplicities, so the number of atoms a cell can hold is strongly constrained by it. On its own, the atom count of the composition string predicts the space group with top-1 {atoms['atoms_only_accuracy']:.3f} — as much as the whole run as it stood, against a majority-class floor of {atoms['majority_accuracy']:.3f} — and shares {atoms['mutual_information_bits']:.1f} bits of information with the label. KMD reduces the string to atomic fractions before anything else happens, so Fe<sub>2</sub>O<sub>3</sub> and Fe<sub>4</sub>O<sub>6</sub> are the same input and the count never reaches the model.",
f"XenonPy classic 表在每种设置下都比 KMD 高 {eff['descriptor: XenonPy classic − KMD']*100:.0f} 个百分点,但去掉 weighted-sum 块——唯一使用原始原子数的块——之后只比 KMD 高 {eff['descriptor: classic w/o sum block − KMD']*100:.1f} 个。增益来自 sum 块,也就是原胞大小。这与配套页面里 volume 的发现相同,而且在这里有物理原因:空间群固定了 Wyckoff 位置的多重度,所以一个原胞能容纳多少原子受它强烈约束。单凭组成串的原子数就能以 top-1 {atoms['atoms_only_accuracy']:.3f} 预测空间群——和原样运行的整个模型一样多,而多数类的下限只有 {atoms['majority_accuracy']:.3f}——它与标签共享 {atoms['mutual_information_bits']:.1f} bit 的信息。KMD 在做任何事之前先把组成串化为原子分数,于是 Fe<sub>2</sub>O<sub>3</sub> 和 Fe<sub>4</sub>O<sub>6</sub> 是同一输入,原子数从未到达模型。",
f"XenonPy classic 表はどの設定でも KMD を {eff['descriptor: XenonPy classic − KMD']*100:.0f} ポイント上回るが、weighted-sum ブロック — 生の原子数を使う唯一のブロック — を除いた同じ表は KMD を {eff['descriptor: classic w/o sum block − KMD']*100:.1f} ポイントしか上回らない。利得は sum ブロック、すなわち単位胞の大きさである。これは併設ページでの volume と同じ知見で、ここには物理的理由がある:空間群は Wyckoff 位置の多重度を固定するので、単位胞が持てる原子数はそれに強く制約される。組成文字列の原子数だけで空間群は top-1 {atoms['atoms_only_accuracy']:.3f} で予測でき — 実行時のままの実行全体と同じで、多数クラスの下限 {atoms['majority_accuracy']:.3f} に対して — ラベルと {atoms['mutual_information_bits']:.1f} ビットの情報を共有する。KMD は何よりも先に文字列を原子分率に落とすので、Fe<sub>2</sub>O<sub>3</sub> と Fe<sub>4</sub>O<sub>6</sub> は同じ入力になり、原子数はモデルに届かない。")}</div>
{FH("Atoms per cell in the four largest groups", "最大的四个群的原胞原子数分布", "最大4群の単位胞原子数分布")}
<figure><div class="figbox"><svg id="fig-atoms" width="960" height="320" role="img" aria-label="Atoms-per-cell distributions of the four largest space groups"></svg></div>
{CAP("Training rows. Fm-3m compositions are mostly written with 1–8 atoms; Pnma and P2<sub>1</sub>/c with 16–63; C2/c with 32 and more. The count separates the cubic group from the two low-symmetry ones almost by itself — and it is exactly what the fraction-based descriptor discards.",
"训练行。Fm-3m 的组成大多写成 1–8 个原子;Pnma 和 P2<sub>1</sub>/c 是 16–63 个;C2/c 是 32 个以上。单凭原子数几乎就能把立方群和两个低对称群分开——而这恰恰是基于分数的描述符丢掉的信息。",
"訓練行。Fm-3m の組成は大半が 1–8 原子で書かれ、Pnma と P2<sub>1</sub>/c は 16–63、C2/c は 32 以上。原子数だけで立方晶群と低対称2群がほぼ分かれる — そしてそれこそ分率ベースの記述子が捨てる情報である。")}</figure>
<div class="col"><h3 style="margin-top:22px">{T("The network, and what did not matter", "网络,以及不起作用的因素", "ネットワーク、そして効かなかったもの")}</h3>
{P(f"The paper's 4-layer GELU network with dropout 0.23 adds about 4 points over the pipeline's encoder-plus-head at the adopted shape, under either optimiser — a real but small effect, and the only one where the two setups' hyper-parameter searches differ rather than their design. Under the balanced loss the wider network is <em>worse</em> ({last('paper_model_fm_recipe_bal'):.3f} vs {last('fm_replica_scaled'):.3f}): more capacity to chase the rare groups. The optimiser recipe (AdamW with plateau decay, batch 256, versus Adam with clipping, batch 1,024) moves accuracy by under a point; so does the paper's input-scaling chain; a random 80/10/10 split instead of the dataset's own adds about 2 points, so the fixed split is slightly harder but not the story.",
f"论文的 4 层 GELU 网络(dropout 0.23)在两种优化器下都比 pipeline 采用形状的编码器加 head 高约 4 个百分点——真实但很小,也是唯一一个源自两边超参数搜索不同、而非设计不同的效应。在 balanced 损失下,更宽的网络反而<em>更差</em>({last('paper_model_fm_recipe_bal'):.3f} vs {last('fm_replica_scaled'):.3f}):容量更大,追稀有群追得更狠。优化器配方(带 plateau 衰减的 AdamW、batch 256,对比带裁剪的 Adam、batch 1,024)对准确率的影响不到一个百分点;论文的输入缩放链也一样;用随机 80/10/10 划分代替数据集自带划分约加 2 个百分点,说明固定划分略难一点,但不是主线。",
f"論文の 4 層 GELU ネットワーク(dropout 0.23)は、どちらの最適化器でも採用形状のパイプラインのエンコーダ + ヘッドを約 4 ポイント上回る — 実在するが小さい効果で、両設定の設計ではなくハイパーパラメータ探索の違いに由来する唯一のもの。balanced 損失の下では広いネットワークは<em>悪化</em>する({last('paper_model_fm_recipe_bal'):.3f} 対 {last('fm_replica_scaled'):.3f}):稀な群を追う容量が増えるだけ。最適化器のレシピ(plateau 減衰付き AdamW・バッチ 256 対 クリップ付き Adam・バッチ 1,024)は正解率を 1 ポイント未満しか動かさず、論文の入力スケーリング連鎖も同様。データセット自身の分割の代わりにランダム 80/10/10 にすると約 2 ポイント上がるので、固定分割はやや難しいが本筋ではない。")}</div>
</section>''')

# ---------------- 5 confirmation ----------------
out.append(f'''<section class="section">{H2("5 · Inside the pipeline", "The same two switches on RIKYU, five seeds each", "同样两个开关在 RIKYU 上,各 5 个 seed", "同じ2スイッチを RIKYU 上で、各 5 シード")}
<div class="col">{P("The local harness settles attribution; the pipeline settles what the baseline is. A per-task switch <code>class_weights = &quot;balanced&quot; | &quot;none&quot;</code> was added to the task catalog (default unchanged, so no existing config moves), and the stage_single recipe was re-run on the space-group head with it off, once with KMD and once with the XenonPy classic table as a precomputed descriptor. The container's own 0.3.2 code was patched in place for the runs; the knob is in the branch.",
"本地框架解决的是归因,pipeline 解决的是基线到底是多少。任务 catalog 新增了任务级开关 <code>class_weights = &quot;balanced&quot; | &quot;none&quot;</code>(默认不变,已有配置不受影响),然后用 stage_single 配方在关掉权重的情况下重跑空间群 head,一次用 KMD,一次用作为预计算描述符的 XenonPy classic 表。这几次运行对容器自带的 0.3.2 代码就地打了补丁;开关已进入分支。",
"ローカルハーネスは帰属を決め、パイプラインはベースラインの値を決める。タスクカタログにタスク単位のスイッチ <code>class_weights = &quot;balanced&quot; | &quot;none&quot;</code> を追加し(既定値は不変で既存設定は動かない)、stage_single レシピで空間群ヘッドを重みなしで再実行した。一度は KMD、一度は事前計算記述子としての XenonPy classic 表で。実行ではコンテナ自身の 0.3.2 コードにその場でパッチを当てた。スイッチはブランチに入っている。")}</div>
{TH("Pipeline runs on RIKYU: as run, class weights off, class weights off with the XenonPy descriptor", "RIKYU 上的 pipeline 运行:原样、关掉类权重、关掉类权重并换 XenonPy 描述符", "RIKYU 上のパイプライン実行:実行時のまま、クラス重みなし、クラス重みなし + XenonPy 記述子")}
{table([T("arm","臂","アーム"), T("seeds","seed 数","シード"), T("accuracy","准确率","正解率"), "macro-F1", T("epochs / best","epoch 数 / 最优","エポック / 最良"), T("val loss: first → min → last","验证损失:首 → 最低 → 末","検証損失:初 → 最小 → 終")], confirm_rows, style="max-width:1000px")}
<div class="col" style="margin-top:16px">{P(f"<b>The pipeline agrees with the harness to within its seed spread</b>: {CW['acc']:.3f} against the replica's {last('fm_plain'):.3f} with the weights off, {CX['acc']:.3f} against {last('fm_plain_classic'):.3f} with the XenonPy descriptor. The space-group baseline to quote on the 2026-09-11 dataset is therefore {CW['acc']:.3f} / macro-F1 {CW['f1']:.3f} with KMD, or {CX['acc']:.3f} / {CX['f1']:.3f} with a scale-aware descriptor — not {CN['acc']:.3f}. The remaining {(last('paper_full_classic')-CX['acc'])*100:.0f} points to the paper's setup are the network, a tuning question rather than a defect.",
f"<b>pipeline 与本地框架的差距在 seed 离散之内</b>:关掉权重后 {CW['acc']:.3f} 对复现版的 {last('fm_plain'):.3f},换 XenonPy 描述符后 {CX['acc']:.3f} 对 {last('fm_plain_classic'):.3f}。因此在 2026-09-11 数据集上应引用的空间群基线是 KMD 下的 {CW['acc']:.3f} / macro-F1 {CW['f1']:.3f},或带尺度描述符下的 {CX['acc']:.3f} / {CX['f1']:.3f}——而不是 {CN['acc']:.3f}。到论文设置还差的 {(last('paper_full_classic')-CX['acc'])*100:.0f} 个百分点来自网络,是调参问题而非缺陷。",
f"<b>パイプラインはハーネスとシードのばらつき以内で一致する</b>:重みなしで {CW['acc']:.3f} 対 再現版の {last('fm_plain'):.3f}、XenonPy 記述子で {CX['acc']:.3f} 対 {last('fm_plain_classic'):.3f}。したがって 2026-09-11 データセットで引用すべき空間群ベースラインは KMD で {CW['acc']:.3f} / macro-F1 {CW['f1']:.3f}、スケールを持つ記述子で {CX['acc']:.3f} / {CX['f1']:.3f} — {CN['acc']:.3f} ではない。論文設定までの残り {(last('paper_full_classic')-CX['acc'])*100:.0f} ポイントはネットワークであり、欠陥ではなく調整の問題。", "callout")}</div>
</section>''')

# ---------------- 6 consequences ----------------
out.append(f'''<section class="section">{H2("6 · What follows", "Three decisions this settles, and one it reopens", "由此定下的三件事,和重新打开的一件", "これで決まる3つのことと、再び開かれる1つ")}
<div class="col plan"><ol class="plan">
<li>{T("<b>Space group trains on unweighted cross-entropy.</b> The balanced weights were designed for the five-class material_type head, where 99% of rows are one class and the rare classes are learnable; on 151 classes they cost 21 points of accuracy and gain nothing in macro-F1. The switch exists; the space-group config uses it. Whether magnetic_ordering (4 classes) and the two binary heads are better off with or without weights is a two-run check each, not yet done.",
"<b>空间群改用不加权的交叉熵。</b>balanced 权重是为五类的 material_type head 设计的——那里 99% 的行属于一类、稀有类是学得会的;在 151 类上它损失 21 个百分点的准确率,macro-F1 也毫无收益。开关已经有了,空间群的配置在用。magnetic_ordering(4 类)和两个二分类 head 带不带权重更好,各需要两次运行来检验,尚未做。",
"<b>空間群は重みなしの交差エントロピーで学習する。</b>balanced 重みは5クラスの material_type ヘッドのために設計された — 行の 99% が1クラスで、稀なクラスは学習可能。151 クラスでは正解率を 21 ポイント失い macro-F1 も得ない。スイッチはあり、空間群の設定で使っている。magnetic_ordering(4 クラス)と2つの二値ヘッドが重みあり・なしのどちらが良いかは各2実行の確認で、未実施。")}</li>
<li>{T(f"<b>Space group is the third task that needs the cell scale</b> — after volume and total magnetisation — and unlike those two it has no per-atom form to fall back on. The descriptor decision on the transferability page's list now carries three tasks: a scale-carrying descriptor gets {CX['acc']:.3f} here and R² 0.997 on volume; KMD alone leaves {CW['acc']:.3f}.",
f"<b>空间群是继 volume 和总磁矩之后第三个需要原胞尺度的任务</b>,而且与前两者不同,它没有每原子形式可以退而求其次。迁移性页面清单上的描述符决策现在牵涉三个任务:带尺度的描述符在这里拿到 {CX['acc']:.3f}、在 volume 上拿到 R² 0.997;只用 KMD 则停在 {CW['acc']:.3f}。",
f"<b>空間群は volume と全磁化に続いて単位胞スケールを必要とする3つ目のタスク</b>で、前2つと違い原子あたりの形に逃げられない。転移性ページのリストにある記述子の決定は今や3タスクに関わる:スケールを持つ記述子はここで {CX['acc']:.3f}、volume で R² 0.997 を得る。KMD だけでは {CW['acc']:.3f} に留まる。")}</li>
<li>{T(f"<b>The baseline table on the companion page is superseded for this task</b>: {CW['acc']:.3f} / {CW['f1']:.3f} (KMD) and {CX['acc']:.3f} / {CX['f1']:.3f} (XenonPy classic) replace {CN['acc']:.3f} / {CN['f1']:.3f}, and the sentence “only high-symmetry families are recognisable from composition” is withdrawn — with the weights off, Pnma and P2<sub>1</sub>/c are recalled at {r_big('plain_kmd','Pnma')*100:.0f}% and {r_big('plain_kmd','P2_1/c')*100:.0f}%.",
f"<b>配套页面上这个任务的基线表作废</b>:{CW['acc']:.3f} / {CW['f1']:.3f}(KMD)和 {CX['acc']:.3f} / {CX['f1']:.3f}(XenonPy classic)取代 {CN['acc']:.3f} / {CN['f1']:.3f},“只凭组成只能认出高对称家族”这句话收回——关掉权重后,Pnma 和 P2<sub>1</sub>/c 的召回率分别是 {r_big('plain_kmd','Pnma')*100:.0f}% 和 {r_big('plain_kmd','P2_1/c')*100:.0f}%。",
f"<b>併設ページのこのタスクのベースライン表は置き換えられる</b>:{CW['acc']:.3f} / {CW['f1']:.3f}(KMD)と {CX['acc']:.3f} / {CX['f1']:.3f}(XenonPy classic)が {CN['acc']:.3f} / {CN['f1']:.3f} に代わり、「組成から分かるのは高対称族だけ」という文は撤回する — 重みなしでは Pnma と P2<sub>1</sub>/c の recall は {r_big('plain_kmd','Pnma')*100:.0f}% と {r_big('plain_kmd','P2_1/c')*100:.0f}%。")}</li>
<li>{T("<b>Reopened: the head shape for many-class tasks.</b> The paper's network adds 4 points with the same descriptor and loss. The adopted encoder and head were tuned on regression tasks and a five-class head; a 151-way output through a 64-unit hidden layer was never tuned. This is the one factor left between the pipeline and the paper, and it is a search, not a fix.",
"<b>重新打开:多类任务的 head 形状。</b>同样的描述符和损失下,论文的网络多出 4 个百分点。采用的编码器和 head 是在回归任务和一个五类 head 上调出来的;经过 64 单元隐藏层的 151 路输出从未调过。这是 pipeline 与论文之间剩下的唯一因素,它是一次搜索,不是一个修复。",
"<b>再び開かれる:多クラスタスクのヘッド形状。</b>同じ記述子と損失で論文のネットワークは 4 ポイント上乗せする。採用したエンコーダとヘッドは回帰タスクと5クラスのヘッドで調整されたもので、64 ユニットの隠れ層を通る 151 出力は一度も調整されていない。これがパイプラインと論文の間に残る唯一の要因で、修正ではなく探索である。")}</li>
</ol></div>
<div class="col caveats" style="margin-top:28px"><h3>{T("What to hold loosely", "需要保留态度的几点", "留保すべき点")}</h3><ul>
<li>{T("<b>151 versus 213 classes.</b> The paper's test set includes every group; ours excludes groups under ten rows (0.8% of rows). Those rows are the hardest, so our 0.61 for the paper setup is, if anything, a touch generous; the paper's random split adds about 2 points over the dataset's split, in the other direction.", "<b>151 类 vs 213 类。</b>论文的测试集包含所有群;我们排除了不足十行的群(0.8% 的行)。这些行最难,所以论文设置在我们这里的 0.61 若有偏差也是略偏高;论文的随机划分则比数据集自带划分高约 2 个百分点,方向相反。", "<b>151 クラス対 213 クラス。</b>論文のテスト集合は全群を含み、我々は 10 行未満の群(行の 0.8%)を除く。それらは最も難しい行なので、論文設定での我々の 0.61 はあるとすればやや甘い。論文のランダム分割はデータセットの分割より約 2 ポイント高く、逆方向。")}</li>
<li>{T("<b>The paper arm uses the 2026 refit's hyper-parameters</b> (4 layers, GELU, dropout 0.23, lr 3e-3, batch 1,024), which were tuned for the 232-column feature set on a 180-class label, not re-tuned here for KMD or for 151 classes. A search on our data could move its numbers by a point or two; it cannot move the ordering of the factors.", "<b>论文臂用的是 2026 年重训的超参数</b>(4 层、GELU、dropout 0.23、lr 3e-3、batch 1,024),那是在 232 列特征和 180 类标签上调的,这里没有为 KMD 或 151 类重新调。在我们的数据上再搜索一次可能让它的数字动一两个百分点,但动不了因素的排序。", "<b>論文アームは 2026 年再学習のハイパーパラメータ</b>(4 層、GELU、dropout 0.23、lr 3e-3、バッチ 1,024)を使い、それは 232 列特徴と 180 クラスのラベルで調整されたもので、ここで KMD や 151 クラス向けに再調整していない。我々のデータでの探索は数値を1〜2ポイント動かしうるが、要因の順序は動かない。")}</li>
<li>{T("<b>Every pipeline number is last-epoch weights</b> under early stopping on the validation loss; this page shows that is the right choice for this task, but it is a choice.", "<b>pipeline 的每个数字都是验证损失早停下最后一个 epoch 的权重</b>;本页表明对这个任务这是正确的选择,但它仍是一个选择。", "<b>パイプラインの全数値は検証損失による早期終了下の最終エポックの重み</b>である。本ページはそれがこのタスクでは正しい選択だと示すが、選択であることに変わりはない。")}</li>
</ul></div>
</section>
<footer><p>{T("Local harness: analysis/space_group_study.py (uses the project's FoundationEncoder and ClassificationHead; the paper network and recipe from notebooks/prediction_models/space_group in shotgun_csp_next); results in summary/space_group_study.json and _table.json. Pipeline runs: stage_single_mp2026/stN_*, stW_*, stX_* on RIKYU with SRC_OVERRIDE on the 0.3.2 container; summary/space_group_confirm.json, _perclass.json, _atoms.json. Paper: Liu et al., npj Comput. Mater. 10, 298 (2024), Fig. 3 and Methods. Figure labels stay in English in every language.",
"本地框架:analysis/space_group_study.py(使用项目自身的 FoundationEncoder 和 ClassificationHead;论文网络与配方来自 shotgun_csp_next 的 notebooks/prediction_models/space_group);结果在 summary/space_group_study.json 与 _table.json。pipeline 运行:RIKYU 上的 stage_single_mp2026/stN_*、stW_*、stX_*,通过 SRC_OVERRIDE 作用于 0.3.2 容器;summary/space_group_confirm.json、_perclass.json、_atoms.json。论文:Liu et al., npj Comput. Mater. 10, 298 (2024),图 3 与 Methods。图内标签在各语言下均保留英文。",
"ローカルハーネス:analysis/space_group_study.py(プロジェクト自身の FoundationEncoder と ClassificationHead を使用。論文のネットワークとレシピは shotgun_csp_next の notebooks/prediction_models/space_group から)。結果は summary/space_group_study.json と _table.json。パイプライン実行:RIKYU 上の stage_single_mp2026/stN_*、stW_*、stX_*、0.3.2 コンテナに SRC_OVERRIDE。summary/space_group_confirm.json、_perclass.json、_atoms.json。論文:Liu et al., npj Comput. Mater. 10, 298 (2024)、図 3 と Methods。図中のラベルはどの言語でも英語のまま。")}</p></footer>
</div>
<script>
{LANGJS}
</script>
<script>
const SG = {json.dumps(SG, separators=(",", ":"))};
{JS}
</script>
''')

OUT.write_text("".join(out), encoding="utf-8")
print(f"  wrote {OUT} ({len(''.join(out)) // 1024} KB)")
