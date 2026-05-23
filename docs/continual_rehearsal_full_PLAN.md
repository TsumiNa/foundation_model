# Continual Rehearsal — 正式训练 Plan Memo

> 状态：**评估方法已敲定（§5），实现停在 CPU smoke**（GPU 被占用 + 等 PR #18 合并）
> 路线决策：先走**缩小版**（`sample_per_dataset` 上限 + 减少 epoch 上限），全量留到论文最终复现阶段
> 日期：2026-05-23 · 分支：`refine-demo-plots`（与 PR #18 同分支）
> 流程蓝本：`run_continual_rehearsal_demo.sh` / `scripts/continual_rehearsal_demo.py`

---

## Handoff — 给接手 PR #18 的 agent

PR #18 合并时本 workstream（`continual_rehearsal_full`）将随 #18 一并打磨入库。本文档是单一信息源；所有决策与背景都在下方各节。

### 当前工作树（未 commit）
本 workstream 的全部产物，都在 working tree 里、未 commit：

| 路径 | 状态 | 备注 |
|---|---|---|
| `docs/continual_rehearsal_full_PLAN.md` | untracked | **本文件** |
| `src/foundation_model/scripts/continual_rehearsal_full.py` | untracked | 主脚本（24-task 目录、分级 rehearsal、不冻结、EarlyStopping、逐步 pred dump、checkpoint、3 剧本 inverse、pptx+md+html） |
| `src/foundation_model/scripts/continual_rehearsal_full_test.py` | untracked | 16 tests 通过（catalogue / config / parser） |
| `samples/continual_rehearsal_full_config.toml` | untracked | 默认配置，已含**新版 task_sequence**（12 reg → 7 kr 升序 → 5 tail） |
| `run_continual_rehearsal_full.sh` | untracked | 仿 demo wrapper，日期戳输出 |
| `pyproject.toml` / `uv.lock` | modified | `uv add python-pptx`（runtime dep） |

`artifacts/continual_rehearsal_full_smoke/` 是 CPU smoke 产物（gitignored，可丢弃）。

### 用户已确认的决策（来自本次会话）
1. **任务顺序**（§2）：12 regression（自由序）→ 7 kernel regression **按非空行升序** → 5 固定 tail（`formation_energy → magnetic_moment → tc → klat → material_type`）。已同步到 config。
2. **分级 rehearsal**（§3）：固定 tail 作为旧 task 被回放时 `replay_ratio_high=0.10`，其余旧 task `replay_ratio=0.05`。
3. **不冻结任何层**：每步 encoder + 所有已激活 head 联合训练，仅靠 rehearsal mask 实现增量。
4. **训练规模**：全量数据 + `max_epochs_per_step=100` + EarlyStopping(`val_final_loss`, patience=8)，MPS。
5. **Inverse design — 强制使用 PR #18 的两条新路径**（§5）：旧无约束 `optimize_latent` (`α=0`) **弃用**；每个剧本必须用 (a) `optimize_latent(ae_align_scale=0.5)`（[0,1] 范围，默认 0.5）和 (b) `optimize_composition(...)` differentiable KMD 跑 3 个 composition 配置对照。三个剧本目标不变（§5 表）。
6. **评估细节已敲定**（§5）：基于 PR #18 的 `paper_inverse_comparison.py` 实测结论 —— `ae_align_scale=0.5`（empirical sweet spot）；composition 路径跑 strict-seed / alloy-palette+blended / random-init 三个配置；种子改为 **17 top-QC 去重 + 3 个固定 Au–Ga–Ln**；alloy palette 见 §5 元素清单。两个用户旋钮 `ae_align_scale` / `diversity_scale` 都在 `[0, 1]` 上，符合直觉（见 §5）。
7. **PPT 规范**（§6）：16:9 · 白底 · 主色 `#2563EB` · 至多两辅助色（`#55A868` / `#C44E52`）· 11 页结构（含 catagoly 短分析与"即插即用 downstream"收尾页）。
8. **PR #18 依赖与 rebase 步骤**：§11。当前 runner 里 `_inverse_design` 是**占位实现**（沿用 demo 老式 latent-only + `KMD.inverse` 解码），**rebase 时必须替换**。

### 已完成 / 验证过
- `ruff format` / `ruff check` / `mypy src/.../continual_rehearsal_full.py` 全绿。
- `pytest` 新增 16 tests 通过。
- **CPU smoke**（`--sample-per-dataset 800 --max-epochs-per-step 2 --accelerator cpu`）端到端 OK，全部交付物产出。注意 smoke 跑的是**旧顺序**（已被本次更新后的新序覆盖），rebase 后建议再跑一次 smoke。

### Rebase 后要做的事（详见 §11）
1. 让 `_inverse_design` 改用双路径（latent+λ 与 `optimize_composition`）。
2. 删除旧无约束 latent 调用与对应配置默认。
3. 按 §6 重写 `_write_pptx`（11 页 + 主色 + ≤2 辅助色 + `_pptx_palette` 常量）。
4. 验证从 demo 模块 import 的 helper 名仍可用（`_apply_plot_style` / `_PALETTE` / `_SCATTER_COLOR` / `_REPORT_TEMPLATE` / `_as_float_array` / `_composition_key` / `_init_kernels`）。
5. smoke 重跑 → GPU 空闲后启动全量 MPS 正式 run（命令在 §10）。

---

## 0a. Narrative arc — 论文 / 项目对外叙事链

**写作时按这条链组织正文与 slides**：

1. **问题提出 — 多属性联合优化是材料开发刚需**
   材料设计 = 在 *很多个* 属性约束下找配方（QC 类别 + 形成能 + 热导率 + 磁矩 + …），传统正向 DFT/实验循环成本极高。

2. **方案 — 持续学习构建一个 downstream 友好的 foundation model**
   共享 encoder + 多任务头 + rehearsal 增量训练；外部数据形态只要是
   "composition + property（或 category）" 即可一行 task config 接入；
   即插即用 downstream。

3. **示例 — Quasicrystal discovery**
   以 QC 形成 + 低形成能 + 高热导率/高磁矩等剧本作 case study；
   展示 model 在三个目标上的反向设计能力。

4. **面向实际可用性 — 不只展示 best number，更展示约束的必要**
   - latent 路径有 AE-roundtrip 失败模式，靠 `ae_align_scale` 修复；
   - composition 路径有 seed-init 锁支撑集问题，靠 `seed_blend` 修复；
   - 无约束 composition（含 random init）虽能找到全局最优 QC 点，但落在 Pu/F/Mn 等 **不可合成元素**上 → **架构的搜索能力强，但反过来证明 alloy palette / 领域知识约束的不可或缺**；
   - 旋钮命名都按用户直觉（`[0, 1]` + 名字朝向 = 大小语义），文档说明背后算法。

5. **系统性分析潜在问题**
   除头条结果外，专门给出**失败模式 + 偏置 + ablation** 三类分析，让读者理解何时该用哪个旋钮、何时该退回 strict-seed、何时该信任 model 的元素发现。

6. **下一步 — agent 化的 inverse design 工作台**
   计划围绕本 foundation model 搭建轻量 agent：用户描述目标（自然语言）→
   AI 分解 + 结合领域知识 → 自动设定 `ae_align_scale` / `diversity_scale` /
   palette 等优化超参 → 自动跑 `optimize_*` → 给出可视化 result + 报告 PDF。

7. **更远期 — AI4S agent 群的一部分**
   把数值模拟 (DFT/MD) + 自动实验 / 表征装置作为额外 agent 接入；
   foundation model 在这个 stack 里扮演**快速预测 + 候选生成**的中枢。

这条链同时是 §6 的 PPT 大纲，也是论文 Introduction / Discussion / Future
Work 的骨架。**slides 与 ANALYSIS.md 最终输出全部用英文撰写**。

---

## 0. 目标

在 **一个共享 encoder** 上做 continual（增量）多任务学习 + rehearsal 回放，覆盖 4 个无机数据集、全部 task 类型；训练完成后用同一个最终模型跑 **3 个独立的 inverse-design 剧本**。每个阶段的**原始数据 + plot** 全部落盘，最后产出 **PPT（.pptx）+ 文字 summary doc（Markdown）+ HTML deck**。

按上一轮确认的 4 个决策执行：
1. 最后固定顺序为 **5 个 task**（重复的 Magnetic moment 是笔误）。
2. **全量数据 + 早停**（`sample_per_dataset=null`，`max_epochs_per_step=100` 作上限，`EarlyStopping` 监控 `val_final_loss`）。
3. **新建专用脚本 + 配置**，复用 demo 的 helper，不改动 demo。
4. **加 `python-pptx`** 生成真正的 .pptx；同时产出 Markdown summary + 现有 HTML deck。

---

## 1. Task 目录（共 24 个监督 task + 常驻 autoencoder）

非空行数为各任务在对应数据集中可用样本（已核实）。kernel 列均为 `(值序列, T/K 序列)` 且长度一致。

### 1a. Regression — 16 个

| task 名 | 数据集 | 列 | 非空行数 |
|---|---|---|---|
| density | qc | `Density (normalized)` | ~49034 |
| efermi | qc | `Efermi (normalized)` | ~49034 |
| final_energy | qc | `Final energy per atom (normalized)` | ~49034 |
| **formation_energy** | qc | `Formation energy per atom (normalized)` | ~49034 |
| total_magnetization | qc | `Total magnetization (normalized)` | ~49034 |
| volume | qc | `Volume (normalized)` | ~49034 |
| dielectric_total | qc | `Dielectric total (normalized)` | （子集，介电仅部分材料有） |
| dielectric_ionic | qc | `Dielectric ionic (normalized)` | （子集） |
| dielectric_electronic | qc | `Dielectric electronic (normalized)` | （子集） |
| kp | phonix | `kp[W/mK]` | 6714 |
| **klat** | phonix | `klat[W/mK]` | 6714 |
| **tc** | superconductor | `Transition temperature[K]` | 10465 |
| **magnetic_moment** | magnetic | `Magnetic moment[μB/f.u.]` | 1222 |
| magnetization | magnetic | `Magnetization[A·m²/mol]` | （子集） |
| curie | magnetic | `Curie temperature[K]` | （子集） |
| neel | magnetic | `Neel temperature[K]` | （子集） |

非-qc 的 raw 回归列（tc/klat/magnetic_moment/magnetization/curie/neel）沿用 demo 处理：`log1p → 用 train 行统计做 z-score → clip 到 ±5`，避免长尾。qc 列已是 normalized，直接用。

### 1b. Kernel Regression — 7 个

| task 名 | 值列 | t 列 | 非空行数 |
|---|---|---|---|
| dos_density | `DOS density (normalized)` | `DOS energy` | 10321 |
| electrical_resistivity | `Electrical resistivity (normalized)` | `Electrical resistivity (T/K)` | 7334 |
| power_factor | `Power factor (normalized)` | `Power factor (T/K)` | 5223 |
| seebeck | `Seebeck coefficient (normalized)` | `Seebeck coefficient (T/K)` | 11722 |
| thermal_conductivity | `Thermal conductivity (normalized)` | `Thermal conductivity (T/K)` | 6158 |
| zt | `ZT (normalized)` | `ZT (T/K)` | 4971 |
| magnetic_susceptibility | `Magnetic susceptibility (normalized)` | `Magnetic susceptibility (T/K)` | **98 ⚠️** |

⚠️ `magnetic_susceptibility` 只有 98 行，test/val 后可用样本极少，R² 可能不稳定 —— 仍按要求纳入，但报告里会标注「low-data」。

### 1c. Classification — 1 个

| task 名 | 列 | 类别 |
|---|---|---|
| **material_type** | `Material type (label)` | 5 类合并为 3 类：AC=DAC+IAC, QC=DQC+IQC, others |

QC 极不平衡（IQC 213 / IAC 126 / DQC 15 / DAC 13 / others 48667），沿用 demo 的 **inverse-frequency class weights**。

### 1d. 数据集来源与规模

| 数据集 | 文件 | 行数 | 提供的 task |
|---|---|---|---|
| qc_ac_te_mp (DOS/material) | `qc_ac_te_mp_dos_reformat_20260515.pd.parquet` | 49034 | 9 reg + 7 kr + 1 clf = 17 |
| phonix-db | `phonix-db-filtered_20260425.parquet` | 6714 | 2 reg (kp, klat) |
| NEMAD superconductor | `NEMAD_superconductor_20260425.parquet` | 10465 | 1 reg (tc) |
| NEMAD magnetic | `NEMAD_magnetic_20260419.parquet` | 20271 | 4 reg (magnetic_moment, magnetization, curie, neel) |

> 注：新 qc 文件**没有**配套 preprocessing pkl（仅有 20250615 版），故 `qc_preprocessing_path=null`，跳过 `dropped_idx` 过滤。各数据集按 composition formula join；qc 用自带 `split` 列（train 34322 / val 7355 / test 7357），其余数据集随机 70/15/15 split。

---

## 2. 训练顺序（continual 增量）

分三段以最小化重复训练开销：

1. **12 个 regression（非 tail）**：顺序自由，按数据集分组一种确定排列保可复现。
2. **7 个 kernel regression**：**按非空样本数升序**。理由：每个新 task 在 intro 时按 100% mask 跑，之后按 5% mask 回放。把**小**数据集摆前面 → intro 时全量也很便宜；后续每步只回放小数据的 5%。把**大**数据集摆后面 → intro 时一次全量，之后剩余步数少（回放次数也少）。kernel regression 训练单步耗时显著，按这个序能把"100% 全量 + 5%·(N−k) 回放"这项总成本压到最小。
3. **5 个固定 tail**：`formation_energy → magnetic_moment → tc → klat → material_type`，保证 inverse-design 用到的头（尤其 QC 分类器）最末最新。

kernel 数据规模（非空行数，已核实）：
`magnetic_susceptibility 98 < zt 4971 < power_factor 5223 < thermal_conductivity 6158 < electrical_resistivity 7334 < dos_density 10321 < seebeck 11722`。

完整最终顺序（12 reg + 7 kr 升序 + 5 tail）：
```
# 12 regression (any order, grouped by dataset)
density, efermi, final_energy, total_magnetization, volume,
dielectric_total, dielectric_ionic, dielectric_electronic,   # 8 qc reg
magnetization, curie, neel,                                  # 3 magnetic (non-tail)
kp,                                                          # 1 phonix (non-tail)
# 7 kernel regression, ascending by non-null row count (cheapest first)
magnetic_susceptibility, zt, power_factor, thermal_conductivity,
electrical_resistivity, dos_density, seebeck,
# 5 fixed tail
formation_energy, magnetic_moment, tc, klat, material_type,
```

### Continual rehearsal 机制（沿用 demo + 本次调整）
- AE 头**全程常驻**。
- **不冻结任何层**：每步 `configure_optimizers` 给 encoder + 所有已激活 task head 各建优化器，联合训练（`freeze_shared_encoder=False`、各 task `freeze_parameters=False`）。增量仅靠 rehearsal mask 实现，非冻结。每步重建 Trainer ⇒ 优化器动量每步重置。
- 每步用 `model.add_task()` 增加一个新头；新 task `task_masking_ratio=1.0`。**旧 task 回放比例分级**：
  - 固定末 5（formation_energy / magnetic_moment / tc / klat / material_type）作为旧 task 被回放时用 **`replay_ratio_high=0.10`**；
  - 其余旧 task 用 **`replay_ratio=0.05`**。
- mask 在每步构建训练集时**抽样一次**（不每 epoch 重抽）。
- 每步在**固定 test split** 上评估**所有已激活头**，记录 forgetting 轨迹。

---

## 3. 训练配置

| 项 | 值 | 说明 |
|---|---|---|
| `sample_per_dataset` | `null` | 全量数据 |
| `max_epochs_per_step` | `100` | 上限 |
| **EarlyStopping** | monitor=`val_final_loss`, patience≈8, mode=min | 通常提前收敛（**对 demo 的新增**） |
| `accelerator` | `mps` | Mac GPU（CUDA 不可用） |
| `batch_size` | 256 | |
| `n_grids` | 8 | KMD-1d 描述子，可逆 |
| `latent_dim` / `encoder_hidden` | 128 / 256 | |
| `head_lr` / `encoder_lr` | 5e-3 | |
| kernel: `n_kernel`/`kr_lr`/`kr_decay` | 15 / 5e-4 / 5e-5 | |
| `replay_ratio` | 0.05 | 一般旧 task 回放比例 |
| `replay_ratio_high` | 0.10 | 固定末 5 task 作为旧 task 时的回放比例 |
| `random_seed` / `datamodule_random_seed` | 2025 / 42 | 可复现 |

---

## 4. 每阶段落盘的「原始数据 + plot」

输出目录：`artifacts/continual_rehearsal_full_<YYMMDD>/`

```
step01_density/
  density_pred.parquet         # 新增：test 集 true/pred 原始数组
  density_parity.png
step02_efermi/ …
…
stepNN_material_type/
  material_type_pred.parquet   # true/pred 标签
  material_type_confusion.png
  <每个已激活 task 的 *_pred.parquet 也在该步落盘，便于看 forgetting 的原始数>
forgetting_trajectory.png
experiment_records.json        # 每步 × 每 task 的 metric（at-intro / running）
metrics_table.csv              # 新增：扁平化指标表（task, type, dataset, at_intro, final, metric）
final_model.ckpt               # 新增：最终模型 checkpoint
final_model_taskconfigs.json   # 新增：重建模型所需 task 配置
inverse_design/
  scenario1_*/ scenario2_*/ scenario3_*/   # 见 §5
report.html                    # 自包含 HTML slide deck（沿用 demo）
summary.pptx                   # 新增：python-pptx
summary.md                     # 新增：文字 summary doc
```

**对 demo 的关键扩展**：除现有「仅新 task 出图」外，每步对**所有已激活 task** dump test 集 `(composition, true, pred)` 为 parquet（kernel task 额外存 t 序列），这样 forgetting 既有曲线也有原始数。

---

## 5. Inverse design — 3 个独立剧本

训练**只跑一次**，最终模型存盘后，对**同一模型**依次跑 3 个剧本，**主目标统一为 QC 概率 ↑**。三个剧本的「目标定义」**保持不变**：

| 剧本 | 主目标 | 副目标（reg task → target） | 输出子目录 |
|---|---|---|---|
| 1 | QC↑ | formation_energy −2.0；magnetic_moment +2.0 | `scenario1_fe_down_moment_up/` |
| 2 | QC↑ | formation_energy −2.0；tc +2.0；magnetic_moment +2.0 | `scenario2_fe_tc_moment/` |
| 3 | QC↑ | formation_energy −2.0；klat +2.0 | `scenario3_fe_down_klat_up/` |

### 用户旋钮命名（重要 — 都在 `[0, 1]` 上，直觉对齐）

PR #18 review 阶段把两个原本"看名字猜不到方向"的反向设计旋钮重新命名 + 限值到 `[0, 1]`：

| API 旋钮 | 空间 | 0 的含义 | 1 的含义 | 默认 | 内部数学 |
|---|---|---|---|---|---|
| `ae_align_scale` | latent | **不约束**（AE-align 罚项关闭，即 #18 之前的"无约束 latent"失败模式）| **最强约束**（强制 latent 落到 decode/encode 不动子集）| **0.5**（#18 实测 sweet spot）| 在 loss 上加 `α · ‖h − encode(decode(h))‖²` |
| `diversity_scale` | composition | **最强 peaky 惩罚**（强制每个解只用极少元素）| **不约束**（每个解可以用任意多元素，自由）| **1.0**（无惩罚 = 用户默认期望）| 在 loss 上加 `(1 − d) · H(w)`，`H` 是 Shannon entropy |

两者都是"越大 = 用户角度名字所指的属性越强"——`ae_align_scale=1` 越向 AE 对齐；`diversity_scale=1` 越自由多元素。命名意义直观，不需要看代码就能用。论文里也按这套写。

### 优化路径 — **基于 PR #18 实测的双路径**

旧的无约束 `optimize_latent`（`ae_align_scale=0`）已被证明问题很多（AE round-trip 是瓶颈，#18 实测 QC 0.97→0.35），**不再用作主路径**。每个剧本对**同一组种子 + 同一组目标**，跑下面 **1 个 latent 配置 + 3 个 composition 配置 = 4 条路径**对照：

| ID | 路径 | 关键参数 | 在 #18 中的作用 | 在 plan 中的作用 |
|---|---|---|---|---|
| L | `optimize_latent` | `ae_align_scale = 0.5`, `optimize_space="latent"` | #18 paper run（16 seeds，剧本 = QC↑/FE↓/klat↑）实测 α=0.5 时 QC=0.96±0.027，FE=+0.92，klat=+1.07，是 [0, 1] 上的 sweet spot；α=0 时 QC 崩到 0.39。 | latent 路径的代表 |
| C-strict | `optimize_composition` | `seed_blend = 1.0`，无 `allowed_elements`，`diversity_scale = 1.0` | **baseline**：复刻"只调种子比例、不引入新元素"。#18 paper run 实测：QC=0.887±0.053，FE=+1.27，klat=+0.76；0/16 越出种子池；解平均 2.6 个元素（成分微调，元素族不变）。 | 锚定 strict-seed 基线 |
| **C-alloy** | `optimize_composition` | `seed_blend = 0.95`，`allowed_elements = ALLOY_PALETTE`（见下），`diversity_scale = 1.0` | **推荐**：#18 paper run 用 **12 元素** alloy palette 实测得 QC=0.870±0.012，FE=+0.84，klat=+1.81；100% 输出落在 Mg–Pd–Al 真实准晶族（Pd 不在任何 seed 里 → **元素发现**）；pairwise L1=0.17（收敛紧致）。本 plan 的 **41 元素** palette 已 smoke 实测（见下「预期基线」表）：QC 接近，但 pairwise L1 跳到 1.02 — 优化器同时落到 *两簇*（Mg–Ni–Sc–Ga–Al–Ge 与 Al–Pd–Sm/Sc–Ti），表明白名单越宽 → 元素发现越多元，论文可推"模型识别出多个 QC-prone 元素族"的故事。 | **paper 头条结果** |
| C-rand | `optimize_composition` | `initial_weights=None`, `n_starts = N_SEEDS`, `diversity_scale = 1.0` | **对照**：完全脱离种子，揭示模型预测面上的"全局吸引子"（#18 实测是 Ti/Pu/F/Mn — 模型偏置；含 Pu 这种不可合成元素，物理不现实） | 验证 alloy palette 约束的必要性 |

### "无约束探索能力"专项 ablation — blended-unconstrained vs random-init

PR #18 paper run 里同时跑了 `composition (blended seed, unconstrained)` 与 `composition (random init)`，两者**只差初值**（一个从种子混 5% uniform 出发，一个从纯随机出发），其他所有约束（无 palette、无 element_step_scale、`diversity_scale=1.0`）全相同。实测：

| 配置 | QC | FE | klat | top 5 元素 | pairwise L1 |
|---|---:|---:|---:|---|---:|
| blended seed, unconstrained | 0.792 ± 0.022 | −0.68 ± 0.20 | +1.77 ± 0.03 | Ti(16), Pu(11), F(10), S(9), Mn(9) | 0.76 |
| random init, unconstrained | 0.793 ± 0.005 | −0.78 ± 0.03 | +1.77 ± 0.02 | Ti(16), Pu(16), Mn(16), F(16), Zr(10) | 0.10 |

**这是个系统性发现，不只是冗余信息**：

1. **同一吸引子**：QC / FE / klat 几乎完全一致，top 元素也都是 Ti/Pu/F/Mn — 两条路径殊途同归。
2. **强大的搜索能力**：现有 encoder 在 *无约束* 情况下，**无论从哪里出发，都能高保真地找到模型内部所能表达的"最优 QC"点**。这是论文里要展示的**架构性能强项**。
3. **同时凸显 constraint 的重要性**：这个最优点含 Pu（不可合成）、F（fluoride 不形成 QC）、Mn（在 Mn-rich 系外不易稳定 QC）—— **模型偏置的产物，物理不现实**。无约束的强搜索能力 ↔ 没有约束就误导 — 两面性正好佐证 alloy palette 这类领域知识约束的必要性。

所以本 plan **保留 random-init 作为正式对照路径**（C-rand），并在每个剧本的报告里**与 C-alloy 并列对比**：
- 主目标达成（QC）：C-rand 0.79 vs C-alloy 0.87，差距合理小；
- 副目标（FE/klat）：C-rand 数值更接近 target 边界（无约束自由发挥），但落点在不可合成元素上 → 失去工程价值；
- C-alloy：略损 QC + 副目标向 target 的逼近不如 C-rand 那么"激进"，但**100% 落在可合成元素族**，论文头条价值。

(其他被尝试但移出主路径的配置：`composition (alloy + peaky, diversity_scale=0)` 在 #18 实测 pairwise L1 从 0.17 跌到 0.01，QC=0.85 几乎不变，输出 16/16 趋同到同一个 Al–Pd–Mg 峰 → 是 *peakiness 旋钮* 不是 diversity 间多样性。如有需要可在论文附录以 ablation 形式呈现。)

### 预期基线（来自 PR #18 paper run + 41-elem smoke）

剧本 = `QC↑ / FE↓(target −2) / klat↑(target +2)`，16 seeds。两组都是同一个 checkpoint，差别只在 `allowed_elements`：

| 路径 | QC after | FE after | klat after | pairwise L1 | mean #elems | top-5 elements |
|---|---:|---:|---:|---:|---:|---|
| latent α=0 (failure) | 0.386 ± 0.315 | +2.46 ± 0.59 | −0.44 ± 0.27 | 1.07 | 5.2 | Na, Mg, Ca, Li, Tm |
| latent α=0.5 (sweet) | 0.960 ± 0.027 | +0.92 ± 1.16 | +1.07 ± 0.31 | 0.82 | 3.4 | Mn, Na, Ca, Mg, Yb |
| latent α=1.0 (max) | 0.951 ± 0.027 | +0.40 ± 1.04 | +1.20 ± 0.35 | 1.06 | 3.6 | Mn, Na, Ca, Mg, Ti |
| C-strict | 0.887 ± 0.053 | +1.27 ± 0.24 | +0.76 ± 0.67 | 1.42 | 2.6 | Mg, Zn, Cu, Al, Ni |
| **C-alloy (12 elem)** | 0.870 ± 0.012 | +0.84 ± 0.03 | +1.81 ± 0.07 | 0.17 | 5.6 | **Al, Pd, Mg, Ga, Ni** |
| **C-alloy (41 elem)** | 0.842 ± 0.018 | +0.68 ± 0.07 | +1.84 ± 0.06 | 1.02 | 6.0 | **Ti, Pd, B, Mg, Ga** |
| C-rand | 0.793 ± 0.005 | −0.78 ± 0.03 | +1.77 ± 0.02 | 0.10 | 6.0 | F, Pu, Mn, Ti, Zr |

注：本表用作**新 runner 跑通后的健全性检查**——剧本 3（FE↓ + klat↑）的全量训练结果应在以上数量级附近；偏差过大需要查 (a) seed 选择是否含 17+3、(b) `ae_align_scale` 是否传对（0.5 是 sweet spot）、(c) `seed_blend` 是否被覆盖、(d) palette 是否裁错。

**41-elem 关键观察**（决定论文叙事）：
1. **不再单族塌缩**：pairwise L1 从 0.17 → 1.02，元素发现的多样性显著上升；论文头条从单一"Mg–Pd–Al"扩为"模型识别多个 QC-prone basin"。
2. **Pd 持续被发现**：14/16 输出含 Pd，但 Pd 不在任何 seed → 强**元素发现**信号（"出现率 ≫ 0%、seed 命中率 = 0%"）。论文用这个口径作主要 evidence。
3. **lanthanide 进入解**：Sm 出现在多个输出（Al–Pd–Sm 团簇），扩展到了 Au–Ga–RE 之外的 RE 体系。Au–Ga–Ln 三个 seed 在剧本 1/2 的表现要单独报告。
4. **strict-seed 与 12-/41-elem palette 结果一致**（QC=0.887/0.888，元素分布几乎相同）——证明 strict-seed 路径对 palette 不敏感（seed 元素早就在任何合理 palette 里），可继续作为不变基线。

### 种子（每个剧本共用 — **17 + 3**）

总 N = 20。

- **17 个 top-QC 去重种子**：在 material_type 训练集中按预测 QC 概率排序，按**元素系**（element symbols set，忽略比例）去重，每个元素系保留最高的代表，取前 17 个。代码已在 PR #18 `_select_seeds` 中实现 `_dedupe_by_element_system`。
- **3 个固定 Au–Ga–Ln 配方**（强制追加，无论 QC 预测值如何）：
  - `Au65 Ga20 Gd15`
  - `Au65 Ga20 Tb15`
  - `Au65 Ga20 Dy15`

  这三组是已知或推测的 i-QC 形成体系（Au-Ga-RE 家族），用来检验模型在"明确属于实验已实现/接近实现的 Au–Ga 重稀土"区域是否仍给出合理的 QC 概率。如果模型把这 3 个 seed 的 QC 拉得不高，本身就是一个值得在论文里说的发现。

`_select_seeds` 改造要点（rebase 时实现）：
1. 新增 `inverse_seed_explicit_append: list[str]`（默认 `[]`），追加种子；
2. 改用 `Composition(s).formula` 做归一化避免 `Au65Ga20Gd15` / `Au0.65Ga0.20Gd0.15` 不一致；
3. 通过 `descriptor_fn` 校验追加种子的描述子可计算（不可计算的 fail-fast，给出明确错误）；
4. 输出 `seeds.json` 区分 `top_qc_seeds` 与 `explicit_seeds` 两段。

### 元素清单（`ALLOY_PALETTE`，**41 个**）

`composition (alloy)` 路径的 `allowed_elements` 白名单。范围设计原则：覆盖常见准晶元素 + 易于实验的 4/5 周期过渡金属 + 部分易得镧系，**剔除放射性元素**与极冷门难合成的稀有元素。

| 类别 | 元素 | 数 |
|---|---|---|
| 轻碱土 | `Mg`, `Ca` | 2 |
| Group 13 | `B`, `Al`, `Ga`, `In`, `Tl` | 5 |
| Group 14 | `Si`, `Ge` | 2 |
| 4th-period TM（Sc–Zn 全） | `Sc`, `Ti`, `V`, `Cr`, `Mn`, `Fe`, `Co`, `Ni`, `Cu`, `Zn` | 10 |
| 5th-period TM（Y–Cd，去 Tc 放射性） | `Y`, `Zr`, `Nb`, `Mo`, `Ru`, `Rh`, `Pd`, `Ag`, `Cd` | 9 |
| 6th-period noble（用于 Au–Ga–Ln seed） | `Au` | 1 |
| 易得镧系（去 Pm 放射性、Tm/Lu 稀贵） | `La`, `Ce`, `Pr`, `Nd`, `Sm`, `Eu`, `Gd`, `Tb`, `Dy`, `Ho`, `Er`, `Yb` | 12 |

合计 41 个，落到 config 里写成：

```toml
composition_allowed_elements = [
    "Mg", "Ca",
    "B", "Al", "Ga", "In", "Tl",
    "Si", "Ge",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Au",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Yb",
]
```

这套白名单同时覆盖：
- 经典 i-QC 三元体系（Mg–Zn–RE、Al–Mn、Al–Cu–Fe、Zn–Mg–RE、Ti–Zr–Ni 等）；
- d-QC 体系（Al–Ni–Co、Al–Cu–Co 等）；
- 重稀土 RE-stabilized 体系（Au–Ga–RE、Mg–Zn–RE）所需的 Au；
- Si/Ge/B 等族 13/14 元素，便于 Mg–Si–Ge 这类边缘体系；
- 3 个追加种子的全部元素（Au/Ga/Gd/Tb/Dy）。

### 评估指标（每剧本 × 每路径）

不止报 QC round-trip 概率。对每条路径在 20 个 seed 上输出：

| 指标 | 形式 | 说明 |
|---|---|---|
| `qc_after` | mean ± std | 主目标，softmax 后 QC 类（merged）的概率 |
| `<reg_task>_after` | mean ± std | 每个副目标 reg task 的解码后预测值 |
| `dist_to_seed_l1` | mean ± std | 每个解 vs 自己的 seed（latent / strict / alloy）或最近 seed（random）的 L1 距离 |
| **`pairwise_l1`** | scalar | **20 个解两两之间** L1 距离的平均（94 维元素权重单纯形上）。**定义**：对 N=20 个解的所有 C(20,2)=190 对 `(w_i, w_j)`，取 `mean Σ_k |w_i[k] − w_j[k]|`。值域 [0, 2]：0 = 20 个解完全一样；2 = 完全正交。**intra-method 多样性**——同一路径给出的候选库本身有多分散。实测参考：strict seed 1.42（每个 seed 都被推到不同方向，最分散）；alloy palette 12-elem 0.17（16 个 Mg-Pd-Al 微变体，紧致）；alloy + `diversity_scale=0` 0.01（全部塌成同一个峰）。 |
| `unique_element_systems` | int / N | 20 个解中不同元素集合的数量 |
| `out_of_seed_pool` | int / N | 解的元素超出种子元素池（17+3 合并）的样本数 |
| `mean_n_elements` | float | 每个解的非零元素数平均 |
| `top_elements` | list[(symbol, count)] | 出现在最多解里的前 8 个元素 |
| **`discovered_elements`** | list[(symbol, hit_rate)] | **元素发现专用**：出现率 ≥ 50% **且** 在 20 个 seed 中出现次数 = 0 的元素。这是论文里"模型发现了 X"的硬证据信号（#18 paper run 里 Pd 是 16/16 出现、0/16 seed → hit_rate = 100% 的发现元素）。**该字段为空意味着该路径只是种子比例微调，不是元素发现**。 |
| `elapsed_s` | scalar | 单次 `optimize_*` 调用耗时 |

`discovered_elements` + `dist_to_seed_l1` + `out_of_seed_pool` 联合回答论文核心问题："这是元素发现还是种子比例微调？"

**Raw arrays（必存）**：除上述聚合指标外，`results.json` 还必须包含每路径的 `optimized_weights`（形状 `(B, n_components)`，元素顺序与 `DEFAULT_ELEMENTS` 一致）和 `optimized_descriptor`（形状 `(B, x_dim)`）。这两份原始数组是日后调整图表方案（per-element bar chart、相似度矩阵、ratio 直方图等）的来源——**不用重跑实验**。已在 `paper_inverse_comparison.py` / `eval_inverse_methods.py` 的两个 runner 中加好。模型权重 `final_model.pt` + seeds + targets + 原始数组 = 论文素材的最小可重现集合。

### Smoke check（正式 run 前必须通过）

在 GPU 启动 24-step 训练 + 3 剧本之前，**必须**先跑一次 smoke 验证 §5 的 4 条路径都能产出**合理的数量级**，免得训练几小时后才发现 inverse-design 配置错了。复用 `paper_inverse_comparison.py` + 现成 `artifacts/paper_inverse_design/final_model.pt` checkpoint，用 17+3 的新 seed 方案跑一次：

```bash
# 1. 在临时输出目录跑 4 路径对比（不污染 artifacts/paper_inverse_design/）
python -m foundation_model.scripts.paper_inverse_comparison \
    --config-file samples/continual_rehearsal_demo_config_inverse_baseline.toml \
    --checkpoint artifacts/paper_inverse_design/final_model.pt \
    --output-dir artifacts/smoke_inverse_4path
# 2. 比对 §5「预期基线」表 — 数量级偏差 < 0.1 即通过；偏差大查 seed/参数。
```

注意 `paper_inverse_comparison.py` 目前用的是 12-elem palette + 16 seeds。smoke check 时如想直接用本 plan 的 17+3 seed + 41-elem palette，临时改三处：
- 把 `DEFAULT_ALLOY_PALETTE` 改成本 plan 的 41-elem 列表；
- `_select_seeds` 在 `ContinualRehearsalRunner` 上的调用前后追加 `Au65Ga20Gd15` / `Tb15` / `Dy15`；
- 用 `_dedupe_by_element_system` 取前 17 个 top-QC 后追加这 3 个 → 共 20。

正式 runner（`continual_rehearsal_full.py`）落地后再把这些写成正式 config 字段，smoke check 用 runner 自己的 `--inverse-only` 模式即可。

### Per-scenario 成功判据（人工核对，不卡 CI）

每个剧本完成后，论文中要主张"实验有效"，至少其中一条必须成立：

1. **C-alloy 路径**在该剧本主目标 QC ≥ 0.80 **且** 至少一个副 reg target 命中目标方向（`(pred − target) · sign(target − seed_mean)` 在合理范围内）；
2. **C-alloy** 的 `discovered_elements` 非空（典型预期：Pd 或某 5th-period TM 被发现）；
3. **L (latent)** 的 QC > C-strict 的 QC 至少 0.05 — 否则说明 cycle 罚项也帮不上忙，本剧本对模型来说"无解"，论文需诚实标记。

若三个剧本里有两个不满足任一条 → 检查训练是否欠拟合（forgetting trajectory 看 tail 5 task 的 final R² / accuracy）或 inverse-design 超参（`inverse_class_weight`、`inverse_steps`、`inverse_lr`）。

### 输出落盘

```
inverse_design/
  scenario1_fe_down_moment_up/
    seeds.json                          # 17 top-QC + 3 explicit, 分两段
    targets.json                        # 主+副目标定义
    latent_lambda1/
      results.json                      # 每 seed 一行：qc/reg/decoded_composition
      metrics.json                      # 上表所有聚合指标
      decoded.txt                       # 人读组成清单（KMD.inverse 解码）
      summary.png                       # 3-panel: QC + 每个 reg target，bar + error
    comp_strict_seed/
      ...（同上结构）
    comp_alloy_blended/                 # **headline**
      ...
    comp_random_init/
      ...
    comparison.png                      # 4 路径 × 3 panels（QC / FE / 副 reg）并列对比，与 paper 主图同款式
    comparison_diversity.png            # 4 路径 × 3 panels（pairwise L1 / out-of-seed / mean_n_elements）
  scenario2_fe_tc_moment/
    ...
  scenario3_fe_down_klat_up/
    ...
  README.md                             # 三个剧本的 takeaway 一页摘要
```

`comparison.png` 与 `comparison_diversity.png` 都直接复用 `paper_inverse_comparison.py` 的绘图风格（`#2563EB` for composition, `#55A868` for latent, `#C44E52` for target line；x-tick rotation 45）。

### 实现路径（rebase 时改写 `_inverse_design`）

旧的 `_inverse_design` 是占位实现，按下面顺序重写：

```python
PATHS = [
    ("latent_align0p5", "latent", {"ae_align_scale": 0.5}),
    ("comp_strict_seed", "composition", {"seed_blend": 1.0}),
    ("comp_alloy_blended", "composition", {
        "seed_blend": 0.95,
        "allowed_elements": ALLOY_PALETTE,
    }),
    ("comp_random_init", "composition", {"initial_weights": None, "n_starts": 20}),
]

for scenario in cfg.inverse_scenarios:
    seeds = _select_seeds(...)  # 17 top-QC 去重 + 3 explicit
    for path_label, mode, extra_kwargs in PATHS:
        result = _run_path(model, seeds, scenario.targets, mode, extra_kwargs)
        _dump_path_results(result, scenario_dir / path_label)
    _plot_comparison(scenario_dir)
    _plot_comparison_diversity(scenario_dir)
```

参考 `paper_inverse_comparison.py` 的 `_run_latent_method` / `_run_composition_config` 实现一对一复用。

---

## 6. 交付物 — `summary.pptx`（16:9 · 白底 · 主色 + 至多两辅助色）

### 配色

- 白底（`background = #FFFFFF`），整体留白多。
- **主色** `#2563EB`（与 demo regression scatter 一致；PR#18 已定为唯一蓝）。
- 辅助色 ≤ 2 个：`#55A868`（绿，正向/达标）、`#C44E52`（红，target line/类别误判）。这三色就是图里在用的，幻灯片元素（标题下划线、强调框、bullet 项 marker、表头底纹）一律从这套色里取，**不引入其他颜色**。
- 任务区分用色（forgetting trajectory 等多线图）仍走 demo 的 12 色 qualitative 调色板，但**只在线图里出现**；其他幻灯片元素严守上述三色。

### 内容与编排

| # | 用意 | 主要素 |
|---|---|---|
| 1 | **Title** | 项目名 + 一行 tagline + 日期 / git SHA |
| 2 | **数据集 & 实验目标 — by task type** | 三栏（regression / kernel regression / classification），列出每类下属 task、来源数据集与数据量；右侧 callout：本次实验目标 = 在 24 个 task 上做 continual learning + 在共享 latent 上做 3 个 inverse design 剧本 |
| 3 | **模型 & 优化算法** | shared encoder + multi-head 架构图（一行示意 + bullet）；KMD-1d 描述子（PR#18 起**可微分**）；说明**为何弃用无约束 latent**（AE round-trip 是瓶颈，#18 实测 α=0 时 QC→0.39）；并列展示 §5 的**四条路径**：(L) latent + AE-align 罚项 (`ae_align_scale=0.5`)、(C-strict) `seed_blend=1.0`、(C-alloy) `seed_blend=0.95` + alloy palette（**头条**）、(C-rand) random init 控制 |
| 4 | **持续学习中的遗忘问题** | 概念：旧任务被新任务覆盖；naive sequential training 的失败模式；展示「想象中（不衰退）vs 现实（衰退）」 |
| 5 | **我们的应对策略** | 分级 rehearsal（5% / 10% 对 inverse-design tail）+ 不冻结任何层 + EarlyStopping；说明为什么对 tail 给更高 ratio；mask once-per-step 的设计 |
| 6 | **遗忘的实测效果** | `forgetting_trajectory.png`（widened，PR#18 风格）+ 一张紧凑表：headline 5 task 的 at-intro / final / Δ |
| 7 | **Inverse design 剧本 1** — `FE↓ + Magnetic Moment↑` | setup（主目标 QC↑、副目标列表、`seeds.json` 中 17+3 的两段）+ **四条路径并列展示** result（QC + 副目标条形图 + 多样性条形图）+ 短分析：QC 概率上升幅度、副目标方向是否对、解的元素分布（**catagoly = 元素组成所在合金族 / 已知 QC 体系**，一句话定性）；Au–Ga–Ln 三个 seed 在剧本 1 的表现单独点评 |
| 8 | **Inverse design 剧本 2** — `FE↓ + Tc↑ + Moment↑` | 同上 |
| 9 | **Inverse design 剧本 3** — `FE↓ + κ_lat↑` | 同上 |
| 10 | **总结** | 一个共享 encoder 覆盖 24 task；分级 rehearsal 把 inverse-design tail 守住；inverse design 三剧本主目标 QC 概率均显著提升；副目标方向正确；解可被解码回可读 composition |
| 11 | **Try it on your data** | 直观示意：「`composition + property`（或 `category`）的数据集 → 一行 task config 注册 → 接入共享 encoder → 即刻开始训练 / 探索 inverse design」；列一个最小 task config 片段；强调任何 downstream 数据形态都能即插即用 |

### 实施备注

- 当前 runner 里 `_write_pptx` 是**老版结构**（9 页、按 dataset 分页等），保留以保 smoke 通跑。**post-#18 rebase 时**改写成上面 11 页结构，并把所有非线图色限制到主+2 辅助色。
- 同步产出 `summary.md`（11 节文字版）与 `report.html`（自包含 deck，能直接打印 PDF）。
- 颜色与字体在 `_apply_plot_style()` 基础上额外加一个 `_pptx_palette` 常量，便于一处改色。

---

## 7. 代码改动清单（新建，不动 demo）

- **新增** `samples/continual_rehearsal_full_config.toml`：全部路径、24-task `task_sequence`、§3 配置、3 个 inverse 剧本（用一个 `[[inverse_scenarios]]` 列表表达）。
- **新增** `scripts/continual_rehearsal_full.py`：
  - 扩充 `TASK_SPECS`（+12 个新 task）与 `TASK_DISPLAY`（中英文友好名）。
  - 复用 demo 的 `descriptor_fn` / KMD / 评估 / 绘图 helper（import 或抽到共享模块）。
  - 训练循环加 **EarlyStopping**（需 val dataloader，CompoundDataModule 已提供）。
  - 每步 dump 所有激活 task 的 `*_pred.parquet`。
  - 训练后 **保存 `final_model.ckpt`** 与 task 配置。
  - inverse design 改为 **遍历 `inverse_scenarios` 列表**，对同一模型跑 3 次，分目录落盘。
  - 新增 `summary.pptx`（python-pptx）+ `summary.md` 生成，保留 `report.html`。
- **新增** `run_continual_rehearsal_full.sh`：仿 `run_continual_rehearsal_demo.sh`，默认配置 + 日期戳输出目录。
- 共享逻辑若从 demo 抽取，会保证 demo 行为不变（仅 import，不改语义）；并补/跑相关 `*_test.py`。
- `uv add python-pptx`，更新 `uv.lock`。

---

## 8. 风险与备注

- **耗时**：24 step × 全量数据，即便 MPS + 早停也可能数小时。建议后台跑并定期回看。
- **magnetic_susceptibility（98 行）**、部分 dielectric / magnetic 列为子集 —— 个别 task R² 可能偏低或不稳定，报告会标注。
- **MPS 兼容性**：极少数算子在 MPS 上可能缺失；若报错，回退 `accelerator=cpu`（更慢）。
- raw 回归 z-score 用 train 行统计，避免泄漏；clip ±5。
- inverse-design 解码用 KMD.inverse（可逆描述子），可能出现 `<undecodable>` 边缘情况（已有 warning 兜底）。

---

## 9. 执行步骤（确认后）

1. `uv add python-pptx` 并 sync。
2. 写 `scripts/continual_rehearsal_full.py` + `samples/continual_rehearsal_full_config.toml` + `run_continual_rehearsal_full.sh`，补测试。
3. `ruff format && ruff check && mypy src` + 跑相关 `pytest`。
4. **小规模 smoke**（`--sample-per-dataset 800 --max-epochs-per-step 2`）验证端到端不报错、产物齐全。
5. 启动**正式全量 run**（后台），完成后核对 forgetting / 5 个目标 task 指标 / 3 个 inverse 剧本 / PPT+MD。

---

## 10. 执行状态（2026-05-23）

- ✅ `uv add python-pptx`（runtime dep，已写入 `uv.lock`）。
- ✅ 新增 `src/foundation_model/scripts/continual_rehearsal_full.py` + `_test.py`（16 tests 通过）、
  `samples/continual_rehearsal_full_config.toml`、`run_continual_rehearsal_full.sh`。
- ✅ `ruff format` / `ruff check` / `mypy`（new module）全绿。
- ✅ **CPU smoke**（`--sample-per-dataset 800 --max-epochs-per-step 2 --accelerator cpu`）端到端通过：
  24 个 step、每步全 task `*_pred.parquet`、3 个 inverse 剧本、`final_model.ckpt`、`metrics_table.csv`、
  `forgetting_trajectory.png`、`report.html`、`summary.md`、9 页 `summary.pptx` 全部产出。产物在
  `artifacts/continual_rehearsal_full_smoke/`（可丢弃）。**注意**：smoke 跑的是「旧顺序」（19 free + 5 tail），正式 run 用本次更新后的「12 reg + 7 kr 升序 + 5 tail」新序。
- ⏸ **正式全量 run 待启动** —— GPU 被另一训练任务占用，且 PR #18 待合并。两者就绪后执行：

  ```bash
  ./run_continual_rehearsal_full.sh            # 默认 config，MPS，全量数据，输出带日期戳
  ```

  （会写到 `artifacts/continual_rehearsal_full_<YYMMDD>/`；建议后台运行。）

> 修复记录：TOML 中 `[[inverse_scenarios]]` array-of-tables 必须置于文件末尾，否则其后的顶层标量键
> 会被并入最后一个 scenario 表（已在配置中调整顺序并加注释）。

---

## 11. PR #18 依赖与 rebase 计划

PR#18 在 #17（differentiable KMD upstream）之上落了若干与本工作流相关的改动：算法（cycle-consistency
latent + differentiable composition）会改变 inverse-design 的可选 backend；配色 / plot 风格会经 demo
模块自动透传到本 runner（因为我 `import` 的就是这些 helper）。#18 PR body 明确说 `continual_rehearsal_full.py` 工作流**不在** #18 范围。

### #18 引入的可被复用 / 必须感知的部分

| 项 | 类型 | 对本 runner 的影响 |
|---|---|---|
| `ae_align_scale` on `optimize_latent` | **必须用** | 给 latent 路径加 AE-alignment 罚项 `α · ‖h − encode(decode(h))‖²`；α=0 = 无约束 = 失败模式（QC→0.39）。**[0, 1] 范围**，默认 0.5（#18 实测 sweet spot）。命名演变：`cycle_consistency_weight` → `ae_cycle_weight` → 最终 `ae_align_scale`（更直觉）。 |
| `optimize_composition`（differentiable KMD） | **必须用** | 94 维 element-weight 单纯形上微分优化；本 plan 跑 3 个配置（strict-seed / alloy-blended / random-init），详见 §5 表。 |
| `seed_blend` (new in #18 fix-up) | **必须用** | composition 路径的核心旋钮：`1.0` = 锁定支撑集（baseline），`0.95` = 允许优化器引入新元素（alloy 路径用此值）。 |
| `diversity_scale` on `optimize_composition` | 可选 | **[0, 1]** 范围，1 = 不约束（默认，最 diverse 多元素），0 = 强惩罚（最 peaky 少元素）。命名演变：`sparsity_weight` → `entropy_weight` → 最终 `diversity_scale`（更直觉）。本 plan 默认 1.0，不主用；仅在论文附录展示 0.0 的 peaky 模式 ablation。 |
| `element_step_scale` hard-lock (#18 PR review fix) | 可选 | `0.0` 现在真正锁定权重（不只是 logit gradient）。本 plan 不主用，但保留作为"锚定 seed 比例 + 只允许新元素进入"的高级手段。 |
| `_dedupe_by_element_system` in `_select_seeds` | **必须用** | top-QC 排序后按元素系去重；本 plan 取前 17 个，再追加 3 个显式 Au–Ga–Ln seed（§5）。 |
| `class_weights` always-registered buffer (#18 PR review fix) | 流程 | state_dict 跨配置 strict-load 不再失败；我们的 `final_model.ckpt` 加载将更稳健。 |
| `material_type` 3-class 合并 + 类权 + plot 通刷（`#2563EB` scatter、widened forgetting、row-normalized confusion、dpi=150） | demo 内部 | 已 `import` 这些 helper、复用同样 merge map；rebase 后视觉自动对齐。需 verify：`_MATERIAL_TYPE_MERGE` / `MATERIAL_TYPE_CLASSES` / `MATERIAL_TYPE_DISPLAY_ORDER` / `_SCATTER_COLOR` 命名是否仍可 import。 |
| `--inverse-only <ckpt>` (demo 端) | 流程 | demo 跑过后可只重跑 inverse；rebase 时给 `continual_rehearsal_full.py` 加同样的 `--inverse-only` + `--checkpoint`。 |
| `final_model.pt` 强制保存（demo 端） | 流程 | 我们已存 `final_model.ckpt`；可改名为 `final_model.pt` 对齐 demo。 |
| `paper_inverse_comparison.py` | 参考实现 | `_run_latent_method` / `_run_composition_config` 是 §5 双路径的直接母版，rebase 时**复用其函数**，不重复实现。 |

### Rebase 步骤（#18 合并到 master 之后）

1. `git fetch origin && git rebase origin/master`；解冲突主要在 demo 的 helper / 配色常量。
2. **验证 imports 依然成立**：`_apply_plot_style` / `_PALETTE` / `_SCATTER_COLOR` / `_REPORT_TEMPLATE` / `_as_float_array` / `_composition_key` / `_init_kernels`；并新增 `QC_CLASSES` / `_dedupe_by_element_system` from `continual_rehearsal_demo`。
3. **接入 PR#18 算法（按 §5 表）**：
   - `ContinualRehearsalFullConfig` 新增字段：
     - `inverse_ae_align_scale: float = 0.5`（latent 路径，[0, 1]，0.5 是 #18 sweet spot）；
     - `inverse_seed_explicit_append: list[str]`（显式追加 seed，**默认即 §5 三个 Au–Ga–Ln**）；
     - `inverse_n_top_qc_seeds: int = 17`（top-QC 去重后取前 N）；
     - `inverse_composition_alloy_palette: list[str]`（默认即 §5 的 41 元素清单）；
     - `inverse_composition_seed_blend: float = 0.95`。
   - `_select_seeds` 改为 "17 top-QC 去重 + 3 显式" 两段拼接，并校验显式 seed 的 descriptor 可计算；输出 `seeds.json` 区分 `top_qc_seeds` / `explicit_seeds`。
   - 重写 `_inverse_design`：对每个 `scenario` 跑 §5 的 4 条路径，复用 `paper_inverse_comparison.py` 的两个 runner 函数（`_run_latent_method` / `_run_composition_config`）。
   - 落盘按 §5 目录结构；`comparison.png` 与 `comparison_diversity.png` 直接复用 paper 风格。
4. **改写 `_write_pptx`** 为 §6 的 11 页结构、主色 + ≤2 辅助色（提取 `_pptx_palette` 常量）。
5. **task_sequence 已在 config 中按新序更新**（本次同步），smoke 再跑一次确保通过。
6. **lint / type / test / smoke**：`ruff format && ruff check && mypy src && pytest src/foundation_model/scripts/continual_rehearsal_full_test.py`，再
   `./run_continual_rehearsal_full.sh ... --sample-per-dataset 800 --max-epochs-per-step 2 --accelerator cpu` 端到端 smoke。
7. **缩小版正式 run**：受时间成本约束，先跑缩小规模（`--sample-per-dataset 5000`、`--max-epochs-per-step 30` 量级，具体值在 rebase 后根据 smoke 时长定）。全量 run 留到论文最终复现阶段。
8. **GPU 空闲后**启动缩小版 MPS 正式 run。
