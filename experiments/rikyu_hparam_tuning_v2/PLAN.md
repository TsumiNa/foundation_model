# 扩展版调参计划 v2（RIKYU）

> 本文档为**全新 session 的执行说明**，不依赖任何对话上下文。执行者应先通读全文，再动手。
> 前一轮（v1）的全部产物在 `experiments/rikyu_hparam_tuning/`，本文档大量引用它。

---

## 0. 为什么要重跑

v1 的全部调参结论，都是在一个**学习率调度器坏掉**的环境下测出来的。

PR #45 之前，`ReduceLROnPlateau` 在 `training_step` 里被调用 —— 即**每个 batch 步进一次**，
而不是每个 epoch。`patience = 5` 于是数的是 batch：在 ~90 batch/epoch 的规模下，
LR 在**第 1 个 epoch 之内**就从 `5e-3` 一路砍到 `min_lr = 1e-4` 地板，然后整个训练都趴在地板上。

修复后实测（v1 的 patience A/B，9 run = 3 臂 × 3 seed，probe3）：

| 对比 | Δ mean R² | vs seed band | 结论 |
|---|---:|---:|---|
| **patience 净效应**（0.2.1 → 0.3.2，weight decay 保持一致） | **+0.0263** | **1.99×** | 超出噪声 |
| weight decay 敏感度（encoder 1e-2→1e-3, head 1e-5→1e-3） | −0.0052 | 0.40× | 噪声内 |

三个 seed 分布**完全不重叠**（新臂最差 0.8369 > 旧臂最好 0.8226）。新臂还**更早收敛**
（平均最终 epoch 114 vs 136）且**快 21%**（994s vs 1255s）。

**这意味着 baseline 被抬高了，而且 v1 调出来的参数是在错误 régime 下选的。**

### 具体哪些 v1 结论失效

| v1 结论 | 状态 | 原因 |
|---|---|---|
| `encoder_lr` 最优 = 1e-3，且是**内点**（A1b 验证 2e-4/5e-4 更差） | ❌ **必须重测** | 旧 régime 下 LR 几乎立刻塌到 1e-4，所谓"调 encoder_lr"实际调的是**塌陷前那一小段窗口的初始 LR**。调度器正常后，最优点几乎肯定移动 |
| `encoder_lr = 0.01` 会发散（−8.6%，最差格 −45.6%） | ❌ **必须重测** | 发散很可能正是"没有有效退火"的后果。调度器能正常退火后，更高的初始 LR 反而可能变好 |
| Stage B 每任务 head 调参：24 个只有 2 个survive | ❌ 重测，且**换设计**（见 §3） | 同样在坏 régime 下测的；且 v1 已证明单任务调出的 head 不迁移到多任务 |
| `latent_dim` / `encoder_hidden_dims` 是弱效应、最优是**平台**不是尖峰 | ⚠️ 可能仍成立 | 但平台的位置会随 LR régime 移动，粗采样复查 |

### 哪些 v1 结论大概率仍成立（**粗采样复查即可，别再花大钱**）

| 结论 | v1 证据 |
|---|---|
| `batch_size = 256`（512 / 1024 都更差） | A2 |
| `n_grids = 8`（16 无改善、4 大幅变差） | A3 |
| `ae_lr = 5e-3`（1e-3 / 1e-2 都在带内） | A6 |
| weight decay 不重要 | patience A/B，0.40× band |

### 新增的可调维度（**v1 时期根本不可调**）

调度器修好之前，下面这些参数写了也没意义 —— 它们现在**第一次真正生效**：

- `[training.scheduler] patience`（现在数 epoch）
- `[training.scheduler] factor`
- `[training.scheduler] min_lr` ← **最重要**。旧 régime 下它其实是**事实上的训练 LR**；
  现在它才回归"地板"的本意。`min_lr` 与 `encoder_lr` 的比值决定了调度器有多少退火空间
  （`OptimizerConfig` 会直接拒绝 `min_lr >= lr`）
- `[training] max_epochs` / `[training.early_stopping] patience` 也值得复查：
  新臂在 114 epoch 就早停（旧臂 136），150 的上限可能不再是约束

---

## 1. 总体设计

三阶段，外加一个"锚定"阶段。**每个阶段都必须先跑完再开下一个**——后一阶段依赖前一阶段固定下来的底座。

```
Stage 0  锚定       在 0.3.2 上复现 v1 采纳配置 + 未调基线，确立新 régime 下的参照系
Stage A' encoder    encoder 架构 × LR × 调度器  联合网格（新增调度器维度）
Stage B' head       多任务联合 head 调参（不再每任务单独调 —— v1 已证明不迁移）
Stage C' 最终       24 任务 hybrid replay + consolidation，对比 v1 的 Stage C
```

### 与 v1 的对比关系

v1 的 Stage C 正在跑（`c_base` 未调 / `c_tuned` v1 调参），会正常跑完。
它们是 v2 的**外部对照**：

| 参照 | 含义 |
|---|---|
| v1 `c_base` | 未调 + 坏调度器 |
| v1 `c_tuned` | v1 调参 + 坏调度器 |
| v2 Stage 0 锚点 | 未调 + **好调度器** ← 隔离出"光升级不调参"能拿多少 |
| v2 Stage C' | v2 调参 + 好调度器 ← 最终交付 |

这四个点合起来才能回答"调参本身值多少、升级值多少"。**不要跳过 Stage 0**，
否则 v2 的收益无法与 patience 修复的收益分离 —— 那正是 v1 犯过的错（见 §5 教训 3）。

---

## 2. Stage A' —— encoder + 调度器联合网格

### 关键：LR 与调度器必须联合搜，不能分开

v1 把 `encoder_lr` 当独立旋钮调。现在 `encoder_lr`（起点）、`min_lr`（地板）、
`factor`（每次砍多少）、`patience`（多久没改善才砍）**共同决定一条 LR 轨迹**。
单独调任何一个都会得到 régime 相关的假最优 —— v1 正是这么栽的。

### 建议网格（分两轮，先粗后细）

**A'1 —— 粗筛（单 seed，宽范围）**

| 维度 | 取值 | 说明 |
|---|---|---|
| `training.encoder_lr` | `1e-3, 3e-3, 1e-2, 3e-2` | **向上扩**。v1 在 0.01 发散，但那很可能是没有退火所致，必须重测 |
| `training.scheduler.min_lr` | `1e-6, 1e-5, 1e-4` | v1 固定 1e-4（且它就是事实训练 LR）。往下扩 |
| `training.scheduler.patience` | `5, 15` | 现在是 epoch |
| `model.latent_dim` | `128, 384` | v1 采纳 384；粗采样确认平台没移动 |

4 × 3 × 2 × 2 = **48 点**。probe3 单 run 约 0.34h（v1 实测），48 点并行约 0.5–1h 墙钟。

固定不动：`encoder_hidden_dims=[256]`（v1 采纳）、`batch_size=256`、`n_grids=8`、
`ae_lr=5e-3`、`factor=0.5`、weight decay 全部默认。

> **必须检查网格边界**：若最优落在 `encoder_lr` 或 `min_lr` 的端点上，
> **必须像 v1 的 A1b 那样加一轮外扩**再定论。v1 这一步做对了，务必保留。

**A'2 —— `factor` 与 early stopping（在 A'1 冠军上）**

| 维度 | 取值 |
|---|---|
| `training.scheduler.factor` | `0.3, 0.5, 0.7` |
| `training.early_stopping.patience` | `24, 40` |

6 点。目的是确认 150 epoch 上限与早停 patience 在新 régime 下不再是约束。

**A'3 —— seed 确认（决定性一步，不可省）**

把 A'1/A'2 的**前 3 名 + 未调基线**各跑 **5 个 seed**（v1 只跑 3 个，不够）。

- 报告口径：均值 ± 极差，**所有排序都必须对着噪声带说话**
- 只有超出噪声带的差异才允许写进结论
- v1 实测该 probe 噪声带 **8.48%**，且它**推翻了 A1 的单 seed 榜首** —— 这是 v1 最重要的教训

4 配置 × 5 seed = **20 run**。

### Stage A' 交付

固定下来的 encoder + 调度器"底座"，后续阶段一律不再改动。

---

## 3. Stage B' —— 多任务联合 head 调参（**设计已改**）

### 为什么不再每任务单独调

v1 对 24 个任务各自单独调 head，结论是：**只有 2/24 的收益扛过 seed 重复，5 个任务反而退化**。
而 v1 的 B-mt 对照组（在多任务 probe 上联合调**一个**共享 head 配置）**是有效的**。

结论很明确：**调参必须在部署的 régime 下做**。单任务调出来的 head 不迁移到 24 任务连续预训练。

所以 v2 直接采用联合调参，不再重复 v1 那条已被证伪的路线。

### 建议网格

在 Stage A' 固定的底座上，用 **probe3（回归）** 和 **probe3_kr（核回归）** 两个多任务 probe：

| 维度 | 取值 |
|---|---|
| `model.head_hidden_dims` | `[64]`, `[256,128]`, `[512,256,128]` |
| `training.head_lr` | `1e-3, 5e-3` |
| `model.kr_x_hidden_dims`（仅 KR probe） | `[128,64]`, `[256,128,64]` |
| `training.kr_lr` | `1e-4, 5e-4` |

回归 probe：3 × 2 = 6 点；KR probe：3 × 2 × 2 × 2 = 24 点。前 3 名各跑 5 seed 确认。

分类任务（`material_type`）v1 已证明对超参不敏感（±0.005），**不单独调**，沿用默认。

---

## 4. Stage C' —— 最终 24 任务

配方沿用 `HYBRID_RECIPE.md`（v1 已验证，不要改动）：

```toml
[pretrain.replay]
interval = 1
resample = "epoch"
amount = 0.30          # amount_t = max(1500, 0.3 * N_t)
per_task = { <每个 N < 5000 的任务> = 1500 }
```

两个 arm，各跑一遍 pretrain + consolidation：

- `c2_tuned` —— Stage A' + B' 的全部采纳参数
- `c2_base` —— 未调默认（在 0.3.2 上）← 这就是 Stage 0 的锚点，可复用

跑完后与 v1 的 `c_base` / `c_tuned` 四点合并对比（见 §1）。

预算参考（v1 实测）：24 任务 hybrid ≈ 17–20h（`c_base` 更慢，KR 步是大头）；
consolidation ≈ 1.5h，但给 10h 预算。`fm finetune` **没有 `--resume`**，墙钟必须一次给够。

---

## 5. 执行环境（**踩过的坑，务必先读**）

### 教训 1：远程命令必须用登录 shell —— 否则 Slurm 会伪装成集群故障

```bash
ssh rikyu-login 'squeue'                  # ❌ DNS SRV lookup failed / Could not establish a configuration source
ssh rikyu-login 'bash -lc "squeue"'       # ✅
```

非登录 shell 下 Slurm 找不到控制器，报错**看起来完全像站点故障**（我为此误判过，还差点让人去联系 R-CCS）。
**所有** `sbatch` / `squeue` / `sacct` 都必须包在 `bash -lc` 里。

### 教训 2：`apptainer` 只在计算节点上有

登录节点没有 `apptainer`，也没有对应 module。**拉镜像必须提交成作业**。
可复用 `~/jobs/pull_image.sbatch`（v1 已建，改 `V=` 即可）。

### 教训 3：把混淆变量钉死，否则结论无法归因

v1 的 patience A/B 一度把"cadence 变化"和"weight decay 变化"混在一起测。
**任何跨镜像/跨版本对比，都要先逐字段核对两边的实际生效值**，
而且要核对**调用点**而不只是字段默认值 —— v1 就是只看了默认值，
没看到 `_engine.py` 里的硬编码，得出了错误的"defaults 变了"的判断。

### 教训 4：结果不进 git

`experiments/*/results/` 在 `.gitignore` 里。**只有 config / 作业脚本 / 分析代码进 git**，
结果用 `rsync` 在机器间同步。

### 关键路径与命令

```bash
# 镜像（arm64，仅限 RIKYU 的 Grace/GB200）
~/containers/foundation-model_rikyu-<version>.sif
docker://ghcr.io/tsumina/foundation_model:rikyu-<version>

# 输出根目录（不在仓库里！）
OUTBASE=/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2

# 提交（沿用 v1 的 submit.sh，需要为 v2 的 stage 增加条目）
experiments/rikyu_hparam_tuning/scripts/submit.sh <stage> [--time HH:MM:SS] [--throttle N]

# 逐 run 切换镜像时用（v1 为 patience A/B 写的独立启动器）
experiments/rikyu_hparam_tuning/scripts/patience_ab.sbatch
```

### 幂等性

所有作业脚本用 **DONE marker** 跳过已完成的点。墙钟被杀或部分失败后，
**重新提交同一条命令**即可只补跑缺的 —— 这是 v1 验证过的恢复方式。
`fm pretrain` 有 `--resume`；`fm finetune` **没有**。

### 强制 smoke

任何长跑之前，先在**真实执行环境**里跑 `--sample 400 --max-epochs 1` 的全链路 smoke。
v1 的 smoke 抓到过配置在新镜像上从未验证过的问题。
**并且要核实 smoke 真的训练了**（产出 step 目录和 metrics JSON），而不是提前退出后报了个假绿。

---

## 6. 分析口径（沿用 v1，不要放松）

1. **一切排序都对着 seed 噪声带说话。** 未超出噪声带的差异一律报"带内"，不作为结论。
2. **指标依赖必须披露。** v1 的 Stage A 增益在 MAE 上是 1.9× 带（成立），在 R² 上只有 0.86× 带（不成立）——
   两个都要报，不能只挑好看的。
3. **饱和任务不参与排名。** `formation_energy` 单任务天花板 R² = 0.995，
   在那个位置 R² 已无分辨率。v1 的做法是每任务自选指标（整格 R² 极差 < 0.005 时退回 MAE），沿用。
4. **读 per-step JSON，不读 `metrics_table.csv`。** `--resume` 过的 run 写出的表是**残缺的**
   （只含恢复进程的内存记录），而 `stepNN_*/<task>_metrics.json` 永远完整。
   参考 `experiments/rikyu_hparam_tuning/analysis/stage_c.py` 的 `final_metrics()`。

### 可直接复用的分析脚本

`experiments/rikyu_hparam_tuning/analysis/` 下：`rank.py`、`adopt.py`、`stage_c.py`、
`patience_ab.py`（其中 `patience_ab.py` 的 arm/band/comparison 结构最接近 v2 需要的口径）。

---

## 7. 预算汇总

| 阶段 | run 数 | 单 run | 备注 |
|---|---:|---:|---|
| Stage 0 锚定 | 2 × 3 seed = 6 | ~0.3h | probe3 |
| A'1 粗筛 | 48 | ~0.3h | 单 seed |
| A'1b 边界外扩 | ~12 | ~0.3h | **仅在最优落在边界时** |
| A'2 factor / early stop | 6 | ~0.3h | |
| A'3 seed 确认 | 20 | ~0.3h | 5 seed × 4 配置 |
| B' 回归 probe | 6 + 15 确认 | ~0.3h | |
| B' KR probe | 24 + 15 确认 | ~0.5h | KR 更慢 |
| Stage C' | 2 pretrain + 2 consolidation | 17–20h / 1.5h | 大头 |

probe 阶段合计约 **150 run × 0.3h ≈ 45 GPU-h**，并行后墙钟数小时。
Stage C' 是真正的大头，约 **40 GPU-h**，需要 48h 墙钟预算。

---

## 8. 起手第一步

```bash
# 1. 确认 Slurm 可用（注意 bash -lc）
ssh rikyu-login 'bash -lc "sinfo -s | head; squeue -u \$USER"'

# 2. 拉最新镜像（提交成作业，登录节点没有 apptainer）
#    先确认 pyproject 的 version，镜像 tag 是 rikyu-<version>

# 3. 建 v2 实验目录，从 v1 复制并修改 configs / scripts
#    v1 的 probe3.toml 是 Stage A'/B' 的起点

# 4. 跑 Stage 0 锚定，确立新 régime 下的参照系 —— 别跳过

# 5. smoke → A'1
```

**第一件该确认的事**：v1 的 Stage C 是否已跑完（`c_base` / `c_tuned` 的
`training/final_model.pt` 是否存在），以及 consolidation 是否已提交。
那批结果是 v2 的外部对照，需要先归档。
