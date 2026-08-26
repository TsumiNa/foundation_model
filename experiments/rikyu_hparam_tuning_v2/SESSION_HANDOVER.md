# v1 session 交接文档

> 写给执行 v2 并做最终收尾的 session。**动手前通读一遍。**
>
> 这份文档记录 v1 session 建立的全部事实、每条结论的**置信度**、哪些已经失效，
> 以及 —— 最重要的 —— **它犯过并修正过的错误**。
> 其中几条错误判断至今仍留在 git 历史的 commit message 里（见 §4），
> 只读 git log 会被误导。

---

## 1. 一句话背景

v1 是一轮完整的调参 campaign（PR #41，分支 `exp/rikyu-hparam-tuning`）：
Stage A 调 encoder/shared block，Stage B 调各任务 head，Stage C 用调好的参数跑 24 任务最终模型。
过程中顺带发现并修复了训练代码的若干问题，其中一个（学习率调度器节奏）**显著抬高了 baseline**，
这正是需要 v2 重跑的原因。

---

## 2. v1 的结论与置信度

### 2.1 高置信（多 seed 确认，超出噪声带）

| 结论 | 证据 |
|---|---|
| **调参本身有收益**：采纳配置 vs 未调 = **+15.8%**（约 1.9× 噪声带） | Stage A4，3 seed |
| **`encoder_lr` 是主导旋钮**，单独改 LR 就拿到约 2/3 的收益 | Stage A1，80 点 |
| **每任务单独调 head 不迁移到多任务**：24 个只有 2 个扛过 seed 重复，5 个反而退化 | Stage B4，3 seed |
| **多任务联合调 head 有效**（B-mt 对照组） | Stage B 对照 |
| **调度器 patience 改逐 epoch 是真实提升**：**+0.0263**（1.99× 带），三 seed 分布完全不重叠 | patience A/B，9 run |

### 2.2 中等置信（单 seed 或范围有限）

| 结论 | 保留意见 |
|---|---|
| `batch_size = 256`（512/1024 更差） | Stage A2，单 seed，仅在一个架构上测 |
| `n_grids = 8`（16 无改善、4 显著更差） | Stage A3，单 seed |
| `ae_lr = 5e-3`（1e-3/1e-2 都在带内） | Stage A6，单 seed |
| weight decay 不重要（0.40× 带） | patience A/B，只测了一个替代值 |

### 2.3 **已失效** —— v2 必须重测

**全部 Stage A/B 的调参结论，都是在学习率调度器坏掉的环境下测出来的。**

修复前（PR #45 之前），`ReduceLROnPlateau` 在 `training_step` 里被调用，即**每 batch 步进一次**。
`patience = 5` 数的是 batch，在 ~90 batch/epoch 的规模下 LR 在**第 1 个 epoch 内**
就从 5e-3 砍到 `min_lr = 1e-4` 地板，整个训练趴在地板上。

后果：

| v1 结论 | 为什么失效 |
|---|---|
| `encoder_lr` 最优 = 1e-3，且经 A1b 验证是**内点** | 那个 régime 下"调 encoder_lr"实际调的是**塌陷前那一小段窗口的初始 LR** |
| `encoder_lr = 0.01` 会发散（最差格 −45.6%） | 发散很可能正是"没有有效退火"的后果，不是 LR 本身太大 |
| Stage B 的全部 head 结论 | 同样在坏 régime 下测 |

**`min_lr = 1e-4` 在旧 régime 下其实是事实上的训练学习率**，不是地板。
这也是为什么 v2 计划把 `min_lr` 列为首要新增维度。

### 2.4 必须与结论一同报告的口径问题

1. **指标依赖**：Stage A 的增益在 **MAE 上是 1.9× 带（成立）**，在 **R² 上只有 0.86× 带（不成立）**。
   两个都要报。只报 MAE 会夸大结论。
2. **seed 噪声带 = 8.48%**，它**推翻过 A1 的单 seed 榜首**（+23.9% → 3 seed 只有 +18.45%）。
   v1 前 3 名彼此只差 1.5–1.8%，3 seed 分辨不了（需要 ~25 seed），所以 v1 只能报"三者并列"。
3. **饱和任务不参与排名**：`formation_energy` 单任务天花板 R² = 0.995，在那里 R² 已无分辨率。
   v1 的做法是每任务自选指标（整格 R² 极差 < 0.005 时退回 MAE）。

---

## 3. v1 产物在哪

| 内容 | 位置 |
|---|---|
| 报告（Markdown） | `experiments/rikyu_hparam_tuning/results/REPORT_20260826.md` |
| PPT | `experiments/rikyu_hparam_tuning/results/REPORT_20260826.pptx` |
| Stage A/B 汇总 CSV/JSON | `experiments/rikyu_hparam_tuning/results/` |
| patience A/B 结果 | `experiments/rikyu_hparam_tuning/results/patience_ab.json` |
| 图 | `experiments/rikyu_hparam_tuning/analysis/*.png` |
| RIKYU 原始输出 | `/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning/` |
| 分析脚本 | `experiments/rikyu_hparam_tuning/analysis/*.py` |

⚠️ **`results/`、`*.png`、`*.csv` 都在 `.gitignore` 里** —— 它们**存在于本地但不在 git 里**。
换机器时靠 `rsync` 传，不要指望 `git clone` 能拿到。**只有 `.py` / `.toml` / `.md` 进了 git。**

可直接复用：`analysis/stage_c.py`（读 per-step JSON 的正确姿势）、
`analysis/patience_ab.py`（arm/per_seed/band/comparison 的 JSON 结构，最接近 v2 需要的形状）、
`build_report_pptx.py`（数据驱动的 deck 生成器）。

---

## 4. ⚠️ 本 session 犯过并修正过的错误（**git 历史里仍有错误陈述**）

### 4.1 「#42 改了 weight decay 默认值」—— **错的**

我曾断言 #42 把全局 `weight_decay = 1e-3` 换成了 per-group 默认值，
导致 encoder 衰减涨 10 倍、head 降 100 倍。**这是错的。**

实际情况（逐字段核对过 #42 之前的代码）：

| group | 0.2.1 实际值 | 来源 | 0.3.2 默认值 |
|---|---|---|---|
| encoder | 1e-2 | `_engine.py:80` **硬编码** | 1e-2 ✅ 相同 |
| head | 1e-5 | `_HEAD_WEIGHT_DECAY` 常量 | 1e-5 ✅ 相同 |
| kr | 5e-5 | `training.kr_weight_decay` | 5e-5 ✅ 相同 |
| ae | 1e-3 | `OptimizerConfig` 默认 | 1e-3 ✅ 相同 |

**四个值完全一致。#42 只是把硬编码值提取成可配置字段，一个都没改。**

我的错误来源：拿 #42 新增的字段去比 `OptimizerConfig.weight_decay` 的全局默认值 1e-3，
**假设那些 group 之前取的是这个默认值** —— 它们从来没取过，值是在调用点硬编码的。

**教训（已写进 PLAN §5 教训 3）**：跨版本比默认值时，必须核对**调用点**，不能只看字段定义。

**影响**：patience A/B 的 arm 标签一度是错的。`pab_new` 我以为是"恢复 0.2.1"，
实际是**引入了一个偏离**；真正保持 weight decay 一致的是 `pab_asis`。
已在 commit `8e391ba` 修正，但**更早的 commit（`2f53702`、`cm5` 那条）里仍写着错误说法**。
**以 `patience_ab.json` 和 `analysis/patience_ab.py` 的当前内容为准。**

修正后的正确结论：

| 对比 | Δ mean R² | vs band | 结论 |
|---|---:|---:|---|
| **patience 净效应**（old→asis，weight decay 一致） | **+0.0263** | **1.99×** | 超出噪声 |
| weight decay 敏感度（asis→new） | −0.0052 | 0.40× | 噪声内 |
| 两者一起（old→new） | +0.0211 | 2.78× | 超出噪声 |

### 4.2 「Slurm 挂了 / 站点 DNS 故障」—— **错的**

我曾判断 RIKYU 的 Slurm 控制器 SRV 记录从 DNS 消失、属站点故障，甚至建议联系 R-CCS 支持。
**实际是我用了非登录 shell。**

```bash
ssh rikyu-login 'squeue'                # ❌ DNS SRV lookup failed
ssh rikyu-login 'bash -lc "squeue"'     # ✅ slurm 25.11.5
```

同一个 `/usr/bin/squeue`，区别只在 shell。登录 profile 才设置了让 Slurm 找到控制器的配置。
**所有远程 Slurm 命令必须包 `bash -lc`。**（已写进 PLAN §5 教训 1）

### 4.3 「weight_decay 配置字段是死配置」—— **错的**

我曾报告 3 个 weight-decay 字段在引擎里没接线。那是因为我 grep 到了 exp 分支的旧代码
（仓库根目录当时停在 `exp/rikyu-hparam-tuning`，而 master 在另一个 worktree 里）。
master 上 encoder/ae 都走 `training.optimizer_config(...)`，**接线是好的**。
`build_model_for_checkpoint` 里那处硬编码是 predict/inverse 的占位符，不参与优化。

---

## 5. 本 session 顺带完成的代码改动（已合并入 master）

| PR | 内容 | 对训练的影响 |
|---|---|---|
| #42 | 暴露 optimizer/scheduler 全部可配置参数，收窄到 AdamW + ReduceLROnPlateau | 无（值未变，见 §4.1） |
| #45 | **调度器改为每 epoch 步进** + 重写三个 `*_step` | **显著**（+0.0263） |
| #46–#50 | 启用 mypy `check_untyped_defs`，清空 backlog，加 per-commit hook | 设计上无 |
| #51 | descriptor 按请求键返回（修静默数据丢失） | 无（实验走 kmd 路径） |

当前 master 版本 **0.3.2**，RIKYU 镜像 `foundation-model_rikyu-0.3.2.sif` 已拉好。

**#51 对实验无影响的依据**（已核实）：实验配置用 `kind = "kmd"`，走 `_kmd_descriptor`，
那是三个 descriptor source 里**本来就遵守约定**的一个；日志实测 49,012 个组分只掉了 1 个 `Cm1`
（锔，KMD 94 元素基不含它，属化学范围限制而非 bug）。

---

## 6. 未决事项

> 本节在 v1 完全结束后由 v1 session 更新为最终状态。若你读到时仍是"进行中"，
> 说明 v1 session 未能收尾，请先按 §3 的路径确认 Stage C 的实际完成情况。

- [ ] Stage C `c_base` 预训练完成 + 提交 `ccon_base`
- [ ] Stage C `c_tuned` consolidation（job 52238）完成
- [ ] Stage C 分析（`analysis/stage_c.py`）+ 报告 Stage C 段落
- [ ] 最终 deck 重建（Stage C 页会自动补上）
- [ ] PR #41 就绪

---

## 7. 给 v2 的重点提醒

1. **不要跳过 Stage 0 锚定。** 没有"未调 + 新镜像"这个点，v2 的收益无法与 #45 的收益分离。
2. **不要跑单 seed 网格。** v1 的 A1 就是这么栽的（赢家诅咒）。
3. **网格边界要检查。** v1 的 A1 最优落在端点，靠 A1b 外扩才确认是内点 —— 这一步做对了，保留。
4. **replay 配方当常量。** 它是上一轮 campaign 专门调出来的；同时改配方和超参会让两者都无法归因。
5. **smoke 要核实真的训练了**（有 step 目录和 metrics JSON），不要被"提前退出 + 假绿"骗过。
