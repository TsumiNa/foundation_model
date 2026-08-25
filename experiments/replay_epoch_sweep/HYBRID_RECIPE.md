# 最终模型训练方案：混合 Replay + 可选末端固化（下阶段工作基准）

日期：2026-08-12 ｜ 状态：**已验证（单 seed）**，可作为下阶段 continual pretraining 的默认配方
来源：`replay_epoch_sweep` 全 campaign（34 runs，REPORT_20260802/20260809）+ 混合验证跑
快速说明用 PPT：`results/RECIPE_20260812.pptx`（构建：`build_recipe_pptx.py`）

---

## 一、方案（拿来即用）

### 1. 主训练：混合 replay，m150 配方

在任意 24 任务 sweep 配置基础上，replay 与训练段配置如下
（完整参照 [configs/hybrid_full24.toml](configs/hybrid_full24.toml)，其余段落与 canonical 配置一致）：

```toml
[training]
max_epochs = 150            # 100 也接近饱和（差 ~0.005）；150 为已验证值

[training.early_stopping]
patience = 24               # 实质关闭早停——重采样下早停是隐性瓶颈

[pretrain.replay]
interval = 1
resample = "epoch"          # 每 epoch 重抽 replay 子集（整个方案的地基）
amount = 0.30               # 全局：每旧任务 30% 自身训练标签
# 下限规则：凡 0.3·N_task < 1500（即 N < 5000）的任务给固定 1500 条（引擎自动截断到 N，
# 小任务因此自动全量）。当前 24 任务集的展开：
per_task = { dielectric_total = 1500, dielectric_ionic = 1500, dielectric_electronic = 1500, magnetization = 1500, neel = 1500, kp = 1500, zt = 1500, power_factor = 1500, thermal_conductivity = 1500, klat = 1500, magnetic_moment = 1500, magnetic_susceptibility = 1500 }
```

**换任务集时的通用规则**：`amount_t = max(1500, 0.3 · N_t)`——全局 `amount = 0.3`，
对所有 `N_t < 5000` 的任务写 `per_task = 1500`。

```bash
fm pretrain --config <sweep 配置> --output-dir <out> --resume   # --resume 保证幂等重提
```

### 2. 可选末端固化（consolidation）

主训练结束后，从 `final_model.pt` 做一次全量联合重训（全部头 + encoder 解冻）：

```bash
fm finetune \
  --config experiments/replay_epoch_sweep/configs/joint_retrain_full24.toml \
  --checkpoint <out>/training/final_model.pt \
  --epochs 250 --output-dir <out>_joint
```

从健康模型出发 **76 epoch 即早停（~1.5h H200）**。何时做：大任务（≥20k 标签）性能优先时
（deficit 0.031→0.022，全场最优）；小任务敏感场景注意其小幅回吐（0.008→0.016）。
它是抛光不是补救——从无 replay 塌方模型出发只能到 0.584。

---

## 二、验证结果（两个交付）

| 臂 | mean R²（23 任务） | 大任务 deficit | 中 | 小 |
| --- | --- | --- | --- | --- |
| **结果 1：混合 replay** | 0.652 | 0.031 | 0.012 | **0.008** |
| **结果 2：+ 末端固化** | **0.658** | **0.022** | **0.005** | 0.016 |
| 纯 ratio 最优（r0.3） | 0.652 | 0.025 | 0.008 | 0.046 |
| 纯 fixed 最优（n2500） | 0.663 | 0.044 | −0.007 | 0.002 |
| 无 replay + 重训（对照） | 0.584 | 0.112 | 0.061 | 0.146 |

- deficit = single-task 天花板 − 终值（组均值）；分组：大 ≥20k（6 任务）/ 中 3k–8.1k（14）/
  小 ≤1.2k（2）。
- 结果 1 是 12 个 replay 设定中的 **minimax 冠军**（最差组 0.031）；结果 2 把 minimax 进一步
  压到 **0.022**。固化的逐任务变化：seebeck +0.066、neel +0.049、final_energy +0.047；
  dielectrics −0.03、magnetization −0.017。
- 2×2 闭合：replay（无→有）决定天花板（0.584 vs 0.65+）；末端重训在其上只加 +0.006。

## 三、每个选择的证据（一行一条）

| 选择 | 证据 |
| --- | --- |
| `resample = "epoch"` | 全部 n 提升 +0.022…+0.126（峰值 @n200）；零回退 |
| patience 24 | p8 下早停是隐性瓶颈；p24 使 n 依赖近乎抹平（n100-p24 0.592 ≈ n2500-step 0.600） |
| max_epochs 100–150 | epoch 预算 ~100 饱和（m150−p24 均值 +0.005、无 n 趋势） |
| ratio 0.3（大任务） | fixed-n 大任务 deficit 停 0.044+；r0.3 → 0.025（"多任务成本"一半是可恢复遗忘） |
| floor 1500（中小任务） | 纯 ratio 饿小任务（r0.5 → 0.085）；n1500 全覆盖时小任务 ≤0.008 |
| 固化可选 | 76ep 早停、均值 +0.006（噪声级）；大任务 0.031→0.022 是唯一实质收益 |

## 四、下阶段实操注意

1. **墙钟计划数**（H200，24 任务全程）：混合臂 ~68k 标签/步 → **21.6h**；纯 r0.3 21.9h、
   n2500 ~25h（含 resume）。kernel-regression 任务主导 replay 成本，晚期步 1–3h/步。
   固化 ~1.5h。A100 约 1.3–1.5×。
2. **TIMEOUT 恢复**：`fm pretrain --resume` 幂等，同命令重提即可（只丢在途步）。注意
   `--resume` 后 `metrics_table.csv` 只含续跑进程的步（上游已知问题）——用
   [analysis/rebuild_metrics_from_stepdirs.py](analysis/rebuild_metrics_from_stepdirs.py)
   从逐步 `*_metrics.json`（权威）重建。`fm finetune` **无 resume**，墙钟一次给足。
3. **代码前提**：`interval > 1` / 无 replay 用法需要 PR #36 修复（master `921ffca` 起已含；
   interval=1 的常规配方不受影响）。`resample="epoch"` 与 `persistent_workers=true` 互斥。
4. **噪声标定**：全部单 seed（2025）、固定任务序；±0.02 属噪声带。发表级数字需 ≥3 seeds。
   遗留对照：step-p24（切"训练时长"与"重采样覆盖"的最后一刀）。
5. **衔接 task-scaling 协议**：其 replay 分支（n1000/n1500，冻结子集假设定尺）应按本配方
   重估；material_type（分类）对 replay 设定不敏感（±0.005），可不作为调参目标。

## 五、文件与溯源

- 配置：[configs/hybrid_full24.toml](configs/hybrid_full24.toml)（主训练）、
  [configs/joint_retrain_full24.toml](configs/joint_retrain_full24.toml)（固化）
- 作业模板：[hybrid_h200.sbatch](hybrid_h200.sbatch)、[hybrid_joint_h200.sbatch](hybrid_joint_h200.sbatch)
- 结果（本地/rsync，不入 git）：`results/mt_hybrid_r03_f1500.csv`、
  `results/hybrid_joint_retrain.json`；raw `artifacts/replay_sweep_hybrid/`（R-CCS + 本地镜像）
- run_provenance：master `0672ac9`（训练时）；完整 campaign 报告
  `results/REPORT_20260802.md` / `results/REPORT_20260809.md`（+ 同名 pptx）
