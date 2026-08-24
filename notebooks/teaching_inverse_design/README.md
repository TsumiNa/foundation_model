# 逆向设计教学材料

两节课的自包含教材，围绕**准晶（quasicrystal）合金的逆向设计**展开，
对应研究产物 `artifacts/inverse_design_run/.../inverse_design_max_elements`。

## 怎么用

```bash
uv sync --frozen --all-groups        # 在仓库根目录执行一次
uv run jupyter lab                   # 打开 notebooks/teaching_inverse_design/
```

按顺序跑：

| Notebook | 内容 | 耗时 |
|---|---|---|
| `01_multitask_model.ipynb` | 数据 → KMD 描述符 → 多任务模型 → 训练 → 评估 | 约 1–2 分钟 |
| `02_inverse_design.ipynb` | KMD 成分路径优化器的原理、参数详解、三个设计场景、可视化分析 | 约 1 分钟 |

**必须先跑 01**：它把训练好的模型写进 `outputs/`，02 从那里读。

仓库里的 `.ipynb` **带着运行结果**，可以先当讲义读一遍，再自己动手跑。

## 目录

```
teaching_inverse_design/
├── README.md                              ← 本文件
├── data/
│   └── qc_inverse_design_teaching.parquet ← 唯一的数据文件（585 KB，29 802 个成分）
├── prepare_data.py                        ← 上面那个 parquet 是怎么来的（学生不需要跑）
├── 01_multitask_model.ipynb
├── 02_inverse_design.ipynb
└── outputs/                               ← 01 生成、02 消费，已 gitignore
    ├── multitask_model.ckpt               ← 训练好的模型
    ├── model_meta.json                    ← 结构参数 + test 集指标
    ├── resolved_split.parquet             ← 成分级 train/val/test 划分
    ├── scenario*__seed_to_optimized.parquet   ← 每个场景逐种子的设计结果
    └── scenario*__trajectory.npz              ← 每个场景的优化轨迹
```

## 数据

一个 parquet，一行 = 一个化学成分，`composition` 是主键。
四个性质列（`formation_energy` / `magnetization` / `tc` / `klat`）是 **z-score 标准化值**，
不是物理单位；`material_type` 是三分类标签（`0=AC`、`1=QC`、`2=others`）。
大量单元格是 NaN——不同性质来自不同数据库，覆盖的成分不重叠，模型按任务做掩码处理。

原始数据来自四个来源（QC/AC 材料库、NEMAD 超导、NEMAD 磁性、phonix-db 热导），
合并、规范化、标准化的全过程写在 `prepare_data.py` 里，有完整注释。
**教学刻意跳过数据整理**，所以正常上课不需要碰这个脚本。

## 这两节课教什么、不教什么

**教**：
- 正向模型（成分 → 性质）怎么搭、怎么评估，以及**评估结果如何决定逆向设计的可信度**
- KMD 描述符，尤其是 `x = w @ K` 这个**线性**关系为什么是逆向设计的前提
- KMD 成分路径优化器的完整原理：`logits → softmax → w → x → 模型 → loss` 全程可微
- 每个重要参数解决什么问题，配一次改一个参数的对照实验
- 三个设计场景的完整流程与结果分析

**不教**：
- 数据清洗与整理（已提前做完）
- 多任务学习本身（同时训 5 个任务只是因为三个场景需要这 5 个性质）
- latent 路径优化器（只教 KMD 成分路径，02 第 2.3 节解释了为什么）

## 图里为什么是英文

matplotlib 的默认字体不含中文字形，换字体在不同机器上不可靠。
所以**图的坐标轴和标题一律用英文，全部讲解在 markdown 里用中文**。
