# KMD 描述符计算与迁移指南

本文说明如何使用 `kmd_plus.py` 将材料组成转换为 Kernel Mean Descriptor（KMD），以及把该模块迁移到其他代码库时需要携带的文件、依赖和约定。

## 1. KMD 在做什么

设系统有 $C$ 个可选组分，每个组分有 $F$ 个特征。组分特征矩阵为

$$
X \in \mathbb{R}^{C \times F},
$$

一个样品的混合权重为

$$
w \in \mathbb{R}^{C}, \qquad w_c \ge 0, \qquad \sum_{c=1}^{C} w_c = 1.
$$

`KMD` 在初始化时根据 $X$ 构造并缓存核矩阵 $K$，之后通过一次矩阵乘法计算描述符：

$$
z = wK.
$$

批量输入时，权重矩阵 $W$ 的形状为 `(n_samples, n_components)`，输出为 `W @ K`。因此，同一个 `KMD` 实例应复用于所有样品，避免重复构造核矩阵。

## 2. 两种核构造方法

### 2.1 `method="1d"`：逐特征网格 KMD

这是本项目计算元素组成描述符时使用的方法。

对于每个组分特征，代码在该特征的最小值和最大值之间建立 `n_grids` 个等距网格点，再计算组分到各网格点的高斯核响应。所有特征的响应最后按列拼接。

- 核矩阵形状：`(n_components, n_features * n_grids)`
- 描述符形状：`(n_samples, n_features * n_grids)`
- `n_grids` 必须为不小于 2 的整数
- `scale=True` 时，每个特征先做 min-max 归一化
- `sigma="auto"` 时，每个特征使用自身网格间距确定核宽度

若元素特征表有 58 个特征，配置为：

```toml
[descriptor]
kind = "kmd"
n_grids = 8
```

则描述符维度为：

$$
58 \times 8 = 464.
$$

`n_grids` 越大，描述符越细致、维度和计算开销也越高。训练、预测和逆向设计必须使用相同的组分特征表、组分顺序、`n_grids`、`sigma` 和 `scale`。

### 2.2 `method="md"`：多维特征空间 KMD

该方法直接计算组分之间在完整特征空间中的高斯核。

- 核矩阵形状：`(n_components, n_components)`
- 描述符形状：`(n_samples, n_components)`
- `n_grids` 不参与计算
- `scale=True` 时，每个特征按样本标准差（`ddof=1`）标准化
- `sigma="auto"` 时，核尺度来自各组分最近邻平方距离的中位数

`md` 更紧凑，但输出维度随组分数量变化；`1d` 的输出维度由特征数和网格数决定。本项目的元素组成路径固定使用 `1d`。

## 3. 最小可运行示例

下面的示例不依赖本项目的训练框架，只需要 `kmd_plus.py` 及其数值计算依赖。

```python
import numpy as np

from kmd_plus import KMD

# 4 个组分，每个组分有 3 个固定特征。
component_features = np.array(
    [
        [0.2, 1.0, 3.0],
        [0.8, 1.5, 2.0],
        [1.4, 0.5, 1.0],
        [2.0, 2.0, 0.0],
    ],
    dtype=float,
)

# 两个样品；列顺序必须与 component_features 的行顺序完全一致。
weights = np.array(
    [
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.2, 0.3, 0.5],
    ],
    dtype=float,
)

kmd = KMD(
    component_features,
    method="1d",
    n_grids=8,
    sigma="auto",
    scale=True,
)

descriptors = kmd.transform(weights)
# 等价写法：descriptors = kmd(weights)

print(descriptors.shape)  # (2, 3 * 8) == (2, 24)
```

`transform` 本身不会检查权重是否非负或每行和是否为 1。调用方应在进入 KMD 前完成校验或归一化：

```python
if np.any(weights < 0):
    raise ValueError("KMD weights must be non-negative.")

row_sums = weights.sum(axis=1, keepdims=True)
if np.any(row_sums == 0):
    raise ValueError("Every sample must contain at least one component.")

weights = weights / row_sums
descriptors = kmd.transform(weights)
```

## 4. 从化学式计算元素 KMD

模块自带 `element_features.csv`，其行索引是从 H 到 Pu 的 94 个元素，列是 58 个元素级特征。`formula_to_composition` 会把化学式转换为与元素表行顺序一致的原子分数向量。

```python
import numpy as np

from kmd_plus import KMD, element_features, formula_to_composition

kmd = KMD(
    element_features.values,
    method="1d",
    n_grids=8,
    sigma="auto",
    scale=True,
)

formulas = ["SiO2", "Al2O3", "LiFePO4"]
weights = np.stack([formula_to_composition(formula) for formula in formulas])
descriptors = kmd.transform(weights)

print(weights.shape)      # (3, 94)
print(descriptors.shape)  # (3, 464)
```

`formula_to_composition` 也接受元素数量字典和 pymatgen `Composition`：

```python
from pymatgen.core import Composition

from kmd_plus import formula_to_composition

from_string = formula_to_composition("SiO2")
from_dict = formula_to_composition({"Si": 1.0, "O": 2.0})
from_object = formula_to_composition(Composition("SiO2"))
```

注意：传入自定义 `elements` 时，其顺序必须与 `component_features` 的行顺序一致，并且必须覆盖待处理化学式中的全部元素。函数不会为不在 `elements` 中的元素增加列，也不会对遗漏元素后的向量重新归一化。

### 4.1 本项目的实际生产调用链

在 `foundation_model` 中，`component_features` 不是从训练任务的数据集读取的，而是来自与 `kmd_plus.py` 同目录的内置文件：

```text
src/foundation_model/utils/element_features.csv
```

模块导入时以第一列作为元素索引读取该文件：

```python
_ELEMENT_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "element_features.csv")
element_features = pd.read_csv(_ELEMENT_FEATURES_PATH, index_col=0)
DEFAULT_ELEMENTS = list(element_features.index)
```

当前文件的形状为 `(94, 58)`：

- 94 行对应从 `H` 到 `Pu` 的元素；
- 58 列对应元素级物理和化学特征，例如 `atomic_number`、`atomic_radius`、`atomic_weight`、`boiling_point` 和 `bulk_modulus`；
- `element_features.index` 同时定义了 `formula_to_composition` 输出向量的默认元素顺序。

当配置指定 KMD 时：

```toml
[descriptor]
kind = "kmd"
n_grids = 8
```

`TaskCatalog` 的实际构造代码等价于：

```python
kmd = KMD(
    element_features.values,
    method="1d",
    n_grids=8,
    sigma="auto",
    scale=True,
)
```

这里传入的 `component_features` 就是 `element_features.values`，形状为 `(94, 58)`。转换为 NumPy 数组后虽然不再携带 DataFrame 的行标签，但行顺序保持不变。

每条任务数据只需要提供化学式，例如 `"SiO2"`。描述符计算过程为：

```python
formula = "SiO2"

# DEFAULT_ELEMENTS 来自 element_features.index，所以 weight 的第 j 列与
# element_features.values 的第 j 行表示同一个元素。
weight = formula_to_composition(formula)[None, :]
descriptor = kmd.transform(weight)

assert weight.shape == (1, 94)
assert descriptor.shape == (1, 464)
assert np.isclose(weight.sum(), 1.0)
```

矩阵关系如下：

```text
element_features.csv
    │  pd.read_csv(..., index_col=0)
    ▼
element_features.values                  shape: (94, 58)
    │  KMD(method="1d", n_grids=8)
    ▼
KMD kernel K                             shape: (94, 58 × 8) = (94, 464)

"SiO2"
    │  formula_to_composition()
    ▼
weight                                   shape: (1, 94)
    │  weight @ K
    ▼
descriptor                               shape: (1, 464)
```

因此，训练任务的数据文件负责提供 composition 字符串和预测目标，不负责提供 `component_features`。如果配置改为 `kind = "precomputed"`，`TaskCatalog` 不会创建 KMD，而是从配置指定的预计算描述符文件读取输入。

## 5. 可微分的 PyTorch 计算

梯度优化组成时，使用 `transform_torch`，不要先把 tensor 转成 NumPy。核矩阵被视为常量，梯度会通过矩阵乘法回传到权重。

```python
import torch

from kmd_plus import KMD, element_features

kmd = KMD(element_features.values, method="1d", n_grids=8)

logits = torch.zeros(
    1,
    len(element_features),
    dtype=torch.float32,
    requires_grad=True,
)
weights = torch.softmax(logits, dim=1)
descriptors = kmd.transform_torch(weights)

loss = descriptors.square().mean()
loss.backward()

assert logits.grad is not None
```

`transform_torch` 默认让核矩阵跟随输入权重的 `device` 和 `dtype`，并按 `(device, dtype)` 缓存只读副本，适合在优化循环中重复调用。

若外部代码需要读取核矩阵，可使用 `kernel_torch()`。每次调用都会返回独立 clone，调用方的原地修改不会影响 KMD 内部状态：

```python
kernel = kmd.kernel_torch(device="cpu", dtype=torch.float64)
```

PyTorch 只在调用这两个方法时才需要安装；纯 NumPy 的 `transform` 和 `inverse` 不导入 PyTorch。

## 6. 从描述符恢复混合权重

`inverse` 为每个样品求解带单纯形约束的二次规划：

$$
\underset{w}{\operatorname{minimize}}\; \frac{1}{2}w^T(KK^T)w - (Kz)^Tw,
\qquad w \ge 0,\quad \mathbf{1}^T w = 1.
$$

```python
descriptors = kmd.transform(weights)
recovered_weights = kmd.inverse(descriptors)

np.testing.assert_allclose(recovered_weights.sum(axis=1), 1.0)
np.testing.assert_allclose(recovered_weights, weights, atol=1e-4)
```

逆变换要求 $KK^T$ 正定。重复的组分特征、过于相似的核响应或过低的描述符维度都可能使核不可逆。代码会在这种情况下抛出 `ValueError`：

- `method="1d"`：优先尝试增加 `n_grids`
- `method="md"`：优先尝试减小显式 `sigma`
- 两种方法：检查是否存在重复组分行或常量特征列

`inverse` 固定使用 `qpsolvers` 的 `quadprog` 后端，因此即使只在少数路径中调用逆变换，也必须安装 `quadprog`。

## 7. `sigma` 和 `scale` 的含义

显式指定正数 `sigma` 时，两种方法都使用标准高斯形式：

$$
k(d)=\exp\left(-\frac{d^2}{2\sigma^2}\right).
$$

`sigma="auto"` 的规则则由方法决定：

- `md`：$\gamma$ 为最近邻平方距离中位数的倒数，核为 $\exp(-\gamma d^2)$
- `1d`：每个特征的 $\gamma$ 为网格间距平方的倒数

实践建议：

1. 首次使用优先保留 `sigma="auto"` 和 `scale=True`。
2. 训练和推理必须复用完全相同的参数。
3. `scale=True` 时，`md` 的每个特征必须有非零标准差，`1d` 的每个特征必须有非零极差。
4. 若特征中存在常量列，应在构造 KMD 前删除该列，或在确认数值尺度合理后使用 `scale=False`。

## 8. 迁移到其他代码库

### 8.1 必须复制的文件

```text
your_package/
├── kmd_plus.py
└── element_features.csv  # 使用内置元素描述符时必须与 .py 文件同目录
```

`kmd_plus.py` 通过 `os.path.dirname(__file__)` 查找 `element_features.csv`。如果新项目不需要内置元素特征，可以改为由调用方传入 `component_features`，并移除模块级 CSV 加载及相关默认元素常量。

复制代码时应保留源文件顶部的版权和 SPDX 许可证声明，并遵守仓库许可证要求。

### 8.2 Python 依赖

纯 NumPy 正向计算需要：

```text
numpy
pandas
scipy
pymatgen
```

逆变换额外需要：

```text
qpsolvers
quadprog
```

可微分计算额外需要：

```text
torch
```

完整安装示例：

```bash
python -m pip install numpy pandas scipy pymatgen qpsolvers quadprog torch
```

若目标平台不提供 `quadprog` wheel，而新项目只做正向计算，可删除 `inverse` 相关代码及 `qpsolvers` 导入；不要仅捕获导入错误后保留一个不可用的 `inverse` 接口。

### 8.3 迁移后最小验收

至少验证以下行为：

```python
import numpy as np

from your_package.kmd_plus import KMD, element_features, formula_to_composition

kmd = KMD(element_features.values, method="1d", n_grids=8)
weights = np.stack(
    [
        formula_to_composition("SiO2"),
        formula_to_composition("Al2O3"),
    ]
)
descriptors = kmd.transform(weights)

assert weights.shape == (2, len(element_features))
assert descriptors.shape == (2, element_features.shape[1] * 8)
assert np.isfinite(descriptors).all()
assert np.allclose(weights.sum(axis=1), 1.0)
```

如果新项目会使用逆变换，再加入 round-trip 检查；如果会做组成优化，再加入 `transform_torch(...).backward()` 检查。

## 9. 常见问题

### 描述符维度与模型输入维度不一致

检查 `component_features.shape[1]` 和 `n_grids`。对于 `1d`：

```text
descriptor_dim = n_features * n_grids
```

改变元素特征列数或 `n_grids` 后，旧模型的输入层通常不能直接复用。

### 不同机器上同一化学式得到不同结果

核矩阵不仅取决于化学式，还取决于完整的组分特征表及其行列顺序。确认两端使用同一份 CSV、相同的预处理参数和相同的元素顺序。

### 输出出现 NaN 或无穷值

最常见原因是 `scale=True` 时存在常量特征列，导致标准化或 min-max 归一化除以零。先检查：

```python
feature_range = np.ptp(component_features, axis=0)
constant_columns = np.flatnonzero(feature_range == 0)
```

### `inverse` 报告 KMD 不可逆

检查重复组分、增加 `1d` 的 `n_grids`，或调整 `md` 的 `sigma`。正向描述符仍可能可计算，但不可逆意味着无法唯一恢复原始组成。

### 自定义组分体系如何使用

不必使用元素。只要 `component_features` 的每一行代表一个组分，权重矩阵的每一列与该行对应，KMD 同样适用于聚合物单体、溶剂、合金端元或其他混合体系。

## 10. 补充：统计描述符

模块还提供 `stats_descriptor(weights, component_features)`，可按顺序拼接加权均值、加权方差、存在组分的最大值和最小值。它不是 KMD，也不使用预计算核矩阵，但可作为简单基线：

```python
from kmd_plus import stats_descriptor

summary = stats_descriptor(
    weights,
    component_features,
    stats=("mean", "var", "max", "min"),
)
```

默认输出维度为 `n_features * 4`。`max` 和 `min` 只统计权重非零的组分，并要求每个样品至少包含一个非零权重。
