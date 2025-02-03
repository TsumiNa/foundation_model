import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def format_title(text, to_snake=False, skip_list=None):
    """
    将输入文本格式化为首字母大写格式或转换为蛇形命名法，并支持指定不需要格式化的单词
    例如: 'aaa_xxxx' -> 'Aaa xxxx' 或 'AaaXxxx' -> 'aaa_xxxx'

    Parameters
    ----------
    text : str
        输入文本
    to_snake : bool, optional
        是否转换为蛇形命名法，默认为 False
    skip_list : list of str, optional
        指定部分单词不做格式化，默认为 None

    Returns
    -------
    str
        格式化后的文本
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    if text is None or text.strip() == "":
        return text

    if skip_list is None:
        skip_list = []

    if to_snake:
        # 将空格拆分为单词，并转换为蛇形命名法，保留skip_list中的单词不变
        words = text.split()
        formatted = [word if word in skip_list else word.lower() for word in words]
        return "_".join(formatted)
    else:
        # 将下划线替换为空格，并分割成单词
        words = text.replace("_", " ").split()
        if not words:
            return text
        # 格式化：第一个单词首字母大写，其他单词全部小写（除非在skip_list中）
        formatted = []
        for i, word in enumerate(words):
            if word in skip_list:
                formatted.append(word)
            else:
                formatted.append(word.capitalize() if i == 0 else word.lower())
        return " ".join(formatted)


def plot_scatter_comparison(
    y_true1,
    y_pred1,
    y_true2=None,
    y_pred2=None,
    labels=None,
    title=None,
    figsize=None,
    return_stat=False,
    dpi=150,
):
    """
    绘制一组或两组预测值与真实值的散点图对比

    Parameters
    ----------
    y_true1, y_pred1 : array-like
        第一组真实值和预测值
    y_true2, y_pred2 : array-like, optional
        第二组真实值和预测值（可选）
    labels : tuple of str, optional
        数据标签，如果为None则不显示标题
    title : str, optional
        图表标题
    figsize : tuple, optional
        图表大小，如果为None则根据子图数量自动设置
    dpi : int, optional
        图像分辨率
    """

    # 设置全局字体大小和样式
    plt.rcParams.update({"font.size": 16})  # 增加默认字体大小
    sns.set_style("whitegrid", {"grid.linestyle": "--"})  # 设置网格为虚线

    # 确定是否有两组数据
    has_two_datasets = y_true2 is not None and y_pred2 is not None

    # 根据数据组数设置图表大小
    if figsize is None:
        figsize = (12, 6) if has_two_datasets else (8, 6)

    # 准备数据
    df1 = pd.DataFrame({"Observation": y_true1, "Prediction": y_pred1})
    if has_two_datasets:
        df2 = pd.DataFrame({"Observation": y_true2, "Prediction": y_pred2})

    # 计算统计指标
    def calculate_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r2, mae, rmse

    # 计算数据范围
    if has_two_datasets:
        min_val = min(min(y_true1), min(y_pred1), min(y_true2), min(y_pred2))
        max_val = max(max(y_true1), max(y_pred1), max(y_true2), max(y_pred2))
    else:
        min_val = min(min(y_true1), min(y_pred1))
        max_val = max(max(y_true1), max(y_pred1))

    # 创建图形
    fig = plt.figure(figsize=figsize, dpi=dpi)

    if has_two_datasets:
        # 创建子图
        gs = plt.GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        axes = [ax1, ax2]
        dfs = [df1, df2]
    else:
        ax1 = fig.add_subplot(111)
        axes = [ax1]
        dfs = [df1]

    # 绘制散点图
    for i, (ax, df) in enumerate(zip(axes, dfs)):
        # 使用seaborn绘制散点图和拟合线
        sns.regplot(
            data=df,
            x="Observation",
            y="Prediction",
            ax=ax,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "blue", "linewidth": 1},
            fit_reg=True,
        )

        # 添加对角线
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=1)

        # 设置坐标轴范围和方形图形
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")

        # 显示四个边框并设置为黑色
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")

        # 设置刻度线宽度、长度和方向
        ax.tick_params(width=1, length=6, colors="black", direction="out")

        # 设置网格样式
        ax.grid(True, linestyle="--", alpha=0.6)

        # 计算并添加统计信息
        r2, mae, rmse = calculate_metrics(df["Observation"], df["Prediction"])
        stats_text = f"R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}"
        ax.text(
            0.95,
            0.05,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            fontsize=16,
        )

        # 设置标签
        ax.set_xlabel("Observation", fontsize=18)
        if i == 0:  # 只在第一个子图添加y轴标签
            ax.set_ylabel("Prediction", fontsize=18)

        # 设置子图标题（双图模式或单图模式下的labels）
        if labels and (has_two_datasets or (not has_two_datasets and title)):
            ax.set_title(labels[i], fontsize=20, pad=10)

    # 调整布局
    # plt.tight_layout()

    # 设置标题
    if title:
        formatted_title = format_title(title)
        if has_two_datasets or (not has_two_datasets and labels):
            # 双图模式或单图模式下同时有labels时：使用suptitle
            fig.suptitle(formatted_title, fontsize=22, y=1.05)
        else:
            # 单图模式下只有title时：直接设置为axes的标题
            ax1.set_title(formatted_title, fontsize=22, pad=10)
    if return_stat:
        return fig, tuple(axes), dict(r2=r2, mae=mae, rmse=rmse)

    return fig, tuple(axes)


# 使用示例
# if __name__ == "__main__":
#     # 生成示例数据
#     np.random.seed(42)

#     # 生成真实值
#     x1 = np.linspace(0, 10, 100)
#     x2 = np.linspace(0, 10, 100)

#     # 添加一些噪声生成预测值
#     y1 = x1 + np.random.normal(0, 0.5, 100)
#     y2 = x2 + np.random.normal(0, 1.0, 100)

#     # 示例1：单图模式（只有title）
#     print("示例1：单图模式（只有title）")
#     fig, axes = plot_scatter_comparison(x1, y1, title="electrical_resistivity")
#     plt.show()

#     # 示例2：单图模式（只有labels）
#     print("\n示例2：单图模式（只有labels）")
#     fig, axes = plot_scatter_comparison(x1, y1, labels=("Single Dataset",))
#     plt.show()

#     # 示例3：单图模式（同时有title和labels）
#     print("\n示例3：单图模式（同时有title和labels）")
#     fig, axes = plot_scatter_comparison(
#         x1, y1, labels=("Single Dataset",), title="electrical_resistivity"
#     )
#     plt.show()

#     # 示例4：双图模式
#     print("\n示例4：双图模式")
#     fig, axes = plot_scatter_comparison(
#         x1, y1, x2, y2, labels=("Dataset 1", "Dataset 2"), title="seebeck_coefficient"
#     )
#     plt.show()
