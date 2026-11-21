# -*- coding: utf-8 -*-
"""
01_data_exploration.py
城市租金预测项目 —— 数据读取、清洗与探索性分析（EDA）

运行方式：
    在 code 目录下执行：python 01_data_exploration.py
    或在项目根目录执行：python code/01_data_exploration.py

输入：data/manhattan.csv（StreetEasy 曼哈顿公开租房挂牌数据）
输出：
    output/cleaned_data.csv        清洗后的数据集
    figures/fig1_rent_distribution.png   租金分布直方图与箱线图
    figures/fig2_rent_vs_size.png         租金与面积散点图
    figures/fig3_correlation_heatmap.png  数值特征相关性热力图
    figures/fig4_neighborhood_rent.png    各片区平均租金对比图
"""

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")  # 无界面环境下后端输出图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# 项目根目录（code 的上一级），保证脚本在任意工作目录下运行都能定位文件
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 统一绘图风格，避免中文字体缺失导致乱码，图中文字采用英文标注
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

RANDOM_STATE = 42


def load_raw_data() -> pd.DataFrame:
    """读取原始公开数据集。"""
    df = pd.read_csv(DATA_DIR / "manhattan.csv")
    print(f"[原始数据] 行数={len(df)}，列数={df.shape[1]}")
    print("字段列表：", list(df.columns))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：
    1. 去除卫生间数量为 0 的异常挂牌（字段缺失或录入错误）；
    2. 过滤面积、租金明显不合理的极端记录；
    3. 去除重复挂牌与缺失值。
    """
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.dropna()

    # 卫生间为 0 视为字段缺失
    df = df[df["bathrooms"] >= 1]
    # 面积与租金的合理区间（依据数据分位数与常识设定）
    df = df[(df["size_sqft"] >= 280) & (df["size_sqft"] <= 3000)]
    df = df[(df["rent"] >= 1500) & (df["rent"] <= 18000)]

    df = df.reset_index(drop=True)
    print(f"[清洗数据] 清洗前行数={n0}，清洗后行数={len(df)}，剔除 {n0 - len(df)} 条")
    return df


def eda(df: pd.DataFrame) -> None:
    """绘制探索性分析图表并打印统计量。"""
    # 数值字段统计摘要
    numeric_cols = ["rent", "bedrooms", "bathrooms", "size_sqft",
                     "min_to_subway", "floor", "building_age_yrs"]
    print("\n[数值特征描述统计]")
    print(df[numeric_cols].describe().round(2).to_string())

    # 图1：租金分布（直方图 + 箱线图）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["rent"], bins=40, color="#4C78A8", edgecolor="white")
    axes[0].set_title("Monthly Rent Distribution")
    axes[0].set_xlabel("Rent (USD/month)")
    axes[0].set_ylabel("Count")
    axes[1].boxplot(df["rent"], vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#4C78A8"))
    axes[1].set_yticks([])
    axes[1].set_title("Rent Boxplot")
    axes[1].set_xlabel("Rent (USD/month)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_rent_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 图2：租金与面积散点图，叠加一元线性拟合趋势线
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["size_sqft"], df["rent"], s=12, alpha=0.55, color="#4C78A8")
    coef = np.polyfit(df["size_sqft"], df["rent"], 1)
    x_line = np.linspace(df["size_sqft"].min(), df["size_sqft"].max(), 100)
    ax.plot(x_line, np.polyval(coef, x_line), color="#E45756",
            linewidth=2, label=f"Fit: rent={coef[0]:.2f}*size+{coef[1]:.1f}")
    ax.set_title("Rent vs. Apartment Size")
    ax.set_xlabel("Size (sqft)")
    ax.set_ylabel("Rent (USD/month)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_rent_vs_size.png", bbox_inches="tight")
    plt.close(fig)

    # 图3：数值特征相关性热力图
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # 图4：样本量前 15 的片区平均租金
    top_neighborhoods = df["neighborhood"].value_counts().head(15).index
    neighborhood_rent = (df[df["neighborhood"].isin(top_neighborhoods)]
                         .groupby("neighborhood")["rent"].mean()
                         .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    neighborhood_rent.plot(kind="bar", color="#54A24B", ax=ax)
    ax.set_title("Average Rent by Neighborhood (Top 15)")
    ax.set_ylabel("Average Rent (USD/month)")
    ax.set_xlabel("")
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_neighborhood_rent.png", bbox_inches="tight")
    plt.close(fig)

    print("\n[EDA 图表已保存至 figures 目录]")


def main() -> None:
    raw = load_raw_data()
    cleaned = clean_data(raw)
    cleaned.to_csv(OUT_DIR / "cleaned_data.csv", index=False, encoding="utf-8")
    eda(cleaned)
    print("\n01_data_exploration.py 运行完成。")


if __name__ == "__main__":
    main()
