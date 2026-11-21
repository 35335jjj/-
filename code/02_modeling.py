# -*- coding: utf-8 -*-
"""
02_modeling.py
城市租金预测项目 —— 特征工程、模型训练、正则化对比与结果分析

运行方式：
    在 code 目录下执行：python 02_modeling.py
    （若未运行 01_data_exploration.py，本脚本会直接基于原始数据完成清洗）

模型对比：
    1. 普通多元线性回归（OLS / LinearRegression）
    2. Ridge 岭回归（L2 正则化，网格搜索 + 5 折交叉验证）
    3. Lasso 套索回归（L1 正则化，网格搜索 + 5 折交叉验证）
    4. ElasticNet 弹性网络（额外探索：L1+L2 混合正则化）

输出：
    output/metrics.csv            各模型测试集指标与交叉验证结果
    output/coefficients.csv       各模型标准化后的特征权重
    output/predictions.csv        最优模型在测试集上的预测明细
    figures/fig5_model_compare.png      模型指标对比柱状图
    figures/fig6_coefficients.png       OLS/Ridge/Lasso 权重对比图
    figures/fig7_pred_vs_actual.png     预测值与真实值散点图
    figures/fig8_residuals.png          残差分布图
    figures/fig9_ridge_alpha_path.png   Ridge 正则化强度-交叉验证曲线
"""

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 原始字段分组
NUMERIC_RAW = ["bedrooms", "bathrooms", "size_sqft", "min_to_subway",
               "floor", "building_age_yrs"]
BINARY_COLS = ["no_fee", "has_roofdeck", "has_washer_dryer", "has_doorman",
               "has_elevator", "has_dishwasher", "has_patio", "has_gym"]
TARGET = "rent"


def load_data() -> pd.DataFrame:
    """读取清洗后的数据；若不存在则从原始数据现场清洗。"""
    cleaned_path = OUT_DIR / "cleaned_data.csv"
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
    else:
        raw = pd.read_csv(DATA_DIR / "manhattan.csv")
        raw = raw.drop_duplicates().dropna()
        raw = raw[raw["bathrooms"] >= 1]
        raw = raw[(raw["size_sqft"] >= 280) & (raw["size_sqft"] <= 3000)]
        raw = raw[(raw["rent"] >= 1500) & (raw["rent"] <= 18000)]
        df = raw.reset_index(drop=True)
    print(f"[建模数据] 样本数={len(df)}，字段数={df.shape[1]}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征构造（中级难度要求的衍生特征在本环节完成）：
    - size_sqft_sq：面积平方项，刻画租金随面积的非线性增长；
    - sqft_per_bed：卧室人均面积，衡量户型宽敞程度；
    - bath_per_bed：每卧室对应卫生间数量；
    - amenity_count：配套设施数量，汇总二值配套字段。
    """
    df = df.copy()
    df["size_sqft_sq"] = df["size_sqft"] ** 2
    df["sqft_per_bed"] = df["size_sqft"] / (df["bedrooms"] + 1)
    df["bath_per_bed"] = df["bathrooms"] / (df["bedrooms"] + 1)
    df["amenity_count"] = df[BINARY_COLS].sum(axis=1)
    return df


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """按字段类型构造预处理器：数值标准化、二值原样保留、片区独热编码。"""
    numeric_cols = ["bedrooms", "bathrooms", "size_sqft", "min_to_subway",
                     "floor", "building_age_yrs", "size_sqft_sq",
                     "sqft_per_bed", "bath_per_bed", "amenity_count"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", BINARY_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
             ["neighborhood"]),
        ],
        remainder="drop",
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算回归常用评估指标。"""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"R2": round(r2, 4), "RMSE": round(rmse, 2),
            "MAE": round(mae, 2), "MAPE(%)": round(mape, 2)}


def make_models() -> dict:
    """定义候选模型与待调超参数网格。"""
    ridge_grid = {"regressor__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    lasso_grid = {"regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}
    enet_grid = {"regressor__alpha": [1e-3, 1e-2, 1e-1, 1],
                 "regressor__l1_ratio": [0.5, 0.9]}
    return {
        "OLS": Pipeline([("prep", None), ("regressor", LinearRegression())]),
        "Ridge": Pipeline([("prep", None),
                          ("regressor", Ridge(random_state=RANDOM_STATE,
                                             max_iter=10000))]),
        "Lasso": Pipeline([("prep", None),
                          ("regressor", Lasso(random_state=RANDOM_STATE,
                                              max_iter=20000))]),
        "ElasticNet": Pipeline([("prep", None),
                                ("regressor", ElasticNet(random_state=RANDOM_STATE,
                                                         max_iter=20000))]),
    }


def main() -> None:
    df = load_data()
    df = feature_engineering(df)

    X = df.drop(columns=[TARGET, "rental_id", "borough"])
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"[数据划分] 训练集={len(X_train)}，测试集={len(X_test)}")

    preprocessor = build_preprocessor(X)

    # 交叉验证折数
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grids = {"Ridge": {"regressor__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
             "Lasso": {"regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]},
             "ElasticNet": {"regressor__alpha": [1e-3, 1e-2, 1e-1, 1],
                            "regressor__l1_ratio": [0.5, 0.9]}}

    results = {}
    fitted = {}
    coef_table = {}

    for name, pipe in make_models().items():
        # 为每个模型绑定独立的预处理步骤，避免数据泄漏
        pipe.steps[0] = ("prep", preprocessor)

        if name in grids:
            search = GridSearchCV(pipe, grids[name], cv=cv,
                                  scoring="r2", n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
            cv_r2 = search.best_score_
            print(f"[{name}] 最优超参数：{best_params}，"
                  f"交叉验证 R2={cv_r2:.4f}")
        else:
            model = pipe
            model.fit(X_train, y_train)
            cv_r2 = None
            best_params = {}
            print(f"[{name}] 普通最小二乘拟合完成")

        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        if cv_r2 is not None:
            metrics["CV_R2"] = round(cv_r2, 4)
        results[name] = metrics
        fitted[name] = model

        # 提取标准化后的回归权重，便于跨模型比较
        reg = model.named_steps["regressor"]
        feature_names = model.named_steps["prep"].get_feature_names_out()
        coef_table[name] = pd.Series(reg.coef_, index=feature_names)

    # 指标汇总表
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "model"
    metrics_df.to_csv(OUT_DIR / "metrics.csv", encoding="utf-8-sig")
    print("\n[测试集指标汇总]")
    print(metrics_df.to_string())

    # 权重对照表
    coef_df = pd.DataFrame(coef_table)
    coef_df.to_csv(OUT_DIR / "coefficients.csv", encoding="utf-8-sig")

    # 选择测试集 R2 最高的模型作为最优模型，输出预测明细
    best_name = metrics_df["R2"].idxmax()
    best_model = fitted[best_name]
    y_best_pred = best_model.predict(X_test)
    pred_df = pd.DataFrame({
        "rent_true": y_test,
        "rent_pred": np.round(y_best_pred, 2),
        "abs_error": np.round(np.abs(y_test - y_best_pred), 2),
    })
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\n[最优模型] {best_name}")

    # 图5：各模型测试集指标对比
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, color in zip(axes, ["R2", "RMSE", "MAE"],
                                ["#4C78A8", "#E45756", "#54A24B"]):
        metrics_df[metric].plot(kind="bar", ax=ax, color=color)
        ax.set_title(f"Model Comparison - {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_model_compare.png", bbox_inches="tight")
    plt.close(fig)

    # 图6：按 OLS 权重绝对值取前 12 个特征，对比三种模型权重
    top_features = coef_df["OLS"].abs().sort_values(ascending=False).head(12).index
    plot_coef = coef_df.loc[top_features]
    plot_coef.index = [s.replace("num__", "").replace("cat__neighborhood_", "nb_")
                       for s in plot_coef.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_coef.plot(kind="bar", ax=ax)
    ax.set_title("Standardized Coefficients: OLS vs Ridge vs Lasso vs ElasticNet")
    ax.set_ylabel("Coefficient (standardized)")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_coefficients.png", bbox_inches="tight")
    plt.close(fig)

    # 图7：最优模型预测值与真实值散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_best_pred, s=14, alpha=0.6, color="#4C78A8")
    lims = [min(y_test.min(), y_best_pred.min()),
            max(y_test.max(), y_best_pred.max())]
    ax.plot(lims, lims, color="#E45756", linewidth=2, label="y = x")
    ax.set_title(f"Predicted vs Actual Rent ({best_name})")
    ax.set_xlabel("Actual Rent")
    ax.set_ylabel("Predicted Rent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_pred_vs_actual.png", bbox_inches="tight")
    plt.close(fig)

    # 图8：残差分布与残差-预测值图
    residuals = y_test - y_best_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_best_pred, residuals, s=12, alpha=0.55, color="#F58518")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Residuals vs Predicted")
    axes[0].set_xlabel("Predicted Rent")
    axes[0].set_ylabel("Residual")
    axes[1].hist(residuals, bins=35, color="#4C78A8", edgecolor="white")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_residuals.png", bbox_inches="tight")
    plt.close(fig)

    # 图9：Ridge 正则化强度 alpha 与交叉验证 R2 的关系
    ridge_search = GridSearchCV(
        Pipeline([("prep", preprocessor),
                  ("regressor", Ridge(random_state=RANDOM_STATE, max_iter=10000))]),
        grids["Ridge"], cv=cv, scoring="r2", n_jobs=-1)
    ridge_search.fit(X_train, y_train)
    alphas = [params["regressor__alpha"] for params in ridge_search.cv_results_["params"]]
    mean_scores = ridge_search.cv_results_["mean_test_score"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, mean_scores, marker="o", color="#4C78A8")
    ax.set_xscale("log")
    ax.set_title("Ridge CV Score vs Regularization Strength (alpha)")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("Cross-Validation R2")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_ridge_alpha_path.png", bbox_inches="tight")
    plt.close(fig)

    print("\n02_modeling.py 运行完成，结果已保存至 output 与 figures 目录。")


if __name__ == "__main__":
    main()
