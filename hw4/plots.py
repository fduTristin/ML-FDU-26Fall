# -*- coding: utf-8 -*-
"""绘图脚本：根据 logs/*.json 生成实验结果图（PDF）。

    python plots.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

C_RIDGE = "#4C72B0"
C_LASSO = "#55A868"


def load(tag):
    with open(os.path.join(LOG_DIR, f"{tag}.json")) as f:
        return json.load(f)


def plot_reg_paths():
    """2x2：Ridge/Lasso 的验证 RMSE-alpha 曲线与系数路径。"""
    ols = load("boston_ols")
    ridge = load("boston_ridge")
    lasso = load("boston_lasso")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    cmap = plt.get_cmap("tab20")

    for col, (res, name, color) in enumerate(
            [(ridge, "Ridge", C_RIDGE), (lasso, "Lasso", C_LASSO)]):
        grid = sorted(res["grid"], key=lambda d: d["alpha"])
        alphas = np.array([g["alpha"] for g in grid])
        val = np.array([g["val_rmse"] for g in grid])
        tr = np.array([g["train_rmse"] for g in grid])
        best = res["best"]["alpha"]

        # 上排：train/val RMSE vs alpha
        ax = axes[0, col]
        ax.semilogx(alphas, tr, "o--", ms=3.5, lw=1.1, color=color,
                    alpha=0.55, label="train")
        ax.semilogx(alphas, val, "o-", ms=3.5, lw=1.3, color=color,
                    label="validation")
        ax.axhline(ols["test_metrics"]["val_rmse"], color="gray",
                   ls=":", lw=1.1, label="OLS validation")
        ax.axvline(best, color="red", ls="--", lw=1.1)
        ax.text(best * 1.25, ax.get_ylim()[0], f"best $\\alpha$={best:g}",
                color="red", fontsize=8.5, va="bottom")
        ax.set_xlabel(r"regularization strength $\alpha$")
        ax.set_ylabel("RMSE (k$)")
        ax.set_title(f"({chr(97 + col * 2)}) {name}: RMSE vs. $\\alpha$")
        ax.legend(fontsize=8)

        # 下排：系数路径（系数路径本身按 alpha 升序存储过，重新用 grid 对齐）
        ax = axes[1, col]
        coef_map = {g["alpha"]: c for g, c in
                    zip(res["grid"], res["coef_path"])}
        path = np.array([coef_map[a] for a in alphas])  # d x len? no: len x d
        for j in range(path.shape[1]):
            ax.semilogx(alphas, path[:, j], lw=1.2, color=cmap(j % 20))
        ax.axvline(best, color="red", ls="--", lw=1.1)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(r"regularization strength $\alpha$")
        ax.set_ylabel("coefficient value")
        ax.set_title(f"({chr(98 + col * 2)}) {name}: coefficient paths")

        if col == 1:  # 在 Lasso 路径图上标注几个重要特征
            names = res["feature_names"]
            # 取最优模型系数绝对值最大的 4 个特征做标注
            order = np.argsort(-np.abs(np.array(
                [res["coef_best"][nm] for nm in names])))[:4]
            # 按左端点纵坐标排序后交错放置标注，避免重叠
            offs = [(4, 6), (4, -13), (30, 12), (30, -19)]
            order = sorted(order, key=lambda j: -path[0, j])
            for rank, j in enumerate(order):
                ax.annotate(names[j], (alphas[0], path[0, j]),
                            textcoords="offset points",
                            xytext=offs[rank % len(offs)], fontsize=8,
                            color=cmap(j % 20))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boston_reg_paths.pdf"))
    plt.close(fig)


def plot_comparison():
    """1x2：各方法测试集指标条形图 + 最优模型预测散点。"""
    ols = load("boston_ols")
    ridge = load("boston_ridge")
    lasso = load("boston_lasso")
    poly = load("boston_poly")
    cv = load("boston_cv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), width_ratios=[1.15, 1])

    # (a) 测试 RMSE 条形图
    ax = axes[0]
    entries = [
        ("OLS", ols["test_metrics"]["rmse"], ols["test_metrics"]["r2"], "#8c8c8c"),
        ("Ridge", ridge["test_metrics"]["rmse"], ridge["test_metrics"]["r2"], C_RIDGE),
        ("Lasso", lasso["test_metrics"]["rmse"], lasso["test_metrics"]["r2"], C_LASSO),
        ("OLS + poly2", poly["ols"]["test_metrics"]["rmse"],
         poly["ols"]["test_metrics"]["r2"], "#8c8c8c"),
        ("Ridge + poly2", poly["ridge"]["test_metrics"]["rmse"],
         poly["ridge"]["test_metrics"]["r2"], C_RIDGE),
        ("Lasso + poly2", poly["lasso"]["test_metrics"]["rmse"],
         poly["lasso"]["test_metrics"]["r2"], C_LASSO),
    ]
    x = np.arange(len(entries))
    bars = ax.bar(x, [e[1] for e in entries], color=[e[3] for e in entries],
                  width=0.62)
    for b, e in zip(bars, entries):
        ax.text(b.get_x() + b.get_width() / 2, e[1] + 0.06,
                f"{e[1]:.2f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.25,
                f"$R^2$={e[2]:.3f}", ha="center", fontsize=7.5, color="white")
    # 5 折 CV 均值（误差条）
    for i, model in enumerate(("ridge", "lasso")):
        ax.errorbar(x[i + 1], cv["folds"][model]["rmse_mean"],
                    yerr=cv["folds"][model]["rmse_std"], fmt="s", ms=4,
                    color="black", capsize=4, lw=1,
                    label="5-fold CV mean±std" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in entries], fontsize=8.5, rotation=12)
    ax.set_ylabel("test RMSE (k$)")
    ax.set_ylim(0, 5.4)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(a) Test RMSE of all models")

    # (b) 最优模型（Ridge+poly2）预测散点
    ax = axes[1]
    y_true = np.array(poly["ridge"]["y_test"])
    y_pred = np.array(poly["ridge"]["y_pred"])
    ax.scatter(y_true, y_pred, s=18, alpha=0.75, edgecolors="none",
               color=C_RIDGE)
    y_ols = np.array(ols["y_pred"])
    ax.scatter(y_true, y_ols, s=18, alpha=0.5, edgecolors="none",
               color="#8c8c8c", marker="^", label="OLS")
    lo, hi = 0, 55
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    m = poly["ridge"]["test_metrics"]
    ax.set_xlabel("True MEDV (k$)")
    ax.set_ylabel("Predicted MEDV (k$)")
    ax.set_title(f"(b) Ridge+poly2 test: RMSE={m['rmse']:.2f}, $R^2$={m['r2']:.3f}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boston_reg_comparison.pdf"))
    plt.close(fig)

    # 残差直方图（Ridge+poly2）
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    res = y_pred - y_true
    ax.hist(res, bins=20, color=C_RIDGE, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="r", ls="--", lw=1.2)
    ax.set_xlabel("Residual (predicted - true, k$)")
    ax.set_ylabel("Count")
    ax.set_title("Ridge+poly2 test residuals")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boston_reg_residuals.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    plot_reg_paths()
    plot_comparison()
    print("figures saved to", FIG_DIR)
