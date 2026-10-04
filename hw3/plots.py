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


def load(tag):
    with open(os.path.join(LOG_DIR, f"{tag}.json")) as f:
        return json.load(f)


def plot_model_comparison():
    """条形图：线性 SVM、子集上各核 SVM 与 RBF SVM（重训）的测试/评估准确率。"""
    lin = load("svm_linear")
    ker = load("svm_kernels")
    rbf = load("svm_rbf")
    names, accs, notes = [], [], []
    names.append("Linear SVM\n(60k train, full test)")
    accs.append(lin["test_acc"] * 100)
    notes.append(f"C={lin['best_C']:g}")
    for r in ker["results"]:
        names.append(f"SVM {r['kernel']}\n({ker['n_sub']//1000}k sub, 5k eval)")
        accs.append(r["acc"] * 100)
        notes.append(f"{r['n_support']} SVs")
    names.append(f"RBF SVM + PCA{rbf['pca_dim']}\n({rbf['n_refit']//1000}k train, full test)")
    accs.append(rbf["test_acc"] * 100)
    notes.append(f"C={rbf['best_C']:g}, $\\gamma$={rbf['best_gamma']:g}")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#4C72B0", "#8c8c8c", "#8c8c8c", "#8c8c8c", "#C44E52"]
    bars = ax.bar(range(len(names)), accs, color=colors, width=0.62)
    for i, (b, a, n) in enumerate(zip(bars, accs, notes)):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.06, f"{a:.2f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 84.5, n,
                ha="center", va="bottom", fontsize=7.5, color="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(84, 100)
    ax.set_ylabel("Accuracy / %")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_svm_comparison.pdf"))
    plt.close(fig)


def plot_rbf_grid():
    """RBF SVM 网格搜索验证准确率热力图（C x gamma）。"""
    rbf = load("svm_rbf")
    Cs = sorted({g["C"] for g in rbf["grid"]})
    gs = sorted({g["gamma"] for g in rbf["grid"]})
    Z = np.zeros((len(gs), len(Cs)))
    T = np.zeros_like(Z)
    for g in rbf["grid"]:
        i, j = gs.index(g["gamma"]), Cs.index(g["C"])
        Z[i, j] = g["val_acc"] * 100
        T[i, j] = g["n_support"]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.imshow(Z, origin="lower", cmap="viridis", aspect="auto",
                   vmin=Z.min() - 0.15, vmax=Z.max() + 0.05)
    for i in range(len(gs)):
        for j in range(len(Cs)):
            ax.text(j, i, f"{Z[i, j]:.2f}\n({int(T[i, j])})", ha="center",
                    va="center", fontsize=8,
                    color="white" if Z[i, j] < (Z.min() + Z.max()) / 2 else "black")
    ax.set_xticks(range(len(Cs)))
    ax.set_xticklabels([f"{c:g}" for c in Cs])
    ax.set_yticks(range(len(gs)))
    ax.set_yticklabels([f"{g:g}" for g in gs])
    ax.set_xlabel("$C$")
    ax.set_ylabel("$\\gamma$")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="validation accuracy / %")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_rbf_grid.pdf"))
    plt.close(fig)


def plot_confusion_matrix():
    rbf = load("svm_rbf")
    cm = np.array(rbf["confusion_matrix"])
    n_err = int(cm.sum() - np.trace(cm))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6.5,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"RBF SVM on MNIST test set ({n_err} errors / 10000)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_svm_confusion_matrix.pdf"))
    plt.close(fig)


def plot_boston():
    """波士顿房价：SVR 网格搜索验证 RMSE 曲线 + 最优模型预测散点与残差。"""
    svr = load("boston_svr")
    # 左：以最优 eps 为基准，画 C-gamma 平面验证 RMSE；右：预测散点
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # (a) LinearSVR：C 扫描下不同 eps 的验证 RMSE
    ax = axes[0]
    lin = load("boston_linsvr")
    for eps in sorted({g["epsilon"] for g in lin["grid"]}):
        xs = [g["C"] for g in lin["grid"] if g["epsilon"] == eps]
        ys_ = [g["val_rmse"] for g in lin["grid"] if g["epsilon"] == eps]
        ax.semilogx(xs, ys_, "o-", label=f"$\\varepsilon$={eps:g}", ms=4)
    ax.set_xlabel("$C$")
    ax.set_ylabel("validation RMSE (k$)")
    ax.set_title("(a) LinearSVR")
    ax.legend(fontsize=8)
    lreg = load("boston_linear")
    ax.axhline(lreg["test_metrics"]["val_rmse"], color="gray", ls="--",
               lw=1, label="LinearReg val")
    ax.legend(fontsize=8)

    # (b) RBF SVR：最优 eps 下的 C-gamma 热力图
    ax = axes[1]
    best_eps = svr["best"]["epsilon"]
    Cs = sorted({g["C"] for g in svr["grid"] if g["epsilon"] == best_eps})
    gs = sorted({g["gamma"] for g in svr["grid"] if g["epsilon"] == best_eps})
    Z = np.full((len(gs), len(Cs)), np.nan)
    for g in svr["grid"]:
        if g["epsilon"] == best_eps:
            i, j = gs.index(g["gamma"]), Cs.index(g["C"])
            Z[i, j] = g["val_rmse"]
    im = ax.imshow(Z, origin="lower", cmap="viridis_r", aspect="auto")
    for i in range(len(gs)):
        for j in range(len(Cs)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if Z[i, j] > (np.nanmin(Z) + np.nanmax(Z)) / 2
                    else "black")
    ax.set_xticks(range(len(Cs)))
    ax.set_xticklabels([f"{c:g}" for c in Cs])
    ax.set_yticks(range(len(gs)))
    ax.set_yticklabels([f"{g:g}" for g in gs])
    ax.set_xlabel("$C$")
    ax.set_ylabel("$\\gamma$")
    ax.set_title(f"(b) SVR-RBF ($\\varepsilon$={best_eps:g})")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="validation RMSE (k$)")

    # (c) 最优 SVR 预测散点
    ax = axes[2]
    y_true = np.array(svr["y_test"])
    y_pred = np.array(svr["y_pred"])
    ax.scatter(y_true, y_pred, s=18, alpha=0.75, edgecolors="none",
               color="#4C72B0")
    lo, hi = 0, 55
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    m = svr["test_metrics"]
    ax.set_xlabel("True MEDV (k$)")
    ax.set_ylabel("Predicted MEDV (k$)")
    ax.set_title(f"(c) SVR-RBF test: RMSE={m['rmse']:.2f}, $R^2$={m['r2']:.3f}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boston_svr.pdf"))
    plt.close(fig)

    # 残差直方图
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    res = y_pred - y_true
    ax.hist(res, bins=20, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="r", ls="--", lw=1.2)
    ax.set_xlabel("Residual (predicted - true, k$)")
    ax.set_ylabel("Count")
    ax.set_title("SVR-RBF test residuals")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boston_svr_residuals.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    plot_model_comparison()
    plot_rbf_grid()
    plot_confusion_matrix()
    plot_boston()
    print("figures saved to", FIG_DIR)
