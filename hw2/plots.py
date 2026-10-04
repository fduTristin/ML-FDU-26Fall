# -*- coding: utf-8 -*-
"""读取 logs/*.json 并绘制作业2的图表（MNIST 曲线/混淆矩阵，房价回归曲线/散点图）。"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

MNIST_EXPS = [
    ("mlp_1x256",        "MLP 784-256-10 (baseline)"),
    ("mlp_2x512_256",    "MLP 784-512-256-10"),
    ("mlp_2x512_256_do", "+ Dropout & weight decay"),
    ("mlp_full",         "+ Augmentation & cosine LR"),
]
BOSTON_EXPS = [
    ("boston_mlp_small", "MLP 13-32-1 (no reg)"),
    ("boston_mlp",       "MLP 13-64-32-1 + reg"),
]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load(tag):
    with open(os.path.join(LOGS, f"{tag}.json")) as f:
        return json.load(f)


def plot_mnist_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for (tag, label), c in zip(MNIST_EXPS, COLORS):
        r = load(tag)
        ep = np.arange(1, r["epochs"] + 1)
        axes[0].plot(ep, r["history"]["train_loss"], color=c, label=label)
        axes[1].plot(ep, [a * 100 for a in r["history"]["test_acc"]], color=c, label=label)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test Accuracy (%)")
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "mnist_training_curves.pdf"))
    plt.close(fig)


def plot_mnist_confusion(tag="mlp_full"):
    r = load(tag)
    cm = np.array(r["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "mnist_confusion_matrix.pdf"))
    plt.close(fig)


def plot_boston_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for (tag, label), c in zip(BOSTON_EXPS, COLORS):
        r = load(tag)
        ep = np.arange(1, len(r["history"]["train_loss"]) + 1)
        axes[0].plot(ep, r["history"]["train_loss"], color=c, label=label, alpha=0.9)
        axes[1].plot(ep, r["history"]["val_rmse"], color=c, label=label, alpha=0.9)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train MSE (standardized)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val RMSE (k$)")
    axes[0].set_yscale("log"); axes[1].set_ylim(bottom=0)
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "boston_training_curves.pdf"))
    plt.close(fig)


def plot_boston_scatter(tag="boston_mlp"):
    r = load(tag)
    y_true = np.array(r["y_test"]); y_pred = np.array(r["y_pred"])
    m = r["test_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    lo = min(y_true.min(), y_pred.min()) - 2
    hi = max(y_true.max(), y_pred.max()) + 2
    axes[0].scatter(y_true, y_pred, s=22, alpha=0.75, edgecolors="none", c=COLORS[0])
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.2)
    axes[0].set_xlabel("True price (k$)"); axes[0].set_ylabel("Predicted price (k$)")
    axes[0].set_title(f"Test: RMSE={m['rmse']:.2f}k$, $R^2$={m['r2']:.3f}", fontsize=10)
    axes[0].grid(alpha=0.3)
    res = y_true - y_pred
    axes[1].hist(res, bins=20, color=COLORS[2], edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="r", ls="--", lw=1.2)
    axes[1].set_xlabel("Residual (k$)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Test residual histogram", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "boston_pred_vs_true.pdf"))
    plt.close(fig)


def print_summary():
    print(f"{'MNIST experiment':<30}{'#params':>12}{'best/%':>9}{'final/%':>9}{'time/s':>9}")
    for tag, label in MNIST_EXPS:
        r = load(tag)
        print(f"{label:<30}{r['params']:>12,}{r['best_test_acc'] * 100:>9.2f}"
              f"{r['final_test_acc'] * 100:>9.2f}{r['time_sec']:>9.1f}")
    print()
    print(f"{'Boston experiment':<30}{'#params':>12}{'RMSE/k$':>9}{'MAE/k$':>9}{'R2':>9}")
    for tag in ["boston_linear"] + [t for t, _ in BOSTON_EXPS]:
        r = load(tag)
        m = r["test_metrics"]
        p = r.get("params", 14)
        print(f"{tag:<30}{p:>12,}{m['rmse']:>9.3f}{m['mae']:>9.3f}{m['r2']:>9.4f}")
    r = load("boston_cv")
    print(f"\nboston_cv {r['cv']}-fold: RMSE={r['rmse_mean']:.3f}±{r['rmse_std']:.3f}k$, "
          f"R2={r['r2_mean']:.4f}±{r['r2_std']:.4f}")


if __name__ == "__main__":
    plot_mnist_curves()
    plot_mnist_confusion()
    plot_boston_curves()
    plot_boston_scatter()
    print_summary()
