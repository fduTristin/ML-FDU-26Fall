# -*- coding: utf-8 -*-
"""读取 logs/*.json 并绘制训练曲线与混淆矩阵。"""

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

EXPS = [
    ("baseline",           "Baseline CNN"),
    ("vggstyle_wobn",      "Deeper CNN (w/o BN)"),
    ("vggstyle_bn",        "Deeper CNN + BN + Dropout"),
    ("vggstyle_aug_cosine","+ Augmentation & cosine LR"),
]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load(tag):
    with open(os.path.join(LOGS, f"{tag}.json")) as f:
        return json.load(f)


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for (tag, label), c in zip(EXPS, COLORS):
        r = load(tag)
        ep = np.arange(1, r["epochs"] + 1)
        axes[0].plot(ep, r["history"]["train_loss"], color=c, label=label)
        axes[1].plot(ep, [a * 100 for a in r["history"]["test_acc"]], color=c, label=label)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test Accuracy (%)")
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "training_curves.pdf"))
    plt.close(fig)


def plot_confusion(tag="vggstyle_aug_cosine"):
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
    fig.savefig(os.path.join(FIGS, "confusion_matrix.pdf"))
    plt.close(fig)


def print_summary():
    rows = []
    for tag, label in EXPS:
        r = load(tag)
        rows.append((label, r["params"], r["best_test_acc"] * 100,
                     r["final_test_acc"] * 100, r["time_sec"]))
    print(f"{'experiment':<28}{'#params':>12}{'best/%':>9}{'final/%':>9}{'time/s':>9}")
    for label, p, best, final, t in rows:
        print(f"{label:<28}{p:>12,}{best:>9.2f}{final:>9.2f}{t:>9.1f}")


if __name__ == "__main__":
    plot_curves()
    plot_confusion()
    print_summary()
