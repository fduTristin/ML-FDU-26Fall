# -*- coding: utf-8 -*-
"""训练 / 评估脚本：在 MNIST 上训练线性 / 非线性 SVM 分类器并记录日志。

实验设计：
  1. 线性 SVM（LinearSVC，hinge 损失，primal）：在完整 60000 训练样本上
     训练，C 通过留出验证集（5000 样本）网格搜索。
  2. 核函数对比（同一 10000 训练子集）：linear / polynomial(3) / rbf，
     固定 C=1, gamma="scale"，验证集取 5000。
  3. RBF 非线性 SVM 网格搜索：PCA 降维后在子训练集上对 C x gamma 做
     网格搜索，用验证集挑选最优超参数，最后在 20000 样本上重训并在
     完整测试集（10000）上评估。

用法示例：
    python train_svm_mnist.py --exp linear --tag svm_linear
    python train_svm_mnist.py --exp kernels --tag svm_kernels
    python train_svm_mnist.py --exp rbf --tag svm_rbf
"""

import argparse
import json
import os
import time

import numpy as np
import torchvision
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
# 若 hw1 已下载 MNIST 则直接复用，避免重复下载
if not os.path.exists(os.path.join(DATA_DIR, "MNIST")):
    hw1_data = os.path.join(BASE_DIR, "..", "hw1", "data")
    if os.path.exists(os.path.join(hw1_data, "MNIST")):
        DATA_DIR = hw1_data


def load_mnist():
    """读入 MNIST，展平并缩放到 [0,1]。返回 (X_train, y_train, X_test, y_test)。"""
    train_set = torchvision.datasets.MNIST(DATA_DIR, train=True, download=False)
    test_set = torchvision.datasets.MNIST(DATA_DIR, train=False, download=False)
    X_train = train_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_set.targets.numpy()
    X_test = test_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_set.targets.numpy()
    return X_train, y_train, X_test, y_test


def split_train_val(X, y, n_val=5000, seed=42):
    """分层留出划分验证集。"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    va, tr = idx[:n_val], idx[n_val:]
    return X[tr], y[tr], X[va], y[va]


def subsample(X, y, n, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))[:n]
    return X[idx], y[idx]


def run_linear(args, X_train, y_train, X_test, y_test):
    """线性 SVM：LinearSVC，在完整训练集上训练，C 用验证集网格搜索。"""
    Xtr, ytr, Xva, yva = split_train_val(X_train, y_train, n_val=args.n_val,
                                         seed=args.seed)
    grid, t_fit_total = [], 0.0
    for C in args.C:
        t0 = time.time()
        clf = LinearSVC(C=C, dual=True, max_iter=20000, random_state=args.seed)
        clf.fit(Xtr, ytr)
        t_fit = time.time() - t0
        val_acc = float(clf.score(Xva, yva))
        grid.append({"C": C, "val_acc": val_acc, "fit_time": t_fit})
        t_fit_total += t_fit
        print(f"[{args.tag}] C={C:g}: val_acc={val_acc:.4f} ({t_fit:.1f}s)")
    best = max(grid, key=lambda d: d["val_acc"])

    # 用最优 C 在 train+val（完整 60000 样本）上重训
    t0 = time.time()
    clf = LinearSVC(C=best["C"], dual=True, max_iter=20000, random_state=args.seed)
    clf.fit(X_train, y_train)
    refit_time = time.time() - t0
    t0 = time.time()
    pred = clf.predict(X_test)
    test_time = time.time() - t0
    test_acc = float((pred == y_test).mean())
    print(f"[{args.tag}] best C={best['C']:g}, test_acc={test_acc:.4f} "
          f"(refit {refit_time:.1f}s)")

    return {
        "tag": args.tag, "task": "mnist", "model": "LinearSVC",
        "n_train": int(len(Xtr)) + args.n_val, "n_val": args.n_val,
        "grid": grid, "best_C": best["C"], "best_val_acc": best["val_acc"],
        "test_acc": test_acc,
        "time_sec": t_fit_total + refit_time,
        "test_time": test_time,
        "y_test": [int(v) for v in y_test],
        "y_pred": [int(v) for v in pred],
    }


def run_kernels(args, X_train, y_train, X_test, y_test):
    """核函数对比：在同一训练子集上比较 linear / poly / rbf。"""
    Xsub, ysub = subsample(X_train, y_train, args.n_sub, seed=args.seed)
    # 从测试集中取一部分作为公平比较用的评估集
    rng = np.random.RandomState(args.seed)
    ev = rng.permutation(len(X_test))[:args.n_val_test]
    Xev, yev = X_test[ev], y_test[ev]

    results = []
    kernels = [("linear", {}), ("poly", {"degree": 3, "coef0": 1.0}), ("rbf", {})]
    for kernel, extra in kernels:
        t0 = time.time()
        clf = SVC(C=1.0, kernel=kernel, gamma="scale", cache_size=2048, **extra)
        clf.fit(Xsub, ysub)
        t_fit = time.time() - t0
        t0 = time.time()
        acc = float(clf.score(Xev, yev))
        t_inf = time.time() - t0
        results.append({
            "kernel": kernel, "params": extra, "acc": acc,
            "n_support": int(clf.n_support_.sum()),
            "fit_time": t_fit, "infer_time": t_inf,
        })
        print(f"[{args.tag}] kernel={kernel}: acc={acc:.4f}, "
              f"SVs={int(clf.n_support_.sum())}, fit={t_fit:.1f}s, infer={t_inf:.1f}s")

    return {
        "tag": args.tag, "task": "mnist", "model": "SVC(kernel compare)",
        "n_sub": args.n_sub, "n_eval": args.n_val_test,
        "C": 1.0, "gamma": "scale", "results": results,
        "time_sec": sum(r["fit_time"] + r["infer_time"] for r in results),
    }


def run_rbf(args, X_train, y_train, X_test, y_test):
    """RBF SVM：PCA 降维 + C x gamma 网格搜索 + 大子集重训。"""
    # PCA 降维（仅在调参子集上 fit，再作用于全部数据）
    Xsub, ysub = subsample(X_train, y_train, args.n_sub, seed=args.seed)
    pca = PCA(n_components=args.pca_dim, random_state=args.seed).fit(Xsub)
    evr = float(pca.explained_variance_ratio_.sum())
    Xp = pca.transform(X_train)
    Xp_test = pca.transform(X_test)
    print(f"[{args.tag}] PCA: 784->{args.pca_dim}, explained_var={evr:.4f}")

    Xtr, ytr, Xva, yva = split_train_val(Xp, y_train, n_val=args.n_val,
                                         seed=args.seed)
    # 在较小子集上做网格搜索（完整 60000 上网格搜索过慢）
    Xgs, ygs = subsample(Xtr, ytr, args.n_grid, seed=args.seed)
    grid, t_grid = [], 0.0
    for C in args.C:
        for gamma in args.gamma:
            t0 = time.time()
            clf = SVC(C=C, kernel="rbf", gamma=gamma, cache_size=2048)
            clf.fit(Xgs, ygs)
            t_fit = time.time() - t0
            val_acc = float(clf.score(Xva, yva))
            grid.append({"C": C, "gamma": gamma, "val_acc": val_acc,
                         "fit_time": t_fit,
                         "n_support": int(clf.n_support_.sum())})
            t_grid += t_fit
            print(f"[{args.tag}] C={C:g}, gamma={gamma:g}: "
                  f"val_acc={val_acc:.4f}, SVs={int(clf.n_support_.sum())} "
                  f"({t_fit:.1f}s)")
    best = max(grid, key=lambda d: d["val_acc"])

    # 最优超参数下在更大子集（含训练+验证共 20000）上重训并测试
    Xbig, ybig = subsample(Xp, y_train, args.n_refit, seed=args.seed)
    t0 = time.time()
    clf = SVC(C=best["C"], kernel="rbf", gamma=best["gamma"], cache_size=2048)
    clf.fit(Xbig, ybig)
    refit_time = time.time() - t0
    t0 = time.time()
    pred = clf.predict(Xp_test)
    test_time = time.time() - t0
    test_acc = float((pred == y_test).mean())
    cm = confusion_matrix(y_test, pred).tolist()
    print(f"[{args.tag}] best C={best['C']:g}, gamma={best['gamma']:g}, "
          f"test_acc={test_acc:.4f}, SVs={int(clf.n_support_.sum())} "
          f"(refit {refit_time:.1f}s)")

    return {
        "tag": args.tag, "task": "mnist", "model": "SVC(RBF)",
        "pca_dim": args.pca_dim, "pca_explained_var": evr,
        "n_grid": args.n_grid, "n_val": args.n_val, "n_refit": args.n_refit,
        "grid": grid, "best_C": best["C"], "best_gamma": best["gamma"],
        "best_val_acc": best["val_acc"],
        "test_acc": test_acc,
        "n_support": int(clf.n_support_.sum()),
        "time_sec": t_grid + refit_time, "test_time": test_time,
        "confusion_matrix": cm,
        "y_test": [int(v) for v in y_test],
        "y_pred": [int(v) for v in pred],
    }


def save_json(result):
    out_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{result['tag']}.json"), "w") as f:
        json.dump(result, f, indent=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="linear",
                   choices=["linear", "kernels", "rbf"])
    p.add_argument("--tag", default=None)
    p.add_argument("--C", type=float, nargs="+", default=None)
    p.add_argument("--gamma", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    p.add_argument("--n-val", type=int, default=5000, help="留出验证集大小")
    p.add_argument("--n-sub", type=int, default=10000, help="核函数对比子集大小")
    p.add_argument("--n-grid", type=int, default=10000, help="RBF 网格搜索训练子集")
    p.add_argument("--n-refit", type=int, default=20000, help="RBF 最终重训子集")
    p.add_argument("--n-val-test", type=int, default=5000,
                   help="核函数对比评估集大小（取自测试集）")
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.tag is None:
        args.tag = f"svm_{args.exp}"
    if args.C is None:
        args.C = [1e-3, 1e-2, 3e-2, 1e-1] if args.exp == "linear" \
            else [1.0, 10.0, 100.0]
    return args


if __name__ == "__main__":
    args = parse_args()
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"[{args.tag}] train={X_train.shape}, test={X_test.shape}")
    t0 = time.time()
    if args.exp == "linear":
        result = run_linear(args, X_train, y_train, X_test, y_test)
    elif args.exp == "kernels":
        result = run_kernels(args, X_train, y_train, X_test, y_test)
    else:
        result = run_rbf(args, X_train, y_train, X_test, y_test)
    save_json(result)
    print(f"[{args.tag}] total wall time: {time.time() - t0:.1f}s")
