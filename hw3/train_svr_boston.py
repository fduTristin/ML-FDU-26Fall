# -*- coding: utf-8 -*-
"""训练 / 评估脚本：在波士顿房价数据集上训练 SVR 回归模型。

实验设计：
  1. 线性回归 baseline（scikit-learn LinearRegression）。
  2. LinearSVR（epsilon-不敏感损失 + 线性核）：C x epsilon 网格搜索，
     用与作业2相同的 train/val/test = 64%/16%/20% 划分（seed=42）。
  3. SVR（RBF 核）：C x gamma x epsilon 网格搜索，同样以验证集 RMSE 选参，
     最后用 5 折交叉验证评估稳健性。
  训练时对标签做 Z-score 标准化，评估时还原为原始量纲（千美元）。

用法示例：
    python train_svr_boston.py --model linear --tag boston_linear
    python train_svr_boston.py --model linsvr --tag boston_linsvr
    python train_svr_boston.py --model svr --tag boston_svr
    python train_svr_boston.py --model svr --cv 5 --tag boston_svr_cv
"""

import argparse
import csv
import json
import os
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "boston.csv")

FEATURE_NAMES = ["crim", "zn", "indus", "chas", "nox", "rm", "age",
                 "dis", "rad", "tax", "ptratio", "b", "lstat"]


def load_data(seed=42):
    """读入 boston.csv 并划分 train/val/test = 64%/16%/20%（与作业2一致），
    按训练集统计量做 Z-score 标准化。"""
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in r] for r in reader], dtype=np.float64)
    assert header[-1] == "medv"
    X, y = rows[:, :-1], rows[:, -1:]

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_train, n_val = int(0.64 * len(X)), int(0.16 * len(X))
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
    y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-8
    Xs = (X - x_mean) / x_std
    ys = (y - y_mean) / y_std  # 标签标准化仅用于训练，评估时还原
    return {
        "X_train": Xs[tr], "y_train": ys[tr], "X_val": Xs[va], "y_val": ys[va],
        "X_test": Xs[te], "y_test": ys[te],
        "y_mean": y_mean, "y_std": y_std,
        "raw": (X, y),
    }


def regress_metrics(y_true_raw, y_pred_raw):
    """在原始量纲上计算 MSE / RMSE / MAE / R^2。"""
    y_true = y_true_raw.ravel()
    y_pred = y_pred_raw.ravel()
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def to_raw(ys, data):
    """标准化的 y 还原为原始量纲（千美元）。"""
    return np.asarray(ys).reshape(-1, 1) * data["y_std"] + data["y_mean"]


def run_linear(args, data):
    reg = LinearRegression().fit(data["X_train"], data["y_train"].ravel())
    pred_raw = to_raw(reg.predict(data["X_test"]), data)
    metrics = regress_metrics(to_raw(data["y_test"], data), pred_raw)
    val_pred = to_raw(reg.predict(data["X_val"]), data)
    metrics["val_rmse"] = regress_metrics(to_raw(data["y_val"], data),
                                          val_pred)["rmse"]
    result = {
        "tag": args.tag, "task": "boston", "model": "linear",
        "test_metrics": metrics,
        "coef": dict(zip(FEATURE_NAMES, [float(c) for c in reg.coef_.ravel()])),
        "y_test": [float(v) for v in to_raw(data["y_test"], data).ravel()],
        "y_pred": [float(v) for v in pred_raw.ravel()],
    }
    print(f"[{args.tag}] linear: test RMSE={metrics['rmse']:.3f}k$, "
          f"R2={metrics['r2']:.4f}")
    return result


def grid_search(args, data, model_kind):
    """在网格上搜索超参数，以验证集 RMSE 选择最优模型，返回网格与最优配置。"""
    Xtr, ytr = data["X_train"], data["y_train"].ravel()
    Xva, yva_raw = data["X_val"], to_raw(data["y_val"], data).ravel()
    grid = []
    if model_kind == "linsvr":
        combos = [(C, None, eps) for C in args.C for eps in args.epsilon]
    else:
        combos = [(C, g, eps) for C in args.C for g in args.gamma
                  for eps in args.epsilon]
    for C, gamma, eps in combos:
        t0 = time.time()
        if model_kind == "linsvr":
            mdl = LinearSVR(C=C, epsilon=eps, max_iter=50000,
                            random_state=args.seed)
        else:
            mdl = SVR(C=C, kernel="rbf", gamma=gamma, epsilon=eps,
                      cache_size=512)
        mdl.fit(Xtr, ytr)
        t_fit = time.time() - t0
        pred_raw = to_raw(mdl.predict(Xva), data).ravel()
        rmse = regress_metrics(yva_raw, pred_raw)["rmse"]
        rec = {"C": C, "epsilon": eps, "val_rmse": rmse, "fit_time": t_fit}
        if gamma is not None:
            rec["gamma"] = gamma
            rec["n_support"] = int(mdl.n_support_.sum())
        grid.append(rec)
        print(f"[{args.tag}] C={C:g}" +
              (f", gamma={gamma:g}" if gamma is not None else "") +
              f", eps={eps:g}: val RMSE={rmse:.3f}k$")
    best = min(grid, key=lambda d: d["val_rmse"])
    return grid, best


def fit_eval_best(args, data, model_kind, best, return_pred=True):
    """用最优超参数在 train 上重训，在测试集上评估。"""
    t0 = time.time()
    if model_kind == "linsvr":
        mdl = LinearSVR(C=best["C"], epsilon=best["epsilon"], max_iter=50000,
                        random_state=args.seed)
    else:
        mdl = SVR(C=best["C"], kernel="rbf", gamma=best["gamma"],
                  epsilon=best["epsilon"], cache_size=512)
    mdl.fit(data["X_train"], data["y_train"].ravel())
    fit_time = time.time() - t0
    pred_raw = to_raw(mdl.predict(data["X_test"]), data)
    y_true_raw = to_raw(data["y_test"], data)
    metrics = regress_metrics(y_true_raw, pred_raw)
    extra = {}
    if model_kind == "svr":
        extra["n_support"] = int(mdl.n_support_.sum())
    return mdl, metrics, fit_time, y_true_raw.ravel(), pred_raw.ravel(), extra


def run_svr(args, data, model_kind):
    grid, best = grid_search(args, data, model_kind)
    mdl, metrics, fit_time, y_true, y_pred, extra = fit_eval_best(
        args, data, model_kind, best)
    result = {
        "tag": args.tag, "task": "boston", "model": model_kind,
        "grid": grid, "best": best,
        "test_metrics": metrics, "fit_time": fit_time, **extra,
        "y_test": [float(v) for v in y_true],
        "y_pred": [float(v) for v in y_pred],
    }
    print(f"[{args.tag}] {model_kind}: best={best} -> "
          f"test RMSE={metrics['rmse']:.3f}k$  MAE={metrics['mae']:.3f}k$  "
          f"R2={metrics['r2']:.4f}")
    return result


def run_cv(args):
    """在全数据上做 k 折交叉验证：每折内部按训练折统计量标准化，
    并在训练折上做一次小型网格搜索，模拟完整的训练+选参流程。"""
    X, y = load_data(args.seed)["raw"]
    k = args.cv
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    fold_rmse, fold_r2, fold_params = [], [], []
    t0 = time.time()
    for i in range(k):
        te = folds[i]
        rest = np.concatenate([folds[j] for j in range(k) if j != i])
        rng2 = np.random.RandomState(args.seed + i)
        va = rng2.choice(rest, size=len(rest) // 8, replace=False)  # ~12.5% of rest
        tr = np.setdiff1d(rest, va)
        x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
        y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-8
        fold_data = {
            "X_train": (X[tr] - x_mean) / x_std, "y_train": (y[tr] - y_mean) / y_std,
            "X_val": (X[va] - x_mean) / x_std, "y_val": (y[va] - y_mean) / y_std,
            "X_test": (X[te] - x_mean) / x_std, "y_test": (y[te] - y_mean) / y_std,
            "y_mean": y_mean, "y_std": y_std,
        }
        _, best = grid_search(args, fold_data, "svr")
        _, metrics, _, _, _, _ = fit_eval_best(args, fold_data, "svr", best)
        fold_rmse.append(metrics["rmse"])
        fold_r2.append(metrics["r2"])
        fold_params.append({kk: best[kk] for kk in ("C", "gamma", "epsilon")})
        print(f"[{args.tag}] fold {i + 1}/{k}: RMSE={metrics['rmse']:.3f}k$  "
              f"R2={metrics['r2']:.4f}  params={fold_params[-1]}")
    result = {
        "tag": args.tag, "task": "boston", "model": "svr", "cv": k,
        "fold_params": fold_params,
        "fold_rmse": fold_rmse, "fold_r2": fold_r2,
        "rmse_mean": float(np.mean(fold_rmse)), "rmse_std": float(np.std(fold_rmse)),
        "r2_mean": float(np.mean(fold_r2)), "r2_std": float(np.std(fold_r2)),
        "time_sec": time.time() - t0,
    }
    print(f"[{args.tag}] {k}-fold CV: RMSE={result['rmse_mean']:.3f}±"
          f"{result['rmse_std']:.3f}k$  R2={result['r2_mean']:.4f}±"
          f"{result['r2_std']:.4f}")
    return result


def save_json(result):
    out_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{result['tag']}.json"), "w") as f:
        json.dump(result, f, indent=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="svr", choices=["linear", "linsvr", "svr"])
    p.add_argument("--tag", default=None)
    p.add_argument("--C", type=float, nargs="+",
                   default=[0.1, 1.0, 10.0, 100.0])
    p.add_argument("--gamma", type=float, nargs="+",
                   default=[0.01, 0.05, 0.1, 0.5])
    p.add_argument("--epsilon", type=float, nargs="+",
                   default=[0.01, 0.05, 0.1, 0.3])
    p.add_argument("--cv", type=int, default=0, help=">0 时执行 k 折交叉验证（仅 svr）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.tag is None:
        args.tag = f"boston_{args.model}" + (f"_cv" if args.cv > 0 else "")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cv > 0:
        save_json(run_cv(args))
    else:
        data = load_data(args.seed)
        t0 = time.time()
        if args.model == "linear":
            result = run_linear(args, data)
        elif args.model == "linsvr":
            result = run_svr(args, data, "linsvr")
        else:
            result = run_svr(args, data, "svr")
        result["time_sec"] = time.time() - t0
        save_json(result)
