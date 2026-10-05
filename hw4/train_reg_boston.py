# -*- coding: utf-8 -*-
"""训练 / 评估脚本：在波士顿房价数据集上实现并评估岭回归与 Lasso 回归。

实验设计（沿用作业2/3的 train/val/test = 64%/16%/20% 划分，seed=42）：
  1. 最小二乘（OLS）baseline。
  2. 岭回归（Ridge）：alpha 网格搜索，验证集 RMSE 选参；
     解析解 w = (X^T X + alpha*I)^{-1} X^T y 手写实现，并与 sklearn Ridge 核对。
  3. Lasso：alpha 网格搜索，验证集 RMSE 选参；
     坐标下降（coordinate descent）手写实现，并与 sklearn Lasso 核对。
  4. 二阶多项式特征（13 -> 104 维）上的 OLS / Ridge / Lasso 对比，
     展示正则化在高维特征下的作用。
  5. 5 折交叉验证：每折内部独立标准化并重新选参，评估结论稳健性。

特征按训练集统计量做 Z-score 标准化；标签直接使用原始量纲（千美元），
截距单独处理、不加惩罚。

用法示例：
    python train_reg_boston.py --model linear --tag boston_ols
    python train_reg_boston.py --model ridge  --tag boston_ridge
    python train_reg_boston.py --model lasso  --tag boston_lasso
    python train_reg_boston.py --model poly   --tag boston_poly
    python train_reg_boston.py --model both   --cv 5 --tag boston_cv
"""

import argparse
import csv
import json
import os
import time

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "boston.csv")

FEATURE_NAMES = ["crim", "zn", "indus", "chas", "nox", "rm", "age",
                 "dis", "rad", "tax", "ptratio", "b", "lstat"]


# ----------------------------------------------------------------------
# 手写实现：岭回归（解析解）与 Lasso（坐标下降）
# ----------------------------------------------------------------------
class RidgeClosedForm:
    """岭回归解析解：w = (X^T X + alpha * I)^{-1} X^T y（截距不加惩罚）。"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.x_mean_ = X.mean(0)          # 居中以便单独处理截距
        self.y_mean_ = y.mean()
        Xc = X - self.x_mean_
        yc = y - self.y_mean_
        d = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(d)
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


class LassoCoordinateDescent:
    """Lasso 坐标下降：min (1/2n)||y - Xw - b||^2 + alpha * ||w||_1。

    软阈值更新：w_j <- S(rho_j, alpha) / z_j，
    rho_j = (1/n) x_j^T (y - b - Xw + w_j x_j), z_j = (1/n) ||x_j||^2。
    支持沿 alpha 网格的 warm start（path 方式）以加速。
    """

    def __init__(self, alpha=0.1, max_iter=10000, tol=1e-7):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, w_init=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        self.x_mean_ = X.mean(0)
        self.y_mean_ = y.mean()
        Xc = X - self.x_mean_
        yc = y - self.y_mean_
        z = (Xc * Xc).mean(0)             # 标准化特征下 z_j = 1
        w = np.zeros(d) if w_init is None else w_init.copy()
        r = yc - Xc @ w                   # 当前残差
        for it in range(self.max_iter):
            w_old = w.copy()
            for j in range(d):
                rho = (Xc[:, j] @ r) / n + z[j] * w[j]
                # 软阈值算子 S(rho, alpha)
                if rho > self.alpha:
                    w[j] = (rho - self.alpha) / z[j]
                elif rho < -self.alpha:
                    w[j] = (rho + self.alpha) / z[j]
                else:
                    w[j] = 0.0
                r += Xc[:, j] * (w_old[j] - w[j])
            if np.max(np.abs(w - w_old)) < self.tol:
                break
        self.coef_ = w
        self.intercept_ = self.y_mean_ - self.x_mean_ @ w
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def load_data(seed=42):
    """读入 boston.csv 并划分 train/val/test = 64%/16%/20%（与作业2/3一致），
    按训练集统计量对特征做 Z-score 标准化；标签保留原始量纲（千美元）。"""
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in r] for r in reader], dtype=np.float64)
    assert header[-1] == "medv"
    X, y = rows[:, :-1], rows[:, -1].ravel()

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_train, n_val = int(0.64 * len(X)), int(0.16 * len(X))
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xs = (X - x_mean) / x_std
    return {
        "X_train": Xs[tr], "y_train": y[tr],
        "X_val": Xs[va], "y_val": y[va],
        "X_test": Xs[te], "y_test": y[te],
        "raw": (X, y),
    }


def regress_metrics(y_true, y_pred):
    """MSE / RMSE / MAE / R^2（原始量纲，千美元）。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "r2": float(1 - np.sum((y_true - y_pred) ** 2)
                        / np.sum((y_true - y_true.mean()) ** 2))}


# ----------------------------------------------------------------------
# 模型流程
# ----------------------------------------------------------------------
def check_against_sklearn(data):
    """随机抽一个 alpha，核对手写实现与 sklearn 的系数/预测差异。"""
    X, y = data["X_train"], data["y_train"]
    check = {}
    for alpha in (0.1, 10.0):
        mine = RidgeClosedForm(alpha=alpha).fit(X, y)
        ref = Ridge(alpha=alpha).fit(X, y)
        check[f"ridge_coef_maxdiff_alpha{alpha}"] = \
            float(np.max(np.abs(mine.coef_ - ref.coef_)))
    for alpha in (0.001, 0.1):
        mine = LassoCoordinateDescent(alpha=alpha).fit(X, y)
        ref = Lasso(alpha=alpha, max_iter=100000, tol=1e-8).fit(X, y)
        check[f"lasso_coef_maxdiff_alpha{alpha}"] = \
            float(np.max(np.abs(mine.coef_ - ref.coef_)))
    print("[check] " + "  ".join(f"{k}={v:.2e}" for k, v in check.items()))
    assert max(check.values()) < 1e-4, "手写实现与 sklearn 差异过大"
    return check


def run_linear(data, tag="boston_ols"):
    reg = LinearRegression().fit(data["X_train"], data["y_train"])
    pred = reg.predict(data["X_test"])
    metrics = regress_metrics(data["y_test"], pred)
    metrics["val_rmse"] = regress_metrics(
        data["y_val"], reg.predict(data["X_val"]))["rmse"]
    result = {
        "tag": tag, "model": "ols", "test_metrics": metrics,
        "coef": dict(zip(FEATURE_NAMES, [float(c) for c in reg.coef_.ravel()])),
        "y_test": [float(v) for v in data["y_test"]],
        "y_pred": [float(v) for v in pred],
    }
    print(f"[{tag}] OLS: test RMSE={metrics['rmse']:.3f}k$  R2={metrics['r2']:.4f}")
    return result


def fit_model(model, X, y, alpha, w_init=None):
    """按 model 类型用给定 alpha 训练，返回（模型, 拟合耗时）。"""
    t0 = time.time()
    if model == "ridge":
        mdl = RidgeClosedForm(alpha=alpha).fit(X, y)
    else:
        mdl = LassoCoordinateDescent(alpha=alpha).fit(X, y, w_init=w_init)
    return mdl, time.time() - t0


def run_grid(data, model, alphas, tag, feature_names=None):
    """在 alpha 网格上训练，以验证集 RMSE 选参，记录系数路径；
    随后用最优 alpha 重训并评估测试集。"""
    if feature_names is None:
        feature_names = FEATURE_NAMES
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    grid, coef_path = [], []
    w_init = None  # Lasso 沿 alpha 递减方向 warm start
    iter_alphas = sorted(alphas, reverse=(model == "lasso"))
    for alpha in iter_alphas:
        mdl, t_fit = fit_model(model, Xtr, ytr, alpha, w_init)
        w_init = mdl.coef_
        val_rmse = regress_metrics(yva, mdl.predict(Xva))["rmse"]
        tr_rmse = regress_metrics(ytr, mdl.predict(Xtr))["rmse"]
        grid.append({"alpha": alpha, "val_rmse": val_rmse,
                     "train_rmse": tr_rmse, "fit_time": t_fit,
                     "n_nonzero": int(np.sum(np.abs(mdl.coef_) > 1e-8))})
        coef_path.append([float(c) for c in mdl.coef_])
        print(f"[{tag}] alpha={alpha:g}: val RMSE={val_rmse:.3f}k$  "
              f"train RMSE={tr_rmse:.3f}k$  n_nonzero={grid[-1]['n_nonzero']}")
    best = min(grid, key=lambda d: d["val_rmse"])
    mdl, fit_time = fit_model(model, Xtr, ytr, best["alpha"])
    pred = mdl.predict(data["X_test"])
    metrics = regress_metrics(data["y_test"], pred)
    result = {
        "tag": tag, "model": model, "grid": grid, "best": best,
        "feature_names": list(feature_names),
        "coef_path": coef_path,
        "coef_best": dict(zip(feature_names,
                              [float(c) for c in mdl.coef_.ravel()])),
        "intercept": float(mdl.intercept_),
        "test_metrics": metrics, "fit_time": fit_time,
        "y_test": [float(v) for v in data["y_test"]],
        "y_pred": [float(v) for v in pred],
    }
    print(f"[{tag}] {model}: best alpha={best['alpha']:g} -> "
          f"test RMSE={metrics['rmse']:.3f}k$  MAE={metrics['mae']:.3f}k$  "
          f"R2={metrics['r2']:.4f}  n_nonzero={int(np.sum(np.abs(mdl.coef_) > 1e-8))}")
    return result


def run_poly(data, tag="boston_poly"):
    """二阶多项式特征（13 -> 104 维）上的 OLS / Ridge / Lasso 对比。"""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtr = poly.fit_transform(data["X_train"])   # 注意输入已标准化
    Xva = poly.transform(data["X_val"])
    Xte = poly.transform(data["X_test"])
    # 多项式展开后各列量纲再次悬殊，需按训练折统计量重新标准化
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xva, Xte = (Xtr - m) / s, (Xva - m) / s, (Xte - m) / s
    sub = {"X_train": Xtr, "y_train": data["y_train"],
           "X_val": Xva, "y_val": data["y_val"],
           "X_test": Xte, "y_test": data["y_test"]}
    results = {"tag": tag, "model": "poly", "degree": 2,
               "n_features": int(Xtr.shape[1])}

    reg = LinearRegression().fit(Xtr, data["y_train"])
    results["ols"] = {"test_metrics": regress_metrics(
        data["y_test"], reg.predict(Xte))}
    print(f"[{tag}] OLS(poly2): test RMSE={results['ols']['test_metrics']['rmse']:.3f}k$  "
          f"R2={results['ols']['test_metrics']['r2']:.4f}")

    alphas = np.logspace(-4, 3, 29)
    names = poly.get_feature_names_out(FEATURE_NAMES).tolist()
    results["ridge"] = run_grid(sub, "ridge", alphas, f"{tag}_ridge",
                                feature_names=names)
    results["lasso"] = run_grid(sub, "lasso", np.logspace(-4, 0, 25),
                                f"{tag}_lasso", feature_names=names)
    # 记录 Lasso 选出的非零特征，便于报告中讨论特征选择
    nz = sorted(((v, k) for k, v in results["lasso"]["coef_best"].items()
                 if abs(v) > 1e-8), key=lambda t: -abs(t[0]))
    results["lasso"]["selected"] = [{"feature": k, "coef": v} for v, k in nz]
    return results


def run_cv(args):
    """5 折交叉验证：每折内部独立标准化并重新以验证集选参。"""
    X, y = load_data(args.seed)["raw"]
    k = args.cv
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    ridge_alphas = np.logspace(-4, 3, 29)
    lasso_alphas = np.logspace(-4, 0, 25)
    out = {"tag": args.tag, "model": "ridge+lasso", "cv": k, "folds": {}}
    t0 = time.time()
    for model in ("ridge", "lasso"):
        fold_rmse, fold_r2, fold_alpha = [], [], []
        alphas = ridge_alphas if model == "ridge" else lasso_alphas
        for i in range(k):
            te = folds[i]
            rest = np.concatenate([folds[j] for j in range(k) if j != i])
            rng2 = np.random.RandomState(args.seed + i)
            va = rng2.choice(rest, size=len(rest) // 8, replace=False)
            tr = np.setdiff1d(rest, va)
            x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
            fold_data = {
                "X_train": (X[tr] - x_mean) / x_std, "y_train": y[tr],
                "X_val": (X[va] - x_mean) / x_std, "y_val": y[va],
                "X_test": (X[te] - x_mean) / x_std, "y_test": y[te],
            }
            grid = []
            w_init = None
            for alpha in sorted(alphas, reverse=(model == "lasso")):
                mdl, _ = fit_model(model, fold_data["X_train"],
                                   fold_data["y_train"], alpha, w_init)
                w_init = mdl.coef_
                grid.append({"alpha": alpha,
                             "val_rmse": regress_metrics(
                                 fold_data["y_val"],
                                 mdl.predict(fold_data["X_val"]))["rmse"]})
            best = min(grid, key=lambda d: d["val_rmse"])
            mdl, _ = fit_model(model, fold_data["X_train"],
                               fold_data["y_train"], best["alpha"])
            m = regress_metrics(fold_data["y_test"],
                                mdl.predict(fold_data["X_test"]))
            fold_rmse.append(m["rmse"])
            fold_r2.append(m["r2"])
            fold_alpha.append(best["alpha"])
            print(f"[{args.tag}] {model} fold {i + 1}/{k}: alpha={best['alpha']:g}  "
                  f"RMSE={m['rmse']:.3f}k$  R2={m['r2']:.4f}")
        out["folds"][model] = {
            "fold_alpha": fold_alpha,
            "fold_rmse": fold_rmse, "fold_r2": fold_r2,
            "rmse_mean": float(np.mean(fold_rmse)),
            "rmse_std": float(np.std(fold_rmse)),
            "r2_mean": float(np.mean(fold_r2)),
            "r2_std": float(np.std(fold_r2)),
        }
        print(f"[{args.tag}] {model} {k}-fold CV: RMSE={out['folds'][model]['rmse_mean']:.3f}±"
              f"{out['folds'][model]['rmse_std']:.3f}k$  "
              f"R2={out['folds'][model]['r2_mean']:.4f}±{out['folds'][model]['r2_std']:.4f}")
    out["time_sec"] = time.time() - t0
    return out


def save_json(result):
    out_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{result['tag']}.json"), "w") as f:
        json.dump(result, f, indent=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ridge",
                   choices=["linear", "ridge", "lasso", "poly", "both"])
    p.add_argument("--tag", default=None)
    p.add_argument("--alphas", type=float, nargs="+", default=None)
    p.add_argument("--cv", type=int, default=0, help=">0 时执行 k 折交叉验证")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.tag is None:
        args.tag = f"boston_{args.model}" + ("_cv" if args.cv > 0 else "")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cv > 0:
        save_json(run_cv(args))
    else:
        data = load_data(args.seed)
        checks = check_against_sklearn(data)
        t0 = time.time()
        if args.model == "linear":
            result = run_linear(data, args.tag)
        elif args.model == "poly":
            result = run_poly(data, args.tag)
        else:
            if args.model == "both":
                result = {"tag": args.tag}
                for model in ("ridge", "lasso"):
                    default = (np.logspace(-4, 3, 29) if model == "ridge"
                               else np.logspace(-4, 0, 25))
                    alphas = np.array(args.alphas) if args.alphas else default
                    result[model] = run_grid(data, model, alphas,
                                             f"{args.tag}_{model}")
            else:
                default = (np.logspace(-4, 3, 29) if args.model == "ridge"
                           else np.logspace(-4, 0, 25))
                alphas = np.array(args.alphas) if args.alphas else default
                result = run_grid(data, args.model, alphas, args.tag)
        result["sklearn_check"] = checks
        result["time_sec"] = time.time() - t0
        save_json(result)
