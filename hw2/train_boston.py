# -*- coding: utf-8 -*-
"""训练 / 评估脚本：在波士顿房价数据集上训练 MLPNN 回归模型。

用法示例：
    python train_boston.py --model linear --tag boston_linear
    python train_boston.py --model mlp --hidden 64,32 --dropout 0.1 \
        --weight-decay 1e-4 --patience 60 --tag boston_mlp
    python train_boston.py --model mlp --hidden 64,32 --dropout 0.1 \
        --weight-decay 1e-4 --cv 5 --tag boston_cv
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from mlp import build_mlp, parse_hidden

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "boston.csv")

FEATURE_NAMES = ["crim", "zn", "indus", "chas", "nox", "rm", "age",
                 "dis", "rad", "tax", "ptratio", "b", "lstat"]


def load_data(seed=42):
    """读入 boston.csv 并划分 train/val/test = 64%/16%/20%，按训练集统计量标准化。"""
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
    data = {
        "Xtr, ytr": None,
        "X_train": Xs[tr], "y_train": ys[tr], "X_val": Xs[va], "y_val": ys[va],
        "X_test": Xs[te], "y_test": ys[te],
        "y_mean": y_mean, "y_std": y_std,
        "raw": (X, y),
    }
    data.pop("Xtr, ytr")
    return data


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
    return ys * data["y_std"] + data["y_mean"]


def train_mlp(args, X_train, y_train, X_val, y_val, X_test, y_test,
              y_mean, y_std, verbose=True, return_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    hidden = parse_hidden(args.hidden)
    model = build_mlp("boston", hidden, activation=args.activation,
                      dropout=args.dropout, use_bn=args.use_bn).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[{args.tag}] hidden={hidden}, params={n_params:,}, device={device}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    t = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    Xtr, ytr, Xva, yva, Xte, yte = map(t, (X_train, y_train, X_val, y_val, X_test, y_test))

    history = {"train_loss": [], "val_rmse": []}
    best_val, best_state, best_epoch, wait = np.inf, None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # 全批量 + 内部 shuffle 小批量
        model.train()
        perm = torch.randperm(Xtr.size(0), device=device)
        ep_loss = 0.0
        for i in range(0, Xtr.size(0), args.batch_size):
            b = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[b]), ytr[b])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * b.numel()
        model.eval()
        with torch.no_grad():
            val_rmse = criterion(model(Xva), yva).sqrt().item() * float(y_std[0])
        history["train_loss"].append(ep_loss / Xtr.size(0))
        history["val_rmse"].append(val_rmse)
        if val_rmse < best_val - 1e-4:
            best_val, best_epoch, wait = val_rmse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"[{args.tag}] epoch {epoch:04d}/{args.epochs}  "
                  f"train_mse={history['train_loss'][-1]:.4f}  val_rmse={val_rmse:.3f}k$")
        if wait >= args.patience:
            if verbose:
                print(f"[{args.tag}] early stop at epoch {epoch} (best={best_epoch})")
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test = model(Xte).cpu().numpy()
    metrics = regress_metrics(to_raw(yte.cpu().numpy(), {"y_std": y_std, "y_mean": y_mean}),
                              to_raw(pred_test, {"y_std": y_std, "y_mean": y_mean}))
    metrics["best_val_rmse"] = float(best_val)
    metrics["best_epoch"] = int(best_epoch)
    metrics["epochs_run"] = len(history["train_loss"])
    metrics["time_sec"] = time.time() - t0
    if return_pred:
        return model, history, metrics, to_raw(yte.cpu().numpy(), {"y_std": y_std, "y_mean": y_mean}), \
               to_raw(pred_test, {"y_std": y_std, "y_mean": y_mean}), n_params
    return model, history, metrics, n_params


def run_linear(args, data):
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    reg = LinearRegression().fit(X_train, y_train)
    pred_raw = to_raw(reg.predict(X_test), data)
    metrics = regress_metrics(to_raw(y_test, data), pred_raw)
    result = {
        "tag": args.tag, "task": "boston", "model": "linear",
        "test_metrics": metrics,
        "coef": dict(zip(FEATURE_NAMES, [float(c) for c in reg.coef_.ravel()])),
        "y_test": [float(v) for v in to_raw(y_test, data).ravel()],
        "y_pred": [float(v) for v in pred_raw.ravel()],
    }
    print(f"[{args.tag}] linear: test RMSE={metrics['rmse']:.3f}k$, R2={metrics['r2']:.4f}")
    return result


def run_mlp(args, data):
    model, history, metrics, y_true, y_pred, n_params = train_mlp(
        args, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        data["X_test"], data["y_test"], data["y_mean"], data["y_std"],
        return_pred=True)
    result = {
        "tag": args.tag, "task": "boston", "model": "mlp",
        "hidden": parse_hidden(args.hidden), "activation": args.activation,
        "dropout": args.dropout, "use_bn": bool(args.use_bn),
        "weight_decay": args.weight_decay, "lr": args.lr,
        "params": n_params, "history": history, "test_metrics": metrics,
        "y_test": [float(v) for v in np.asarray(y_true).ravel()],
        "y_pred": [float(v) for v in np.asarray(y_pred).ravel()],
    }
    print(f"[{args.tag}] test RMSE={metrics['rmse']:.3f}k$  MAE={metrics['mae']:.3f}k$  "
          f"R2={metrics['r2']:.4f}  (best val RMSE={metrics['best_val_rmse']:.3f}k$ "
          f"@epoch {metrics['best_epoch']})")
    return result


def run_cv(args):
    """在全数据上做 k 折交叉验证（每折内部按训练折统计量标准化，10% 训练折作验证）。"""
    data_all = load_data(args.seed)["raw"]
    X, y = data_all
    k = args.cv
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    fold_rmse, fold_r2 = [], []
    t0 = time.time()
    for i in range(k):
        te = folds[i]
        rest = np.concatenate([folds[j] for j in range(k) if j != i])
        rng2 = np.random.RandomState(args.seed + i)
        va = rng2.choice(rest, size=len(rest) // 8, replace=False)  # ~12.5% of rest as val
        tr = np.setdiff1d(rest, va)
        x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
        y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-8
        _, _, metrics, _ = train_mlp(
            args, (X[tr] - x_mean) / x_std, (y[tr] - y_mean) / y_std,
            (X[va] - x_mean) / x_std, (y[va] - y_mean) / y_std,
            (X[te] - x_mean) / x_std, (y[te] - y_mean) / y_std,
            y_mean, y_std, verbose=False)
        fold_rmse.append(metrics["rmse"])
        fold_r2.append(metrics["r2"])
        print(f"[{args.tag}] fold {i + 1}/{k}: RMSE={metrics['rmse']:.3f}k$  R2={metrics['r2']:.4f}")
    result = {
        "tag": args.tag, "task": "boston", "model": "mlp", "cv": k,
        "hidden": parse_hidden(args.hidden), "dropout": args.dropout,
        "weight_decay": args.weight_decay, "lr": args.lr,
        "fold_rmse": fold_rmse, "fold_r2": fold_r2,
        "rmse_mean": float(np.mean(fold_rmse)), "rmse_std": float(np.std(fold_rmse)),
        "r2_mean": float(np.mean(fold_r2)), "r2_std": float(np.std(fold_r2)),
        "time_sec": time.time() - t0,
    }
    print(f"[{args.tag}] {k}-fold CV: RMSE={result['rmse_mean']:.3f}±{result['rmse_std']:.3f}k$  "
          f"R2={result['r2_mean']:.4f}±{result['r2_std']:.4f}")
    return result


def save_json(result):
    out_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{result['tag']}.json"), "w") as f:
        json.dump(result, f, indent=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlp", choices=["linear", "mlp"])
    p.add_argument("--hidden", default="64,32", help="隐藏层，如 '64,32'")
    p.add_argument("--activation", default="relu",
                   choices=["relu", "lrelu", "tanh", "sigmoid"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use-bn", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=60, help="早停耐心（epoch）")
    p.add_argument("--cv", type=int, default=0, help=">0 时执行 k 折交叉验证")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.tag is None:
        args.tag = f"boston_{args.model}"
    return args


def dump_data_stats(seed=42):
    """保存数据集基本统计信息（供报告使用）。"""
    data = load_data(seed)
    X, y = data["raw"]
    stats = {
        "n_samples": int(len(X)), "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "n_train": int(len(data["X_train"])), "n_val": int(len(data["X_val"])),
        "n_test": int(len(data["X_test"])),
        "y_min": float(y.min()), "y_max": float(y.max()), "y_mean": float(y.mean()),
    }
    save_json({**stats, "tag": "boston_data"})
    print(f"[data] n={stats['n_samples']}, features={stats['n_features']}, "
          f"train/val/test={stats['n_train']}/{stats['n_val']}/{stats['n_test']}")


if __name__ == "__main__":
    args = parse_args()
    if args.cv > 0:
        save_json(run_cv(args))
    else:
        data = load_data(args.seed)
        if args.model == "linear":
            save_json(run_linear(args, data))
        else:
            save_json(run_mlp(args, data))
    dump_data_stats(args.seed)
