# -*- coding: utf-8 -*-
"""训练 / 评估脚本：在 MNIST 上训练 MLPNN 分类器并记录日志。

用法示例：
    python train_mnist.py --hidden 256 --tag mlp_1x256 --epochs 15
    python train_mnist.py --hidden 512,256 --dropout 0.3 --weight-decay 5e-4 \
        --augment --lr-sched cosine --tag mlp_full --epochs 15
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from mlp import build_mlp, parse_hidden

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
# 若 hw1 已下载 MNIST 则直接复用，避免重复下载
if not os.path.exists(os.path.join(DATA_DIR, "MNIST")):
    hw1_data = os.path.join(BASE_DIR, "..", "hw1", "data")
    if os.path.exists(os.path.join(hw1_data, "MNIST")):
        DATA_DIR = hw1_data


def get_dataloaders(batch_size, augment, num_workers=4):
    mean, std = 0.1307, 0.3081
    train_tf = [torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean,), (std,))]
    if augment:
        train_tf.insert(1, torchvision.transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    tf_train = torchvision.transforms.Compose(train_tf)
    tf_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((mean,), (std,)),
    ])
    train_set = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf_train)
    test_set = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf_test)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1024, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            preds_all.append(pred.cpu().numpy())
            labels_all.append(y.cpu().numpy())
    return correct / total, np.concatenate(preds_all), np.concatenate(labels_all)


def train_one_exp(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader, test_loader = get_dataloaders(args.batch_size, args.augment)
    hidden = parse_hidden(args.hidden)
    model = build_mlp("mnist", hidden, activation=args.activation,
                      dropout=args.dropout, use_bn=args.use_bn).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] hidden={hidden}, activation={args.activation}, "
          f"params={n_params:,}, device={device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.lr_sched == "cosine" else None)

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "lr": []}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * y.numel()
            ep_correct += (out.argmax(1) == y).sum().item()
            ep_total += y.numel()
        if scheduler is not None:
            scheduler.step()
        train_loss = ep_loss / ep_total
        train_acc = ep_correct / ep_total
        test_acc, preds, labels = evaluate(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"[{args.tag}] epoch {epoch:02d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"test_acc={test_acc:.4f}")
    elapsed = time.time() - t0
    final_acc, preds, labels = evaluate(model, test_loader, device)

    result = {
        "tag": args.tag,
        "task": "mnist",
        "hidden": hidden,
        "activation": args.activation,
        "dropout": args.dropout,
        "use_bn": bool(args.use_bn),
        "augment": bool(args.augment),
        "lr_sched": args.lr_sched,
        "epochs": args.epochs,
        "params": n_params,
        "history": history,
        "best_test_acc": max(history["test_acc"]),
        "final_test_acc": final_acc,
        "time_sec": elapsed,
    }

    # 混淆矩阵
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    result["confusion_matrix"] = cm.tolist()

    out_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{args.tag}] done in {elapsed:.1f}s, best_test_acc={result['best_test_acc']:.4f}")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", default="512,256", help="隐藏层，如 '512,256'")
    p.add_argument("--activation", default="relu",
                   choices=["relu", "lrelu", "tanh", "sigmoid"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use-bn", action="store_true")
    p.add_argument("--tag", default=None, help="实验标签（输出文件名）")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--lr-sched", default="none", choices=["none", "cosine"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.tag is None:
        args.tag = "mlp_" + "x".join(args.hidden.split(","))
    return args


if __name__ == "__main__":
    train_one_exp(parse_args())
