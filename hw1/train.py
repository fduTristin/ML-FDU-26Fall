# -*- coding: utf-8 -*-
"""训练 / 评估脚本：在 MNIST 上训练 CNN 并记录日志与图表。

用法示例：
    python train.py --model baseline --epochs 10
    python train.py --model vggstyle --epochs 10 --augment --lr-sched cosine
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_model

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


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
            out = model(x)
            pred = out.argmax(1)
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
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] model={args.model}, params={n_params:,}, device={device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.lr_sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

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
        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr_now)
        print(f"[{args.tag}] epoch {epoch:02d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"test_acc={test_acc:.4f}  lr={lr_now:.5f}")
    elapsed = time.time() - t0
    final_acc, preds, labels = evaluate(model, test_loader, device)

    result = {
        "tag": args.tag,
        "model": args.model,
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

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{args.tag}] done in {elapsed:.1f}s, best_test_acc={result['best_test_acc']:.4f}")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vggstyle",
                   choices=["baseline", "vggstyle", "vggstyle_wobn"])
    p.add_argument("--tag", default=None, help="实验标签（输出文件名）")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--lr-sched", default="none", choices=["none", "cosine"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.tag is None:
        args.tag = args.model
        if args.augment:
            args.tag += "_aug"
        if args.lr_sched != "none":
            args.tag += "_" + args.lr_sched
    return args


if __name__ == "__main__":
    train_one_exp(parse_args())
