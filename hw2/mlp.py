# -*- coding: utf-8 -*-
"""作业2：多层感知机（MLPNN）模型定义。

实现一个通用的全连接多层感知机，同时用于两个任务：
  1. MNIST 手写数字 10 分类（输入 784 维，输出 10 维，配合交叉熵损失）；
  2. 波士顿房价回归（输入 13 维，输出 1 维，配合 MSE 损失）。
"""

import torch.nn as nn

ACTS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """通用多层感知机: Input -> [Linear -> (BN) -> Act -> (Dropout)] x L -> Linear(out)"""

    def __init__(self, input_dim, hidden_sizes, output_dim,
                 activation="relu", dropout=0.0, use_bn=False):
        super().__init__()
        act = ACTS[activation]
        layers = []
        d_in = input_dim
        for d in hidden_sizes:
            layers.append(nn.Linear(d_in, d, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(d))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = d
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:  # 图像输入时展平为向量
            x = x.view(x.size(0), -1)
        return self.net(x)


def build_mlp(task, hidden_sizes, activation="relu", dropout=0.0, use_bn=False):
    """按任务构建 MLP。

    task: "mnist" -> 784 输入、10 输出（分类）
          "boston" -> 13 输入、1 输出（回归）
    """
    if task == "mnist":
        return MLP(784, hidden_sizes, 10, activation, dropout, use_bn)
    if task == "boston":
        return MLP(13, hidden_sizes, 1, activation, dropout, use_bn)
    raise ValueError(f"unknown task: {task}")


def parse_hidden(s):
    """将 '512,256' 形式的字符串解析为隐藏层列表。"""
    s = s.strip()
    return [int(x) for x in s.split(",")] if s else []
