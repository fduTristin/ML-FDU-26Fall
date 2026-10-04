# -*- coding: utf-8 -*-
"""作业1：基于 MNIST 的卷积神经网络（CNN）设计与实现。

包含三个网络结构：
  1. NetBaseline：简单 CNN（2 个卷积层 + 池化 + 全连接）
  2. NetVGGStyle：改进的 VGG 风格 CNN（3 个卷积块 + BatchNorm + Dropout）
  3. NetBN：在 VGG 风格基础上去掉 BN（用于消融实验）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetBaseline(nn.Module):
    """简单 CNN: conv(32) -> pool -> conv(64) -> pool -> fc(128) -> fc(10)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14 -> 7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ConvBlock(nn.Module):
    """卷积块: Conv -> ReLU -> Conv -> (BN) -> Pool (+Dropout)"""

    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_bn)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.pool(F.relu(x))
        if self.drop is not None:
            x = self.drop(x)
        return x


class NetVGGStyle(nn.Module):
    """VGG 风格 CNN: 3 个卷积块 + BN + Dropout。

    28x28 -> 14x14 -> 7x7 -> 3x3
    """

    def __init__(self, use_bn=True, dropout=0.3):
        super().__init__()
        self.block1 = ConvBlock(1, 32, use_bn)
        self.block2 = ConvBlock(32, 64, use_bn)
        self.block3 = ConvBlock(64, 128, use_bn, dropout=dropout)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.block1(x)  # 28 -> 14
        x = self.block2(x)  # 14 -> 7
        x = self.block3(x)  # 7 -> 3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def build_model(name: str) -> nn.Module:
    if name == "baseline":
        return NetBaseline()
    if name == "vggstyle":
        return NetVGGStyle(use_bn=True)
    if name == "vggstyle_wobn":
        return NetVGGStyle(use_bn=False)
    raise ValueError(f"unknown model name: {name}")
