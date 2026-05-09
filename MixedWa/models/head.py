# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.base import HeadBase


class LinearClassifier(HeadBase):
    """
    线性分类头：GAP 输出 → Dropout → Linear → logits
    支持单标签（softmax + CE）和多标签（sigmoid + BCE）两种模式。
    """
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class MLPProjector(HeadBase):
    """
    MLP 投影头，用于对比学习（WaPIRL 阶段三）：
    Linear → BN → ReLU → Linear → L2 归一化
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=1)
