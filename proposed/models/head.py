# -*- coding: utf-8 -*-
"""Projection heads for cluster contrastive pretraining."""

import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, hidden_channels: int = None):
        super().__init__()
        hidden_channels = int(hidden_channels or in_channels)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class LinearHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=1)

