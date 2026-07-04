# -*- coding: utf-8 -*-
"""Small cluster patch encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_dim: int = 256, width: int = 32):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.net = nn.Sequential(
            _block(in_channels, width),
            nn.MaxPool2d(2),
            _block(width, width * 2),
            nn.MaxPool2d(2),
            _block(width * 2, width * 4),
            nn.MaxPool2d(2),
            _block(width * 4, width * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(width * 4, self.embedding_dim)

    def forward(self, x):
        z = self.net(x).flatten(1)
        z = self.fc(z)
        return F.normalize(z, dim=1)


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

