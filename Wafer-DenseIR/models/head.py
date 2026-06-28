# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_channels: int, num_features: int, dropout: float = 0.0):
        super(LinearHead, self).__init__()
        self.num_features = num_features
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, num_features),
        )

    def forward(self, x: torch.FloatTensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPHead(nn.Module):
    def __init__(self, in_channels: int, num_features: int):
        super(MLPHead, self).__init__()
        self.num_features = num_features
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_features),
        )

    def forward(self, x: torch.FloatTensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
