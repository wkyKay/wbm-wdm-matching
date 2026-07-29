# -*- coding: utf-8 -*-
"""Cluster patch encoders for contrastive pretraining and retrieval."""

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
        z = self.net(self.forward_features(x)).flatten(1)
        z = self.fc(z)
        return F.normalize(z, dim=1)

    def forward_features(self, x):
        """Return the final spatial feature map for dense retrieval consumers."""
        return self.net[:-1](x)


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResNet18Encoder(nn.Module):
    """ResNet-18 adapted to preserve detail in 64x64 sparse cluster patches."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, self.embedding_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = [_ResidualBlock(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.forward_features(x)
        z = self.pool(z).flatten(1)
        z = self.fc(z)
        return F.normalize(z, dim=1)

    def forward_features(self, x):
        """Return the final spatial feature map before global pooling."""
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        return z


class _ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(z + identity)


def build_encoder(name: str = 'simple', in_channels: int = 3, embedding_dim: int = 256, width: int = 32):
    normalized_name = str(name).lower()
    if normalized_name == 'simple':
        return ClusterEncoder(in_channels=in_channels, embedding_dim=embedding_dim, width=width)
    if normalized_name == 'resnet18':
        return ResNet18Encoder(in_channels=in_channels, embedding_dim=embedding_dim)
    raise ValueError(f'Unknown encoder {name!r}; expected "simple" or "resnet18".')
