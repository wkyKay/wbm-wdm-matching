# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


def balanced_loader(dataset, batch_size: int, num_workers: int = 0,
                    shuffle: bool = False, **kwargs) -> DataLoader:
    """
    针对单标签数据集的类别平衡采样 DataLoader。
    dataset 需要有 .samples 属性，每个元素为 (path_or_data, label_int)。
    """
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers,
                      drop_last=True, **kwargs)


def standard_loader(dataset, batch_size: int, num_workers: int = 0,
                    shuffle: bool = True, **kwargs) -> DataLoader:
    """标准 DataLoader，用于多标签数据集或验证/测试集。"""
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      drop_last=False, **kwargs)
