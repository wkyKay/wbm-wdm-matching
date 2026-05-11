# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def _get_labels(dataset) -> np.ndarray:
    """从 dataset.samples 中提取整数标签数组。"""
    return np.array([s[1] for s in dataset.samples], dtype=np.int64)


def balanced_loader(dataset, batch_size: int, num_workers: int = 0,
                    shuffle: bool = False,
                    max_per_class: int = None,
                    **kwargs) -> DataLoader:
    """
    类别平衡采样 DataLoader：少数类过采样，多数类下采样。

    每类目标样本数 = min(该类实际数量上限, max_per_class)，
    上限取各类样本数的中位数（若未指定 max_per_class）。
    总采样数 = num_classes × target_per_class。

    Args:
        dataset:       需要有 .samples 属性，每个元素为 (data, label_int)
        batch_size:    批大小
        num_workers:   DataLoader 工作进程数
        max_per_class: 每类最多采样数，None 时取各类数量的中位数
    """
    labels = _get_labels(dataset)
    class_counts = np.bincount(labels)
    num_classes = len(class_counts)

    if max_per_class is None:
        # 取各类数量的中位数作为目标，避免多数类主导
        max_per_class = int(np.median(class_counts))

    # 每类目标采样数：少数类过采样到 max_per_class，多数类下采样到 max_per_class
    target_per_class = np.minimum(class_counts, max_per_class).astype(np.float64)

    # 每个样本的采样权重 = 该类目标数 / 该类实际数
    weights = target_per_class[labels] / class_counts[labels]
    num_samples = int(target_per_class.sum())

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=num_samples,
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
