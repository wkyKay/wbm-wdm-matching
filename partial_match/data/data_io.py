# -*- coding: utf-8 -*-
"""
MixedWM38K 数据读取模块
"""

import numpy as np
from typing import Tuple, Dict, List


# 类别名称
CLASS_NAMES = [
    'center',
    'donut',
    'edge-loc',
    'edge-ring',
    'loc',
    'random',
    'scratch',
    'near-full'
]


def load_wm38k(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 MixedWM38K 数据集。

    Args:
        npz_path: Wafer_Map_Datasets.npz 文件路径

    Returns:
        (maps, labels)
        - maps: (N, H, W) 晶圆图
        - labels: (N, 8) multi-hot 标签
    """
    data = np.load(npz_path)
    maps = data['arr_0']
    labels = data['arr_1']
    return maps, labels


def filter_valid_samples(
    maps: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    过滤有效样本（有至少一个标签）。

    Args:
        maps: (N, H, W) 原始晶圆图
        labels: (N, 8) 原始标签

    Returns:
        (valid_maps, valid_labels, original_indices)
        - valid_maps: 过滤后的晶圆图
        - valid_labels: 过滤后的标签
        - original_indices: 对应原始数据集的下标
    """
    label_sums = labels.sum(axis=1)
    valid_mask = label_sums > 0

    valid_maps = maps[valid_mask]
    valid_labels = labels[valid_mask]
    original_indices = np.where(valid_mask)[0]

    return valid_maps, valid_labels, original_indices


def get_label_info(labels: np.ndarray) -> Dict:
    """
    获取标签统计信息。

    Args:
        labels: (N, 8) multi-hot 标签

    Returns:
        标签统计字典
    """
    n_samples = len(labels)
    label_sums = labels.sum(axis=1)

    # 标签基数统计
    cardinality_counts = {}
    for c in range(0, 5):
        cardinality_counts[c] = (label_sums == c).sum()

    # 每个类别的样本数
    class_counts = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_counts[class_name] = labels[:, i].sum()

    # 标签 signature 统计
    signatures = []
    signature_counts = {}
    for label_vec in labels:
        sig = tuple(np.where(label_vec == 1)[0])
        signatures.append(sig)
        if sig not in signature_counts:
            signature_counts[sig] = 0
        signature_counts[sig] += 1

    return {
        'n_samples': n_samples,
        'cardinality_counts': cardinality_counts,
        'class_counts': class_counts,
        'signatures': signatures,
        'signature_counts': signature_counts,
    }


def get_label_signature(label_vec: np.ndarray) -> Tuple[int, ...]:
    """
    获取标签 signature。

    Args:
        label_vec: (8,) multi-hot 标签向量

    Returns:
        标签 signature（类别下标的 tuple）
    """
    return tuple(np.where(label_vec == 1)[0])
