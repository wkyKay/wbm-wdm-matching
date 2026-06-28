# -*- coding: utf-8 -*-
"""
数据集分层 split 模块
"""

import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict

from .data_io import get_label_signature


def split_by_signature(
    labels: np.ndarray,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2026
) -> Dict[str, List[int]]:
    """
    按标签 signature 分层 split。

    Args:
        labels: (N, 8) multi-hot 标签
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        {'train': [indices], 'validation': [indices], 'test': [indices]}
    """
    np.random.seed(seed)

    # 按 signature 分组
    signature_groups = defaultdict(list)
    for idx, label_vec in enumerate(labels):
        sig = get_label_signature(label_vec)
        signature_groups[sig].append(idx)

    train_indices = []
    valid_indices = []
    test_indices = []

    for sig, indices in signature_groups.items():
        n_samples = len(indices)

        # 打乱顺序
        shuffled = np.random.permutation(indices)

        if n_samples >= 10:
            # 样本数足够，按比例 split
            n_train = int(n_samples * train_ratio)
            n_valid = int(n_samples * valid_ratio)
            n_test = n_samples - n_train - n_valid

            train_indices.extend(shuffled[:n_train])
            valid_indices.extend(shuffled[n_train:n_train+n_valid])
            test_indices.extend(shuffled[n_train+n_valid:])
        elif n_samples >= 3:
            # 样本数较少，保证至少 1 个 valid 和 1 个 test
            train_indices.extend(shuffled[:-2])
            valid_indices.append(shuffled[-2])
            test_indices.append(shuffled[-1])
        else:
            # 样本数太少，全部放入 train
            train_indices.extend(shuffled)

    # 再次打乱每个集合的顺序
    train_indices = list(np.random.permutation(train_indices))
    valid_indices = list(np.random.permutation(valid_indices))
    test_indices = list(np.random.permutation(test_indices))

    return {
        'train': train_indices,
        'validation': valid_indices,
        'test': test_indices,
    }


def get_split_info(
    split_indices: Dict[str, List[int]],
    labels: np.ndarray
) -> Dict:
    """
    获取 split 的统计信息。

    Args:
        split_indices: split 结果
        labels: (N, 8) multi-hot 标签

    Returns:
        split 统计信息
    """
    info = {
        'train': {},
        'validation': {},
        'test': {},
    }

    for split_name, indices in split_indices.items():
        split_labels = labels[indices]
        n_samples = len(indices)
        label_sums = split_labels.sum(axis=1)

        # 标签基数统计
        cardinality_counts = {}
        for c in range(1, 5):
            cardinality_counts[c] = (label_sums == c).sum()

        # 每个类别的样本数
        class_counts = {}
        for i in range(8):
            class_counts[i] = int(split_labels[:, i].sum())

        info[split_name] = {
            'n_samples': n_samples,
            'cardinality_counts': cardinality_counts,
            'class_counts': class_counts,
        }

    return info


class DataSplitter:
    """
    数据集分割器，封装数据读取和 split 逻辑。
    """

    def __init__(
        self,
        npz_path: str,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 2026
    ):
        """
        Args:
            npz_path: Wafer_Map_Datasets.npz 文件路径
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.npz_path = npz_path
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self._data_loaded = False
        self._maps = None
        self._labels = None
        self._valid_maps = None
        self._valid_labels = None
        self._original_indices = None
        self._split_indices = None

    def load_data(self):
        """
        加载并过滤数据。
        """
        from .data_io import load_wm38k, filter_valid_samples

        self._maps, self._labels = load_wm38k(self.npz_path)
        self._valid_maps, self._valid_labels, self._original_indices = filter_valid_samples(
            self._maps, self._labels
        )
        self._data_loaded = True

    def split(self):
        """
        执行 split。
        """
        if not self._data_loaded:
            self.load_data()

        self._split_indices = split_by_signature(
            self._valid_labels,
            self.train_ratio,
            self.valid_ratio,
            self.test_ratio,
            self.seed
        )
        return self._split_indices

    def get_split_data(self, split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取某个 split 的数据。

        Args:
            split_name: 'train', 'validation', 或 'test'

        Returns:
            (maps, labels, sample_ids)
            - maps: 晶圆图
            - labels: 标签
            - sample_ids: 样本 ID（过滤后的连续 ID）
        """
        if self._split_indices is None:
            self.split()

        indices = self._split_indices[split_name]
        maps = self._valid_maps[indices]
        labels = self._valid_labels[indices]
        sample_ids = np.array(indices)

        return maps, labels, sample_ids

    @property
    def valid_maps(self):
        if not self._data_loaded:
            self.load_data()
        return self._valid_maps

    @property
    def valid_labels(self):
        if not self._data_loaded:
            self.load_data()
        return self._valid_labels

    @property
    def original_indices(self):
        if not self._data_loaded:
            self.load_data()
        return self._original_indices

    @property
    def split_indices(self):
        if self._split_indices is None:
            self.split()
        return self._split_indices
