# -*- coding: utf-8 -*-
"""Shared stratified split generation for MixedWM38K."""

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from .io import label_signature


def stratified_split_by_signature(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 2026,
) -> Dict[str, List[int]]:
    total = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f'Split ratios must sum to 1.0, got {total}')

    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for valid_index, label in enumerate(labels):
        groups[label_signature(label)].append(valid_index)

    split_indices = {'train': [], 'valid': [], 'test': []}
    for _, indices in sorted(groups.items(), key=lambda kv: kv[0]):
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(np.floor(n * train_ratio))
        n_valid = int(np.floor(n * valid_ratio))
        n_test = n - n_train - n_valid

        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_valid == 0:
                n_valid = 1
            if n_test == 0:
                n_test = 1
            while n_train + n_valid + n_test > n:
                if n_train >= n_valid and n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_valid >= n_test and n_valid > 1:
                    n_valid -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break

        split_indices['train'].extend(_as_ints(shuffled[:n_train]))
        split_indices['valid'].extend(_as_ints(shuffled[n_train:n_train + n_valid]))
        split_indices['test'].extend(_as_ints(shuffled[n_train + n_valid:]))

    for split_name in split_indices:
        values = np.asarray(split_indices[split_name], dtype=np.int64)
        rng.shuffle(values)
        split_indices[split_name] = _as_ints(values)
    return split_indices


def _as_ints(values: Iterable[int]) -> List[int]:
    return [int(x) for x in values]

