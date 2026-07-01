# -*- coding: utf-8 -*-
"""Shared MixedWM38K loading utilities."""

from collections import Counter
from typing import Tuple

import numpy as np


CLASS_NAMES = [
    'center',
    'donut',
    'edge-loc',
    'edge-ring',
    'loc',
    'random',
    'scratch',
    'near-full',
]


def load_valid_wm38k(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    maps = _pick_array(data, ('maps', 'x', 'X', 'images', 'arr_0'))
    labels = _pick_array(data, ('labels', 'y', 'Y', 'targets', 'arr_1'))
    valid_mask = labels.sum(axis=1) > 0
    original_indices = np.where(valid_mask)[0].astype(np.int64)
    return maps[valid_mask], labels[valid_mask].astype(np.int32), original_indices


def label_signature(label_vec) -> tuple:
    return tuple(np.where(np.asarray(label_vec).astype(np.int32) == 1)[0].tolist())


def signature_string(label_vec) -> str:
    sig = label_signature(label_vec)
    return '|'.join(str(x) for x in sig)


def signature_counts(labels: np.ndarray) -> Counter:
    return Counter(label_signature(label) for label in labels)


def _pick_array(npz, names):
    for name in names:
        if name in npz.files:
            return npz[name]
    raise KeyError(f'None of {names} found in npz file. Available keys: {npz.files}')

