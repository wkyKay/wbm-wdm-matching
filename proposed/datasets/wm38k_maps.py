# -*- coding: utf-8 -*-
"""WM38K loading helpers for the proposed method."""

from __future__ import annotations

import numpy as np

from partial_match.data.data_io import filter_valid_samples, load_wm38k
from shared.wm38k.manifest import load_split_manifest


def load_valid_wm38k(data_file: str):
    maps, labels = load_wm38k(data_file)
    return filter_valid_samples(maps, labels)


def load_split_records(data_file: str, split_manifest: str = None, split: str = 'train'):
    maps, labels, original_indices = load_valid_wm38k(data_file)
    if split_manifest:
        rows = load_split_manifest(split_manifest, split=split)
        valid_indices = np.asarray([int(row['valid_index']) for row in rows], dtype=np.int64)
        maps = maps[valid_indices]
        labels = labels[valid_indices]
        original_indices = original_indices[valid_indices]
    return [
        {'map_id': int(original_indices[i]), 'raw_map': maps[i], 'label': labels[i]}
        for i in range(len(maps))
    ]


def records_by_id(records):
    return {int(record['map_id']): record for record in records}

