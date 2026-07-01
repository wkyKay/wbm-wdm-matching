# -*- coding: utf-8 -*-
"""CSV manifest helpers for shared WM38K splits and query sets."""

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .io import CLASS_NAMES, signature_string


SPLIT_COLUMNS = [
    'sample_id',
    'original_index',
    'valid_index',
    'split',
    'label_signature',
] + [f'label_{i}' for i in range(len(CLASS_NAMES))]


QUERY_COLUMNS = [
    'sample_id',
    'original_index',
    'valid_index',
    'label_signature',
] + [f'label_{i}' for i in range(len(CLASS_NAMES))]


def write_split_manifest(path, labels: np.ndarray, original_indices: np.ndarray, split_indices: Dict[str, List[int]]):
    rows = []
    for split_name in ('train', 'valid', 'test'):
        for valid_index in split_indices[split_name]:
            label = labels[int(valid_index)].astype(np.int32)
            original_index = int(original_indices[int(valid_index)])
            row = {
                'sample_id': original_index,
                'original_index': original_index,
                'valid_index': int(valid_index),
                'split': split_name,
                'label_signature': signature_string(label),
            }
            row.update({f'label_{i}': int(label[i]) for i in range(len(CLASS_NAMES))})
            rows.append(row)
    _write_csv(path, SPLIT_COLUMNS, rows)


def write_query_manifest(path, labels: np.ndarray, original_indices: np.ndarray, valid_indices: Iterable[int]):
    rows = []
    for valid_index in valid_indices:
        label = labels[int(valid_index)].astype(np.int32)
        original_index = int(original_indices[int(valid_index)])
        row = {
            'sample_id': original_index,
            'original_index': original_index,
            'valid_index': int(valid_index),
            'label_signature': signature_string(label),
        }
        row.update({f'label_{i}': int(label[i]) for i in range(len(CLASS_NAMES))})
        rows.append(row)
    _write_csv(path, QUERY_COLUMNS, rows)


def load_split_manifest(path, split: str = None) -> List[dict]:
    rows = _read_csv(path)
    if split is not None and split != 'all':
        rows = [row for row in rows if row['split'] == split]
    return rows


def load_query_ids(path) -> List[int]:
    return [int(row['sample_id']) for row in _read_csv(path)]


def manifest_valid_indices(path, split: str = None) -> np.ndarray:
    rows = load_split_manifest(path, split=split)
    return np.asarray([int(row['valid_index']) for row in rows], dtype=np.int64)


def manifest_original_indices(path, split: str = None) -> np.ndarray:
    rows = load_split_manifest(path, split=split)
    return np.asarray([int(row['original_index']) for row in rows], dtype=np.int64)


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with Path(path).open('r', newline='') as f:
        return list(csv.DictReader(f))

