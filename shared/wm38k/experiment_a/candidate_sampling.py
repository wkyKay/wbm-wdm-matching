# -*- coding: utf-8 -*-
"""Experiment A per-query candidate pool sampling."""

from __future__ import annotations

from typing import List

import numpy as np

from shared.wm38k.io import CLASS_NAMES
from shared.wm38k.manifest import load_query_ids, load_split_manifest


def build_stratified_candidate_rows(
    split_manifest,
    query_manifest,
    split='test',
    seed=2026,
    exact=50,
    high=100,
    weak=100,
    none=750,
    total=1000,
) -> List[dict]:
    rows = load_split_manifest(split_manifest, split=split)
    query_ids = load_query_ids(query_manifest)
    label_by_id = _labels_by_id(rows)
    signature_by_id = {int(row['sample_id']): row['label_signature'] for row in rows}
    candidate_ids = [int(row['sample_id']) for row in rows]
    rng = np.random.default_rng(seed)
    out = []
    for query_id in query_ids:
        if query_id not in label_by_id:
            raise ValueError(f'Query id {query_id} is not in split manifest split={split}')
        buckets = {'exact': [], 'high_overlap': [], 'weak_overlap': [], 'no_overlap': []}
        q_label = label_by_id[query_id]
        for candidate_id in candidate_ids:
            if candidate_id == query_id:
                continue
            value = _jaccard(q_label, label_by_id[candidate_id])
            if value >= 1.0:
                bucket = 'exact'
            elif value >= 0.5:
                bucket = 'high_overlap'
            elif value > 0.0:
                bucket = 'weak_overlap'
            else:
                bucket = 'no_overlap'
            buckets[bucket].append((candidate_id, value))
        selected = []
        budgets = [
            ('exact', exact),
            ('high_overlap', high),
            ('weak_overlap', weak),
            ('no_overlap', none),
        ]
        for bucket, count in budgets:
            selected.extend(_sample_bucket(buckets[bucket], count, rng))
        selected_ids = {candidate_id for candidate_id, _ in selected}
        if total is not None and len(selected) < total:
            leftovers = []
            for bucket in ('exact', 'high_overlap', 'weak_overlap', 'no_overlap'):
                leftovers.extend([item for item in buckets[bucket] if item[0] not in selected_ids])
            selected.extend(_sample_bucket(leftovers, total - len(selected), rng))
        if total is not None and len(selected) > total:
            selected = _sample_bucket(selected, total, rng)
        selected.sort(key=lambda item: (-item[1], item[0]))
        for candidate_id, value in selected:
            out.append({
                'query_id': int(query_id),
                'candidate_id': int(candidate_id),
                'bucket': _bucket_from_jaccard(value),
                'jaccard': f'{value:.8f}',
                'query_signature': signature_by_id[int(query_id)],
                'candidate_signature': signature_by_id[int(candidate_id)],
            })
    return out


def _labels_by_id(rows):
    out = {}
    for row in rows:
        out[int(row['sample_id'])] = np.asarray([int(row[f'label_{i}']) for i in range(len(CLASS_NAMES))], dtype=np.int32)
    return out


def _jaccard(a, b):
    inter = int(np.dot(a, b))
    union = int(a.sum() + b.sum() - inter)
    return float(inter / max(union, 1))


def _bucket_from_jaccard(value):
    if value >= 1.0:
        return 'exact'
    if value >= 0.5:
        return 'high_overlap'
    if value > 0.0:
        return 'weak_overlap'
    return 'no_overlap'


def _sample_bucket(items, count, rng):
    if count is None or count >= len(items):
        return list(items)
    indices = rng.choice(len(items), size=count, replace=False)
    return [items[int(i)] for i in indices]
