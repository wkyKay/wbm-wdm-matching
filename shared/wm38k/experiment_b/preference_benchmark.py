# -*- coding: utf-8 -*-
"""Build method-independent Experiment B benchmark manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from shared.wm38k.io import CLASS_NAMES, load_valid_wm38k, signature_string
from shared.wm38k.manifest import load_split_manifest
from .transforms import defect_mask, transform_map


SYNTHETIC_TRANSFORMS = [
    ('identity', 'none'),
    ('rot_90', 'moderate'),
    ('rot_180', 'moderate'),
    ('shift_mild', 'mild'),
    ('shift_strong', 'strong'),
    ('scale_mild', 'mild'),
    ('noise_mild', 'mild'),
    ('noise_strong', 'strong'),
    ('dropout_mild', 'mild'),
    ('dropout_strong', 'strong'),
    ('cluster_extra', 'moderate'),
    ('cluster_dropout', 'moderate'),
]


NEGATIVE_TYPES = [
    'easy_diff_label_random',
    'easy_diff_label_random',
    'same_area_wrong_shape',
    'same_area_wrong_shape',
    'same_position_wrong_shape',
    'same_position_wrong_shape',
    'same_label_hard_negative',
    'diff_label_similar_morphology',
]


def build_experiment_b(
    data_file: str,
    split_manifest: str,
    out_dir: str,
    split: str = 'test',
    num_queries: int = 1000,
    seed: int = 2026,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    maps, labels, original_indices = load_valid_wm38k(data_file)
    split_rows = load_split_manifest(split_manifest, split=split)
    test_valid_indices = np.asarray([int(row['valid_index']) for row in split_rows], dtype=np.int64)
    query_valid_indices = _stratified_queries(labels, test_valid_indices, num_queries, rng)

    benchmark_maps = []
    benchmark_labels = []
    split_manifest_rows = []
    query_rows = []
    candidate_rows = []
    preference_rows = []
    source_rows = []
    next_id = 0

    for q_order, valid_idx in enumerate(query_valid_indices):
        source_id = int(original_indices[int(valid_idx)])
        query_new_id = next_id
        next_id += 1
        raw = maps[int(valid_idx)]
        label = labels[int(valid_idx)].astype(np.int32)
        benchmark_maps.append(raw)
        benchmark_labels.append(label)
        split_manifest_rows.append(_split_row(query_new_id, source_id, 'test', label, row_type='query'))
        query_rows.append(_query_row(query_new_id, source_id, label))
        source_rows.append(_source_row(query_new_id, source_id, 'query', 'query', 'none', 'none', q_order))

        synthetic_ids = {}
        for transform_type, strength in SYNTHETIC_TRANSFORMS:
            candidate_id = next_id
            next_id += 1
            variant_rng = np.random.default_rng(seed + q_order * 1009 + candidate_id)
            variant = transform_map(raw, transform_type, variant_rng)
            benchmark_maps.append(variant)
            benchmark_labels.append(label)
            split_manifest_rows.append(_split_row(candidate_id, source_id, 'test', label, row_type='synthetic'))
            candidate_rows.append(_candidate_row(
                query_new_id, candidate_id, 'synthetic', source_id, transform_type, strength, '', label
            ))
            source_rows.append(_source_row(candidate_id, source_id, 'synthetic', 'synthetic', transform_type, strength, q_order))
            synthetic_ids[transform_type] = candidate_id

        negative_ids = {}
        negatives = _sample_negatives(
            query_valid_idx=int(valid_idx),
            all_valid_indices=test_valid_indices,
            maps=maps,
            labels=labels,
            original_indices=original_indices,
            rng=rng,
        )
        for negative_type, neg_valid_idx in negatives:
            candidate_id = next_id
            next_id += 1
            neg_label = labels[int(neg_valid_idx)].astype(np.int32)
            neg_source_id = int(original_indices[int(neg_valid_idx)])
            benchmark_maps.append(maps[int(neg_valid_idx)])
            benchmark_labels.append(neg_label)
            split_manifest_rows.append(_split_row(candidate_id, neg_source_id, 'test', neg_label, row_type='real_negative'))
            candidate_rows.append(_candidate_row(
                query_new_id, candidate_id, 'real_negative', neg_source_id, 'none', 'none', negative_type, neg_label
            ))
            source_rows.append(_source_row(candidate_id, neg_source_id, 'real_negative', negative_type, 'none', 'none', q_order))
            negative_ids.setdefault(negative_type, []).append(candidate_id)

        preference_rows.extend(_preference_rows(query_new_id, synthetic_ids, negative_ids))

    map_array = np.asarray(benchmark_maps, dtype=np.uint8)
    label_array = np.asarray(benchmark_labels, dtype=np.int32)
    np.savez_compressed(
        out_dir / 'b_data.npz',
        maps=map_array,
        labels=label_array,
        arr_0=map_array,
        arr_1=label_array,
    )
    _write_csv(out_dir / 'b_split_manifest.csv', _split_columns(), split_manifest_rows)
    _write_csv(out_dir / 'b_queries.csv', _query_columns(), query_rows)
    _write_csv(out_dir / 'b_candidates.csv', _candidate_columns(), candidate_rows)
    _write_csv(out_dir / 'b_preferences.csv', _preference_columns(), preference_rows)
    _write_csv(out_dir / 'b_sources.csv', _source_columns(), source_rows)
    config = {
        'protocol': 'Experiment B: transformation-derived preference accuracy',
        'data_file': data_file,
        'source_split_manifest': split_manifest,
        'split': split,
        'seed': seed,
        'num_queries_requested': num_queries,
        'num_queries': len(query_rows),
        'synthetic_transforms': SYNTHETIC_TRANSFORMS,
        'negative_types': NEGATIVE_TYPES,
        'outputs': {
            'data': 'b_data.npz',
            'split_manifest': 'b_split_manifest.csv',
            'query_manifest': 'b_queries.csv',
            'candidate_manifest': 'b_candidates.csv',
            'preferences': 'b_preferences.csv',
            'sources': 'b_sources.csv',
        },
    }
    (out_dir / 'b_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config


def _stratified_queries(labels, valid_indices, num_queries, rng):
    groups = {}
    for valid_idx in valid_indices:
        sig = signature_string(labels[int(valid_idx)])
        groups.setdefault(sig, []).append(int(valid_idx))
    selected = []
    shuffled = []
    for values in groups.values():
        values = list(values)
        rng.shuffle(values)
        shuffled.append(values)
    while len(selected) < min(num_queries, len(valid_indices)):
        progressed = False
        for values in shuffled:
            if values and len(selected) < num_queries:
                selected.append(values.pop())
                progressed = True
        if not progressed:
            break
    return np.asarray(selected, dtype=np.int64)


def _sample_negatives(query_valid_idx, all_valid_indices, maps, labels, original_indices, rng):
    q_label = labels[query_valid_idx].astype(np.int32)
    q_area = int(defect_mask(maps[query_valid_idx]).sum())
    q_centroid = _centroid(maps[query_valid_idx])
    rows = []
    used = {query_valid_idx}
    for negative_type in NEGATIVE_TYPES:
        candidate = _pick_negative(negative_type, query_valid_idx, all_valid_indices, maps, labels, q_label, q_area, q_centroid, used, rng)
        used.add(candidate)
        rows.append((negative_type, candidate))
    return rows


def _pick_negative(negative_type, query_valid_idx, all_valid_indices, maps, labels, q_label, q_area, q_centroid, used, rng):
    pool = [int(i) for i in all_valid_indices if int(i) not in used and int(i) != query_valid_idx]
    if negative_type == 'same_label_hard_negative':
        same = [i for i in pool if _jaccard(q_label, labels[i]) >= 1.0]
        if same:
            return max(same, key=lambda i: _centroid_distance(q_centroid, _centroid(maps[i])) + _area_delta(q_area, maps[i]))
    if negative_type == 'diff_label_similar_morphology':
        diff = [i for i in pool if _jaccard(q_label, labels[i]) <= 0.0]
        if diff:
            return min(diff, key=lambda i: _area_delta(q_area, maps[i]) + _centroid_distance(q_centroid, _centroid(maps[i])))
    if negative_type == 'same_area_wrong_shape':
        diff = [i for i in pool if _jaccard(q_label, labels[i]) <= 0.0]
        if diff:
            return min(diff, key=lambda i: _area_delta(q_area, maps[i]))
    if negative_type == 'same_position_wrong_shape':
        diff = [i for i in pool if _jaccard(q_label, labels[i]) <= 0.0]
        if diff:
            return min(diff, key=lambda i: _centroid_distance(q_centroid, _centroid(maps[i])))
    diff = [i for i in pool if _jaccard(q_label, labels[i]) <= 0.0]
    if diff:
        return int(diff[int(rng.integers(0, len(diff)))])
    return int(pool[int(rng.integers(0, len(pool)))])


def _preference_rows(query_id, syn, neg):
    out = []

    def add(preferred, less, rule, group):
        if preferred is None or less is None:
            return
        out.append({
            'query_id': query_id,
            'preferred_candidate_id': preferred,
            'less_preferred_candidate_id': less,
            'rule_type': rule,
            'rule_group': group,
        })

    easy = _first(neg, 'easy_diff_label_random')
    area = _first(neg, 'same_area_wrong_shape')
    pos = _first(neg, 'same_position_wrong_shape')
    same_bad = _first(neg, 'same_label_hard_negative')
    diff_sim = _first(neg, 'diff_label_similar_morphology')
    for variant in ('rot_90', 'rot_180'):
        add(syn.get(variant), easy, f'{variant}_over_easy_negative', 'rotation')
        add(syn.get(variant), pos, f'{variant}_over_same_position_wrong_shape', 'hard_negative')
    add(syn.get('shift_mild'), syn.get('shift_strong'), 'mild_shift_over_strong_shift', 'shift')
    add(syn.get('shift_mild'), area, 'mild_shift_over_same_area_wrong_shape', 'shift')
    add(syn.get('noise_mild'), syn.get('noise_strong'), 'mild_noise_over_strong_noise', 'noise')
    add(syn.get('noise_mild'), easy, 'mild_noise_over_easy_negative', 'noise')
    add(syn.get('dropout_mild'), syn.get('dropout_strong'), 'mild_dropout_over_strong_dropout', 'dropout')
    add(syn.get('dropout_mild'), easy, 'mild_dropout_over_easy_negative', 'dropout')
    add(syn.get('identity'), syn.get('cluster_extra'), 'identity_over_cluster_extra', 'cluster_extra')
    add(syn.get('cluster_extra'), easy, 'cluster_extra_over_easy_negative', 'cluster_extra')
    add(syn.get('identity'), syn.get('cluster_dropout'), 'identity_over_cluster_dropout', 'cluster_dropout')
    add(syn.get('cluster_dropout'), easy, 'cluster_dropout_over_easy_negative', 'cluster_dropout')
    add(syn.get('scale_mild'), easy, 'scale_mild_over_easy_negative', 'scale')
    add(same_bad, diff_sim, 'same_label_hard_negative_over_diff_label_spatially_close', 'hard_negative')
    return out


def _first(values, key):
    items = values.get(key, [])
    return items[0] if items else None


def _centroid(raw):
    coords = np.argwhere(defect_mask(raw))
    if len(coords) == 0:
        return np.asarray(raw.shape, dtype=np.float32) / 2.0
    return coords.mean(axis=0)


def _centroid_distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _area_delta(q_area, raw):
    return abs(int(q_area) - int(defect_mask(raw).sum())) / max(float(q_area), 1.0)


def _jaccard(a, b):
    a = np.asarray(a).astype(np.int32)
    b = np.asarray(b).astype(np.int32)
    inter = int(np.dot(a, b))
    union = int(a.sum() + b.sum() - inter)
    return float(inter / max(union, 1))


def _split_columns():
    return ['sample_id', 'original_index', 'valid_index', 'split', 'label_signature', 'row_type', 'source_id'] + [f'label_{i}' for i in range(len(CLASS_NAMES))]


def _query_columns():
    return ['sample_id', 'original_index', 'valid_index', 'label_signature', 'source_id'] + [f'label_{i}' for i in range(len(CLASS_NAMES))]


def _candidate_columns():
    return ['query_id', 'candidate_id', 'candidate_kind', 'source_id', 'transform_type', 'transform_strength', 'negative_type', 'label_signature'] + [f'label_{i}' for i in range(len(CLASS_NAMES))]


def _preference_columns():
    return ['query_id', 'preferred_candidate_id', 'less_preferred_candidate_id', 'rule_type', 'rule_group']


def _source_columns():
    return ['sample_id', 'source_id', 'row_type', 'candidate_kind', 'transform_type', 'transform_strength', 'query_order']


def _split_row(sample_id, source_id, split, label, row_type):
    row = {
        'sample_id': int(sample_id),
        'original_index': int(sample_id),
        'valid_index': int(sample_id),
        'split': split,
        'label_signature': signature_string(label),
        'row_type': row_type,
        'source_id': int(source_id),
    }
    row.update({f'label_{i}': int(label[i]) for i in range(len(CLASS_NAMES))})
    return row


def _query_row(sample_id, source_id, label):
    row = {
        'sample_id': int(sample_id),
        'original_index': int(sample_id),
        'valid_index': int(sample_id),
        'label_signature': signature_string(label),
        'source_id': int(source_id),
    }
    row.update({f'label_{i}': int(label[i]) for i in range(len(CLASS_NAMES))})
    return row


def _candidate_row(query_id, candidate_id, kind, source_id, transform_type, strength, negative_type, label):
    row = {
        'query_id': int(query_id),
        'candidate_id': int(candidate_id),
        'candidate_kind': kind,
        'source_id': int(source_id),
        'transform_type': transform_type,
        'transform_strength': strength,
        'negative_type': negative_type,
        'label_signature': signature_string(label),
    }
    row.update({f'label_{i}': int(label[i]) for i in range(len(CLASS_NAMES))})
    return row


def _source_row(sample_id, source_id, row_type, candidate_kind, transform_type, strength, query_order):
    return {
        'sample_id': int(sample_id),
        'source_id': int(source_id),
        'row_type': row_type,
        'candidate_kind': candidate_kind,
        'transform_type': transform_type,
        'transform_strength': strength,
        'query_order': int(query_order),
    }


def _write_csv(path, fieldnames, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
