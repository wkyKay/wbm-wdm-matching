# -*- coding: utf-8 -*-
"""Label-derived relevance functions for MixedWM38K retrieval."""

import numpy as np


def label_set(label):
    return tuple(np.where(np.asarray(label).astype(np.int32) == 1)[0].tolist())


def overlap_hit(query_label, candidate_label):
    query_label = np.asarray(query_label).astype(np.int32)
    candidate_label = np.asarray(candidate_label).astype(np.int32)
    return int(np.dot(query_label, candidate_label) > 0)


def exact_match(query_label, candidate_label):
    return int(label_set(query_label) == label_set(candidate_label))


def jaccard(query_label, candidate_label):
    query_label = np.asarray(query_label).astype(np.int32)
    candidate_label = np.asarray(candidate_label).astype(np.int32)
    inter = int(np.dot(query_label, candidate_label))
    union = int(query_label.sum() + candidate_label.sum() - inter)
    return float(inter / max(union, 1))


def jaccard_tier(value):
    if value >= 1.0:
        return 3
    if value >= 0.5:
        return 2
    if value > 0.0:
        return 1
    return 0


def relevance_value(query_label, candidate_label, mode='jaccard'):
    value = jaccard(query_label, candidate_label)
    if mode == 'jaccard':
        return value
    if mode == 'tier':
        return float(jaccard_tier(value))
    raise ValueError(f'Unknown relevance mode: {mode}')


def gain(value, mode='identity'):
    if mode == 'identity':
        return float(value)
    if mode == 'exp2':
        return float((2.0 ** value) - 1.0)
    raise ValueError(f'Unknown gain mode: {mode}')

