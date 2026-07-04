# -*- coding: utf-8 -*-
"""Ranking metrics for label-derived wafer-map retrieval evaluation."""

import numpy as np


def dcg(gains):
    gains = np.asarray(gains, dtype=np.float64)
    if gains.size == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float((gains * discount).sum())


def ndcg_at_k(actual_gains, ideal_gains, k):
    actual = np.asarray(actual_gains, dtype=np.float64)[:k]
    ideal = np.asarray(sorted(np.asarray(ideal_gains, dtype=np.float64), reverse=True), dtype=np.float64)[:k]
    if actual.size == 0:
        return 0.0
    idcg = dcg(ideal)
    return dcg(actual) / idcg if idcg > 0 else None


def average_precision(binary_rel, total_relevant=None):
    binary_rel = np.asarray(binary_rel, dtype=np.float64)
    if binary_rel.size == 0:
        return 0.0
    if total_relevant is None:
        total_relevant = float(binary_rel.sum())
    total_relevant = max(float(total_relevant), 1.0)
    positive = np.where(binary_rel > 0)[0]
    if positive.size == 0:
        return 0.0
    cumulative = np.cumsum(binary_rel)
    return float((cumulative[positive] / (positive + 1)).sum() / total_relevant)


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None

