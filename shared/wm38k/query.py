# -*- coding: utf-8 -*-
"""Fixed query subset sampling for WM38K retrieval evaluation."""

from collections import defaultdict
from typing import Iterable, List

import numpy as np

from .io import label_signature


def stratified_query_sample(
    labels: np.ndarray,
    candidate_valid_indices: Iterable[int],
    num_queries: int = 2000,
    seed: int = 2026,
) -> List[int]:
    candidate_valid_indices = [int(x) for x in candidate_valid_indices]
    if num_queries is None or num_queries >= len(candidate_valid_indices):
        return sorted(candidate_valid_indices)

    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for valid_index in candidate_valid_indices:
        groups[label_signature(labels[valid_index])].append(valid_index)

    shuffled_groups = []
    for _, values in sorted(groups.items(), key=lambda kv: kv[0]):
        values = np.asarray(values, dtype=np.int64)
        rng.shuffle(values)
        shuffled_groups.append(values.tolist())

    selected = []
    while len(selected) < num_queries:
        made_progress = False
        for values in shuffled_groups:
            if values and len(selected) < num_queries:
                selected.append(int(values.pop()))
                made_progress = True
        if not made_progress:
            break
    return sorted(selected)

