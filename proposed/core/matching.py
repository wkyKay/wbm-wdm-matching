# -*- coding: utf-8 -*-
"""Local token matching for learned cluster descriptors."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def token_match_components(query: Dict, candidate: Dict, sigma_pos: float = 0.35, sigma_area: float = 1.0) -> Dict:
    q_desc = query.get('shape_descriptor', query.get('embedding'))
    c_desc = candidate.get('shape_descriptor', candidate.get('embedding'))
    desc_sim = float(np.dot(q_desc, c_desc))
    desc_sim = max(desc_sim, 0.0)
    pos_dist2 = float(((query['pos'] - candidate['pos']) ** 2).sum())
    pos_affinity = float(np.exp(-pos_dist2 / max(sigma_pos ** 2, 1e-6)))
    scale_q = max(query.get('area_ratio', query.get('area', 1.0)), 1e-12)
    scale_c = max(candidate.get('area_ratio', candidate.get('area', 1.0)), 1e-12)
    scale_affinity = float(np.exp(-abs(np.log(scale_q / scale_c)) / max(sigma_area, 1e-6)))
    type_affinity = _type_affinity(query['geometry_type'], candidate['geometry_type'])
    score = desc_sim * pos_affinity * scale_affinity * type_affinity
    return {
        'score': float(score),
        'shape_sim': float(desc_sim),
        'position_affinity': float(pos_affinity),
        'scale_affinity': float(scale_affinity),
        'type_affinity': float(type_affinity),
    }


def token_match_score(query: Dict, candidate: Dict, sigma_pos: float = 0.35, sigma_area: float = 1.0) -> float:
    return token_match_components(query, candidate, sigma_pos=sigma_pos, sigma_area=sigma_area)['score']


def map_similarity(query_tokens: List[Dict], candidate_tokens: List[Dict],
                   sigma_pos: float = 0.35, sigma_area: float = 1.0, topk: int = 1) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    scores = []
    weights = []
    for qt in query_tokens:
        pair_scores = [token_match_score(qt, ct, sigma_pos=sigma_pos, sigma_area=sigma_area) for ct in candidate_tokens]
        k = min(topk, len(pair_scores))
        scores.append(float(np.sort(pair_scores)[-k:].mean()))
        weights.append(float(np.sqrt(max(qt['area'], 1.0))))
    scores = np.asarray(scores, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    return float((scores * weights).sum() / max(weights.sum(), 1e-6))


def _type_affinity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    compatible = {('line', 'irregular'), ('blob', 'irregular'), ('central', 'blob')}
    if (a, b) in compatible or (b, a) in compatible:
        return 0.6
    return 0.25

