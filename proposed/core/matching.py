"""Greedy local matching for learned cluster embeddings.

This module shares partial_match's proposal and scoring protocol. The learned
embedding is the only shape representation: it replaces the handcrafted
moment-and-geometry descriptor used by the traditional baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from partial_match.core.arc_ring_local_matching.scoring import (
    _greedy_one_to_one_matches,
    _max_token_area,
    _normalized_pair_weights,
    _normalized_score_weights,
    _pair_record,
    _relative_area,
    _token_match_components,
)


@dataclass(frozen=True)
class MatchingConfig:
    sigma_pos: float = 0.35
    sigma_scale: float = 1.5
    min_token_score: float = 0.30
    min_relative_token_area: float = 0.10
    scale_ratio_min: float = 0.20
    shape_weight: float = 0.60
    position_weight: float = 0.25
    scale_weight: float = 0.15
    scale_area_weight: float = 0.30
    scale_pca_weight: float = 0.70


def explain_map_similarity(query_tokens: List[Dict], candidate_tokens: List[Dict], config: MatchingConfig) -> Dict:
    """Apply the partial-match hard gates and greedy one-to-one aggregation."""
    if not query_tokens or not candidate_tokens:
        return {"score": 0.0, "matches": []}

    score_weights = _normalized_score_weights(config.shape_weight, config.position_weight, config.scale_weight)
    scale_weights = _normalized_pair_weights(
        config.scale_area_weight,
        config.scale_pca_weight,
        default_a=0.30,
        default_b=0.70,
    )
    query_max_area = _max_token_area(query_tokens)
    candidate_max_area = _max_token_area(candidate_tokens)
    scored_query_ids = {
        index for index, token in enumerate(query_tokens)
        if _relative_area(token, query_max_area) >= config.min_relative_token_area
    }
    pairs = []
    for query_id, query in enumerate(query_tokens):
        if query_id not in scored_query_ids:
            continue
        for candidate_id, candidate in enumerate(candidate_tokens):
            if _relative_area(candidate, candidate_max_area) < config.min_relative_token_area:
                continue
            components = _token_match_components(
                query,
                candidate,
                sigma_pos=config.sigma_pos,
                sigma_scale=config.sigma_scale,
                score_weights=score_weights,
                scale_component_weights=scale_weights,
                scale_ratio_min=config.scale_ratio_min,
            )
            if components["score"] >= config.min_token_score:
                pairs.append(_pair_record(query_id, candidate_id, query, candidate, components))

    pairs.sort(key=lambda item: item[0], reverse=True)
    matches = _greedy_one_to_one_matches(pairs)
    match_by_query = {match["query_token_id"]: match for match in matches}
    weights = []
    weighted_scores = []
    for query_id, query in enumerate(query_tokens):
        if query_id not in scored_query_ids:
            continue
        weight = float(np.sqrt(max(float(query["area"]), 1.0)))
        weights.append(weight)
        weighted_scores.append(weight * float(match_by_query.get(query_id, {}).get("score", 0.0)))
    return {
        "score": float(np.sum(weighted_scores) / max(np.sum(weights), 1e-6)),
        "matches": matches,
    }


def map_similarity(query_tokens: List[Dict], candidate_tokens: List[Dict], config: MatchingConfig) -> float:
    return float(explain_map_similarity(query_tokens, candidate_tokens, config)["score"])
