"""Arc-ring local retrieval compatible with the legacy partial_match outputs.

The arc-ring proposal and scoring implementation is vendored in
``partial_match.core.arc_ring_local_matching``. This adapter caches proposals
per map and exposes the old partial_match record and explanation shapes, so
ranking and evaluation artifacts keep their schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from partial_match.core.arc_ring_local_matching.proposal import _proposal_config, _tokens_from_mask
from partial_match.core.arc_ring_local_matching.scoring import (
    _greedy_one_to_one_matches,
    _max_token_area,
    _normalized_pair_weights,
    _normalized_score_weights,
    _pair_record,
    _relative_area,
    _token_match_components,
)
from partial_match.core.arc_ring_local_matching.descriptors import DEFAULT_GEOMETRY_WEIGHT, DEFAULT_MOMENT_WEIGHT


@dataclass(frozen=True)
class ArcRingConfig:
    min_area: int = 5
    top_k: int = 5
    sigma_pos: float = 0.35
    sigma_scale: float = 1.5
    min_token_score: float = 0.30
    min_relative_token_area: float = 0.10
    scale_ratio_min: float = 0.50
    shape_weight: float = 0.60
    position_weight: float = 0.25
    scale_weight: float = 0.15
    scale_area_weight: float = 0.30
    scale_pca_weight: float = 0.70
    moment_weight: float = DEFAULT_MOMENT_WEIGHT
    geometry_weight: float = DEFAULT_GEOMETRY_WEIGHT
    rotation_tolerance: bool = False


def prepare_tokens(defect_mask: np.ndarray, valid_mask: np.ndarray, config: ArcRingConfig) -> List[Dict]:
    """Extract the same arc-ring-residual tokens used by the match experiment."""
    proposal_config = _proposal_config(
        defect_mask.shape,
        int(valid_mask.sum()),
        config.min_area,
        config.top_k,
        proposal_mode="arc-ring-residual",
        rotation_tolerance=config.rotation_tolerance,
        moment_weight=config.moment_weight,
        geometry_weight=config.geometry_weight,
    )
    return _tokens_from_mask(
        defect_mask & valid_mask,
        valid_mask,
        proposal_config=proposal_config,
    )


def score_tokens(query_tokens: List[Dict], candidate_tokens: List[Dict], config: ArcRingConfig) -> Dict:
    """Apply hard gates and greedy one-to-one token matching before aggregation."""
    if not query_tokens or not candidate_tokens:
        return _empty_explanation(query_tokens, candidate_tokens)

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
    score = float(np.sum(weighted_scores) / max(np.sum(weights), 1e-6))
    return {
        "score": score,
        "matches": matches,
        "query_tokens": query_tokens,
        "candidate_tokens": candidate_tokens,
    }


def token_row(map_id: int, token_id: int, token: Dict) -> Dict:
    """Render an enhanced token as the legacy tokens.csv row schema."""
    return {
        "map_id": int(map_id),
        "token_id": int(token_id),
        "geometry_type": token.get("geometry_type", "irregular"),
        "area": float(token.get("area", 0.0)),
        "area_ratio": float(token.get("support_area_ratio", 0.0)),
        "centroid_row": float(token.get("centroid_row", 0.0)),
        "centroid_col": float(token.get("centroid_col", 0.0)),
        "bbox_row_min": int(token.get("bbox_row_min", 0)),
        "bbox_col_min": int(token.get("bbox_col_min", 0)),
        "bbox_row_max": int(token.get("bbox_row_max", 0)),
        "bbox_col_max": int(token.get("bbox_col_max", 0)),
        "bbox_height": int(token.get("bbox_height", 0)),
        "bbox_width": int(token.get("bbox_width", 0)),
        "compactness": float(token.get("compactness", 0.0)),
        "orientation": float(token.get("orientation", 0.0)),
        "radial_distance_norm": float(token.get("radial_distance_norm", 0.0)),
        "proposal_type": token.get("proposal_type", ""),
        "proposal_source": token.get("proposal_source", "arc_ring_residual"),
        "angular_coverage": float(token.get("angular_coverage", 0.0)),
        "radial_std": float(token.get("radial_std", 0.0)),
    }


def _empty_explanation(query_tokens: List[Dict], candidate_tokens: List[Dict]) -> Dict:
    return {
        "score": 0.0,
        "matches": [],
        "query_tokens": query_tokens,
        "candidate_tokens": candidate_tokens,
    }
