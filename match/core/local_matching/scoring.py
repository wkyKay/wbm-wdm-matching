from __future__ import annotations

from typing import Dict

import numpy as np

from .models import LocalMatchResult
from .proposal import (_proposal_config, _tokens_from_mask, _tokens_from_weighted_mask,)


MIN_SHAPE_SIM_FOR_MATCH = 0.45
SHAPE_SCORE_POWER = 2.0


def compute_count_partial_match(
    reference,
    candidate,
    min_area: int = 5,
    top_k: int = 6,
    token_match_top_k: int = 3,
    map_match_top_k: int = 20,
    sigma_pos: float = 0.35,
    sigma_scale: float = 1.5,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
) -> LocalMatchResult:
    explanation = explain_count_partial_match(
        reference,
        candidate,
        min_area=min_area,
        top_k=top_k,
        token_match_top_k=token_match_top_k,
        map_match_top_k=map_match_top_k,
        sigma_pos=sigma_pos,
        sigma_scale=sigma_scale,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
    )
    return explanation["result"]


def compute_binary_partial_match(
    reference,
    candidate,
    min_area: int = 5,
    top_k: int = 6,
    token_match_top_k: int = 3,
    map_match_top_k: int = 20,
    sigma_pos: float = 0.35,
    sigma_scale: float = 1.5,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
) -> LocalMatchResult:
    explanation = explain_binary_partial_match(
        reference,
        candidate,
        min_area=min_area,
        top_k=top_k,
        token_match_top_k=token_match_top_k,
        map_match_top_k=map_match_top_k,
        sigma_pos=sigma_pos,
        sigma_scale=sigma_scale,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
    )
    return explanation["result"]


def explain_count_partial_match(
    reference,
    candidate,
    min_area: int = 5,
    top_k: int = 6,
    token_match_top_k: int = 3,
    map_match_top_k: int = 20,
    sigma_pos: float = 0.35,
    sigma_scale: float = 1.5,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
) -> Dict:
    valid_mask = (reference.status_map == 1) | (reference.status_map == 2)
    wbm_mask = reference.status_map == 2
    wdm_count = np.where(valid_mask, candidate.count_map, 0).astype(np.float32)
    wdm_mask = wdm_count > 0

    return _explain_local_partial_match(
        reference=reference,
        wbm_mask=wbm_mask & valid_mask,
        wdm_mask=wdm_mask & valid_mask,
        wdm_weight_map=wdm_count,
        valid_mask=valid_mask,
        min_area=min_area,
        top_k=top_k,
        token_match_top_k=token_match_top_k,
        map_match_top_k=map_match_top_k,
        sigma_pos=sigma_pos,
        sigma_scale=sigma_scale,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
    )


def explain_binary_partial_match(
    reference,
    candidate,
    min_area: int = 5,
    top_k: int = 6,
    token_match_top_k: int = 3,
    map_match_top_k: int = 20,
    sigma_pos: float = 0.35,
    sigma_scale: float = 1.5,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
) -> Dict:
    valid_mask = (reference.status_map == 1) | (reference.status_map == 2)
    wbm_mask = reference.status_map == 2
    wdm_mask = (candidate.binary_map > 0) & valid_mask
    wdm_weight_map = wdm_mask.astype(np.float32)

    return _explain_local_partial_match(
        reference=reference,
        wbm_mask=wbm_mask & valid_mask,
        wdm_mask=wdm_mask,
        wdm_weight_map=wdm_weight_map,
        valid_mask=valid_mask,
        min_area=min_area,
        top_k=top_k,
        token_match_top_k=token_match_top_k,
        map_match_top_k=map_match_top_k,
        sigma_pos=sigma_pos,
        sigma_scale=sigma_scale,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
    )


def _explain_local_partial_match(
    reference,
    wbm_mask: np.ndarray,
    wdm_mask: np.ndarray,
    wdm_weight_map: np.ndarray,
    valid_mask: np.ndarray,
    min_area: int,
    top_k: int,
    token_match_top_k: int,
    map_match_top_k: int,
    sigma_pos: float,
    sigma_scale: float,
    proposal_mode: str,
    rotation_tolerance: bool,
) -> Dict:
    proposal_config = _proposal_config(
        reference.status_map.shape,
        int(valid_mask.sum()),
        min_area,
        top_k,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
    )
    wbm_tokens = _tokens_from_mask(wbm_mask & valid_mask, valid_mask, proposal_config=proposal_config)
    wdm_tokens = _tokens_from_weighted_mask(
        wdm_mask & valid_mask,
        valid_mask,
        wdm_weight_map.astype(np.float32),
        proposal_config=proposal_config,
    )

    if not wbm_tokens or not wdm_tokens:
        return {
            "result": LocalMatchResult(
                score=0.0,
                mean_shape=0.0,
                mean_position=0.0,
                mean_scale=0.0,
                mean_type=0.0,
                matched_tokens=0,
                wbm_tokens=len(wbm_tokens),
                wdm_tokens=len(wdm_tokens),
            ),
            "wbm_tokens": wbm_tokens,
            "wdm_tokens": wdm_tokens,
            "matches": [],
            "token_topk_matches": [],
            "map_topk_matches": [],
        }

    all_pairs = []
    for query_id, qt in enumerate(wbm_tokens):
        for candidate_id, ct in enumerate(wdm_tokens):
            comp = _token_match_components(qt, ct, sigma_pos=sigma_pos, sigma_scale=sigma_scale)
            if comp["score"] > 0:
                all_pairs.append(_pair_record(query_id, candidate_id, qt, ct, comp))
    all_pairs.sort(key=lambda item: item[0], reverse=True)

    map_topk_matches = _ranked_matches(all_pairs[:max(int(map_match_top_k), 0)])
    token_topk_matches = _token_topk_matches(all_pairs, len(wbm_tokens), max(int(token_match_top_k), 0))

    # Evidence tables keep top-k candidates, but summary scoring uses a
    # one-to-one greedy match so neither WBM nor WDM tokens are counted twice.
    matches = _greedy_one_to_one_matches(all_pairs)

    weighted_scores = []
    weights = []
    best_components = []
    for query_id, qt in enumerate(wbm_tokens):
        weight = float(np.sqrt(max(qt["area"], 1.0)))
        weights.append(weight)
        match = next((m for m in matches if m["query_token_id"] == query_id), None)
        if match:
            weighted_scores.append(match["score"] * weight)
            best_components.append({k: match[k] for k in ("score", "shape_sim", "position_affinity", "scale_affinity", "type_affinity")})
        else:
            weighted_scores.append(0.0)
            best_components.append({"score": 0.0, "shape_sim": 0.0, "position_affinity": 0.0, "scale_affinity": 0.0, "type_affinity": 0.0})

    score = float(np.sum(weighted_scores) / max(np.sum(weights), 1e-6))
    comp_weights = np.asarray(weights, dtype=np.float32)

    def _weighted_mean(name: str) -> float:
        vals = np.asarray([c[name] for c in best_components], dtype=np.float32)
        return float((vals * comp_weights).sum() / max(comp_weights.sum(), 1e-6))

    result = LocalMatchResult(
        score=score,
        mean_shape=_weighted_mean("shape_sim"),
        mean_position=_weighted_mean("position_affinity"),
        mean_scale=_weighted_mean("scale_affinity"),
        mean_type=_weighted_mean("type_affinity"),
        matched_tokens=len(matches),
        wbm_tokens=len(wbm_tokens),
        wdm_tokens=len(wdm_tokens),
    )
    return {
        "result": result,
        "wbm_tokens": wbm_tokens,
        "wdm_tokens": wdm_tokens,
        "matches": matches,
        "token_topk_matches": token_topk_matches,
        "map_topk_matches": map_topk_matches,
    }


def _token_match_components(query: Dict, candidate: Dict, sigma_pos: float, sigma_scale: float) -> Dict:
    shape_sim = max(float(np.dot(query["descriptor"], candidate["descriptor"])), 0.0)
    pos_dist2 = float(((query["pos"] - candidate["pos"]) ** 2).sum())
    position_affinity = float(np.exp(-pos_dist2 / max(sigma_pos**2, 1e-6)))
    q_scale = max(float(query.get("support_area_ratio", 0.0)), 1e-12)
    c_scale = max(float(candidate.get("support_area_ratio", 0.0)), 1e-12)
    scale_affinity = float(np.exp(-abs(np.log(q_scale / c_scale)) / max(sigma_scale, 1e-6)))
    # geometry_type remains an explanation/diversity label; shape matching lives in descriptor space.
    type_affinity = 1.0
    if shape_sim < MIN_SHAPE_SIM_FOR_MATCH:
        score = 0.0
    else:
        score = (shape_sim**SHAPE_SCORE_POWER) * position_affinity * scale_affinity
    return {
        "score": float(score),
        "shape_sim": shape_sim,
        "position_affinity": position_affinity,
        "scale_affinity": scale_affinity,
        "type_affinity": type_affinity,
    }


def _pair_record(query_id: int, candidate_id: int, query: Dict, candidate: Dict, comp: Dict) -> tuple:
    return (
        float(comp["score"]),
        int(query_id),
        int(candidate_id),
        query,
        candidate,
        comp,
    )


def _ranked_matches(pair_records: list[tuple]) -> list[Dict]:
    matches = []
    for rank, (_, query_id, candidate_id, qt, ct, comp) in enumerate(pair_records, 1):
        matches.append({
            "rank": rank,
            "query_token_id": query_id,
            "candidate_token_id": candidate_id,
            "query_token": qt,
            "candidate_token": ct,
            **comp,
        })
    return matches


def _token_topk_matches(pair_records: list[tuple], query_count: int, top_k: int) -> list[list[Dict]]:
    if top_k <= 0:
        return [[] for _ in range(query_count)]
    grouped = [[] for _ in range(query_count)]
    for record in pair_records:
        _, query_id, _, _, _, _ = record
        if len(grouped[query_id]) >= top_k:
            continue
        grouped[query_id].append(record)
    return [_ranked_matches(items) for items in grouped]


def _greedy_one_to_one_matches(pair_records: list[tuple]) -> list[Dict]:
    matched_queries = set()
    matched_candidates = set()
    selected = []
    for record in pair_records:
        _, query_id, candidate_id, _, _, _ = record
        if query_id in matched_queries or candidate_id in matched_candidates:
            continue
        matched_queries.add(query_id)
        matched_candidates.add(candidate_id)
        selected.append(record)
    return _ranked_matches(selected)
