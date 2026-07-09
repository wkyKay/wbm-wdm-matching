from __future__ import annotations

from typing import Dict

import numpy as np

from .models import LocalMatchResult
from .proposal import (_proposal_config, _tokens_from_mask, _tokens_from_weighted_mask,)


MIN_SHAPE_SIM_FOR_MATCH = 0.45
MIN_TOKEN_SCORE_FOR_MATCH = 0.45
DEFAULT_SHAPE_SCORE_WEIGHT = 0.60
DEFAULT_POSITION_SCORE_WEIGHT = 0.25
DEFAULT_SCALE_SCORE_WEIGHT = 0.15


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
    min_token_score: float = MIN_TOKEN_SCORE_FOR_MATCH,
    score_shape_weight: float = DEFAULT_SHAPE_SCORE_WEIGHT,
    score_position_weight: float = DEFAULT_POSITION_SCORE_WEIGHT,
    score_scale_weight: float = DEFAULT_SCALE_SCORE_WEIGHT,
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
        min_token_score=min_token_score,
        score_shape_weight=score_shape_weight,
        score_position_weight=score_position_weight,
        score_scale_weight=score_scale_weight,
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
    min_token_score: float = MIN_TOKEN_SCORE_FOR_MATCH,
    score_shape_weight: float = DEFAULT_SHAPE_SCORE_WEIGHT,
    score_position_weight: float = DEFAULT_POSITION_SCORE_WEIGHT,
    score_scale_weight: float = DEFAULT_SCALE_SCORE_WEIGHT,
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
        min_token_score=min_token_score,
        score_shape_weight=score_shape_weight,
        score_position_weight=score_position_weight,
        score_scale_weight=score_scale_weight,
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
    min_token_score: float = MIN_TOKEN_SCORE_FOR_MATCH,
    score_shape_weight: float = DEFAULT_SHAPE_SCORE_WEIGHT,
    score_position_weight: float = DEFAULT_POSITION_SCORE_WEIGHT,
    score_scale_weight: float = DEFAULT_SCALE_SCORE_WEIGHT,
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
        min_token_score=min_token_score,
        score_shape_weight=score_shape_weight,
        score_position_weight=score_position_weight,
        score_scale_weight=score_scale_weight,
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
    min_token_score: float = MIN_TOKEN_SCORE_FOR_MATCH,
    score_shape_weight: float = DEFAULT_SHAPE_SCORE_WEIGHT,
    score_position_weight: float = DEFAULT_POSITION_SCORE_WEIGHT,
    score_scale_weight: float = DEFAULT_SCALE_SCORE_WEIGHT,
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
        min_token_score=min_token_score,
        score_shape_weight=score_shape_weight,
        score_position_weight=score_position_weight,
        score_scale_weight=score_scale_weight,
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
    min_token_score: float,
    score_shape_weight: float,
    score_position_weight: float,
    score_scale_weight: float,
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

    empty_result = LocalMatchResult(
        score=0.0,
        mean_shape=0.0,
        mean_position=0.0,
        mean_scale=0.0,
        mean_type=0.0,
        matched_tokens=0,
        wbm_tokens=len(wbm_tokens),
        wdm_tokens=len(wdm_tokens),
    )
    if not wbm_tokens or not wdm_tokens:
        return {
            "result": empty_result,
            "result_matched_only": empty_result,
            "wbm_tokens": wbm_tokens,
            "wdm_tokens": wdm_tokens,
            "matches": [],
            "token_topk_matches": [],
            "map_topk_matches": [],
        }

    all_pairs = []
    min_token_score = max(float(min_token_score), 0.0)
    score_weights = _normalized_score_weights(
        score_shape_weight,
        score_position_weight,
        score_scale_weight,
    )
    for query_id, qt in enumerate(wbm_tokens):
        for candidate_id, ct in enumerate(wdm_tokens):
            comp = _token_match_components(
                qt,
                ct,
                sigma_pos=sigma_pos,
                sigma_scale=sigma_scale,
                score_weights=score_weights,
            )
            passes_score_gate = comp["score"] >= min_token_score if min_token_score > 0 else comp["score"] > 0
            if passes_score_gate:
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

    # --- matched-only scoring: only matched tokens contribute to the weighted average ---
    matched_weighted_scores = []
    matched_weights = []
    matched_components = []
    for query_id, qt in enumerate(wbm_tokens):
        weight = float(np.sqrt(max(qt["area"], 1.0)))
        match = next((m for m in matches if m["query_token_id"] == query_id), None)
        if match:
            matched_weighted_scores.append(match["score"] * weight)
            matched_weights.append(weight)
            matched_components.append({k: match[k] for k in ("score", "shape_sim", "position_affinity", "scale_affinity", "type_affinity")})
    mo_total_weight = float(np.sum(matched_weights))
    score_mo = float(np.sum(matched_weighted_scores) / max(mo_total_weight, 1e-6)) if matched_weights else 0.0
    mo_weights_arr = np.asarray(matched_weights, dtype=np.float32)

    def _mo_weighted_mean(name: str) -> float:
        if mo_weights_arr.size == 0:
            return 0.0
        vals = np.asarray([c[name] for c in matched_components], dtype=np.float32)
        return float((vals * mo_weights_arr).sum() / max(mo_weights_arr.sum(), 1e-6))

    result_matched_only = LocalMatchResult(
        score=score_mo,
        mean_shape=_mo_weighted_mean("shape_sim"),
        mean_position=_mo_weighted_mean("position_affinity"),
        mean_scale=_mo_weighted_mean("scale_affinity"),
        mean_type=_mo_weighted_mean("type_affinity"),
        matched_tokens=len(matches),
        wbm_tokens=len(wbm_tokens),
        wdm_tokens=len(wdm_tokens),
    )

    return {
        "result": result,
        "result_matched_only": result_matched_only,
        "wbm_tokens": wbm_tokens,
        "wdm_tokens": wdm_tokens,
        "matches": matches,
        "token_topk_matches": token_topk_matches,
        "map_topk_matches": map_topk_matches,
    }


def _normalized_score_weights(
    shape_weight: float,
    position_weight: float,
    scale_weight: float,
) -> tuple[float, float, float]:
    weights = np.asarray([shape_weight, position_weight, scale_weight], dtype=np.float32)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 1e-8:
        return (
            DEFAULT_SHAPE_SCORE_WEIGHT,
            DEFAULT_POSITION_SCORE_WEIGHT,
            DEFAULT_SCALE_SCORE_WEIGHT,
        )
    weights = weights / total
    return float(weights[0]), float(weights[1]), float(weights[2])


def _token_match_components(
    query: Dict,
    candidate: Dict,
    sigma_pos: float,
    sigma_scale: float,
    score_weights: tuple[float, float, float],
) -> Dict:
    shape_parts = _shape_similarity_components(query, candidate)
    shape_sim = shape_parts["shape_sim"]
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
        w_shape, w_position, w_scale = score_weights
        score = w_shape * shape_sim + w_position * position_affinity + w_scale * scale_affinity
    return {
        "score": float(score),
        "shape_sim": shape_sim,
        "moment_sim": shape_parts["moment_sim"],
        "geometry_sim": shape_parts["geometry_sim"],
        "position_affinity": position_affinity,
        "scale_affinity": scale_affinity,
        "type_affinity": type_affinity,
    }


def _shape_similarity_components(query: Dict, candidate: Dict) -> Dict:
    q_parts = query.get("descriptor_parts")
    c_parts = candidate.get("descriptor_parts")
    if q_parts and c_parts and q_parts.get("kind") == c_parts.get("kind") == "zernike_geometry":
        moment_sim = _cosine_sim(q_parts.get("moment"), c_parts.get("moment"))
        geometry_sim = _geometry_sim(q_parts.get("geometry"), c_parts.get("geometry"))
        moment_weight = float(q_parts.get("moment_weight", 0.75))
        geometry_weight = float(q_parts.get("geometry_weight", 0.25))
        total = max(moment_weight + geometry_weight, 1e-6)
        shape_sim = (moment_weight * moment_sim + geometry_weight * geometry_sim) / total
        return {
            "shape_sim": float(np.clip(shape_sim, 0.0, 1.0)),
            "moment_sim": float(moment_sim),
            "geometry_sim": float(geometry_sim),
        }

    shape_sim = max(float(np.dot(query["descriptor"], candidate["descriptor"])), 0.0)
    return {
        "shape_sim": float(np.clip(shape_sim, 0.0, 1.0)),
        "moment_sim": float(shape_sim),
        "geometry_sim": 0.0,
    }


def _cosine_sim(a, b) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return 0.0
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(av, bv) / denom, 0.0, 1.0))


def _geometry_sim(a, b) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return 0.0
    mean_abs_diff = float(np.mean(np.abs(av - bv)))
    return float(np.exp(-mean_abs_diff / 0.25))


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
