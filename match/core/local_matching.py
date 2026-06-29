# Local partial matching between WBM failure tokens and WDM count-map evidence tokens.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


GEOMETRY_TYPES = ["blob", "line", "edge_ring", "central", "irregular"]


@dataclass(frozen=True)
class LocalMatchResult:
    score: float
    mean_shape: float
    mean_position: float
    mean_scale: float
    mean_type: float
    matched_tokens: int
    wbm_tokens: int
    wdm_tokens: int


def compute_count_partial_match(
    reference: GridMaps,
    candidate: GridMaps,
    min_area: int = 5,
    top_k: int = 6,
    sigma_pos: float = 0.35,
    sigma_scale: float = 1.5,
) -> LocalMatchResult:
    """Score how well a WDM count map explains local WBM failure tokens.

    WBM tokens are extracted from the reference status map. WDM tokens are
    extracted from candidate.count_map and keep count values as weights.
    """
    explanation = explain_count_partial_match(
        reference,
        candidate,
        min_area=min_area,
        top_k=top_k,
        sigma_pos=sigma_pos,
        sigma_scale=sigma_scale,
    )
    return explanation["result"]


def explain_count_partial_match(
    reference: GridMaps,
    candidate: GridMaps,
    min_area: int = 5,
    top_k: int = 6,
    sigma_pos: float = 0.35,
    sigma_scale: float = 1.5,
) -> Dict:
    """Return local matching score plus tokens and best token-pair details."""
    valid_mask = (reference.status_map == VALID_NO_DEFECT) | (reference.status_map == VALID_HAS_DEFECT)
    wbm_mask = reference.status_map == VALID_HAS_DEFECT
    wdm_count = np.where(valid_mask, candidate.count_map, 0).astype(np.float32)

    wbm_tokens = _tokens_from_mask(wbm_mask & valid_mask, valid_mask, min_area=min_area, top_k=top_k)
    wdm_tokens = _tokens_from_count(wdm_count, valid_mask, min_area=min_area, top_k=top_k)

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
        }

    weighted_scores = []
    weights = []
    best_components = []
    matches = []
    for query_id, qt in enumerate(wbm_tokens):
        pairs = []
        for candidate_id, ct in enumerate(wdm_tokens):
            comp = _token_match_components(qt, ct, sigma_pos=sigma_pos, sigma_scale=sigma_scale)
            pairs.append((candidate_id, ct, comp))
        candidate_id, ct, best = max(pairs, key=lambda item: item[2]["score"])
        weight = float(np.sqrt(max(qt["area"], 1.0)))
        weighted_scores.append(best["score"] * weight)
        weights.append(weight)
        best_components.append(best)
        matches.append({
            "query_token_id": query_id,
            "candidate_token_id": candidate_id,
            "query_token": qt,
            "candidate_token": ct,
            **best,
        })

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
        matched_tokens=len(best_components),
        wbm_tokens=len(wbm_tokens),
        wdm_tokens=len(wdm_tokens),
    )
    return {
        "result": result,
        "wbm_tokens": wbm_tokens,
        "wdm_tokens": wdm_tokens,
        "matches": matches,
    }


def _tokens_from_mask(mask: np.ndarray, valid_mask: np.ndarray, min_area: int, top_k: int) -> List[Dict]:
    weight_map = mask.astype(np.float32)
    return _tokens_from_components(mask, valid_mask, weight_map, min_area=min_area, top_k=top_k, source="wbm")


def _tokens_from_count(count_map: np.ndarray, valid_mask: np.ndarray, min_area: int, top_k: int) -> List[Dict]:
    mask = (count_map > 0) & valid_mask
    return _tokens_from_components(mask, valid_mask, count_map.astype(np.float32), min_area=min_area, top_k=top_k, source="wdm")


def _tokens_from_components(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    weight_map: np.ndarray,
    min_area: int,
    top_k: int,
    source: str,
) -> List[Dict]:
    h, w = mask.shape
    tokens = []
    total_mass = float(weight_map[mask].sum())
    for comp in _connected_components(mask):
        if len(comp) < min_area:
            continue
        token = _token_stats(comp, weight_map, valid_mask, total_mass=total_mass, source=source)
        token["descriptor"] = _shape_descriptor(token, (h, w))
        tokens.append(token)

    tokens.sort(key=_token_importance, reverse=True)
    return tokens[:top_k]


def _token_stats(
    pixels: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    total_mass: float,
    source: str,
) -> Dict:
    h, w = weight_map.shape
    rows = pixels[:, 0].astype(np.int64)
    cols = pixels[:, 1].astype(np.int64)
    weights = weight_map[rows, cols].astype(np.float32)
    weights = np.maximum(weights, 1e-6)
    mass = float(weights.sum())
    support_area = int(len(pixels))

    centroid_row = float((rows * weights).sum() / mass)
    centroid_col = float((cols * weights).sum() / mass)
    bbox_row_min = int(rows.min())
    bbox_col_min = int(cols.min())
    bbox_row_max = int(rows.max())
    bbox_col_max = int(cols.max())
    bbox_h = bbox_row_max - bbox_row_min + 1
    bbox_w = bbox_col_max - bbox_col_min + 1

    rr = rows.astype(np.float32) - centroid_row
    cc = cols.astype(np.float32) - centroid_col
    cov_rr = float((weights * rr * rr).sum() / mass)
    cov_cc = float((weights * cc * cc).sum() / mass)
    cov_rc = float((weights * rr * cc).sum() / mass)
    cov = np.array([[cov_rr, cov_rc], [cov_rc, cov_cc]], dtype=np.float32)
    vals, vecs = np.linalg.eigh(cov)
    pca_l1 = float(vals[1])
    pca_l2 = float(vals[0])
    orientation = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))) if vals[1] > 1e-10 else 0.0

    perimeter = _perimeter(rows, cols)
    center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    valid_points = np.argwhere(valid_mask).astype(np.float32)
    radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max()) if len(valid_points) else float(np.linalg.norm(center))
    centroid_radial = np.linalg.norm(np.array([centroid_row, centroid_col], dtype=np.float32) - center)
    radial_distance_norm = float(centroid_radial / max(radius_ref, 1e-6))

    rel = pixels.astype(np.float32) - center
    radial = np.linalg.norm(rel, axis=1) / max(radius_ref, 1e-6)
    theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
    occupied = np.unique(np.floor(theta / 5.0).astype(np.int64)) if len(theta) else []
    angular_coverage = float(len(occupied) / 72.0)
    radial_std = float(radial.std()) if len(radial) else 0.0

    token = {
        "source": source,
        "map_shape": (h, w),
        "area": support_area,
        "support_area": support_area,
        "support_area_ratio": float(support_area / max(int(valid_mask.sum()), 1)),
        "mass": mass,
        "mass_ratio": float(mass / max(total_mass, 1e-6)),
        "peak": float(weights.max()) if len(weights) else 0.0,
        "mean_weight": float(weights.mean()) if len(weights) else 0.0,
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
        "pos": np.array([centroid_row / max(h, 1), centroid_col / max(w, 1)], dtype=np.float32),
        "bbox_row_min": bbox_row_min,
        "bbox_col_min": bbox_col_min,
        "bbox_row_max": bbox_row_max,
        "bbox_col_max": bbox_col_max,
        "bbox_height": bbox_h,
        "bbox_width": bbox_w,
        "pca_lambda1": pca_l1,
        "pca_lambda2": pca_l2,
        "orientation": orientation,
        "perimeter": perimeter,
        "compactness": float(perimeter / max(support_area, 1)),
        "radial_distance_norm": radial_distance_norm,
        "angular_coverage": angular_coverage,
        "radial_std": radial_std,
        "pixels": [(int(r), int(c)) for r, c in pixels],
    }
    token["geometry_type"] = _classify_token(token)
    return token


def _shape_descriptor(token: Dict, map_shape) -> np.ndarray:
    bbox_h = float(token.get("bbox_height", 1))
    bbox_w = float(token.get("bbox_width", 1))
    area = float(token.get("area", 0))
    bbox_area = max(bbox_h * bbox_w, 1.0)
    fill_ratio = area / bbox_area
    aspect = max(bbox_h / max(bbox_w, 1.0), bbox_w / max(bbox_h, 1.0))
    elongation = float(token.get("pca_lambda1", 0.0)) / max(float(token.get("pca_lambda2", 0.0)), 1e-6)
    compactness = float(token.get("compactness", 0.0))
    orientation = float(token.get("orientation", 0.0))

    features = np.array([
        fill_ratio,
        np.log1p(aspect) / np.log(16.0),
        np.log1p(elongation) / np.log(64.0),
        min(compactness / 4.0, 1.0),
        np.cos(np.deg2rad(orientation)),
        np.sin(np.deg2rad(orientation)),
        float(token.get("angular_coverage", 0.0)),
        float(token.get("radial_std", 0.0)),
    ], dtype=np.float32)
    desc = np.concatenate([features, _shape_profiles(token, map_shape, bins=8)]).astype(np.float32)
    norm = float(np.linalg.norm(desc))
    if norm > 1e-8:
        desc /= norm
    return desc


def _token_match_components(query: Dict, candidate: Dict, sigma_pos: float, sigma_scale: float) -> Dict:
    shape_sim = max(float(np.dot(query["descriptor"], candidate["descriptor"])), 0.0)
    pos_dist2 = float(((query["pos"] - candidate["pos"]) ** 2).sum())
    position_affinity = float(np.exp(-pos_dist2 / max(sigma_pos**2, 1e-6)))
    q_scale = max(float(query.get("support_area_ratio", 0.0)), 1e-12)
    c_scale = max(float(candidate.get("support_area_ratio", 0.0)), 1e-12)
    scale_affinity = float(np.exp(-abs(np.log(q_scale / c_scale)) / max(sigma_scale, 1e-6)))
    type_affinity = _type_affinity(query["geometry_type"], candidate["geometry_type"])
    score = shape_sim * position_affinity * scale_affinity * type_affinity
    return {
        "score": float(score),
        "shape_sim": shape_sim,
        "position_affinity": position_affinity,
        "scale_affinity": scale_affinity,
        "type_affinity": type_affinity,
    }


def _classify_token(token: Dict) -> str:
    if (
        token.get("radial_distance_norm", 0.0) >= 0.65
        and token.get("angular_coverage", 0.0) >= 0.16
        and token.get("radial_std", 1.0) <= 0.14
    ):
        return "edge_ring"

    area = max(token.get("area", 1), 1)
    bbox_area = max(token.get("bbox_height", 1) * token.get("bbox_width", 1), 1)
    fill_ratio = area / bbox_area
    elongation = token.get("pca_lambda1", 0.0) / max(token.get("pca_lambda2", 0.0), 1e-6)
    aspect = max(
        token.get("bbox_height", 1) / max(token.get("bbox_width", 1), 1),
        token.get("bbox_width", 1) / max(token.get("bbox_height", 1), 1),
    )
    if elongation >= 6.0 or aspect >= 4.0:
        return "line"
    if fill_ratio >= 0.45 and token.get("compactness", 0.0) <= 1.6:
        return "blob"
    if token.get("radial_distance_norm", 1.0) <= 0.35:
        return "central"
    return "irregular"


def _shape_profiles(token: Dict, map_shape, bins: int) -> np.ndarray:
    pixels = np.asarray(token.get("pixels", []), dtype=np.float32)
    if len(pixels) == 0:
        return np.zeros(bins * 2, dtype=np.float32)
    h, w = map_shape
    center = np.array([token.get("centroid_row", h / 2.0), token.get("centroid_col", w / 2.0)], dtype=np.float32)
    rel = pixels - center
    radius = np.linalg.norm(rel, axis=1)
    radius = radius / max(float(radius.max()), 1.0)
    radial_hist, _ = np.histogram(radius, bins=bins, range=(0.0, 1.0))
    theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
    theta_hist, _ = np.histogram(theta, bins=bins, range=(0.0, 360.0))
    profile = np.concatenate([radial_hist, theta_hist]).astype(np.float32)
    profile /= max(float(profile.sum()), 1.0)
    return profile


def _token_importance(token: Dict) -> float:
    mass = float(token.get("mass", token.get("area", 0)))
    area = float(token.get("area", 0))
    type_bonus = {
        "edge_ring": 2.0,
        "central": 1.5,
        "blob": 1.2,
        "line": 1.0,
        "irregular": 0.8,
    }.get(token.get("geometry_type"), 0.8)
    return float(np.sqrt(max(mass, 0.0)) + 0.25 * np.sqrt(max(area, 0.0)) + type_bonus)


def _type_affinity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    compatible = {
        ("line", "irregular"),
        ("blob", "irregular"),
        ("central", "blob"),
    }
    if (a, b) in compatible or (b, a) in compatible:
        return 0.6
    return 0.25


def _perimeter(rows: np.ndarray, cols: np.ndarray) -> int:
    pixel_set = set((int(r), int(c)) for r, c in zip(rows, cols))
    perimeter = 0
    for r, c in pixel_set:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) not in pixel_set:
                perimeter += 1
    return perimeter


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for row in range(h):
        for col in range(w):
            if not mask[row, col] or visited[row, col]:
                continue
            queue = [(row, col)]
            visited[row, col] = True
            comp = []
            while queue:
                r, c = queue.pop(0)
                comp.append((r, c))
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(comp, dtype=np.int64))
    return components
