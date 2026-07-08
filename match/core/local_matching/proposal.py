from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import ProposalConfig
from .morphology import (_connected_components, _perimeter,)
from .descriptors import _classify_token, _shape_descriptor


def _tokens_from_mask(mask: np.ndarray, valid_mask: np.ndarray, proposal_config: ProposalConfig) -> List[Dict]:
    weight_map = mask.astype(np.float32)
    return _tokens_from_components(mask, valid_mask, weight_map, proposal_config=proposal_config, source="wbm")


def _tokens_from_count(count_map: np.ndarray, valid_mask: np.ndarray, proposal_config: ProposalConfig) -> List[Dict]:
    mask = (count_map > 0) & valid_mask
    return _tokens_from_weighted_mask(mask, valid_mask, count_map.astype(np.float32), proposal_config=proposal_config)


def _tokens_from_weighted_mask(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    weight_map: np.ndarray,
    proposal_config: ProposalConfig,
) -> List[Dict]:
    return _tokens_from_components(mask, valid_mask, weight_map, proposal_config=proposal_config, source="wdm")


def _tokens_from_components(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    weight_map: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
) -> List[Dict]:
    h, w = mask.shape
    if proposal_config.proposal_mode == "compact":
        tokens = _retrieval_compact_tokens(mask & valid_mask, weight_map, valid_mask, proposal_config, source=source)
        for token in tokens:
            _finalize_token(token, (h, w), proposal_config)
        return tokens

    tokens = []
    total_mass = float(weight_map[mask].sum())
    for comp in _connected_components(mask, connectivity=proposal_config.connectivity):
        if len(comp) < proposal_config.min_area:
            continue
        token = _token_stats(comp, weight_map, valid_mask, total_mass=total_mass, source=source)
        tokens.append(token)

    tokens.sort(key=_token_importance, reverse=True)
    tokens = tokens[:proposal_config.top_k]
    for token in tokens:
        _finalize_token(token, (h, w), proposal_config)
    return tokens


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


def _retrieval_compact_tokens(
    mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
) -> List[Dict]:
    h, w = weight_map.shape
    original = mask & valid_mask
    denoised = np.zeros_like(original, dtype=bool)
    for comp in _connected_components(original, connectivity=proposal_config.connectivity):
        if len(comp) >= proposal_config.min_area:
            denoised[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True

    ring_token, ring_mask, _ = _extract_retrieval_ring_token(
        denoised,
        weight_map,
        valid_mask=valid_mask,
        source=source,
    )
    residual = denoised & (~ring_mask)
    component_tokens = _retrieval_component_tokens(
        residual,
        weight_map,
        valid_mask,
        min_area=proposal_config.min_area,
        source=source,
    )
    return _select_retrieval_tokens(ring_token, component_tokens, proposal_config.top_k)


def _extract_retrieval_ring_token(
    mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: Optional[np.ndarray],
    source: str,
    min_area: int = 12,
    edge_r_min: float = 0.65,
    band_width: float = 0.10,
    min_angular_coverage: float = 0.16,
    min_ring_area_ratio: float = 0.12,
    max_radial_std: float = 0.12,
    max_defect_ratio: float = 0.45,
    min_edge_defect_fraction: float = 0.45,
) -> Tuple[Optional[Dict], np.ndarray, Dict]:
    h, w = mask.shape
    ring_mask = np.zeros_like(mask, dtype=bool)
    points = np.argwhere(mask).astype(np.float32)
    debug = {"accepted": False, "reason": "no_points", "candidate_area": 0, "angular_coverage": 0.0}
    if len(points) < min_area:
        return None, ring_mask, debug

    center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    if valid_mask is not None and valid_mask.any():
        valid_points = np.argwhere(valid_mask).astype(np.float32)
        radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max())
        valid_area = int(valid_mask.sum())
    else:
        radius_ref = float(max(np.linalg.norm(points - center, axis=1).max(), 1.0))
        valid_area = h * w
    if radius_ref <= 1e-6:
        debug["reason"] = "bad_radius"
        return None, ring_mask, debug

    rel = points - center
    radial = np.linalg.norm(rel, axis=1) / radius_ref
    defect_ratio = float(len(points) / max(valid_area, 1))
    edge_keep = radial >= edge_r_min
    edge_fraction = float(edge_keep.sum() / max(len(points), 1))
    debug.update(defect_ratio=defect_ratio, edge_fraction=edge_fraction)
    if defect_ratio > max_defect_ratio:
        debug["reason"] = "skip_high_defect_ratio"
        return None, ring_mask, debug
    if edge_fraction < min_edge_defect_fraction:
        debug["reason"] = "skip_low_edge_fraction"
        return None, ring_mask, debug

    edge_points = points[edge_keep]
    edge_radial = radial[edge_keep]
    if len(edge_points) < min_area:
        debug.update(reason="too_few_edge_points", candidate_area=int(len(edge_points)))
        return None, ring_mask, debug

    hist, edges = np.histogram(edge_radial, bins=12, range=(edge_r_min, 1.05))
    peak = int(hist.argmax())
    band_center = float((edges[peak] + edges[peak + 1]) / 2.0)
    band_keep = np.abs(edge_radial - band_center) <= band_width
    ring_points = edge_points[band_keep]
    ring_radial = edge_radial[band_keep]

    theta = (np.degrees(np.arctan2(ring_points[:, 0] - center[0], ring_points[:, 1] - center[1])) + 360.0) % 360.0
    angular_bins = 72
    occupied = np.unique(np.floor(theta / (360.0 / angular_bins)).astype(int)) if len(theta) else []
    angular_coverage = float(len(occupied) / angular_bins)
    area_ratio = float(len(ring_points) / max(len(points), 1))
    radial_std = float(ring_radial.std()) if len(ring_points) else 0.0

    debug.update(
        reason="candidate",
        candidate_area=int(len(ring_points)),
        angular_coverage=angular_coverage,
        radial_mean=float(ring_radial.mean()) if len(ring_points) else 0.0,
        radial_std=radial_std,
        radial_band_center=band_center,
        area_ratio=area_ratio,
    )
    if len(ring_points) < min_area:
        debug["reason"] = "too_few_ring_points"
        return None, ring_mask, debug
    if angular_coverage < min_angular_coverage:
        debug["reason"] = "low_angular_coverage"
        return None, ring_mask, debug
    if area_ratio < min_ring_area_ratio:
        debug["reason"] = "low_ring_area_ratio"
        return None, ring_mask, debug
    if radial_std > max_radial_std:
        debug["reason"] = "high_radial_std"
        return None, ring_mask, debug

    ring_pixels = ring_points.astype(np.int64)
    ring_mask[ring_pixels[:, 0], ring_pixels[:, 1]] = True
    token = _token_stats(
        ring_pixels,
        weight_map,
        valid_mask if valid_mask is not None else np.ones_like(mask, dtype=bool),
        total_mass=float(weight_map[mask].sum()),
        source=source,
    )
    token.update(
        proposal_source="retrieval_compact",
        proposal_type="ring_band",
        geometry_type="edge_ring",
        radial_mean=debug["radial_mean"],
        radial_std=debug["radial_std"],
        radial_band_center=band_center,
        angular_coverage=angular_coverage,
    )
    debug["accepted"] = True
    debug["reason"] = "accepted"
    return token, ring_mask, debug


def _retrieval_component_tokens(
    mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    min_area: int,
    source: str,
) -> List[Dict]:
    tokens = []
    total_mass = float(weight_map[mask].sum())
    for comp in _connected_components(mask):
        if len(comp) < min_area:
            continue
        token = _token_stats(comp, weight_map, valid_mask, total_mass=total_mass, source=source)
        token["proposal_source"] = "retrieval_compact"
        token["proposal_type"] = "component"
        token["geometry_type"] = _classify_component(token)
        tokens.append(token)
    tokens.sort(key=lambda item: item.get("area", 0), reverse=True)
    return tokens


def _classify_component(item: Dict) -> str:
    area = max(item.get("area", 1), 1)
    bbox_area = max(item.get("bbox_height", 1) * item.get("bbox_width", 1), 1)
    fill_ratio = area / bbox_area
    elongation = item.get("pca_lambda1", 0.0) / max(item.get("pca_lambda2", 0.0), 1e-6)
    aspect = max(
        item.get("bbox_height", 1) / max(item.get("bbox_width", 1), 1),
        item.get("bbox_width", 1) / max(item.get("bbox_height", 1), 1),
    )
    if elongation >= 6.0 or aspect >= 4.0:
        return "line"
    if fill_ratio >= 0.45 and item.get("compactness", 0.0) <= 1.6:
        return "blob"
    if item.get("radial_distance_norm", 1.0) <= 0.35:
        return "central"
    return "irregular"


def _select_retrieval_tokens(
    ring_token: Optional[Dict],
    component_tokens: List[Dict],
    top_k: int,
    min_residual_types: int = 3,
) -> List[Dict]:
    selected = []
    if ring_token is not None and top_k > 0:
        selected.append(ring_token)

    remaining_slots = max(top_k - len(selected), 0)
    if remaining_slots == 0 or not component_tokens:
        return selected

    by_type: Dict[str, List[Dict]] = {}
    for token in component_tokens:
        by_type.setdefault(token.get("geometry_type", "irregular"), []).append(token)
    for items in by_type.values():
        items.sort(key=_residual_importance, reverse=True)

    type_order = sorted(
        by_type.keys(),
        key=lambda geometry_type: _type_group_priority(geometry_type, by_type[geometry_type][0]),
        reverse=True,
    )
    residual_selected = []
    used_ids = set()
    for geometry_type in type_order:
        if len(residual_selected) >= min(remaining_slots, min_residual_types):
            break
        token = by_type[geometry_type][0]
        residual_selected.append(token)
        used_ids.add(id(token))

    leftovers = [token for token in component_tokens if id(token) not in used_ids]
    leftovers.sort(key=_residual_importance, reverse=True)
    for token in leftovers:
        if len(residual_selected) >= remaining_slots:
            break
        residual_selected.append(token)

    residual_selected.sort(key=_display_order, reverse=True)
    return selected + residual_selected[:remaining_slots]


def _residual_importance(item: Dict) -> float:
    area_score = np.sqrt(max(item.get("area", 0), 0))
    radial = item.get("radial_distance_norm", 0.5)
    central_bonus = 2.0 if radial <= 0.35 else 0.0
    fill = item.get("area", 0) / max(item.get("bbox_height", 1) * item.get("bbox_width", 1), 1)
    structure_bonus = {
        "central": 2.5,
        "blob": 1.5,
        "line": 1.2,
        "irregular": 0.8,
    }.get(item.get("geometry_type"), 0.8)
    return float(area_score + central_bonus + structure_bonus + min(fill, 1.0))


def _type_group_priority(geometry_type: str, representative: Dict) -> float:
    base = {
        "central": 5.0,
        "blob": 4.0,
        "line": 3.5,
        "irregular": 3.0,
    }.get(geometry_type, 3.0)
    return base + 0.01 * _residual_importance(representative)


def _display_order(item: Dict) -> float:
    order = {
        "central": 5.0,
        "blob": 4.0,
        "line": 3.5,
        "irregular": 3.0,
    }.get(item.get("geometry_type"), 3.0)
    return order + 0.01 * item.get("area", 0)


def _finalize_token(token: Dict, map_shape: tuple[int, int], proposal_config: ProposalConfig) -> None:
    token["descriptor"] = _shape_descriptor(
        token,
        map_shape,
        mode=proposal_config.descriptor_mode,
        rotation_tolerance=proposal_config.rotation_tolerance,
    )
    token["proposal_config"] = {
        "min_area": proposal_config.min_area,
        "top_k": proposal_config.top_k,
        "connectivity": proposal_config.connectivity,
        "descriptor_mode": proposal_config.descriptor_mode,
        "proposal_mode": proposal_config.proposal_mode,
        "rotation_tolerance": proposal_config.rotation_tolerance,
    }


def _proposal_config(
    shape: tuple[int, int],
    valid_area: int,
    min_area: int,
    top_k: int,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
) -> ProposalConfig:
    proposal_mode = proposal_mode.lower().strip()
    if proposal_mode not in {"cc", "compact"}:
        raise ValueError(f"Unsupported count-partial proposal mode: {proposal_mode}")
    short_side = min(int(shape[0]), int(shape[1]))
    valid_area = max(int(valid_area), 1)

    if short_side <= 12:
        adaptive_min_area = 2
        adaptive_top_k = 4
        descriptor_mode = "coarse"
    elif short_side <= 25:
        adaptive_min_area = max(3, int(round(valid_area * 0.01)))
        adaptive_top_k = 6
        descriptor_mode = "normal"
    else:
        adaptive_min_area = max(5, int(round(valid_area * 0.005)))
        adaptive_top_k = 8
        descriptor_mode = "normal"

    return ProposalConfig(
        min_area=max(1, min(int(min_area), adaptive_min_area)),
        top_k=max(1, min(int(top_k), adaptive_top_k)),
        connectivity=8,
        descriptor_mode=descriptor_mode,
        proposal_mode=proposal_mode,
        rotation_tolerance=bool(rotation_tolerance),
    )


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
