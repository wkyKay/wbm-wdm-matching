# -*- coding: utf-8 -*-
"""
Retrieval-oriented compact cluster proposal.

This proposal is not intended to be semantic segmentation. It produces a small
set of stable local tokens for handcrafted local retrieval:

  defect mask -> optional ring-aware token -> residual components -> compact top-k
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from partial_match.core.clustering import _compute_cluster_stats, _connected_components


def retrieval_compact_proposal(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_area: int = 5,
    top_k: int = 6,
    edge_r_min: float = 0.65,
    ring_band_width: float = 0.10,
    min_ring_area: int = 12,
    min_ring_angular_coverage: float = 0.16,
    min_ring_area_ratio: float = 0.12,
    max_ring_radial_std: float = 0.12,
    max_defect_ratio_for_ring: float = 0.45,
    min_edge_defect_fraction_for_ring: float = 0.45,
    min_residual_types: int = 3,
    enable_ring_aware: bool = True,
    keep_ring_pixels_in_residual: bool = False,
    return_steps: bool = False,
) -> List[Dict] or Tuple[List[Dict], Dict[str, object]]:
    H, W = defect_mask.shape
    original = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()

    tiny_removed = np.zeros_like(original, dtype=bool)
    denoised = np.zeros_like(original, dtype=bool)
    for comp in _connected_components(original):
        if len(comp) >= min_area:
            denoised[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True
        else:
            tiny_removed[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True

    if enable_ring_aware:
        ring_token, ring_mask, ring_debug = _extract_ring_token(
            denoised,
            valid_mask=valid_mask,
            min_area=min_ring_area,
            edge_r_min=edge_r_min,
            band_width=ring_band_width,
            min_angular_coverage=min_ring_angular_coverage,
            min_ring_area_ratio=min_ring_area_ratio,
            max_radial_std=max_ring_radial_std,
            max_defect_ratio=max_defect_ratio_for_ring,
            min_edge_defect_fraction=min_edge_defect_fraction_for_ring,
        )
    else:
        ring_token = None
        ring_mask = np.zeros_like(denoised, dtype=bool)
        ring_debug = {'accepted': False, 'reason': 'disabled', 'candidate_area': 0, 'angular_coverage': 0.0}

    residual = denoised.copy() if keep_ring_pixels_in_residual else denoised & (~ring_mask)
    component_tokens = _component_tokens(residual, H, W, min_area=min_area)

    final_tokens = _select_final_tokens(
        ring_token=ring_token,
        component_tokens=component_tokens,
        top_k=top_k,
        min_residual_types=min_residual_types,
    )

    steps = {
        'original_mask': original,
        'tiny_removed_mask': tiny_removed,
        'denoised_mask': denoised,
        'ring_mask': ring_mask,
        'ring_debug': ring_debug,
        'residual_mask': residual,
        'component_tokens': component_tokens,
        'final_tokens': final_tokens,
    }
    if return_steps:
        return final_tokens, steps
    return final_tokens


def _extract_ring_token(
    mask: np.ndarray,
    valid_mask: Optional[np.ndarray],
    min_area: int,
    edge_r_min: float,
    band_width: float,
    min_angular_coverage: float,
    min_ring_area_ratio: float,
    max_radial_std: float,
    max_defect_ratio: float,
    min_edge_defect_fraction: float,
) -> Tuple[Optional[Dict], np.ndarray, Dict]:
    H, W = mask.shape
    ring_mask = np.zeros_like(mask, dtype=bool)
    points = np.argwhere(mask).astype(np.float32)
    debug = {
        'accepted': False,
        'reason': 'no_points',
        'candidate_area': 0,
        'angular_coverage': 0.0,
        'radial_mean': 0.0,
        'radial_std': 0.0,
    }
    if len(points) < min_area:
        return None, ring_mask, debug

    center = np.array([H / 2.0, W / 2.0], dtype=np.float32)
    if valid_mask is not None and valid_mask.any():
        valid_points = np.argwhere(valid_mask).astype(np.float32)
        radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max())
        valid_area = int(valid_mask.sum())
    else:
        radius_ref = float(max(np.linalg.norm(points - center, axis=1).max(), 1.0))
        valid_area = H * W
    if radius_ref <= 1e-6:
        debug['reason'] = 'bad_radius'
        return None, ring_mask, debug

    rel = points - center
    radial = np.linalg.norm(rel, axis=1) / radius_ref
    defect_ratio = float(len(points) / max(valid_area, 1))
    edge_keep = radial >= edge_r_min
    edge_fraction = float(edge_keep.sum() / max(len(points), 1))
    debug.update(defect_ratio=defect_ratio, edge_fraction=edge_fraction)
    if defect_ratio > max_defect_ratio:
        debug['reason'] = 'skip_high_defect_ratio'
        return None, ring_mask, debug
    if edge_fraction < min_edge_defect_fraction:
        debug['reason'] = 'skip_low_edge_fraction'
        return None, ring_mask, debug

    edge_points = points[edge_keep]
    edge_radial = radial[edge_keep]
    if len(edge_points) < min_area:
        debug.update(reason='too_few_edge_points', candidate_area=int(len(edge_points)))
        return None, ring_mask, debug

    hist, edges = np.histogram(edge_radial, bins=12, range=(edge_r_min, 1.05))
    peak = int(hist.argmax())
    band_center = float((edges[peak] + edges[peak + 1]) / 2.0)
    band_keep = np.abs(edge_radial - band_center) <= band_width
    ring_points = edge_points[band_keep]

    theta = (np.degrees(np.arctan2(ring_points[:, 0] - center[0], ring_points[:, 1] - center[1])) + 360.0) % 360.0
    angular_bins = 72
    occupied = np.unique(np.floor(theta / (360.0 / angular_bins)).astype(int)) if len(theta) else []
    angular_coverage = float(len(occupied) / angular_bins)
    area_ratio = float(len(ring_points) / max(len(points), 1))
    radial_std = float(edge_radial[band_keep].std()) if len(ring_points) else 0.0

    debug.update(
        reason='candidate',
        candidate_area=int(len(ring_points)),
        angular_coverage=angular_coverage,
        radial_mean=float(edge_radial[band_keep].mean()) if len(ring_points) else 0.0,
        radial_std=radial_std,
        radial_band_center=band_center,
        area_ratio=area_ratio,
    )

    if len(ring_points) < min_area:
        debug['reason'] = 'too_few_ring_points'
        return None, ring_mask, debug
    if angular_coverage < min_angular_coverage:
        debug['reason'] = 'low_angular_coverage'
        return None, ring_mask, debug
    if area_ratio < min_ring_area_ratio:
        debug['reason'] = 'low_ring_area_ratio'
        return None, ring_mask, debug
    if radial_std > max_radial_std:
        debug['reason'] = 'high_radial_std'
        return None, ring_mask, debug

    ring_mask[ring_points[:, 0].astype(int), ring_points[:, 1].astype(int)] = True
    token = _compute_cluster_stats(
        ring_points,
        H,
        W,
        proposal_source='retrieval_compact',
        proposal_type='ring_band',
        geometry_type='edge_ring',
        radial_mean=debug['radial_mean'],
        radial_std=debug['radial_std'],
        radial_band_center=band_center,
        angular_coverage=angular_coverage,
    )
    debug['accepted'] = True
    debug['reason'] = 'accepted'
    return token, ring_mask, debug


def _component_tokens(mask: np.ndarray, H: int, W: int, min_area: int) -> List[Dict]:
    tokens = []
    for comp in _connected_components(mask):
        if len(comp) < min_area:
            continue
        base = _compute_cluster_stats(
            comp,
            H,
            W,
            proposal_source='retrieval_compact',
        )
        geometry_type = _classify_component(base)
        base['proposal_type'] = 'component'
        base['geometry_type'] = geometry_type
        tokens.append(base)
    tokens.sort(key=lambda x: x.get('area', 0), reverse=True)
    return tokens


def _classify_component(item: Dict) -> str:
    area = max(item.get('area', 1), 1)
    bbox_area = max(item.get('bbox_height', 1) * item.get('bbox_width', 1), 1)
    fill_ratio = area / bbox_area
    elongation = item.get('pca_lambda1', 0.0) / max(item.get('pca_lambda2', 0.0), 1e-6)
    aspect = max(
        item.get('bbox_height', 1) / max(item.get('bbox_width', 1), 1),
        item.get('bbox_width', 1) / max(item.get('bbox_height', 1), 1),
    )
    if elongation >= 6.0 or aspect >= 4.0:
        return 'line'
    if fill_ratio >= 0.45 and item.get('compactness', 0.0) <= 1.6:
        return 'blob'
    if item.get('radial_distance_norm', 1.0) <= 0.35:
        return 'central'
    return 'irregular'


def _select_final_tokens(
    ring_token: Optional[Dict],
    component_tokens: List[Dict],
    top_k: int,
    min_residual_types: int = 3,
) -> List[Dict]:
    """
    Select final compact tokens without making scratch/line dominate residuals.

    Ring is handled as a special optional token. Residual components are selected
    with diversity first, then area/centrality. This keeps center, donut-like,
    blob and loc patterns from being dropped only because a line token has a
    higher type priority.
    """
    selected = []
    if ring_token is not None and top_k > 0:
        selected.append(ring_token)

    remaining_slots = max(top_k - len(selected), 0)
    if remaining_slots == 0 or not component_tokens:
        return selected

    by_type = {}
    for token in component_tokens:
        by_type.setdefault(token.get('geometry_type', 'irregular'), []).append(token)
    for items in by_type.values():
        items.sort(key=_residual_importance, reverse=True)

    type_order = sorted(
        by_type.keys(),
        key=lambda t: _type_group_priority(t, by_type[t][0]),
        reverse=True,
    )

    residual_selected = []
    used_ids = set()

    # Diversity pass: keep at most one strong token from each residual type.
    for geometry_type in type_order:
        if len(residual_selected) >= min(remaining_slots, min_residual_types):
            break
        token = by_type[geometry_type][0]
        residual_selected.append(token)
        used_ids.add(id(token))

    # Fill pass: add the strongest remaining tokens regardless of type.
    leftovers = [t for t in component_tokens if id(t) not in used_ids]
    leftovers.sort(key=_residual_importance, reverse=True)
    for token in leftovers:
        if len(residual_selected) >= remaining_slots:
            break
        residual_selected.append(token)

    residual_selected.sort(key=_display_order, reverse=True)
    return selected + residual_selected[:remaining_slots]


def _residual_importance(item: Dict) -> float:
    area_score = np.sqrt(max(item.get('area', 0), 0))
    radial = item.get('radial_distance_norm', 0.5)
    central_bonus = 2.0 if radial <= 0.35 else 0.0
    fill = item.get('area', 0) / max(item.get('bbox_height', 1) * item.get('bbox_width', 1), 1)
    structure_bonus = {
        'central': 2.5,
        'blob': 1.5,
        'line': 1.2,
        'irregular': 0.8,
    }.get(item.get('geometry_type'), 0.8)
    return float(area_score + central_bonus + structure_bonus + min(fill, 1.0))


def _type_group_priority(geometry_type: str, representative: Dict) -> float:
    base = {
        'central': 5.0,
        'blob': 4.0,
        'line': 3.5,
        'irregular': 3.0,
    }.get(geometry_type, 3.0)
    return base + 0.01 * _residual_importance(representative)


def _display_order(item: Dict) -> float:
    order = {
        'central': 5.0,
        'blob': 4.0,
        'line': 3.5,
        'irregular': 3.0,
    }.get(item.get('geometry_type'), 3.0)
    return order + 0.01 * item.get('area', 0)
