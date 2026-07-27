from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .morphology import _connected_components
from .proposal_utils import (
    SPARSE_RING_EDGE_R_MIN,
    SPARSE_RING_MAX_ANGULAR_BINS,
    SPARSE_RING_MAX_RADIAL_STD,
    SPARSE_RING_MIN_ANGULAR_BINS,
    SPARSE_RING_MIN_ARC_BINS,
    SPARSE_RING_MIN_ARC_COVERAGE,
    SPARSE_RING_MIN_FULL_COVERAGE,
    SPARSE_RING_RADIUS_TOL_CELLS,
    _chain_pixels,
    _finalize_token,
    _pixel_iou,
    _token_stats,
    _unique_pixel_array,
    _wafer_center_and_radius,
)


def _tokens_from_sparse_density(
    impulse_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config,
    source: str,
    raw_weight_map: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Create comparable WBM/WDM tokens from a multi-scale sparse density field."""
    density_weights = np.where(valid_mask, np.maximum(impulse_map, 0.0), 0.0).astype(np.float32)
    raw_weights = density_weights if raw_weight_map is None else np.where(
        valid_mask, np.maximum(raw_weight_map, 0.0), 0.0
    ).astype(np.float32)
    raw_points = raw_weights > 0
    total_raw_mass = float(raw_weights.sum())
    if not raw_points.any() or total_raw_mass <= 0.0:
        return []

    tokens: List[Dict] = []
    for sigma in proposal_config.density_sigmas:
        sigma_tokens: List[Dict] = []
        density = _masked_gaussian_density(density_weights, valid_mask, sigma)
        support = _density_support_mask(density, valid_mask, proposal_config.density_threshold)
        total_density_mass = float(density[support].sum())
        for comp in _connected_components(support, connectivity=proposal_config.connectivity):
            rows = comp[:, 0].astype(np.int64)
            cols = comp[:, 1].astype(np.int64)
            component_raw = raw_weights[rows, cols]
            raw_point_count = int((component_raw > 0).sum())
            raw_mass = float(component_raw.sum())
            if raw_point_count < proposal_config.density_min_raw_points:
                continue
            if raw_mass < proposal_config.density_min_raw_mass:
                continue
            # KDE support groups nearby defects, but it must not become the
            # matched region: all token geometry is computed from raw defects.
            raw_component = comp[component_raw > 0]
            token = _token_stats(raw_component, raw_weights, valid_mask, total_mass=total_raw_mass, source=source)
            token.update(
                proposal_source="sparse_density",
                proposal_type="density_support",
                proposal_scale=float(sigma),
                raw_mass=raw_mass,
                raw_point_count=raw_point_count,
                raw_pixels=[(int(r), int(c)) for r, c in raw_component],
                kde_support_pixels=[(int(r), int(c)) for r, c in comp],
                kde_support_area=int(len(comp)),
                kde_support_mass=total_density_mass,
                density_peak=float(density[rows, cols].max()),
            )
            _annotate_sparse_ring_arc(token, valid_mask)
            sigma_tokens.append(token)
        tokens.extend(_merge_sparse_density_ring_arcs(sigma_tokens, raw_weights, valid_mask, total_raw_mass, source))

    tokens = _deduplicate_density_tokens(tokens, proposal_config.density_merge_iou)
    tokens.sort(key=_density_token_importance, reverse=True)
    tokens = tokens[:proposal_config.top_k]
    for token in tokens:
        _finalize_token(token, impulse_map.shape, proposal_config)
    return tokens


def _masked_gaussian_density(
    weights: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Add per-point, valid-region-normalized Gaussian kernels without SciPy."""
    if sigma <= 0.0:
        raise ValueError("density sigma must be positive")
    h, w = weights.shape
    radius = max(1, int(np.ceil(3.0 * sigma)))
    density = np.zeros_like(weights, dtype=np.float32)
    for row, col in np.argwhere(weights > 0):
        r0, r1 = max(0, row - radius), min(h, row + radius + 1)
        c0, c1 = max(0, col - radius), min(w, col + radius + 1)
        rr = np.arange(r0, r1, dtype=np.float32) - float(row)
        cc = np.arange(c0, c1, dtype=np.float32) - float(col)
        kernel = np.exp(-(rr[:, None] ** 2 + cc[None, :] ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        local_valid = valid_mask[r0:r1, c0:c1]
        kernel *= local_valid
        kernel_sum = float(kernel.sum())
        if kernel_sum > 1e-8:
            density[r0:r1, c0:c1] += float(weights[row, col]) * kernel / kernel_sum
    return density


def _density_support_mask(density: np.ndarray, valid_mask: np.ndarray, threshold: float) -> np.ndarray:
    peak = float(density[valid_mask].max()) if valid_mask.any() else 0.0
    if peak <= 0.0:
        return np.zeros_like(valid_mask, dtype=bool)
    return valid_mask & (density >= max(float(threshold), 0.0) * peak)


def _annotate_sparse_ring_arc(token: Dict, valid_mask: np.ndarray) -> None:
    features = _sparse_ring_features(token.get("raw_pixels", token.get("pixels", [])), valid_mask)
    token.update(features)
    if not features.get("sparse_ring_arc_candidate", False):
        return
    coverage = float(features.get("ring_angular_coverage", 0.0))
    if coverage >= SPARSE_RING_MIN_FULL_COVERAGE:
        token.update(proposal_type="density_ring", geometry_type="edge_ring")
    elif coverage >= SPARSE_RING_MIN_ARC_COVERAGE:
        token.update(proposal_type="density_ring_arc", geometry_type="ring_arc")


def _sparse_ring_features(pixels, valid_mask: np.ndarray) -> Dict:
    arr = np.asarray(pixels, dtype=np.int64)
    if arr.size == 0:
        return {"sparse_ring_arc_candidate": False}
    if arr.ndim != 2 or arr.shape[1] != 2:
        arr = arr.reshape(-1, 2)

    center, radius_ref = _wafer_center_and_radius(valid_mask)
    if radius_ref <= 1e-6:
        return {"sparse_ring_arc_candidate": False}

    rel = arr.astype(np.float32) - center
    radial = np.linalg.norm(rel, axis=1) / radius_ref
    radius = float(np.median(radial))
    bins = int(np.clip(
        np.ceil(2.0 * np.pi * radius * radius_ref),
        SPARSE_RING_MIN_ANGULAR_BINS,
        SPARSE_RING_MAX_ANGULAR_BINS,
    ))
    theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
    occupied = np.unique(np.floor(theta / (360.0 / bins)).astype(np.int64) % bins)
    radial_std = float(radial.std()) if len(radial) else 0.0
    coverage = float(len(occupied) / max(bins, 1))
    arc_candidate = (
        radius >= SPARSE_RING_EDGE_R_MIN
        and radial_std <= SPARSE_RING_MAX_RADIAL_STD
        and len(occupied) >= SPARSE_RING_MIN_ARC_BINS
    )
    return {
        "sparse_ring_arc_candidate": bool(arc_candidate),
        "ring_radius_norm": radius,
        "ring_radial_std": radial_std,
        "ring_angular_bins": bins,
        "ring_occupied_bins": occupied.astype(int).tolist(),
        "ring_angular_coverage": coverage,
    }


def _merge_sparse_density_ring_arcs(
    tokens: List[Dict],
    raw_weights: np.ndarray,
    valid_mask: np.ndarray,
    total_raw_mass: float,
    source: str,
) -> List[Dict]:
    arc_tokens = [token for token in tokens if token.get("sparse_ring_arc_candidate")]
    if not arc_tokens:
        return tokens

    _, radius_ref = _wafer_center_and_radius(valid_mask)
    if radius_ref <= 1e-6:
        return tokens

    groups = _group_sparse_ring_arcs_by_radius(arc_tokens, radius_ref)
    merged_tokens: List[Dict] = []
    used_ids = set()
    for group in groups:
        if len(group) < 2:
            continue
        raw_pixels = _unique_pixel_array(_chain_pixels(group, "raw_pixels"))
        if len(raw_pixels) == 0:
            continue
        merged_features = _sparse_ring_features(raw_pixels, valid_mask)
        coverage = float(merged_features.get("ring_angular_coverage", 0.0))
        if coverage < SPARSE_RING_MIN_ARC_COVERAGE:
            continue
        support_pixels = _unique_pixel_array(_chain_pixels(group, "kde_support_pixels"))
        merged = _token_stats(raw_pixels, raw_weights, valid_mask, total_mass=total_raw_mass, source=source)
        raw_rows = raw_pixels[:, 0].astype(np.int64)
        raw_cols = raw_pixels[:, 1].astype(np.int64)
        raw_mass = float(raw_weights[raw_rows, raw_cols].sum())
        geometry_type = "edge_ring" if coverage >= SPARSE_RING_MIN_FULL_COVERAGE else "ring_arc"
        merged.update(
            proposal_source="sparse_density_ring_merge",
            proposal_type="merged_ring" if geometry_type == "edge_ring" else "merged_ring_arc",
            proposal_scale=float(np.median([float(token.get("proposal_scale", 0.0)) for token in group])),
            geometry_type=geometry_type,
            raw_mass=raw_mass,
            raw_point_count=int((raw_weights[raw_rows, raw_cols] > 0).sum()),
            raw_pixels=[(int(r), int(c)) for r, c in raw_pixels],
            kde_support_pixels=[(int(r), int(c)) for r, c in support_pixels],
            kde_support_area=int(len(support_pixels)),
            kde_support_mass=float(sum(float(token.get("kde_support_mass", 0.0)) for token in group)),
            density_peak=float(max(float(token.get("density_peak", 0.0)) for token in group)),
            sparse_ring_arc_candidate=True,
            ring_radius_norm=float(merged_features.get("ring_radius_norm", 0.0)),
            ring_radial_std=float(merged_features.get("ring_radial_std", 0.0)),
            ring_angular_bins=int(merged_features.get("ring_angular_bins", SPARSE_RING_MIN_ANGULAR_BINS)),
            ring_occupied_bins=list(merged_features.get("ring_occupied_bins", [])),
            ring_angular_coverage=coverage,
            ring_arc_count=len(group),
        )
        merged_tokens.append(merged)
        used_ids.update(id(token) for token in group)

    if not merged_tokens:
        return tokens
    return merged_tokens + [token for token in tokens if id(token) not in used_ids]


def _group_sparse_ring_arcs_by_radius(tokens: List[Dict], radius_ref: float) -> List[List[Dict]]:
    radius_tol = max(SPARSE_RING_RADIUS_TOL_CELLS / max(radius_ref, 1e-6), 0.04)
    groups: List[List[Dict]] = []
    for token in sorted(tokens, key=lambda item: float(item.get("ring_radius_norm", 0.0))):
        radius = float(token.get("ring_radius_norm", 0.0))
        for group in groups:
            group_radius = float(np.median([float(item.get("ring_radius_norm", 0.0)) for item in group]))
            if abs(radius - group_radius) <= radius_tol:
                group.append(token)
                break
        else:
            groups.append([token])
    return groups


def _deduplicate_density_tokens(tokens: List[Dict], min_iou: float) -> List[Dict]:
    selected: List[Dict] = []
    for token in sorted(tokens, key=_density_token_importance, reverse=True):
        support_pixels = set(token.get("kde_support_pixels", []))
        raw_pixels = set(token.get("raw_pixels", []))
        duplicate = False
        for existing in selected:
            other = set(existing.get("kde_support_pixels", []))
            other_raw = set(existing.get("raw_pixels", []))
            support_iou = _pixel_iou(support_pixels, other)
            raw_iou = _pixel_iou(raw_pixels, other_raw)
            if support_iou >= min_iou or raw_iou >= min_iou:
                duplicate = True
                break
        if not duplicate:
            selected.append(token)
    return selected


def _density_token_importance(token: Dict) -> float:
    raw_mass = float(token.get("raw_mass", 0.0))
    area = float(token.get("area", 0.0))
    peak = float(token.get("density_peak", 0.0))
    ring_bonus = 0.0
    if token.get("geometry_type") == "edge_ring":
        ring_bonus = 2.0 * float(token.get("ring_angular_coverage", 0.0))
    elif token.get("geometry_type") == "ring_arc":
        ring_bonus = 0.5 * float(token.get("ring_angular_coverage", 0.0))
    return float(np.sqrt(max(raw_mass, 0.0)) + 0.25 * np.sqrt(max(area, 0.0)) + 0.05 * peak + ring_bonus)
