from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import ProposalConfig
from .morphology import (_binary_closing_square_constrained, _connected_components, _perimeter,)
from .descriptors import _classify_token, _shape_descriptor


def _tokens_from_mask(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    proposal_debug: Optional[Dict] = None,
) -> List[Dict]:
    weight_map = mask.astype(np.float32)
    if proposal_config.proposal_mode == "sparse-density":
        return _tokens_from_sparse_density(weight_map, valid_mask, proposal_config, source="wbm")
    return _tokens_from_components(
        mask,
        valid_mask,
        weight_map,
        proposal_config=proposal_config,
        source="wbm",
        proposal_debug=proposal_debug,
    )


def _tokens_from_count(count_map: np.ndarray, valid_mask: np.ndarray, proposal_config: ProposalConfig) -> List[Dict]:
    mask = (count_map > 0) & valid_mask
    return _tokens_from_weighted_mask(mask, valid_mask, count_map.astype(np.float32), proposal_config=proposal_config)


def _tokens_from_weighted_mask(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    weight_map: np.ndarray,
    proposal_config: ProposalConfig,
    raw_weight_map: Optional[np.ndarray] = None,
    proposal_debug: Optional[Dict] = None,
) -> List[Dict]:
    if proposal_config.proposal_mode == "sparse-density":
        return _tokens_from_sparse_density(
            weight_map,
            valid_mask,
            proposal_config,
            source="wdm",
            raw_weight_map=raw_weight_map,
        )
    return _tokens_from_components(
        mask,
        valid_mask,
        weight_map,
        proposal_config=proposal_config,
        source="wdm",
        proposal_debug=proposal_debug,
    )


def _tokens_from_sparse_density(
    impulse_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
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
            tokens.append(token)

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


def _pixel_iou(a: set, b: set) -> float:
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def _density_token_importance(token: Dict) -> float:
    raw_mass = float(token.get("raw_mass", 0.0))
    area = float(token.get("area", 0.0))
    peak = float(token.get("density_peak", 0.0))
    return float(np.sqrt(max(raw_mass, 0.0)) + 0.25 * np.sqrt(max(area, 0.0)) + 0.05 * peak)


def _tokens_from_components(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    weight_map: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
    proposal_debug: Optional[Dict] = None,
) -> List[Dict]:
    h, w = mask.shape
    if proposal_config.proposal_mode == "compact":
        tokens, ring_debug = _retrieval_compact_tokens(mask & valid_mask, weight_map, valid_mask, proposal_config, source=source)
        if proposal_debug is not None:
            proposal_debug[source] = ring_debug
        for token in tokens:
            _finalize_token(token, (h, w), proposal_config)
        return tokens
    if proposal_config.proposal_mode == "tangential-ring":
        tokens, ring_debug = _tangential_ring_tokens(mask & valid_mask, weight_map, valid_mask, proposal_config, source=source)
        if proposal_debug is not None:
            proposal_debug[source] = ring_debug
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
) -> Tuple[List[Dict], Dict]:
    h, w = weight_map.shape
    original = mask & valid_mask
    denoised = np.zeros_like(original, dtype=bool)
    for comp in _connected_components(original, connectivity=proposal_config.connectivity):
        if len(comp) >= proposal_config.min_area:
            denoised[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True

    ring_input = _small_map_ring_input(original, valid_mask, weight_map.shape)
    ring_token, ring_mask, ring_debug = _extract_retrieval_ring_token(
        ring_input,
        weight_map,
        raw_mask=original,
        valid_mask=valid_mask,
        source=source,
        min_area=proposal_config.ring_min_area,
        edge_r_min=proposal_config.ring_edge_r_min,
        band_width=proposal_config.ring_band_width,
        min_angular_coverage=proposal_config.ring_min_angular_coverage,
        angular_bins=proposal_config.ring_angular_bins,
        max_radial_std=proposal_config.ring_max_radial_std,
        max_defect_ratio=proposal_config.ring_max_defect_ratio,
        min_edge_defect_fraction=proposal_config.ring_min_edge_defect_fraction,
    )
    residual = denoised & (~ring_mask)
    component_tokens = _retrieval_component_tokens(
        residual,
        weight_map,
        valid_mask,
        min_area=proposal_config.min_area,
        source=source,
    )
    ring_debug.update(
        source=source,
        original_area=int(original.sum()),
        denoised_area=int(denoised.sum()),
        ring_input_area=int(ring_input.sum()),
    )
    return _select_retrieval_tokens(ring_token, component_tokens, proposal_config.top_k), ring_debug


def _tangential_ring_tokens(
    mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
) -> Tuple[List[Dict], Dict]:
    """Extract a high-recall ring from raw points with tangential-only gap bridging."""
    original = mask & valid_mask
    ring_token, ring_mask, ring_debug = _extract_tangential_ring_token(
        original,
        weight_map,
        valid_mask,
        source=source,
        edge_r_min=proposal_config.ring_edge_r_min,
        max_defect_ratio=proposal_config.ring_max_defect_ratio,
    )

    denoised = np.zeros_like(original, dtype=bool)
    for comp in _connected_components(original, connectivity=proposal_config.connectivity):
        if len(comp) >= proposal_config.min_area:
            denoised[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True
    component_tokens = _retrieval_component_tokens(
        denoised & (~ring_mask),
        weight_map,
        valid_mask,
        min_area=proposal_config.min_area,
        source=source,
    )
    ring_debug.update(source=source, original_area=int(original.sum()), denoised_area=int(denoised.sum()))
    return _select_retrieval_tokens(ring_token, component_tokens, proposal_config.top_k), ring_debug


def _extract_tangential_ring_token(
    raw_mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    source: str,
    edge_r_min: float,
    max_defect_ratio: float,
    max_gap_cells: float = 2.0,
    max_half_width_cells: float = 2.0,
) -> Tuple[Optional[Dict], np.ndarray, Dict]:
    """Bridge only short angular gaps; the returned mask always contains raw pixels."""
    h, w = raw_mask.shape
    empty_mask = np.zeros_like(raw_mask, dtype=bool)
    points = np.argwhere(raw_mask).astype(np.int64)
    debug = {"accepted": False, "reason": "no_points", "raw_ring_area": 0}
    if len(points) < 3:
        return None, empty_mask, debug

    center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    valid_points = np.argwhere(valid_mask).astype(np.float32)
    radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max()) if len(valid_points) else 0.0
    if radius_ref <= 1e-6:
        debug["reason"] = "bad_radius"
        return None, empty_mask, debug

    defect_ratio = float(len(points) / max(int(valid_mask.sum()), 1))
    if defect_ratio > max_defect_ratio:
        debug.update(reason="skip_high_defect_ratio", defect_ratio=defect_ratio)
        return None, empty_mask, debug

    rel = points.astype(np.float32) - center
    radial = np.linalg.norm(rel, axis=1) / radius_ref
    outer = radial >= edge_r_min
    if int(outer.sum()) < 3:
        debug.update(reason="too_few_outer_points", defect_ratio=defect_ratio)
        return None, empty_mask, debug

    outer_points = points[outer]
    outer_radial = radial[outer]
    outer_weights = np.maximum(weight_map[outer_points[:, 0], outer_points[:, 1]], 1e-6)
    radial_cell = 1.0 / radius_ref
    bin_count = max(1, int(np.ceil((1.05 - edge_r_min) / radial_cell)))
    hist, edges = np.histogram(outer_radial, bins=bin_count, range=(edge_r_min, 1.05), weights=outer_weights)
    peak = int(hist.argmax())
    seed = (outer_radial >= edges[peak]) & (outer_radial <= edges[peak + 1])
    if not seed.any():
        debug["reason"] = "no_radial_peak"
        return None, empty_mask, debug

    ring_radius = float(np.median(outer_radial[seed]))
    radial_mad = float(np.median(np.abs(outer_radial[seed] - ring_radius)))
    half_width = min(max_half_width_cells * radial_cell, max(radial_cell, 2.0 * radial_mad))
    in_band = np.abs(outer_radial - ring_radius) <= half_width
    ring_points = outer_points[in_band]
    if len(ring_points) < 3:
        debug.update(reason="too_few_band_points", ring_radius=ring_radius, half_width_cells=half_width * radius_ref)
        return None, empty_mask, debug

    angular_bins = int(np.clip(np.ceil(2.0 * np.pi * ring_radius * radius_ref), 12, 144))
    theta = (np.degrees(np.arctan2(ring_points[:, 0] - center[0], ring_points[:, 1] - center[1])) + 360.0) % 360.0
    bin_ids = np.floor(theta / (360.0 / angular_bins)).astype(np.int64) % angular_bins
    occupied = np.zeros(angular_bins, dtype=bool)
    occupied[bin_ids] = True
    arc_cells_per_bin = 2.0 * np.pi * ring_radius * radius_ref / angular_bins
    max_gap_bins = max(1, int(np.floor(max_gap_cells / max(arc_cells_per_bin, 1e-6))))
    contour = _bridge_short_circular_gaps(occupied, max_gap_bins)
    raw_coverage = float(occupied.mean())
    contour_coverage = float(contour.mean())
    bridged_gap_bins = int(contour.sum() - occupied.sum())
    min_coverage = min(0.08, max(3.0 / angular_bins, 0.0))
    if contour_coverage < min_coverage:
        debug.update(reason="low_tangential_coverage", raw_angular_coverage=raw_coverage, contour_angular_coverage=contour_coverage)
        return None, empty_mask, debug

    ring_mask = np.zeros_like(raw_mask, dtype=bool)
    ring_mask[ring_points[:, 0], ring_points[:, 1]] = True
    token = _token_stats(
        ring_points,
        weight_map,
        valid_mask,
        total_mass=float(weight_map[raw_mask].sum()),
        source=source,
    )
    token.update(
        proposal_source="tangential_ring",
        proposal_type="raw_radial_band",
        geometry_type="edge_ring",
        ring_radius_norm=ring_radius,
        ring_half_width_cells=float(half_width * radius_ref),
        ring_raw_angular_coverage=raw_coverage,
        ring_contour_angular_coverage=contour_coverage,
        ring_angular_bins=angular_bins,
        ring_max_tangential_gap_cells=float(max_gap_bins * arc_cells_per_bin),
        ring_bridged_gap_bins=bridged_gap_bins,
        ring_contour_bins=np.flatnonzero(contour).astype(int).tolist(),
    )
    debug.update(
        accepted=True,
        reason="accepted",
        defect_ratio=defect_ratio,
        raw_ring_area=int(len(ring_points)),
        ring_radius_norm=ring_radius,
        ring_half_width_cells=float(half_width * radius_ref),
        raw_angular_coverage=raw_coverage,
        contour_angular_coverage=contour_coverage,
        angular_bins=angular_bins,
        bridged_gap_bins=bridged_gap_bins,
    )
    return token, ring_mask, debug


def _bridge_short_circular_gaps(occupied: np.ndarray, max_gap_bins: int) -> np.ndarray:
    """Mark gaps bounded by occupied angular bins without creating spatial pixels."""
    bridged = occupied.copy()
    count = len(bridged)
    if count == 0 or not occupied.any():
        return bridged
    for start in range(count):
        if occupied[start] or not occupied[(start - 1) % count]:
            continue
        gap_len = 0
        while gap_len < count and not occupied[(start + gap_len) % count]:
            gap_len += 1
        if gap_len <= max_gap_bins and gap_len < count and occupied[(start + gap_len) % count]:
            bridged[(start + np.arange(gap_len)) % count] = True
    return bridged


def _small_map_ring_input(mask: np.ndarray, valid_mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if min(int(shape[0]), int(shape[1])) > 12:
        return mask
    return _binary_closing_square_constrained(mask, valid_mask)


def _extract_retrieval_ring_token(
    mask: np.ndarray,
    weight_map: np.ndarray,
    raw_mask: np.ndarray,
    valid_mask: Optional[np.ndarray],
    source: str,
    min_area: int = 12,
    edge_r_min: float = 0.65,
    band_width: float = 0.10,
    min_angular_coverage: float = 0.16,
    min_ring_area_ratio: float = 0.12,
    angular_bins: int = 72,
    max_radial_std: float = 0.12,
    max_defect_ratio: float = 0.45,
    min_edge_defect_fraction: float = 0.45,
) -> Tuple[Optional[Dict], np.ndarray, Dict]:
    h, w = mask.shape
    contour_mask = np.zeros_like(mask, dtype=bool)
    raw_ring_mask = np.zeros_like(mask, dtype=bool)
    points = np.argwhere(mask).astype(np.float32)
    debug = {"accepted": False, "reason": "no_points", "candidate_area": 0, "angular_coverage": 0.0}
    if len(points) < min_area:
        return None, raw_ring_mask, debug

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
        return None, raw_ring_mask, debug

    rel = points - center
    radial = np.linalg.norm(rel, axis=1) / radius_ref
    defect_ratio = float(len(points) / max(valid_area, 1))
    edge_keep = radial >= edge_r_min
    edge_fraction = float(edge_keep.sum() / max(len(points), 1))
    debug.update(defect_ratio=defect_ratio, edge_fraction=edge_fraction)
    if defect_ratio > max_defect_ratio:
        debug["reason"] = "skip_high_defect_ratio"
        return None, raw_ring_mask, debug
    if edge_fraction < min_edge_defect_fraction:
        debug["reason"] = "skip_low_edge_fraction"
        return None, raw_ring_mask, debug

    edge_points = points[edge_keep]
    edge_radial = radial[edge_keep]
    if len(edge_points) < min_area:
        debug.update(reason="too_few_edge_points", candidate_area=int(len(edge_points)))
        return None, raw_ring_mask, debug

    hist, edges = np.histogram(edge_radial, bins=12, range=(edge_r_min, 1.05))
    peak = int(hist.argmax())
    band_center = float((edges[peak] + edges[peak + 1]) / 2.0)
    band_keep = np.abs(edge_radial - band_center) <= band_width
    ring_points = edge_points[band_keep]
    ring_radial = edge_radial[band_keep]

    theta = (np.degrees(np.arctan2(ring_points[:, 0] - center[0], ring_points[:, 1] - center[1])) + 360.0) % 360.0
    angular_bins = max(int(angular_bins), 1)
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
        return None, raw_ring_mask, debug
    if angular_coverage < min_angular_coverage:
        debug["reason"] = "low_angular_coverage"
        return None, raw_ring_mask, debug
    if area_ratio < min_ring_area_ratio:
        debug["reason"] = "low_ring_area_ratio"
        return None, raw_ring_mask, debug
    if radial_std > max_radial_std:
        debug["reason"] = "high_radial_std"
        return None, raw_ring_mask, debug

    contour_pixels = ring_points.astype(np.int64)
    contour_mask[contour_pixels[:, 0], contour_pixels[:, 1]] = True
    raw_ring_mask = contour_mask & raw_mask
    raw_ring_pixels = np.argwhere(raw_ring_mask).astype(np.int64)
    if not len(raw_ring_pixels):
        debug["reason"] = "no_raw_ring_points"
        return None, raw_ring_mask, debug

    token = _token_stats(
        raw_ring_pixels,
        weight_map,
        valid_mask if valid_mask is not None else np.ones_like(mask, dtype=bool),
        total_mass=float(weight_map[raw_mask].sum()),
        source=source,
    )
    token.update(
        proposal_source="retrieval_compact",
        proposal_type="ring_band",
        geometry_type="edge_ring",
        ring_contour_pixels=[(int(r), int(c)) for r, c in contour_pixels],
        ring_contour_area=int(len(contour_pixels)),
        ring_contour_radial_mean=debug["radial_mean"],
        ring_contour_radial_std=debug["radial_std"],
        ring_contour_angular_coverage=angular_coverage,
        radial_band_center=band_center,
    )
    debug.update(raw_ring_area=int(len(raw_ring_pixels)), contour_area=int(len(contour_pixels)))
    debug["accepted"] = True
    debug["reason"] = "accepted"
    return token, raw_ring_mask, debug


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
        "density_sigmas": proposal_config.density_sigmas,
        "density_threshold": proposal_config.density_threshold,
        "ring_min_area": proposal_config.ring_min_area,
        "ring_edge_r_min": proposal_config.ring_edge_r_min,
        "ring_band_width": proposal_config.ring_band_width,
        "ring_min_angular_coverage": proposal_config.ring_min_angular_coverage,
        "ring_angular_bins": proposal_config.ring_angular_bins,
        "ring_max_radial_std": proposal_config.ring_max_radial_std,
        "ring_max_defect_ratio": proposal_config.ring_max_defect_ratio,
        "ring_min_edge_defect_fraction": proposal_config.ring_min_edge_defect_fraction,
    }


def _proposal_config(
    shape: tuple[int, int],
    valid_area: int,
    min_area: int,
    top_k: int,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    density_sigmas: tuple[float, ...] = (0.8, 1.6, 3.2),
    density_threshold: float = 0.20,
    density_min_raw_points: int = 3,
    density_min_raw_mass: float = 3.0,
    density_merge_iou: float = 0.60,
    density_weight_transform: str = "sqrt",
    ring_min_area: Optional[int] = None,
    ring_edge_r_min: Optional[float] = None,
    ring_band_width: Optional[float] = None,
    ring_min_angular_coverage: Optional[float] = None,
    ring_angular_bins: Optional[int] = None,
    ring_max_radial_std: Optional[float] = None,
    ring_max_defect_ratio: Optional[float] = None,
    ring_min_edge_defect_fraction: Optional[float] = None,
) -> ProposalConfig:
    proposal_mode = proposal_mode.lower().strip()
    if proposal_mode not in {"cc", "compact", "tangential-ring", "sparse-density"}:
        raise ValueError(f"Unsupported count-partial proposal mode: {proposal_mode}")
    density_sigmas = tuple(float(sigma) for sigma in density_sigmas)
    if not density_sigmas or any(sigma <= 0.0 for sigma in density_sigmas):
        raise ValueError("density_sigmas must contain positive values")
    if density_weight_transform not in {"count", "sqrt", "log1p"}:
        raise ValueError(f"Unsupported density weight transform: {density_weight_transform}")
    short_side = min(int(shape[0]), int(shape[1]))
    valid_area = max(int(valid_area), 1)

    if short_side <= 12:
        adaptive_min_area = 2
        adaptive_top_k = 4
        descriptor_mode = "coarse"
        ring_defaults = (6, 0.65, 0.10, 0.10, 24, 0.18, 0.60, 0.45)
    elif short_side <= 25:
        adaptive_min_area = max(3, int(round(valid_area * 0.01)))
        adaptive_top_k = 6
        descriptor_mode = "normal"
        ring_defaults = (12, 0.65, 0.10, 0.16, 72, 0.12, 0.45, 0.45)
    else:
        adaptive_min_area = max(5, int(round(valid_area * 0.005)))
        adaptive_top_k = 8
        descriptor_mode = "normal"
        ring_defaults = (12, 0.65, 0.10, 0.16, 72, 0.12, 0.45, 0.45)

    (
        default_ring_min_area,
        default_ring_edge_r_min,
        default_ring_band_width,
        default_ring_min_angular_coverage,
        default_ring_angular_bins,
        default_ring_max_radial_std,
        default_ring_max_defect_ratio,
        default_ring_min_edge_defect_fraction,
    ) = ring_defaults
    ring_min_area = default_ring_min_area if ring_min_area is None else ring_min_area
    ring_edge_r_min = default_ring_edge_r_min if ring_edge_r_min is None else ring_edge_r_min
    ring_band_width = default_ring_band_width if ring_band_width is None else ring_band_width
    ring_min_angular_coverage = default_ring_min_angular_coverage if ring_min_angular_coverage is None else ring_min_angular_coverage
    ring_angular_bins = default_ring_angular_bins if ring_angular_bins is None else ring_angular_bins
    ring_max_radial_std = default_ring_max_radial_std if ring_max_radial_std is None else ring_max_radial_std
    ring_max_defect_ratio = default_ring_max_defect_ratio if ring_max_defect_ratio is None else ring_max_defect_ratio
    ring_min_edge_defect_fraction = (
        default_ring_min_edge_defect_fraction
        if ring_min_edge_defect_fraction is None
        else ring_min_edge_defect_fraction
    )

    return ProposalConfig(
        min_area=max(1, min(int(min_area), adaptive_min_area)),
        top_k=max(1, min(int(top_k), adaptive_top_k)),
        connectivity=8,
        descriptor_mode=descriptor_mode,
        proposal_mode=proposal_mode,
        rotation_tolerance=bool(rotation_tolerance),
        density_sigmas=tuple(sorted(density_sigmas)),
        density_threshold=max(float(density_threshold), 0.0),
        density_min_raw_points=max(int(density_min_raw_points), 1),
        density_min_raw_mass=max(float(density_min_raw_mass), 0.0),
        density_merge_iou=min(max(float(density_merge_iou), 0.0), 1.0),
        density_weight_transform=density_weight_transform,
        ring_min_area=max(int(ring_min_area), 1),
        ring_edge_r_min=min(max(float(ring_edge_r_min), 0.0), 1.0),
        ring_band_width=max(float(ring_band_width), 0.0),
        ring_min_angular_coverage=min(max(float(ring_min_angular_coverage), 0.0), 1.0),
        ring_angular_bins=max(int(ring_angular_bins), 1),
        ring_max_radial_std=max(float(ring_max_radial_std), 0.0),
        ring_max_defect_ratio=min(max(float(ring_max_defect_ratio), 0.0), 1.0),
        ring_min_edge_defect_fraction=min(max(float(ring_min_edge_defect_fraction), 0.0), 1.0),
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
