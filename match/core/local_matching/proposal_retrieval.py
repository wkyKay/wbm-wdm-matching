from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import ProposalConfig
from .morphology import _binary_closing_square_constrained, _connected_components
from .descriptors import _classify_token, _shape_descriptor
from .proposal_utils import (
    _wafer_center_and_radius,
    _circular_true_runs,
    _circular_true_run_indices,
    _circular_bins_between,
    _circular_arc_runs_with_gap_limits,
    _bridge_short_circular_gaps,
    _ring_break_stats,
    _effective_arc_band_width,
    _small_map_ring_input,
    _component_label_map,
    _arc_parent_component_fraction,
    _points_to_mask,
    _contour_points_for_raw_region,
    _chain_pixels,
    _unique_pixel_array,
    _component_min_chebyshev_distance,
    _region_parent_component_fraction,
    _token_stats,
    COMPACT_RING_ARC_MIN_ANGULAR_COVERAGE,
    COMPACT_RING_ARC_MAX_ANGULAR_COVERAGE,
    COMPACT_RING_ARC_ALLOWED_GAP_CELLS,
    COMPACT_RING_ARC_MAX_GAP_COUNT,
    COMPACT_RING_ARC_MIN_PARENT_FRACTION,
    COMPACT_ARC_MAX_ANGULAR_COVERAGE,
    COMPACT_ARC_ALLOWED_GAP_CELLS,
    COMPACT_ARC_MAX_GAP_COUNT,
    COMPACT_ARC_MAX_BAND_WIDTH_CELLS,
    COMPACT_ARC_MIN_PARENT_BAND_FRACTION,
    COMPACT_ARC_MIN_FULL_COVERAGE,
)



def _retrieval_compact_tokens(
    mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
) -> Tuple[List[Dict], Dict]:
    original = mask & valid_mask
    arc_mode = proposal_config.proposal_mode == "arc"
    arc_band_residual_mode = proposal_config.proposal_mode == "arc-band-residual"
    arc_ring_residual_mode = proposal_config.proposal_mode == "arc-ring-residual"
    denoised = np.zeros_like(original, dtype=bool)
    for comp in _connected_components(original, connectivity=proposal_config.connectivity):
        if len(comp) >= proposal_config.min_area:
            denoised[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True

    ring_input = _small_map_ring_input(original, valid_mask, weight_map.shape)
    ring_token: Optional[Dict] = None
    ring_mask = np.zeros_like(original, dtype=bool)
    ring_debug = {"accepted": False, "reason": "skipped_arc_mode" if arc_mode else "not_run"}
    if not (arc_mode or arc_band_residual_mode or arc_ring_residual_mode):
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
            connectivity=proposal_config.connectivity,
        )
    residual = denoised & (~ring_mask)
    arc_tokens: List[Dict] = []
    arc_mask = np.zeros_like(original, dtype=bool)
    arc_debug = {"accepted_count": 0, "reason": "skipped_ring_accepted" if ring_token is not None else "not_run"}
    if arc_mode or arc_band_residual_mode or arc_ring_residual_mode or ring_token is None:
        arc_input = _binary_closing_square_constrained(original, valid_mask) if (arc_mode or arc_band_residual_mode or arc_ring_residual_mode) else (original & (~ring_mask))
        arc_tokens, arc_mask, arc_debug = _extract_retrieval_arc_tokens(
            arc_input,
            weight_map,
            valid_mask,
            source=source,
            raw_mask=original,
            min_area=proposal_config.min_area,
            edge_r_min=proposal_config.ring_edge_r_min,
            band_width=proposal_config.ring_band_width,
            min_angular_coverage=COMPACT_RING_ARC_MIN_ANGULAR_COVERAGE,
            max_angular_coverage=COMPACT_ARC_MAX_ANGULAR_COVERAGE if arc_mode else COMPACT_RING_ARC_MAX_ANGULAR_COVERAGE,
            angular_bins=proposal_config.ring_angular_bins,
            max_radial_std=proposal_config.ring_max_radial_std,
            allowed_gap_cells=COMPACT_ARC_ALLOWED_GAP_CELLS if arc_mode else COMPACT_RING_ARC_ALLOWED_GAP_CELLS,
            max_gap_count=COMPACT_ARC_MAX_GAP_COUNT if arc_mode else COMPACT_RING_ARC_MAX_GAP_COUNT,
            min_parent_fraction=COMPACT_RING_ARC_MIN_PARENT_FRACTION,
            cc_aware=False,
            connectivity=proposal_config.connectivity,
            parent_fraction_mask=original if arc_band_residual_mode else None,
            proposal_source="arc_band_residual" if arc_band_residual_mode else "retrieval_compact",
        )
        if arc_ring_residual_mode:
            tokens, arc_ring_debug = _extract_arc_ring_residual_tokens(
                original,
                weight_map,
                valid_mask,
                proposal_config,
                source=source,
                arc_debug=arc_debug,
            )
            ring_debug.update(
                source=source,
                original_area=int(original.sum()),
                denoised_area=int(denoised.sum()),
                ring_input_area=int(ring_input.sum()),
                arc_detection=arc_debug,
                arc_ring_debug=arc_ring_debug,
            )
            return tokens, ring_debug
        residual = (original if arc_band_residual_mode else residual) & (~arc_mask)
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
        arc_detection=arc_debug,
    )
    if arc_band_residual_mode:
        return _select_arc_band_residual_tokens(arc_tokens, component_tokens, proposal_config.top_k), ring_debug
    return _select_retrieval_tokens(ring_token, arc_tokens + component_tokens, proposal_config.top_k), ring_debug


def _extract_arc_ring_residual_tokens(
    original: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
    arc_debug: Dict,
) -> Tuple[List[Dict], Dict]:
    center, _radius_ref = _wafer_center_and_radius(valid_mask)
    band_groups = arc_debug.get("band_groups_raw", [])
    ring_debug = {
        "band_pixels_raw": arc_debug.get("band_pixels_raw", []),
        "band_pixels": arc_debug.get("band_pixels", []),
        "band_groups": band_groups,
        "radial_band_center": arc_debug.get("radial_band_center"),
    }
    if not band_groups:
        component_tokens = _retrieval_component_tokens(
            original,
            weight_map,
            valid_mask,
            min_area=proposal_config.min_area,
            source=source,
        )
        for token in component_tokens:
            token["proposal_source"] = "arc_ring_residual"
        return component_tokens[:proposal_config.top_k], ring_debug

    angular_bins = 72
    min_angular_degrees = 45.0
    max_contact_ratio = 0.40
    parent_labels, parent_areas = _component_label_map(original, connectivity=proposal_config.connectivity)
    parent_components: Dict[int, np.ndarray] = {}
    for label_val, area in enumerate(parent_areas):
        if area >= proposal_config.min_area:
            parent_components[label_val] = np.argwhere(parent_labels == label_val)

    total_mass = float(weight_map[original].sum())
    arc_tokens: List[Dict] = []
    arc_mask = np.zeros_like(original, dtype=bool)
    angle_rejected = 0
    contact_rejected = 0

    for group_pixels in band_groups:
        group_arr = np.asarray(group_pixels, dtype=np.int64)
        if len(group_arr) < proposal_config.min_area:
            continue

        rel = group_arr.astype(np.float32) - center
        theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
        group_bin_ids = np.unique(np.floor(theta / (360.0 / angular_bins)).astype(np.int64) % angular_bins)
        angular_degrees = float(len(group_bin_ids) / angular_bins * 360.0)
        if angular_degrees < min_angular_degrees:
            angle_rejected += 1
            continue

        raw_keep = original[group_arr[:, 0], group_arr[:, 1]]
        raw_group_points = group_arr[raw_keep]
        if len(raw_group_points) < proposal_config.min_area:
            continue

        group_labels = parent_labels[raw_group_points[:, 0], raw_group_points[:, 1]]
        valid_labels = group_labels[group_labels >= 0]
        parent_label = int(np.bincount(valid_labels).argmax()) if len(valid_labels) else -1
        contact_ratio = 0.0
        arc_length = 0.0
        contact_length = 0.0
        if parent_label >= 0 and parent_label in parent_components:
            arc_rel = raw_group_points.astype(np.float32) - center
            arc_radii = np.sqrt(arc_rel[:, 0] ** 2 + arc_rel[:, 1] ** 2)
            arc_length = float(len(group_bin_ids) * (2.0 * np.pi / angular_bins) * np.mean(arc_radii))

            arc_point_set = {(int(r), int(c)) for r, c in raw_group_points}
            parent_non_arc_set = {
                (int(r), int(c))
                for r, c in parent_components[parent_label]
                if (int(r), int(c)) not in arc_point_set
            }
            if parent_non_arc_set:
                for r, c in raw_group_points:
                    ri, ci = int(r), int(c)
                    if any(
                        (ri + dr, ci + dc) in parent_non_arc_set
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if dr != 0 or dc != 0
                    ):
                        contact_length += 1.0
            if arc_length > 0.0:
                contact_ratio = float(contact_length / arc_length)
            if contact_ratio > max_contact_ratio:
                contact_rejected += 1
                continue

        angular_coverage = float(len(group_bin_ids) / angular_bins)
        geometry_type = "edge_ring" if angular_coverage >= COMPACT_ARC_MIN_FULL_COVERAGE else "ring_arc"
        token = _token_stats(raw_group_points, weight_map, valid_mask, total_mass=total_mass, source=source)
        token.update(
            proposal_source="arc_ring_residual",
            proposal_type="ring_band" if geometry_type == "edge_ring" else "ring_arc_band",
            geometry_type=geometry_type,
            raw_point_count=int(len(raw_group_points)),
            ring_arc_angular_coverage=angular_coverage,
            ring_arc_angle_degrees=angular_degrees,
            ring_arc_angular_bins=angular_bins,
            ring_arc_occupied_bins=int(len(group_bin_ids)),
            ring_arc_mean_radius=float(np.mean(np.sqrt(np.sum((raw_group_points.astype(np.float32) - center) ** 2, axis=1)))),
            ring_arc_length=float(arc_length),
            ring_arc_contact_length=float(contact_length),
            ring_arc_contact_ratio=float(contact_ratio),
        )
        arc_tokens.append(token)
        arc_mask[raw_group_points[:, 0], raw_group_points[:, 1]] = True

    component_tokens = _retrieval_component_tokens(
        original & (~arc_mask),
        weight_map,
        valid_mask,
        min_area=proposal_config.min_area,
        source=source,
    )
    for token in component_tokens:
        token["proposal_source"] = "arc_ring_residual"

    selected = _select_arc_band_residual_tokens(arc_tokens, component_tokens, proposal_config.top_k)
    ring_debug.update(
        band_group_input_count=int(len(band_groups)),
        arc_token_count=int(len(arc_tokens)),
        min_angular_degrees=float(min_angular_degrees),
        max_contact_ratio=float(max_contact_ratio),
        angle_rejected_count=int(angle_rejected),
        contact_rejected_count=int(contact_rejected),
    )
    return selected, ring_debug


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


def _filter_ring_band_components(
    band_mask: np.ndarray,
    min_component_area: int,
    connectivity: int,
    parent_mask: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    radius_ref: Optional[float] = None,
    band_center: Optional[float] = None,
    band_width: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    filtered = np.zeros_like(band_mask, dtype=bool)
    components = _connected_components(band_mask, connectivity=connectivity)
    parent_labels: Optional[np.ndarray] = None
    parent_areas: List[int] = []
    parent_radial_max_delta: List[float] = []
    enforce_parent_band = (
        parent_mask is not None
        and center is not None
        and radius_ref is not None
        and band_center is not None
        and band_width is not None
        and float(radius_ref) > 1e-6
    )
    if enforce_parent_band:
        parent_labels, parent_areas = _component_label_map(parent_mask, connectivity=connectivity)
        parent_radial_max_delta = [0.0 for _ in parent_areas]
        for label, area in enumerate(parent_areas):
            if area <= 0:
                continue
            parent_points = np.argwhere(parent_labels == label).astype(np.float32)
            radial = np.linalg.norm(parent_points - center, axis=1) / float(radius_ref)
            parent_radial_max_delta[label] = float(np.max(np.abs(radial - float(band_center)))) if len(radial) else 0.0

    kept_count = 0
    kept_area = 0
    rejected_parent_band_count = 0
    min_component_area = max(int(min_component_area), 1)
    for comp in components:
        if len(comp) < min_component_area:
            continue
        if enforce_parent_band and parent_labels is not None:
            labels = parent_labels[comp[:, 0].astype(int), comp[:, 1].astype(int)]
            parent_ids = sorted({int(label) for label in labels.tolist() if int(label) >= 0})
            radial_cell = 1.0 / max(float(radius_ref), 1e-6)
            max_allowed_delta = float(band_width) + radial_cell
            parent_area = int(sum(parent_areas[label] for label in parent_ids))
            parent_fraction = float(len(comp) / max(parent_area, 1))
            parent_exceeds_band = any(parent_radial_max_delta[label] > max_allowed_delta for label in parent_ids)
            if parent_exceeds_band and parent_fraction < COMPACT_ARC_MIN_PARENT_BAND_FRACTION:
                rejected_parent_band_count += 1
                continue
        rows = comp[:, 0].astype(int)
        cols = comp[:, 1].astype(int)
        filtered[rows, cols] = True
        kept_count += 1
        kept_area += int(len(comp))
    return filtered, {
        "cc_aware_band": True,
        "band_area_before_cc": int(band_mask.sum()),
        "band_area_after_cc": int(kept_area),
        "band_cc_count": int(len(components)),
        "band_cc_kept_count": int(kept_count),
        "band_min_cc_area": int(min_component_area),
        "band_parent_width_filter": bool(enforce_parent_band),
        "band_parent_width_rejected_count": int(rejected_parent_band_count),
        "band_min_parent_fraction": float(COMPACT_ARC_MIN_PARENT_BAND_FRACTION),
    }


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
    connectivity: int = 8,
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
    effective_band_width = _effective_arc_band_width(band_width, radius_ref, False)
    band_keep = np.abs(edge_radial - band_center) <= effective_band_width
    ring_points = edge_points[band_keep]
    cc_filter_debug = {
        "cc_aware_band": False,
        "band_area_before_cc": int(len(ring_points)),
        "band_area_after_cc": int(len(ring_points)),
        "band_cc_count": 0,
        "band_cc_kept_count": 0,
    }
    ring_radial = edge_radial[band_keep]

    theta = (np.degrees(np.arctan2(ring_points[:, 0] - center[0], ring_points[:, 1] - center[1])) + 360.0) % 360.0
    angular_bins = max(int(angular_bins), 1)
    occupied = np.unique(np.floor(theta / (360.0 / angular_bins)).astype(int)) if len(theta) else []
    break_stats = _ring_break_stats(np.asarray(occupied, dtype=np.int64), angular_bins, band_center * radius_ref)
    angular_coverage = float(len(occupied) / angular_bins)
    bridged_angular_coverage = float(break_stats["ring_arc_bridged_angular_coverage"])
    area_ratio = float(len(ring_points) / max(len(points), 1))
    radial_std = float(ring_radial.std()) if len(ring_points) else 0.0

    debug.update(
        reason="candidate",
        candidate_area=int(len(ring_points)),
        angular_coverage=angular_coverage,
        bridged_angular_coverage=bridged_angular_coverage,
        radial_mean=float(ring_radial.mean()) if len(ring_points) else 0.0,
        radial_std=radial_std,
        radial_band_center=band_center,
        requested_radial_band_width=float(band_width),
        effective_radial_band_width=float(effective_band_width),
        arc_max_band_width_cells=float(COMPACT_ARC_MAX_BAND_WIDTH_CELLS),
        area_ratio=area_ratio,
        **cc_filter_debug,
        **break_stats,
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
        ring_contour_bridged_angular_coverage=bridged_angular_coverage,
        radial_band_center=band_center,
        **break_stats,
    )
    debug.update(raw_ring_area=int(len(raw_ring_pixels)), contour_area=int(len(contour_pixels)))
    debug["accepted"] = True
    debug["reason"] = "accepted"
    return token, raw_ring_mask, debug


def _extract_retrieval_arc_tokens(
    mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    source: str,
    min_area: int,
    edge_r_min: float,
    band_width: float,
    min_angular_coverage: float,
    max_angular_coverage: float,
    angular_bins: int,
    max_radial_std: float,
    allowed_gap_cells: float,
    max_gap_count: int,
    min_parent_fraction: float,
    cc_aware: bool = False,
    connectivity: int = 8,
    raw_mask: Optional[np.ndarray] = None,
    parent_fraction_mask: Optional[np.ndarray] = None,
    proposal_source: str = "retrieval_compact",
    max_gap_ratio: float = 0.20,
    max_merge_gap_count: Optional[int] = None,
    min_band_width_cells: float = 0.0,
    enforce_group_radial_std: bool = False,
    full_ring_coverage: float = COMPACT_ARC_MIN_FULL_COVERAGE,
    merge_dilation_radius: int = 1,
) -> Tuple[List[Dict], np.ndarray, Dict]:
    arc_mask = np.zeros_like(mask, dtype=bool)
    points = np.argwhere(mask).astype(np.int64)
    debug = {
        "accepted_count": 0,
        "reason": "no_points",
        "candidate_area": 0,
        "min_angular_coverage": float(min_angular_coverage),
        "max_angular_coverage": float(max_angular_coverage),
        "allowed_gap_cells": float(allowed_gap_cells),
        "max_gap_count": int(max_gap_count),
        "cc_aware_band": bool(cc_aware),
        "min_parent_component_fraction": float(min_parent_fraction),
        "parent_fraction_rejected_count": 0,
        "connectivity_rejected_count": 0,
        "max_gap_ratio": float(max_gap_ratio),
        "max_merge_gap_count": max_merge_gap_count,
        "min_band_width_cells": float(min_band_width_cells),
        "enforce_group_radial_std": bool(enforce_group_radial_std),
        "full_ring_coverage": float(full_ring_coverage),
        "merge_dilation_radius": int(merge_dilation_radius),
    }
    if len(points) < min_area:
        return [], arc_mask, debug

    h, w = mask.shape
    center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    valid_points = np.argwhere(valid_mask).astype(np.float32)
    radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max()) if len(valid_points) else 0.0
    if radius_ref <= 1e-6:
        debug["reason"] = "bad_radius"
        return [], arc_mask, debug

    rel = points.astype(np.float32) - center
    radial = np.linalg.norm(rel, axis=1) / radius_ref
    edge_keep = radial >= edge_r_min
    if int(edge_keep.sum()) < min_area:
        debug.update(reason="too_few_edge_points", candidate_area=int(edge_keep.sum()))
        return [], arc_mask, debug

    edge_points = points[edge_keep]
    edge_radial = radial[edge_keep]
    hist, edges = np.histogram(edge_radial, bins=12, range=(edge_r_min, 1.05))
    peak = int(hist.argmax())
    band_center = float((edges[peak] + edges[peak + 1]) / 2.0)

    effective_band_width = _effective_arc_band_width(band_width, radius_ref, True)
    if min_band_width_cells > 0.0:
        effective_band_width = max(effective_band_width, float(min_band_width_cells) / radius_ref)
    # Extract band pixels from raw_mask (actual defect pixels), not the closed mask
    band_source = raw_mask if raw_mask is not None else mask
    band_points = np.argwhere(band_source).astype(np.int64)
    band_rel = band_points.astype(np.float32) - center
    band_radial = np.linalg.norm(band_rel, axis=1) / radius_ref
    all_band_keep = (band_radial >= edge_r_min) & (np.abs(band_radial - band_center) <= effective_band_width)
    ring_band_mask = np.zeros_like(mask, dtype=bool)
    ring_band_raw = band_points[all_band_keep]
    ring_band_raw_count = int(len(ring_band_raw))
    if len(ring_band_raw):
        ring_band_mask[ring_band_raw[:, 0], ring_band_raw[:, 1]] = True
    # Dilation only establishes raw-component groups; it never adds token pixels.
    merge_dilation_radius = max(int(merge_dilation_radius), 0)
    if ring_band_mask.any():
        raw_components = _connected_components(ring_band_mask, connectivity=8)
        raw_cc_count = len(raw_components)
        n_comp = len(raw_components)
        if n_comp <= 1:
            merged_groups = [list(range(n_comp))]
        else:
            h, w = mask.shape
            dilated_sets: list[set] = []
            for comp in raw_components:
                dset = set()
                for r, c in comp:
                    r, c = int(r), int(c)
                    for dr in range(-merge_dilation_radius, merge_dilation_radius + 1):
                        for dc in range(-merge_dilation_radius, merge_dilation_radius + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                dset.add((nr, nc))
                dilated_sets.append(dset)
            parent = list(range(n_comp))
            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def _union(x: int, y: int) -> None:
                rx, ry = _find(x), _find(y)
                if rx != ry:
                    parent[rx] = ry
            for i in range(n_comp):
                for j in range(i + 1, n_comp):
                    if dilated_sets[i] & dilated_sets[j]:
                        _union(i, j)
            groups: dict[int, list[int]] = {}
            for i in range(n_comp):
                root = _find(i)
                groups.setdefault(root, []).append(i)
            merged_groups = list(groups.values())
        # Filter only the grouping relationship, not the original token pixels.
        valid_groups = []
        rejected_by_gap = 0
        for comp_indices in merged_groups:
            total_pixels = sum(len(raw_components[i]) for i in comp_indices)
            gap_count = len(comp_indices) - 1
            gap_count_ok = max_merge_gap_count is None or gap_count <= int(max_merge_gap_count)
            if total_pixels > 0 and gap_count_ok and gap_count / total_pixels <= max(float(max_gap_ratio), 0.0):
                valid_groups.append((total_pixels, comp_indices))
            else:
                rejected_by_gap += 1
        valid_groups.sort(key=lambda x: x[0], reverse=True)
        filtered = np.zeros_like(mask, dtype=bool)
        kept_count = 0
        band_groups: list = []
        for total_pixels, comp_indices in valid_groups[:5]:
            group_pixels: list = []
            for i in comp_indices:
                comp = raw_components[i]
                filtered[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True
                group_pixels.extend((int(r), int(c)) for r, c in comp)
            band_groups.append(group_pixels)
            kept_count += 1
        ring_band_mask = filtered
        merged_cc_count = len(merged_groups)
    else:
        raw_cc_count = 0
        merged_cc_count = 0
        kept_count = 0
        rejected_by_gap = 0
        band_groups = []
    cc_filter_debug = {
        "band_area_before_cc": ring_band_raw_count,
        "band_area_after_cc": int(ring_band_mask.sum()),
        "band_cc_count": raw_cc_count,
        "band_cc_merged_count": merged_cc_count,
        "band_cc_kept_count": kept_count,
        "band_cc_rejected_by_gap": rejected_by_gap,
        "band_groups_raw": band_groups,
    }
    if cc_aware:
        ring_band_mask, cc_filter_debug = _filter_ring_band_components(
            ring_band_mask,
            min_component_area=min_area,
            connectivity=connectivity,
            parent_mask=mask,
            center=center,
            radius_ref=radius_ref,
            band_center=band_center,
            band_width=effective_band_width,
        )
    candidate_mask = ring_band_mask
    candidate_points = np.argwhere(candidate_mask).astype(np.int64)
    debug.update(
        reason="candidate",
        candidate_area=int(len(candidate_points)),
        band_pixels_raw=[(int(r), int(c)) for r, c in ring_band_raw],
        band_pixels=[(int(r), int(c)) for r, c in candidate_points],
        radial_band_center=band_center,
        requested_radial_band_width=float(band_width),
        effective_radial_band_width=float(effective_band_width),
        arc_max_band_width_cells=float(COMPACT_ARC_MAX_BAND_WIDTH_CELLS),
        **cc_filter_debug,
    )
    if len(candidate_points) < min_area:
        debug["reason"] = "too_few_ring_band_points"
        return [], arc_mask, debug

    token_mask = mask if raw_mask is None else (raw_mask & valid_mask)
    total_mass = float(weight_map[token_mask].sum())
    parent_source_mask = candidate_mask if parent_fraction_mask is None else (parent_fraction_mask & valid_mask)
    parent_labels, parent_areas = _component_label_map(parent_source_mask, connectivity=connectivity)
    tokens: List[Dict] = []
    angle_rejected_count = 0
    parent_fraction_rejected_count = 0
    for group_pixels in band_groups:
        group_arr = np.array(group_pixels, dtype=np.int64)
        if len(group_arr) < min_area:
            continue
        # Compute angular coverage as fraction of angular bins occupied by this group
        rel = group_arr.astype(np.float32) - center
        theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
        angular_bins_val = max(int(angular_bins), 1)
        group_bin_ids = np.unique(np.floor(theta / (360.0 / angular_bins_val)).astype(np.int64) % angular_bins_val)
        angular_coverage = float(len(group_bin_ids) / angular_bins_val)
        angular_degrees = float(angular_coverage * 360.0)
        if angular_coverage < min_angular_coverage:
            angle_rejected_count += 1
            continue
        parent_fraction, parent_area = _arc_parent_component_fraction(
            group_arr, parent_labels, parent_areas,
        )
        if parent_fraction < min_parent_fraction:
            parent_fraction_rejected_count += 1
            continue
        # Keep only raw (original defect) pixels within the group
        raw_keep = token_mask[group_arr[:, 0], group_arr[:, 1]]
        raw_group_points = group_arr[raw_keep]
        if len(raw_group_points) < min_area:
            continue
        raw_rel = raw_group_points.astype(np.float32) - center
        raw_radial_std = float((np.linalg.norm(raw_rel, axis=1) / radius_ref).std())
        if enforce_group_radial_std and raw_radial_std > max_radial_std:
            continue
        geometry_type = "edge_ring" if angular_coverage >= full_ring_coverage else "ring_arc"
        token = _token_stats(raw_group_points, weight_map, valid_mask, total_mass=total_mass, source=source)
        token.update(
            proposal_source=proposal_source,
            proposal_type="ring_band" if geometry_type == "edge_ring" else "ring_arc_band",
            geometry_type=geometry_type,
            raw_point_count=int(len(raw_group_points)),
            ring_arc_angular_coverage=angular_coverage,
            ring_arc_angle_degrees=angular_degrees,
            ring_arc_angular_bins=angular_bins_val,
            ring_arc_occupied_bins=int(len(group_bin_ids)),
            ring_arc_parent_component_fraction=parent_fraction,
            ring_arc_parent_component_area=parent_area,
            ring_arc_min_parent_component_fraction=float(min_parent_fraction),
            radial_band_center=band_center,
            ring_arc_radial_std=raw_radial_std,
        )
        tokens.append(token)
        arc_mask[raw_group_points[:, 0], raw_group_points[:, 1]] = True

    tokens.sort(key=lambda item: (item.get("ring_arc_angular_coverage", 0.0), item.get("area", 0)), reverse=True)
    debug.update(
        accepted_count=int(len(tokens)),
        reason="accepted" if tokens else "no_valid_arc_groups",
        band_group_count=int(len(band_groups)),
        angle_rejected_count=int(angle_rejected_count),
        parent_fraction_rejected_count=int(parent_fraction_rejected_count),
        parent_fraction_source="raw_map" if parent_fraction_mask is not None else "candidate_band",
    )
    return tokens, arc_mask, debug


def _arc_connected_regions(
    raw_points: np.ndarray,
    contour_points: np.ndarray,
    shape: Tuple[int, int],
    valid_mask: np.ndarray,
    connectivity: int,
    min_area: int,
    max_bridge_pixels: int,
    parent_mask: np.ndarray,
    min_parent_fraction: float,
) -> List[Dict]:
    raw_mask = np.zeros(shape, dtype=bool)
    if len(raw_points):
        raw_mask[raw_points[:, 0].astype(int), raw_points[:, 1].astype(int)] = True
    raw_components = _connected_components(raw_mask, connectivity=connectivity)

    contour_mask = np.zeros(shape, dtype=bool)
    if len(contour_points):
        contour_mask[contour_points[:, 0].astype(int), contour_points[:, 1].astype(int)] = True
    contour_mask &= valid_mask
    regions: List[Dict] = []
    min_area = max(int(min_area), 1)
    max_bridge_pixels = max(int(max_bridge_pixels), 0)

    for idx, comp in enumerate(raw_components):
        if len(comp) >= min_area:
            region = _arc_region(
                comp,
                comp,
                mode="8_connected",
                raw_component_count=1,
                bridge_pixel_count=0,
                contour_component_count=1,
                parent_mask=parent_mask,
                min_parent_fraction=min_parent_fraction,
                max_bridge_pixels=max_bridge_pixels,
            )
            if region is not None:
                regions.append(
                    region
                )
        for jdx in range(idx + 1, len(raw_components)):
            other = raw_components[jdx]
            bridge_pixel_count = _component_min_chebyshev_distance(comp, other) - 1
            if bridge_pixel_count < 0 or bridge_pixel_count > max_bridge_pixels:
                continue
            combined_raw = np.vstack([comp, other])
            if len(combined_raw) < min_area:
                continue
            combined_contour = _contour_points_for_raw_region(combined_raw, contour_mask, raw_mask)
            region = _arc_region(
                combined_raw,
                combined_contour,
                mode="pixel_gap",
                raw_component_count=2,
                bridge_pixel_count=bridge_pixel_count,
                contour_component_count=max(1, len(_connected_components(_points_to_mask(combined_contour, shape), connectivity=connectivity))),
                parent_mask=parent_mask,
                min_parent_fraction=min_parent_fraction,
                max_bridge_pixels=max_bridge_pixels,
            )
            if region is not None:
                regions.append(
                    region
                )

    if not regions and len(raw_components) == 1 and len(raw_components[0]) >= min_area:
        comp = raw_components[0]
        region = _arc_region(
            comp,
            comp,
            mode="8_connected",
            raw_component_count=1,
            bridge_pixel_count=0,
            contour_component_count=1,
            parent_mask=parent_mask,
            min_parent_fraction=min_parent_fraction,
            max_bridge_pixels=max_bridge_pixels,
        )
        if region is not None:
            regions.append(
                region
            )
    regions.sort(
        key=lambda item: (
            item["connectivity_stats"]["arc_connectivity_mode"] == "pixel_gap",
            len(item["raw_points"]),
        ),
        reverse=True,
    )
    return _deduplicate_arc_regions(regions)


def _arc_region(
    raw_points: np.ndarray,
    contour_points: np.ndarray,
    mode: str,
    raw_component_count: int,
    bridge_pixel_count: int,
    contour_component_count: int,
    parent_mask: np.ndarray,
    min_parent_fraction: float,
    max_bridge_pixels: int,
) -> Optional[Dict]:
    parent_fraction, parent_area = _region_parent_component_fraction(raw_points, parent_mask)
    if parent_fraction < min_parent_fraction:
        return None
    return {
        "raw_points": raw_points.astype(np.int64),
        "contour_points": contour_points.astype(np.int64),
        "connectivity_stats": {
            "arc_connected": True,
            "arc_connectivity_mode": mode,
            "arc_raw_component_count": int(raw_component_count),
            "arc_bridge_pixel_count": int(bridge_pixel_count),
            "arc_contour_component_count": int(contour_component_count),
            "arc_max_allowed_bridge_pixels": int(max_bridge_pixels),
            "arc_parent_component_fraction": float(parent_fraction),
            "arc_parent_component_area": int(parent_area),
        },
    }


def _deduplicate_arc_regions(regions: List[Dict]) -> List[Dict]:
    selected: List[Dict] = []
    for region in regions:
        points = {(int(r), int(c)) for r, c in region["raw_points"]}
        if any(points.issubset({(int(r), int(c)) for r, c in item["raw_points"]}) for item in selected):
            continue
        selected.append(region)
    return selected


def _deduplicate_arc_tokens(tokens: List[Dict]) -> List[Dict]:
    selected: List[Dict] = []
    seen: set[Tuple[Tuple[int, int], ...]] = set()
    for token in sorted(tokens, key=lambda item: (item.get("area", 0), item.get("ring_contour_angular_coverage", 0.0)), reverse=True):
        key = tuple(sorted((int(r), int(c)) for r, c in token.get("pixels", [])))
        if key in seen:
            continue
        seen.add(key)
        selected.append(token)
    return selected


def _retrieval_component_tokens(
    mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    min_area: int,
    source: str,
    grouping_mask: Optional[np.ndarray] = None,
) -> List[Dict]:
    tokens = []
    total_mass = float(weight_map[mask].sum())
    component_mask = mask if grouping_mask is None else (grouping_mask & valid_mask)
    for comp in _connected_components(component_mask):
        rows = comp[:, 0].astype(int)
        cols = comp[:, 1].astype(int)
        raw_keep = mask[rows, cols]
        raw_comp = comp[raw_keep]
        if len(raw_comp) < min_area:
            continue
        token = _token_stats(raw_comp, weight_map, valid_mask, total_mass=total_mass, source=source)
        token["proposal_source"] = "retrieval_compact"
        token["proposal_type"] = "component"
        if grouping_mask is not None:
            token["component_grouping_area"] = int(len(comp))
            token["component_raw_area"] = int(len(raw_comp))
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


# ---------------------------------------------------------------------------
# Token selection helpers (called from _retrieval_compact_tokens)
# ---------------------------------------------------------------------------


def _select_retrieval_tokens(
    ring_token: Optional[Dict],
    component_tokens: List[Dict],
    top_k: int,
    min_residual_types: int = 3,
) -> List[Dict]:
    selected: List[Dict] = []
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
    residual_selected: List[Dict] = []
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


def _select_arc_band_residual_tokens(
    arc_tokens: List[Dict],
    component_tokens: List[Dict],
    top_k: int,
) -> List[Dict]:
    if top_k <= 0:
        return []
    kept_arcs = sorted(arc_tokens, key=_display_order, reverse=True)
    if len(kept_arcs) >= top_k:
        return kept_arcs
    remaining_slots = top_k - len(kept_arcs)
    components = sorted(component_tokens, key=_residual_importance, reverse=True)
    selected = kept_arcs + components[:remaining_slots]
    selected.sort(key=_display_order, reverse=True)
    return selected


def _residual_importance(item: Dict) -> float:
    area_score = np.sqrt(max(item.get("area", 0), 0))
    radial = item.get("radial_distance_norm", 0.5)
    central_bonus = 2.0 if radial <= 0.35 else 0.0
    fill = item.get("area", 0) / max(item.get("bbox_height", 1) * item.get("bbox_width", 1), 1)
    structure_bonus = {
        "edge_ring": 3.0,
        "central": 2.5,
        "ring_arc": 2.0,
        "blob": 1.5,
        "line": 1.2,
        "irregular": 0.8,
    }.get(item.get("geometry_type"), 0.8)
    return float(area_score + central_bonus + structure_bonus + min(fill, 1.0))


def _type_group_priority(geometry_type: str, representative: Dict) -> float:
    base = {
        "edge_ring": 5.5,
        "central": 5.0,
        "ring_arc": 4.5,
        "blob": 4.0,
        "line": 3.5,
        "irregular": 3.0,
    }.get(geometry_type, 3.0)
    return base + 0.01 * _residual_importance(representative)


def _display_order(item: Dict) -> float:
    order = {
        "edge_ring": 5.5,
        "central": 5.0,
        "ring_arc": 4.5,
        "blob": 4.0,
        "line": 3.5,
        "irregular": 3.0,
    }.get(item.get("geometry_type"), 3.0)
    return order + 0.01 * item.get("area", 0)
