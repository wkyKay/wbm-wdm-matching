from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from .descriptors import _classify_token, _shape_descriptor
from .models import ProposalConfig
from .morphology import _binary_closing_square_constrained, _connected_components, _perimeter

SPARSE_RING_EDGE_R_MIN = 0.60
SPARSE_RING_MAX_RADIAL_STD = 0.16
SPARSE_RING_RADIUS_TOL_CELLS = 2.0
SPARSE_RING_MIN_ANGULAR_BINS = 12
SPARSE_RING_MAX_ANGULAR_BINS = 144
SPARSE_RING_MIN_ARC_BINS = 2
SPARSE_RING_MIN_ARC_COVERAGE = 0.25
SPARSE_RING_MIN_FULL_COVERAGE = 0.65
COMPACT_RING_ARC_MIN_ANGULAR_COVERAGE = 30.0 / 360.0
COMPACT_RING_ARC_MAX_ANGULAR_COVERAGE = 180.0 / 360.0
COMPACT_RING_ARC_ALLOWED_GAP_CELLS = 2.0
COMPACT_RING_ARC_MAX_GAP_COUNT = 4
COMPACT_RING_ARC_MIN_PARENT_FRACTION = 0.50
COMPACT_ARC_MAX_ANGULAR_COVERAGE = 1.0
COMPACT_ARC_ALLOWED_GAP_CELLS = 2.0
COMPACT_ARC_MAX_GAP_COUNT = 1
COMPACT_ARC_MAX_BAND_WIDTH_CELLS = 2.5
COMPACT_ARC_MIN_PARENT_BAND_FRACTION = 0.45
COMPACT_ARC_MIN_FULL_COVERAGE = 0.65


def _wafer_center_and_radius(valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = valid_mask.shape
    center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    valid_points = np.argwhere(valid_mask).astype(np.float32)
    radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max()) if len(valid_points) else float(np.linalg.norm(center))
    return center, radius_ref


def _wafer_center_and_axes(valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ellipse coordinates from the valid-region bounding box."""
    points = np.argwhere(valid_mask)
    if not len(points):
        h, w = valid_mask.shape
        return np.array([h / 2.0, w / 2.0], dtype=np.float32), np.array(
            [max(h / 2.0, 0.5), max(w / 2.0, 0.5)], dtype=np.float32
        )
    mins = points.min(axis=0).astype(np.float32)
    maxs = points.max(axis=0).astype(np.float32)
    return (mins + maxs) / 2.0, np.maximum((maxs - mins) / 2.0, 0.5)


def _elliptical_radial(points: np.ndarray, center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    rel = points.astype(np.float32) - center
    return np.sqrt((rel[:, 0] / axes[0]) ** 2 + (rel[:, 1] / axes[1]) ** 2)


def _elliptical_theta(points: np.ndarray, center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    rel = points.astype(np.float32) - center
    return (np.degrees(np.arctan2(rel[:, 0] / axes[0], rel[:, 1] / axes[1])) + 360.0) % 360.0


def _circular_true_runs(values: np.ndarray) -> list[int]:
    if len(values) == 0 or not values.any():
        return []
    if values.all():
        return [int(len(values))]
    false_positions = np.flatnonzero(~values)
    start = int((false_positions[0] + 1) % len(values))
    ordered = np.concatenate([values[start:], values[:start]])
    runs: list[int] = []
    run = 0
    for value in ordered:
        if value:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def _circular_true_run_indices(values: np.ndarray) -> List[np.ndarray]:
    values = np.asarray(values, dtype=bool)
    count = len(values)
    if count == 0 or not values.any():
        return []
    if values.all():
        return [np.arange(count, dtype=np.int64)]
    false_positions = np.flatnonzero(~values)
    start = int((false_positions[0] + 1) % count)
    ordered = np.concatenate([values[start:], values[:start]])
    runs: List[np.ndarray] = []
    run_start: Optional[int] = None
    for offset, value in enumerate(ordered):
        if value and run_start is None:
            run_start = offset
        elif not value and run_start is not None:
            indices = (start + np.arange(run_start, offset)) % count
            runs.append(indices.astype(np.int64))
            run_start = None
    if run_start is not None:
        indices = (start + np.arange(run_start, count)) % count
        runs.append(indices.astype(np.int64))
    return runs


def _circular_bins_between(start_bin: int, end_bin: int, bin_count: int) -> np.ndarray:
    if bin_count <= 0:
        return np.asarray([], dtype=np.int64)
    start_bin %= bin_count
    end_bin %= bin_count
    if start_bin <= end_bin:
        return np.arange(start_bin, end_bin + 1, dtype=np.int64)
    return np.concatenate(
        [
            np.arange(start_bin, bin_count, dtype=np.int64),
            np.arange(0, end_bin + 1, dtype=np.int64),
        ]
    )


def _circular_arc_runs_with_gap_limits(
    occupied: np.ndarray,
    max_gap_bins: int,
    max_gap_count: int,
    min_angular_coverage: float,
    max_angular_coverage: float,
) -> List[Dict]:
    occupied = np.asarray(occupied, dtype=bool)
    bin_count = len(occupied)
    occupied_bins = np.flatnonzero(occupied).astype(np.int64)
    if bin_count == 0 or not len(occupied_bins):
        return []

    max_gap_bins = max(int(max_gap_bins), 0)
    max_gap_count = max(int(max_gap_count), 0)
    if len(occupied_bins) == 1:
        bins = occupied_bins.astype(np.int64)
        coverage = float(len(bins) / bin_count)
        if min_angular_coverage <= coverage <= max_angular_coverage:
            return [{"bins": bins, "gap_count": 0, "max_gap_bins": 0}]
        return []

    gaps = np.asarray(
        [
            int((occupied_bins[(idx + 1) % len(occupied_bins)] - occupied_bins[idx] - 1) % bin_count)
            for idx in range(len(occupied_bins))
        ],
        dtype=np.int64,
    )
    break_indices = np.flatnonzero(gaps > max_gap_bins).astype(np.int64)
    if not len(break_indices):
        run_bins = np.arange(bin_count, dtype=np.int64)
        gap_count = int((gaps > 0).sum())
        if gap_count <= max_gap_count and min_angular_coverage <= 1.0 <= max_angular_coverage:
            return [{"bins": run_bins, "gap_count": gap_count, "max_gap_bins": int(gaps.max(initial=0))}]
        return []

    runs: List[Dict] = []
    seen: set[Tuple[int, ...]] = set()
    for break_idx in break_indices:
        start_pos = int((break_idx + 1) % len(occupied_bins))
        end_pos = int(break_idx)
        group_positions = []
        pos = start_pos
        while True:
            group_positions.append(pos)
            if pos == end_pos:
                break
            pos = int((pos + 1) % len(occupied_bins))

        group_occupied = occupied_bins[np.asarray(group_positions, dtype=np.int64)]
        start_bin = int(group_occupied[0])
        end_bin = int(group_occupied[-1])
        run_bins = _circular_bins_between(start_bin, end_bin, bin_count)
        gap_values = []
        for pos in group_positions[:-1]:
            gap = int(gaps[pos])
            if gap > 0:
                gap_values.append(gap)
        gap_count = int(len(gap_values))
        coverage = float(len(run_bins) / bin_count)
        key = tuple(int(bin_id) for bin_id in run_bins.tolist())
        if key in seen:
            continue
        seen.add(key)
        if gap_count > max_gap_count:
            continue
        if coverage < min_angular_coverage or coverage > max_angular_coverage:
            continue
        runs.append(
            {
                "bins": run_bins,
                "gap_count": gap_count,
                "max_gap_bins": int(max(gap_values, default=0)),
            }
        )
    return runs


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


def _ring_break_stats(occupied_bins: np.ndarray, bin_count: int, ring_radius_cells: float) -> Dict:
    occupied = np.zeros(max(int(bin_count), 1), dtype=bool)
    if len(occupied_bins):
        occupied[np.asarray(occupied_bins, dtype=np.int64) % len(occupied)] = True
    arc_cells_per_bin = 2.0 * np.pi * max(float(ring_radius_cells), 1e-6) / max(len(occupied), 1)
    max_gap_bins = max(1, int(np.floor(COMPACT_RING_ARC_ALLOWED_GAP_CELLS / max(arc_cells_per_bin, 1e-6))))
    bridged = _bridge_short_circular_gaps(occupied, max_gap_bins)
    gap_runs = _circular_true_runs(~occupied)
    gap_cells = [float(run * arc_cells_per_bin) for run in gap_runs if run < len(occupied)]
    large_gap_cells = [gap for gap in gap_cells if gap > COMPACT_RING_ARC_ALLOWED_GAP_CELLS]
    return {
        "ring_arc_allowed_gap_cells": float(COMPACT_RING_ARC_ALLOWED_GAP_CELLS),
        "ring_arc_allowed_gap_bins": int(max_gap_bins),
        "ring_arc_gap_count": int(len(gap_cells)),
        "ring_arc_large_gap_count": int(len(large_gap_cells)),
        "ring_arc_max_gap_cells": float(max(gap_cells, default=0.0)),
        "ring_arc_large_gap_cells": large_gap_cells,
        "ring_arc_bridged_angular_coverage": float(bridged.mean()),
    }


def _effective_arc_band_width(band_width: float, radius_ref: float, enabled: bool) -> float:
    if not enabled:
        return float(band_width)
    radial_cell = 1.0 / max(float(radius_ref), 1e-6)
    return min(float(band_width), COMPACT_ARC_MAX_BAND_WIDTH_CELLS * radial_cell)


def _small_map_ring_input(mask: np.ndarray, valid_mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if min(int(shape[0]), int(shape[1])) > 12:
        return mask
    return _binary_closing_square_constrained(mask, valid_mask)


def _component_label_map(mask: np.ndarray, connectivity: int) -> Tuple[np.ndarray, List[int]]:
    labels = np.full(mask.shape, -1, dtype=np.int32)
    areas: List[int] = []
    for label, comp in enumerate(_connected_components(mask, connectivity=connectivity)):
        rows = comp[:, 0].astype(int)
        cols = comp[:, 1].astype(int)
        labels[rows, cols] = int(label)
        areas.append(int(len(comp)))
    return labels, areas


def _arc_parent_component_fraction(arc_points: np.ndarray, labels: np.ndarray, areas: List[int]) -> Tuple[float, int]:
    if not len(arc_points):
        return 0.0, 0
    arc_labels = labels[arc_points[:, 0].astype(int), arc_points[:, 1].astype(int)]
    parent_labels = sorted({int(label) for label in arc_labels.tolist() if int(label) >= 0})
    parent_area = int(sum(areas[label] for label in parent_labels))
    if parent_area <= 0:
        return 0.0, 0
    return float(len(arc_points) / parent_area), parent_area


def _points_to_mask(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(points):
        mask[points[:, 0].astype(int), points[:, 1].astype(int)] = True
    return mask


def _contour_points_for_raw_region(raw_points: np.ndarray, contour_mask: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    region_mask = _points_to_mask(raw_points, raw_mask.shape)
    bridge_mask = contour_mask & (~raw_mask) & _binary_closing_square_constrained(region_mask, contour_mask)
    combined = region_mask | bridge_mask
    return np.argwhere(combined).astype(np.int64)


def _chain_pixels(tokens: List[Dict], key: str) -> list[tuple[int, int]]:
    pixels: list[tuple[int, int]] = []
    for token in tokens:
        pixels.extend((int(row), int(col)) for row, col in token.get(key, []))
    return pixels


def _unique_pixel_array(pixels) -> np.ndarray:
    if not pixels:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(set((int(row), int(col)) for row, col in pixels)), dtype=np.int64)


def _pixel_iou(a: set, b: set) -> float:
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def _is_sparse_density_mode(proposal_mode: str) -> bool:
    return proposal_mode in {"sparse-density", "sparse-density-arc-ring-residual"}


def _component_min_chebyshev_distance(first: np.ndarray, second: np.ndarray) -> int:
    if not len(first) or not len(second):
        return 10**9
    deltas = np.abs(first[:, None, :].astype(np.int64) - second[None, :, :].astype(np.int64))
    return int(deltas.max(axis=2).min())


def _region_parent_component_fraction(points: np.ndarray, parent_mask: np.ndarray) -> Tuple[float, int]:
    if not len(points):
        return 0.0, 0
    labels, areas = _component_label_map(parent_mask, connectivity=8)
    point_labels = labels[points[:, 0].astype(int), points[:, 1].astype(int)]
    parent_ids = sorted({int(label) for label in point_labels.tolist() if int(label) >= 0})
    parent_area = int(sum(areas[label] for label in parent_ids))
    if parent_area <= 0:
        return 0.0, 0
    return float(len(points) / parent_area), parent_area


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
    occupied_bins = np.zeros(72, dtype=bool)
    if len(occupied):
        occupied_bins[np.asarray(occupied, dtype=np.int64) % 72] = True
    max_angular_run_coverage = float(max(_circular_true_runs(occupied_bins), default=0) / 72.0)
    max_gap_coverage = float(max(_circular_true_runs(~occupied_bins), default=0) / 72.0)
    if len(theta):
        sector_counts, _ = np.histogram(theta, bins=12, range=(0.0, 360.0))
        sector_mean = float(sector_counts.mean())
        angular_count_cv = float(sector_counts.std() / sector_mean) if sector_mean > 1e-8 else 0.0
    else:
        angular_count_cv = 0.0
    radial_std = float(radial.std()) if len(radial) else 0.0
    radial_band_width = float(np.percentile(radial, 90) - np.percentile(radial, 10)) if len(radial) else 0.0

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
        "max_angular_run_coverage": max_angular_run_coverage,
        "max_gap_coverage": max_gap_coverage,
        "angular_count_cv": angular_count_cv,
        "radial_std": radial_std,
        "radial_band_width": radial_band_width,
        "pixels": [(int(r), int(c)) for r, c in pixels],
    }
    token["geometry_type"] = _classify_token(token)
    return token


def _finalize_token(token: Dict, map_shape: tuple[int, int], proposal_config: ProposalConfig) -> None:
    token["descriptor"] = _shape_descriptor(
        token,
        map_shape,
        mode=proposal_config.descriptor_mode,
        rotation_tolerance=proposal_config.rotation_tolerance,
        moment_weight=proposal_config.moment_weight,
        geometry_weight=proposal_config.geometry_weight,
    )
    token["proposal_config"] = {
        "min_area": proposal_config.min_area,
        "top_k": proposal_config.top_k,
        "connectivity": proposal_config.connectivity,
        "descriptor_mode": proposal_config.descriptor_mode,
        "moment_weight": proposal_config.moment_weight,
        "geometry_weight": proposal_config.geometry_weight,
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
