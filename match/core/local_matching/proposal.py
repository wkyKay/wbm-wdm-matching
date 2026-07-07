from __future__ import annotations

from typing import Dict, List

import numpy as np

from .models import ProposalConfig
from .morphology import (_binary_closing_square, _connected_components, _perimeter,)
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
    tokens = []
    total_mass = float(weight_map[mask].sum())
    for comp in _connected_components(mask, connectivity=proposal_config.connectivity):
        if len(comp) < proposal_config.min_area:
            continue
        token = _token_stats(comp, weight_map, valid_mask, total_mass=total_mass, source=source)
        tokens.append(token)

    if proposal_config.proposal_mode == "compact":
        tokens = _compact_tokens(tokens, weight_map, valid_mask, proposal_config, source=source)

    tokens.sort(key=_token_importance, reverse=True)
    tokens = (
        _select_diverse_tokens(tokens, proposal_config.top_k)
        if proposal_config.proposal_mode == "compact"
        else tokens[:proposal_config.top_k]
    )
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


def _compact_tokens(
    tokens: List[Dict],
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
) -> List[Dict]:
    if not tokens:
        return tokens
    compacted = list(tokens)
    compacted = _apply_gap_aware_grouping(compacted, weight_map, valid_mask, proposal_config, source)
    compacted = _apply_ring_aware_token(compacted, weight_map, valid_mask, proposal_config, source)
    compacted = _merge_geometry_fragments(compacted, weight_map, valid_mask, proposal_config, source)
    return compacted


def _apply_gap_aware_grouping(
    tokens: List[Dict],
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
) -> List[Dict]:
    h, w = weight_map.shape
    short_side = min(h, w)
    if short_side > 25:
        return tokens

    original = (weight_map > 0) & valid_mask
    if int(original.sum()) < proposal_config.min_area:
        return tokens

    grouped_mask = _binary_closing_square(original) & valid_mask
    groups = _connected_components(grouped_mask, connectivity=proposal_config.connectivity)
    max_virtual_gap_ratio = 0.75 if short_side <= 12 else 0.45
    total_mass = float(weight_map[original].sum())
    grouped_tokens = []
    used_pixels = set()

    for group in groups:
        group_mask = np.zeros((h, w), dtype=bool)
        group_mask[group[:, 0], group[:, 1]] = True
        original_pixels = np.argwhere(original & group_mask).astype(np.int64)
        if len(original_pixels) < proposal_config.min_area:
            continue
        virtual_gap_area = int(group_mask.sum() - len(original_pixels))
        virtual_gap_ratio = virtual_gap_area / max(len(original_pixels), 1)
        if virtual_gap_ratio > max_virtual_gap_ratio:
            continue

        token = _token_stats(
            original_pixels,
            weight_map,
            valid_mask,
            total_mass=total_mass,
            source=source,
        )
        token.update(
            proposal_source="compact_gap_grouping",
            grouping_area=int(group_mask.sum()),
            virtual_gap_area=virtual_gap_area,
            virtual_gap_ratio=float(virtual_gap_ratio),
        )
        grouped_tokens.append(token)
        used_pixels.update((int(r), int(c)) for r, c in original_pixels)

    for token in tokens:
        token_pixels = set((int(r), int(c)) for r, c in token.get("pixels", []))
        if not token_pixels or token_pixels.issubset(used_pixels):
            continue
        grouped_tokens.append(token)

    return grouped_tokens if grouped_tokens else tokens


def _apply_ring_aware_token(
    tokens: List[Dict],
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
) -> List[Dict]:
    h, w = weight_map.shape
    short_side = min(h, w)
    edge_r_min = 0.50 if short_side <= 12 else 0.65
    min_edge_fraction = 0.30 if short_side <= 12 else 0.35
    radial_bins = 4 if short_side <= 12 else (8 if short_side < 26 else 12)
    angular_bins = 12 if short_side <= 12 else (36 if short_side < 26 else 72)
    band_width = 0.24 if short_side <= 12 else (0.14 if short_side < 26 else 0.10)
    max_radial_std = 0.22 if short_side <= 12 else 0.16
    min_angular_coverage = 0.14 if short_side <= 12 else 0.16
    min_ring_points = max(proposal_config.min_area, 3 if short_side <= 12 else 6)
    max_defect_ratio_for_ring = 0.45 if short_side <= 12 else 0.35

    mask = (weight_map > 0) & valid_mask
    points = np.argwhere(mask)
    if len(points) < min_ring_points:
        return tokens
    defect_ratio = float(len(points) / max(int(valid_mask.sum()), 1))
    if defect_ratio > max_defect_ratio_for_ring:
        return tokens

    center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    valid_points = np.argwhere(valid_mask).astype(np.float32)
    radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max()) if len(valid_points) else float(np.linalg.norm(center))
    if radius_ref <= 1e-6:
        return tokens

    rel = points.astype(np.float32) - center
    radial = np.linalg.norm(rel, axis=1) / radius_ref
    edge_keep = radial >= edge_r_min
    edge_fraction = float(edge_keep.sum() / max(len(points), 1))
    if edge_fraction < min_edge_fraction:
        return tokens

    edge_points = points[edge_keep]
    edge_radial = radial[edge_keep]
    hist, edges = np.histogram(edge_radial, bins=radial_bins, range=(edge_r_min, 1.05))
    if hist.max() < min_ring_points:
        return tokens

    peak = int(hist.argmax())
    band_center = float((edges[peak] + edges[peak + 1]) / 2.0)
    band_keep = np.abs(edge_radial - band_center) <= band_width
    ring_points = edge_points[band_keep]
    if len(ring_points) < min_ring_points:
        return tokens

    theta = (np.degrees(np.arctan2(ring_points[:, 0] - center[0], ring_points[:, 1] - center[1])) + 360.0) % 360.0
    occupied = np.unique(np.floor(theta / (360.0 / angular_bins)).astype(np.int64)) if len(theta) else []
    angular_coverage = float(len(occupied) / angular_bins)
    radial_std = float(radial[edge_keep][band_keep].std()) if len(ring_points) else 0.0
    if angular_coverage < min_angular_coverage or radial_std > max_radial_std:
        return tokens

    ring_set = set((int(r), int(c)) for r, c in ring_points)
    ring_token = _token_stats(
        ring_points.astype(np.int64),
        weight_map,
        valid_mask,
        total_mass=float(weight_map[mask].sum()),
        source=source,
    )
    ring_token.update(
        geometry_type="edge_ring",
        proposal_source="compact_ring_aware",
        proposal_type="ring_band",
        radial_band_center=band_center,
    )

    residual_tokens = []
    for token in tokens:
        pixels = [pixel for pixel in token.get("pixels", []) if (int(pixel[0]), int(pixel[1])) not in ring_set]
        if len(pixels) < proposal_config.min_area:
            continue
        if len(pixels) == token.get("area", 0):
            residual_tokens.append(token)
        else:
            residual_tokens.append(_token_stats(
                np.asarray(pixels, dtype=np.int64),
                weight_map,
                valid_mask,
                total_mass=float(weight_map[mask].sum()),
                source=source,
            ))
    return [ring_token] + residual_tokens


def _merge_geometry_fragments(
    tokens: List[Dict],
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
) -> List[Dict]:
    if len(tokens) <= 1:
        return tokens

    h, w = weight_map.shape
    short_side = min(h, w)
    max_gap = 2 if short_side <= 12 else (3 if short_side <= 25 else 5)
    mask = (weight_map > 0) & valid_mask
    total_mass = float(weight_map[mask].sum())
    remaining = [dict(token) for token in tokens]
    changed = True
    while changed:
        changed = False
        best_pair = None
        best_gap = float("inf")
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                if not _merge_compatible(remaining[i], remaining[j]):
                    continue
                gap = _bbox_gap(remaining[i], remaining[j])
                if gap <= max_gap and gap < best_gap:
                    best_pair = (i, j)
                    best_gap = gap
        if best_pair is None:
            break
        i, j = best_pair
        pixels = _unique_pixels(remaining[i].get("pixels", []) + remaining[j].get("pixels", []))
        merged = _token_stats(
            np.asarray(pixels, dtype=np.int64),
            weight_map,
            valid_mask,
            total_mass=total_mass,
            source=source,
        )
        merged["proposal_source"] = "compact_geometry_merge"
        merged["merged_count"] = int(remaining[i].get("merged_count", 1)) + int(remaining[j].get("merged_count", 1))
        for idx in sorted((i, j), reverse=True):
            remaining.pop(idx)
        remaining.append(merged)
        changed = True
    return remaining


def _merge_compatible(a: Dict, b: Dict) -> bool:
    ta = a.get("geometry_type", "irregular")
    tb = b.get("geometry_type", "irregular")
    if ta == "edge_ring" and tb == "edge_ring":
        return True
    if ta == "line" and tb == "line":
        return _angle_delta(float(a.get("orientation", 0.0)), float(b.get("orientation", 0.0))) <= 25.0
    if ta in {"blob", "central", "irregular"} and tb in {"blob", "central", "irregular"}:
        return True
    return False


def _bbox_gap(a: Dict, b: Dict) -> float:
    row_gap = max(0, max(a["bbox_row_min"], b["bbox_row_min"]) - min(a["bbox_row_max"], b["bbox_row_max"]) - 1)
    col_gap = max(0, max(a["bbox_col_min"], b["bbox_col_min"]) - min(a["bbox_col_max"], b["bbox_col_max"]) - 1)
    return float(np.hypot(row_gap, col_gap))


def _angle_delta(a: float, b: float) -> float:
    delta = abs((a - b + 90.0) % 180.0 - 90.0)
    return float(delta)


def _unique_pixels(pixels: List[tuple]) -> List[tuple[int, int]]:
    return sorted({(int(pixel[0]), int(pixel[1])) for pixel in pixels})


def _select_diverse_tokens(tokens: List[Dict], top_k: int) -> List[Dict]:
    if len(tokens) <= top_k:
        return tokens[:top_k]
    by_type: Dict[str, List[Dict]] = {}
    for token in tokens:
        by_type.setdefault(token.get("geometry_type", "irregular"), []).append(token)
    for items in by_type.values():
        items.sort(key=_token_importance, reverse=True)

    type_order = sorted(
        by_type,
        key=lambda key: _type_priority(key) + 0.01 * _token_importance(by_type[key][0]),
        reverse=True,
    )
    selected = []
    used = set()
    for geometry_type in type_order:
        if len(selected) >= top_k:
            break
        token = by_type[geometry_type][0]
        selected.append(token)
        used.add(id(token))

    leftovers = [token for token in tokens if id(token) not in used]
    leftovers.sort(key=_token_importance, reverse=True)
    for token in leftovers:
        if len(selected) >= top_k:
            break
        selected.append(token)
    selected.sort(key=_token_importance, reverse=True)
    return selected[:top_k]


def _type_priority(geometry_type: str) -> float:
    return {
        "edge_ring": 5.0,
        "central": 4.0,
        "blob": 3.5,
        "line": 3.0,
        "irregular": 2.5,
    }.get(geometry_type, 2.5)


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
