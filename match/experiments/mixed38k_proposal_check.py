"""Mixed38K proposal sanity check.

For every Mixed38K pattern label combination, sample one wafer map, run the
current proposal generator, and render raw/proposal visualizations.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from match.core.local_matching.proposal import _finalize_token, _proposal_config, _tokens_from_mask
from match.core.local_matching.proposal_utils import (
    _component_label_map,
    _token_stats,
    _wafer_center_and_radius,
)
from match.core.local_matching.morphology import _connected_components
from match.core.local_matching.scoring import _is_fragmented_sparse_map
from match.core.models import BACKGROUND, GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


CLASS_NAMES = (
    "center",
    "donut",
    "edge-loc",
    "edge-ring",
    "loc",
    "random",
    "scratch",
    "near-full",
)
DEFAULT_DATA_FILE = WORKSPACE_ROOT / "data" / "wm38k" / "Wafer_Map_Datasets.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "match" / "experiments" / "artifacts" / "mixed38k_proposal_check_all_patterns"
TOKEN_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#ca8a04",
    "#9333ea",
    "#0891b2",
    "#ea580c",
    "#4f46e5",
    "#65a30d",
    "#be185d",
)


def main() -> None:
    args = parse_args()
    outputs = run_check(args)
    print(json.dumps({key: _output_value(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample one Mixed38K map per pattern type and visualize proposal results.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE, help="Mixed38K/WM38K npz file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--proposal-mode",
        choices=("cc", "compact", "arc", "arc-band-residual", "arc-ring-residual", "tangential-ring", "sparse-density", "auto"),
        default="arc-ring-residual",
    )
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--top-k-proposals", type=int, default=6)
    parser.add_argument("--density-sigmas", nargs="+", type=float, default=(0.8, 1.6, 3.2))
    parser.add_argument("--density-threshold", type=float, default=0.20)
    parser.add_argument("--density-min-raw-points", type=int, default=3)
    parser.add_argument("--density-min-raw-mass", type=float, default=3.0)
    parser.add_argument("--density-merge-iou", type=float, default=0.60)
    parser.add_argument("--density-weight-transform", choices=("count", "sqrt", "log1p"), default="sqrt")
    parser.add_argument("--rotation-tolerance", action="store_true")
    parser.add_argument("--overview-cols", type=int, default=8, help="Number of pattern types per overview page.")
    return parser.parse_args(argv)


def run_check(args: argparse.Namespace) -> Dict[str, object]:
    maps, labels, original_ids = _load_mixed38k(args.data_file)
    indices_by_pattern = _pattern_indices(labels)
    _validate_pattern_pools(indices_by_pattern)

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    records = []
    token_rows: List[Dict] = []
    for pattern_idx, (label_tuple, pool) in enumerate(indices_by_pattern.items()):
        pattern_name = _pattern_name(label_tuple)
        sample_pos = int(rng.choice(pool))
        grid = _grid_from_raw(
            maps[sample_pos],
            role="proposal_check",
            map_id=int(original_ids[sample_pos]),
            class_name=pattern_name,
        )
        tokens, proposal_debug, effective_mode = _extract_tokens(grid, args)
        band_pixels = []
        band_pixels_raw = []
        band_groups_raw = []
        ring_sim_debug = []
        if args.proposal_mode == "arc-ring-residual":
            ring_debug = proposal_debug.get("arc_ring_debug", {})
            band_pixels_raw = ring_debug.get("band_pixels_raw", [])
            band_pixels = ring_debug.get("band_pixels", [])
            band_groups_raw = ring_debug.get("band_groups", [])
            ring_sim_debug = ring_debug.get("group_similarities", [])
        else:
            arc_detection = proposal_debug.get("wbm", {}).get("arc_detection", {})
            if arc_detection:
                band_pixels = arc_detection.get("band_pixels", [])
                band_pixels_raw = arc_detection.get("band_pixels_raw", [])
                band_groups_raw = arc_detection.get("band_groups_raw", [])
        record = {
            "pattern_idx": int(pattern_idx),
            "pattern_name": pattern_name,
            "label_count": int(sum(label_tuple)),
            "labels": list(label_tuple),
            "source_map_id": int(original_ids[sample_pos]),
            "sample_position": sample_pos,
            "grid": grid,
            "tokens": tokens,
            "proposal_mode": effective_mode,
            "proposal_debug": proposal_debug,
            "band_pixels": band_pixels,
            "band_pixels_raw": band_pixels_raw,
            "band_groups": band_groups_raw,
            "ring_sim_debug": ring_sim_debug,
        }
        records.append(record)
        token_rows.extend(_token_rows(record))
        _save_single_figure(figure_dir / f"{pattern_idx:02d}_{_slug(pattern_name)}.png", record)

    overview_paths = _save_overview_figures(
        args.out_dir,
        records,
        title=f"Mixed38K proposals: {args.proposal_mode}",
        cols=max(int(args.overview_cols), 1),
    )

    summary_csv = args.out_dir / "summary.csv"
    summary_json = args.out_dir / "summary.json"
    config_path = args.out_dir / "config.json"
    _write_csv(summary_csv, token_rows, _summary_fieldnames())
    summary_json.write_text(json.dumps(_summary_payload(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path.write_text(json.dumps(_config_dict(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "overviews": args.out_dir,
        "overview_pages": overview_paths,
        "figures": figure_dir,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "config": config_path,
    }


def _extract_tokens(grid: GridMaps, args: argparse.Namespace) -> tuple[List[Dict], Dict, str]:
    valid_mask = (grid.status_map == VALID_NO_DEFECT) | (grid.status_map == VALID_HAS_DEFECT)
    defect_mask = (grid.status_map == VALID_HAS_DEFECT) & valid_mask
    requested_mode = args.proposal_mode

    if requested_mode == "arc-ring-residual":
        # Step 1: run arc mode first to get band_groups debug data
        arc_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != "proposal_mode"})
        setattr(arc_args, "proposal_mode", "arc")
        mode = _resolve_proposal_mode("arc", defect_mask, valid_mask, arc_args.density_min_raw_points)
        config = _proposal_config(
            tuple(grid.status_map.shape),
            int(valid_mask.sum()),
            min_area=arc_args.min_area,
            top_k=arc_args.top_k_proposals,
            proposal_mode=mode,
            rotation_tolerance=arc_args.rotation_tolerance,
            density_sigmas=tuple(arc_args.density_sigmas),
            density_threshold=arc_args.density_threshold,
            density_min_raw_points=arc_args.density_min_raw_points,
            density_min_raw_mass=arc_args.density_min_raw_mass,
            density_merge_iou=arc_args.density_merge_iou,
            density_weight_transform=arc_args.density_weight_transform,
        )
        proposal_debug = {"requested_proposal_mode": "arc-ring-residual", "proposal_mode": "arc"}
        _ = _tokens_from_mask(defect_mask, valid_mask, config, proposal_debug=proposal_debug)
        # Step 2: extract arc-ring-residual tokens from the arc debug data
        tokens, _ring_debug = _extract_arc_ring_residual_tokens(grid, defect_mask, valid_mask, args, proposal_debug)
        # Merge ring_debug into proposal_debug for row 3/4 visualization
        proposal_debug["arc_ring_debug"] = _ring_debug
        return tokens, proposal_debug, "arc-ring-residual"

    mode = _resolve_proposal_mode(requested_mode, defect_mask, valid_mask, args.density_min_raw_points)
    config = _proposal_config(
        tuple(grid.status_map.shape),
        int(valid_mask.sum()),
        min_area=args.min_area,
        top_k=args.top_k_proposals,
        proposal_mode=mode,
        rotation_tolerance=args.rotation_tolerance,
        density_sigmas=tuple(args.density_sigmas),
        density_threshold=args.density_threshold,
        density_min_raw_points=args.density_min_raw_points,
        density_min_raw_mass=args.density_min_raw_mass,
        density_merge_iou=args.density_merge_iou,
        density_weight_transform=args.density_weight_transform,
    )
    proposal_debug = {"requested_proposal_mode": requested_mode, "proposal_mode": mode}
    tokens = _tokens_from_mask(defect_mask, valid_mask, config, proposal_debug=proposal_debug)
    return tokens, proposal_debug, mode


def _resolve_proposal_mode(requested: str, defect_mask: np.ndarray, valid_mask: np.ndarray, min_raw_points: int) -> str:
    if requested != "auto":
        return requested
    if _is_fragmented_sparse_map(defect_mask & valid_mask, min_raw_points):
        return "sparse-density"
    return "cc"


def _extract_arc_ring_residual_tokens(
    grid: GridMaps,
    defect_mask: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    proposal_debug: Dict,
) -> tuple[List[Dict], Dict]:
    """arc-ring-residual mode: row4 band_groups → filter each group independently → residue 8-connect → top-k.

    Each band_group from row 4 (8-connected, 1-gap bridge, 20% break ratio, top-5)
    is evaluated independently — kept or discarded, never further split.

    Filtering per group:
      1. Angular coverage >= 45° (around wafer center, 72 angular bins).
      2. Contact ratio = contact_pixel_count / arc_length <= 40%.
         - Arc length = angular_span (rad) × mean radius from wafer center.
         - Contact pixels: arc pixels with >=1 8-neighbor in parent-minus-arc.
         - If ratio > 40%, the arc is mostly attached to its parent blob → discard.

    Surviving groups become individual arc tokens.  Their raw defect pixels are
    removed from the map before ordinary 8-connected component extraction.
    Arc tokens are always retained; residue components fill remaining top-k slots.
    """
    h, w = defect_mask.shape
    weight_map = defect_mask.astype(np.float32)
    total_mass = float(weight_map[defect_mask & valid_mask].sum())
    center, radius_ref = _wafer_center_and_radius(valid_mask)

    # --- Step 1: get band_groups from arc detection debug (row 4's output) ---
    wbm_debug = proposal_debug.get("wbm", {})
    arc_detection = wbm_debug.get("arc_detection", {})
    band_groups = arc_detection.get("band_groups_raw", [])
    band_pixels_raw = arc_detection.get("band_pixels_raw", [])
    band_pixels = arc_detection.get("band_pixels", [])
    band_center = arc_detection.get("radial_band_center", None)

    ring_debug = {
        "band_pixels_raw": band_pixels_raw,
        "band_pixels": band_pixels,
        "band_groups": band_groups,
        "radial_band_center": band_center,
    }

    if not band_groups:
        component_tokens = _retrieval_component_tokens_local(
            defect_mask & valid_mask, weight_map, valid_mask, min_area=args.min_area, source="wbm",
        )
        return component_tokens[:args.top_k_proposals], ring_debug

    angular_bins = 72
    min_angular_coverage = 45.0 / 360.0  # >= 45°
    max_contact_ratio = 0.40  # discard if >40% of arc boundary contacts parent component

    # --- Step 2: each band_group filtered by angular coverage + contact ratio ---
    # Build 8-connected component label map once
    comp_labels, comp_areas = _component_label_map(defect_mask & valid_mask, connectivity=8)
    comp_kept: Dict[int, np.ndarray] = {}
    for label_val, area in enumerate(comp_areas):
        if area < args.min_area:
            continue
        comp_kept[label_val] = np.argwhere(comp_labels == label_val)

    arc_tokens: List[Dict] = []
    arc_mask = np.zeros_like(defect_mask, dtype=bool)
    contact_rejected = 0
    angle_rejected = 0

    for group_pixels in band_groups:
        group_arr = np.array(group_pixels, dtype=np.int64)
        if len(group_arr) < args.min_area:
            continue

        # Angular coverage check (per-group, >= 45°)
        rel = group_arr.astype(np.float32) - center
        theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
        group_bin_ids = np.unique(np.floor(theta / (360.0 / angular_bins)).astype(np.int64) % angular_bins)
        group_ang = len(group_bin_ids) / angular_bins * 360.0
        if group_ang < 45.0:
            angle_rejected += 1
            continue

        # Keep only raw defect pixels within this group
        token_mask = defect_mask & valid_mask
        raw_keep = token_mask[group_arr[:, 0], group_arr[:, 1]]
        raw_group_points = group_arr[raw_keep]
        if len(raw_group_points) < args.min_area:
            continue

        # Contact ratio check: arc-length (angular × radius) vs connecting pixel count
        group_labels = comp_labels[raw_group_points[:, 0], raw_group_points[:, 1]]
        valid_lbl = group_labels[group_labels >= 0]  # -1 = background, >= 0 = component labels
        parent_label = int(np.bincount(valid_lbl).argmax()) if len(valid_lbl) > 0 else -1
        contact_ratio = 0.0
        arc_length_val = 0.0
        contact_length_val = 0.0
        if parent_label >= 0 and parent_label in comp_kept:
            # Arc mean radius for arc-length calculation
            arc_rel = raw_group_points.astype(np.float32) - center
            arc_radii = np.sqrt(arc_rel[:, 0] ** 2 + arc_rel[:, 1] ** 2)
            arc_mean_radius = float(np.mean(arc_radii))

            # Arc length = angular span (rad) × mean radius
            num_arc_bins = len(group_bin_ids)
            bin_angle_rad = 2.0 * np.pi / angular_bins
            arc_length_val = num_arc_bins * bin_angle_rad * arc_mean_radius

            # Contact length = # of arc pixels adjacent (8-neighbor) to parent-minus-arc pixels
            parent_points = comp_kept[parent_label]
            arc_point_set = set((int(r), int(c)) for r, c in raw_group_points)
            # parent-minus-arc pixel set
            parent_non_arc_set = set()
            for pt in parent_points:
                pr, pc = int(pt[0]), int(pt[1])
                if (pr, pc) not in arc_point_set:
                    parent_non_arc_set.add((pr, pc))

            contact_length_val = 0
            if parent_non_arc_set:
                for r, c in raw_group_points:
                    ri, ci = int(r), int(c)
                    has_contact = False
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            if (ri + dr, ci + dc) in parent_non_arc_set:
                                has_contact = True
                                break
                        if has_contact:
                            break
                    if has_contact:
                        contact_length_val += 1

            if arc_length_val > 0:
                contact_ratio = contact_length_val / arc_length_val
            if contact_ratio > max_contact_ratio:
                contact_rejected += 1
                continue
        # (If no parent component or parent too small, keep the group — it's a standalone arc)

        group_angular_coverage = float(len(group_bin_ids) / angular_bins)
        geometry_type = "edge_ring" if group_angular_coverage >= 0.65 else "ring_arc"
        token = _token_stats(raw_group_points, weight_map, valid_mask, total_mass=total_mass, source="wbm")
        token.update(
            proposal_source="arc_ring_residual",
            proposal_type="ring_band" if geometry_type == "edge_ring" else "ring_arc_band",
            geometry_type=geometry_type,
            raw_point_count=int(len(raw_group_points)),
            ring_arc_angular_coverage=group_angular_coverage,
            ring_arc_angle_degrees=group_ang,
            ring_arc_angular_bins=angular_bins,
            ring_arc_occupied_bins=int(len(group_bin_ids)),
            ring_arc_mean_radius=float(np.mean(np.sqrt(np.sum((raw_group_points.astype(np.float32) - center) ** 2, axis=1)))),
            ring_arc_length=float(arc_length_val),
            ring_arc_contact_length=float(contact_length_val),
            ring_arc_contact_ratio=float(contact_ratio),
        )
        arc_tokens.append(token)
        arc_mask[raw_group_points[:, 0], raw_group_points[:, 1]] = True

    arc_tokens.sort(key=lambda t: (t.get("ring_arc_angular_coverage", 0.0), t.get("area", 0)), reverse=True)

    ring_debug.update(
        band_group_input_count=int(len(band_groups)),
        arc_token_count=int(len(arc_tokens)),
        min_angular_degrees=45.0,
        max_contact_ratio=float(max_contact_ratio),
        angle_rejected_count=int(angle_rejected),
        contact_rejected_count=int(contact_rejected),
    )

    # --- Step 3 & 4: residue 8-connected on raw map minus arc token pixels ---
    residue_mask = (defect_mask & valid_mask) & (~arc_mask)
    component_tokens = _retrieval_component_tokens_local(
        residue_mask, weight_map, valid_mask, min_area=args.min_area, source="wbm",
    )

    # --- Step 5: arc tokens (must keep) + residue component tokens, top-k ---
    combined: List[Dict] = list(arc_tokens)
    remaining_slots = max(args.top_k_proposals - len(combined), 0)
    components = sorted(component_tokens, key=_residual_importance_key, reverse=True)
    combined.extend(components[:remaining_slots])
    combined.sort(key=_display_order_key, reverse=True)

    config = _proposal_config(
        tuple(defect_mask.shape),
        int(valid_mask.sum()),
        min_area=args.min_area,
        top_k=args.top_k_proposals,
        proposal_mode="arc",
        rotation_tolerance=args.rotation_tolerance,
        density_sigmas=tuple(args.density_sigmas),
        density_threshold=args.density_threshold,
        density_min_raw_points=args.density_min_raw_points,
        density_min_raw_mass=args.density_min_raw_mass,
        density_merge_iou=args.density_merge_iou,
        density_weight_transform=args.density_weight_transform,
    )
    for token in combined:
        _finalize_token(token, (h, w), config)

    return combined, ring_debug


def _retrieval_component_tokens_local(
    mask: np.ndarray,
    weight_map: np.ndarray,
    valid_mask: np.ndarray,
    min_area: int,
    source: str,
) -> List[Dict]:
    """8-connected component tokens from a binary mask (local copy for this module)."""
    tokens = []
    total_mass = float(weight_map[mask].sum())
    for comp in _connected_components(mask):
        rows = comp[:, 0].astype(int)
        cols = comp[:, 1].astype(int)
        raw_comp = comp
        if len(raw_comp) < min_area:
            continue
        token = _token_stats(raw_comp, weight_map, valid_mask, total_mass=total_mass, source=source)
        token["proposal_source"] = "arc_ring_residual"
        token["proposal_type"] = "component"
        token["geometry_type"] = _classify_component_local(token)
        tokens.append(token)
    tokens.sort(key=lambda item: item.get("area", 0), reverse=True)
    return tokens


def _classify_component_local(item: Dict) -> str:
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


def _display_order_key(item: Dict) -> float:
    order = {
        "edge_ring": 5.5,
        "central": 5.0,
        "ring_arc": 4.5,
        "blob": 4.0,
        "line": 3.5,
        "irregular": 3.0,
    }.get(item.get("geometry_type"), 3.0)
    return order + 0.01 * item.get("area", 0)


def _residual_importance_key(item: Dict) -> float:
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


def _load_mixed38k(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Mixed38K dataset not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        maps = _pick_array(data, ("maps", "x", "X", "images", "arr_0"))
        labels = _pick_array(data, ("labels", "y", "Y", "targets", "arr_1")).astype(np.int32)
    valid = labels.sum(axis=1) > 0
    return maps[valid], labels[valid], np.where(valid)[0].astype(np.int64)


def _pick_array(npz, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return npz[name]
    raise KeyError(f"None of {tuple(names)} found in npz file. Available keys: {npz.files}")


def _pattern_indices(labels: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    groups: dict[tuple[int, ...], np.ndarray] = {}
    label_tuples = [tuple(int(value) for value in row.tolist()) for row in labels]
    for label_tuple in sorted(set(label_tuples), key=lambda item: (sum(item), item)):
        mask = np.asarray([item == label_tuple for item in label_tuples], dtype=bool)
        groups[label_tuple] = np.flatnonzero(mask).astype(np.int64)
    return groups


def _validate_pattern_pools(indices_by_pattern: dict[tuple[int, ...], np.ndarray]) -> None:
    missing = [_pattern_name(label_tuple) for label_tuple, pool in indices_by_pattern.items() if len(pool) == 0]
    if missing:
        raise ValueError(f"No samples for pattern types: {', '.join(missing)}")


def _pattern_name(label_tuple: tuple[int, ...]) -> str:
    names = [CLASS_NAMES[idx] for idx, value in enumerate(label_tuple) if value]
    return "+".join(names) if names else "none"


def _slug(value: str) -> str:
    return value.replace("+", "__").replace("/", "_").replace(" ", "_")


def _normalize_map(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.uint8)
    return np.where(raw >= 3, VALID_HAS_DEFECT, raw).astype(np.uint8)


def _grid_from_raw(raw: np.ndarray, role: str, map_id: int, class_name: str) -> GridMaps:
    raw = _normalize_map(raw)
    defects = raw == VALID_HAS_DEFECT
    valid = raw > 0
    status = np.full(raw.shape, BACKGROUND, dtype=np.uint8)
    status[valid] = VALID_NO_DEFECT
    status[defects & valid] = VALID_HAS_DEFECT
    count = (status == VALID_HAS_DEFECT).astype(np.float32)
    density = count / max(float(count.sum()), 1.0)
    return GridMaps(
        count_map=count,
        binary_map=count.astype(np.uint8),
        density_map=density,
        status_map=status,
        representation_map=count,
        representation_maps={"count": count, "binary": count.astype(np.uint8), "density": density},
        metadata={"source": "mixed38k", "role": role, "map_id": int(map_id), "class_name": class_name},
    )


def _save_overview_figures(out_dir: Path, records: List[Dict], title: str, cols: int) -> List[Path]:
    paths = []
    total_pages = int(np.ceil(len(records) / float(cols))) if records else 0
    for page_idx in range(total_pages):
        page_records = records[page_idx * cols:(page_idx + 1) * cols]
        path = out_dir / f"mixed38k_all_pattern_proposals_page{page_idx + 1:02d}.png"
        _save_overview_figure(path, page_records, title=f"{title} page {page_idx + 1}/{total_pages}")
        paths.append(path)
    return paths


def _save_overview_figure(path: Path, records: List[Dict], title: str) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt

    cols = len(records)
    fig, axes = plt.subplots(4, cols, figsize=(2.2 * cols, 10.0))
    axes = np.asarray(axes).reshape(4, cols)
    for col, record in enumerate(records):
        label = f"{record['pattern_name']}\nmap={record['source_map_id']}"
        _draw_grid(axes[0, col], record["grid"], label)
        _draw_proposal(axes[1, col], record["grid"], record["tokens"], str(record["proposal_mode"]),
                       ring_sim_debug=record.get("ring_sim_debug"))
        _draw_band(axes[2, col], record["grid"], record.get("band_pixels_raw", []), label="radial band")
        _draw_band(axes[3, col], record["grid"], record.get("band_pixels", []),
                   band_groups=record.get("band_groups", []), label="filtered top-5")
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.92, wspace=0.08, hspace=0.12)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_single_figure(path: Path, record: Dict) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.7))
    label = f"{record['pattern_name']} map={record['source_map_id']}"
    _draw_grid(axes[0], record["grid"], label)
    _draw_proposal(axes[1], record["grid"], record["tokens"], str(record["proposal_mode"]),
                   ring_sim_debug=record.get("ring_sim_debug"))
    _draw_band(axes[2], record["grid"], record.get("band_pixels_raw", []), label="radial band")
    _draw_band(axes[3], record["grid"], record.get("band_pixels", []),
               band_groups=record.get("band_groups", []), label="filtered top-5")
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.82, wspace=0.10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_sim_info_text(ring_sim_debug: List[Dict] | None) -> str:
    """Build a compact similarity info string for the proposal title."""
    if not ring_sim_debug:
        return ""
    kept = [g for g in ring_sim_debug if g["kept"]]
    disc = [g for g in ring_sim_debug if not g["kept"]]
    parts = []
    if kept:
        sims = [f"{g['sim']:.2f}" for g in kept if g.get("sim") is not None]
        parts.append(f"+{len(kept)}(sim:{','.join(sims)})" if sims else f"+{len(kept)}")
    if disc:
        sims = [f"{g['sim']:.2f}" for g in disc if g.get("sim") is not None]
        parts.append(f"-{len(disc)}(sim:{','.join(sims)})" if sims else f"-{len(disc)}")
    return " ".join(parts)


def _draw_grid(ax, grid: GridMaps, label: str) -> None:
    image = np.zeros((*grid.status_map.shape, 3), dtype=np.float32)
    valid = grid.status_map == VALID_NO_DEFECT
    defects = grid.status_map == VALID_HAS_DEFECT
    image[valid] = (0.58, 0.58, 0.58)
    image[defects] = (0.92, 0.92, 0.92)
    ax.imshow(image, interpolation="nearest")
    ax.set_title(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_proposal(ax, grid: GridMaps, tokens: List[Dict], proposal_mode: str, ring_sim_debug: List[Dict] | None = None) -> None:
    _ensure_mpl()
    import matplotlib.patches as patches

    image = _proposal_image(grid, tokens)
    ax.imshow(image, interpolation="nearest")
    title = f"{proposal_mode} proposal n={len(tokens)}"
    # Build similarity info text from ring_sim_debug
    sim_text = _build_sim_info_text(ring_sim_debug)
    if sim_text:
        title += f"\n{sim_text}"
    ax.set_title(title, fontsize=7)
    for idx, token in enumerate(tokens):
        color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        x = int(token.get("bbox_col_min", 0)) - 0.5
        y = int(token.get("bbox_row_min", 0)) - 0.5
        width = int(token.get("bbox_width", 1))
        height = int(token.get("bbox_height", 1))
        ax.add_patch(patches.Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=0.8))
        ax.text(
            x,
            y,
            str(idx + 1),
            color="white",
            fontsize=5,
            ha="left",
            va="top",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 0.7, "alpha": 0.9},
        )
    ax.set_xticks([])
    ax.set_yticks([])


GROUP_COLORS = [
    np.array([0.94, 0.65, 0.13], dtype=np.float32),  # orange
    np.array([0.13, 0.70, 0.30], dtype=np.float32),  # green
    np.array([0.90, 0.30, 0.30], dtype=np.float32),  # red
    np.array([0.20, 0.50, 0.95], dtype=np.float32),  # blue
    np.array([0.80, 0.30, 0.85], dtype=np.float32),  # purple
]


def _draw_band(ax, grid: GridMaps, band_pixels: list[tuple[int, int]],
               label: str = "radial band",
               band_groups: list[list[tuple[int, int]]] | None = None) -> None:
    h, w = grid.status_map.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    valid = grid.status_map == VALID_NO_DEFECT
    defects = grid.status_map == VALID_HAS_DEFECT
    image[valid] = (0.58, 0.58, 0.58)
    image[defects] = (0.92, 0.92, 0.92)
    colored = 0
    if band_groups:
        for idx, group_pixels in enumerate(band_groups):
            color = GROUP_COLORS[idx % len(GROUP_COLORS)]
            for row, col in group_pixels:
                row, col = int(row), int(col)
                if 0 <= row < h and 0 <= col < w and grid.status_map[row, col] == VALID_HAS_DEFECT:
                    image[row, col] = color
                    colored += 1
    else:
        band_color = np.array([0.94, 0.65, 0.13], dtype=np.float32)
        for row, col in band_pixels:
            row = int(row)
            col = int(col)
            if 0 <= row < h and 0 <= col < w and grid.status_map[row, col] == VALID_HAS_DEFECT:
                image[row, col] = band_color
                colored += 1
    ax.imshow(image, interpolation="nearest")
    suffix = f" n={colored}"
    if band_groups:
        suffix += f" groups={len(band_groups)}"
    ax.set_title(f"{label}{suffix}", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _proposal_image(grid: GridMaps, tokens: List[Dict]) -> np.ndarray:
    h, w = grid.status_map.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    valid = (grid.status_map == VALID_NO_DEFECT) | (grid.status_map == VALID_HAS_DEFECT)
    defects = grid.status_map == VALID_HAS_DEFECT
    image[valid] = (0.48, 0.48, 0.48)
    image[defects] = (0.92, 0.92, 0.92)
    for idx, token in enumerate(tokens):
        color = np.asarray(_hex_to_rgb(TOKEN_COLORS[idx % len(TOKEN_COLORS)]), dtype=np.float32)
        support_color = 0.45 * color
        for row, col in _token_visual_support_pixels(token):
            if 0 <= row < h and 0 <= col < w:
                image[row, col] = np.clip(0.55 * image[row, col] + support_color, 0.0, 1.0)
        for row, col in token.get("pixels", []):
            row = int(row)
            col = int(col)
            if 0 <= row < h and 0 <= col < w:
                image[row, col] = color
    return image


def _token_visual_support_pixels(token: Dict) -> List[tuple[int, int]]:
    if token.get("kde_support_pixels"):
        return [(int(row), int(col)) for row, col in token["kde_support_pixels"]]
    if token.get("ring_contour_pixels"):
        return [(int(row), int(col)) for row, col in token["ring_contour_pixels"]]
    return []


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def _token_rows(record: Dict) -> List[Dict]:
    tokens = record["tokens"]
    if not tokens:
        return [{
            "pattern_idx": record["pattern_idx"],
            "pattern_name": record["pattern_name"],
            "label_count": record["label_count"],
            "source_map_id": record["source_map_id"],
            "proposal_mode": record["proposal_mode"],
            "token_idx": "",
            "token_count": 0,
        }]
    rows = []
    for idx, token in enumerate(tokens, start=1):
        rows.append({
            "pattern_idx": record["pattern_idx"],
            "pattern_name": record["pattern_name"],
            "label_count": record["label_count"],
            "source_map_id": record["source_map_id"],
            "proposal_mode": record["proposal_mode"],
            "token_idx": idx,
            "token_count": len(tokens),
            "proposal_source": token.get("proposal_source", ""),
            "proposal_type": token.get("proposal_type", ""),
            "geometry_type": token.get("geometry_type", ""),
            "area": int(token.get("area", 0)),
            "mass": float(token.get("mass", 0.0)),
            "centroid_row": float(token.get("centroid_row", 0.0)),
            "centroid_col": float(token.get("centroid_col", 0.0)),
            "bbox_row_min": int(token.get("bbox_row_min", 0)),
            "bbox_col_min": int(token.get("bbox_col_min", 0)),
            "bbox_row_max": int(token.get("bbox_row_max", 0)),
            "bbox_col_max": int(token.get("bbox_col_max", 0)),
            "support_area": int(token.get("support_area", token.get("area", 0))),
            "raw_point_count": int(token.get("raw_point_count", token.get("area", 0))),
            "ring_angular_coverage": _optional_float(token.get("ring_angular_coverage")),
        })
    return rows


def _summary_payload(records: List[Dict]) -> Dict:
    samples = []
    for record in records:
        samples.append({
            "pattern_idx": record["pattern_idx"],
            "pattern_name": record["pattern_name"],
            "label_count": record["label_count"],
            "labels": record["labels"],
            "source_map_id": record["source_map_id"],
            "sample_position": record["sample_position"],
            "proposal_mode": record["proposal_mode"],
            "token_count": len(record["tokens"]),
            "proposal_debug": record["proposal_debug"],
            "ring_sim_debug": record.get("ring_sim_debug", []),
            "tokens": [_compact_token(token, idx) for idx, token in enumerate(record["tokens"], start=1)],
        })
    return {"dataset": "Mixed38K", "samples": samples}


def _compact_token(token: Dict, idx: int) -> Dict:
    keys = (
        "proposal_source",
        "proposal_type",
        "geometry_type",
        "area",
        "mass",
        "centroid_row",
        "centroid_col",
        "bbox_row_min",
        "bbox_col_min",
        "bbox_row_max",
        "bbox_col_max",
        "raw_point_count",
        "ring_angular_coverage",
        "ring_arc_angular_coverage",
        "ring_arc_angle_degrees",
        "ring_arc_occupied_bins",
        "ring_arc_mean_radius",
        "ring_arc_length",
        "ring_arc_contact_length",
        "ring_arc_contact_ratio",
    )
    compact = {"token_idx": idx}
    for key in keys:
        if key in token:
            compact[key] = _json_value(token[key])
    return compact


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _optional_float(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _summary_fieldnames() -> List[str]:
    return [
        "pattern_idx",
        "pattern_name",
        "label_count",
        "source_map_id",
        "proposal_mode",
        "token_idx",
        "token_count",
        "proposal_source",
        "proposal_type",
        "geometry_type",
        "area",
        "mass",
        "centroid_row",
        "centroid_col",
        "bbox_row_min",
        "bbox_col_min",
        "bbox_row_max",
        "bbox_col_max",
        "support_area",
        "raw_point_count",
        "ring_angular_coverage",
    ]


def _config_dict(args: argparse.Namespace) -> Dict:
    return {
        "data_file": str(args.data_file),
        "out_dir": str(args.out_dir),
        "seed": int(args.seed),
        "proposal_mode": args.proposal_mode,
        "effective_proposal_mode": _resolve_config_mode(args.proposal_mode),
        "min_area": int(args.min_area),
        "top_k_proposals": int(args.top_k_proposals),
        "density_sigmas": [float(value) for value in args.density_sigmas],
        "density_threshold": float(args.density_threshold),
        "density_min_raw_points": int(args.density_min_raw_points),
        "density_min_raw_mass": float(args.density_min_raw_mass),
        "density_merge_iou": float(args.density_merge_iou),
        "density_weight_transform": args.density_weight_transform,
        "rotation_tolerance": bool(args.rotation_tolerance),
        "overview_cols": int(args.overview_cols),
    }


def _resolve_config_mode(requested: str) -> str:
    return requested


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _output_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [str(item) if isinstance(item, Path) else item for item in value]
    return value


def _ensure_mpl() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")


if __name__ == "__main__":
    main()
