from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..core.pipeline import map_klarf_to_grid
from ..data.fileio import load_defect_tables, save_grid_maps
from ..core.similarity import compute_similarity
from ..core.local_matching import explain_count_partial_match
from ..core.classnumber_matching import classnumber_scores_dict, compute_classnumber_matches
from .cli_args import SIMILARITY_COLUMNS, PARTIAL_MATCH_COLUMNS, PARTIAL_MATCH_MO_COLUMNS, CLASSNUMBER_COLUMNS, derive_classnumber_match_mode


def process_one(
    klarf_path: Path,
    args,
    shape: Tuple[int, int],
    ref_gm: "GridMaps",
    npz_dir: Path | None = None,
) -> dict:
    result: dict = {}

    skip_reason = _check_defect_threshold(klarf_path, args)
    if skip_reason is not None:
        return skip_reason

    grid_maps_or_error = _map_klarf_file(klarf_path, args, shape, npz_dir)
    if "_status" in grid_maps_or_error:
        return grid_maps_or_error
    grid_maps = grid_maps_or_error["_grid_maps"]

    mode = getattr(args, "mode", "count-partial")
    result.update(_compute_similarity_scores(ref_gm, grid_maps))
    result.update(_compute_count_partial_scores(ref_gm, grid_maps, args))
    if mode == "classnumber":
        result.update(_compute_classnumber_scores(klarf_path, ref_gm, shape, args))
    result.update(_attach_metadata(grid_maps, args))
    result["_status"] = "OK"
    return result


def _check_defect_threshold(klarf_path: Path, args) -> dict | None:
    if args.defect_threshold <= 0:
        return None
    try:
        tables = load_defect_tables(klarf_path)
        idx = args.defect_table_index
        if idx >= len(tables) or len(tables[idx].rows) < args.defect_threshold:
            actual = len(tables[idx].rows) if idx < len(tables) else 0
            return {
                "_status": "SKIPPED",
                "_reason": f"defect count {actual} < {args.defect_threshold}",
            }
    except Exception as e:
        return {"_status": "ERROR", "_reason": f"count check failed: {e}"}
    return None


def _map_klarf_file(klarf_path: Path, args, shape: Tuple[int, int], npz_dir: Path | None = None) -> dict:
    try:
        grid_maps = map_klarf_to_grid(
            klarf_path,
            shape=shape,
            mapper_name=args.mapper,
            representation_name=args.representation,
            defect_table_index=args.defect_table_index,
            die_x_range=tuple(args.die_x_range) if args.die_x_range else None,
            die_y_range=tuple(args.die_y_range) if args.die_y_range else None,
            die_defect_threshold=args.die_defect_threshold,
        )
    except ValueError as e:
        msg = str(e)
        if "Single-die wafer" in msg or "not supported" in msg:
            return {"_status": "SKIPPED", "_reason": msg.split("\n")[0]}
        return {"_status": "ERROR", "_reason": msg}
    except Exception as e:
        return {"_status": "ERROR", "_reason": f"{type(e).__name__}: {e}"}

    # if npz_dir:
    #     npz_dir.mkdir(parents=True, exist_ok=True)
    #     out_path = npz_dir / f"{klarf_path.stem}.npz"
    #     save_grid_maps(out_path, grid_maps)
    return {"_grid_maps": grid_maps}


def _compute_similarity_scores(ref_gm: "GridMaps", grid_maps: "GridMaps") -> dict:
    result: dict = {}
    for method in SIMILARITY_COLUMNS:
        try:
            result[method] = compute_similarity(
                ref_gm.representation_map,
                grid_maps.representation_map,
                method=method,
                reference_status=ref_gm.status_map,
                candidate_status=grid_maps.status_map,
            )
        except Exception as e:
            result[method] = f"ERR:{e}"
    return result


def _compute_count_partial_scores(ref_gm: "GridMaps", grid_maps: "GridMaps", args) -> dict:
    try:
        explanation = explain_count_partial_match(
            ref_gm,
            grid_maps,
            min_area=args.proposal_min_area,
            top_k=args.proposal_top_k,
            token_match_top_k=args.token_match_top_k,
            map_match_top_k=args.map_match_top_k,
            proposal_mode=args.proposal_mode,
            rotation_tolerance=args.proposal_rotation_tolerance,
            min_token_score=args.token_min_score,
            score_shape_weight=args.token_score_shape_weight,
            score_position_weight=args.token_score_position_weight,
            score_scale_weight=args.token_score_scale_weight,
            min_relative_token_area=args.proposal_min_relative_token_area,
            scale_area_weight=args.token_scale_area_weight,
            scale_pca_weight=args.token_scale_pca_weight,
            scale_ratio_min=args.token_scale_ratio_min,
            density_sigmas=tuple(args.density_sigmas),
            density_threshold=args.density_threshold,
            density_min_raw_points=args.density_min_raw_points,
            density_min_raw_mass=args.density_min_raw_mass,
            density_merge_iou=args.density_merge_iou,
            density_weight_transform=args.density_weight_transform,
            ring_min_area=args.ring_min_area,
            ring_edge_r_min=args.ring_edge_r_min,
            ring_band_width=args.ring_band_width,
            ring_min_angular_coverage=args.ring_min_angular_coverage,
            ring_angular_bins=args.ring_angular_bins,
            ring_max_radial_std=args.ring_max_radial_std,
            ring_max_defect_ratio=args.ring_max_defect_ratio,
            ring_min_edge_defect_fraction=args.ring_min_edge_defect_fraction,
        )
        partial = explanation["result"]
        partial_mo = explanation["result_matched_only"]
        return {
            "count-partial": partial.score,
            "count-partial-shape": partial.mean_shape,
            "count-partial-position": partial.mean_position,
            "count-partial-scale": partial.mean_scale,
            "count-partial-type": partial.mean_type,
            "count-partial-tokens": partial,
            "count-partial-mo": partial_mo.score,
            "count-partial-mo-shape": partial_mo.mean_shape,
            "count-partial-mo-position": partial_mo.mean_position,
            "count-partial-mo-scale": partial_mo.mean_scale,
            "count-partial-mo-type": partial_mo.mean_type,
            "count-partial-mo-tokens": partial_mo,
            "_token_topk_matches": explanation.get("token_topk_matches", []),
            "_map_topk_matches": explanation.get("map_topk_matches", []),
        }
    except Exception as e:
        return {col: f"ERR:{e}" for col in PARTIAL_MATCH_COLUMNS + PARTIAL_MATCH_MO_COLUMNS}


def _compute_classnumber_scores(
    klarf_path: Path,
    ref_gm: "GridMaps",
    shape: Tuple[int, int],
    args,
) -> dict:
    try:
        class_result = compute_classnumber_matches(
            klarf_path,
            reference=ref_gm,
            shape=shape,
            mapper_name=args.mapper,
            representation_name=args.representation,
            defect_table_index=args.defect_table_index,
            die_x_range=tuple(args.die_x_range) if args.die_x_range else None,
            die_y_range=tuple(args.die_y_range) if args.die_y_range else None,
            min_area=args.proposal_min_area,
            top_k=args.proposal_top_k,
            match_mode=derive_classnumber_match_mode(args.representation),
            binary_dilation=args.classnumber_binary_dilation,
            binary_beta=args.classnumber_binary_beta,
            die_defect_threshold=args.die_defect_threshold,
            proposal_mode=args.proposal_mode,
            rotation_tolerance=args.proposal_rotation_tolerance,
            min_token_score=args.token_min_score,
            score_shape_weight=args.token_score_shape_weight,
            score_position_weight=args.token_score_position_weight,
            score_scale_weight=args.token_score_scale_weight,
            min_relative_token_area=args.proposal_min_relative_token_area,
            scale_area_weight=args.token_scale_area_weight,
            scale_pca_weight=args.token_scale_pca_weight,
            scale_ratio_min=args.token_scale_ratio_min,
            density_sigmas=tuple(args.density_sigmas),
            density_threshold=args.density_threshold,
            density_min_raw_points=args.density_min_raw_points,
            density_min_raw_mass=args.density_min_raw_mass,
            density_merge_iou=args.density_merge_iou,
            density_weight_transform=args.density_weight_transform,
            ring_min_area=args.ring_min_area,
            ring_edge_r_min=args.ring_edge_r_min,
            ring_band_width=args.ring_band_width,
            ring_min_angular_coverage=args.ring_min_angular_coverage,
            ring_angular_bins=args.ring_angular_bins,
            ring_max_radial_std=args.ring_max_radial_std,
            ring_max_defect_ratio=args.ring_max_defect_ratio,
            ring_min_edge_defect_fraction=args.ring_min_edge_defect_fraction,
        )
        result = classnumber_scores_dict(class_result)
        result["_classnumber_result"] = class_result
        return result
    except Exception as e:
        return {col: f"ERR:{e}" for col in CLASSNUMBER_COLUMNS}


def _attach_metadata(grid_maps: "GridMaps", args) -> dict:
    return {
        "_mapped": grid_maps.metadata.get("mapped_defects", 0),
        "_input": grid_maps.metadata.get("input_defects", 0),
        "_die_defect_threshold": grid_maps.metadata.get("die_defect_threshold", args.die_defect_threshold),
        "_grid_maps": grid_maps,
    }
