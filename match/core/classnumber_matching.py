# Optional classnumber-based WDM split matching.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ..data.fileio import load_defect_tables, split_defect_table_by_classnumber
from .local_matching import LocalMatchResult, compute_binary_partial_match, explain_binary_partial_match, explain_count_partial_match
from .models import GridMaps
from .pipeline import build_mapper, map_defect_table_to_grid


@dataclass(frozen=True)
class ClassSplitMatch:
    classnumber: int
    grid_maps: GridMaps
    partial: LocalMatchResult | None
    partial_matched_only: LocalMatchResult | None
    binary: LocalMatchResult | None
    binary_matched_only: LocalMatchResult | None
    rank_score: float
    rank_mode: str


@dataclass(frozen=True)
class ClassNumberMatchResult:
    splits: List[ClassSplitMatch]
    best: ClassSplitMatch | None
    match_mode: str


def compute_classnumber_matches(
    klarf_path: str | Path,
    reference: GridMaps,
    shape: Tuple[int, int],
    mapper_name: str,
    representation_name: str,
    defect_table_index: int = 0,
    die_x_range: Tuple[int, int] | None = None,
    die_y_range: Tuple[int, int] | None = None,
    min_area: int = 5,
    top_k: int = 6,
    match_mode: str = "count",
    binary_dilation: int = 1,
    binary_beta: float = 0.5,
    die_defect_threshold: int = 1,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    min_token_score: float = 0.45,
    score_shape_weight: float = 0.60,
    score_position_weight: float = 0.25,
    score_scale_weight: float = 0.15,
    min_relative_token_area: float = 0.10,
    scale_area_weight: float = 0.50,
    scale_pca_weight: float = 0.50,
    density_sigmas: tuple[float, ...] = (0.8, 1.6, 3.2),
    density_threshold: float = 0.20,
    density_min_raw_points: int = 3,
    density_min_raw_mass: float = 3.0,
    density_merge_iou: float = 0.60,
    density_weight_transform: str = "sqrt",
    moment_weight: float = 0.75,
    geometry_weight: float = 0.25,
    zernike_degree: int = 8,
    ring_min_area: int | None = None,
    ring_edge_r_min: float | None = None,
    ring_band_width: float | None = None,
    ring_min_angular_coverage: float | None = None,
    ring_angular_bins: int | None = None,
    ring_max_radial_std: float | None = None,
    ring_max_defect_ratio: float | None = None,
    ring_min_edge_defect_fraction: float | None = None,
) -> ClassNumberMatchResult:
    """Split one KLARF DefectTable by classnumber and score each split."""
    match_mode = _normalize_match_mode(match_mode)
    tables = load_defect_tables(klarf_path)
    if defect_table_index < 0 or defect_table_index >= len(tables):
        raise IndexError(
            f"defect_table_index={defect_table_index} out of range; found {len(tables)} DefectList tables"
        )

    split_tables = split_defect_table_by_classnumber(tables[defect_table_index])
    if not split_tables:
        return ClassNumberMatchResult(splits=[], best=None, match_mode=match_mode)

    mapper = build_mapper(
        klarf_path=klarf_path,
        mapper_name=mapper_name,
        die_x_range=die_x_range,
        die_y_range=die_y_range,
    )

    splits: List[ClassSplitMatch] = []
    for classnumber, defect_table in split_tables.items():
        if classnumber == 0:
            continue
        gm = map_defect_table_to_grid(
            defect_table,
            shape=shape,
            mapper=mapper,
            representation_name=representation_name,
            metadata={
                "klarf_path": str(klarf_path),
                "defect_table_index": defect_table_index,
                "defect_table_source": defect_table.source,
                "classnumber": classnumber,
                "representation": representation_name,
            },
            die_defect_threshold=die_defect_threshold,
        )
        partial = None
        partial_matched_only = None
        binary = None
        binary_matched_only = None
        if match_mode == "count":
            count_explanation = explain_count_partial_match(
                reference,
                gm,
                min_area=min_area,
                top_k=top_k,
                proposal_mode=proposal_mode,
                rotation_tolerance=rotation_tolerance,
                min_token_score=min_token_score,
                score_shape_weight=score_shape_weight,
                score_position_weight=score_position_weight,
                score_scale_weight=score_scale_weight,
                min_relative_token_area=min_relative_token_area,
                scale_area_weight=scale_area_weight,
                scale_pca_weight=scale_pca_weight,
                density_sigmas=density_sigmas,
                density_threshold=density_threshold,
                density_min_raw_points=density_min_raw_points,
                density_min_raw_mass=density_min_raw_mass,
                density_merge_iou=density_merge_iou,
                density_weight_transform=density_weight_transform,
                moment_weight=moment_weight,
                geometry_weight=geometry_weight,
                zernike_degree=zernike_degree,
                ring_min_area=ring_min_area,
                ring_edge_r_min=ring_edge_r_min,
                ring_band_width=ring_band_width,
                ring_min_angular_coverage=ring_min_angular_coverage,
                ring_angular_bins=ring_angular_bins,
                ring_max_radial_std=ring_max_radial_std,
                ring_max_defect_ratio=ring_max_defect_ratio,
                ring_min_edge_defect_fraction=ring_min_edge_defect_fraction,
            )
            partial = count_explanation["result"]
            partial_matched_only = count_explanation["result_matched_only"]
        if match_mode == "binary":
            binary, binary_matched_only = compute_binary_class_score(
                reference,
                gm,
                min_area=min_area,
                top_k=top_k,
                proposal_mode=proposal_mode,
                rotation_tolerance=rotation_tolerance,
                min_token_score=min_token_score,
                score_shape_weight=score_shape_weight,
                score_position_weight=score_position_weight,
                score_scale_weight=score_scale_weight,
                min_relative_token_area=min_relative_token_area,
                scale_area_weight=scale_area_weight,
                scale_pca_weight=scale_pca_weight,
                density_sigmas=density_sigmas,
                density_threshold=density_threshold,
                density_min_raw_points=density_min_raw_points,
                density_min_raw_mass=density_min_raw_mass,
                density_merge_iou=density_merge_iou,
                density_weight_transform=density_weight_transform,
                moment_weight=moment_weight,
                geometry_weight=geometry_weight,
                ring_min_area=ring_min_area,
                ring_edge_r_min=ring_edge_r_min,
                ring_band_width=ring_band_width,
                ring_min_angular_coverage=ring_min_angular_coverage,
                ring_angular_bins=ring_angular_bins,
                ring_max_radial_std=ring_max_radial_std,
                ring_max_defect_ratio=ring_max_defect_ratio,
                ring_min_edge_defect_fraction=ring_min_edge_defect_fraction,
            )

        rank_score = _rank_score(partial, binary, match_mode)
        splits.append(
            ClassSplitMatch(
                classnumber=classnumber,
                grid_maps=gm,
                partial=partial,
                partial_matched_only=partial_matched_only,
                binary=binary,
                binary_matched_only=binary_matched_only,
                rank_score=rank_score,
                rank_mode=match_mode,
            )
        )

    best = max(splits, key=lambda item: item.rank_score) if splits else None
    return ClassNumberMatchResult(splits=splits, best=best, match_mode=match_mode)


def classnumber_scores_dict(result: ClassNumberMatchResult) -> Dict[str, object]:
    """Flatten classnumber matching result for TSV output."""
    if result.best is None:
        return {
            "classnumber-count": "0",
            "best-classnumber": "",
            "best-classnumber-partial": "",
            "best-classnumber-tokens": "",
            "best-classnumber-binary": "",
            "best-classnumber-binary-shape": "",
            "best-classnumber-binary-position": "",
            "best-classnumber-binary-scale": "",
            "best-classnumber-binary-type": "",
            "best-classnumber-binary-tokens": "",
            "best-classnumber-binary-coverage": "",
            "best-classnumber-binary-leakage": "",
            "best-classnumber-rank-mode": result.match_mode,
            "best-classnumber-rank-score": "",
            "best-classnumber-mo-partial": "",
            "best-classnumber-mo-tokens": "",
            "best-classnumber-mo-binary": "",
            "best-classnumber-mo-binary-shape": "",
            "best-classnumber-mo-binary-position": "",
            "best-classnumber-mo-binary-scale": "",
            "best-classnumber-mo-binary-type": "",
            "best-classnumber-mo-binary-tokens": "",
            "best-classnumber-mo-rank-score": "",
        }
    partial_score = result.best.partial.score if result.best.partial is not None else ""
    partial_tokens = result.best.partial if result.best.partial is not None else ""
    binary_score = result.best.binary.score if result.best.binary is not None else ""
    mo_partial_score = result.best.partial_matched_only.score if result.best.partial_matched_only is not None else ""
    mo_partial_tokens = result.best.partial_matched_only if result.best.partial_matched_only is not None else ""
    mo_binary_score = result.best.binary_matched_only.score if result.best.binary_matched_only is not None else ""
    mo_rank_score = result.best.partial_matched_only.score if result.best.partial_matched_only is not None else ""
    return {
        "classnumber-count": str(len(result.splits)),
        "best-classnumber": str(result.best.classnumber),
        "best-classnumber-partial": partial_score,
        "best-classnumber-tokens": partial_tokens,
        "best-classnumber-binary": binary_score,
        "best-classnumber-binary-shape": result.best.binary.mean_shape if result.best.binary is not None else "",
        "best-classnumber-binary-position": result.best.binary.mean_position if result.best.binary is not None else "",
        "best-classnumber-binary-scale": result.best.binary.mean_scale if result.best.binary is not None else "",
        "best-classnumber-binary-type": result.best.binary.mean_type if result.best.binary is not None else "",
        "best-classnumber-binary-tokens": result.best.binary if result.best.binary is not None else "",
        "best-classnumber-binary-coverage": "",
        "best-classnumber-binary-leakage": "",
        "best-classnumber-rank-mode": result.match_mode,
        "best-classnumber-rank-score": result.best.rank_score,
        "best-classnumber-mo-partial": mo_partial_score,
        "best-classnumber-mo-tokens": mo_partial_tokens,
        "best-classnumber-mo-binary": mo_binary_score,
        "best-classnumber-mo-binary-shape": result.best.binary_matched_only.mean_shape if result.best.binary_matched_only is not None else "",
        "best-classnumber-mo-binary-position": result.best.binary_matched_only.mean_position if result.best.binary_matched_only is not None else "",
        "best-classnumber-mo-binary-scale": result.best.binary_matched_only.mean_scale if result.best.binary_matched_only is not None else "",
        "best-classnumber-mo-binary-type": result.best.binary_matched_only.mean_type if result.best.binary_matched_only is not None else "",
        "best-classnumber-mo-binary-tokens": result.best.binary_matched_only if result.best.binary_matched_only is not None else "",
        "best-classnumber-mo-rank-score": mo_rank_score,
    }


def compute_binary_class_score(
    reference: GridMaps,
    candidate: GridMaps,
    min_area: int = 5,
    top_k: int = 6,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    min_token_score: float = 0.45,
    score_shape_weight: float = 0.60,
    score_position_weight: float = 0.25,
    score_scale_weight: float = 0.15,
    min_relative_token_area: float = 0.10,
    scale_area_weight: float = 0.50,
    scale_pca_weight: float = 0.50,
    density_sigmas: tuple[float, ...] = (0.8, 1.6, 3.2),
    density_threshold: float = 0.20,
    density_min_raw_points: int = 3,
    density_min_raw_mass: float = 3.0,
    density_merge_iou: float = 0.60,
    density_weight_transform: str = "sqrt",
    moment_weight: float = 0.75,
    geometry_weight: float = 0.25,
    zernike_degree: int = 8,
    ring_min_area: int | None = None,
    ring_edge_r_min: float | None = None,
    ring_band_width: float | None = None,
    ring_min_angular_coverage: float | None = None,
    ring_angular_bins: int | None = None,
    ring_max_radial_std: float | None = None,
    ring_max_defect_ratio: float | None = None,
    ring_min_edge_defect_fraction: float | None = None,
) -> tuple[LocalMatchResult, LocalMatchResult]:
    """Score a classnumber split using binary-token partial matching. Returns (result, result_matched_only)."""
    explanation = explain_binary_partial_match(
        reference,
        candidate,
        min_area=min_area,
        top_k=top_k,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
        min_token_score=min_token_score,
        score_shape_weight=score_shape_weight,
        score_position_weight=score_position_weight,
        score_scale_weight=score_scale_weight,
        min_relative_token_area=min_relative_token_area,
        scale_area_weight=scale_area_weight,
        scale_pca_weight=scale_pca_weight,
        density_sigmas=density_sigmas,
        density_threshold=density_threshold,
        density_min_raw_points=density_min_raw_points,
        density_min_raw_mass=density_min_raw_mass,
        density_merge_iou=density_merge_iou,
        density_weight_transform=density_weight_transform,
        moment_weight=moment_weight,
        geometry_weight=geometry_weight,
        zernike_degree=zernike_degree,
        ring_min_area=ring_min_area,
        ring_edge_r_min=ring_edge_r_min,
        ring_band_width=ring_band_width,
        ring_min_angular_coverage=ring_min_angular_coverage,
        ring_angular_bins=ring_angular_bins,
        ring_max_radial_std=ring_max_radial_std,
        ring_max_defect_ratio=ring_max_defect_ratio,
        ring_min_edge_defect_fraction=ring_min_edge_defect_fraction,
    )
    return explanation["result"], explanation["result_matched_only"]


def split_score(split: ClassSplitMatch, mode: str) -> float:
    mode = mode.lower()
    if mode == "count":
        return float(split.partial.score) if split.partial is not None else float("-inf")
    if mode == "binary":
        return float(split.binary.score) if split.binary is not None else float("-inf")
    raise ValueError(f"Unsupported classnumber score mode: {mode}")


def _rank_score(
    partial: LocalMatchResult | None,
    binary: LocalMatchResult | None,
    match_mode: str,
) -> float:
    if match_mode == "count":
        return float(partial.score) if partial is not None else float("-inf")
    return float(binary.score) if binary is not None else float("-inf")


def _normalize_match_mode(match_mode: str) -> str:
    match_mode = match_mode.lower()
    if match_mode not in {"count", "binary"}:
        raise ValueError(f"Unsupported classnumber match mode: {match_mode}")
    return match_mode
