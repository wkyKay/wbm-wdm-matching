# Optional classnumber-based WDM split matching.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ..data.fileio import load_defect_tables, split_defect_table_by_classnumber
from .local_matching import LocalMatchResult, compute_binary_partial_match, compute_count_partial_match
from .models import GridMaps
from .pipeline import build_mapper, map_defect_table_to_grid


@dataclass(frozen=True)
class ClassSplitMatch:
    classnumber: int
    grid_maps: GridMaps
    partial: LocalMatchResult | None
    binary: LocalMatchResult | None
    rank_score: float
    rank_mode: str


@dataclass(frozen=True)
class ClassNumberMatchResult:
    splits: List[ClassSplitMatch]
    best: ClassSplitMatch | None
    match_mode: str
    rank_by: str


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
    rank_by: str = "count",
    binary_dilation: int = 1,
    binary_beta: float = 0.5,
    die_defect_threshold: int = 1,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    min_token_score: float = 0.45,
    score_shape_weight: float = 0.60,
    score_position_weight: float = 0.25,
    score_scale_weight: float = 0.15,
) -> ClassNumberMatchResult:
    """Split one KLARF DefectTable by classnumber and score each split."""
    match_mode = _normalize_match_mode(match_mode)
    rank_by = _normalize_rank_by(match_mode, rank_by)
    tables = load_defect_tables(klarf_path)
    if defect_table_index < 0 or defect_table_index >= len(tables):
        raise IndexError(
            f"defect_table_index={defect_table_index} out of range; found {len(tables)} DefectList tables"
        )

    split_tables = split_defect_table_by_classnumber(tables[defect_table_index])
    if not split_tables:
        return ClassNumberMatchResult(splits=[], best=None, match_mode=match_mode, rank_by=rank_by)

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
        binary = None
        if match_mode in ("count", "both"):
            partial = compute_count_partial_match(
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
            )
        if match_mode in ("binary", "both"):
            binary = compute_binary_class_score(
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
            )

        rank_score = _rank_score(partial, binary, rank_by)
        splits.append(
            ClassSplitMatch(
                classnumber=classnumber,
                grid_maps=gm,
                partial=partial,
                binary=binary,
                rank_score=rank_score,
                rank_mode=rank_by,
            )
        )

    best = max(splits, key=lambda item: item.rank_score) if splits else None
    return ClassNumberMatchResult(splits=splits, best=best, match_mode=match_mode, rank_by=rank_by)


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
            "best-classnumber-rank-mode": result.rank_by,
            "best-classnumber-rank-score": "",
        }
    partial_score = result.best.partial.score if result.best.partial is not None else ""
    partial_tokens = result.best.partial if result.best.partial is not None else ""
    binary_score = result.best.binary.score if result.best.binary is not None else ""
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
        "best-classnumber-rank-mode": result.rank_by,
        "best-classnumber-rank-score": result.best.rank_score,
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
) -> LocalMatchResult:
    """Score a classnumber split using binary-token partial matching."""
    return compute_binary_partial_match(
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
    )


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
    rank_by: str,
) -> float:
    if rank_by == "count":
        return float(partial.score) if partial is not None else float("-inf")
    return float(binary.score) if binary is not None else float("-inf")


def _normalize_match_mode(match_mode: str) -> str:
    match_mode = match_mode.lower()
    if match_mode not in {"count", "binary", "both"}:
        raise ValueError(f"Unsupported classnumber match mode: {match_mode}")
    return match_mode


def _normalize_rank_by(match_mode: str, rank_by: str) -> str:
    rank_by = rank_by.lower()
    if match_mode == "binary":
        return "binary"
    if match_mode == "count":
        return "count"
    if rank_by not in {"count", "binary"}:
        raise ValueError(f"Unsupported classnumber rank mode: {rank_by}")
    return rank_by
