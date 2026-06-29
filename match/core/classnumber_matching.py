# Optional classnumber-based WDM split matching.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..data.fileio import load_defect_tables, split_defect_table_by_classnumber
from .local_matching import LocalMatchResult, compute_count_partial_match
from .models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT
from .pipeline import build_mapper, map_defect_table_to_grid


@dataclass(frozen=True)
class BinaryClassScore:
    score: float
    coverage: float
    leakage: float
    wbm_pixels: int
    wdm_pixels: int
    overlap_pixels: int


@dataclass(frozen=True)
class ClassSplitMatch:
    classnumber: int
    grid_maps: GridMaps
    partial: LocalMatchResult | None
    binary: BinaryClassScore | None
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
        )
        partial = None
        binary = None
        if match_mode in ("count", "both"):
            partial = compute_count_partial_match(reference, gm, min_area=min_area, top_k=top_k)
        if match_mode in ("binary", "both"):
            binary = compute_binary_class_score(
                reference,
                gm,
                dilation=binary_dilation,
                beta=binary_beta,
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
            "best-classnumber-binary-coverage": "",
            "best-classnumber-binary-leakage": "",
            "best-classnumber-rank-mode": result.rank_by,
            "best-classnumber-rank-score": "",
        }
    partial_score = result.best.partial.score if result.best.partial is not None else ""
    partial_tokens = result.best.partial if result.best.partial is not None else ""
    binary_score = result.best.binary.score if result.best.binary is not None else ""
    binary_coverage = result.best.binary.coverage if result.best.binary is not None else ""
    binary_leakage = result.best.binary.leakage if result.best.binary is not None else ""
    return {
        "classnumber-count": str(len(result.splits)),
        "best-classnumber": str(result.best.classnumber),
        "best-classnumber-partial": partial_score,
        "best-classnumber-tokens": partial_tokens,
        "best-classnumber-binary": binary_score,
        "best-classnumber-binary-coverage": binary_coverage,
        "best-classnumber-binary-leakage": binary_leakage,
        "best-classnumber-rank-mode": result.rank_by,
        "best-classnumber-rank-score": result.best.rank_score,
    }


def compute_binary_class_score(
    reference: GridMaps,
    candidate: GridMaps,
    dilation: int = 1,
    beta: float = 0.5,
) -> BinaryClassScore:
    """Score a classnumber split using binary coverage minus leakage."""
    valid_mask = (reference.status_map == VALID_NO_DEFECT) | (reference.status_map == VALID_HAS_DEFECT)
    wbm_mask = (reference.status_map == VALID_HAS_DEFECT) & valid_mask
    wdm_mask = (candidate.binary_map > 0) & valid_mask
    if dilation > 0:
        wdm_match_mask = _dilate_binary(wdm_mask, radius=dilation) & valid_mask
    else:
        wdm_match_mask = wdm_mask

    overlap = wbm_mask & wdm_match_mask
    wbm_pixels = int(wbm_mask.sum())
    wdm_pixels = int(wdm_mask.sum())
    overlap_pixels = int(overlap.sum())
    coverage = float(overlap_pixels / max(wbm_pixels, 1))

    no_defect_mask = (reference.status_map == VALID_NO_DEFECT) & valid_mask
    leakage_pixels = int((wdm_mask & no_defect_mask).sum())
    leakage = float(leakage_pixels / max(wdm_pixels, 1))
    score = float(coverage - beta * leakage)

    return BinaryClassScore(
        score=score,
        coverage=coverage,
        leakage=leakage,
        wbm_pixels=wbm_pixels,
        wdm_pixels=wdm_pixels,
        overlap_pixels=overlap_pixels,
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
    binary: BinaryClassScore | None,
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


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    if radius == 0:
        return mask.copy()

    out = mask.astype(bool).copy()
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    h, w = mask.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            r0 = radius + dr
            c0 = radius + dc
            out |= padded[r0:r0 + h, c0:c0 + w]
    return out
