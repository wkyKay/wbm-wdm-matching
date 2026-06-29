# Optional classnumber-based WDM split matching.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ..data.fileio import load_defect_tables, split_defect_table_by_classnumber
from .local_matching import LocalMatchResult, compute_count_partial_match
from .models import GridMaps
from .pipeline import build_mapper, map_defect_table_to_grid


@dataclass(frozen=True)
class ClassSplitMatch:
    classnumber: int
    grid_maps: GridMaps
    partial: LocalMatchResult


@dataclass(frozen=True)
class ClassNumberMatchResult:
    splits: List[ClassSplitMatch]
    best: ClassSplitMatch | None


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
) -> ClassNumberMatchResult:
    """Split one KLARF DefectTable by classnumber and score each split."""
    tables = load_defect_tables(klarf_path)
    if defect_table_index < 0 or defect_table_index >= len(tables):
        raise IndexError(
            f"defect_table_index={defect_table_index} out of range; found {len(tables)} DefectList tables"
        )

    split_tables = split_defect_table_by_classnumber(tables[defect_table_index])
    if not split_tables:
        return ClassNumberMatchResult(splits=[], best=None)

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
        partial = compute_count_partial_match(reference, gm, min_area=min_area, top_k=top_k)
        splits.append(ClassSplitMatch(classnumber=classnumber, grid_maps=gm, partial=partial))

    best = max(splits, key=lambda item: item.partial.score) if splits else None
    return ClassNumberMatchResult(splits=splits, best=best)


def classnumber_scores_dict(result: ClassNumberMatchResult) -> Dict[str, object]:
    """Flatten classnumber matching result for TSV output."""
    if result.best is None:
        return {
            "classnumber-count": "0",
            "best-classnumber": "",
            "best-classnumber-partial": "",
            "best-classnumber-tokens": "",
        }
    return {
        "classnumber-count": str(len(result.splits)),
        "best-classnumber": str(result.best.classnumber),
        "best-classnumber-partial": result.best.partial.score,
        "best-classnumber-tokens": result.best.partial,
    }
