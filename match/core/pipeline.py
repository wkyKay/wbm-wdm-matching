# Main pipeline: coordinate mapping → representation building → GridMaps.
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ..data.fileio import load_defect_tables, load_die_pitch, load_die_index_bounds
from .mappers import MAPPERS, GridMapper, DieIndexGridMapper, PhysicalCoordinateGridMapper
from .representations import REPRESENTATIONS, DensityMapBuilder
from .models import DefectTable, GridMaps


def map_klarf_to_grid(
    klarf_path: str | Path,
    shape: Tuple[int, int],
    mapper_name: str = DieIndexGridMapper.name,
    representation_name: str = DensityMapBuilder.name,
    defect_table_index: int = 0,
    die_x_range: Tuple[int, int] | None = None,
    die_y_range: Tuple[int, int] | None = None,
    die_defect_threshold: int = 1,
) -> GridMaps:
    """第一层负责把缺陷放进目标网格，第二层把同一份 count/status 转成不同 baseline 表达。"""
    tables = load_defect_tables(klarf_path)
    if defect_table_index < 0 or defect_table_index >= len(tables):
        raise IndexError(
            f"defect_table_index={defect_table_index} out of range; found {len(tables)} DefectList tables"
        )
    defect_table = tables[defect_table_index]

    mapper = build_mapper(
        klarf_path=klarf_path,
        mapper_name=mapper_name,
        die_x_range=die_x_range,
        die_y_range=die_y_range,
    )
    metadata = {
        "klarf_path": str(klarf_path),
        "defect_table_index": defect_table_index,
        "defect_table_source": defect_table.source,
        "defect_table_count": len(tables),
        "representation": representation_name,
    }
    return map_defect_table_to_grid(
        defect_table,
        shape=shape,
        mapper=mapper,
        representation_name=representation_name,
        metadata=metadata,
        die_defect_threshold=die_defect_threshold,
    )


def build_mapper(
    klarf_path: str | Path,
    mapper_name: str = DieIndexGridMapper.name,
    die_x_range: Tuple[int, int] | None = None,
    die_y_range: Tuple[int, int] | None = None,
) -> GridMapper:
    """Build a mapper instance for a KLARF file and mapper configuration."""

    mapper_cls = MAPPERS.get(mapper_name)
    if mapper_cls is None:
        choices = ", ".join(sorted(MAPPERS))
        raise KeyError(f"Unknown mapper {mapper_name!r}. Available mappers: {choices}")

    # 构建 mapper 实例
    if mapper_cls is PhysicalCoordinateGridMapper:
        if die_x_range is None or die_y_range is None:
            x_min, x_max, y_min, y_max = load_die_index_bounds(klarf_path)
            if die_x_range is None:
                die_x_range = (x_min, x_max)
            if die_y_range is None:
                die_y_range = (y_min, y_max)
        die_pitch_x, die_pitch_y = load_die_pitch(klarf_path)
        mapper: GridMapper = PhysicalCoordinateGridMapper(
            die_pitch_x=die_pitch_x,
            die_pitch_y=die_pitch_y,
            x_index_min=float(die_x_range[0]),
            x_index_max=float(die_x_range[1]),
            y_index_min=float(die_y_range[0]),
            y_index_max=float(die_y_range[1]),
        )
    elif die_x_range is not None and mapper_cls is DieIndexGridMapper:
        mapper = DieIndexGridMapper(
            x_index_min=float(die_x_range[0]),
            x_index_max=float(die_x_range[1]),
            y_index_min=float(die_y_range[0]) if die_y_range else None,
            y_index_max=float(die_y_range[1]) if die_y_range else None,
        )
    else:
        mapper = mapper_cls()
    return mapper


def map_defect_table_to_grid(
    defect_table: DefectTable,
    shape: Tuple[int, int],
    mapper: GridMapper,
    representation_name: str = DensityMapBuilder.name,
    metadata: dict | None = None,
    die_defect_threshold: int = 1,
) -> GridMaps:
    """Map an already-loaded DefectTable to GridMaps."""
    if representation_name not in REPRESENTATIONS:
        choices = ", ".join(sorted(REPRESENTATIONS))
        raise KeyError(f"Unknown representation {representation_name!r}. Available representations: {choices}")

    mapped = mapper.map(defect_table, shape)
    count_map = mapped["count_map"]
    status_map = mapped["status_map"]
    row_indices = mapped["row_indices"]
    col_indices = mapped["col_indices"]
    representation_maps = {
        name: builder.build(count_map, status_map, row_indices, col_indices)
        for name, builder in REPRESENTATIONS.items()
    }
    threshold = max(int(die_defect_threshold), 1)
    representation_maps["binary"] = (count_map >= threshold).astype("uint8")

    merged_metadata = mapped["metadata"]
    if metadata:
        merged_metadata.update(metadata)
    merged_metadata.update({
        "representation": representation_name,
        "die_defect_threshold": threshold,
    })
    return GridMaps(
        count_map=representation_maps["count"],
        binary_map=representation_maps["binary"],
        density_map=representation_maps["density"],
        status_map=status_map,
        representation_map=representation_maps[representation_name],
        representation_maps=representation_maps,
        metadata=merged_metadata,
    )
