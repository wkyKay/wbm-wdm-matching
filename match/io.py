# KLARF / WBM file I/O: read shapes, parse defect tables, load die pitch, save outputs.
from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Mapping, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import klarfio  # noqa: E402

from .models import DefectTable, GridMaps


def read_wbm_shape(path: str | Path) -> Tuple[int, int]:
    """不依赖额外图像库，直接从 PNG 头读取 HxW。"""
    path = Path(path)
    with path.open("rb") as file:
        header = file.read(24)

    if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 24:
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return height, width

    raise ValueError(
        f"Cannot read WBM shape from {path}. Only PNG is supported without image dependencies; "
        "pass --height and --width instead."
    )


def load_defect_tables(klarf_path: str | Path) -> List[DefectTable]:
    """解析 KLARF 文件，收集其中所有的 DefectList 表。"""
    parsed = klarfio.klarf(filename=str(klarf_path))
    tables: List[DefectTable] = []
    _collect_defect_tables(parsed.data, tables, source="root")
    if not tables:
        raise ValueError(f"No DefectList found in {klarf_path}")
    return tables


def _collect_defect_tables(node: object, tables: List[DefectTable], source: str) -> None:
    """递归遍历解析后的 KLARF 树，寻找包含缺陷列的表。"""
    if not isinstance(node, Mapping):
        return

    if "Columns" in node and "Data" in node:
        columns = [item["Column"] for item in node["Columns"]]
        if {"DEFECTID", "XINDEX", "YINDEX"}.issubset(columns):
            rows = np.asarray(node["Data"], dtype=float)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
            tables.append(DefectTable(columns=columns, rows=rows, source=source))

    for key, value in node.items():
        if isinstance(value, Mapping):
            _collect_defect_tables(value, tables, source=f"{source}/{key}")


def load_die_pitch(klarf_path: str | Path) -> tuple[float, float]:
    """从 KLARF LotRecord 中提取 DiePitch。"""
    parsed = klarfio.klarf(filename=str(klarf_path))
    file_data = parsed.data["FileRecord_1.8"]
    lot_keys = [k for k in file_data if k.startswith("LotRecord")]
    if not lot_keys:
        raise ValueError(f"No LotRecord found in {klarf_path}")
    lot_data = file_data[lot_keys[0]]
    if "DiePitch" not in lot_data:
        raise ValueError(f"No DiePitch found in LotRecord of {klarf_path}")
    die_pitch = lot_data["DiePitch"]
    return float(die_pitch[0]), float(die_pitch[1])


def save_grid_maps(path: str | Path, grid_maps: GridMaps) -> None:
    """把所有 map 一起保存为 npz，后续相似度代码可以直接复用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        count_map=grid_maps.count_map,
        binary_map=grid_maps.binary_map,
        density_map=grid_maps.density_map,
        soft_map=grid_maps.representation_maps["soft"],
        three_value_map=grid_maps.representation_maps["three-value"],
        mountain_map=grid_maps.representation_maps["mountain"],
        representation_map=grid_maps.representation_map,
        status_map=grid_maps.status_map,
        metadata=np.asarray(grid_maps.metadata, dtype=object),
    )
