# Coordinate mapping strategies: die-index, relative-coordinate, physical-coordinate.
from __future__ import annotations
from typing import Dict, Sequence, Tuple

import numpy as np

from .models import DefectTable, BACKGROUND, VALID_NO_DEFECT, VALID_HAS_DEFECT, UNINSPECTED


class GridMapper:
    name = "base"

    def map(self, defects: DefectTable, shape: Tuple[int, int]) -> Dict[str, object]:
        raise NotImplementedError

    def _maps_from_indices(
        self,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        shape: Tuple[int, int],
        metadata: Dict[str, object],
        status_map: np.ndarray | None = None,
    ) -> Dict[str, object]:
        """共享的低层逻辑：把缺陷点累加到目标网格单元中。"""
        height, width = shape
        valid = (
            (row_indices >= 0)
            & (row_indices < height)
            & (col_indices >= 0)
            & (col_indices < width)
        )

        count_map = np.zeros((height, width), dtype=np.int32)
        np.add.at(count_map, (row_indices[valid], col_indices[valid]), 1)

        if status_map is None:
            status_map = np.full((height, width), VALID_NO_DEFECT, dtype=np.uint8)
        status_map[count_map > 0] = VALID_HAS_DEFECT

        metadata = dict(metadata)
        metadata.update(
            {
                "coordinate_mapper": self.name,
                "target_height": height,
                "target_width": width,
                "input_defects": int(len(row_indices)),
                "mapped_defects": int(valid.sum()),
            }
        )
        return {
            "count_map": count_map,
            "status_map": status_map,
            "row_indices": row_indices[valid],
            "col_indices": col_indices[valid],
            "metadata": metadata,
        }


class DieIndexGridMapper(GridMapper):
    name = "die-index"

    def __init__(
        self,
        x_index_min: float | None = None,
        x_index_max: float | None = None,
        y_index_min: float | None = None,
        y_index_max: float | None = None,
    ):
        self.x_index_min = x_index_min
        self.x_index_max = x_index_max
        self.y_index_min = y_index_min
        self.y_index_max = y_index_max

    def map(self, defects: DefectTable, shape: Tuple[int, int]) -> Dict[str, object]:
        xindex = defects.column("XINDEX")
        yindex = defects.column("YINDEX")
        col_indices, x_min, x_max = _scale_to_grid(
            xindex, shape[1], value_min=self.x_index_min, value_max=self.x_index_max,
        )
        row_indices, y_min, y_max = _scale_to_grid(
            yindex, shape[0], invert=True, value_min=self.y_index_min, value_max=self.y_index_max,
        )
        return self._maps_from_indices(
            row_indices,
            col_indices,
            shape,
            {
                "x_column": "XINDEX",
                "y_column": "YINDEX",
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            },
        )


class RelativeCoordinateGridMapper(GridMapper):
    name = "relative-coordinate"

    def map(self, defects: DefectTable, shape: Tuple[int, int]) -> Dict[str, object]:
        xrel = defects.column("XREL")
        yrel = defects.column("YREL")
        col_indices, x_min, x_max = _scale_to_grid(xrel, shape[1])
        row_indices, y_min, y_max = _scale_to_grid(yrel, shape[0], invert=True)
        return self._maps_from_indices(
            row_indices,
            col_indices,
            shape,
            {
                "x_column": "XREL",
                "y_column": "YREL",
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            },
        )


class PhysicalCoordinateGridMapper(GridMapper):
    """将 die 索引 + die 内相对坐标合并为归一化位置，直接映射到 WBM 网格。

    参数:
      die_pitch_x, die_pitch_y  — 从 KLARF DiePitch 自动读取
      x_index_min, x_index_max  — die 网格 X 方向最小/最大索引 (如 -20, 20)
      y_index_min, y_index_max  — die 网格 Y 方向最小/最大索引 (如 -20, 20)
    """

    name = "physical-coordinate"

    def __init__(
        self,
        die_pitch_x: float,
        die_pitch_y: float,
        x_index_min: float,
        x_index_max: float,
        y_index_min: float,
        y_index_max: float,
    ):
        self.die_pitch_x = die_pitch_x
        self.die_pitch_y = die_pitch_y
        self.x_index_min = x_index_min
        self.x_index_max = x_index_max
        self.y_index_min = y_index_min
        self.y_index_max = y_index_max

    def map(self, defects: DefectTable, shape: Tuple[int, int]) -> Dict[str, object]:
        height, width = shape

        wafer_die_cols = int(self.x_index_max - self.x_index_min + 1)
        wafer_die_rows = int(self.y_index_max - self.y_index_min + 1)

        if wafer_die_cols == 1 and wafer_die_rows == 1:
            raise ValueError(
                f"Single-die wafer (cols={wafer_die_cols}, rows={wafer_die_rows}) is not supported. "
                "Skip this wafer."
            )

        xindex = defects.column("XINDEX")
        yindex = defects.column("YINDEX")
        xrel = defects.column("XREL")
        yrel = defects.column("YREL")

        # 归一化到 [0, 1]：整数部分=die 偏移，小数部分=die 内位置
        x_norm = (xindex - self.x_index_min + xrel / self.die_pitch_x) / wafer_die_cols
        y_norm = (yindex - self.y_index_min + yrel / self.die_pitch_y) / wafer_die_rows

        col_indices = np.floor(x_norm * width).astype(np.int64)
        row_indices = np.floor((1.0 - y_norm) * height).astype(np.int64)  # Y 翻转

        # 圆形 wafer mask
        center = ((height - 1) / 2.0, (width - 1) / 2.0)
        radius = min(height, width) / 2.0
        rows_grid, cols_grid = np.ogrid[:height, :width]
        inside_wafer = (rows_grid - center[0]) ** 2 + (cols_grid - center[1]) ** 2 <= radius ** 2
        status_map = np.where(inside_wafer, VALID_NO_DEFECT, UNINSPECTED).astype(np.uint8)

        return self._maps_from_indices(
            row_indices,
            col_indices,
            shape,
            {
                "die_pitch_x": self.die_pitch_x,
                "die_pitch_y": self.die_pitch_y,
                "x_index_min": self.x_index_min,
                "x_index_max": self.x_index_max,
                "y_index_min": self.y_index_min,
                "y_index_max": self.y_index_max,
                "wafer_die_cols": wafer_die_cols,
                "wafer_die_rows": wafer_die_rows,
            },
            status_map=status_map,
        )


MAPPERS: Dict[str, type[GridMapper]] = {
    DieIndexGridMapper.name: DieIndexGridMapper,
    RelativeCoordinateGridMapper.name: RelativeCoordinateGridMapper,
    PhysicalCoordinateGridMapper.name: PhysicalCoordinateGridMapper,
}


def _scale_to_grid(
    values: Sequence[float],
    size: int,
    invert: bool = False,
    value_min: float | None = None,
    value_max: float | None = None,
) -> Tuple[np.ndarray, float, float]:
    """把坐标线性缩放到 [0, size-1]。
    若传入 value_min/value_max 则作为固定参考系，否则从数据自动推算。
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=np.int64), 0.0, 0.0

    if value_min is None:
        value_min = float(np.nanmin(values))
    if value_max is None:
        value_max = float(np.nanmax(values))

    if value_max == value_min:
        scaled = np.zeros_like(values, dtype=float)
    else:
        scaled = (values - value_min) / (value_max - value_min)
    if invert:
        scaled = 1.0 - scaled
    indices = np.floor(scaled * size).astype(np.int64)
    indices = np.clip(indices, 0, size - 1)
    return indices, value_min, value_max
