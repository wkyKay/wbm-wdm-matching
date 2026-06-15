# Coordinate mapping strategies: die-index, relative-coordinate, physical-coordinate.
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
from .models import DefectTable, VALID_NO_DEFECT, VALID_HAS_DEFECT


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
    name = "physical-coordinate"

    def __init__(
        self,
        die_pitch_x: float = 1.0,
        die_pitch_y: float = 1.0,
        x_index_min: float | None = None,
        x_index_max: float | None = None,
        y_index_min: float | None = None,
        y_index_max: float | None = None,
    ):
        self.die_pitch_x = die_pitch_x
        self.die_pitch_y = die_pitch_y
        self.x_index_min = x_index_min
        self.x_index_max = x_index_max
        self.y_index_min = y_index_min
        self.y_index_max = y_index_max

    def map(self, defects: DefectTable, shape: Tuple[int, int]) -> Dict[str, object]:
        xindex = defects.column("XINDEX")
        yindex = defects.column("YINDEX")
        xrel = defects.column("XREL")
        yrel = defects.column("YREL")
        physical_x = xindex * self.die_pitch_x + xrel
        physical_y = yindex * self.die_pitch_y + yrel

        # die 网格范围 → 物理边界；未指定则从缺陷数据自动推算
        if self.x_index_min is not None and self.x_index_max is not None:
            x_frame_min = self.x_index_min * self.die_pitch_x
            x_frame_max = (self.x_index_max + 1.0) * self.die_pitch_x
        else:
            x_frame_min, x_frame_max = None, None
        if self.y_index_min is not None and self.y_index_max is not None:
            y_frame_min = self.y_index_min * self.die_pitch_y
            y_frame_max = (self.y_index_max + 1.0) * self.die_pitch_y
        else:
            y_frame_min, y_frame_max = None, None

        col_indices, _, _ = _scale_to_grid(physical_x, shape[1], value_min=x_frame_min, value_max=x_frame_max)
        row_indices, _, _ = _scale_to_grid(physical_y, shape[0], invert=True, value_min=y_frame_min, value_max=y_frame_max)
        return self._maps_from_indices(
            row_indices,
            col_indices,
            shape,
            {
                "x_column": "physical_x (XINDEX*DiePitchX+XREL)",
                "y_column": "physical_y (YINDEX*DiePitchY+YREL)",
                "die_pitch_x": self.die_pitch_x,
                "die_pitch_y": self.die_pitch_y,
                "x_frame_min": x_frame_min,
                "x_frame_max": x_frame_max,
                "y_frame_min": y_frame_min,
                "y_frame_max": y_frame_max,
            },
        )


MAPPERS: Dict[str, GridMapper] = {
    DieIndexGridMapper.name: DieIndexGridMapper(),
    RelativeCoordinateGridMapper.name: RelativeCoordinateGridMapper(),
    PhysicalCoordinateGridMapper.name: PhysicalCoordinateGridMapper(),
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
