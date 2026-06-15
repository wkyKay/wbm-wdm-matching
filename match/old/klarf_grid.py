from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import klarfio  # noqa: E402


BACKGROUND = 0
VALID_NO_DEFECT = 1
VALID_HAS_DEFECT = 2
UNINSPECTED = 3


@dataclass(frozen=True)
class DefectTable:
    # 对一个 KLARF DefectList 表的轻量封装。
    columns: List[str]
    rows: np.ndarray
    source: str

    def column(self, name: str) -> np.ndarray:
        try:
            index = self.columns.index(name)
        except ValueError as exc:
            raise KeyError(f"KLARF DefectList has no column {name!r}") from exc
        return self.rows[:, index]


@dataclass(frozen=True)
class GridMaps:
    # 一个 DefectList 派生出的所有网格表达。
    count_map: np.ndarray
    binary_map: np.ndarray
    density_map: np.ndarray
    status_map: np.ndarray
    representation_map: np.ndarray
    representation_maps: Dict[str, np.ndarray]
    metadata: Dict[str, object]


def read_wbm_shape(path: str | Path) -> Tuple[int, int]:
    # 不依赖额外图像库，直接从 PNG 头读取尺寸。
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
    # 先解析 KLARF，再收集文件中的所有 DefectList。
    parsed = klarfio.klarf(filename=str(klarf_path))
    tables: List[DefectTable] = []
    _collect_defect_tables(parsed.data, tables, source="root")
    if not tables:
        raise ValueError(f"No DefectList found in {klarf_path}")
    return tables


def _collect_defect_tables(node: object, tables: List[DefectTable], source: str) -> None:
    # 递归遍历解析后的 KLARF 树，寻找包含缺陷列的表。
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
        # 共享的低层逻辑：把缺陷点累加到目标网格单元中。
        height, width = shape #wbm的尺寸
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

    def map(self, defects: DefectTable, shape: Tuple[int, int]) -> GridMaps:
        # 按 die 索引映射，再把索引归一化到目标网格。
        xindex = defects.column("XINDEX")
        yindex = defects.column("YINDEX")
        col_indices, x_min, x_max = _scale_to_grid(xindex, shape[1])
        row_indices, y_min, y_max = _scale_to_grid(yindex, shape[0], invert=True)
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

    def map(self, defects: DefectTable, shape: Tuple[int, int]) -> GridMaps:
        # 使用相对坐标字段进行映射，而不是 die 索引。
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

    def __init__(self, die_pitch_x: float = 1.0, die_pitch_y: float = 1.0):
        self.die_pitch_x = die_pitch_x
        self.die_pitch_y = die_pitch_y

    def map(self, defects: DefectTable, shape: Tuple[int, int]) -> Dict[str, object]:
        xindex = defects.column("XINDEX")
        yindex = defects.column("YINDEX")
        xrel = defects.column("XREL")
        yrel = defects.column("YREL")
        physical_x = xindex * self.die_pitch_x + xrel
        physical_y = yindex * self.die_pitch_y + yrel
        col_indices, x_min, x_max = _scale_to_grid(physical_x, shape[1])
        row_indices, y_min, y_max = _scale_to_grid(physical_y, shape[0], invert=True)
        return self._maps_from_indices(
            row_indices,
            col_indices,
            shape,
            {
                "x_column": "physical_x (XINDEX*DiePitchX+XREL)",
                "y_column": "physical_y (YINDEX*DiePitchY+YREL)",
                "die_pitch_x": self.die_pitch_x,
                "die_pitch_y": self.die_pitch_y,
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            },
        )


MAPPERS: Dict[str, GridMapper] = {
    DieIndexGridMapper.name: DieIndexGridMapper(),
    RelativeCoordinateGridMapper.name: RelativeCoordinateGridMapper(),
    PhysicalCoordinateGridMapper.name: PhysicalCoordinateGridMapper(),
}


class RepresentationBuilder:
    name = "base"

    def build(
        self,
        count_map: np.ndarray,
        status_map: np.ndarray,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class BinaryMapBuilder(RepresentationBuilder):
    name = "binary"

    def build(self, count_map, status_map, row_indices, col_indices):
        # 单元格内只要存在缺陷就是 1，否则为 0。
        return (count_map > 0).astype(np.uint8)


class CountMapBuilder(RepresentationBuilder):
    name = "count"

    def build(self, count_map, status_map, row_indices, col_indices):
        # 每个单元格内的原始缺陷数量。
        return count_map.astype(np.int32)


class DensityMapBuilder(RepresentationBuilder):
    name = "density"

    def build(self, count_map, status_map, row_indices, col_indices):
        # 将 count map 归一化为类似概率分布的 density map。
        density_map = count_map.astype(np.float32)
        total = float(density_map.sum())
        if total > 0:
            density_map /= total
        return density_map


class SoftMapBuilder(RepresentationBuilder):
    name = "soft"

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

        radius = max(1, int(round(3 * sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
        self.kernel = kernel / kernel.sum()

    def build(self, count_map, status_map, row_indices, col_indices):
        # 用高斯核平滑归一化后的 density map。
        density_map = DensityMapBuilder().build(count_map, status_map, row_indices, col_indices)
        return _separable_convolution(density_map, self.kernel).astype(np.float32)


class ThreeValueMapBuilder(RepresentationBuilder):
    name = "three-value"

    def __init__(self, strong_threshold: int = 2):
        self.strong_threshold = strong_threshold

    def build(self, count_map, status_map, row_indices, col_indices):
        # 把强证据、弱证据、无证据编码为 1.0、0.5、0.0。
        tri_map = np.zeros(count_map.shape, dtype=np.float32)
        tri_map[count_map >= self.strong_threshold] = 1.0
        tri_map[(count_map > 0) & (count_map < self.strong_threshold)] = 0.5
        neighbor_support = _neighbor_sum(count_map > 0) > 0
        tri_map[(count_map == 0) & neighbor_support] = 0.5
        return tri_map


class MountainMapBuilder(RepresentationBuilder):
    name = "mountain"

    def __init__(self, sigma: float = 1.5):
        self.sigma = sigma

    def build(self, count_map, status_map, row_indices, col_indices):
        # 每个缺陷点在目标网格上形成一个高斯“山峰”。
        height, width = count_map.shape
        mountain_map = np.zeros((height, width), dtype=np.float32)
        if len(row_indices) == 0:
            return mountain_map

        rows = np.arange(height, dtype=np.float32)[:, None]
        cols = np.arange(width, dtype=np.float32)[None, :]
        for row, col in zip(row_indices, col_indices):
            distance2 = (rows - row) ** 2 + (cols - col) ** 2
            mountain_map += np.exp(-distance2 / (2 * self.sigma ** 2))
        total = float(mountain_map.sum())
        if total > 0:
            mountain_map /= total
        return mountain_map


REPRESENTATIONS: Dict[str, RepresentationBuilder] = {
    BinaryMapBuilder.name: BinaryMapBuilder(),
    CountMapBuilder.name: CountMapBuilder(),
    DensityMapBuilder.name: DensityMapBuilder(),
    SoftMapBuilder.name: SoftMapBuilder(),
    ThreeValueMapBuilder.name: ThreeValueMapBuilder(),
    MountainMapBuilder.name: MountainMapBuilder(),
}


def map_klarf_to_grid(
    klarf_path: str | Path,
    shape: Tuple[int, int],
    mapper_name: str = DieIndexGridMapper.name,
    representation_name: str = DensityMapBuilder.name,
    defect_table_index: int = 0,
) -> GridMaps:
    # 故意把“坐标映射”和“表达形式”拆开。
    # 第一层负责把缺陷放进目标网格，第二层把同一份 count/status 转成不同 baseline 表达。
    tables = load_defect_tables(klarf_path)
    if defect_table_index < 0 or defect_table_index >= len(tables):
        raise IndexError(
            f"defect_table_index={defect_table_index} out of range; found {len(tables)} DefectList tables"
        )
    try:
        mapper = MAPPERS[mapper_name]
    except KeyError as exc:
        choices = ", ".join(sorted(MAPPERS))
        raise KeyError(f"Unknown mapper {mapper_name!r}. Available mappers: {choices}") from exc

    if representation_name not in REPRESENTATIONS:
        choices = ", ".join(sorted(REPRESENTATIONS))
        raise KeyError(f"Unknown representation {representation_name!r}. Available representations: {choices}")

    if mapper_name == PhysicalCoordinateGridMapper.name:
        die_pitch_x, die_pitch_y = load_die_pitch(klarf_path)
        mapper = PhysicalCoordinateGridMapper(die_pitch_x, die_pitch_y)

    mapped = mapper.map(tables[defect_table_index], shape)
    count_map = mapped["count_map"]
    status_map = mapped["status_map"]
    row_indices = mapped["row_indices"]
    col_indices = mapped["col_indices"]
    representation_maps = {
        name: builder.build(count_map, status_map, row_indices, col_indices)
        for name, builder in REPRESENTATIONS.items()
    }

    metadata = mapped["metadata"]
    metadata.update(
        {
            "klarf_path": str(klarf_path),
            "defect_table_index": defect_table_index,
            "defect_table_source": tables[defect_table_index].source,
            "defect_table_count": len(tables),
            "representation": representation_name,
        }
    )
    return GridMaps(
        count_map=representation_maps["count"],
        binary_map=representation_maps["binary"],
        density_map=representation_maps["density"],
        status_map=status_map,
        representation_map=representation_maps[representation_name],
        representation_maps=representation_maps,
        metadata=metadata,
    )


def save_grid_maps(path: str | Path, grid_maps: GridMaps) -> None:
    # 把所有 map 一起保存，后续相似度代码可以直接复用。
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


def _separable_convolution(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # 只用 NumPy 实现一个轻量级的高斯平滑。
    radius = len(kernel) // 2
    padded = np.pad(array, ((0, 0), (radius, radius)), mode="edge")
    horizontal = np.zeros_like(array, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        horizontal += weight * padded[:, offset : offset + array.shape[1]]

    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    vertical = np.zeros_like(array, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        vertical += weight * padded[offset : offset + array.shape[0], :]
    return vertical


def _neighbor_sum(mask: np.ndarray) -> np.ndarray:
    # 统计 8 邻域支持数，供三值图使用。
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    result = np.zeros(mask.shape, dtype=np.uint8)
    for row_offset in range(3):
        for col_offset in range(3):
            if row_offset == 1 and col_offset == 1:
                continue
            result += padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return result


def _scale_to_grid(values: Sequence[float], size: int, invert: bool = False) -> Tuple[np.ndarray, float, float]:
    # 把坐标线性缩放到 [0, size-1]。
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=np.int64), 0.0, 0.0

    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if max_value == min_value:
        scaled = np.zeros_like(values, dtype=float)
    else:
        scaled = (values - min_value) / (max_value - min_value)
    if invert:
        scaled = 1.0 - scaled
    indices = np.floor(scaled * size).astype(np.int64)
    indices = np.clip(indices, 0, size - 1)
    return indices, min_value, max_value
