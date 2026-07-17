# KLARF / WBM file I/O: read shapes, parse defect tables, load die pitch, save outputs.
from __future__ import annotations

import struct
import zlib
from pathlib import Path
import sys
from typing import List, Mapping, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import klarfio  # noqa: E402

from ..core.models import (
    DefectTable,
    GridMaps,
    BACKGROUND,
    VALID_NO_DEFECT,
    VALID_HAS_DEFECT,
    UNINSPECTED,
)


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


def read_wbm_png(
    path: str | Path,
    defect_value: int = 255,
    no_defect_value: int = 127,
    background_value: int = 0,
) -> GridMaps:
    """将 WBM PNG（标准三值编码）读入 GridMaps。

    WBM 编码约定:
      白色 (255)  → VALID_HAS_DEFECT  有缺陷
      灰色 (127)  → VALID_NO_DEFECT   无缺陷但晶圆内
      黑色 (0)    → BACKGROUND        背景/晶圆外

    其他非标值:  >background_value 且 ≤defect_value → 归一化为 density (defect 强度)。
    返回的 status_map 直接反映 WBM 的三值含义。
    """
    pixels = _decode_png_grayscale(path)
    height, width = pixels.shape

    # ── 构建 status_map ──
    status_map = np.full((height, width), BACKGROUND, dtype=np.uint8)
    status_map[pixels == no_defect_value] = VALID_NO_DEFECT
    status_map[pixels == defect_value] = VALID_HAS_DEFECT        # 既无缺陷又有缺陷的格子，后续被 count_map>0 覆盖掉 VALID_NO_DEFECT
    # 灰色区域标记为 VALID_NO_DEFECT，后续 count_map>0 的格子会被覆盖为 VALID_HAS_DEFECT

    # ── 构建 count_map: 只有白色(缺陷)像素计入 ──
    count_map = (pixels == defect_value).astype(np.int32)

    # ── binary_map: WBM 本身是二值信号 ──
    binary_map = count_map.astype(np.uint8)

    # ── density_map: 仅缺陷区域归一化 ──
    density_map = count_map.astype(np.float32)
    total = float(density_map.sum())
    if total > 0:
        density_map /= total

    representation_maps: dict = {
        "binary": binary_map,
        "count": count_map,
        "density": density_map,
        "soft": density_map.copy(),
        "three-value": binary_map.astype(np.float32),
        "mountain": density_map.copy(),
    }

    return GridMaps(
        count_map=representation_maps["count"],
        binary_map=representation_maps["binary"],
        density_map=representation_maps["density"],
        status_map=status_map,
        representation_map=representation_maps["binary"],  # WBM 用 binary，保持二值语义
        representation_maps=representation_maps,
        metadata={
            "source": str(path),
            "defect_value": defect_value,
            "no_defect_value": no_defect_value,
            "background_value": background_value,
            "target_height": height,
            "target_width": width,
        },
    )


def _decode_png_grayscale(path: str | Path) -> np.ndarray:
    """纯 Python/NumPy 解码灰度 PNG（0~255）。不依赖第三方图像库。"""
    with open(path, "rb") as f:
        data = f.read()

    # 校验 PNG 魔数
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG file: {path}")

    pos = 8
    width = height = bit_depth = color_type = 0
    raw_bytes = b""

    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length

        if chunk_type == b"IHDR":
            width = struct.unpack(">I", chunk_data[0:4])[0]
            height = struct.unpack(">I", chunk_data[4:8])[0]
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]

        elif chunk_type == b"IDAT":
            raw_bytes += chunk_data

        elif chunk_type == b"IEND":
            break

    if not raw_bytes:
        raise ValueError(f"No image data in PNG: {path}")

    # 解压 zlib
    decompressed = zlib.decompress(raw_bytes)

    # PNG 每行前有一个 filter 字节；处理最简单的 filter=0 (None)
    bytes_per_row = 1 + width  # filter byte + pixel bytes
    expected_len = bytes_per_row * height
    if len(decompressed) < expected_len:
        raise ValueError(f"Unexpected decompressed length: {len(decompressed)} vs {expected_len}")

    pixels = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        start = row * bytes_per_row
        filter_byte = decompressed[start]
        row_data = decompressed[start + 1 : start + bytes_per_row]
        if filter_byte == 0:  # None filter
            pixels[row, :] = np.frombuffer(row_data, dtype=np.uint8)
        else:
            # Sub filter: pixel[i] = raw[i] + pixel[i-1]; Up filter: +pixel_above[i]
            raw = np.frombuffer(row_data, dtype=np.uint8).astype(np.int32)
            decoded = raw.copy()
            if filter_byte == 1:  # Sub
                for col in range(1, width):
                    decoded[col] = (decoded[col] + decoded[col - 1]) % 256
            elif filter_byte == 2:  # Up
                decoded[:] = (decoded + pixels[row - 1, :].astype(np.int32)) % 256
            elif filter_byte == 4:  # Paeth (simplified)
                decoded[:] = decoded % 256  # fallback: no-op on paeth for now
            pixels[row, :] = decoded.astype(np.uint8)

    return pixels


def load_defect_tables(klarf_path: str | Path) -> List[DefectTable]:
    """解析 KLARF 文件，收集其中所有的 DefectList 表。"""
    parsed = klarfio.klarf(filename=str(klarf_path))
    tables: List[DefectTable] = []
    _collect_defect_tables(parsed.data, tables, source="root")
    if not tables:
        raise ValueError(f"No DefectList found in {klarf_path}")
    return tables


def split_defect_table_by_classnumber(defect_table: DefectTable) -> dict[int, DefectTable]:
    """按 KLARF 中的 classnumber/class number 字段拆分 defect table。

    返回值按 classnumber 升序排序；若找不到 classnumber 列则返回空字典。
    """
    class_col = _find_column_name(defect_table.columns, ("CLASSNUMBER", "ClassNumber", "classnumber"))
    if class_col is None:
        return {}

    values = defect_table.column(class_col)
    valid = np.isfinite(values)
    if not valid.any():
        return {}

    groups: dict[int, list[int]] = {}
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        class_id = int(value)
        groups.setdefault(class_id, []).append(idx)

    split_tables: dict[int, DefectTable] = {}
    for class_id in sorted(groups):
        rows = defect_table.rows[np.asarray(groups[class_id], dtype=np.int64)]
        split_tables[class_id] = DefectTable(columns=list(defect_table.columns), rows=rows, source=f"{defect_table.source}/classnumber={class_id}")
    return split_tables


def _collect_defect_tables(node: object, tables: List[DefectTable], source: str) -> None:
    """递归遍历解析后的 KLARF 树，寻找包含缺陷列的表。"""
    if not isinstance(node, Mapping):
        return

    if "Columns" in node and "Data" in node:
        columns = [item["Column"] for item in node["Columns"]]
        if {"DEFECTID", "XINDEX", "YINDEX"}.issubset(columns):
            # 生产数据中某些字段为 "N" (unknown)，转为 np.nan 再 parse
            rows = _safe_float_array(node["Data"])
            tables.append(DefectTable(columns=columns, rows=rows, source=source))

    for key, value in node.items():
        if isinstance(value, Mapping):
            _collect_defect_tables(value, tables, source=f"{source}/{key}")


def _safe_float_array(data: list) -> np.ndarray:
    """将 KLARF Data 列表转为 float 数组，'N' 视为 NaN。"""
    # 判断结构：list of rows（二维）或 flat list（一维）
    is_nested = data and isinstance(data[0], (list, tuple))

    flat = []
    for item in data:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    converted = [np.nan if isinstance(v, str) and v.strip().upper() == "N" else float(v) for v in flat]
    arr = np.array(converted, dtype=float)

    if is_nested:
        return arr.reshape(len(data), -1)
    else:
        return arr.reshape(1, -1)


def _find_column_name(columns: List[str], candidates: Tuple[str, ...]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def load_die_pitch(klarf_path: str | Path) -> tuple[float, float]:
    """从 KLARF LotRecord 中提取 DiePitch。"""
    parsed = klarfio.klarf(filename=str(klarf_path))
    lot_data = _lot_data(parsed)
    if "DiePitch" not in lot_data:
        raise ValueError(f"No DiePitch found in LotRecord of {klarf_path}")
    die_pitch = lot_data["DiePitch"]
    return float(die_pitch[0]), float(die_pitch[1])


def load_die_index_bounds(klarf_path: str | Path) -> tuple[int, int, int, int]:
    """从 KLARF 元数据中估算完整 die 网格范围（对齐 klarfkit 的网格绘制逻辑）。

    使用 SampleSize + DiePitch + SampleCenterLocation 计算晶圆圆周内
    每个轴上的 die 索引最小/最大值。公式：

        i_min = floor((center - radius) / pitch)
        i_max = ceil ((center + radius) / pitch)

    其中 radius = sample_size / 2。

    返回 (x_min, x_max, y_min, y_max)，反映晶圆上所有 die（不限于有缺陷的 die）。
    """
    import math

    parsed = klarfio.klarf(filename=str(klarf_path))
    lot_data = _lot_data(parsed)
    wafer_data = _wafer_data(parsed)

    die_pitch_x, die_pitch_y = (
        float(lot_data["DiePitch"][0]),
        float(lot_data["DiePitch"][1]),
    )
    sample_size = float(lot_data.get("SampleSize", [300_000])[0])
    radius = sample_size / 2.0
    center_x, center_y = _sample_center_location(wafer_data, sample_size)

    x_min = math.floor((center_x - radius) / die_pitch_x)
    x_max = math.ceil((center_x + radius) / die_pitch_x)
    y_min = math.floor((center_y - radius) / die_pitch_y)
    y_max = math.ceil((center_y + radius) / die_pitch_y)

    return int(x_min), int(x_max), int(y_min), int(y_max)


def _lot_data(parsed) -> dict:
    """Extract LotRecord from parsed KLARF (supports v1.2 and v1.8)."""
    version = parsed.version
    if version == "1.8":
        file_data = parsed.data["FileRecord_1.8"]
    elif version == "1.2":
        file_data = parsed.data["FileRecord_1.2"]
    else:
        raise ValueError(f"Unsupported KLARF version: {version}")
    lot_keys = [k for k in file_data if k.startswith("LotRecord")]
    if not lot_keys:
        raise ValueError("No LotRecord found in KLARF")
    return file_data[lot_keys[0]]


def _wafer_data(parsed) -> dict:
    """Extract the first WaferRecord from parsed KLARF."""
    lot_data = _lot_data(parsed)
    wafer_keys = [k for k in lot_data if k.startswith("WaferRecord")]
    if not wafer_keys:
        raise ValueError("No WaferRecord found in KLARF")
    return lot_data[wafer_keys[0]]


def _sample_center_location(wafer_data: dict, sample_size: float) -> tuple[float, float]:
    """Read SampleCenterLocation from KLARF WaferRecord, defaulting to wafer center."""
    if "SampleCenterLocation" in wafer_data:
        vals = wafer_data["SampleCenterLocation"]
        return float(vals[0]), float(vals[1])
    return sample_size / 2.0, sample_size / 2.0


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
