# Data models and constants shared across the matching pipeline.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

BACKGROUND = 0
VALID_NO_DEFECT = 1
VALID_HAS_DEFECT = 2
UNINSPECTED = 3


@dataclass(frozen=True)
class DefectTable:
    """对一个 KLARF DefectList 表的轻量封装。"""

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
    """一个 DefectList 派生出的所有网格表达。"""

    count_map: np.ndarray
    binary_map: np.ndarray
    density_map: np.ndarray
    status_map: np.ndarray
    representation_map: np.ndarray
    representation_maps: Dict[str, np.ndarray]
    metadata: Dict[str, object]
