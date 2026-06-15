# Grid map representations: binary, count, density, soft, three-value, mountain.
from __future__ import annotations

from typing import Dict

import numpy as np


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
        return (count_map > 0).astype(np.uint8)


class CountMapBuilder(RepresentationBuilder):
    name = "count"

    def build(self, count_map, status_map, row_indices, col_indices):
        return count_map.astype(np.int32)


class DensityMapBuilder(RepresentationBuilder):
    name = "density"

    def build(self, count_map, status_map, row_indices, col_indices):
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
        kernel = np.exp(-(offsets**2) / (2 * sigma**2))
        self.kernel = kernel / kernel.sum()

    def build(self, count_map, status_map, row_indices, col_indices):
        density_map = DensityMapBuilder().build(count_map, status_map, row_indices, col_indices)
        return _separable_convolution(density_map, self.kernel).astype(np.float32)


class ThreeValueMapBuilder(RepresentationBuilder):
    name = "three-value"

    def __init__(self, strong_threshold: int = 2):
        self.strong_threshold = strong_threshold

    def build(self, count_map, status_map, row_indices, col_indices):
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
        height, width = count_map.shape
        mountain_map = np.zeros((height, width), dtype=np.float32)
        if len(row_indices) == 0:
            return mountain_map

        rows = np.arange(height, dtype=np.float32)[:, None]
        cols = np.arange(width, dtype=np.float32)[None, :]
        for row, col in zip(row_indices, col_indices):
            distance2 = (rows - row) ** 2 + (cols - col) ** 2
            mountain_map += np.exp(-distance2 / (2 * self.sigma**2))
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


def _separable_convolution(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """只用 NumPy 实现一个轻量级的高斯平滑。"""
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
    """统计 8 邻域支持数，供三值图使用。"""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    result = np.zeros(mask.shape, dtype=np.uint8)
    for row_offset in range(3):
        for col_offset in range(3):
            if row_offset == 1 and col_offset == 1:
                continue
            result += padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return result
