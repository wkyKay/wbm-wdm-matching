# -*- coding: utf-8 -*-
"""
晶圆图预处理模块
"""

import numpy as np
from typing import Dict, Tuple


# 状态图的取值定义
STATUS_BACKGROUND = 0
STATUS_VALID_NO_DEFECT = 1
STATUS_VALID_HAS_DEFECT = 2
STATUS_UNINSPECTED = 3


def preprocess_single_map(raw_map: np.ndarray) -> Dict[str, np.ndarray]:
    """
    预处理单张晶圆图。

    Args:
        raw_map: (H, W) 原始晶圆图，取值 0/1/2/3

    Returns:
        包含各种预处理结果的字典
    """
    # 1. 生成状态图
    status_map = raw_map.copy()

    # 2. 生成各种 mask
    valid_mask = (raw_map == STATUS_VALID_NO_DEFECT) | (raw_map == STATUS_VALID_HAS_DEFECT)
    defect_mask = (raw_map == STATUS_VALID_HAS_DEFECT)
    ignored_mask = (raw_map == STATUS_UNINSPECTED)

    # 3. 生成基础 map
    binary_map = defect_mask.astype(np.uint8)
    count_map = binary_map.copy()

    # 4. 密度图
    defect_area = defect_mask.sum()
    if defect_area > 0:
        density_map = defect_mask.astype(np.float32) / defect_area
    else:
        density_map = np.zeros_like(defect_mask, dtype=np.float32)

    # 5. 高斯平滑的 soft map
    soft_map = gaussian_filter_2d(density_map, sigma=1.0)

    # 6. 三值图
    three_value_map = create_three_value_map(defect_mask)

    return {
        'status_map': status_map,
        'valid_mask': valid_mask,
        'defect_mask': defect_mask,
        'ignored_mask': ignored_mask,
        'binary_map': binary_map,
        'count_map': count_map,
        'density_map': density_map,
        'soft_map': soft_map,
        'three_value_map': three_value_map,
    }


def preprocess_batch(maps: np.ndarray, skip_soft_map: bool = True) -> Dict[str, np.ndarray]:
    """
    批量预处理晶圆图。

    Args:
        maps: (N, H, W) 原始晶圆图数组
        skip_soft_map: Whether to skip computing soft_map to speed up processing

    Returns:
        包含各种预处理结果的字典，每个值是 (N, H, W) 的数组
    """
    N, H, W = maps.shape

    # 初始化输出数组 - use vectorized operations where possible
    status_maps = maps.astype(np.uint8)
    
    # Vectorized mask creation
    valid_masks = (maps == STATUS_VALID_NO_DEFECT) | (maps == STATUS_VALID_HAS_DEFECT)
    defect_masks = (maps == STATUS_VALID_HAS_DEFECT)
    ignored_masks = (maps == STATUS_UNINSPECTED)
    
    binary_maps = defect_masks.astype(np.uint8)
    count_maps = binary_maps.copy()
    
    # Density maps - vectorized
    defect_areas = defect_masks.sum(axis=(1, 2), keepdims=True)
    defect_areas_safe = np.maximum(defect_areas, 1)
    density_maps = defect_masks.astype(np.float32) / defect_areas_safe
    
    # Soft maps - skip for speed if requested
    if skip_soft_map:
        soft_maps = density_maps.copy()
    else:
        soft_maps = np.zeros_like(density_maps, dtype=np.float32)
        for i in range(N):
            soft_maps[i] = gaussian_filter_2d(density_maps[i], sigma=1.0)
    
    # Three value maps - vectorized
    three_value_maps = create_three_value_map_batch(defect_masks)

    return {
        'status_maps': status_maps,
        'valid_masks': valid_masks,
        'defect_masks': defect_masks,
        'ignored_masks': ignored_masks,
        'binary_maps': binary_maps,
        'count_maps': count_maps,
        'density_maps': density_maps,
        'soft_maps': soft_maps,
        'three_value_maps': three_value_maps,
    }


def gaussian_filter_2d(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    二维高斯滤波（简化实现，避免依赖 scipy）。

    Args:
        image: (H, W) 输入图像
        sigma: 高斯核标准差

    Returns:
        滤波后的图像
    """
    H, W = image.shape
    output = np.zeros_like(image, dtype=np.float32)

    # 计算核大小
    kernel_size = int(max(3, 2 * int(3 * sigma) + 1))
    kernel_radius = kernel_size // 2

    # 生成高斯核
    x = np.arange(-kernel_radius, kernel_radius + 1)
    y = np.arange(-kernel_radius, kernel_radius + 1)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # 填充边界
    padded = np.pad(image, kernel_radius, mode='constant')

    # 卷积
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = (region * kernel).sum()

    return output


def create_three_value_map(defect_mask: np.ndarray) -> np.ndarray:
    """
    生成三值图。

    规则：
    - 1.0: 缺陷像素
    - 0.5: 缺陷像素的 8-邻域非缺陷像素
    - 0.0: 其他

    Args:
        defect_mask: (H, W) 缺陷 mask

    Returns:
        (H, W) 三值图
    """
    return create_three_value_map_batch(defect_mask[np.newaxis, ...])[0]


def create_three_value_map_batch(defect_masks: np.ndarray) -> np.ndarray:
    """
    批量生成三值图（向量化版本）。

    Args:
        defect_masks: (N, H, W) 缺陷 masks

    Returns:
        (N, H, W) 三值图
    """
    N, H, W = defect_masks.shape
    three_value = np.zeros((N, H, W), dtype=np.float32)
    
    # 1. 缺陷像素设为 1.0
    three_value[defect_masks] = 1.0
    
    # 2. 快速计算 8-邻域
    padded = np.pad(defect_masks, ((0, 0), (1, 1), (1, 1)), mode='constant')
    
    # Use sliding window approach - check if any neighbor is defective
    # For efficiency, use convolution-like approach with max pooling
    from functools import reduce
    import operator
    
    # Create neighbor masks
    neighbor_masks = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbor_mask = padded[:, 1+di:H+1+di, 1+dj:W+1+dj]
            neighbor_masks.append(neighbor_mask)
    
    # Combine all neighbor masks
    has_defect_neighbor = reduce(operator.or_, neighbor_masks)
    
    # Set 0.5 where there's a defect neighbor and it's not a defect itself
    three_value[has_defect_neighbor & (~defect_masks)] = 0.5
    
    return three_value


def compute_map_statistics(
    defect_mask: np.ndarray,
    valid_mask: np.ndarray = None
) -> Dict:
    """
    计算单张图的统计信息。

    Args:
        defect_mask: (H, W) 缺陷 mask
        valid_mask: (H, W) 有效区域 mask，可选

    Returns:
        统计信息字典
    """
    H, W = defect_mask.shape

    valid_area = valid_mask.sum() if valid_mask is not None else H * W
    defect_area = defect_mask.sum()
    defect_ratio = defect_area / max(valid_area, 1)

    # 缺陷 bbox
    defect_coords = np.argwhere(defect_mask)
    if len(defect_coords) > 0:
        row_min, col_min = defect_coords.min(axis=0)
        row_max, col_max = defect_coords.max(axis=0)
        centroid_row, centroid_col = defect_coords.mean(axis=0)
    else:
        row_min = col_min = row_max = col_max = 0
        centroid_row = H / 2
        centroid_col = W / 2

    # 归一化质心
    centroid_row_norm = centroid_row / H
    centroid_col_norm = centroid_col / W

    return {
        'valid_area': int(valid_area),
        'defect_area': int(defect_area),
        'defect_ratio': float(defect_ratio),
        'defect_bbox_row_min': int(row_min),
        'defect_bbox_col_min': int(col_min),
        'defect_bbox_row_max': int(row_max),
        'defect_bbox_col_max': int(col_max),
        'defect_centroid_row': float(centroid_row),
        'defect_centroid_col': float(centroid_col),
        'defect_centroid_row_norm': float(centroid_row_norm),
        'defect_centroid_col_norm': float(centroid_col_norm),
    }


# ============================================================
# Adjacency-Clustering 清洗（Ezzat et al. 2020）
# ============================================================

def ac_clean_mask(
    defect_mask: np.ndarray,
    min_degree: int = 2,
    max_rounds: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjacency-Clustering 空间清洗。

    基于图论的最小割思想：通过迭代移除 8-邻域中邻接缺陷数不足
    的孤立噪声点，保留成群结队的系统性缺陷。

    原理：
    - 构建缺陷芯片的 8-邻接图
    - 计算每个芯片的度（相邻缺陷芯片数）
    - 度 < min_degree 的芯片视为噪声并移除
    - 迭代至收敛（无更多噪声可移除）

    Args:
        defect_mask: (H, W) bool，True = 缺陷点
        min_degree:  保留所需的最小邻接缺陷数（默认 2）
        max_rounds:  最大迭代轮数（默认 3）

    Returns:
        (cleaned_mask, removed_mask)
        - cleaned_mask: (H, W) bool，清洗后的缺陷 mask
        - removed_mask: (H, W) bool，被移除的噪声点
    """
    H, W = defect_mask.shape
    current = defect_mask.copy()
    neighbors8 = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for _round in range(max_rounds):
        pts = np.argwhere(current)
        if len(pts) == 0:
            break

        # 构建坐标 → 下标映射
        pt_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(pts)}

        # 计算每个点的度
        degree = np.zeros(len(pts), dtype=int)
        for (r, c), idx in pt_to_idx.items():
            for dr, dc in neighbors8:
                if (r + dr, c + dc) in pt_to_idx:
                    degree[idx] += 1

        noise = degree < min_degree
        if noise.sum() == 0:
            break  # 收敛

        for r, c in pts[noise]:
            current[int(r), int(c)] = False

    removed = defect_mask & (~current)
    return current, removed


def ac_clean_batch(
    defect_masks: np.ndarray,
    valid_masks: np.ndarray = None,
    min_degree: int = 2,
    max_rounds: int = 3,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    批量 Adjacency-Clustering 清洗。

    Args:
        defect_masks: (N, H, W) bool，缺陷 masks
        valid_masks:  (N, H, W) bool，有效区域 masks（可选）
        min_degree:   保留所需的最小邻接缺陷数
        max_rounds:   最大迭代轮数

    Returns:
        (cleaned_masks, removed_masks, stats)
        - cleaned_masks: (N, H, W) bool
        - removed_masks: (N, H, W) bool
        - stats: List[Dict] 每张图的清洗统计 {n_before, n_after, n_removed, ratio}
    """
    N = len(defect_masks)
    cleaned = np.zeros_like(defect_masks, dtype=bool)
    removed = np.zeros_like(defect_masks, dtype=bool)
    stats = []

    for i in range(N):
        mask = defect_masks[i]
        if valid_masks is not None:
            mask = mask & valid_masks[i]

        n_before = int(mask.sum())
        clean, rm = ac_clean_mask(mask, min_degree, max_rounds)
        n_after = int(clean.sum())

        cleaned[i] = clean
        removed[i] = rm
        stats.append({
            'sample_id': i,
            'n_before': n_before,
            'n_after': n_after,
            'n_removed': n_before - n_after,
            'removal_ratio': (n_before - n_after) / max(n_before, 1),
        })

    return cleaned, removed, stats
