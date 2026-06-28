# -*- coding: utf-8 -*-
"""
DBSCAN Cluster Proposal
Reference: Koo & Hwang (2021), Jin et al. (2019)

流程：
  缺陷点 → k-distance 自动估计 eps → DBSCAN → 噪声过滤 → 连通域聚类
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def estimate_eps(points: np.ndarray, min_samples: int = 5) -> float:
    """
    用 k-distance 图自动估计 DBSCAN 的 eps 参数。
    取 k-distance 排序后拐点处的距离。
    """
    if len(points) < min_samples + 1:
        # 点太少，用固定值
        return 3.0

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(points)
    distances, _ = nbrs.kneighbors(points)
    k_distances = np.sort(distances[:, -1])  # 第 k 近邻距离，升序

    # 拐点：最大曲率处
    n = len(k_distances)
    idx = np.arange(n)
    # 计算二阶差分找拐点
    d2 = np.gradient(np.gradient(k_distances))
    elbow = np.argmax(d2[len(d2)//4:]) + len(d2)//4  # 忽略开头噪声
    eps = k_distances[elbow]

    return max(eps, 0.5)


def dbscan_cluster_proposal(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: Optional[float] = None,
    min_samples: int = 5,
    auto_eps: bool = True,
) -> List[Dict]:
    """
    基于 DBSCAN 的 cluster proposal。

    Args:
        defect_mask: (H, W) bool 缺陷 mask
        valid_mask:  (H, W) bool 有效区域
        eps:         邻域半径，None 则自动估计
        min_samples: 核心点最少邻居数
        auto_eps:    是否自动估计 eps

    Returns:
        clusters 列表
    """
    H, W = defect_mask.shape
    if valid_mask is not None:
        mask = defect_mask & valid_mask
    else:
        mask = defect_mask

    points = np.argwhere(mask).astype(np.float32)  # (N, 2)
    if len(points) < min_samples:
        return []

    # 自动估计 eps
    if auto_eps and eps is None:
        eps = estimate_eps(points, min_samples)
    elif eps is None:
        eps = 2.0  # 默认值

    # DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # 分簇
    clusters = []
    unique_labels = set(labels)
    n_noise = int((labels == -1).sum())

    for label in unique_labels:
        if label == -1:
            continue  # 跳过噪声
        pts = points[labels == label]
        clusters.append(_compute_stats(pts, H, W, label))

    # 按面积降序排列
    clusters.sort(key=lambda c: c['area'], reverse=True)

    return clusters


def _compute_stats(points: np.ndarray, H: int, W: int, label: int) -> Dict:
    """计算 cluster 统计信息"""
    rows = points[:, 0]
    cols = points[:, 1]
    area = len(points)

    centroid_row = float(rows.mean())
    centroid_col = float(cols.mean())

    bbox_row_min = int(rows.min())
    bbox_col_min = int(cols.min())
    bbox_row_max = int(rows.max())
    bbox_col_max = int(cols.max())

    # PCA
    pca_lambda1 = pca_lambda2 = 0.0
    orientation = 0.0
    if area > 1:
        cov_rr = float(np.mean((rows - centroid_row) ** 2))
        cov_cc = float(np.mean((cols - centroid_col) ** 2))
        cov_rc = float(np.mean((rows - centroid_row) * (cols - centroid_col)))
        cov = np.array([[cov_rr, cov_rc], [cov_rc, cov_cc]])
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_lambda1 = float(eigvals[1])
        pca_lambda2 = float(eigvals[0])
        if eigvals[1] > 1e-10:
            orientation = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))

    # Perimeter
    pixel_set = set((int(r), int(c)) for r, c in points)
    perimeter = 0
    for x, y in pixel_set:
        if (x-1, y) not in pixel_set or x-1 < 0:
            perimeter += 1
        if (x+1, y) not in pixel_set or x+1 >= H:
            perimeter += 1
        if (x, y-1) not in pixel_set or y-1 < 0:
            perimeter += 1
        if (x, y+1) not in pixel_set or y+1 >= W:
            perimeter += 1

    radial_dist = np.sqrt((centroid_row - H/2)**2 + (centroid_col - W/2)**2)
    radial_dist_norm = float(radial_dist / max(np.sqrt((H/2)**2 + (W/2)**2), 1))

    return {
        'area': area,
        'centroid_row': centroid_row,
        'centroid_col': centroid_col,
        'centroid_row_norm': float(centroid_row / H),
        'centroid_col_norm': float(centroid_col / W),
        'bbox_row_min': bbox_row_min,
        'bbox_col_min': bbox_col_min,
        'bbox_row_max': bbox_row_max,
        'bbox_col_max': bbox_col_max,
        'bbox_height': bbox_row_max - bbox_row_min + 1,
        'bbox_width': bbox_col_max - bbox_col_min + 1,
        'pca_lambda1': pca_lambda1,
        'pca_lambda2': pca_lambda2,
        'orientation': orientation,
        'perimeter': perimeter,
        'compactness': perimeter / max(area, 1),
        'radial_distance_norm': radial_dist_norm,
        'pixels': [(int(r), int(c)) for r, c in points],
        'pixel_coords': [(int(r), int(c)) for r, c in points],
    }
