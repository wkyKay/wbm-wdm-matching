# -*- coding: utf-8 -*-
"""
Spectral Clustering Cluster Proposal
Reference: Wang et al. — spectral clustering + fuzzy c-means

流程：
  缺陷点 → 相似度矩阵（空间+RBF） → 图拉普拉斯特征分解 → K-means on eigenvectors

关键优势：
  - 天然适合分离粘连的图案（通过图割视角）
  - 不需要预定义 cluster 数量（可用 eigengap 启发式自动确定）
  - 处理任意形状
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def _gaussian_similarity(points: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """构建空间 RBF 相似度矩阵"""
    N = len(points)
    W = np.zeros((N, N), dtype=np.float32)
    sigma2 = 2.0 * sigma ** 2

    for i in range(N):
        for j in range(i + 1, N):
            d2 = np.sum((points[i] - points[j]) ** 2)
            if d2 > (5 * sigma) ** 2:
                continue
            val = np.exp(-d2 / sigma2)
            W[i, j] = val
            W[j, i] = val
    return W


def _estimate_n_clusters(eigvals: np.ndarray, max_k: int = 8) -> int:
    """
    用 eigengap 启发式自动确定 cluster 数量。
    取相邻特征值之差最大的位置（跳过 λ₀=0）。
    """
    n = min(len(eigvals), max_k + 1)
    vals = eigvals[:n]

    # eigengap: λ_{k+1} - λ_k
    gaps = np.diff(vals)
    if len(gaps) == 0:
        return 2

    best_k = np.argmax(gaps) + 1  # +1 因为 diff 偏移

    # 至少 2 个 cluster
    return max(2, int(best_k))


def spectral_cluster_proposal(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sigma: float = 3.0,
    n_clusters: Optional[int] = None,
    auto_k: bool = True,
) -> List[Dict]:
    """
    基于 Spectral Clustering 的 cluster proposal。

    Args:
        defect_mask: (H, W) bool
        valid_mask:  (H, W) bool
        sigma:       RBF 核宽度
        n_clusters:  cluster 数量，None 则自动估计
        auto_k:      是否自动估计 k

    Returns:
        clusters 列表
    """
    H, W = defect_mask.shape
    if valid_mask is not None:
        mask = defect_mask & valid_mask
    else:
        mask = defect_mask

    points = np.argwhere(mask).astype(np.float32)
    N = len(points)
    if N < 3:
        return []

    # 1. 构建相似度矩阵
    W = _gaussian_similarity(points, sigma)

    # 2. 度矩阵和归一化拉普拉斯
    D_diag = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D_diag, 1e-10)))
    L_sym = np.eye(N, dtype=np.float32) - D_inv_sqrt @ W @ D_inv_sqrt

    # 3. 特征分解
    max_k = min(10, N)
    try:
        eigvals, eigvecs = np.linalg.eigh(L_sym)
    except np.linalg.LinAlgError:
        return []

    # 4. 自动确定 k
    if auto_k and n_clusters is None:
        n_clusters = _estimate_n_clusters(eigvals, max_k)
    elif n_clusters is None:
        n_clusters = 3

    n_clusters = min(n_clusters, N)

    # 5. 取前 k 个特征向量，用 K-means 聚类
    X = eigvecs[:, :n_clusters]  # (N, k)

    # 归一化行
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X = X / norms

    # K-means on eigenvectors
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # 分组
    clusters = []
    for lbl in range(n_clusters):
        mask_lbl = labels == lbl
        pts = points[mask_lbl]
        if len(pts) < 2:
            continue
        clusters.append(_compute_spectral_stats(pts, H, W, lbl))

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


def _compute_spectral_stats(points: np.ndarray, H: int, W: int, label: int) -> Dict:
    """统计信息"""
    area = len(points)
    return {
        'area': area,
        'centroid_row': float(points[:, 0].mean()),
        'centroid_col': float(points[:, 1].mean()),
        'centroid_row_norm': float(points[:, 0].mean() / H),
        'centroid_col_norm': float(points[:, 1].mean()) / W,
        'bbox_row_min': int(points[:, 0].min()),
        'bbox_col_min': int(points[:, 1].min()),
        'bbox_row_max': int(points[:, 0].max()),
        'bbox_col_max': int(points[:, 1].max()),
        'bbox_height': int(points[:, 0].max() - points[:, 0].min() + 1),
        'bbox_width': int(points[:, 1].max() - points[:, 1].min() + 1),
        'pixels': [(int(r), int(c)) for r, c in points],
        'pixel_coords': [(int(r), int(c)) for r, c in points],
    }
