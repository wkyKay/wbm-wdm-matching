# -*- coding: utf-8 -*-
"""
Adjacency-Clustering + iWMM (Infinite Warped Mixture Model) Cluster Proposal
Reference: Ezzat, Liu, Hochbaum, Ding (2020)

两阶段流程：
  Stage 1: Adjacency-Clustering（图论空间过滤）
    - 构建缺陷芯片的邻接图（8-connected）
    - 利用空间依赖过滤随机噪声
    - 只保留系统性缺陷芯片
  Stage 2: iWMM / DP-GMM（Dirichlet Process GMM）
    - 在过滤后的缺陷芯片上用 DP-GMM 自动分离混合图案
    - 不需要预定义 cluster 数量
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


# ============================================================
# Stage 1: Adjacency-Clustering — 图论空间过滤
# ============================================================

def _build_adjacency_graph(defect_mask: np.ndarray) -> Dict:
    """
    构建缺陷芯片的 8-连通邻接图。
    返回每个节点的邻居列表。
    """
    H, W = defect_mask.shape
    points = np.argwhere(defect_mask)
    point_to_idx = {(r, c): i for i, (r, c) in enumerate(points)}

    adj_list = {i: [] for i in range(len(points))}
    neighbors_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for (r, c), idx in point_to_idx.items():
        for dr, dc in neighbors_8:
            nr, nc = r + dr, c + dc
            if (nr, nc) in point_to_idx:
                adj_list[idx].append(point_to_idx[(nr, nc)])

    return {
        'points': points,
        'adj_list': adj_list,
        'n_nodes': len(points),
    }


def adjacency_filter(defect_mask: np.ndarray,
                     min_degree: int = 2,
                     max_rounds: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    论文 Stage 1: Adjacency-Clustering 空间过滤。
    反复移除邻接度 < min_degree 的孤立点及其邻居，
    直到收敛或达到 max_rounds。

    Args:
        defect_mask: (H, W) bool
        min_degree:  保留点最少需要的邻接缺陷数
        max_rounds:  最大迭代轮数

    Returns:
        filtered_mask: (H, W) bool 过滤后的 mask
        removed_mask:  (H, W) bool 被移除的噪声点
    """
    H, W = defect_mask.shape
    current = defect_mask.copy()

    for _ in range(max_rounds):
        graph = _build_adjacency_graph(current)
        if graph['n_nodes'] == 0:
            break

        degrees = np.array([len(nbrs) for nbrs in graph['adj_list'].values()])
        noise_mask = degrees < min_degree

        if noise_mask.sum() == 0:
            break  # 收敛

        # 移除低度节点
        noise_points = graph['points'][noise_mask]
        for r, c in noise_points:
            current[int(r), int(c)] = False

    removed = defect_mask & (~current)
    return current, removed


# ============================================================
# Stage 2: Dirichlet Process GMM（iWMM 的实用近似）
# ============================================================

from sklearn.mixture import BayesianGaussianMixture


def dp_gmm_clustering(points: np.ndarray,
                      max_components: int = 10,
                      weight_concentration_prior: float = 0.1) -> Tuple[np.ndarray, int]:
    """
    用 Dirichlet Process GMM (BayesianGaussianMixture) 做聚类。
    这是 iWMM 的实用近似：DP 先验让模型自动决定有效成分数。

    Args:
        points: (N, 2) 缺陷点坐标
        max_components: 最大成分数（上限）
        weight_concentration_prior: DP 浓度参数，越小越倾向于少数成分

    Returns:
        labels: (N,) cluster 标签，-1 为噪声
        n_components: 实际有效成分数
    """
    N = len(points)
    if N < 2:
        return np.zeros(N, dtype=int), 1

    n_comp = min(max_components, N)

    bgm = BayesianGaussianMixture(
        n_components=n_comp,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=weight_concentration_prior,
        covariance_type='full',
        max_iter=200,
        random_state=42,
    )
    bgm.fit(points)
    labels = bgm.predict(points)

    # 统计有效成分（权重 > 1e-3）
    weights = bgm.weights_
    active = np.where(weights > 1e-3)[0]
    n_active = len(active)

    return labels, n_active


def adjacency_iwmm_cluster_proposal(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_degree: int = 2,
    max_rounds: int = 3,
    max_components: int = 8,
    weight_concentration_prior: float = 0.15,
) -> List[Dict]:
    """
    完整的 Adjacency-Clustering + iWMM cluster proposal。

    Args:
        defect_mask: (H, W) bool
        valid_mask:  (H, W) bool
        min_degree:  AC 过滤的最小邻接度
        max_rounds:  AC 最大迭代轮数
        max_components: DP-GMM 最大成分数
        weight_concentration_prior: DP 浓度参数

    Returns:
        clusters 列表
    """
    H, W = defect_mask.shape
    if valid_mask is not None:
        mask = defect_mask & valid_mask
    else:
        mask = defect_mask

    # Stage 1: Adjacency-Clustering 过滤
    filtered_mask, removed_mask = adjacency_filter(mask, min_degree, max_rounds)

    points = np.argwhere(filtered_mask).astype(np.float32)
    if len(points) < 2:
        return []

    # Stage 2: DP-GMM 聚类
    labels, n_active = dp_gmm_clustering(
        points,
        max_components=max_components,
        weight_concentration_prior=weight_concentration_prior,
    )

    # 按标签分组
    clusters = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask_lbl = labels == label
        pts = points[mask_lbl]
        if len(pts) < 2:
            continue
        cluster = _compute_iwmm_stats(pts, H, W, int(label),
                                       int(removed_mask.sum()))
        clusters.append(cluster)

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


def _compute_iwmm_stats(points: np.ndarray, H: int, W: int,
                        label: int, n_removed: int) -> Dict:
    """统计 + AC 过滤信息"""
    stats = {
        'area': len(points),
        'centroid_row': float(points[:, 0].mean()),
        'centroid_col': float(points[:, 1].mean()),
        'centroid_row_norm': float(points[:, 0].mean() / H),
        'centroid_col_norm': float(points[:, 1].mean() / W),
        'bbox_row_min': int(points[:, 0].min()),
        'bbox_col_min': int(points[:, 1].min()),
        'bbox_row_max': int(points[:, 0].max()),
        'bbox_col_max': int(points[:, 1].max()),
        'n_removed_noise': n_removed,
        'pixels': [(int(r), int(c)) for r, c in points],
        'pixel_coords': [(int(r), int(c)) for r, c in points],
    }
    stats['bbox_height'] = stats['bbox_row_max'] - stats['bbox_row_min'] + 1
    stats['bbox_width'] = stats['bbox_col_max'] - stats['bbox_col_min'] + 1
    return stats
