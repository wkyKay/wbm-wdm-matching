# -*- coding: utf-8 -*-
"""
统一聚类模块 — 所有 Cluster Proposal 方法共享同一接口。

调用方式:
    from partial_match.core.clustering import cluster

    clusters = cluster(defect_mask, valid_mask, method='dbscan', eps=2.0)

支持的方法:
  'raw'            8-连通域（原始）
  'filtered'       8-连通域 + 面积过滤
  'adhesion'       8-连通域 + 可疑粘连区域二次拆分
  'dilated_group'  膨胀仅用于 grouping，最终 token 使用原始像素
  'dilated_adhesion' 先 dilated grouping，再拆分可疑 group
  'group_then_adhesion' filtered -> dilated grouping -> selective adhesion split
  'geometry_merge' filtered -> adhesion -> component-level geometry merge
  'topk'           基于候选 proposal 选择面积最大的 K 个区域
  'closing'        形态学闭运算后连通域
  'simi_paper'     SIMI Paper: Closing + Spatial Filter → 强缺陷聚类
  'dbscan'         DBSCAN（Koo & Hwang 2021）
  'adjacency_iwmm' Adjacency-Clustering + iWMM（Ezzat et al. 2020）
  'spectral'       Spectral Clustering（Wang et al.）
  'tensor_voting'  Tensor Voting → 噪声过滤（Wang et al. 2022）

所有方法返回: List[Dict]，每个 cluster 含 area, centroid, bbox, pixels 等。
"""

import numpy as np
from typing import Dict, List, Optional


# ============================================================
# 主入口
# ============================================================

def cluster(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    method: str = 'raw',
    use_clean: bool = False,
    clean_min_degree: int = 2,
    clean_max_rounds: int = 3,
    **kwargs
) -> List[Dict]:
    """
    统一的 cluster proposal 接口。

    Args:
        defect_mask: (H, W) bool 缺陷点 mask
        valid_mask:  (H, W) bool 有效区域 mask（可选）
        method:      聚类方法名
        use_clean:   是否先做 AC 空间清洗再聚类（默认 False）
        clean_min_degree: AC 清洗的最小邻接度（默认 2）
        clean_max_rounds: AC 清洗的最大迭代轮数（默认 3）
        **kwargs:    传递给各方法的参数

    Returns:
        List[Dict]: 每个 cluster 含 area, centroid, bbox, pixels, ...

    Methods & Kwargs:
      raw             — 无额外参数
      filtered        — min_area: int (default 3)
      adhesion        — min_area: int (3), split_method: str ('tv_hybrid'),
                        suspicious_area: int (40), min_split_area: int (3),
                        min_suspicious_cues: int (2), max_split_count: int (6)
      dilated_group   — min_area: int (3), dilation_radius: int (1),
                        use_closing: bool (False), structure: str ('cross')
      dilated_adhesion — same as dilated_group, plus split suspicious groups
      group_then_adhesion — min_area filtering before dilation, dilation_radius: int (1),
                        selective adhesion split, skip_ring_like: bool (True)
      geometry_merge  — component-level merge for truncated ring/line/blob fragments
      topk            — top_k: int (5), base_method: str ('geometry_merge')
      closing         — 无额外参数
      simi_paper      — 无额外参数
      dbscan          — eps: float | None, min_samples: int (5), auto_eps: bool (True)
      adjacency_iwmm  — 已内含 AC 清洗，use_clean 跳过内部 Stage1
      spectral        — sigma: float (3.0), n_clusters: int | None, auto_k: bool (True)
      tensor_voting   — sigma: float (5.0), noise_ratio: float (0.3)
      tv              — alias for tensor_voting
    """
    method = method.lower().strip()

    # ── 应用 AC 清洗 ──
    n_removed_noise = 0
    if use_clean:
        from partial_match.data.preprocessing import ac_clean_mask
        mask = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()
        cleaned, _ = ac_clean_mask(mask, clean_min_degree, clean_max_rounds)
        n_removed_noise = int(mask.sum() - cleaned.sum())
        defect_mask = cleaned
        if valid_mask is not None:
            valid_mask = valid_mask | (cleaned > 0)  # keep original valid region

    # ── 路由到各方法 ──
    if method in ('raw', 'filtered', 'adhesion', 'closing', 'simi_paper'):
        return _cluster_legacy(defect_mask, valid_mask, method, **kwargs)

    if method in ('topk', 'compact', 'adhesion_topk'):
        return _cluster_topk(defect_mask, valid_mask, **kwargs)

    if method == 'topk_dilated':
        kwargs.setdefault('base_method', 'dilated_adhesion')
        return _cluster_topk(defect_mask, valid_mask, **kwargs)

    if method in ('dilated_group', 'dilated'):
        return _cluster_dilated_group(defect_mask, valid_mask, split_suspicious=False, **kwargs)

    if method in ('dilated_adhesion', 'dilated_group_adhesion'):
        return _cluster_dilated_group(defect_mask, valid_mask, split_suspicious=True, **kwargs)

    if method in ('group_then_adhesion', 'dilated_group_then_adhesion', 'gta'):
        kwargs.setdefault('dilation_radius', 1)
        kwargs.setdefault('pre_filter', True)
        kwargs.setdefault('skip_ring_like', True)
        kwargs.setdefault('min_suspicious_cues', 1)
        kwargs.setdefault('max_split_count', 12)
        kwargs.setdefault('min_split_coverage', 0.5)
        return _cluster_dilated_group(
            defect_mask,
            valid_mask,
            split_suspicious=True,
            proposal_source='group_then_adhesion_group',
            split_source='group_then_adhesion',
            **kwargs,
        )

    if method in ('topk_group_then_adhesion', 'topk_gta'):
        kwargs.setdefault('base_method', 'group_then_adhesion')
        return _cluster_topk(defect_mask, valid_mask, **kwargs)

    if method in ('geometry_merge', 'geom_merge', 'fragment_merge'):
        return _cluster_geometry_merge(defect_mask, valid_mask, **kwargs)

    if method in ('topk_geometry_merge', 'topk_geom_merge'):
        kwargs.setdefault('base_method', 'geometry_merge')
        return _cluster_topk(defect_mask, valid_mask, **kwargs)

    if method == 'dbscan':
        return _cluster_dbscan(defect_mask, valid_mask, **kwargs)

    if method == 'adjacency_iwmm':
        # use_clean 时跳过内部 Stage1 AC 过滤，直接 DP-GMM
        if use_clean:
            return _cluster_iwmm_only(defect_mask, valid_mask,
                                      n_removed_noise=n_removed_noise, **kwargs)
        return _cluster_adjacency_iwmm(defect_mask, valid_mask, **kwargs)

    if method == 'spectral':
        return _cluster_spectral(defect_mask, valid_mask, **kwargs)

    if method in ('tensor_voting', 'tv'):
        return _cluster_tensor_voting(defect_mask, valid_mask, **kwargs)

    raise ValueError(f"Unknown clustering method: {method}. "
                     f"Available: raw, filtered, adhesion, dilated_group, "
                     f"dilated_adhesion, group_then_adhesion, geometry_merge, topk, closing, simi_paper, "
                     f"dbscan, adjacency_iwmm, spectral, tensor_voting")


# ============================================================
# 0. 公共工具函数
# ============================================================

def _compute_cluster_stats(
    pixels: np.ndarray,  # (N, 2) [row, col]
    H: int,
    W: int,
    **extra
) -> Dict:
    """计算单个 cluster 的统计信息"""
    rows = pixels[:, 0]
    cols = pixels[:, 1]
    area = len(pixels)

    centroid_row = float(rows.mean())
    centroid_col = float(cols.mean())

    bbox_row_min = int(rows.min())
    bbox_col_min = int(cols.min())
    bbox_row_max = int(rows.max())
    bbox_col_max = int(cols.max())

    # PCA
    pca_l1 = pca_l2 = 0.0
    orientation = 0.0
    if area > 1:
        cov_rr = float(np.mean((rows - centroid_row) ** 2))
        cov_cc = float(np.mean((cols - centroid_col) ** 2))
        cov_rc = float(np.mean((rows - centroid_row) * (cols - centroid_col)))
        cov = np.array([[cov_rr, cov_rc], [cov_rc, cov_cc]])
        vals, vecs = np.linalg.eigh(cov)
        pca_l1, pca_l2 = float(vals[1]), float(vals[0])
        if vals[1] > 1e-10:
            orientation = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))

    # Perimeter
    pixel_set = set((int(r), int(c)) for r, c in pixels)
    perimeter = 0
    for x, y in pixel_set:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (x + dx, y + dy) not in pixel_set:
                perimeter += 1

    radial = np.sqrt((centroid_row - H / 2) ** 2 + (centroid_col - W / 2) ** 2)
    radial_norm = float(radial / max(np.sqrt((H / 2) ** 2 + (W / 2) ** 2), 1))

    result = {
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
        'pca_lambda1': pca_l1,
        'pca_lambda2': pca_l2,
        'orientation': orientation,
        'perimeter': perimeter,
        'compactness': perimeter / max(area, 1),
        'radial_distance_norm': radial_norm,
        'pixels': [(int(r), int(c)) for r, c in pixels],
        'pixel_coords': [{'row': int(r), 'col': int(c)} for r, c in pixels],
    }
    result.update(extra)
    return result


def _connected_components(mask: np.ndarray, connectivity: int = 2) -> List[np.ndarray]:
    """8-连通域/4-连通域分割，返回每组点的 (N,2) 数组"""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    if connectivity == 2:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                queue = [(i, j)]
                visited[i, j] = True
                comp = []
                while queue:
                    x, y = queue.pop(0)
                    comp.append((x, y))
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                if comp:
                    components.append(np.array(comp, dtype=np.float32))
    return components


# ============================================================
# 1. Legacy: raw / filtered / closing / simi_paper
# ============================================================

def _build_cross_se():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _build_se(radius: int = 1, structure: str = 'cross') -> np.ndarray:
    """Build a small binary structuring element."""
    radius = max(int(radius), 1)
    size = 2 * radius + 1
    if structure == 'square':
        return np.ones((size, size), dtype=bool)

    center = radius
    se = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) <= radius:
                se[r, c] = True
    return se


def _custom_binary_dilation(mask: np.ndarray, structure: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small fallback binary dilation for environments without scipy."""
    result = mask.copy()
    sH, sW = structure.shape
    pad_h, pad_w = sH // 2, sW // 2
    for _ in range(max(int(iterations), 1)):
        padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        out = np.zeros_like(result, dtype=bool)
        H, W = result.shape
        for i in range(H):
            for j in range(W):
                region = padded[i:i + sH, j:j + sW]
                out[i, j] = bool(np.any(region[structure]))
        result = out
    return result


def _custom_binary_closing(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """自定义二值闭运算"""
    H, W = mask.shape
    sH, sW = structure.shape
    pad_h, pad_w = sH // 2, sW // 2
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Dilation
    dilated = np.zeros_like(padded, dtype=bool)
    for i in range(H):
        for j in range(W):
            region = padded[i:i + sH, j:j + sW]
            if np.any(region[structure]):
                dilated[i + pad_h, j + pad_w] = True

    # Erosion
    eroded = np.zeros_like(dilated, dtype=bool)
    for i in range(H):
        for j in range(W):
            if dilated[i + pad_h, j + pad_w]:
                region = dilated[i:i + sH, j:j + sW]
                if np.all(region[structure]):
                    eroded[i + pad_h, j + pad_w] = True

    return eroded[pad_h:pad_h + H, pad_w:pad_w + W]


def _spatial_filter(mask: np.ndarray) -> np.ndarray:
    """5x5 Spatial Filter: 0/0.5/1 三值图"""
    H, W = mask.shape
    output = np.zeros_like(mask, dtype=np.float32)
    padded = np.pad(mask, 2, mode='constant')
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            region = padded[i:i + 5, j:j + 5]
            near = region[1:4, 1:4].sum()
            far = region.sum()
            if near >= 3:
                output[i, j] = 1.0
            elif far >= 5:
                output[i, j] = 0.5
    return output


def _cluster_legacy(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    method: str = 'raw',
    **kwargs
) -> List[Dict]:
    """Legacy 方法分发"""
    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()

    if method == 'raw':
        components = _connected_components(mask)
        return [_compute_cluster_stats(comp, H, W) for comp in components]

    if method == 'filtered':
        min_area = kwargs.get('min_area', 3)
        components = _connected_components(mask)
        return [_compute_cluster_stats(comp, H, W)
                for comp in components if len(comp) >= min_area]

    if method == 'adhesion':
        min_area = kwargs.get('min_area', 3)
        split_method = kwargs.get('split_method', 'tv_hybrid')
        suspicious_area = kwargs.get('suspicious_area', 40)
        min_suspicious_cues = kwargs.get('min_suspicious_cues', 2)
        min_split_area = kwargs.get('min_split_area', min_area)
        max_split_count = kwargs.get('max_split_count', 6)
        min_split_coverage = kwargs.get('min_split_coverage', 0.75)
        components = _connected_components(mask)
        clusters = []
        for comp in components:
            if len(comp) < min_area:
                continue
            base = _compute_cluster_stats(comp, H, W)
            if not _is_suspicious_adhesion(
                base,
                suspicious_area=suspicious_area,
                min_cues=min_suspicious_cues,
            ):
                clusters.append(base)
                continue

            comp_mask = np.zeros((H, W), dtype=bool)
            comp_mask[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True
            split_clusters = _split_adhesion_component(
                comp_mask,
                valid_mask,
                split_method=split_method,
                min_split_area=min_split_area,
                fallback=base,
                max_split_count=max_split_count,
                min_split_coverage=min_split_coverage,
                **_adhesion_cluster_kwargs(kwargs),
            )
            clusters.extend(split_clusters)

        clusters.sort(key=lambda c: c['area'], reverse=True)
        return clusters

    if method == 'closing':
        cross = _build_cross_se()
        try:
            from scipy.ndimage import binary_closing
            closed = binary_closing(mask, cross)
        except ImportError:
            closed = _custom_binary_closing(mask, cross)
        components = _connected_components(closed)
        return [_compute_cluster_stats(comp, H, W) for comp in components]

    if method == 'simi_paper':
        cross = _build_cross_se()
        try:
            from scipy.ndimage import binary_closing
            closed = binary_closing(mask, cross)
        except ImportError:
            closed = _custom_binary_closing(mask, cross)
        soft = _spatial_filter(closed)
        strong = soft >= 0.9
        components = _connected_components(strong)
        return [_compute_cluster_stats(comp, H, W) for comp in components]

    return []


def _is_suspicious_adhesion(
    cluster_stats: Dict,
    suspicious_area: int = 40,
    min_cues: int = 2,
    aspect_threshold: float = 2.5,
    elongation_threshold: float = 8.0,
    compactness_threshold: float = 1.2,
    fill_threshold: float = 0.25,
) -> bool:
    """Return True for connected components likely to contain merged patterns."""
    area = cluster_stats['area']
    if area < suspicious_area:
        return False

    height = max(cluster_stats['bbox_height'], 1)
    width = max(cluster_stats['bbox_width'], 1)
    aspect = max(height / width, width / height)
    fill_ratio = area / max(height * width, 1)
    compactness = cluster_stats['compactness']
    pca_l1 = cluster_stats['pca_lambda1']
    pca_l2 = max(cluster_stats['pca_lambda2'], 1e-6)
    elongation = pca_l1 / pca_l2

    cues = [
        aspect >= aspect_threshold,
        elongation >= elongation_threshold,
        compactness >= compactness_threshold,
        fill_ratio <= fill_threshold,
    ]
    return sum(bool(cue) for cue in cues) >= min_cues


def _adhesion_cluster_kwargs(kwargs: Dict) -> Dict:
    """Remove wrapper-level adhesion controls before calling adhesion_cluster."""
    split_kwargs = dict(kwargs)
    for key in (
        'min_area',
        'suspicious_area',
        'min_suspicious_cues',
        'min_split_area',
        'split_method',
        'max_split_count',
        'min_split_coverage',
    ):
        split_kwargs.pop(key, None)
    return split_kwargs


def _split_adhesion_component(
    comp_mask: np.ndarray,
    valid_mask: Optional[np.ndarray],
    split_method: str,
    min_split_area: int,
    fallback: Dict,
    max_split_count: int = 6,
    min_split_coverage: float = 0.75,
    **kwargs
) -> List[Dict]:
    """Split one suspicious component and fall back if the split is not useful."""
    from partial_match.core.adhesion_split import adhesion_cluster

    split_kwargs = _adhesion_cluster_kwargs(kwargs)
    radial_split = _split_radial_ring_component(
        comp_mask,
        valid_mask,
        min_split_area=min_split_area,
        fallback=fallback,
        max_split_count=max_split_count,
        min_split_coverage=min_split_coverage,
    )
    if radial_split is not None:
        return radial_split

    try:
        split_clusters = adhesion_cluster(
            comp_mask,
            valid_mask,
            method=split_method,
            **split_kwargs,
        )
    except Exception:
        return [fallback]

    split_clusters = [c for c in split_clusters if c.get('area', 0) >= min_split_area]
    if len(split_clusters) <= 1:
        return [fallback]
    if len(split_clusters) > max_split_count:
        return [fallback]

    split_area = sum(c.get('area', 0) for c in split_clusters)
    if split_area < fallback.get('area', 1) * min_split_coverage:
        return [fallback]

    for cluster in split_clusters:
        cluster['proposal_source'] = 'adhesion'
        cluster['adhesion_method'] = split_method
    return split_clusters


def _split_radial_ring_component(
    comp_mask: np.ndarray,
    valid_mask: Optional[np.ndarray],
    min_split_area: int,
    fallback: Dict,
    max_split_count: int = 6,
    min_split_coverage: float = 0.75,
    radial_bins: int = 28,
    angular_bins: int = 36,
    band_half_width: float = 0.075,
    min_angular_coverage: float = 0.18,
    min_ring_area_ratio: float = 0.18,
    min_residual_area_ratio: float = 0.06,
) -> Optional[List[Dict]]:
    """
    Split donut/edge-ring-like pixels from attached scratch/loc fragments.

    This is intentionally radial rather than morphological: it finds a dense
    wafer-centered radius band, keeps that as the ring token, and lets geometry
    merge reconnect residual scratch fragments later.
    """
    H, W = comp_mask.shape
    mask = comp_mask & valid_mask if valid_mask is not None else comp_mask.copy()
    points = np.argwhere(mask).astype(np.float32)
    area = len(points)
    if area < max(min_split_area * 3, 30):
        return None

    center = np.array([H / 2.0, W / 2.0], dtype=np.float32)
    if valid_mask is not None and valid_mask.any():
        valid_points = np.argwhere(valid_mask).astype(np.float32)
        radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max())
    else:
        radius_ref = float(max(np.linalg.norm(points - center, axis=1).max(), 1.0))
    if radius_ref <= 1e-6:
        return None

    rel = points - center
    radial = np.linalg.norm(rel, axis=1) / radius_ref
    hist, edges = np.histogram(radial, bins=radial_bins, range=(0.0, 1.05))
    if hist.max() < max(min_split_area * 2, area * min_ring_area_ratio):
        return None

    peak = int(hist.argmax())
    band_center = float((edges[peak] + edges[peak + 1]) / 2.0)
    ring_keep = np.abs(radial - band_center) <= band_half_width
    ring_points = points[ring_keep]
    residual_points = points[~ring_keep]

    if len(ring_points) < max(min_split_area, area * min_ring_area_ratio):
        return None
    if len(residual_points) < max(min_split_area, area * min_residual_area_ratio):
        return None

    theta = (np.degrees(np.arctan2(ring_points[:, 0] - center[0], ring_points[:, 1] - center[1])) + 360.0) % 360.0
    occupied = np.unique(np.floor(theta / (360.0 / angular_bins)).astype(int))
    angular_coverage = len(occupied) / angular_bins
    if angular_coverage < min_angular_coverage:
        return None

    ring_mask = np.zeros((H, W), dtype=bool)
    ring_mask[ring_points[:, 0].astype(int), ring_points[:, 1].astype(int)] = True
    residual_mask = mask & (~ring_mask)

    clusters = [
        _compute_cluster_stats(
            ring_points,
            H,
            W,
            proposal_source='radial_ring_split',
            adhesion_method='radial_ring_split',
            radial_band_center=band_center,
            angular_coverage=float(angular_coverage),
        )
    ]
    for comp in _connected_components(residual_mask):
        if len(comp) >= min_split_area:
            clusters.append(_compute_cluster_stats(
                comp,
                H,
                W,
                proposal_source='radial_ring_split',
                adhesion_method='radial_ring_split',
                radial_band_center=band_center,
            ))

    if len(clusters) <= 1 or len(clusters) > max_split_count:
        return None
    split_area = sum(c.get('area', 0) for c in clusters)
    if split_area < fallback.get('area', 1) * min_split_coverage:
        return None

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


def _is_ring_like_group(
    cluster_stats: Dict,
    H: int,
    W: int,
    radial_threshold: float = 0.45,
    fill_threshold: float = 0.35,
    min_area: int = 40,
) -> bool:
    """
    Conservative ring/edge guard.

    This is intentionally weak: it only prevents adhesion split for large,
    sparse groups whose centroid is away from wafer center. It is not a full
    ring detector.
    """
    area = cluster_stats.get('area', 0)
    if area < min_area:
        return False

    bbox_area = max(cluster_stats.get('bbox_height', 1) * cluster_stats.get('bbox_width', 1), 1)
    fill_ratio = area / bbox_area
    radial = cluster_stats.get('radial_distance_norm', 0.0)

    touches_edge_band = (
        cluster_stats.get('bbox_row_min', H) <= 4
        or cluster_stats.get('bbox_col_min', W) <= 4
        or cluster_stats.get('bbox_row_max', 0) >= H - 5
        or cluster_stats.get('bbox_col_max', 0) >= W - 5
    )

    return radial >= radial_threshold and fill_ratio <= fill_threshold and touches_edge_band


def _cluster_dilated_group(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_area: int = 3,
    dilation_radius: int = 1,
    use_closing: bool = False,
    structure: str = 'cross',
    split_suspicious: bool = False,
    split_method: str = 'tv_hybrid',
    suspicious_area: int = 40,
    min_split_area: Optional[int] = None,
    proposal_source: str = 'dilated_group',
    split_source: str = 'dilated_adhesion',
    skip_ring_like: bool = False,
    pre_filter: bool = False,
    **kwargs
) -> List[Dict]:
    """
    Group fragments on a dilated mask, but compute final token stats from original pixels.

    If split_suspicious=True, suspicious grouped original pixels are passed to adhesion split.
    """
    H, W = defect_mask.shape
    original = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()
    grouping_input = original
    filtered_area = int(original.sum())
    if pre_filter:
        grouping_input = np.zeros_like(original, dtype=bool)
        for comp in _connected_components(original):
            if len(comp) >= min_area:
                grouping_input[comp[:, 0].astype(int), comp[:, 1].astype(int)] = True
        filtered_area = int(grouping_input.sum())

    se = _build_se(dilation_radius, structure)

    try:
        from scipy.ndimage import binary_dilation, binary_closing
        if use_closing:
            grouping = binary_closing(grouping_input, structure=se)
        else:
            grouping = binary_dilation(grouping_input, structure=se)
    except ImportError:
        if use_closing:
            grouping = _custom_binary_closing(grouping_input, se)
        else:
            grouping = _custom_binary_dilation(grouping_input, se)

    if valid_mask is not None:
        grouping = grouping & valid_mask

    groups = _connected_components(grouping)
    clusters = []
    split_min_area = min_area if min_split_area is None else min_split_area

    for group in groups:
        group_mask = np.zeros((H, W), dtype=bool)
        group_mask[group[:, 0].astype(int), group[:, 1].astype(int)] = True
        original_pixels = np.argwhere(original & group_mask).astype(np.float32)
        if len(original_pixels) < min_area:
            continue

        base = _compute_cluster_stats(
            original_pixels,
            H,
            W,
            proposal_source=proposal_source,
            grouping_area=int(group_mask.sum()),
            virtual_gap_area=int(group_mask.sum() - len(original_pixels)),
            dilation_radius=int(dilation_radius),
            use_closing=bool(use_closing),
            pre_filter=bool(pre_filter),
            filtered_area=filtered_area,
            split_status='not_attempted',
        )

        ring_like = _is_ring_like_group(base, H, W)
        should_split = (
            split_suspicious
            and not (skip_ring_like and ring_like)
            and _is_suspicious_adhesion(
                base,
                suspicious_area=suspicious_area,
                min_cues=kwargs.get('min_suspicious_cues', 1),
            )
        )

        if should_split:
            original_group_mask = np.zeros((H, W), dtype=bool)
            original_group_mask[
                original_pixels[:, 0].astype(int),
                original_pixels[:, 1].astype(int),
            ] = True
            split_clusters = _split_adhesion_component(
                original_group_mask,
                valid_mask,
                split_method=split_method,
                min_split_area=split_min_area,
                fallback=base,
                max_split_count=kwargs.get('max_split_count', 6),
                min_split_coverage=kwargs.get('min_split_coverage', 0.75),
                **_adhesion_cluster_kwargs(kwargs),
            )
            split_accepted = not (len(split_clusters) == 1 and split_clusters[0] is base)
            if split_accepted:
                for item in split_clusters:
                    item['proposal_source'] = split_source
                    item['grouping_area'] = int(group_mask.sum())
                    item['virtual_gap_area'] = int(group_mask.sum() - len(original_pixels))
                    item['dilation_radius'] = int(dilation_radius)
                    item['use_closing'] = bool(use_closing)
                    item['pre_filter'] = bool(pre_filter)
                    item['filtered_area'] = filtered_area
                    item['split_status'] = 'accepted'
                    item['ring_like_guard'] = bool(ring_like)
                clusters.extend(split_clusters)
            else:
                base['split_status'] = 'rejected_fallback'
                base['ring_like_guard'] = bool(ring_like)
                clusters.append(base)
        else:
            if split_suspicious:
                base['split_status'] = 'skipped_ring_like' if (skip_ring_like and ring_like) else 'skipped_not_suspicious'
                base['ring_like_guard'] = bool(ring_like)
            clusters.append(base)

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


def _cluster_pixels_array(cluster_item: Dict) -> np.ndarray:
    coords = cluster_item.get('pixels', cluster_item.get('pixel_coords', []))
    pixels = []
    for coord in coords:
        if isinstance(coord, dict):
            pixels.append((int(coord['row']), int(coord['col'])))
        else:
            pixels.append((int(coord[0]), int(coord[1])))
    return np.array(pixels, dtype=np.float32)


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + 90.0) % 180.0 - 90.0)
    return float(diff)


def _min_pixel_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    diff = a[:, None, :] - b[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(dist2.min()))


def _line_merge_ok(a: Dict, b: Dict, max_gap: float, max_angle_diff: float, max_perp_gap: float) -> bool:
    if a.get('area', 0) < 4 or b.get('area', 0) < 4:
        return False
    if _angle_diff_deg(a.get('orientation', 0.0), b.get('orientation', 0.0)) > max_angle_diff:
        return False

    elong_a = a.get('pca_lambda1', 0.0) / max(a.get('pca_lambda2', 0.0), 1e-6)
    elong_b = b.get('pca_lambda1', 0.0) / max(b.get('pca_lambda2', 0.0), 1e-6)
    if max(elong_a, elong_b) < 4.0:
        return False

    p1 = np.array([a['centroid_row'], a['centroid_col']], dtype=float)
    p2 = np.array([b['centroid_row'], b['centroid_col']], dtype=float)
    d = p2 - p1
    center_gap = float(np.linalg.norm(d))
    if center_gap > max_gap:
        return False

    theta = np.radians((a.get('orientation', 0.0) + b.get('orientation', 0.0)) / 2.0)
    axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    perp_gap = abs(float(d[0] * axis[1] - d[1] * axis[0]))
    return perp_gap <= max_perp_gap


def _cluster_theta_bins(cluster_item: Dict, H: int, W: int, bins: int = 72) -> np.ndarray:
    pixels = _cluster_pixels_array(cluster_item)
    if len(pixels) == 0:
        return np.array([], dtype=int)
    center = np.array([H / 2.0, W / 2.0], dtype=np.float32)
    rel = pixels - center
    theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
    return np.unique(np.floor(theta / (360.0 / bins)).astype(int))


def _theta_bin_gap_deg(a_bins: np.ndarray, b_bins: np.ndarray, bins: int = 72) -> float:
    if len(a_bins) == 0 or len(b_bins) == 0:
        return 360.0
    diff = np.abs(a_bins[:, None] - b_bins[None, :])
    circular = np.minimum(diff, bins - diff)
    return float(circular.min() * (360.0 / bins))


def _ring_merge_ok(a: Dict, b: Dict, H: int, W: int, max_radial_gap: float, max_theta_gap: float) -> bool:
    ra = a.get('radial_distance_norm', 0.0)
    rb = b.get('radial_distance_norm', 0.0)
    if min(ra, rb) < 0.25:
        return False
    if abs(ra - rb) > max_radial_gap:
        return False

    theta_gap = _theta_bin_gap_deg(
        _cluster_theta_bins(a, H, W),
        _cluster_theta_bins(b, H, W),
    )
    return theta_gap <= max_theta_gap


def _blob_merge_ok(a: Dict, b: Dict, pix_a: np.ndarray, pix_b: np.ndarray, max_gap: float, max_bbox_area: int) -> bool:
    if _min_pixel_distance(pix_a, pix_b) > max_gap:
        return False
    row_min = min(a['bbox_row_min'], b['bbox_row_min'])
    row_max = max(a['bbox_row_max'], b['bbox_row_max'])
    col_min = min(a['bbox_col_min'], b['bbox_col_min'])
    col_max = max(a['bbox_col_max'], b['bbox_col_max'])
    return (row_max - row_min + 1) * (col_max - col_min + 1) <= max_bbox_area


def _merge_cluster_items(items: List[Dict], H: int, W: int, proposal_source: str, merge_reason: str) -> Dict:
    pixels = np.vstack([_cluster_pixels_array(item) for item in items])
    pixels = np.unique(pixels.astype(int), axis=0).astype(np.float32)
    sources = sorted(set(item.get('proposal_source', 'candidate') for item in items))
    return _compute_cluster_stats(
        pixels,
        H,
        W,
        proposal_source=proposal_source,
        merge_reason=merge_reason,
        merged_count=len(items),
        merged_sources=sources,
    )


def _cluster_geometry_merge(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_area: int = 5,
    base_method: str = 'adhesion',
    split_method: str = 'tv_hybrid',
    suspicious_area: int = 40,
    min_suspicious_cues: int = 1,
    max_split_count: int = 8,
    min_split_coverage: float = 0.6,
    enable_ring_merge: bool = True,
    enable_line_merge: bool = True,
    enable_blob_merge: bool = True,
    ring_radial_gap: float = 0.14,
    ring_theta_gap: float = 55.0,
    line_gap: float = 11.0,
    line_angle_gap: float = 30.0,
    line_perp_gap: float = 5.0,
    blob_gap: float = 3.0,
    blob_max_bbox_area: int = 220,
    **kwargs
) -> List[Dict]:
    """
    Merge truncated fragments at component level. No mask dilation is used.

    The graph links candidate components only when ring, line, or local-blob
    geometry is compatible, then each connected graph component becomes a token.
    """
    H, W = defect_mask.shape
    adhesion_extra = dict(kwargs)
    for key in (
        'top_k',
        'dilation_radius',
        'use_closing',
        'structure',
        'skip_ring_like',
        'pre_filter',
    ):
        adhesion_extra.pop(key, None)

    if base_method == 'adhesion':
        candidates = _cluster_legacy(
            defect_mask,
            valid_mask,
            'adhesion',
            min_area=min_area,
            split_method=split_method,
            suspicious_area=suspicious_area,
            min_suspicious_cues=min_suspicious_cues,
            max_split_count=max_split_count,
            min_split_coverage=min_split_coverage,
            **adhesion_extra,
        )
    else:
        candidates = cluster(
            defect_mask,
            valid_mask,
            method=base_method,
            min_area=min_area,
            **kwargs,
        )

    candidates = [c for c in candidates if c.get('area', 0) >= min_area]
    n = len(candidates)
    if n <= 1:
        for item in candidates:
            item['proposal_source'] = item.get('proposal_source', 'geometry_merge')
            item['merge_reason'] = 'single'
            item['merged_count'] = 1
        return candidates

    pix = [_cluster_pixels_array(item) for item in candidates]
    parent = list(range(n))
    reasons = ['single'] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a_idx: int, b_idx: int, reason: str) -> None:
        ra, rb = find(a_idx), find(b_idx)
        if ra == rb:
            if reasons[ra] == 'single':
                reasons[ra] = reason
            return
        parent[rb] = ra
        if reasons[ra] == 'single':
            reasons[ra] = reason
        elif reason not in reasons[ra].split('+'):
            reasons[ra] = reasons[ra] + '+' + reason

    for i in range(n):
        for j in range(i + 1, n):
            if enable_ring_merge and _ring_merge_ok(candidates[i], candidates[j], H, W, ring_radial_gap, ring_theta_gap):
                union(i, j, 'ring')
                continue
            if enable_line_merge and _line_merge_ok(candidates[i], candidates[j], line_gap, line_angle_gap, line_perp_gap):
                union(i, j, 'line')
                continue
            if enable_blob_merge and _blob_merge_ok(candidates[i], candidates[j], pix[i], pix[j], blob_gap, blob_max_bbox_area):
                union(i, j, 'blob')

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    clusters = []
    for root, indices in groups.items():
        items = [candidates[idx] for idx in indices]
        if len(items) == 1:
            item = dict(items[0])
            item['proposal_source'] = item.get('proposal_source', 'geometry_merge_candidate')
            item['merge_reason'] = 'single'
            item['merged_count'] = 1
            clusters.append(item)
        else:
            clusters.append(_merge_cluster_items(
                items,
                H,
                W,
                proposal_source='geometry_merge',
                merge_reason=reasons[find(root)],
            ))

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


def _cluster_topk(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    top_k: int = 5,
    base_method: str = 'geometry_merge',
    min_area: int = 5,
    **kwargs
) -> List[Dict]:
    """
    Compact retrieval proposal: generate candidates, then keep the largest K regions.

    This intentionally produces a small pattern-level token set for 52x52 maps.
    """
    base_method = base_method.lower().strip()
    if base_method in ('topk', 'compact', 'adhesion_topk'):
        raise ValueError("topk base_method cannot be another topk method")

    candidates = cluster(
        defect_mask,
        valid_mask,
        method=base_method,
        min_area=min_area,
        **kwargs,
    )
    candidates = [c for c in candidates if c.get('area', 0) >= min_area]
    candidates.sort(key=lambda c: c['area'], reverse=True)

    selected = candidates[:max(int(top_k), 0)]
    for rank, item in enumerate(selected):
        item['proposal_source'] = item.get('proposal_source', base_method)
        item['proposal_type'] = 'topk'
        item['topk_rank'] = rank
        item['topk_base_method'] = base_method
    return selected


# ============================================================
# 2. DBSCAN（Koo & Hwang 2021）
# ============================================================

def _estimate_eps(points: np.ndarray, min_samples: int = 5) -> float:
    """k-distance 图自动估计 eps"""
    if len(points) < min_samples + 1:
        return 3.0
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(points)
    distances, _ = nbrs.kneighbors(points)
    k_dist = np.sort(distances[:, -1])
    d2 = np.gradient(np.gradient(k_dist))
    elbow = np.argmax(d2[len(d2) // 4:]) + len(d2) // 4
    return max(k_dist[elbow], 0.5)


def _cluster_dbscan(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: Optional[float] = None,
    min_samples: int = 5,
    auto_eps: bool = True,
    **_
) -> List[Dict]:
    """DBSCAN 聚类"""
    from sklearn.cluster import DBSCAN

    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask
    points = np.argwhere(mask).astype(np.float32)
    if len(points) < min_samples:
        return []

    if auto_eps and eps is None:
        eps = _estimate_eps(points, min_samples)
    elif eps is None:
        eps = 2.0

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    clusters = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        pts = points[labels == lbl]
        clusters.append(_compute_cluster_stats(pts, H, W))
    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ============================================================
# 3. Adjacency-Clustering + iWMM（Ezzat et al. 2020）
# ============================================================

def _iwmm_dp_gmm(points: np.ndarray,
                 max_components: int = 8,
                 weight_concentration_prior: float = 0.15) -> np.ndarray:
    """DP-GMM 聚类（iWMM 的 Stage 2，不包含 AC 过滤）"""
    from sklearn.mixture import BayesianGaussianMixture

    N = len(points)
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
    return bgm.predict(points)


def _cluster_iwmm_only(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_components: int = 8,
    weight_concentration_prior: float = 0.15,
    n_removed_noise: int = 0,
    **_
) -> List[Dict]:
    """仅 DP-GMM 聚类（已预清洗，跳过 Stage1 AC）"""
    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask
    points = np.argwhere(mask).astype(np.float32)
    if len(points) < 2:
        return []

    labels = _iwmm_dp_gmm(points, max_components, weight_concentration_prior)

    clusters = []
    for lbl in set(labels):
        pts = points[labels == lbl]
        if len(pts) < 2:
            continue
        clusters.append(_compute_cluster_stats(pts, H, W,
                                                n_removed_noise=n_removed_noise))
    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters

def _adjacency_filter(defect_mask: np.ndarray,
                      min_degree: int = 2,
                      max_rounds: int = 3) -> np.ndarray:
    """Stage 1: 图论空间过滤，移除孤立噪声"""
    from collections import defaultdict

    H, W = defect_mask.shape
    current = defect_mask.copy()
    neighbors8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for _ in range(max_rounds):
        pts = np.argwhere(current)
        if len(pts) == 0:
            break
        pt_to_idx = {(r, c): i for i, (r, c) in enumerate(pts)}
        degree = np.zeros(len(pts), dtype=int)
        for (r, c), idx in pt_to_idx.items():
            for dr, dc in neighbors8:
                if (r + dr, c + dc) in pt_to_idx:
                    degree[idx] += 1
        noise = degree < min_degree
        if noise.sum() == 0:
            break
        for r, c in pts[noise]:
            current[int(r), int(c)] = False
    return current


def _cluster_adjacency_iwmm(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_degree: int = 2,
    max_rounds: int = 3,
    max_components: int = 8,
    weight_concentration_prior: float = 0.15,
    **_
) -> List[Dict]:
    """Adjacency-Clustering + DP-GMM"""
    from sklearn.mixture import BayesianGaussianMixture

    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask

    # Stage 1: AC 空间过滤
    filtered = _adjacency_filter(mask, min_degree, max_rounds)
    n_removed = int(mask.sum() - filtered.sum())

    points = np.argwhere(filtered).astype(np.float32)
    if len(points) < 2:
        return []

    # Stage 2: DP-GMM
    n_comp = min(max_components, len(points))
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

    clusters = []
    for lbl in set(labels):
        pts = points[labels == lbl]
        if len(pts) < 2:
            continue
        clusters.append(_compute_cluster_stats(pts, H, W, n_removed_noise=n_removed))

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ============================================================
# 4. Spectral Clustering（Wang et al.）
# ============================================================

def _gaussian_similarity(points: np.ndarray, sigma: float) -> np.ndarray:
    """RBF 相似度矩阵"""
    N = len(points)
    W = np.zeros((N, N), dtype=np.float32)
    s2 = 2.0 * sigma ** 2
    for i in range(N):
        for j in range(i + 1, N):
            d2 = np.sum((points[i] - points[j]) ** 2)
            if d2 > (5 * sigma) ** 2:
                continue
            v = np.exp(-d2 / s2)
            W[i, j] = W[j, i] = v
    return W


def _estimate_k(eigvals: np.ndarray, max_k: int = 8) -> int:
    """eigengap 自动确定 k"""
    n = min(len(eigvals), max_k + 1)
    vals = eigvals[:n]
    gaps = np.diff(vals)
    if len(gaps) == 0:
        return 2
    return max(2, int(np.argmax(gaps) + 1))


def _cluster_spectral(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sigma: float = 3.0,
    n_clusters: Optional[int] = None,
    auto_k: bool = True,
    **_
) -> List[Dict]:
    """Spectral Clustering"""
    from sklearn.cluster import KMeans

    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask
    points = np.argwhere(mask).astype(np.float32)
    N = len(points)
    if N < 3:
        return []

    # 1. 相似度矩阵
    W_sim = _gaussian_similarity(points, sigma)

    # 2. 归一化拉普拉斯
    D_diag = W_sim.sum(axis=1)
    inv_sqrt = np.zeros_like(D_diag, dtype=np.float32)
    nonzero = D_diag > 1e-8
    inv_sqrt[nonzero] = 1.0 / np.sqrt(D_diag[nonzero])
    W_norm = W_sim * inv_sqrt[:, None] * inv_sqrt[None, :]
    L_sym = np.eye(N, dtype=np.float32) - W_norm
    L_sym = np.nan_to_num(L_sym, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. 特征分解
    try:
        eigvals, eigvecs = np.linalg.eigh(L_sym)
    except np.linalg.LinAlgError:
        return []

    # 4. 确定 k
    if auto_k and n_clusters is None:
        n_clusters = _estimate_k(eigvals, min(10, N))
    elif n_clusters is None:
        n_clusters = 3
    n_clusters = min(n_clusters, N)

    # 5. K-means on eigenvectors
    X = eigvecs[:, :n_clusters]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-10)

    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X)

    clusters = []
    for lbl in range(n_clusters):
        pts = points[labels == lbl]
        if len(pts) < 2:
            continue
        clusters.append(_compute_cluster_stats(pts, H, W))
    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ============================================================
# 5. Tensor Voting（Wang et al. 2022）
# ============================================================

def _tensor_voting_core(points: np.ndarray, sigma: float) -> np.ndarray:
    """Guy-Medioni ball voting: T += DF * (I - vv^T)"""
    N = len(points)
    tensors = np.zeros((N, 2, 2), dtype=np.float32)
    I_eye = np.eye(2, dtype=np.float32)
    cutoff = (3 * sigma) ** 2

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            vec = points[i] - points[j]
            d2 = vec @ vec
            if d2 > cutoff:
                continue
            v = vec / np.sqrt(d2)
            projection = I_eye - np.outer(v, v)
            tensors[i] += np.exp(-d2 / sigma ** 2) * projection

    return tensors


def _cluster_tensor_voting(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sigma: float = 5.0,
    noise_ratio: float = 0.3,
    **_
) -> List[Dict]:
    """Tensor Voting → 噪声过滤 → 连通域聚类"""
    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask
    points = np.argwhere(mask).astype(np.float32)
    N = len(points)
    if N < 2:
        return []

    # 1. Tensor Voting
    tensors = _tensor_voting_core(points, sigma)

    # 2. Saliency
    stick_sal = np.zeros(N, dtype=np.float32)
    ball_sal = np.zeros(N, dtype=np.float32)
    for k in range(N):
        vals, _ = np.linalg.eigh(tensors[k])
        stick_sal[k] = max(0.0, vals[1] - vals[0])
        ball_sal[k] = max(0.0, vals[0])

    # 3. 噪声过滤
    th_stick = noise_ratio * max(stick_sal.max(), 1e-10)
    th_ball = noise_ratio * max(ball_sal.max(), 1e-10)
    keep = (stick_sal >= th_stick) | (ball_sal >= th_ball)

    # 4. 重建 mask → 连通域
    kept_mask = np.zeros((H, W), dtype=bool)
    for k, (r, c) in enumerate(points[keep]):
        kept_mask[int(round(r)), int(round(c))] = True

    components = _connected_components(kept_mask)
    clusters = []
    for comp in components:
        clusters.append(_compute_cluster_stats(comp, H, W,
                                                n_removed=int((~keep).sum())))

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ============================================================
# 便捷方法列表
# ============================================================

AVAILABLE_METHODS = [
    'raw',
    'filtered',
    'closing',
    'simi_paper',
    'dbscan',
    'adjacency_iwmm',
    'spectral',
    'tensor_voting',
    'tv',
]

METHOD_NAMES = {
    'raw':             '8-Connected (Raw)',
    'filtered':        'Filtered (area >= 3)',
    'closing':         'Closing (Morphological)',
    'simi_paper':      'SIMI Paper',
    'dbscan':          'DBSCAN',
    'adjacency_iwmm':  'Adj-Cluster + iWMM',
    'spectral':        'Spectral Clustering',
    'tensor_voting':   'Tensor Voting',
    'tv':              'Tensor Voting',
}
