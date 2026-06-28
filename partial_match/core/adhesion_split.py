# -*- coding: utf-8 -*-
"""
粘连分离模块 — 针对头尾粘连线、线与圆形粘连等场景。

提供两种方法：
  - skeleton_split:    骨架化 → 检测分叉点 → 切除 → 重新连通域
  - watershed_split:   距离变换 → 局部极大值 → 分水岭分割

两种方法均返回 List[Dict]，格式与 clustering.cluster() 兼容。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt, label


def _connected_components_points(mask: np.ndarray) -> List[np.ndarray]:
    """8-连通域分割，返回每组点的 (N,2) 数组"""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    components = []

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


def _compute_stats(pixels: np.ndarray, H: int, W: int, **extra) -> Dict:
    """计算单个 cluster 的统计信息（与 clustering.py 兼容）"""
    rows = pixels[:, 0]
    cols = pixels[:, 1]
    area = len(pixels)

    centroid_row = float(rows.mean())
    centroid_col = float(cols.mean())

    bbox_row_min = int(rows.min())
    bbox_col_min = int(cols.min())
    bbox_row_max = int(rows.max())
    bbox_col_max = int(cols.max())

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


# ============================================================
# 方法 1: 骨架分叉切割 (Skeleton Branch-Point Split)
# ============================================================

def _find_branch_points(skeleton: np.ndarray) -> np.ndarray:
    """
    在骨架图中检测分叉点（branch points）。
    分叉点定义：骨架像素，其 8-邻域内骨架点数 >= 3。
    """
    H, W = skeleton.shape
    branch_mask = np.zeros((H, W), dtype=bool)

    # 8-邻域偏移（包含自身中心也需要计算）
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if skeleton[i, j]:
                nbr_count = sum(skeleton[i + di, j + dj] for di, dj in offsets)
                if nbr_count >= 3:
                    branch_mask[i, j] = True

    return branch_mask


def _expand_branch_region(branch_mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """将分叉点膨胀 radius 像素，确保粘连处彻底断开"""
    if radius <= 0:
        return branch_mask
    H, W = branch_mask.shape
    result = branch_mask.copy()
    for _ in range(radius):
        new = result.copy()
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if result[i, j]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            new[i + di, j + dj] = True
        result = new
    return result


def skeleton_split(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    branch_radius: int = 1,
    **kwargs
) -> List[Dict]:
    """
    骨架化 + 分叉点切除 → 重新连通域。

    流程：
      1. 对 defect_mask 做骨架化
      2. 检测骨架中的分叉点（≥3 个骨架邻居）
      3. 在原始 mask 上切除分叉点区域
      4. 重新做 8-连通域分割

    Args:
        defect_mask: (H, W) bool 缺陷点
        valid_mask:  (H, W) bool 有效区域
        branch_radius: 分叉点膨胀半径（默认 1）

    Returns:
        List[Dict] clusters
    """
    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()

    if mask.sum() < 3:
        # 点太少，直接连通域
        comps = _connected_components_points(mask)
        return [_compute_stats(c, H, W) for c in comps]

    # 1. 骨架化
    skeleton = skeletonize(mask)

    # 2. 检测分叉点
    branch_pts = _find_branch_points(skeleton)

    # 3. 膨胀分叉点区域
    branch_region = _expand_branch_region(branch_pts, radius=branch_radius)

    # 4. 从 mask 中切除分叉区域
    split_mask = mask & (~branch_region)

    # 5. 重新连通域
    comps = _connected_components_points(split_mask)
    clusters = []
    for comp in comps:
        if len(comp) >= 2:  # 过滤掉切除后过小的碎片
            clusters.append(_compute_stats(comp, H, W,
                                           n_branch_pts=int(branch_pts.sum())))
        else:
            # 把这些碎片重新加到最近的 cluster（或用原始 mask 补充）
            pass

    # 6. 把切除点重新分配给最近的 cluster（复原丢失的像素）
    if branch_region.any():
        clusters = _reassign_removed_points(mask, branch_region, clusters, H, W)

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ============================================================
# 方法 2: 距离变换 + 分水岭 (Distance Transform Watershed)
# ============================================================

def watershed_split(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_distance: int = 3,
    **kwargs
) -> List[Dict]:
    """
    距离变换 → 局部极大值 → 分水岭分割。

    流程：
      1. 对 defect_mask 做距离变换
      2. 找距离图局部极大值作为 seed
      3. 分水岭分割
      4. 每个 label 作为一个 cluster

    Args:
        defect_mask: (H, W) bool 缺陷点
        valid_mask:  (H, W) bool 有效区域
        min_distance: 局部极大值最小间距

    Returns:
        List[Dict] clusters
    """
    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()

    if mask.sum() < 3:
        comps = _connected_components_points(mask)
        return [_compute_stats(c, H, W) for c in comps]

    # 1. 距离变换（背景=0，前景=到最近边界的距离）
    dist = distance_transform_edt(mask)

    # 2. 找局部极大值
    coords = peak_local_max(dist, min_distance=min_distance, labels=mask,
                            exclude_border=False)

    # 3. 构建 markers
    markers = np.zeros((H, W), dtype=np.int32)
    for idx, (r, c) in enumerate(coords):
        markers[r, c] = idx + 1

    # 4. 分水岭
    labels = watershed(-dist, markers, mask=mask)

    # 5. 按 label 收集 cluster
    clusters = []
    n_seeds = len(coords)
    for lbl in range(1, labels.max() + 1):
        pts = np.argwhere(labels == lbl).astype(np.float32)
        if len(pts) < 2:
            continue
        clusters.append(_compute_stats(pts, H, W, n_seeds=n_seeds))

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ============================================================
# 辅助：将被切除的分叉点重新分配
# ============================================================

def _reassign_removed_points(
    full_mask: np.ndarray,
    removed_region: np.ndarray,
    clusters: List[Dict],
    H: int, W: int
) -> List[Dict]:
    """将被切除的点分配给最近（按曼哈顿距离）的 cluster"""
    if not clusters:
        return clusters

    removed_coords = np.argwhere(removed_region)  # (K, 2)

    # 构建 cluster 索引图
    label_map = np.full((H, W), -1, dtype=np.int32)
    for ci, cl in enumerate(clusters):
        for r, c in cl['pixels']:
            label_map[r, c] = ci

    for rr, rc in removed_coords:
        # 搜索最近的非切除 cluster 像素（BFS）
        best_ci = -1
        for radius in range(1, max(H, W)):
            found = False
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) + abs(dc) != radius:
                        continue
                    nr, nc = rr + dr, rc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        ci = label_map[nr, nc]
                        if ci >= 0:
                            best_ci = ci
                            found = True
                            break
                if found:
                    break
            if found:
                break

        if best_ci >= 0:
            clusters[best_ci]['pixels'].append((int(rr), int(rc)))
            clusters[best_ci]['pixel_coords'].append({'row': int(rr), 'col': int(rc)})
            clusters[best_ci]['area'] += 1

    return clusters


# ============================================================
# TV 基础设施：计算 + 噪声清洗
# ============================================================

def _compute_tv_raw(points: np.ndarray, sigma: float) -> dict:
    """对给定点集计算 TV stick/ball saliency 和主方向"""
    N = len(points)
    if N < 2:
        return {
            'stick_sal': np.zeros(N, dtype=np.float32),
            'ball_sal': np.zeros(N, dtype=np.float32),
            'principal_dir': np.zeros((N, 2), dtype=np.float32),
        }

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

    stick_sal = np.zeros(N, dtype=np.float32)
    ball_sal = np.zeros(N, dtype=np.float32)
    principal_dir = np.zeros((N, 2), dtype=np.float32)

    for k in range(N):
        vals, vecs = np.linalg.eigh(tensors[k])
        stick_sal[k] = max(0.0, vals[1] - vals[0])
        ball_sal[k] = max(0.0, vals[0])
        if vals[1] > 1e-10:
            principal_dir[k] = vecs[:, 1]

    return {
        'stick_sal': stick_sal,
        'ball_sal': ball_sal,
        'principal_dir': principal_dir,
    }


def _tv_cleaned_mask(
    defect_mask: np.ndarray,
    sigma: float = 5.0,
    noise_ratio: float = 0.3,
) -> Tuple[np.ndarray, dict]:
    """
    计算 TV → 按 saliency 阈值清洗稀疏噪声点 → 返回清洗后的 mask 和 TV 结果。

    对应论文：保留 stick_sal ≥ noise_ratio × max(stick) 或 ball_sal ≥ noise_ratio × max(ball) 的点。

    Args:
        defect_mask: (H, W) bool
        sigma:       TV 尺度参数
        noise_ratio: 显著性阈值比例（论文默认 0.3）

    Returns:
        (cleaned_mask, tv_result)
        - cleaned_mask: (H, W) bool，只保留结构显著的点
        - tv_result:    dict，包含 'points', 'stick_sal', 'ball_sal', 'principal_dir'
                        仅含清洗后的保留点
    """
    H, W = defect_mask.shape
    all_points = np.argwhere(defect_mask).astype(np.float32)
    N = len(all_points)

    if N < 3:
        # 点太少，不做清洗
        tv = _compute_tv_raw(all_points, sigma)
        return defect_mask.copy(), {
            'points': all_points,
            **tv,
        }

    # 1. 第一次 TV（全量点）
    tv_all = _compute_tv_raw(all_points, sigma)

    # 2. 噪声过滤：保留结构显著的点
    stick_max = tv_all['stick_sal'].max()
    ball_max = tv_all['ball_sal'].max()
    th_stick = noise_ratio * max(stick_max, 1e-10)
    th_ball = noise_ratio * max(ball_max, 1e-10)

    keep = (tv_all['stick_sal'] >= th_stick) | (tv_all['ball_sal'] >= th_ball)

    # 3. 构建清洗后的 mask 和点集
    cleaned_mask = np.zeros((H, W), dtype=bool)
    kept_points = all_points[keep]
    for r, c in kept_points:
        cleaned_mask[int(round(r)), int(round(c))] = True

    n_removed = int((~keep).sum())
    kept_points_int = kept_points.astype(np.float32)

    # 4. 第二次 TV（仅对保留点，得到更准确的结构信息）
    tv_kept = _compute_tv_raw(kept_points_int, sigma) if len(kept_points_int) >= 2 else {
        'stick_sal': np.array([]),
        'ball_sal': np.array([]),
        'principal_dir': np.array([]),
    }

    return cleaned_mask, {
        'points': kept_points_int,
        'stick_sal': tv_kept['stick_sal'],
        'ball_sal': tv_kept['ball_sal'],
        'principal_dir': tv_kept['principal_dir'],
        'n_removed': n_removed,
        'n_kept': len(kept_points_int),
    }


def _compute_tv_saliency(defect_mask: np.ndarray, sigma: float = 5.0) -> dict:
    """计算 TV 的 stick 和 ball saliency（不做清洗，向后兼容）"""
    points = np.argwhere(defect_mask).astype(np.float32)
    tv = _compute_tv_raw(points, sigma)
    return {
        'points': points,
        'stick_sal': tv['stick_sal'],
        'ball_sal': tv['ball_sal'],
        'principal_dir': tv['principal_dir'],
    }


# ============================================================
# 方法 3: TV Saliency Junction Split（利用 TV stick saliency 检测粘连点）
# ============================================================


def tv_junction_split(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sigma: float = 5.0,
    noise_ratio: float = 0.3,
    junction_percentile: float = 30.0,
    expand_radius: int = 1,
    **kwargs
) -> List[Dict]:
    """
    TV 噪声清洗 → stick saliency 低谷检测 → 切除粘连点 → 重新连通域。

    原理：
      1. TV 计算 + 噪声过滤（保留 stick/ball saliency ≥ 30% 最大值的结构点）
      2. 在清洗后的保留点上找 stick saliency 低谷（粘连候选）
      3. 切除粘连点 → 连通域

    Args:
        defect_mask: (H, W) bool
        valid_mask:  (H, W) bool
        sigma:       TV 尺度参数
        noise_ratio: 噪声过滤阈值（论文默认 0.3）
        junction_percentile: stick saliency 分位数阈值
        expand_radius: 粘连点膨胀半径

    Returns:
        List[Dict] clusters
    """
    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()

    if mask.sum() < 3:
        return skeleton_split(mask, valid_mask=None)

    # 1. TV 计算 + 噪声清洗
    cleaned_mask, tv = _tv_cleaned_mask(mask, sigma=sigma, noise_ratio=noise_ratio)
    n_removed = tv.get('n_removed', 0)

    if tv.get('n_kept', 0) < 2:
        return skeleton_split(mask, valid_mask=None)

    # 2. 找 stick saliency 低谷点（粘连候选）
    stick = tv['stick_sal']
    if len(stick) == 0 or stick.max() < 1e-8:
        return skeleton_split(mask, valid_mask=None)

    threshold = np.percentile(stick, junction_percentile)
    low_sal = stick <= threshold

    # 3. 在图像空间标记粘连点
    junction_mask = np.zeros((H, W), dtype=bool)
    for k in np.where(low_sal)[0]:
        r, c = int(round(tv['points'][k][0])), int(round(tv['points'][k][1]))
        if 0 <= r < H and 0 <= c < W:
            junction_mask[r, c] = True

    # 4. 膨胀粘连区域
    if expand_radius > 0:
        junction_mask = _expand_branch_region(junction_mask, radius=expand_radius)

    # 5. 在清洗后的 mask 上切除粘连点
    split_mask = cleaned_mask & (~junction_mask)
    comps = _connected_components_points(split_mask)
    clusters = []
    for comp in comps:
        if len(comp) >= 2:
            clusters.append(_compute_stats(comp, H, W,
                                           n_junctions=int(junction_mask.sum()),
                                           n_removed_noise=n_removed))

    # 6. 重新分配被切除的点
    if junction_mask.any() and clusters:
        clusters = _reassign_removed_points(cleaned_mask, junction_mask, clusters, H, W)

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ============================================================
# 方法 4: TV Direction-Aware Clustering（方向连续性聚类）
# ============================================================

def tv_direction_cluster(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sigma: float = 5.0,
    noise_ratio: float = 0.3,
    angle_threshold: float = 45.0,
    spatial_radius: float = 3.0,
    **kwargs
) -> List[Dict]:
    """
    TV 噪声清洗 → 方向感知连通域聚类。

    原理：
      1. TV 计算 + 噪声过滤（保留结构显著的点）
      2. 在保留点上用主方向的连续性做图聚类

    Args:
        defect_mask: (H, W) bool
        sigma:       TV 尺度参数
        noise_ratio: 噪声过滤阈值（论文默认 0.3）
        angle_threshold: 方向一致性阈值（度）
        spatial_radius:  空间邻域半径

    Returns:
        List[Dict] clusters
    """
    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()

    if mask.sum() < 3:
        return skeleton_split(mask, valid_mask=None)

    # 1. TV 计算 + 噪声清洗
    _, tv = _tv_cleaned_mask(mask, sigma=sigma, noise_ratio=noise_ratio)
    points = tv['points']
    N = len(points)
    if N < 2:
        return skeleton_split(mask, valid_mask=None)

    # 构建点索引映射
    coord_to_idx = {}
    for k, (r, c) in enumerate(points):
        coord_to_idx[(int(round(r)), int(round(c)))] = k

    # 对每个点，找方向一致的邻域点作为图边
    import math
    angle_th_rad = math.radians(angle_threshold)
    cos_th = math.cos(angle_th_rad)

    # 构建邻接图
    adj = {i: set() for i in range(N)}
    directions = tv['principal_dir']

    for i in range(N):
        ri, ci = points[i]
        dir_i = directions[i]
        if np.linalg.norm(dir_i) < 1e-8:
            continue
        dir_i = dir_i / (np.linalg.norm(dir_i) + 1e-10)

        # 搜索空间邻域
        for dr in range(-int(spatial_radius), int(spatial_radius) + 1):
            for dc in range(-int(spatial_radius), int(spatial_radius) + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(round(ri + dr)), int(round(ci + dc))
                key = (nr, nc)
                if key not in coord_to_idx:
                    continue
                j = coord_to_idx[key]
                dir_j = directions[j]
                if np.linalg.norm(dir_j) < 1e-8:
                    continue
                dir_j = dir_j / (np.linalg.norm(dir_j) + 1e-10)

                # 方向一致性检查
                dot = abs(np.dot(dir_i, dir_j))  # |cos| 因为切线方向 ±180° 等价
                if dot >= cos_th:
                    adj[i].add(j)
                    adj[j].add(i)

    # DFS 聚类（方向一致的连通分量）
    visited = np.zeros(N, dtype=bool)
    clusters = []

    for i in range(N):
        if visited[i]:
            continue
        # BFS
        queue = [i]
        visited[i] = True
        comp_indices = []
        while queue:
            cur = queue.pop(0)
            comp_indices.append(cur)
            for nbr in adj[cur]:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)

        if len(comp_indices) >= 2:
            pts = points[comp_indices]
            clusters.append(_compute_stats(pts, H, W))
        else:
            # 孤立点：尝试合并到最近的已存在 cluster
            pass

    # 将未被任何 cluster 包含的点分配回最近 cluster
    unassigned = points[visited == False] if (~visited).any() else np.empty((0, 2))
    if len(unassigned) > 0 and clusters:
        for p in unassigned:
            best_c = 0
            best_d = float('inf')
            for ci, cl in enumerate(clusters):
                cx, cy = cl['centroid_row'], cl['centroid_col']
                d = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
                if d < best_d:
                    best_d = d
                    best_c = ci
            r, c = int(p[0]), int(p[1])
            clusters[best_c]['pixels'].append((r, c))
            clusters[best_c]['pixel_coords'].append({'row': r, 'col': c})
            clusters[best_c]['area'] += 1

    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ============================================================
# 方法 5: TV Hybrid — 先分类（curve/region）再分别聚类
# ============================================================

def tv_hybrid_cluster(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sigma: float = 5.0,
    noise_ratio: float = 0.3,
    curve_ratio: float = 0.3,
    **kwargs
) -> List[Dict]:
    """
    TV 噪声清洗 → saliency 分类（curve/region）→ 分别聚类 → 合并。

    流程：
      1. TV 计算 + 噪声过滤（保留结构显著的点）
      2. 分类：stick 主导 → curve，ball 主导 → region
      3. curve 点：用方向感知聚类
      4. region 点：用 watershed
      5. 合并结果

    Args:
        defect_mask: (H, W) bool
        sigma:       TV 尺度参数
        noise_ratio: 噪声过滤阈值（论文默认 0.3）
        curve_ratio: stick/(stick+ball) 大于此值为 curve 点

    Returns:
        List[Dict] clusters
    """
    H, W = defect_mask.shape
    mask = defect_mask & valid_mask if valid_mask is not None else defect_mask.copy()

    if mask.sum() < 3:
        return skeleton_split(mask, valid_mask=None)

    # 1. TV 计算 + 噪声清洗
    cleaned_mask, tv = _tv_cleaned_mask(mask, sigma=sigma, noise_ratio=noise_ratio)
    points = tv['points']
    N = len(points)
    if N < 2:
        return skeleton_split(mask, valid_mask=None)

    stick = tv['stick_sal']
    ball = tv['ball_sal']
    total = stick + ball

    # 避免除零
    total_safe = np.where(total > 1e-10, total, 1e-10)
    stick_ratio = stick / total_safe

    is_curve = stick_ratio >= curve_ratio
    is_region = ~is_curve

    all_clusters = []

    # Curve 点 → 方向感知聚类
    if is_curve.sum() >= 2:
        curve_mask = np.zeros((H, W), dtype=bool)
        for k in np.where(is_curve)[0]:
            r, c = int(round(points[k][0])), int(round(points[k][1]))
            if 0 <= r < H and 0 <= c < W:
                curve_mask[r, c] = True

        curve_clusters = tv_direction_cluster(
            curve_mask, valid_mask=None, sigma=sigma, noise_ratio=0.0, **kwargs
        )
        for cl in curve_clusters:
            cl['type'] = 'curve'
        all_clusters.extend(curve_clusters)

    # Region 点 → watershed
    if is_region.sum() >= 2:
        region_mask = np.zeros((H, W), dtype=bool)
        for k in np.where(is_region)[0]:
            r, c = int(round(points[k][0])), int(round(points[k][1]))
            if 0 <= r < H and 0 <= c < W:
                region_mask[r, c] = True

        region_clusters = watershed_split(region_mask, valid_mask=None, min_distance=3)
        for cl in region_clusters:
            cl['type'] = 'region'
        all_clusters.extend(region_clusters)

    # 如果没有成功聚类，退回到连通域
    if not all_clusters:
        comps = _connected_components_points(mask)
        all_clusters = [_compute_stats(c, H, W) for c in comps]

    # 合并太小的碎片
    all_clusters.sort(key=lambda c: c['area'], reverse=True)
    return all_clusters


# ============================================================
# 公共入口
# ============================================================

def adhesion_cluster(
    defect_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    method: str = 'skeleton',
    **kwargs
) -> List[Dict]:
    """
    粘连分离统一入口。

    Args:
        defect_mask: (H, W) bool
        valid_mask:  (H, W) bool
        method:      'skeleton', 'watershed', 'tv_junction',
                     'tv_direction', 'tv_hybrid'

    Returns:
        List[Dict] clusters
    """
    method = method.lower().strip()
    if method == 'skeleton':
        return skeleton_split(defect_mask, valid_mask, **kwargs)
    elif method == 'watershed':
        return watershed_split(defect_mask, valid_mask, **kwargs)
    elif method == 'tv_junction':
        return tv_junction_split(defect_mask, valid_mask, **kwargs)
    elif method == 'tv_direction':
        return tv_direction_cluster(defect_mask, valid_mask, **kwargs)
    elif method == 'tv_hybrid':
        return tv_hybrid_cluster(defect_mask, valid_mask, **kwargs)
    else:
        raise ValueError(
            f"Unknown adhesion method: {method}. "
            f"Use 'skeleton', 'watershed', 'tv_junction', 'tv_direction', or 'tv_hybrid'."
        )
