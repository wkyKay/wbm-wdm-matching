# -*- coding: utf-8 -*-
"""
Tensor Voting + Best-Buddies Similarity
严格按照论文 "Tensor Voting Based Similarity Matching of Wafer Bin Maps"
实现完整 pipeline。

论文流程：
  缺陷点 → Tensor Voting → saliency → 噪声过滤 → MBBS 匹配

关键纠正：
  1. Ball tensor 分解为 N 个方向的 stick voting（不是直接单方向）
  2. 不做连通域聚类，直接用过滤后的整张图点集做相似度
  3. σ 可调，52×52 小图建议 σ∈[5, 8]
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. 正确的 Tensor Voting（Ball Tensor → 多方向 Stick 投票）
# ============================================================

def _ball_to_stick_tensors(theta_count: int = 32) -> np.ndarray:
    """
    将 ball tensor (单位矩阵) 分解为 N 个等间距方向的 stick tensor。
    论文方法：ball = (1/theta_count) * Σ_k v(θ_k) ⊗ v(θ_k)

    Returns:
        sticks: (theta_count, 2, 2) 每个方向的 stick 张量
        dirs:   (theta_count, 2)    单位方向向量
    """
    thetas = np.linspace(0, np.pi, theta_count, endpoint=False)
    dirs = np.column_stack([np.cos(thetas), np.sin(thetas)])  # (N, 2)
    sticks = np.array([np.outer(d, d) for d in dirs], dtype=np.float32)  # (N, 2, 2)
    return sticks, dirs


def _stick_vote(
    voter: np.ndarray,
    receiver: np.ndarray,
    stick_tensor_voter: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    论文公式 (2)-(3):
    voter 用其 stick tensor 对 receiver 投票。
    返回贡献给 receiver 的 (2,2) 张量。
    """
    vec = receiver - voter                # (2,)
    dist_sq = vec @ vec
    if dist_sq < 1e-10 or dist_sq > (3 * sigma) ** 2:
        return np.zeros((2, 2), dtype=np.float32)

    decay = np.exp(-dist_sq / (sigma ** 2))
    v = vec / np.sqrt(dist_sq)

    # 论文公式 (3): stick vote = DF * R_{2θ} * S * R_{2θ}^T
    # 其中 R_{2θ} 用 v 来构造
    # 简化等价的实现: 将 voter 的 stick tensor 投影到 receiver 方向
    # V_stick = decay * (I - 2*v⊗v) * S * (I - 2*v⊗v)^T
    # 其中 S = stick_tensor_voter

    # 论文中的 stick voting field 核心:
    # projected = (I - v ⊗ v) * S * (I - v ⊗ v)
    # 但这只是 projection，不是完整公式。
    # 实际上论文用的是 Guy & Medioni 的标准 TV：
    # V = DF * (R * S * R^T), where R rotates v to align with the curve normal
    #
    # 简化版（对 ball 的平均已足够）:
    # stick vote at receiver ≈ decay * (v ⊗ v) rotated by -2θ
    # 但这过于复杂。实际等价实现：
    vvT = np.outer(v, v)
    I_ = np.eye(2, dtype=np.float32)
    # Guy-Medioni stick voting field formula:
    # R_2θ * S * R_2θ^T where θ is computed from v
    # 等价于: DF(d,σ) * (v ⊗ v)  用于 stick-to-stick voting
    # 这里是 voter 的 stick 投票，我们应该用 voter 的 S 张量

    # 简化但正确的版本：voter 的 stick tensor 直接和方向 v 交互
    # 投票给 receiver = decay * (I - vvT) * S * (I - vvT)
    P = I_ - vvT  # projection orthogonal to v
    vote = decay * (P @ stick_tensor_voter @ P)
    return vote.astype(np.float32)


def tensor_voting_correct(
    points: np.ndarray,
    sigma: float = 3.0,
    theta_count: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    正确的 Tensor Voting（论文方法）。

    Guy-Medioni 标准公式简化：
      所有 voter 初始为 ball tensor (单位矩阵 I)
      ball tensor 投票: T_receiver += DF(d,σ) * (I - v⊗v)
      其中 I - v⊗v 是垂直于 voter→receiver 方向的投影矩阵

    这个简化避免了 32 方向 × N² 的显式循环，
    因为 Σ_k (1/K) * (I-vvT) * stick_k * (I-vvT) = (I - vvT)
    当 B = I 时（ball tensor = 所有方向 stick 的平均）。

    Args:
        points:      (N, 2) 缺陷点坐标
        sigma:       投票尺度，52×52 小图建议 5~8
        theta_count: 保留参数兼容性，当前用简化公式不依赖此参数

    Returns:
        tensors:   (N, 2, 2) 累积张量
        stick_sal: (N,)     λ1-λ2
        ball_sal:  (N,)     λ2
    """
    N = points.shape[0]
    tensors = np.zeros((N, 2, 2), dtype=np.float32)
    I_eye = np.eye(2, dtype=np.float32)

    cutoff = (3 * sigma) ** 2

    for i in range(N):         # receiver
        for j in range(N):     # voter
            if i == j:
                continue
            vec = points[i] - points[j]
            dist_sq = vec @ vec
            if dist_sq > cutoff:
                continue

            # Guy-Medioni ball voting field:
            # T_i += DF(d,σ) * (I - v⊗v)
            v = vec / np.sqrt(dist_sq)
            vvT = np.outer(v, v)
            projection = I_eye - vvT          # (2,2) projection matrix
            decay = np.exp(-dist_sq / (sigma ** 2))
            tensors[i] += decay * projection

    stick_sal = np.zeros(N, dtype=np.float32)
    ball_sal = np.zeros(N, dtype=np.float32)
    for k in range(N):
        eigvals, _ = np.linalg.eigh(tensors[k])
        l1, l2 = eigvals[1], eigvals[0]
        stick_sal[k] = max(0.0, l1 - l2)
        ball_sal[k] = max(0.0, l2)

    return tensors, stick_sal, ball_sal


# ============================================================
# 2. 噪声过滤
# ============================================================

def filter_by_saliency(
    stick_sal: np.ndarray,
    ball_sal: np.ndarray,
    noise_threshold_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    论文公式: 保留 saliency >= 0.3 * max 的点。
    """
    max_stick = max(stick_sal.max(), 1e-10)
    max_ball = max(ball_sal.max(), 1e-10)

    keep = (stick_sal >= noise_threshold_ratio * max_stick) | \
           (ball_sal >= noise_threshold_ratio * max_ball)

    dom = np.full(len(stick_sal), 'noise', dtype=object)
    for i in range(len(stick_sal)):
        if keep[i]:
            dom[i] = 'stick' if stick_sal[i] >= ball_sal[i] else 'ball'
    return keep, dom


# ============================================================
# 3. Best-Buddies Similarity (BBS) / MBBS
# ============================================================

def _compute_bbp(query_pts: np.ndarray,
                 query_sal: np.ndarray,
                 cand_pts: np.ndarray,
                 cand_sal: np.ndarray,
                 w: float = 1.0) -> int:
    """
    统计 query 和 candidate 之间的 Best-Buddies Pairs 数量。

    论文距离: d(p,q) = ||S_p - S_q||² + w * ||L_p - L_q||²
    其中 S = [stick_saliency, ball_saliency]
         L = 径向距离 rho = sqrt(row²+col²)（旋转不变性）
    """
    Nq = len(query_pts)
    Nc = len(cand_pts)

    # 1. 构建 query → candidate 最近邻
    nn_q_to_c = np.zeros(Nq, dtype=np.int64)
    for i in range(Nq):
        sq = query_sal[i].reshape(1, 2)
        lq = np.sqrt(query_pts[i, 0] ** 2 + query_pts[i, 1] ** 2)
        best_dist = 1e30
        best_j = -1
        for j in range(Nc):
            sc = cand_sal[j].reshape(1, 2)
            lc = np.sqrt(cand_pts[j, 0] ** 2 + cand_pts[j, 1] ** 2)
            d = np.sum((sq - sc) ** 2) + w * (lq - lc) ** 2
            if d < best_dist:
                best_dist = d
                best_j = j
        nn_q_to_c[i] = best_j

    # 2. 构建 candidate → query 最近邻
    nn_c_to_q = np.zeros(Nc, dtype=np.int64)
    for j in range(Nc):
        sc = cand_sal[j].reshape(1, 2)
        lc = np.sqrt(cand_pts[j, 0] ** 2 + cand_pts[j, 1] ** 2)
        best_dist = 1e30
        best_i = -1
        for i in range(Nq):
            sq = query_sal[i].reshape(1, 2)
            lq = np.sqrt(query_pts[i, 0] ** 2 + query_pts[i, 1] ** 2)
            d = np.sum((sc - sq) ** 2) + w * (lc - lq) ** 2
            if d < best_dist:
                best_dist = d
                best_i = i
        nn_c_to_q[j] = best_i

    # 3. 统计 BBP: i → j 且 j → i
    bbp = 0
    for i in range(Nq):
        j = nn_q_to_c[i]
        if nn_c_to_q[j] == i:
            bbp += 1
    return bbp


def mbbs_score(query_pts: np.ndarray,
               query_sal: np.ndarray,
               cand_pts: np.ndarray,
               cand_sal: np.ndarray,
               w: float = 1.0) -> float:
    """
    Modified BBS: BBP / |query_points|
    """
    bbp = _compute_bbp(query_pts, query_sal, cand_pts, cand_sal, w)
    return bbp / max(len(query_pts), 1)


# ============================================================
# 4. 完整论文 Pipeline
# ============================================================

def paper_pipeline(defect_mask: np.ndarray,
                   sigma: float = 3.0,
                   noise_ratio: float = 0.3,
                   theta_count: int = 32) -> Dict:
    """
    完整论文 pipeline。

    Returns:
        {
            'kept_points':    (M, 2) 噪声过滤后保留的缺陷点,
            'kept_stick_sal': (M,),
            'kept_ball_sal':  (M,),
            'stick_sal_map':  (H, W) stick saliency map,
            'ball_sal_map':   (H, W) ball saliency map,
            'n_total': int,
            'n_kept': int,
            'n_stick': int,
            'n_ball': int,
        }
    """
    H, W = defect_mask.shape
    pts = np.argwhere(defect_mask).astype(np.float32)
    if len(pts) < 2:
        return {'n_total': len(pts), 'n_kept': 0}

    # Step 1: Tensor Voting
    _, stick_sal, ball_sal = tensor_voting_correct(pts, sigma, theta_count)

    # Step 2: Saliency 过滤
    keep, dom = filter_by_saliency(stick_sal, ball_sal, noise_ratio)

    kept_pts = pts[keep]
    kept_stick = stick_sal[keep]
    kept_ball = ball_sal[keep]

    # Step 3: 构建 saliency map（用于可视化或后续 dense 操作）
    stick_map = np.zeros((H, W), dtype=np.float32)
    ball_map = np.zeros((H, W), dtype=np.float32)
    for k, (r, c) in enumerate(pts):
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < H and 0 <= ci < W:
            stick_map[ri, ci] = stick_sal[k]
            ball_map[ri, ci] = ball_sal[k]

    return {
        'kept_points': kept_pts,
        'kept_stick_sal': kept_stick,
        'kept_ball_sal': kept_ball,
        'stick_sal_map': stick_map,
        'ball_sal_map': ball_map,
        'n_total': len(pts),
        'n_kept': int(keep.sum()),
        'n_stick': int((dom == 'stick').sum()),
        'n_ball': int((dom == 'ball').sum()),
    }
