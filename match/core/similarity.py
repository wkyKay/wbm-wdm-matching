# Pluggable similarity methods for WBM-WDM matching.
#
# 所有方法现在接受 (map, status) 对，以 WBM 的 status_map 定义"有意义区域"：
#   - WBM status = VALID_HAS_DEFECT / VALID_NO_DEFECT → 参与计算
#   - WBM status = BACKGROUND / UNINSPECTED             → 忽略
#
# 当前所有方法均为逐像素严格对齐的简单相似度，不具备以下几何不变性：
#   - 平移不变（translation invariance）：位置偏移会导致得分下降。
#   - 旋转不变（rotation invariance）：wafer 旋转后无法匹配。
#   - 缩放不变（scale invariance）：已被上游 mapper 的坐标归一化解决。
# 小幅平移可由 SoftMap / MountainMap 等带邻域扩散的表达间接缓解；
# 旋转容错（如小角度旋转搜索）待后续按需加入。
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


class SimilarityMethod:
    """所有相似度方法的基类。"""

    name: str = "base"

    def compute(
        self,
        reference_map: np.ndarray,
        candidate_map: np.ndarray,
        reference_status: np.ndarray | None = None,
        candidate_status: np.ndarray | None = None,
    ) -> float | "SimilarityResult":
        raise NotImplementedError


@dataclass(frozen=True)
class SimilarityResult:
    """单次相似度计算的完整结果，包含可解释分量。"""

    score: float
    coverage: float | None = None
    leakage: float | None = None
    method: str = ""


# ── 辅助函数 ──────────────────────────────────────────────────────


def _meaningful_mask(reference_status: np.ndarray | None, shape: tuple | None = None) -> np.ndarray:
    """以 WBM status_map 为准：非背景、非未检测的像素才有意义。"""
    if reference_status is None:
        if shape is None:
            raise ValueError("Must provide either reference_status or shape")
        return np.ones(shape, dtype=np.bool_)
    return (reference_status >= 1) & (reference_status <= 2)


def _ref_defect_mask(reference_map: np.ndarray, reference_status: np.ndarray | None) -> np.ndarray:
    """WBM 中有缺陷的像素。优先用 status，否则用 map>0。"""
    if reference_status is not None:
        return reference_status == 2  # VALID_HAS_DEFECT
    return reference_map > 0


def _masked_float_maps(
    reference_map: np.ndarray,
    candidate_map: np.ndarray,
    reference_status: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """按 WBM 有意义区域裁剪后，保留 map 的连续权重。"""
    mask = _meaningful_mask(reference_status, reference_map.shape).astype(np.float32)
    a = reference_map.astype(np.float32) * mask
    b = candidate_map.astype(np.float32) * mask
    return a, b


# ── 基础像素/网格相似度 ──────────────────────────────────────────


class DiceSimilarity(SimilarityMethod):
    """Dice 系数（Dice Similarity Coefficient, DSC）。

    公式：
        Dice = 2 * |A ∩ B| / (|A| + |B|)

    其中交集使用逐像素 min(A, B) 计算，支持连续值权重图（density map 等）。
    当 reference 与 candidate 完全匹配时得分为 1.0；完全不重叠时得分为 0.0。
    两个 map 同时为零时返回 1.0（完美匹配）。

    特点：
        - 对交集给予 2 倍权重，比 IoU 更"宽容"——小目标重叠也能获得中等分数。
        - 对连续值 map：捕获权重分布的整体重叠程度，而非二值化的精确像素匹配。
        - 无几何不变性：要求 wafer 严格对齐。

    适用范围：需要平衡覆盖与漏检的通用场景，是最常用的二值/连续图相似度指标之一。
    """

    name = "dice"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        intersection = np.minimum(a, b).sum()
        denominator = a.sum() + b.sum()
        if denominator == 0:
            return 1.0
        return float(2.0 * intersection / denominator)


class IoUSimilarity(SimilarityMethod):
    """交并比（Intersection over Union, IoU / Jaccard Index）。

    公式：
        IoU = |A ∩ B| / |A ∪ B|

    其中交集使用逐像素 min(A, B)、并集使用逐像素 max(A, B) 计算，支持连续值权重图。
    得分为 1.0 表示完美匹配；0.0 表示完全不重叠。两个 map 同时为零时返回 1.0。

    特点：
        - 比 Dice 更"严格"：并集受双方共同影响，任意一方扩张都会拉低得分。
        - 对缺陷数量/面积差异敏感：candidate 多出缺陷会导致并集变大、得分下降。
        - 对连续值 map 同样适用，衡量的是权重分布的重叠占比。

    适用范围：需要严格控制漏检（candidate 多报会立即惩罚）的场景。
    """

    name = "iou"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        intersection = np.minimum(a, b).sum()
        union = np.maximum(a, b).sum()
        if union == 0:
            return 1.0
        return float(intersection / union)


class NccSimilarity(SimilarityMethod):
    """归一化互相关（Normalized Cross-Correlation, NCC）。

    公式：
        NCC = Σ[(A_i - μ_A)(B_i - μ_B)] / (σ_A * σ_B)

    将两个 map 在"有意义区域"内的像素拉平为向量，减去各自均值后计算余弦相似度
    （等价于 Pearson 相关系数）。

    特点：
        - 对整体亮度/偏移不敏感：减均值操作使得两个 map 的绝对数值大小不影响得分。
          例如 reference = [1, 2, 3] 与 candidate = [10, 20, 30] 的 NCC = 1.0。
        - 衡量的是"缺陷分布模式"的线性相关性，而非像素级重叠。
        - 对稀疏或均匀分布的两个 map 区分力较弱（因为减均值后信号被削弱）。

    得分范围 [-1, 1]，1.0 表示完全正相关。无有意义像素时返回 0.0。

    适用范围：关注缺陷空间分布模式而非绝对数量的场景；容忍信号强度差异。
    """

    name = "ncc"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        mask = _meaningful_mask(reference_status, reference_map.shape)
        if not mask.any():
            return 0.0
        a = reference_map.astype(np.float32).ravel()
        b = candidate_map.astype(np.float32).ravel()
        m = mask.ravel()
        a = a[m]
        b = b[m]
        a_mean = a.mean()
        b_mean = b.mean()
        num = float(((a - a_mean) * (b - b_mean)).sum())
        den_a = float(np.sqrt(((a - a_mean) ** 2).sum()))
        den_b = float(np.sqrt(((b - b_mean) ** 2).sum()))
        den = den_a * den_b
        if den == 0:
            return 1.0 if num == 0 else 0.0
        return num / den


class CosineSimilarity(SimilarityMethod):
    """余弦相似度（Cosine Similarity）。

    公式：
        cos(θ) = (A · B) / (||A|| * ||B||)

    将两个 map 在"有意义区域"内的像素拉平为向量，直接计算夹角余弦（不去均值）。

    特点：
        - 对向量长度（总权重大小）不敏感，只关心方向（分布比例）。
          例如 reference = [1, 0, 1] 与 candidate = [2, 0, 2] 的 cosine = 1.0。
        - 与 NCC 的区别：NCC 减均值后衡量"模式相关"，cosine 不减去均值，衡量"方向一致"。
        - 当一个向量全零时，另一向量也全零返回 1.0，否则返回 0.0。

    得分范围 [0, 1]（因为 map 值非负），1.0 表示分布比例完全一致。

    适用范围：关注缺陷分布比例而非绝对数量的场景；可容忍整体强度缩放。
    """

    name = "cosine"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        mask = _meaningful_mask(reference_status, reference_map.shape)
        if not mask.any():
            return 0.0
        a = reference_map.astype(np.float32).ravel()
        b = candidate_map.astype(np.float32).ravel()
        m = mask.ravel()
        a = a[m]
        b = b[m]
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 1.0 if norm_a == norm_b else 0.0
        return float(np.dot(a, b)) / (norm_a * norm_b)


# ── 点集距离 ──────────────────────────────────────────────────────


class ChamferSimilarity(SimilarityMethod):
    """倒角距离相似度（Chamfer Distance Similarity）。

    先将两个二值缺陷图转化为点集坐标（缺陷像素位置），再计算双向倒角距离：

        CD(A, B) = mean_{a∈A}(min_{b∈B} ||a-b||) + mean_{b∈B}(min_{a∈A} ||b-a||)

    最终归一化得分：
        score = 1.0 - min(CD / max_dim, 1.0)

    其中 max_dim 为网格对角线长度，作为归一化上限。

    特点：
        - 基于几何距离而非像素重叠，因此对小幅平移有一定的鲁棒性。
        - 仅使用二值缺陷位置（有/无缺陷），忽略像素级权重（density）。
        - 计算复杂度为 O(n*m)，缺陷点极多时较慢。
        - 两边都没缺陷点 → 1.0；仅一边有 → 0.0。

    适用范围：容忍小幅空间偏移、关注缺陷位置整体分布的场景。
    """

    name = "chamfer"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        mask = _meaningful_mask(reference_status, reference_map.shape)
        ref_defect = _ref_defect_mask(reference_map, reference_status) & mask
        cnd_defect = (candidate_map > 0) & mask

        a_coords = np.argwhere(ref_defect).astype(np.float32)
        b_coords = np.argwhere(cnd_defect).astype(np.float32)
        if len(a_coords) == 0 and len(b_coords) == 0:
            return 1.0
        if len(a_coords) == 0 or len(b_coords) == 0:
            return 0.0

        diff = a_coords[:, None, :] - b_coords[None, :, :]
        dists_ab = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
        dists_ba = np.sqrt((diff**2).sum(axis=2)).min(axis=0)

        chamfer = float(np.mean(dists_ab) + np.mean(dists_ba))
        max_dim = float(np.linalg.norm(ref_defect.shape))
        return 1.0 - min(chamfer / max_dim, 1.0)


# ── Coverage-Leakage 分数 ─────────────────────────────────────────


class CoverageSimilarity(SimilarityMethod):
    """覆盖率（Coverage Rate）。

    公式：
        Coverage = |A ∩ B| / |A|

    即 reference 缺陷总权重中，被 candidate "覆盖"到的比例。交集使用逐像素 min(A, B)。

    特点：
        - 单向衡量：只看 reference 被覆盖了多少，不关心 candidate 多出来的部分。
        - 得分总是 0~1，reference 全零时返回 1.0。
        - 与 Leakage 配合使用：Coverage 看"找回率"，Leakage 看"多报率"。

    适用范围：关注"是否漏掉了真实缺陷"的场景。
    """

    name = "coverage"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        matched = np.minimum(a, b).sum()
        total_a = a.sum()
        if total_a == 0:
            return 1.0
        return float(matched / total_a)


class LeakagePenalty(SimilarityMethod):
    """泄漏控制得分（1 - Leakage Rate）。

    公式：
        score = 1.0 - (|B| - |A ∩ B|) / |B|
              = |A ∩ B| / |B|

    即 candidate 缺陷总权重中，被 reference 支持的比例。得分越高越好（1.0 = 无泄漏）。

    特点：
        - 1.0 = candidate 所有缺陷都在 reference 缺陷区内（零泄漏）。
        - 0.0 = candidate 缺陷全在 reference 无缺陷区（完全泄漏）。
        - 与 Coverage 配合：Coverage 看"找回率"，Leakage 看"精确率"。
        - candidate 全零时返回 1.0（无泄漏）。

    适用范围：关注"是否多报了不存在的缺陷"的场景。
    """

    name = "leakage"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        matched = np.minimum(a, b).sum()
        total_b = b.sum()
        if total_b == 0:
            return 1.0
        return float(matched / total_b)


class CoverageLeakageScore(SimilarityMethod):
    """Coverage-Leakage 综合得分。

    公式：
        score = Coverage - beta * Leakage
        其中 Coverage = |A ∩ B| / |A|,  Leakage = |B - (A ∩ B)| / |B|

    将"找回真实缺陷"与"惩罚多报缺陷"合成单一分数。

    特点：
        - beta 控制漏检与多报的权衡：beta < 1 偏向覆盖率（容忍多报），beta > 1 偏向漏检惩罚。
        - 默认 beta = 0.5，即 Coverage 权重是 Leakage 的两倍。
        - 返回 SimilarityResult（含 coverage 和 leakage 分量），便于分项分析。
        - 得分无下界：若 Leakage 极高且 Coverage 很低，可跌至负值。

    适用范围：需要单一可比较分数、且需同时关心覆盖与多报的排名场景。
    """

    name = "coverage-leakage"

    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        matched = float(np.minimum(a, b).sum())
        coverage = matched / (float(a.sum()) + 1e-12)
        leakage = (float(b.sum()) - matched) / (float(b.sum()) + 1e-12)

        score = coverage - self.beta * leakage
        return SimilarityResult(score=score, coverage=coverage, leakage=leakage, method=self.name)


# ── 注册表 ────────────────────────────────────────────────────────

SIMILARITIES: Dict[str, SimilarityMethod] = {
    DiceSimilarity.name: DiceSimilarity(),
    IoUSimilarity.name: IoUSimilarity(),
    NccSimilarity.name: NccSimilarity(),
    CosineSimilarity.name: CosineSimilarity(),
    CoverageSimilarity.name: CoverageSimilarity(),
    LeakagePenalty.name: LeakagePenalty(),
    CoverageLeakageScore.name: CoverageLeakageScore(),
    ChamferSimilarity.name: ChamferSimilarity(),
}


def compute_similarity(
    reference_map: np.ndarray,
    candidate_map: np.ndarray,
    method: str = "coverage",
    reference_status: np.ndarray | None = None,
    candidate_status: np.ndarray | None = None,
) -> float | SimilarityResult:
    """单一相似度计算的便捷入口。"""
    sim = SIMILARITIES[method]
    return sim.compute(reference_map, candidate_map, reference_status, candidate_status)


def top_k_retrieval(
    reference_map: np.ndarray,
    candidates: Dict[str, np.ndarray],
    method: str = "coverage",
    k: int = 5,
    reference_status: np.ndarray | None = None,
    candidates_status: Dict[str, np.ndarray] | None = None,
) -> list[SimilarityResult]:
    """从候选 WDM map 中检索 top-k。返回 SimilarityResult 列表（降序）。"""
    sim = SIMILARITIES[method]
    results: list[SimilarityResult] = []
    for name, candidate_map in candidates.items():
        cnd_status = candidates_status.get(name) if candidates_status else None
        result = sim.compute(reference_map, candidate_map, reference_status, cnd_status)
        if isinstance(result, float):
            result = SimilarityResult(score=result, method=method)
        results.append(result)
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]
