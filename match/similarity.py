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
    name = "dice"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        intersection = np.minimum(a, b).sum()
        denominator = a.sum() + b.sum()
        if denominator == 0:
            return 1.0
        return float(2.0 * intersection / denominator)


class IoUSimilarity(SimilarityMethod):
    name = "iou"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        intersection = np.minimum(a, b).sum()
        union = np.maximum(a, b).sum()
        if union == 0:
            return 1.0
        return float(intersection / union)


class NccSimilarity(SimilarityMethod):
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
    name = "coverage"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        matched = np.minimum(a, b).sum()
        total_a = a.sum()
        if total_a == 0:
            return 1.0
        return float(matched / total_a)


class LeakagePenalty(SimilarityMethod):
    """candidate 中未被 reference 支持的权重比例。

    有意义区域仍以 WBM status_map 为准；二值图下退化为原来的漏检惩罚。
    """

    name = "leakage"

    def compute(self, reference_map, candidate_map, reference_status=None, candidate_status=None):
        a, b = _masked_float_maps(reference_map, candidate_map, reference_status)
        matched = np.minimum(a, b).sum()
        total_b = b.sum()
        if total_b == 0:
            return 0.0
        return float((total_b - matched) / total_b)


class CoverageLeakageScore(SimilarityMethod):
    """Coverage - beta × Leakage，方案中推荐的第一版简化得分。"""

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
