# Pluggable similarity methods for WBM-WDM matching.
# Each method takes two same-shape numpy arrays and returns a float score.
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
    """所有相似度方法的基类。均为逐像素对齐计算，不包含几何不变性。"""

    name: str = "base"

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class SimilarityResult:
    """单次相似度计算的完整结果，包含可解释分量。"""

    score: float
    coverage: float | None = None
    leakage: float | None = None
    method: str = ""


# ── 基础像素/网格相似度 ──────────────────────────────────────────


class DiceSimilarity(SimilarityMethod):
    name = "dice"

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        a = (reference > 0).astype(np.float32)
        b = (candidate > 0).astype(np.float32)
        intersection = (a * b).sum()
        denominator = a.sum() + b.sum()
        if denominator == 0:
            return 1.0
        return float(2.0 * intersection / denominator)


class IoUSimilarity(SimilarityMethod):
    name = "iou"

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        a = (reference > 0).astype(np.float32)
        b = (candidate > 0).astype(np.float32)
        intersection = (a * b).sum()
        union = ((a + b) > 0).sum()
        if union == 0:
            return 1.0
        return float(intersection / union)


class NccSimilarity(SimilarityMethod):
    name = "ncc"

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        a = reference.astype(np.float32).ravel()
        b = candidate.astype(np.float32).ravel()
        a_mean = a.mean()
        b_mean = b.mean()
        num = float(((a - a_mean) * (b - b_mean)).sum())
        den = float(np.sqrt(((a - a_mean) ** 2).sum()) * np.sqrt(((b - b_mean) ** 2).sum()))
        if den == 0:
            return 1.0 if num == 0 else 0.0
        return num / den


class CosineSimilarity(SimilarityMethod):
    name = "cosine"

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        a = reference.astype(np.float32).ravel()
        b = candidate.astype(np.float32).ravel()
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 1.0 if norm_a == norm_b else 0.0
        return float(np.dot(a, b)) / (norm_a * norm_b)


# ── 点集距离 ──────────────────────────────────────────────────────


class ChamferSimilarity(SimilarityMethod):
    name = "chamfer"

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        a_coords = np.argwhere(reference > 0).astype(np.float32)
        b_coords = np.argwhere(candidate > 0).astype(np.float32)
        if len(a_coords) == 0 and len(b_coords) == 0:
            return 1.0
        if len(a_coords) == 0 or len(b_coords) == 0:
            return 0.0

        # 纯 NumPy 最近邻距离（小网格下足够快）
        diff = a_coords[:, None, :] - b_coords[None, :, :]
        dists_ab = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
        dists_ba = np.sqrt((diff**2).sum(axis=2)).min(axis=0)

        chamfer = float(np.mean(dists_ab) + np.mean(dists_ba))
        max_dim = float(np.linalg.norm(reference.shape))
        return 1.0 - min(chamfer / max_dim, 1.0)


# ── Coverage-Leakage 分数 ─────────────────────────────────────────


class CoverageSimilarity(SimilarityMethod):
    name = "coverage"

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        a = (reference > 0).astype(np.float32)
        b = (candidate > 0).astype(np.float32)
        matched = (a * b).sum()
        total_a = a.sum()
        if total_a == 0:
            return 1.0
        return float(matched / total_a)


class LeakagePenalty(SimilarityMethod):
    name = "leakage"

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        a = (reference > 0).astype(np.float32)
        b = (candidate > 0).astype(np.float32)
        leakage = ((1 - a) * b).sum()
        total_b = b.sum()
        if total_b == 0:
            return 0.0
        return float(leakage / total_b)


class CoverageLeakageScore(SimilarityMethod):
    """Coverage - beta × Leakage，方案中推荐的第一版简化得分。"""

    name = "coverage-leakage"

    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def compute(self, reference: np.ndarray, candidate: np.ndarray) -> SimilarityResult:
        a = (reference > 0).astype(np.float32)
        b = (candidate > 0).astype(np.float32)

        matched = float((a * b).sum())
        coverage = matched / (float(a.sum()) + 1e-12)
        leakage = float(((1 - a) * b).sum()) / (float(b.sum()) + 1e-12)

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
    reference: np.ndarray,
    candidate: np.ndarray,
    method: str = "coverage",
) -> float | SimilarityResult:
    """单一相似度计算的便捷入口。"""
    sim = SIMILARITIES[method]
    return sim.compute(reference, candidate)


def top_k_retrieval(
    reference: np.ndarray,
    candidates: Dict[str, np.ndarray],
    method: str = "coverage",
    k: int = 5,
) -> list[SimilarityResult]:
    """从候选 WDM map 中检索 top-k。返回 SimilarityResult 列表（降序）。"""
    sim = SIMILARITIES[method]
    results: list[SimilarityResult] = []
    for name, candidate in candidates.items():
        result = sim.compute(reference, candidate)
        if isinstance(result, float):
            result = SimilarityResult(score=result, method=method)
        results.append(result)
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]
