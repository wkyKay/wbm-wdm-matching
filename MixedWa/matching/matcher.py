# -*- coding: utf-8 -*-
"""
WBM-WDM 匹配推理模块。

匹配得分 = α × 重叠率 + β × 位置相似度 + γ × 面积相似度
过滤条件：重叠率 ≥ θ

- 重叠率：|S_wdm ∩ S_wbm| / |S_wdm|，衡量 pattern 类型一致性
- 位置相似度：cosine(z_wbm, z_wdm)，由位置感知训练的 embedding 隐含位置信息
- 面积相似度：1 - |area_wbm_i - area_wdm_i| / max(...)，衡量 pattern 大小一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# 面积估算
# ---------------------------------------------------------------------------

def estimate_pattern_areas(probs: torch.Tensor,
                            binary_map: torch.Tensor,
                            threshold: float = 0.5) -> Dict[int, float]:
    """
    估算每个激活 pattern 类别的像素面积占比。

    Args:
        probs:      (C,) sigmoid 输出，用于确定激活类别
        binary_map: (1, H, W) 或 (2, H, W) 的 tensor，用于估算总激活面积
        threshold:  分类阈值

    Returns:
        {class_idx: area_ratio} 字典，area_ratio ∈ [0, 1]
    """
    active_classes = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    if len(active_classes) == 0:
        return {}

    # 使用 existence mask（第 1 通道）或单通道估算总激活面积
    if binary_map.shape[0] == 2:
        mask = binary_map[1]  # existence mask
    else:
        mask = binary_map[0]

    total_pixels = mask.numel()
    active_pixels = mask.gt(0).sum().item()
    total_area = active_pixels / total_pixels if total_pixels > 0 else 0.0

    # 将总面积均分给各激活类别（轻量近似）
    area_per_class = total_area / len(active_classes) if active_classes else 0.0
    return {c: area_per_class for c in active_classes}


def area_similarity(areas_wbm: Dict[int, float],
                    areas_wdm: Dict[int, float]) -> float:
    """
    计算共同 pattern 类别的面积相似度均值。
    对每个共同类别：sim = 1 - |a_wbm - a_wdm| / max(a_wbm, a_wdm)
    """
    common = set(areas_wbm.keys()) & set(areas_wdm.keys())
    if not common:
        return 0.0

    sims = []
    for c in common:
        a1, a2 = areas_wbm[c], areas_wdm[c]
        denom = max(a1, a2)
        if denom < 1e-8:
            sims.append(1.0)
        else:
            sims.append(1.0 - abs(a1 - a2) / denom)
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# 匹配得分
# ---------------------------------------------------------------------------

def match_score(z_wbm: torch.Tensor,
                z_wdm: torch.Tensor,
                s_wbm: set,
                s_wdm: set,
                areas_wbm: Dict[int, float] = None,
                areas_wdm: Dict[int, float] = None,
                alpha: float = 0.6,
                beta: float = 0.2,
                gamma: float = 0.2,
                theta: float = 0.6) -> float:
    """
    计算单对 (WBM, WDM) 的匹配得分。

    Args:
        z_wbm, z_wdm: (D,) L2 归一化 embedding
        s_wbm, s_wdm: pattern 类别集合（int 集合）
        areas_wbm, areas_wdm: {class_idx: area_ratio}，可为 None（跳过面积项）
        alpha: 重叠率权重
        beta:  位置相似度权重
        gamma: 面积相似度权重（alpha + beta + gamma 应 = 1）
        theta: 重叠率过滤阈值

    Returns:
        匹配得分 ∈ [0, 1]，0 表示被过滤
    """
    if len(s_wdm) == 0:
        return 0.0

    overlap_ratio = len(s_wdm & s_wbm) / len(s_wdm)
    if overlap_ratio < theta:
        return 0.0

    pos_score = F.cosine_similarity(
        z_wbm.unsqueeze(0), z_wdm.unsqueeze(0)
    ).item()
    # cosine 值域 [-1, 1]，归一化到 [0, 1]
    pos_score = (pos_score + 1.0) / 2.0

    if areas_wbm is not None and areas_wdm is not None:
        area_score = area_similarity(areas_wbm, areas_wdm)
        score = alpha * overlap_ratio + beta * pos_score + gamma * area_score
    else:
        # 无面积信息时，将权重重新分配给前两项
        total = alpha + beta
        score = (alpha / total) * overlap_ratio + (beta / total) * pos_score

    return float(score)


# ---------------------------------------------------------------------------
# 批量匹配器
# ---------------------------------------------------------------------------

class WaferMatcher:
    """
    给定一张 WBM 和一批 WDM，输出 top-k 匹配结果。
    """
    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module,
                 device: str = 'cpu',
                 alpha: float = 0.6,
                 beta: float = 0.2,
                 gamma: float = 0.2,
                 theta: float = 0.6,
                 cls_threshold: float = 0.5):
        """
        Args:
            backbone:      训练好的 encoder
            classifier:    多标签分类头（8 类）
            alpha/beta/gamma: 重叠率/位置/面积权重
            theta:         重叠率过滤阈值
            cls_threshold: 多标签分类阈值
        """
        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.cls_threshold = cls_threshold

        self.backbone.eval()
        self.classifier.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单张或批量图像编码。
        Returns:
            z:     (B, D) L2 归一化 embedding
            probs: (B, C) sigmoid 概率
            logits:(B, C) 原始 logits
        """
        x = x.to(self.device)
        feat = self.backbone(x)
        logits = self.classifier(feat)
        probs = torch.sigmoid(logits)
        z = F.normalize(feat, dim=1)
        return z, probs, logits

    def match(self,
              wbm_tensor: torch.Tensor,
              wdm_tensors: torch.Tensor,
              wbm_map: torch.Tensor = None,
              wdm_maps: torch.Tensor = None,
              top_k: int = 3) -> List[Dict]:
        """
        Args:
            wbm_tensor:  (1, C, H, W) WBM 输入
            wdm_tensors: (N, C, H, W) WDM 输入批量
            wbm_map:     (1, C, H, W) WBM 原始 tensor（用于面积估算），可为 None
            wdm_maps:    (N, C, H, W) WDM 原始 tensor（用于面积估算），可为 None
            top_k:       返回前 k 个匹配结果

        Returns:
            List of dicts，按得分降序排列：
            [{'wdm_idx': int, 'score': float, 'overlap': float,
              'pos_sim': float, 'area_sim': float, 's_wdm': set}, ...]
        """
        # 编码 WBM
        z_wbm, probs_wbm, _ = self.encode(wbm_tensor)
        z_wbm = z_wbm.squeeze(0)   # (D,)
        probs_wbm = probs_wbm.squeeze(0)  # (C,)
        s_wbm = set((probs_wbm > self.cls_threshold).nonzero(as_tuple=True)[0].tolist())

        # 面积估算（WBM）
        areas_wbm = None
        if wbm_map is not None:
            areas_wbm = estimate_pattern_areas(probs_wbm, wbm_map.squeeze(0),
                                               self.cls_threshold)

        # 批量编码 WDM
        z_wdms, probs_wdms, _ = self.encode(wdm_tensors)  # (N, D), (N, C)

        results = []
        for i in range(len(wdm_tensors)):
            z_wdm_i = z_wdms[i]
            probs_wdm_i = probs_wdms[i]
            s_wdm_i = set((probs_wdm_i > self.cls_threshold).nonzero(as_tuple=True)[0].tolist())

            # 面积估算（WDM）
            areas_wdm_i = None
            if wdm_maps is not None:
                areas_wdm_i = estimate_pattern_areas(probs_wdm_i, wdm_maps[i],
                                                     self.cls_threshold)

            score = match_score(
                z_wbm, z_wdm_i, s_wbm, s_wdm_i,
                areas_wbm, areas_wdm_i,
                self.alpha, self.beta, self.gamma, self.theta,
            )

            if score > 0:
                overlap = len(s_wdm_i & s_wbm) / len(s_wdm_i) if s_wdm_i else 0.0
                pos_sim = ((F.cosine_similarity(z_wbm.unsqueeze(0),
                                                z_wdm_i.unsqueeze(0)).item() + 1) / 2)
                area_sim = (area_similarity(areas_wbm, areas_wdm_i)
                            if areas_wbm and areas_wdm_i else None)
                results.append({
                    'wdm_idx':  i,
                    'score':    score,
                    'overlap':  overlap,
                    'pos_sim':  pos_sim,
                    'area_sim': area_sim,
                    's_wdm':    s_wdm_i,
                })

        results.sort(key=lambda r: r['score'], reverse=True)
        return results[:top_k]
