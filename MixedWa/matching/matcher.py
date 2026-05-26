# -*- coding: utf-8 -*-
"""
WBM-WDM 匹配推理模块。

匹配得分 = α × 标签重叠率 + β × 形状相似度 + γ × 大小相似度
        + δ × 显式位置相似度 + ε × 局部特征相似度
过滤条件：重叠率 ≥ θ

- 重叠率：|S_wdm ∩ S_wbm| / |S_wdm|，衡量 pattern 类型一致性
- 形状相似度：共同 pattern 类别 CAM weak mask IoU 的均值
- 大小相似度：共同 pattern 类别 CAM weak mask 面积相似度的均值
- 显式位置相似度：共同 pattern 类别 CAM weak mask 质心距离相似度的均值
- 局部特征相似度：共同 pattern 类别 CAM 加权局部特征 cosine 的均值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

from matching.cam import CAMExtractor, compute_cam_similarity_components


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
                explicit_scores: Dict[str, float] = None,
                alpha: float = 0.6,
                beta: float = 0.2,
                gamma: float = 0.2,
                delta: float = 0.0,
                epsilon: float = 0.0,
                theta: float = 0.6) -> float:
    """
    计算单对 (WBM, WDM) 的匹配得分。

    Args:
        z_wbm, z_wdm: (D,) L2 归一化 embedding，仅作为无 CAM 时的局部特征兜底
        s_wbm, s_wdm: pattern 类别集合（int 集合）
        areas_wbm, areas_wdm: {class_idx: area_ratio}，可为 None（跳过面积项）
        explicit_scores: CAM 或 token 产生的显式 shape/size/position/local_feature 指标
        alpha/beta/gamma/delta/epsilon: label/shape/size/position/local_feature 权重
        theta: 重叠率过滤阈值

    Returns:
        匹配得分 ∈ [0, 1]，0 表示被过滤
    """
    if len(s_wdm) == 0:
        return 0.0

    overlap_ratio = len(s_wdm & s_wbm) / len(s_wdm)
    if overlap_ratio < theta:
        return 0.0

    global_score = F.cosine_similarity(
        z_wbm.unsqueeze(0), z_wdm.unsqueeze(0)
    ).item()
    # cosine 值域 [-1, 1]，归一化到 [0, 1]
    global_score = (global_score + 1.0) / 2.0

    explicit_scores = explicit_scores or {}
    shape_score = explicit_scores.get('shape_sim')
    size_score = explicit_scores.get('size_sim')
    position_score = explicit_scores.get('position_sim')
    local_feature_score = explicit_scores.get('local_feature_sim')

    # 未启用 CAM/token 时，保留全局 embedding cosine 作为 local_feature 的弱兜底，
    # 但不再把它解释为位置相似度。
    if local_feature_score is None:
        local_feature_score = global_score

    if size_score is None and areas_wbm is not None and areas_wdm is not None:
        size_score = area_similarity(areas_wbm, areas_wdm)

    terms = [(alpha, overlap_ratio)]
    if shape_score is not None and beta > 0:
        terms.append((beta, shape_score))
    if size_score is not None and gamma > 0:
        terms.append((gamma, size_score))
    if position_score is not None and delta > 0:
        terms.append((delta, position_score))
    if local_feature_score is not None and epsilon > 0:
        terms.append((epsilon, local_feature_score))

    total = sum(weight for weight, _ in terms)
    if total <= 0:
        return 0.0
    score = sum((weight / total) * value for weight, value in terms)
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
                 delta: float = 0.0,
                 epsilon: float = 0.0,
                 theta: float = 0.6,
                 cls_threshold: float = 0.5,
                 use_cam: bool = False,
                 cam_lambda: float = 0.5,
                 cam_threshold: float = 0.5,
                 cam_min_area: float = 0.005,
                 cam_classes: str = 'common'):
        """
        Args:
            backbone:      训练好的 encoder
            classifier:    多标签分类头（8 类）
            alpha/beta/gamma/delta/epsilon: 标签/形状/大小/位置/局部特征权重
            theta:         重叠率过滤阈值
            cls_threshold: 多标签分类阈值
        """
        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.theta = theta
        self.cls_threshold = cls_threshold
        self.use_cam = use_cam
        self.cam_lambda = cam_lambda
        self.cam_threshold = cam_threshold
        self.cam_min_area = cam_min_area
        self.cam_classes = cam_classes
        self.cam_extractor = None

        self.backbone.eval()
        self.classifier.eval()

        if self.use_cam:
            self.cam_extractor = CAMExtractor(
                self.backbone,
                self.classifier,
                device=self.device,
                cam_threshold=self.cam_threshold,
                cam_min_area=self.cam_min_area,
            )

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
              'shape_sim': float, 'size_sim': float, 'position_sim': float,
              'local_feature_sim': float, 'global_sim': float, 's_wdm': set}, ...]
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

        cam_wbm = None
        if self.use_cam:
            if self.cam_classes == 'all':
                wbm_cam_classes = None
            else:
                wbm_cam_classes = sorted(s_wbm)
            cam_wbm = self.cam_extractor.compute_cam(wbm_tensor, wbm_cam_classes)

        results = []
        for i in range(len(wdm_tensors)):
            z_wdm_i = z_wdms[i]
            probs_wdm_i = probs_wdms[i]
            s_wdm_i = set((probs_wdm_i > self.cls_threshold).nonzero(as_tuple=True)[0].tolist())
            common_classes = s_wdm_i & s_wbm
            overlap = len(common_classes) / len(s_wdm_i) if s_wdm_i else 0.0
            if overlap < self.theta:
                continue

            # 面积估算（WDM）
            areas_wdm_i = None
            if wdm_maps is not None:
                areas_wdm_i = estimate_pattern_areas(probs_wdm_i, wdm_maps[i],
                                                     self.cls_threshold)

            explicit_scores = {}
            if self.use_cam and common_classes:
                if self.cam_classes == 'all':
                    wdm_cam_classes = None
                elif self.cam_classes == 'active':
                    wdm_cam_classes = sorted(s_wdm_i)
                else:
                    wdm_cam_classes = sorted(common_classes)

                cam_wdm = self.cam_extractor.compute_cam(
                    wdm_tensors[i].unsqueeze(0), wdm_cam_classes
                )
                explicit_scores = compute_cam_similarity_components(
                    cam_wbm,
                    cam_wdm,
                    sorted(common_classes),
                    self.cam_extractor,
                )

            score = match_score(
                z_wbm, z_wdm_i, s_wbm, s_wdm_i,
                areas_wbm, areas_wdm_i,
                explicit_scores,
                self.alpha, self.beta, self.gamma, self.delta, self.epsilon, self.theta,
            )

            if score > 0:
                global_sim = ((F.cosine_similarity(z_wbm.unsqueeze(0),
                                                   z_wdm_i.unsqueeze(0)).item() + 1) / 2)
                size_sim = (area_similarity(areas_wbm, areas_wdm_i)
                            if areas_wbm and areas_wdm_i else None)
                size_sim = explicit_scores.get('size_sim', size_sim)
                results.append({
                    'wdm_idx':           i,
                    'score':             score,
                    'overlap':           overlap,
                    'shape_sim':         explicit_scores.get('shape_sim'),
                    'size_sim':          size_sim,
                    'position_sim':      explicit_scores.get('position_sim'),
                    'local_feature_sim': explicit_scores.get('local_feature_sim', global_sim),
                    'global_sim':        global_sim,
                    's_wdm':             s_wdm_i,
                    'common_classes': common_classes,
                })

        results.sort(key=lambda r: r['score'], reverse=True)
        return results[:top_k]
