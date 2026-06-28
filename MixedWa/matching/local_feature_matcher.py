
# -*- coding: utf-8 -*-
"""
CNN局部特征 + 规则方法的融合匹配器。

核心思路：
1. CNN提取多尺度局部特征图（保留空间信息）
2. 规则方法做几何不变性匹配（平移、旋转搜索）
3. 多视图相似度融合（形状、面积、位置、局部特征）
4. 支持部分匹配（只匹配相似的聚类区域）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class LocalFeatureMatchResult:
    """局部特征匹配的完整结果，包含可解释分量。"""
    score: float
    shape_sim: float
    size_sim: float
    position_sim: float
    local_feature_sim: float
    coverage: float
    best_offset: tuple  # (dy, dx) 最佳平移偏移
    matched_regions: list  # 匹配到的区域列表


class LocalFeatureExtractor:
    """
    从CNN backbone提取多尺度局部特征图。
    
    输出：
        - multi_scale_features: 不同层的特征图列表，从浅到深
        - global_feature: 全局平均池化特征
    """
    
    def __init__(self, backbone, device='cpu'):
        self.backbone = backbone.to(device)
        self.device = device
        self.backbone.eval()
        
    @torch.no_grad()
    def extract(self, x):
        """
        提取多尺度特征。
        
        Args:
            x: (B, C, H, W) 输入tensor
            
        Returns:
            (multi_scale_features, global_feature)
            - multi_scale_features: [feat1, feat2, feat3, feat4]，每层特征图
            - global_feature: (B, D) 全局特征
        """
        x = x.to(self.device)
        
        # 逐层提取特征
        features = []
        x = self.backbone.stem(x)
        features.append(x)  # layer0 (stem)
        
        x = self.backbone.layer1(x)
        features.append(x)  # layer1
        
        x = self.backbone.layer2(x)
        features.append(x)  # layer2
        
        x = self.backbone.layer3(x)
        features.append(x)  # layer3
        
        x = self.backbone.layer4(x)
        features.append(x)  # layer4
        
        # 全局特征
        global_feat = self.backbone.gap(x).flatten(1)
        
        return features, global_feat


class GeometricMatcher:
    """
    几何不变性匹配器：支持平移、小角度旋转搜索。
    
    使用互相关（cross-correlation）在特征图上做滑动窗口匹配。
    """
    
    def __init__(self, max_shift=10, device='cpu'):
        """
        Args:
            max_shift: 最大平移搜索范围（像素）
        """
        self.max_shift = max_shift
        self.device = device
        
    def match_translation(self, feat_wbm, feat_wdm, mask_wbm=None):
        """
        平移搜索：找到最优平移偏移，返回偏移量和相似度热图。
        
        Args:
            feat_wbm: (C, H, W) WBM特征图
            feat_wdm: (C, H, W) WDM特征图
            mask_wbm: (H, W) WBM有效区域mask，可选
            
        Returns:
            (best_offset, max_sim, similarity_map)
            - best_offset: (dy, dx) 最佳平移偏移
            - max_sim: 最大相似度
            - similarity_map: (2*max_shift+1, 2*max_shift+1) 相似度热图
        """
        C, H, W = feat_wbm.shape
        
        # 归一化特征 - 每个通道独立归一化
        feat_wbm_norm = feat_wbm.clone()
        feat_wdm_norm = feat_wdm.clone()
        for c in range(C):
            norm_wbm = torch.norm(feat_wbm[c])
            norm_wdm = torch.norm(feat_wdm[c])
            if norm_wbm > 1e-8:
                feat_wbm_norm[c] = feat_wbm[c] / norm_wbm
            if norm_wdm > 1e-8:
                feat_wdm_norm[c] = feat_wdm[c] / norm_wdm
        
        # 应用mask（如果有）
        if mask_wbm is not None:
            mask_tensor = mask_wbm.clone().float() if torch.is_tensor(mask_wbm) else torch.from_numpy(mask_wbm).float()
            mask_tensor = mask_tensor.to(feat_wbm.device)
            # 上采样mask到特征图尺寸
            if mask_tensor.shape != (H, W):
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    (H, W),
                    mode='nearest'
                ).squeeze()
            feat_wbm_norm = feat_wbm_norm * mask_tensor.unsqueeze(0)
        
        # 简单高效的方法：计算不同偏移下的相似度
        # 避免复杂的conv2d逻辑，改用循环（对于小max_shift足够快）
        pad = self.max_shift
        best_dy, best_dx = 0, 0
        max_sim = -1.0
        similarity_map = np.zeros((2*pad+1, 2*pad+1), dtype=np.float32)
        
        # 搜索范围内的所有偏移
        for dy_idx in range(2*pad+1):
            dy = dy_idx - pad
            for dx_idx in range(2*pad+1):
                dx = dx_idx - pad
                
                # 计算两个特征图在该偏移下的重叠区域
                y1_wbm, y2_wbm = max(0, dy), min(H, H + dy)
                x1_wbm, x2_wbm = max(0, dx), min(W, W + dx)
                y1_wdm, y2_wdm = max(0, -dy), min(H, H - dy)
                x1_wdm, x2_wdm = max(0, -dx), min(W, W - dx)
                
                # 计算点积相似度
                if (y2_wbm > y1_wbm and x2_wbm > x1_wbm and
                    y2_wdm > y1_wdm and x2_wdm > x1_wdm):
                    
                    wbm_patch = feat_wbm_norm[:, y1_wbm:y2_wbm, x1_wbm:x2_wbm]
                    wdm_patch = feat_wdm_norm[:, y1_wdm:y2_wdm, x1_wdm:x2_wdm]
                    
                    sim = (wbm_patch * wdm_patch).sum().item()
                    # 归一化到 [0,1]
                    n_elem = (y2_wbm - y1_wbm) * (x2_wbm - x1_wbm) * C
                    if n_elem > 0:
                        sim = sim / n_elem
                else:
                    sim = 0.0
                
                similarity_map[dy_idx, dx_idx] = sim
                
                if sim > max_sim:
                    max_sim = sim
                    best_dy = dy
                    best_dx = dx
        
        return (best_dy, best_dx), float(max_sim), similarity_map
    
    def apply_shift(self, x, offset):
        """
        对tensor应用平移偏移。
        
        Args:
            x: (C, H, W) 或 (H, W) 输入tensor
            offset: (dy, dx) 偏移量
            
        Returns:
            平移后的tensor
        """
        dy, dx = offset
        if torch.is_tensor(x):
            shifted = torch.zeros_like(x)
        else:
            shifted = np.zeros_like(x)
        
        if len(x.shape) == 3:
            C, H, W = x.shape
            y_start, y_end = max(0, dy), min(H, H + dy)
            x_start, x_end = max(0, dx), min(W, W + dx)
            src_y_start, src_y_end = max(0, -dy), min(H, H - dy)
            src_x_start, src_x_end = max(0, -dx), min(W, W - dx)
            shifted[:, y_start:y_end, x_start:x_end] = x[:, src_y_start:src_y_end, src_x_start:src_x_end]
        else:
            H, W = x.shape
            y_start, y_end = max(0, dy), min(H, H + dy)
            x_start, x_end = max(0, dx), min(W, W + dx)
            src_y_start, src_y_end = max(0, -dy), min(H, H - dy)
            src_x_start, src_x_end = max(0, -dx), min(W, W - dx)
            shifted[y_start:y_end, x_start:x_end] = x[src_y_start:src_y_end, src_x_start:src_x_end]
        return shifted


def compute_shape_similarity(map1, map2, mask=None):
    """
    计算形状相似度（IoU）。
    
    Args:
        map1, map2: (H, W) 二值图或连续值图
        mask: (H, W) 有效区域mask
        
    Returns:
        形状相似度 ∈ [0, 1]
    """
    if mask is not None:
        map1 = map1 * mask
        map2 = map2 * mask
    
    intersection = np.minimum(map1, map2).sum()
    union = np.maximum(map1, map2).sum()
    
    if union < 1e-8:
        return 1.0
    return float(intersection / union)


def compute_size_similarity(area1, area2):
    """
    计算面积相似度。
    
    sim = 1 - |area1 - area2| / max(area1, area2)
    """
    max_area = max(area1, area2)
    if max_area < 1e-8:
        return 1.0
    return 1.0 - abs(area1 - area2) / max_area


def compute_position_similarity(center1, center2, max_dist):
    """
    计算位置相似度，基于质心距离。
    
    Args:
        center1, center2: (y, x) 质心坐标
        max_dist: 最大距离归一化值
        
    Returns:
        位置相似度 ∈ [0, 1]
    """
    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    return float(max(0.0, 1.0 - dist / max_dist))


def compute_centroid(binary_map):
    """计算二值图的质心。"""
    coords = np.argwhere(binary_map > 0)
    if len(coords) == 0:
        return (binary_map.shape[0] / 2, binary_map.shape[1] / 2)
    return (float(coords[:, 0].mean()), float(coords[:, 1].mean()))


class LocalFeatureMatcher:
    """
    CNN局部特征 + 规则方法的融合匹配器。
    
    工作流程：
    1. 提取多尺度局部特征
    2. 几何不变性搜索（平移）
    3. 多视图相似度计算
    4. 分数融合与排序
    """
    
    def __init__(self,
                 backbone,
                 device='cpu',
                 max_shift=10,
                 alpha=0.3,  # 形状权重
                 beta=0.2,   # 面积权重
                 gamma=0.2,  # 位置权重
                 delta=0.3): # 局部特征权重
        self.device = device
        self.feature_extractor = LocalFeatureExtractor(backbone, device)
        self.geometric_matcher = GeometricMatcher(max_shift, device)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
    @torch.no_grad()
    def match_single(self,
                     wbm_tensor,
                     wdm_tensor,
                     wbm_map,
                     wdm_map,
                     wbm_status=None):
        """
        匹配单对 (WBM, WDM)。
        
        Args:
            wbm_tensor: (1, C, H, W) WBM输入tensor
            wdm_tensor: (1, C, H, W) WDM输入tensor
            wbm_map: (H, W) WBM二值图
            wdm_map: (H, W) WDM二值图
            wbm_status: (H, W) WBM状态图，可选
            
        Returns:
            LocalFeatureMatchResult
        """
        # 1. 提取局部特征（使用中间层，保留更多空间细节）
        wbm_feats, _ = self.feature_extractor.extract(wbm_tensor)
        wdm_feats, _ = self.feature_extractor.extract(wdm_tensor)
        
        # 使用layer2作为主要特征（平衡细节和语义）
        feat_wbm = wbm_feats[2].squeeze(0)  # (C, H/4, W/4)
        feat_wdm = wdm_feats[2].squeeze(0)
        
        # 2. 几何匹配：平移搜索
        # 先resize map到特征图尺寸
        H, W = wbm_map.shape
        feat_H, feat_W = feat_wbm.shape[1:]
        
        mask = None
        if wbm_status is not None:
            wbm_status_resized = torch.nn.functional.interpolate(
                torch.from_numpy(wbm_status).float().unsqueeze(0).unsqueeze(0),
                (feat_H, feat_W),
                mode='nearest'
            ).squeeze().numpy()
            mask = (wbm_status_resized > 0).astype(float)
        
        # 平移搜索
        best_offset, local_feature_sim, _ = self.geometric_matcher.match_translation(
            feat_wbm, feat_wdm,
            mask_wbm=mask
        )
        
        # 将偏移量转换回原始尺度
        scale_y = H / feat_H
        scale_x = W / feat_W
        best_offset_orig = (int(round(best_offset[0] * scale_y)), 
                            int(round(best_offset[1] * scale_x)))
        
        # 3. 应用最佳偏移到WDM map
        wdm_map_shifted = self.geometric_matcher.apply_shift(wdm_map, best_offset_orig)
        
        # 4. 计算多视图相似度
        # 形状相似度
        shape_sim = compute_shape_similarity(wbm_map, wdm_map_shifted, wbm_status)
        
        # 面积相似度
        area_wbm = wbm_map.sum()
        area_wdm = wdm_map.sum()
        size_sim = compute_size_similarity(area_wbm, area_wdm)
        
        # 位置相似度
        center_wbm = compute_centroid(wbm_map)
        center_wdm_shifted = compute_centroid(wdm_map_shifted)
        max_dist = np.sqrt(H**2 + W**2) / 2
        position_sim = compute_position_similarity(center_wbm, center_wdm_shifted, max_dist)
        
        # 覆盖率
        if wbm_status is not None:
            meaningful = (wbm_status > 0)
            intersection = ((wbm_map > 0) & (wdm_map_shifted > 0) & meaningful).sum()
            coverage = float(intersection / max((wbm_map > 0).sum(), 1))
        else:
            intersection = ((wbm_map > 0) & (wdm_map_shifted > 0)).sum()
            coverage = float(intersection / max((wbm_map > 0).sum(), 1))
        
        # 5. 融合分数
        total = self.alpha + self.beta + self.gamma + self.delta
        score = (self.alpha * shape_sim + 
                 self.beta * size_sim + 
                 self.gamma * position_sim + 
                 self.delta * local_feature_sim) / total
        
        return LocalFeatureMatchResult(
            score=score,
            shape_sim=shape_sim,
            size_sim=size_sim,
            position_sim=position_sim,
            local_feature_sim=local_feature_sim,
            coverage=coverage,
            best_offset=best_offset_orig,
            matched_regions=[{
                'wbm_center': center_wbm,
                'wdm_center': center_wdm_shifted,
                'area_wbm': float(area_wbm),
                'area_wdm': float(area_wdm),
            }]
        )
        
    def match_batch(self,
                    wbm_tensor,
                    wdm_tensors,
                    wbm_map,
                    wdm_maps,
                    wbm_status=None,
                    top_k=5):
        """
        批量匹配，返回top-k结果。
        
        Args:
            wbm_tensor: (1, C, H, W) WBM输入
            wdm_tensors: N个(1, C, H, W) WDM输入列表
            wbm_map: (H, W) WBM二值图
            wdm_maps: N个(H, W) WDM二值图列表
            wbm_status: (H, W) WBM状态图，可选
            top_k: 返回前k个
            
        Returns:
            [(wdm_idx, result), ...] 按score降序排列
        """
        results = []
        for i, (wdm_tensor, wdm_map) in enumerate(zip(wdm_tensors, wdm_maps)):
            result = self.match_single(wbm_tensor, wdm_tensor, wbm_map, wdm_map, wbm_status)
            results.append((i, result))
            
        results.sort(key=lambda x: x[1].score, reverse=True)
        return results[:top_k]
