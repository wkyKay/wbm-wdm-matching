# -*- coding: utf-8 -*-

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAMExtractor:
    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module,
                 device: str = 'cpu',
                 cam_threshold: float = 0.5,
                 cam_min_area: float = 0.005):
        if not hasattr(backbone, 'forward_features'):
            raise ValueError('CAM matching currently requires a backbone with forward_features(), e.g. resnet18.')
        if not hasattr(classifier, 'fc'):
            raise ValueError('CAM matching requires a classifier with a linear fc layer.')

        self.backbone = backbone
        self.classifier = classifier
        self.device = device
        self.cam_threshold = cam_threshold
        self.cam_min_area = cam_min_area

    @torch.no_grad()
    def compute_cam(self,
                    x: torch.Tensor,
                    class_indices: Optional[Iterable[int]] = None) -> Dict[str, torch.Tensor]:
        x = x.to(self.device)
        feature_map, pooled = self.backbone.forward_features(x)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)

        if class_indices is None:
            class_indices = list(range(logits.shape[1]))
        else:
            class_indices = list(class_indices)

        if len(class_indices) == 0:
            cams = feature_map.new_zeros((x.shape[0], 0, feature_map.shape[2], feature_map.shape[3]))
        else:
            weights = self.classifier.fc.weight[class_indices].to(feature_map.device)
            cams = torch.einsum('bchw,kc->bkhw', feature_map, weights)
            cams = F.relu(cams)
            cams = self._normalize_cams(cams)

        return {
            'feature_map': feature_map,
            'pooled': pooled,
            'logits': logits,
            'probs': probs,
            'cams': cams,
            'class_indices': torch.tensor(class_indices, device=feature_map.device, dtype=torch.long),
        }

    @staticmethod
    def _normalize_cams(cams: torch.Tensor) -> torch.Tensor:
        flat = cams.flatten(2)
        mins = flat.min(dim=2).values[:, :, None, None]
        maxs = flat.max(dim=2).values[:, :, None, None]
        return (cams - mins) / (maxs - mins + 1e-8)

    def cam_to_mask(self, cam: torch.Tensor) -> Optional[torch.Tensor]:
        mask = cam >= self.cam_threshold
        area_ratio = mask.float().mean().item()
        if area_ratio >= self.cam_min_area:
            return mask

        flat = cam.flatten()
        k = max(1, int(round(flat.numel() * self.cam_min_area)))
        if flat.max().item() <= 0:
            return None
        threshold = torch.topk(flat, k).values[-1]
        return cam >= threshold


def weighted_pool(feature_map: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
    weights = cam / (cam.sum() + 1e-8)
    local_feat = (feature_map * weights.unsqueeze(0)).sum(dim=(1, 2))
    return F.normalize(local_feat, dim=0)


def cam_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    intersection = (mask_a & mask_b).float().sum().item()
    union = (mask_a | mask_b).float().sum().item()
    if union <= 0:
        return 0.0
    return intersection / union


def compute_cam_local_score(cam_wbm: Dict[str, torch.Tensor],
                            cam_wdm: Dict[str, torch.Tensor],
                            common_classes: Iterable[int],
                            extractor: CAMExtractor,
                            cam_lambda: float = 0.5) -> float:
    common_classes = list(common_classes)
    if len(common_classes) == 0:
        return 0.0

    wbm_class_to_pos = {int(c): idx for idx, c in enumerate(cam_wbm['class_indices'].tolist())}
    wdm_class_to_pos = {int(c): idx for idx, c in enumerate(cam_wdm['class_indices'].tolist())}
    feature_wbm = cam_wbm['feature_map'][0]
    feature_wdm = cam_wdm['feature_map'][0]

    scores: List[float] = []
    for class_idx in common_classes:
        if class_idx not in wbm_class_to_pos or class_idx not in wdm_class_to_pos:
            continue

        cam_a = cam_wbm['cams'][0, wbm_class_to_pos[class_idx]]
        cam_b = cam_wdm['cams'][0, wdm_class_to_pos[class_idx]]
        mask_a = extractor.cam_to_mask(cam_a)
        mask_b = extractor.cam_to_mask(cam_b)
        if mask_a is None or mask_b is None:
            continue

        mask_score = cam_iou(mask_a, mask_b)
        local_a = weighted_pool(feature_wbm, cam_a)
        local_b = weighted_pool(feature_wdm, cam_b)
        feat_score = F.cosine_similarity(local_a.unsqueeze(0), local_b.unsqueeze(0)).item()
        feat_score = (feat_score + 1.0) / 2.0
        scores.append(cam_lambda * mask_score + (1.0 - cam_lambda) * feat_score)

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
