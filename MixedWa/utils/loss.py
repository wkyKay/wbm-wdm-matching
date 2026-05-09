# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """多标签二元交叉熵损失，支持可选的类别权重。"""
    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class PositionAwareLoss(nn.Module):
    """
    阶段二位置感知损失：
      L = BCE(z_orig, label) + λ * max(0, margin - cosine_distance(z_orig, z_shift))

    通过推远平移版本的 embedding，迫使 encoder 学习位置敏感特征。
    """
    def __init__(self, margin: float = 0.5, lam: float = 0.1,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.margin = margin
        self.lam = lam
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                z_orig: torch.Tensor,
                z_shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) 分类 logits
            targets: (B, C) 多热标签
            z_orig:  (B, D) 原图 embedding（L2 归一化后）
            z_shift: (B, D) 平移图 embedding（L2 归一化后）
        """
        loss_cls = self.bce(logits, targets)

        # cosine_distance = 1 - cosine_similarity，值域 [0, 2]
        cos_sim = F.cosine_similarity(z_orig, z_shift, dim=1)  # (B,)
        cos_dist = 1.0 - cos_sim
        loss_neg = torch.clamp(self.margin - cos_dist, min=0.0).mean()

        return loss_cls + self.lam * loss_neg


class WaPIRLLoss(nn.Module):
    """
    WaPIRL 对比学习损失（NCE Loss）。
    参考 Kahng & Kim (2021) Eq.(6)。
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.sim1d = nn.CosineSimilarity(dim=1)
        self.sim2d = nn.CosineSimilarity(dim=2)
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self,
                anchors: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor):
        """
        Args:
            anchors:   (B, D) 记忆库中的表示
            positives: (B, D) 当前 batch 的正样本表示
            negatives: (K, D) 负样本表示
        Returns:
            loss, logits
        """
        B, D = anchors.size()
        K = negatives.size(0)

        # 正样本相似度：(B,)
        pos_sim = self.sim1d(anchors, positives) / self.temperature  # (B,)

        # 负样本相似度：(B, K)
        anchors_exp = anchors.unsqueeze(1).expand(B, K, D)       # (B, K, D)
        neg_exp = negatives.unsqueeze(0).expand(B, K, D)         # (B, K, D)
        neg_sim = self.sim2d(anchors_exp, neg_exp) / self.temperature  # (B, K)

        # logits: (B, 1+K)，正样本在第 0 列
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=anchors.device)

        loss = self.ce(logits, labels)
        return loss, logits
