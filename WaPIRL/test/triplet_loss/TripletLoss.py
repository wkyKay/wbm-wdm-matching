import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================
# 1. Triplet Loss 定义
# ============================================================

class TripletLoss(nn.Module):
    """
    Triplet Loss: 拉近正样本对，推远负样本对
    
    Anchor (WBM) ─┬─ → Positive (匹配的WDM)  ─→ 距离近
                  │
                  └─→ Negative (不匹配的WDM) ─→ 距离远
    """
    def __init__(self, margin=0.2, distance='cosine'):
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(self, anchor, positive):
        """
        Args:
            anchor: (B, D) - WBM的嵌入
            positive: (B, D) - 配对的WDM嵌入
            negative: (B, D) 或 (B, K, D) - 不配对的WDM嵌入（单个或多个）
        
        Returns:
            loss: scalar
        """
         # 构建负样本: 从同batch中随机采样不配对的样本
        B = anchor.size(0)
        neg_indices = torch.randperm(B)
        negative = positive[neg_indices]  # 打乱顺序作为负样本

        if self.distance == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=1)
            neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=1) if negative.dim() == 2 else \
                       1 - F.cosine_similarity(anchor.unsqueeze(1), negative, dim=2).mean(dim=1)
        else:
            # 欧氏距离
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative) if negative.dim() == 2 else \
                       F.pairwise_distance(anchor.unsqueeze(1), negative).mean(dim=1)
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Mining: 在每个batch中自动挖掘难负样本
    
    策略：
    - All positive: 所有配对样本
    - All negative: batch中所有不配对的样本
    - Semi-hard: 距离在 pos_dist 和 pos_dist + margin 之间的负样本
    - Hardest: 距离最近的负样本
    """
    def __init__(self, margin=0.2, mining='semi-ha'):
        super().__init__()
        self.margin = margin
        self.mining = mining  # 'all', 'semi-hard', 'hardest'
    
    def forward(self, wbm_emb, wdm_emb):
        """
        Args:
            wbm_emb: (B, D) - batch中所有WBM的嵌入
            wdm_emb: (B, D) - batch中所有WDM的嵌入（配对的）
        
        Returns:
            loss: scalar
        """
        B = wbm_emb.size(0)
        
        # 计算相似度矩阵 (B, B)
        # wbm_emb[i] 与 wdm_emb[j] 的相似度
        sim_matrix = torch.mm(wbm_emb, wdm_emb.T)  # (B, B)
        
        # 正样本对在对角线上
        positive_pairs = torch.arange(B, device=wbm_emb.device)
        
        # 计算距离矩阵
        dist_matrix = 1 - sim_matrix  # (B, B)
        
        if self.mining == 'all':
            # 所有负样本
            loss = 0
            count = 0
            for i in range(B):
                pos_dist = dist_matrix[i, i]
                for j in range(B):
                    if j != i:  # 排除正样本
                        neg_dist = dist_matrix[i, j]
                        loss += F.relu(pos_dist - neg_dist + self.margin)
                        count += 1
            loss = loss / count
        
        elif self.mining == 'semi-hard':
            # 半难负样本: neg_dist < pos_dist + margin 但 neg_dist > pos_dist
            loss = 0
            count = 0
            for i in range(B):
                pos_dist = dist_matrix[i, i]
                for j in range(B):
                    if j != i:
                        neg_dist = dist_matrix[i, j]
                        if neg_dist > pos_dist and neg_dist < pos_dist + self.margin:
                            loss += F.relu(pos_dist - neg_dist + self.margin)
                            count += 1
            if count == 0:
                return torch.tensor(0.0, device=wbm_emb.device)
            loss = loss / count
        
        elif self.mining == 'hardest':
            # 最难负样本: 距离最近的负样本
            loss = 0
            for i in range(B):
                pos_dist = dist_matrix[i, i]
                # 排除自己，取最近的负样本
                neg_dists = dist_matrix[i].clone()
                neg_dists[i] = float('inf')
                hard_neg_dist = neg_dists.min()
                loss += F.relu(pos_dist - hard_neg_dist + self.margin)
            loss = loss / B
        
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.similarity_1d = nn.CosineSimilarity(dim=1)   # (B, F) -> (B,)
        self.similarity_2d = nn.CosineSimilarity(dim=2)   # (B, N, F) -> (B, N)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self,
                anchors: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor):
        """
        Args:
            anchors:   (B, F)  查询样本特征
            positives: (B, F)  与 anchors 一一对应的正样本特征
            negatives: (numN, F) 负样本特征集合（通常来自记忆库或额外负样本）
        Returns:
            loss: 标量损失
            logits: (B, 1+numN) 未归一化的相似度 logits，可用于监控
        """
        assert anchors.size() == positives.size()
        batch_size, _ = anchors.size()
        num_negatives, _ = negatives.size()

        # 将负样本扩展为与 batch 相同的维度 (B, numN, F)
        negatives = negatives.unsqueeze(0).repeat(batch_size, 1, 1)   # (B, numN, F)
        negatives = negatives.detach()   # 固定负样本，不计算梯度（通常做法）

        # ---------- 1. anchor - positive 相似度 ----------
        sim_a2p = self.similarity_1d(anchors, positives)             # (B,)
        sim_a2p = sim_a2p.div(self.temperature).unsqueeze(1)         # (B, 1)

        # ---------- 2. anchor - negative 相似度 ----------
        # 将 anchors 扩展为 (B, numN, F)，与 negatives 对齐
        anchors_expanded = anchors.unsqueeze(1).repeat(1, num_negatives, 1)  # (B, numN, F)
        sim_a2n = self.similarity_2d(anchors_expanded, negatives)           # (B, numN)
        sim_a2n = sim_a2n.div(self.temperature)                              # (B, numN)

        # ---------- 3. 拼接 logits ----------
        logits = torch.cat([sim_a2p, sim_a2n], dim=1)   # (B, 1 + numN)

        # ---------- 4. 交叉熵损失，正样本位于第 0 列 ----------
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = self.cross_entropy(logits, labels)

        return loss, logits.detach()


import torch
import torch.nn as nn

class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 40.0, lambda_val: float = 0.5):
        """
        Multi-Similarity Loss for deep metric learning.
        Args:
            alpha: weight for negative term (usually >0)
            beta:  weight for positive term (usually >0)
            lambda_val: margin threshold
        """
        super(MultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_val = lambda_val
        self.similarity_1d = nn.CosineSimilarity(dim=1)   # (B, F) -> (B,)
        self.similarity_2d = nn.CosineSimilarity(dim=2)   # (B, N, F) -> (B, N)

    def forward(self,
                anchors: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor):
        """
        Args:
            anchors:   (B, F)  查询样本特征
            positives: (B, F)  与 anchors 一一对应的正样本特征
            negatives: (numN, F) 负样本特征集合（通常来自记忆库或额外负样本）
        Returns:
            loss: 标量损失
            logits: (B, 1+numN) 未归一化的相似度（余弦相似度，未经温度缩放），可用于监控
        """
        assert anchors.size() == positives.size()
        batch_size, _ = anchors.size()
        num_negatives, _ = negatives.size()

        # 将负样本扩展为与 batch 相同的维度 (B, numN, F)
        negatives = negatives.unsqueeze(0).repeat(batch_size, 1, 1)   # (B, numN, F)
        negatives = negatives.detach()   # 固定负样本，不计算梯度

        # ---------- 1. 计算相似度 ----------
        # anchor-positive 相似度
        sim_ap = self.similarity_1d(anchors, positives)               # (B,)
        # anchor-negative 相似度（每个 anchor 与所有负样本）
        anchors_exp = anchors.unsqueeze(1).repeat(1, num_negatives, 1) # (B, numN, F)
        sim_an = self.similarity_2d(anchors_exp, negatives)           # (B, numN)

        # ---------- 2. 计算正项：1/β * log(1 + Σ exp(-β (S_{pos} - λ))) ----------
        # 正项只涉及一个正样本，但公式中 Σ 对每个正样本求和，这里只有一个
        pos_exp = -self.beta * (sim_ap - self.lambda_val)             # (B,)
        # 为了计算 log(1 + exp(...))，将 0 与 exp 值拼接后使用 logsumexp
        pos_logits = torch.stack([torch.zeros_like(pos_exp), pos_exp], dim=1)  # (B, 2)
        pos_term = (1.0 / self.beta) * torch.logsumexp(pos_logits, dim=1)      # (B,)

        # ---------- 3. 计算负项：1/α * log(1 + Σ exp(α (S_{neg} - λ))) ----------
        neg_exp = self.alpha * (sim_an - self.lambda_val)             # (B, numN)
        # 在每行的最前面加一个 0，以便 log(1 + sum(exp(...)))
        neg_logits = torch.cat([torch.zeros(batch_size, 1, device=neg_exp.device), neg_exp], dim=1)  # (B, 1+numN)
        neg_term = (1.0 / self.alpha) * torch.logsumexp(neg_logits, dim=1)      # (B,)

        # ---------- 4. 损失：对 batch 取平均 ----------
        loss = (pos_term + neg_term).mean()

        # 返回 logits（与 WaPIRLLoss/InfoNCELoss 格式一致，用于监控）
        logits = torch.cat([sim_ap.unsqueeze(1), sim_an], dim=1)      # (B, 1+numN)

        return loss, logits.detach()


# class InfoNCELoss(nn.Module):
#     """
#     InfoNCE 损失（对称版本）
#     对于双塔模型，计算 wbm 和 wdm 之间的对比损失。
#     正样本对为对角线元素 (i, i)，负样本为同一 batch 内的其他所有非对角线元素。
#     """
#     def __init__(self, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, wbm_emb, wdm_emb):
#         """
#         Args:
#             wbm_emb: (B, D)
#             wdm_emb: (B, D)
#         Returns:
#             loss: scalar
#         """
#         B = wbm_emb.size(0)
        
#         # L2 归一化，使相似度为余弦相似度
#         wbm_emb = F.normalize(wbm_emb, dim=1)   # (B, D)
#         wdm_emb = F.normalize(wdm_emb, dim=1)   # (B, D)
        
#         # 计算相似度矩阵 (B, B)
#         logits = torch.mm(wbm_emb, wdm_emb.T) / self.temperature  # 缩放
        
#         # 标签：正样本在 diag 位置，即第 i 个 wbm 对应第 i 个 wdm
#         labels = torch.arange(B, device=wbm_emb.device)
        
#         # 对称损失：两个方向的交叉熵平均
#         loss_w2d = F.cross_entropy(logits, labels)          # wbm -> wdm
#         loss_d2w = F.cross_entropy(logits.T, labels)        # wdm -> wbm
#         loss = (loss_w2d + loss_d2w) / 2
        
#         return loss


# class MultiSimilarityLoss(nn.Module):
#     """
#     Multi-Similarity Loss
#     论文：https://arxiv.org/abs/1904.06627
#     对每个 anchor，自动挖掘 hard positive 和 hard negative 并赋予不同权重。
#     """
#     def __init__(self, margin=0.1, alpha=2.0, beta=50.0):
#         """
#         Args:
#             margin: 正负样本相似度的边界，论文中常用 0.1
#             alpha: 正样本加权指数底数，论文中常用 2.0
#             beta:  负样本加权指数底数，论文中常用 50.0
#         """
#         super().__init__()
#         self.margin = margin
#         self.alpha = alpha
#         self.beta = beta

#     def forward(self, wbm_emb, wdm_emb):
#         """
#         Args:
#             wbm_emb: (B, D)
#             wdm_emb: (B, D)
#         Returns:
#             loss: scalar
#         """
#         B = wbm_emb.size(0)
#         device = wbm_emb.device
        
#         # L2 归一化
#         wbm_emb = F.normalize(wbm_emb, dim=1)
#         wdm_emb = F.normalize(wdm_emb, dim=1)
        
#         # 相似度矩阵 (B, B)，范围 [-1, 1]
#         sim_mat = torch.mm(wbm_emb, wdm_emb.T)   # (B, B)
        
#         # 正样本相似度：对角线元素
#         pos_sim = torch.diag(sim_mat)            # (B,)
        
#         # 构造 mask：正样本 mask 为对角线的单位矩阵，负样本 mask 为 1 - 单位矩阵
#         mask_pos = torch.eye(B, device=device, dtype=torch.bool)
#         mask_neg = ~mask_pos
        
#         # 获取所有负样本相似度 (B, B-1)
#         neg_sim = sim_mat[mask_neg].view(B, -1)  # (B, B-1)
        
#         # ---- 正样本加权 ----
#         # 论文公式：对于每个 anchor，计算所有正样本对的权重
#         # 但实际上每个 anchor 只有一个正样本（对角线），因此简化
#         # 正样本损失部分：exp(-alpha * (pos_sim - margin))
#         pos_exp = torch.exp(-self.alpha * (pos_sim - self.margin))
#         pos_loss = (1.0 / self.alpha) * torch.log(1.0 + pos_exp.sum())
        
#         # ---- 负样本加权 ----
#         # 对每个 anchor，对所有负样本计算权重
#         # 负样本损失部分：exp(beta * (neg_sim - margin))
#         neg_exp = torch.exp(self.beta * (neg_sim - self.margin))
#         # 对每个 anchor 内的负样本求和，然后取平均
#         neg_loss_per_anchor = (1.0 / self.beta) * torch.log(1.0 + neg_exp.sum(dim=1))
#         neg_loss = neg_loss_per_anchor.mean()
        
#         loss = pos_loss + neg_loss
#         return loss