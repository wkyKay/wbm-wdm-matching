# -*- coding: utf-8 -*-
"""WaPIRL-style contrastive loss for cluster tokens."""

import torch
import torch.nn as nn


class ClusterNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.similarity_1d = nn.CosineSimilarity(dim=1)
        self.similarity_2d = nn.CosineSimilarity(dim=2)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):
        if anchors.size() != positives.size():
            raise ValueError(f'anchors {anchors.size()} and positives {positives.size()} must match')
        batch_size, _ = anchors.size()
        if negatives.numel() == 0:
            logits = self.similarity_1d(anchors, positives).div(self.temperature).unsqueeze(1)
        else:
            num_negatives, _ = negatives.size()
            negatives = negatives.unsqueeze(0).repeat(batch_size, 1, 1).detach()
            sim_a2p = self.similarity_1d(anchors, positives).div(self.temperature).unsqueeze(1)
            sim_a2n = self.similarity_2d(
                positives.unsqueeze(1).repeat(1, num_negatives, 1),
                negatives,
            ).div(self.temperature)
            logits = torch.cat([sim_a2p, sim_a2n], dim=1)
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return self.cross_entropy(logits, targets), logits.detach()

