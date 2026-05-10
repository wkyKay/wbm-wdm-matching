# -*- coding: utf-8 -*-

import torch
import numpy as np
from sklearn.metrics import average_precision_score


def multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                        threshold: float = 0.5) -> float:
    """多标签精确匹配准确率（所有类别都正确才算对）。"""
    preds = (torch.sigmoid(logits) > threshold).float()
    correct = (preds == targets).all(dim=1).float()
    return correct.mean().item()


def per_class_ap(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    每类 Average Precision（多标签）。
    返回 shape (C,) 的 numpy 数组。
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    tgts = targets.detach().cpu().numpy()
    aps = []
    for c in range(tgts.shape[1]):
        if tgts[:, c].sum() == 0:
            aps.append(float('nan'))
        else:
            aps.append(average_precision_score(tgts[:, c], probs[:, c]))
    return np.array(aps)


def mean_ap(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """mAP（忽略 NaN 类别）。"""
    aps = per_class_ap(logits, targets)
    return float(np.nanmean(aps))


def subset_match_recall(pred_probs: torch.Tensor, gt_probs: torch.Tensor,
                        threshold: float = 0.5) -> float:
    """
    子集召回率：预测的 WDM pattern 集合是否是 WBM 的子集。
    pred_probs: (C,) WDM 的 sigmoid 输出
    gt_probs:   (C,) WBM 的 sigmoid 输出
    """
    pred_set = set((pred_probs > threshold).nonzero(as_tuple=True)[0].tolist())
    wbm_set  = set((gt_probs  > threshold).nonzero(as_tuple=True)[0].tolist())
    if len(pred_set) == 0:
        return 0.0
    return len(pred_set & wbm_set) / len(pred_set)


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """单标签 Top-K 准确率。"""
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    单标签分类的 macro recall 和 macro F1。
    返回 {'recall': float, 'f1': float}。
    """
    preds = logits.argmax(dim=1).cpu().numpy()
    tgts  = targets.cpu().numpy()
    num_classes = logits.size(1)

    recalls, f1s = [], []
    for c in range(num_classes):
        tp = ((preds == c) & (tgts == c)).sum()
        fn = ((preds != c) & (tgts == c)).sum()
        fp = ((preds == c) & (tgts != c)).sum()
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        recalls.append(r)
        f1s.append(f1)

    acc = float((preds == tgts).mean())
    return {
        'acc':    acc,
        'recall': float(np.mean(recalls)),
        'f1':     float(np.mean(f1s)),
    }
