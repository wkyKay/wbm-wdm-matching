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


def multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor,
                       threshold: float = 0.5) -> dict:
    """
    多标签分类综合指标，返回：
      mAP          — macro Average Precision（与阈值无关，基于排序）
      f1_macro     — macro F1（每类单独计算后取均值，对稀有类敏感）
      f1_micro     — micro F1（全局 TP/FP/FN 汇总后计算，反映整体表现）
      exact_match  — Exact Match Ratio，所有类别完全预测正确的样本比例
      hamming_acc  — 1 - Hamming Loss，单个标签级别的平均正确率
    """
    probs = torch.sigmoid(logits).detach().cpu()
    preds = (probs > threshold).float()
    tgts  = targets.detach().cpu().float()

    preds_np = preds.numpy()
    tgts_np  = tgts.numpy()
    N, C = tgts_np.shape

    # mAP
    map_score = mean_ap(logits, targets)

    # macro F1：每类单独算 F1，再取均值
    f1s = []
    for c in range(C):
        tp = ((preds_np[:, c] == 1) & (tgts_np[:, c] == 1)).sum()
        fp = ((preds_np[:, c] == 1) & (tgts_np[:, c] == 0)).sum()
        fn = ((preds_np[:, c] == 0) & (tgts_np[:, c] == 1)).sum()
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    f1_macro = float(np.mean(f1s))

    # micro F1：全局汇总 TP/FP/FN
    tp_sum = ((preds_np == 1) & (tgts_np == 1)).sum()
    fp_sum = ((preds_np == 1) & (tgts_np == 0)).sum()
    fn_sum = ((preds_np == 0) & (tgts_np == 1)).sum()
    denom = 2 * tp_sum + fp_sum + fn_sum
    f1_micro = float((2 * tp_sum / denom) if denom > 0 else 0.0)

    # Exact Match Ratio（所有标签完全正确）
    exact_match = float((preds == tgts).all(dim=1).float().mean().item())

    # Hamming Accuracy = 1 - Hamming Loss
    hamming_acc = float(1.0 - (preds != tgts).float().mean().item())

    return {
        'mAP':         map_score,
        'f1_macro':    f1_macro,
        'f1_micro':    f1_micro,
        'exact_match': exact_match,
        'hamming_acc': hamming_acc,
    }


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
    单标签分类的 accuracy / macro recall / macro F1。
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
