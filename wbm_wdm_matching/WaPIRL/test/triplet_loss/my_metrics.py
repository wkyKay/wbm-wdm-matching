import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRecall(nn.Module):
    """Macro-average recall considering top-k predictions with probability threshold."""
    def __init__(self, num_classes: int, k: int, threshold: float = 0.):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.threshold = threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.ndim == 2
        assert labels.ndim == 1
        assert len(logits) == len(labels)
        B, C = logits.shape
        device = logits.device

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)                     # (B, C)
            topk_probs, topk_indices = torch.topk(probs, self.k, dim=1)  # (B, k)

            # Mask for predictions that meet the threshold
            valid_mask = (topk_probs >= self.threshold)         # (B, k)

            # Build one-hot prediction matrix: (B, C), 1 if class is in top-k and prob>=threshold
            pred_one_hot = torch.zeros(B, C, device=device).scatter_add(
                1, topk_indices, valid_mask.float()
            )  # scatter_add works because each (b, i) is unique per sample

            # One-hot ground truth
            label_one_hot = torch.zeros(B, C, device=device).scatter_(
                1, labels.view(-1, 1).long(), 1
            )  # (B, C)

            # Compute per-class TP, FP, FN
            TP = (pred_one_hot * label_one_hot).sum(dim=0)       # (C,)
            FP = (pred_one_hot * (1 - label_one_hot)).sum(dim=0) # (C,)
            FN = ((1 - pred_one_hot) * label_one_hot).sum(dim=0) # (C,)

            # Per-class recall (avoid division by zero)
            denominator = TP + FN
            recall_per_class = torch.where(denominator > 0, TP / denominator, torch.tensor(0.0, device=device))

            # Macro recall: average over classes that actually appear in labels
            present_classes = (label_one_hot.sum(dim=0) > 0)    # (C,)
            if present_classes.any():
                macro_recall = recall_per_class[present_classes].mean()
            else:
                macro_recall = torch.tensor(0.0, device=device)

            return macro_recall


class TopKF1Score(nn.Module):
    """Macro-average F1 score considering top-k predictions with probability threshold."""
    def __init__(self, num_classes: int, k: int, threshold: float = 0.):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.threshold = threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.ndim == 2
        assert labels.ndim == 1
        assert len(logits) == len(labels)
        B, C = logits.shape
        device = logits.device

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, self.k, dim=1)
            valid_mask = (topk_probs >= self.threshold)

            pred_one_hot = torch.zeros(B, C, device=device).scatter_add(
                1, topk_indices, valid_mask.float()
            )
            label_one_hot = torch.zeros(B, C, device=device).scatter_(
                1, labels.view(-1, 1).long(), 1
            )

            TP = (pred_one_hot * label_one_hot).sum(dim=0)
            FP = (pred_one_hot * (1 - label_one_hot)).sum(dim=0)
            FN = ((1 - pred_one_hot) * label_one_hot).sum(dim=0)

            # Per-class precision and recall
            prec_num = TP
            prec_den = TP + FP
            recall_num = TP
            recall_den = TP + FN

            # Avoid division by zero: set precision/recall to 0 if undefined
            precision = torch.where(prec_den > 0, prec_num / prec_den, torch.tensor(0.0, device=device))
            recall = torch.where(recall_den > 0, recall_num / recall_den, torch.tensor(0.0, device=device))

            # Per-class F1
            f1_num = 2 * precision * recall
            f1_den = precision + recall
            f1_per_class = torch.where(f1_den > 0, f1_num / f1_den, torch.tensor(0.0, device=device))

            # Macro F1: average over classes that actually appear in labels
            present_classes = (label_one_hot.sum(dim=0) > 0)
            if present_classes.any():
                macro_f1 = f1_per_class[present_classes].mean()
            else:
                macro_f1 = torch.tensor(0.0, device=device)

            return macro_f1