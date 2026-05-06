import torch
import torch.nn as nn
import torch.nn.functional as F

# [修改点 1] 导入路径变更
from torchmetrics.classification import MulticlassROC, MulticlassPrecisionRecallCurve, MulticlassF1Score, MulticlassAUROC, MulticlassAveragePrecision

class MultiAUPRC(nn.Module):
    def __init__(self, num_classes: int):
        super(MultiAUPRC, self).__init__()
        self.num_classes = num_classes
        self.auprc_metric = MulticlassAveragePrecision(num_classes=num_classes)

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        probs = logits.softmax(dim=1)
        with torch.no_grad():
            avg_auprc = self.auprc_metric(probs, labels)
            if isinstance(avg_auprc, torch.Tensor):
                return avg_auprc
            return torch.tensor(avg_auprc)


class MultiAUROC(nn.Module):
    def __init__(self, num_classes: int):
        super(MultiAUROC, self).__init__()
        self.num_classes = num_classes
        self.auroc_metric = MulticlassAUROC(num_classes=num_classes)

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        probs = logits.softmax(dim=1)
        with torch.no_grad():
            avg_auroc = self.auroc_metric(probs, labels)
            if isinstance(avg_auroc, torch.Tensor):
                return avg_auroc
            return torch.tensor(avg_auroc)


class MultiAccuracy(nn.Module):
    def __init__(self, num_classes: int):
        super(MultiAccuracy, self).__init__()
        self.num_classes = num_classes
        # 可选：也可以直接用 torchmetrics.classification.MulticlassAccuracy
        # 但为了保持原有 "Function-like" 的调用习惯，保留手动计算逻辑也没问题，
        # 或者替换为官方实现以保证一致性。这里保留原逻辑，因为原逻辑很简单且无依赖。
        pass

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        assert logits.ndim == 2
        assert labels.ndim == 1
        assert len(logits) == len(labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = torch.eq(preds, labels)
            return torch.mean(correct.float())


class TopKAccuracy(nn.Module):
    def __init__(self, num_classes: int, k: int, threshold: float = 0.):
        super(TopKAccuracy, self).__init__()
        self.num_classes = num_classes
        self.k = k
        self.threshold = threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        assert logits.ndim == 2
        assert labels.ndim == 1
        assert len(logits) == len(labels)

        with torch.no_grad():
            topk_probs, topk_indices = torch.topk(F.softmax(logits, dim=1), self.k, dim=1)
            labels = labels.view(-1, 1).expand_as(topk_indices)                 # (B, k)
            correct = labels.eq(topk_indices) * (topk_probs >= self.threshold)  # (B, k)
            correct = correct.sum(dim=1).bool().float()                         # (B, ) & {0, 1}

            return torch.mean(correct)


class MultiPrecision(nn.Module):
    def __init__(self, num_classes: int, average='macro'):
        super(MultiPrecision, self).__init__()
        self.num_classes = num_classes
        assert average in ['macro', 'micro', 'weighted']
        self.average = average
        # 预定义 functional 的参数映射
        self.avg_map = {'macro': 'macro', 'micro': 'micro', 'weighted': 'weighted'}

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        assert logits.ndim == 2
        assert labels.ndim == 1

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            # [修改点 4] 使用新的 functional 接口
            # 旧版 reduction='elementwise_mean' 对应新版的 average='macro'
            return multiclass_precision(
                preds=probs,
                target=labels,
                num_classes=self.num_classes,
                average=self.avg_map[self.average]
            )


class MultiRecall(nn.Module):
    def __init__(self, num_classes: int, average='macro'):
        super(MultiRecall, self).__init__()
        self.num_classes = num_classes
        assert average in ['macro', 'micro', 'weighted']
        self.average = average
        self.avg_map = {'macro': 'macro', 'micro': 'micro', 'weighted': 'weighted'}

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        assert logits.ndim == 2
        assert labels.ndim == 1

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            # [修改点 4] 使用新的 functional 接口
            return multiclass_recall(
                preds=probs,
                target=labels,
                num_classes=self.num_classes,
                average=self.avg_map[self.average]
            )


class MultiF1Score(nn.Module):
    def __init__(self, num_classes: int, average: str = 'macro'):
        super(MultiF1Score, self).__init__()
        self.num_classes = num_classes
        assert average in ['macro', 'micro', 'weighted']
        self.average = average
        
        # [修改点 5] 强烈建议直接使用 torchmetrics 的官方实现
        # 它的逻辑更健壮，且支持 GPU 加速和分布式同步
        self.f1_metric = MulticlassF1Score(num_classes=num_classes, average=average)

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        assert logits.ndim == 2
        assert labels.ndim == 1
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            # 直接调用官方类
            return self.f1_metric(preds, labels)


# 下面的 BinaryFBetaScore 和 BinaryF1Score 如果没有被外部其他文件依赖，
# 且 MultiF1Score 已经改用官方实现，其实可以简化或删除。
# 但为了保持代码结构完整，这里保留并微调以适配 torch 原生操作（去除了对旧 PL 的潜在依赖）

class BinaryFBetaScore(nn.Module):
    def __init__(self, beta=1, threshold=.5, average='macro'):
        super(BinaryFBetaScore, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.average = average

    def forward(self, logit: torch.Tensor, label: torch.Tensor):
        assert logit.ndim == 1
        assert label.ndim == 1

        with torch.no_grad():
            pred = torch.sigmoid(logit)
            pred = pred > self.threshold   # boolean
            true = label > self.threshold  # boolean

            if self.average == 'macro':
                return self.macro_f_beta_score(pred, true, self.beta)
            elif self.average == 'micro':
                return self.micro_f_beta_score(pred, true, self.beta)
            elif self.average == 'weighted':
                return self.weighted_f_beta_score(pred, true, self.beta)
            else:
                raise NotImplementedError

    @staticmethod
    def macro_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):
        assert true.ndim == 1
        assert pred.ndim == 1

        pred = pred.float()
        true = true.float()

        tp = (pred * true).sum().float()
        fp = ((pred) * (1-true)).sum().float()
        fn = ((1-pred) * true).sum().float()

        precision_ = tp / (tp + fp + 1e-7)
        recall_ = tp / (tp + fn + 1e-7)

        f_beta = (1 + beta**2) * precision_ * recall_ / (beta**2 * precision_ + recall_ + 1e-7)
        return f_beta

    @staticmethod
    def micro_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):
        raise NotImplementedError

    @staticmethod
    def weighted_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):
        raise NotImplementedError


class BinaryF1Score(BinaryFBetaScore):
    def __init__(self, threshold=.5, average='macro'):
        super(BinaryF1Score, self).__init__(beta=1, threshold=threshold, average=average)


if __name__ == '__main__':
    targets = torch.LongTensor([2, 2, 0, 2, 1, 1, 1])
    predictions = torch.FloatTensor(
        [
            [1, 2, 7],  # 2
            [1, 3, 7],  # 2
            [3, 9, 0],  # 1 (Max is 9 at index 1) -> Wait, original comment said 1? 
                        # Row 2: [3, 9, 0], argmax is 1. Target is 0. Mismatch.
            [1, 2, 3],  # 2
            [3, 7, 0],  # 1
            [8, 1, 1],  # 0
            [9, 1, 1],  # 0
        ]
    )

    print("Testing MultiF1Score...")
    f1_function = MultiF1Score(num_classes=3, average='macro')
    f1_val = f1_function(logits=predictions, labels=targets)
    print(f"F1 Score: {f1_val}")

    print("Testing MultiAUROC...")
    auroc_function = MultiAUROC(num_classes=3)
    auroc_val = auroc_function(logits=predictions, labels=targets)
    print(f"AUROC: {auroc_val}")
    
    print("Testing MultiAUPRC...")
    auprc_function = MultiAUPRC(num_classes=3)
    auprc_val = auprc_function(logits=predictions, labels=targets)
    print(f"AUPRC: {auprc_val}")