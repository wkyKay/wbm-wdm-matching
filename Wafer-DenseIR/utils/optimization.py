# -*- coding: utf-8 -*-

import math

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_scale=1e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_scale = min_lr_scale
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.min_lr_scale + float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return self.min_lr_scale + max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def get_optimizer(params, name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    if name == 'adamw':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f'Unknown optimizer: {name}')


def get_scheduler(optimizer, name: str, epochs: int, warmup_epochs: int = 0):
    if name == 'none':
        return None
    if name == 'cosine':
        return WarmupCosineSchedule(optimizer, warmup_steps=warmup_epochs, total_steps=epochs)
    raise ValueError(f'Unknown scheduler: {name}')
