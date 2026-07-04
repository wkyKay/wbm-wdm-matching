# -*- coding: utf-8 -*-
"""Optimization helpers."""

import torch


def get_optimizer(params, name='adamw', lr=1e-3, weight_decay=1e-4, momentum=0.9):
    name = name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f'Unknown optimizer: {name}')


def get_scheduler(optimizer, name='cosine', epochs=100, warmup_epochs=0):
    name = None if name in (None, 'none') else str(name).lower()
    if name is None:
        return None
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs - warmup_epochs), 1))
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(int(epochs // 3), 1), gamma=0.1)
    raise ValueError(f'Unknown scheduler: {name}')

