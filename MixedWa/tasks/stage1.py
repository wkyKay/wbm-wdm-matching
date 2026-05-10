# -*- coding: utf-8 -*-
"""
阶段一：WM811K 有监督单标签分类训练。
使用 ImageNet 预训练权重初始化 ResNet-18，cross-entropy 损失，类别平衡采样。
"""

import os
import json
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tasks.base import Task
from datasets.loaders import balanced_loader, standard_loader
from utils.metrics import classification_metrics
from utils.logging import get_tqdm_config, make_epoch_description, get_logger


class Stage1Classification(Task):
    """
    阶段一：WM811K 单标签有监督分类。
    训练完成后 backbone 权重可作为阶段二的初始化。
    """
    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 loss_function: nn.Module,
                 device: str = 'cpu',
                 checkpoint_dir: str = './checkpoints/stage1'):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger = get_logger('stage1', checkpoint_dir)

    def run(self, train_set, valid_set, epochs: int, batch_size: int,
            num_workers: int = 0, test_set=None):
        self.backbone.to(self.device)
        self.classifier.to(self.device)

        self.logger.info(f"Stage1 started | epochs={epochs} batch_size={batch_size} "
                         f"device={self.device} train={len(train_set)} valid={len(valid_set)}")

        train_loader = balanced_loader(train_set, batch_size, num_workers=num_workers)
        valid_loader = standard_loader(valid_set, batch_size, num_workers=num_workers, shuffle=False)

        best_valid_loss = float('inf')
        best_epoch = 0
        full_history = []  # 每个 epoch 的完整指标，最终写入 checkpoint

        with tqdm.tqdm(**get_tqdm_config(epochs, leave=True, color='blue')) as pbar:
            for epoch in range(1, epochs + 1):
                train_hist = self.train(train_loader)
                valid_hist = self.evaluate(valid_loader)

                lr = self.scheduler.get_last_lr()[0] if self.scheduler else \
                     self.optimizer.param_groups[0]['lr']

                epoch_history = {
                    'epoch': epoch,
                    'lr': lr,
                    'loss':   {'train': train_hist['loss'],   'valid': valid_hist['loss']},
                    'recall': {'train': train_hist['recall'], 'valid': valid_hist['recall']},
                    'f1':     {'train': train_hist['f1'],     'valid': valid_hist['f1']},
                }
                full_history.append(epoch_history)

                if valid_hist['loss'] < best_valid_loss:
                    best_valid_loss = valid_hist['loss']
                    best_epoch = epoch
                    self.save_checkpoint(self.best_ckpt, epoch=epoch,
                                         history=full_history)

                if self.scheduler is not None:
                    self.scheduler.step()

                desc = make_epoch_description(epoch_history, epoch, epochs, best_epoch)
                pbar.set_description_str(desc)
                pbar.update(1)
                self.logger.info(desc.strip())

        self.save_checkpoint(self.last_ckpt, epoch=epochs, history=full_history)
        self._save_history_json(full_history, 'train_history.json')
        self.logger.info(f"Stage1 finished | best_epoch={best_epoch} "
                         f"best_valid_loss={best_valid_loss:.4f}")

        if test_set is not None:
            test_loader = standard_loader(test_set, batch_size, num_workers=num_workers, shuffle=False)
            test_hist = self.evaluate(test_loader)
            self.logger.info(f"[Test] loss={test_hist['loss']:.4f} "
                             f"recall={test_hist['recall']:.4f} f1={test_hist['f1']:.4f}")
            self._save_history_json({'test': test_hist}, 'test_history.json')

    def train(self, data_loader: DataLoader) -> dict:
        self.backbone.train()
        self.classifier.train()
        total_loss, total_recall, total_f1, steps = 0., 0., 0., 0

        with tqdm.tqdm(**get_tqdm_config(len(data_loader), leave=False, color='green')) as pbar:
            for batch in data_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)

                logits = self.classifier(self.backbone(x))
                loss = self.loss_function(logits, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                m = classification_metrics(logits.detach(), y)
                total_recall += m['recall']
                total_f1     += m['f1']
                steps += 1
                pbar.set_description_str(
                    f" loss: {total_loss/steps:.4f} | recall: {total_recall/steps:.4f}"
                    f" | f1: {total_f1/steps:.4f}")
                pbar.update(1)

        return {'loss': total_loss / steps, 'recall': total_recall / steps, 'f1': total_f1 / steps}

    def evaluate(self, data_loader: DataLoader) -> dict:
        self.backbone.eval()
        self.classifier.eval()
        total_loss, total_recall, total_f1, steps = 0., 0., 0., 0

        with torch.no_grad():
            for batch in data_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                logits = self.classifier(self.backbone(x))
                loss = self.loss_function(logits, y)
                total_loss += loss.item()
                m = classification_metrics(logits, y)
                total_recall += m['recall']
                total_f1     += m['f1']
                steps += 1

        return {'loss': total_loss / steps, 'recall': total_recall / steps, 'f1': total_f1 / steps}

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone':   self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'scheduler':  self.scheduler.state_dict() if self.scheduler else None,
        }
        ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.classifier.load_state_dict(ckpt['classifier'])

    def _save_history_json(self, history, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
