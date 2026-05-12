# -*- coding: utf-8 -*-
"""
主训练任务：WM38K 多标签分类 + 位置感知训练。

直接从 ImageNet 预训练权重初始化 backbone，在 WM38K（含单类/两类/三类组合）
上训练多标签分类器。损失 = BCE + λ * margin_loss(z_orig, z_shift)。
支持 early stopping 和 ReduceLROnPlateau 调度器。
"""

import os
import json
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tasks.base import Task
from datasets.loaders import standard_loader
from utils.loss import PositionAwareLoss
from utils.metrics import mean_ap
from utils.logging import get_tqdm_config, make_epoch_description, get_logger


class WM38KTrainer(Task):
    """
    WM38K 多标签分类训练器。
    从 ImageNet 预训练权重出发，直接在 WM38K 上训练，无需 WM811K 预训练阶段。
    """
    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 loss_function: PositionAwareLoss,
                 device: str = 'cpu',
                 checkpoint_dir: str = './checkpoints/train',
                 freeze_layers: list = None,
                 patience: int = 15):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.freeze_layers = freeze_layers or []
        self.patience = patience
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger = get_logger('train', checkpoint_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard'))

    def run(self, train_set, valid_set, epochs: int, batch_size: int,
            num_workers: int = 0, test_set=None):
        self.backbone.to(self.device)
        self.classifier.to(self.device)

        if self.freeze_layers:
            self.backbone.freeze_layers(self.freeze_layers)
            self.logger.info(f"Frozen layers: {self.freeze_layers}")

        self.logger.info(
            f"Training started | epochs={epochs} batch_size={batch_size} "
            f"device={self.device} train={len(train_set)} valid={len(valid_set)}"
        )

        train_loader = standard_loader(train_set, batch_size, num_workers=num_workers, shuffle=True)
        valid_loader = standard_loader(valid_set, batch_size, num_workers=num_workers, shuffle=False)

        best_valid_loss = float('inf')
        best_epoch = 0
        no_improve = 0
        full_history = []

        with tqdm.tqdm(**get_tqdm_config(epochs, leave=True, color='blue')) as pbar:
            for epoch in range(1, epochs + 1):
                train_hist = self.train(train_loader)
                valid_hist = self.evaluate(valid_loader)

                lr = self.optimizer.param_groups[0]['lr']

                epoch_history = {
                    'epoch': epoch,
                    'lr': lr,
                    'loss': {'train': train_hist['loss'], 'valid': valid_hist['loss']},
                    'mAP':  {'train': train_hist['mAP'],  'valid': valid_hist['mAP']},
                }
                full_history.append(epoch_history)

                if valid_hist['loss'] < best_valid_loss:
                    best_valid_loss = valid_hist['loss']
                    best_epoch = epoch
                    no_improve = 0
                    self.save_checkpoint(self.best_ckpt, epoch=epoch, history=full_history)
                else:
                    no_improve += 1

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(valid_hist['loss'])
                    else:
                        self.scheduler.step()

                # TensorBoard
                self.writer.add_scalars('loss', {'train': train_hist['loss'], 'valid': valid_hist['loss']}, epoch)
                self.writer.add_scalars('mAP',  {'train': train_hist['mAP'],  'valid': valid_hist['mAP']},  epoch)
                self.writer.add_scalar('lr', lr, epoch)

                desc = make_epoch_description(epoch_history, epoch, epochs, best_epoch)
                pbar.set_description_str(desc)
                pbar.update(1)
                self.logger.info(desc.strip())

                if no_improve >= self.patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        self.save_checkpoint(self.last_ckpt, epoch=epoch, history=full_history)
        self._save_history_json(full_history, 'train_history.json')
        self.logger.info(
            f"Training finished | best_epoch={best_epoch} "
            f"best_valid_loss={best_valid_loss:.4f}"
        )

        if test_set is not None:
            test_loader = standard_loader(test_set, batch_size, num_workers=num_workers, shuffle=False)
            test_hist = self.evaluate(test_loader)
            self.logger.info(f"[Test] loss={test_hist['loss']:.4f} mAP={test_hist['mAP']:.4f}")
            self._save_history_json({'test': test_hist}, 'test_history.json')
            self.writer.add_scalars('test', test_hist, 0)

        self.writer.close()

    def train(self, data_loader: DataLoader) -> dict:
        self.backbone.train()
        self.classifier.train()
        total_loss, all_logits, all_targets = 0., [], []
        steps = 0

        with tqdm.tqdm(**get_tqdm_config(len(data_loader), leave=False, color='green')) as pbar:
            for batch in data_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)

                feat_orig = self.backbone(x)
                logits = self.classifier(feat_orig)

                if 'x_shift' in batch:
                    x_shift = batch['x_shift'].to(self.device)
                    feat_shift = self.backbone(x_shift)
                    z_orig  = F.normalize(feat_orig,  dim=1)
                    z_shift = F.normalize(feat_shift, dim=1)
                    loss = self.loss_function(logits, y, z_orig, z_shift)
                else:
                    loss = self.loss_function.bce(logits, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())
                steps += 1
                pbar.set_description_str(f" loss: {total_loss/steps:.4f}")
                pbar.update(1)

        all_logits  = torch.cat(all_logits,  dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return {'loss': total_loss / steps, 'mAP': mean_ap(all_logits, all_targets)}

    def evaluate(self, data_loader: DataLoader) -> dict:
        self.backbone.eval()
        self.classifier.eval()
        total_loss, all_logits, all_targets = 0., [], []
        steps = 0

        with torch.no_grad():
            for batch in data_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                logits = self.classifier(self.backbone(x))
                loss = self.loss_function.bce(logits, y)
                total_loss += loss.item()
                all_logits.append(logits.cpu())
                all_targets.append(y.cpu())
                steps += 1

        all_logits  = torch.cat(all_logits,  dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return {'loss': total_loss / steps, 'mAP': mean_ap(all_logits, all_targets)}

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
