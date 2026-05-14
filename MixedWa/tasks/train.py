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
from utils.metrics import multilabel_metrics, DEFAULT_METRICS, build_metrics
from utils.logging import get_tqdm_config, make_epoch_description, get_logger
import numpy as np

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
                 patience: int = 15,
                 metrics: dict = None):
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
        self.metrics = metrics if metrics is not None else build_metrics(DEFAULT_METRICS)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger = get_logger('train', checkpoint_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard'))

    def run(self, train_set, valid_set, epochs: int, batch_size: int,
            num_workers: int = 0, test_set=None, shift_transform=None):
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
        best_valid_map = 0.0
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
                    **{
                        name: {'train': train_hist[name], 'valid': valid_hist[name]}
                        for name in self.metrics
                    },
                }
                full_history.append(epoch_history)

                if valid_hist['loss'] < best_valid_loss:
                    best_valid_loss = valid_hist['loss']

                if valid_hist['mAP'] > best_valid_map:
                    best_valid_map = valid_hist['mAP']
                    best_epoch = epoch
                    no_improve = 0
                    self.save_checkpoint(self.best_ckpt, epoch=epoch, history=full_history)
                else:
                    no_improve += 1

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(valid_hist['mAP'])
                    else:
                        self.scheduler.step()

                # TensorBoard
                self.writer.add_scalars('loss', {'train': train_hist['loss'], 'valid': valid_hist['loss']}, epoch)
                for name in self.metrics:
                    self.writer.add_scalars(name, {'train': train_hist[name], 'valid': valid_hist[name]}, epoch)
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
            f"best_valid_mAP={best_valid_map:.4f} best_valid_loss={best_valid_loss:.4f}"
        )

        if test_set is not None:
            test_loader = standard_loader(test_set, batch_size, num_workers=num_workers, shuffle=False)
            test_hist = self.evaluate(test_loader, shift_transform=shift_transform)
            self.logger.info(
                f"[Test] loss={test_hist['loss']:.4f} mAP={test_hist['mAP']:.4f} "
                f"f1_macro={test_hist['f1_macro']:.4f} f1_micro={test_hist['f1_micro']:.4f} "
                f"exact_match={test_hist['exact_match']:.4f} hamming_acc={test_hist['hamming_acc']:.4f}"
            )
            if 'shift_dist' in test_hist:
                self.logger.info(
                    f"[Test] shift_dist={test_hist['shift_dist']:.4f} "
                    f"shift_false_accept={test_hist['shift_false_accept']:.4f}"
                )
            self._save_history_json({'test': test_hist}, 'test_history.json')
            self.writer.add_scalars('test', {k: v for k, v in test_hist.items() if isinstance(v, float)}, 0)

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
        scores = {name: metric(all_logits, all_targets) for name, metric in self.metrics.items()}
        return {'loss': total_loss / steps, **scores}

    def evaluate(self, data_loader: DataLoader, shift_transform=None) -> dict:
        self.backbone.eval()
        self.classifier.eval()
        total_loss, all_logits, all_targets = 0., [], []
        shift_dists, shift_false_accepts = [], []
        steps = 0

        with torch.no_grad():
            for batch in data_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                feat_orig = self.backbone(x)
                logits = self.classifier(feat_orig)
                loss = self.loss_function.bce(logits, y)
                total_loss += loss.item()
                all_logits.append(logits.cpu())
                all_targets.append(y.cpu())

                # 位置敏感性评估：对每张图动态生成平移版本
                if shift_transform is not None:
                    x_np_list = batch.get('x_np')  # 原始 numpy，若 dataset 提供
                    if x_np_list is None:
                        # 从 tensor 反推：直接对 tensor 做平移（近似）
                        x_shift = torch.roll(x, shifts=int(x.shape[-1] * 0.3), dims=-1)
                    else:
                        x_shift = torch.stack([
                            shift_transform(xn) for xn in x_np_list
                        ]).to(self.device)
                        from datasets.datasets import decouple_mask
                        if x.shape[1] == 2:
                            x_shift = torch.stack([decouple_mask(s) for s in x_shift])

                    feat_shift = self.backbone(x_shift)
                    z_orig  = F.normalize(feat_orig,  dim=1)
                    z_shift = F.normalize(feat_shift, dim=1)

                    # 平均余弦距离（越大越位置敏感）
                    dist = (1.0 - F.cosine_similarity(z_orig, z_shift, dim=1))
                    shift_dists.extend(dist.cpu().tolist())

                    # 位置误接受率：平移后与原图的相似度 > 同类其他图的相似度 → 误接受
                    # 用 batch 内同标签对的相似度作为参考基准
                    sim_orig  = F.cosine_similarity(z_orig.unsqueeze(1),
                                                    z_orig.unsqueeze(0), dim=2)  # (B, B)
                    label_match = (y.unsqueeze(1) * y.unsqueeze(0)).sum(dim=2) > 0  # (B, B) 同标签掩码
                    diag = torch.eye(x.shape[0], dtype=torch.bool, device=self.device)
                    label_match = label_match & ~diag  # 排除自身

                    if label_match.any():
                        # 对每个样本：平移版本的相似度 > 同类最高相似度 → 误接受
                        same_class_max = (sim_orig * label_match.float()).max(dim=1).values
                        self_shift_sim  = F.cosine_similarity(z_orig, z_shift, dim=1)
                        false_accept = (self_shift_sim > same_class_max).float()
                        shift_false_accepts.extend(false_accept.cpu().tolist())

                steps += 1

        all_logits  = torch.cat(all_logits,  dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        scores = {name: metric(all_logits, all_targets) for name, metric in self.metrics.items()}
        result = {'loss': total_loss / steps, **scores}

        if shift_dists:
            result['shift_dist']         = float(np.mean(shift_dists))
            result['shift_false_accept'] = float(np.mean(shift_false_accepts)) if shift_false_accepts else 0.0

        return result

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
