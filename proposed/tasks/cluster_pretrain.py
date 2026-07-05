# -*- coding: utf-8 -*-
"""Cluster-level WaPIRL-style pretraining."""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm


class MemoryBank:
    def __init__(self, size: tuple, device, weight: float = 0.5):
        self.size = size
        self.device = device
        self.weight = float(weight)
        self.buffer = torch.zeros(*size, device=device)
        self.initialized = False

    @torch.no_grad()
    def initialize(self, encoder: nn.Module, projector: nn.Module, data_loader: DataLoader):
        encoder.eval()
        projector.eval()
        non_blocking = self.device.type == 'cuda'
        for batch in tqdm(data_loader, desc='Initializing memory', leave=False):
            x = batch['x'].to(self.device, non_blocking=non_blocking)
            idx = batch['idx'].long().to(self.device, non_blocking=non_blocking)
            self.buffer[idx, :] = projector(encoder(x)).detach()
        self.initialized = True

    @torch.no_grad()
    def update(self, index, values):
        index = index.long().to(self.device)
        self.buffer[index, :] = self.weight * self.buffer[index, :] + (1.0 - self.weight) * values

    def get_representations(self, index):
        return self.buffer[index.long().to(self.device), :]

    def get_negatives(self, size: int, exclude):
        probs = torch.ones(self.buffer.size(0), device=self.device)
        probs[exclude.long().to(self.device)] = 0
        size = min(int(size), int((probs > 0).sum().item()))
        if size <= 0:
            return self.buffer[:0, :]
        sampled = Categorical(probs=probs).sample(torch.Size([size]))
        return self.buffer[sampled, :]

    def save(self, path, **kwargs):
        ckpt = {'weight': self.weight, 'buffer': self.buffer.detach().cpu()}
        ckpt.update(kwargs)
        torch.save(ckpt, path)


class ClusterPretrainTask:
    def __init__(self, encoder, projector, memory, optimizer, scheduler, loss_function, device, output_dir,
                 loss_weight=0.5, num_negatives=1024):
        self.encoder = encoder.to(device)
        self.projector = projector.to(device)
        self.memory = memory
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.output_dir = output_dir
        self.loss_weight = float(loss_weight)
        self.num_negatives = int(num_negatives)
        os.makedirs(output_dir, exist_ok=True)

    def run(self, train_set, valid_set, epochs: int, batch_size: int, num_workers: int = 0, logger=None, save_every: int = 25):
        use_cuda = self.device.type == 'cuda'
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=use_cuda, persistent_workers=num_workers > 0)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=use_cuda, persistent_workers=num_workers > 0)
        if len(train_set) < 2:
            raise ValueError('Cluster pretraining needs at least 2 train cluster tokens.')
        if not self.memory.initialized:
            self.memory.initialize(self.encoder, self.projector, train_loader)

        best_valid_loss = float('inf')
        best_epoch = 0
        history = []
        for epoch in range(1, int(epochs) + 1):
            train_history = self._run_epoch(train_loader, train=True)
            valid_history = self._run_epoch(valid_loader, train=False) if len(valid_set) else {'loss': 0.0, 'top1': 0.0}
            epoch_history = {
                'epoch': epoch,
                'loss': {'train': train_history['loss'], 'valid': valid_history['loss']},
                'top1': {'train': train_history['top1'], 'valid': valid_history['top1']},
            }
            history.append(epoch_history)
            valid_loss = epoch_history['loss']['valid']
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                self.save_checkpoint(self.best_ckpt, **epoch_history)
                self.memory.save(os.path.join(self.output_dir, 'best_memory.pt'), epoch=epoch)
            if save_every and epoch % int(save_every) == 0:
                self.save_checkpoint(os.path.join(self.output_dir, f'epoch_{epoch:04d}.loss_{valid_loss:.4f}.pt'), **epoch_history)
            if self.scheduler is not None:
                self.scheduler.step()
            msg = f'Epoch {epoch}/{epochs} best={best_epoch} train_loss={train_history["loss"]:.4f} valid_loss={valid_loss:.4f} valid_top1={valid_history["top1"]:.4f}'
            print(msg)
            if logger is not None:
                logger.info(msg)
        self.save_checkpoint(self.last_ckpt, **history[-1])
        self.memory.save(os.path.join(self.output_dir, 'last_memory.pt'), epoch=epochs)
        with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    def _run_epoch(self, data_loader, train: bool):
        self.encoder.train(train)
        self.projector.train(train)
        out = {'loss': 0.0, 'top1': 0.0}
        steps = max(len(data_loader), 1)
        non_blocking = self.device.type == 'cuda'
        iterator = tqdm(data_loader, desc='train' if train else 'valid', leave=False)
        for batch in iterator:
            idx = batch['idx'].long().to(self.device, non_blocking=non_blocking)
            x = batch['x'].to(self.device, non_blocking=non_blocking)
            x_t = batch['x_t'].to(self.device, non_blocking=non_blocking)
            if train:
                z_concat = self.projector(self.encoder(torch.cat([x, x_t], dim=0)))
            else:
                with torch.no_grad():
                    z_concat = self.projector(self.encoder(torch.cat([x, x_t], dim=0)))
            z = z_concat[:x.size(0)]
            z_t = z_concat[x.size(0):]
            negatives = self.memory.get_negatives(self.num_negatives, exclude=idx)
            if train:
                anchors = self.memory.get_representations(idx)
                loss_z, _ = self.loss_function(anchors=anchors, positives=z, negatives=negatives)
                loss_z_t, logits = self.loss_function(anchors=anchors, positives=z_t, negatives=negatives)
                loss = (1.0 - self.loss_weight) * loss_z + self.loss_weight * loss_z_t
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.memory.update(idx, z.detach())
            else:
                with torch.no_grad():
                    loss, logits = self.loss_function(anchors=z, positives=z_t, negatives=negatives)
            out['loss'] += float(loss.item())
            out['top1'] += _top1(logits)
        return {k: v / steps for k, v in out.items()}

    def save_checkpoint(self, path, **kwargs):
        ckpt = {
            'encoder': self.encoder.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        ckpt.update(kwargs)
        torch.save(ckpt, path)

    @property
    def best_ckpt(self):
        return os.path.join(self.output_dir, 'best_model.pt')

    @property
    def last_ckpt(self):
        return os.path.join(self.output_dir, 'last_model.pt')


def _top1(logits):
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return float((logits.argmax(dim=1) == targets).float().mean().item())
