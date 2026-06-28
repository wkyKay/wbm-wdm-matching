# -*- coding: utf-8 -*-

import json
import os

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
import tqdm

from tasks.base import Task
from utils.logging import make_epoch_description


class MemoryBank(object):
    def __init__(self, size: tuple, device, weight: float = 0.5):
        self.size = size
        self.device = device
        self.weight = weight
        self.buffer = torch.zeros(*size, device=device)
        self.initialized = False

    @torch.no_grad()
    def initialize(self, backbone: nn.Module, projector: nn.Module, data_loader: DataLoader):
        backbone.eval()
        projector.eval()
        non_blocking = self.device.type == 'cuda'
        for batch in tqdm.tqdm(data_loader, desc='Initializing memory', leave=False):
            x = batch['x'].to(self.device, non_blocking=non_blocking)
            idx = batch['idx'].long().to(self.device, non_blocking=non_blocking)
            self.buffer[idx, :] = projector(backbone(x)).detach()
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
        size = min(size, int((probs > 0).sum().item()))
        if size == 0:
            return self.buffer[:0, :]
        sampled = Categorical(probs=probs).sample(torch.Size([size]))
        return self.buffer[sampled, :]

    def save(self, path: str, **kwargs):
        ckpt = {'weight': self.weight, 'buffer': self.buffer.detach().cpu()}
        ckpt.update(kwargs)
        torch.save(ckpt, path)


class WaPIRLPretrain(Task):
    def __init__(self, backbone, projector, memory, optimizer, scheduler, loss_function, device,
                 output_dir, loss_weight=0.5, num_negatives=1024, write_summary=False):
        super(WaPIRLPretrain, self).__init__()
        self.backbone = backbone.to(device)
        self.projector = projector.to(device)
        self.memory = memory
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.output_dir = output_dir
        self.loss_weight = loss_weight
        self.num_negatives = num_negatives
        self.writer = self._make_writer(output_dir) if write_summary else None

    def run(self, train_set, valid_set, epochs: int, batch_size: int, num_workers: int = 0,
            logger=None, save_every: int = 25):
        use_cuda = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )

        if len(train_set) < 2:
            raise ValueError('WaPIRL pretraining needs at least 2 training samples after filtering.')
        if not self.memory.initialized:
            self.memory.initialize(self.backbone, self.projector, train_loader)

        best_valid_loss = float('inf')
        best_epoch = 0
        history = []
        for epoch in range(1, epochs + 1):
            train_history = self.train(train_loader)
            valid_history = self.evaluate(valid_loader)
            epoch_history = {
                'loss': {'train': train_history['loss'], 'valid': valid_history['loss']},
                'top1': {'train': train_history['top1'], 'valid': valid_history['top1']},
            }
            history.append({'epoch': epoch, **epoch_history})

            if self.writer is not None:
                self.writer.add_scalars('loss', epoch_history['loss'], epoch)
                self.writer.add_scalars('top1', epoch_history['top1'], epoch)
                if self.scheduler is not None:
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], epoch)

            valid_loss = epoch_history['loss']['valid']
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                self.memory.save(os.path.join(self.output_dir, 'best_memory.pt'), epoch=epoch)

            if save_every and epoch % save_every == 0:
                path = os.path.join(self.output_dir, f'epoch_{epoch:04d}.loss_{valid_loss:.4f}.pt')
                self.save_checkpoint(path, epoch=epoch, **epoch_history)

            if self.scheduler is not None:
                self.scheduler.step()

            desc = make_epoch_description(epoch_history, current=epoch, total=epochs, best=best_epoch)
            print(desc)
            if logger is not None:
                logger.info(desc)

        self.save_checkpoint(self.last_ckpt, epoch=epochs, **epoch_history)
        self.memory.save(os.path.join(self.output_dir, 'last_memory.pt'), epoch=epochs)
        with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    def train(self, data_loader):
        self._set_learning_phase(True)
        return self._run_epoch(data_loader, train=True)

    @torch.no_grad()
    def evaluate(self, data_loader):
        self._set_learning_phase(False)
        return self._run_epoch(data_loader, train=False)

    def _run_epoch(self, data_loader, train: bool):
        out = {'loss': 0.0, 'top1': 0.0}
        steps = max(len(data_loader), 1)
        non_blocking = self.device.type == 'cuda'
        iterator = tqdm.tqdm(data_loader, desc='train' if train else 'valid', leave=False)
        for batch in iterator:
            idx = batch['idx'].long().to(self.device, non_blocking=non_blocking)
            x = batch['x'].to(self.device, non_blocking=non_blocking)
            x_t = batch['x_t'].to(self.device, non_blocking=non_blocking)
            z_concat = self.predict(torch.cat([x, x_t], dim=0))
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
                loss, logits = self.loss_function(anchors=z, positives=z_t, negatives=negatives)

            out['loss'] += float(loss.item())
            out['top1'] += self._top1(logits)
            iterator.set_postfix(loss=out['loss'] / max(iterator.n, 1), top1=out['top1'] / max(iterator.n, 1))
        return {k: v / steps for k, v in out.items()}

    def predict(self, x):
        return self.projector(self.backbone(x))

    def _set_learning_phase(self, train: bool):
        self.backbone.train(train)
        self.projector.train(train)

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone': self.backbone.state_dict(),
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

    @staticmethod
    def _top1(logits):
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return float((logits.argmax(dim=1) == targets).float().mean().item())

    @staticmethod
    def _make_writer(output_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print('TensorBoard is not installed; continuing without summary writing.')
            return None
        return SummaryWriter(output_dir)
