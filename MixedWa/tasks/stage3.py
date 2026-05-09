# -*- coding: utf-8 -*-
"""
阶段三：生产数据自监督域适应（WaPIRL NCE Loss）。
- 冻结 backbone 前两层，解冻后两层 + 重新初始化 projection head
- 正样本对：(WDM, 由该 WDM 生成的伪 WBM)
- 记忆库用生产数据 embedding 初始化
"""

import os
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
from scipy.ndimage import gaussian_filter
from skimage.morphology import closing, disk
from skimage.filters import threshold_otsu
from skimage.transform import resize as sk_resize

from tasks.base import Task
from utils.loss import WaPIRLLoss
from utils.logging import get_tqdm_config, make_epoch_description, get_logger
from datasets.transforms import WaferTransform


# ---------------------------------------------------------------------------
# 伪 WBM 生成
# ---------------------------------------------------------------------------

def generate_pseudo_wbm(wdm: np.ndarray, out_size: int = 96) -> np.ndarray:
    """
    将 WDM 转换为伪 WBM：
      形态学闭运算 → 高斯模糊 → 下采样到 11×11 → 二值化 → 上采样回 out_size
    物理假设：缺陷密集区域对应芯片失效区域，pattern 形状在低分辨率下保持拓扑一致。
    """
    arr = wdm.astype(np.float32)
    # 形态学闭运算（填充稀疏缺陷点）
    closed = closing(arr > 0, disk(3)).astype(np.float32)
    # 高斯模糊（模拟芯片级空间平均效应）
    blurred = gaussian_filter(closed, sigma=2)
    # 下采样到 11×11
    small = sk_resize(blurred, (11, 11), order=0, anti_aliasing=False)
    # 二值化
    try:
        thresh = threshold_otsu(small)
    except Exception:
        thresh = 0.5
    binary = (small > thresh).astype(np.float32)
    # 上采样回 out_size
    pseudo = sk_resize(binary, (out_size, out_size), order=0, anti_aliasing=False)
    return (pseudo * 2).astype(np.uint8)  # 值域 {0, 2}，与 WBM 格式一致


# ---------------------------------------------------------------------------
# 生产数据 Dataset（WDM + 伪 WBM 对）
# ---------------------------------------------------------------------------

class ProductionWDMDataset(Dataset):
    """
    生产数据集：每个样本返回 (WDM tensor, 伪WBM tensor)。
    支持从 npz 文件或图像目录加载 WDM。
    """
    def __init__(self, wdm_arrays: np.ndarray, transform=None,
                 decouple_input: bool = True, img_size: int = 96):
        """
        Args:
            wdm_arrays: (N, H, W) numpy 数组，WDM 原始数据
            transform: 应用于 WDM 的变换（test 模式即可）
            decouple_input: 是否解耦为双通道
            img_size: 输出图像尺寸
        """
        self.wdm_arrays = wdm_arrays
        self.transform = transform or WaferTransform(size=(img_size, img_size), mode='test')
        self.decouple_input = decouple_input
        self.img_size = img_size

        # 预生成伪 WBM（避免训练时重复计算）
        print("Generating pseudo WBMs...")
        self.pseudo_wbms = np.stack([
            generate_pseudo_wbm(wdm, out_size=img_size)
            for wdm in tqdm.tqdm(wdm_arrays, dynamic_ncols=True)
        ])

    def __len__(self):
        return len(self.wdm_arrays)

    def __getitem__(self, idx):
        wdm_np = np.expand_dims(self.wdm_arrays[idx].astype(np.uint8), axis=2)
        pwbm_np = np.expand_dims(self.pseudo_wbms[idx], axis=2)

        x_wdm  = self.transform(wdm_np)
        x_pwbm = self.transform(pwbm_np)

        if self.decouple_input:
            from datasets.datasets import decouple_mask
            x_wdm  = decouple_mask(x_wdm)
            x_pwbm = decouple_mask(x_pwbm)

        return dict(x=x_wdm, x_t=x_pwbm, idx=idx)


# ---------------------------------------------------------------------------
# 记忆库
# ---------------------------------------------------------------------------

class MemoryBank:
    def __init__(self, size: tuple, device: str, weight: float = 0.5):
        self.size = size
        self.device = device
        self.weight = weight
        self.buffer = torch.zeros(*size, device=device)
        self.initialized = False

    def initialize(self, backbone: nn.Module, projector: nn.Module,
                   data_loader: DataLoader):
        backbone.eval()
        projector.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc='Initializing memory bank',
                                   dynamic_ncols=True):
                x = batch['x'].to(self.device)
                j = batch['idx']
                z = projector(backbone(x)).detach()
                self.buffer[j, :] = z
        self.initialized = True

    def update(self, indices, values: torch.Tensor):
        self.buffer[indices, :] = (
            self.weight * self.buffer[indices, :] +
            (1 - self.weight) * values.detach()
        )

    def get_representations(self, indices):
        return self.buffer[indices, :]

    def get_negatives(self, size: int, exclude):
        logits = torch.ones(self.buffer.size(0), device=self.device)
        logits[exclude] = 0
        return self.buffer[Categorical(logits=logits).sample(torch.Size([size])), :]

    def save(self, path: str):
        torch.save({'weight': self.weight, 'buffer': self.buffer.cpu()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.weight = ckpt['weight']
        self.buffer = ckpt['buffer'].to(self.device)
        self.initialized = True


# ---------------------------------------------------------------------------
# 阶段三 Task
# ---------------------------------------------------------------------------

class Stage3DomainAdaptation(Task):
    """
    阶段三：生产数据自监督域适应。
    冻结 backbone 前两层，解冻后两层 + projection head，用 WaPIRL NCE Loss 训练。
    """
    def __init__(self,
                 backbone: nn.Module,
                 projector: nn.Module,
                 memory: MemoryBank,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 loss_function: WaPIRLLoss,
                 num_negatives: int = 2000,
                 loss_weight: float = 0.5,
                 device: str = 'cpu',
                 checkpoint_dir: str = './checkpoints/stage3'):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.memory = memory
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.num_negatives = num_negatives
        self.loss_weight = loss_weight
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger = get_logger('stage3', checkpoint_dir)

    def run(self, train_set: Dataset, epochs: int, batch_size: int,
            num_workers: int = 0):
        self.backbone.to(self.device)
        self.projector.to(self.device)

        # 冻结前两层，解冻后两层
        self.backbone.freeze_layers(['layer1', 'layer2'])
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(ln) for ln in ['layer3', 'layer4']):
                param.requires_grad = True

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=False, drop_last=True)

        self.logger.info(f"Stage3 started | epochs={epochs} batch_size={batch_size} "
                         f"device={self.device} train={len(train_set)} "
                         f"num_negatives={self.num_negatives}")
        self.logger.info("Frozen layers: layer1, layer2 | Trainable: layer3, layer4, projector")

        # 初始化记忆库
        if not self.memory.initialized:
            self.memory.initialize(self.backbone, self.projector, train_loader)

        best_loss = float('inf')
        best_epoch = 0
        full_history = []

        with tqdm.tqdm(**get_tqdm_config(epochs, leave=True, color='blue')) as pbar:
            for epoch in range(1, epochs + 1):
                train_hist = self.train(train_loader)

                lr = self.scheduler.get_last_lr()[0] if self.scheduler else \
                     self.optimizer.param_groups[0]['lr']

                epoch_history = {
                    'epoch': epoch,
                    'lr': lr,
                    'loss': {'train': train_hist['loss'], 'valid': None},
                }
                full_history.append(epoch_history)

                if train_hist['loss'] < best_loss:
                    best_loss = train_hist['loss']
                    best_epoch = epoch
                    self.save_checkpoint(self.best_ckpt, epoch=epoch,
                                         history=full_history)
                    self.memory.save(os.path.join(self.checkpoint_dir, 'best_memory.pt'))

                if self.scheduler is not None:
                    self.scheduler.step()

                desc = make_epoch_description(epoch_history, epoch, epochs, best_epoch)
                pbar.set_description_str(desc)
                pbar.update(1)
                self.logger.info(desc.strip())

        self.save_checkpoint(self.last_ckpt, epoch=epochs, history=full_history)
        self.memory.save(os.path.join(self.checkpoint_dir, 'last_memory.pt'))
        self._save_history_json(full_history, 'train_history.json')
        self.logger.info(f"Stage3 finished | best_epoch={best_epoch} "
                         f"best_loss={best_loss:.4f}")

    def train(self, data_loader: DataLoader) -> dict:
        self.backbone.train()
        self.projector.train()
        total_loss, steps = 0., 0

        with tqdm.tqdm(**get_tqdm_config(len(data_loader), leave=False, color='green')) as pbar:
            for batch in data_loader:
                j    = batch['idx']
                x    = batch['x'].to(self.device)
                x_t  = batch['x_t'].to(self.device)

                # 合并前向，减少两次 backbone 调用
                z_concat = self.projector(self.backbone(torch.cat([x, x_t], dim=0)))
                z   = z_concat[:x.size(0)]
                z_t = z_concat[x.size(0):]

                m = self.memory.get_representations(j).to(self.device)
                negatives = self.memory.get_negatives(self.num_negatives, exclude=j)

                loss_z,   _ = self.loss_function(m, z,   negatives)
                loss_z_t, _ = self.loss_function(m, z_t, negatives)
                loss = (1 - self.loss_weight) * loss_z + self.loss_weight * loss_z_t

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.memory.update(j, z.detach())

                total_loss += loss.item()
                steps += 1
                pbar.set_description_str(f" loss: {total_loss/steps:.4f}")
                pbar.update(1)

        return {'loss': total_loss / steps}

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone':  self.backbone.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.projector.load_state_dict(ckpt['projector'])

    def _save_history_json(self, history, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
