# -*- coding: utf-8 -*-
"""
Dataset 类，直接从原始 npz 文件或图像目录读取 WM38K 数据。
WM38K 包含单类、两类、三类组合的多标签晶圆图，是主训练数据集。
"""

import os
import glob
import pathlib

import numpy as np
import torch
import cv2

from torch.utils.data import Dataset


# 8 个缺陷类别（WM38K 多标签）
DEFECT_CLASSES = ['center', 'donut', 'edge-loc', 'edge-ring', 'loc', 'random', 'scratch', 'near-full']
CLASS2IDX = {c: i for i, c in enumerate(DEFECT_CLASSES)}
NUM_CLASSES = len(DEFECT_CLASSES)  # 8


def decouple_mask(x: torch.Tensor) -> torch.Tensor:
    """
    将单通道 WBM tensor 解耦为 [defect_map, existence_mask] 双通道。
    值域：缺陷格=2，正常格=1，背景=0
      channel 0（缺陷图）：clamp(x-1, 0, 1)，缺陷格→1，其余→0
      channel 1（存在掩码）：x>0，非背景格→1
    """
    m = x.gt(0).float()
    d = torch.clamp(x - 1, min=0., max=1.)
    return torch.cat([d, m], dim=0)


# ---------------------------------------------------------------------------
# WM38K：从原始 npz 直接读取（多标签，8 类）
# ---------------------------------------------------------------------------

class WM38KRaw(Dataset):
    """
    直接从 Wafer_Map_Datasets.npz 读取 WM38K 数据。
    标签：8 类多热编码 [center, donut, edge-loc, edge-ring, loc, random, scratch, near-full]
    包含单类、两类、三类组合 pattern。
    """
    CLASS_NAMES = DEFECT_CLASSES
    NUM_CLASSES = NUM_CLASSES

    def __init__(self, npz_file: str, split: str = 'train',
                 transform=None, shift_transform=None,
                 decouple_input: bool = True,
                 train_ratio: float = 0.8, seed: int = 0,
                 img_size: int = 96):
        """
        Args:
            npz_file:        Wafer_Map_Datasets.npz 路径
            split:           'train' | 'valid' | 'test'
            transform:       主变换（训练时用 crop，验证/测试用 test）
            shift_transform: 平移变换（用于位置感知负样本）
            decouple_input:  是否解耦为双通道
            train_ratio:     训练集比例（默认 0.8，剩余各半为 valid/test）
            img_size:        resize 目标尺寸
        """
        super().__init__()
        self.transform = transform
        self.shift_transform = shift_transform
        self.decouple_input = decouple_input
        self.img_size = img_size

        data = np.load(npz_file)
        wafer_maps = data['arr_0']   # (N, H, W)
        labels = data['arr_1']       # (N, 8) 多热编码

        # 过滤掉全零标签（unknown）
        valid_mask = labels.sum(axis=1) > 0
        wafer_maps = wafer_maps[valid_mask]
        labels = labels[valid_mask]

        n = len(wafer_maps)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)

        n_train = int(n * train_ratio)
        n_valid = int(n * (1 - train_ratio) / 2)
        train_idx = perm[:n_train]
        valid_idx = perm[n_train:n_train + n_valid]
        test_idx  = perm[n_train + n_valid:]

        split_map = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        use_idx = split_map[split]

        self.wafer_maps = wafer_maps[use_idx]
        self.labels = labels[use_idx].astype(np.float32)

    def __len__(self):
        return len(self.wafer_maps)

    def __getitem__(self, idx):
        wmap = self.wafer_maps[idx]
        y = torch.from_numpy(self.labels[idx])  # (8,) float32 多热

        x_np = self._to_input(wmap, self.img_size)

        if self.transform is not None:
            x = self.transform(x_np)
        else:
            x = torch.from_numpy(x_np.transpose(2, 0, 1)).float()

        x_shift = None
        if self.shift_transform is not None:
            x_shift = self.shift_transform(x_np)

        if self.decouple_input:
            x = decouple_mask(x)
            if x_shift is not None:
                x_shift = decouple_mask(x_shift)

        result = dict(x=x, y=y, idx=idx)
        if x_shift is not None:
            result['x_shift'] = x_shift
        return result

    @staticmethod
    def _to_input(wmap: np.ndarray, img_size: int = 96) -> np.ndarray:
        """将晶圆图 resize 到 img_size×img_size，返回 (H, W, 1) uint8 numpy 数组。"""
        arr = wmap.astype(np.uint8)
        arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(arr, axis=2)


# ---------------------------------------------------------------------------
# WM38K：从图像目录读取（目录名即为多标签字符串，如 center_edge-ring_loc）
# ---------------------------------------------------------------------------

class WM38KFromDir(Dataset):
    """
    从图像目录读取 WM38K（多标签）。
    目录结构：root/{label_str}/*.png，label_str 为下划线分隔的类别组合。
    例：center_edge-ring_loc 表示同时含 center、edge-ring、loc 三类 pattern。
    """
    CLASS_NAMES = DEFECT_CLASSES
    NUM_CLASSES = NUM_CLASSES

    def __init__(self, root: str, transform=None, shift_transform=None,
                 decouple_input: bool = True, img_size: int = 96):
        super().__init__()
        self.transform = transform
        self.shift_transform = shift_transform
        self.decouple_input = decouple_input
        self.img_size = img_size

        images = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))
        self.samples = []
        for img_path in images:
            label_str = pathlib.PurePath(img_path).parent.name
            label_vec = self._parse_label(label_str)
            self.samples.append((img_path, label_vec))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_vec = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x_np = np.expand_dims(
            cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST),
            axis=2,
        )
        y = torch.tensor(label_vec, dtype=torch.float32)

        if self.transform is not None:
            x = self.transform(x_np)
        else:
            x = torch.from_numpy(x_np.transpose(2, 0, 1)).float()

        x_shift = None
        if self.shift_transform is not None:
            x_shift = self.shift_transform(x_np)

        if self.decouple_input:
            x = decouple_mask(x)
            if x_shift is not None:
                x_shift = decouple_mask(x_shift)

        result = dict(x=x, y=y, idx=idx)
        if x_shift is not None:
            result['x_shift'] = x_shift
        return result

    @classmethod
    def _parse_label(cls, label_str: str) -> list:
        """将目录名解析为多热向量，例如 'center_edge-ring' → [1,0,0,1,0,0,0,0]。"""
        vec = [0.0] * NUM_CLASSES
        for part in label_str.split('_'):
            if part in CLASS2IDX:
                vec[CLASS2IDX[part]] = 1.0
        return vec
