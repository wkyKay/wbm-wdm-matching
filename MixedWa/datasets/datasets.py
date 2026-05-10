# -*- coding: utf-8 -*-
"""
Dataset 类，直接从原始 pkl/npz 文件读取数据，无需预先处理为图像文件。
也支持从已处理的图像目录读取（与 WaPIRL 兼容）。
"""

import os
import glob
import pathlib

import numpy as np
import torch
import cv2
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# 8 个缺陷类别（WM38K 多标签）
DEFECT_CLASSES = ['center', 'donut', 'edge-loc', 'edge-ring', 'loc', 'random', 'scratch', 'near-full']
CLASS2IDX = {c: i for i, c in enumerate(DEFECT_CLASSES)}
NUM_CLASSES = len(DEFECT_CLASSES)  # 8


def decouple_mask(x: torch.Tensor) -> torch.Tensor:
    """
    将单通道 WBM tensor 解耦为 [defect_map, existence_mask] 双通道。
    缺陷格 = 2，正常格 = 1，空格 = 0
    """
    m = x.gt(0).float()
    x = torch.clamp(x - 1, min=0., max=1.)
    return torch.cat([x, m], dim=0)


# ---------------------------------------------------------------------------
# WM811K：从原始 pkl 直接读取（单标签，9 类）
# ---------------------------------------------------------------------------

class WM811KRaw(Dataset):
    """
    直接从 LSWMD.pkl 读取 WM811K 数据，无需预处理为图像文件。
    用于阶段一有监督训练。
    标签：9 类单标签（center/donut/edge-loc/edge-ring/loc/random/scratch/near-full/none）
    """
    LABEL2IDX = {
        'center': 0, 'donut': 1, 'edge-loc': 2, 'edge-ring': 3,
        'loc': 4, 'random': 5, 'scratch': 6, 'near-full': 7, 'none': 8,
    }
    IDX2LABEL = list(LABEL2IDX.keys())
    NUM_CLASSES = 9
    IMG_SIZE = (96, 96)

    def __init__(self, pkl_file: str, split: str = 'train',
                 transform=None, decouple_input: bool = True,
                 proportion: float = 1.0, seed: int = 0,
                 img_size: int = 96):
        """
        Args:
            pkl_file: LSWMD.pkl 路径
            split: 'train' | 'valid' | 'test'（仅有标签数据）
            transform: 图像变换
            decouple_input: 是否解耦为双通道
            proportion: 使用数据的比例
        """
        super().__init__()
        self.transform = transform
        self.decouple_input = decouple_input
        self.img_size = img_size

        data = pd.read_pickle(pkl_file)
        data['labelString'] = data['failureType'].apply(self._get_label)
        data['trainTestLabel'] = data['trianTestLabel'].apply(self._get_split)

        # 只取有标签数据
        labeled = data[data['trainTestLabel'] != -1].copy()
        labeled = labeled[labeled['labelString'] != '-'].copy()

        # 按 split 划分
        indices = labeled.index.tolist()
        labels = labeled['labelString'].tolist()
        train_idx, temp_idx, _, temp_lbl = train_test_split(
            indices, labels, test_size=0.2, stratify=labels,
            shuffle=True, random_state=2015010720,
        )
        valid_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=temp_lbl,
            shuffle=True, random_state=2015010720,
        )
        split_map = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        use_idx = split_map[split]

        self.samples = [
            (data.loc[i, 'waferMap'], self.LABEL2IDX[data.loc[i, 'labelString']])
            for i in use_idx
            if data.loc[i, 'labelString'] in self.LABEL2IDX
        ]

        if proportion < 1.0:
            n = max(1, int(len(self.samples) * proportion))
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(self.samples), n, replace=False)
            self.samples = [self.samples[i] for i in idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wmap, y = self.samples[idx]
        x = self._to_input(wmap, self.img_size)
        if self.transform is not None:
            x = self.transform(x)
        if self.decouple_input:
            x = decouple_mask(x)
        return dict(x=x, y=y, idx=idx)

    @staticmethod
    def _to_input(wmap: np.ndarray, img_size: int = 96) -> np.ndarray:
        """将晶圆图 resize 到 img_size×img_size，返回 (H, W, 1) uint8 numpy 数组。"""
        arr = wmap.astype(np.uint8)
        arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(arr, axis=2)

    @staticmethod
    def _get_label(x):
        if len(x) == 1:
            return x[0][0].strip().lower()
        return '-'

    @staticmethod
    def _get_split(x):
        d = {'unlabeled': -1, 'training': 0, 'test': 1}
        if len(x) == 1:
            return d.get(x[0][0].strip().lower(), -1)
        return -1


# ---------------------------------------------------------------------------
# WM38K：从原始 npz 直接读取（多标签，8 类）
# ---------------------------------------------------------------------------

class WM38KRaw(Dataset):
    """
    直接从 Wafer_Map_Datasets.npz 读取 WM38K 数据。
    用于阶段二多标签微调。
    标签：8 类多热编码 [center, donut, edge-loc, edge-ring, loc, random, scratch, near-full]
    """
    CLASS_NAMES = DEFECT_CLASSES
    NUM_CLASSES = NUM_CLASSES
    IMG_SIZE = (96, 96)

    def __init__(self, npz_file: str, split: str = 'train',
                 transform=None, shift_transform=None,
                 decouple_input: bool = True,
                 train_ratio: float = 0.8, seed: int = 0):
        """
        Args:
            npz_file: Wafer_Map_Datasets.npz 路径
            split: 'train' | 'valid' | 'test'
            transform: 主变换
            shift_transform: 平移变换（用于位置感知负样本，阶段二必选）
            decouple_input: 是否解耦为双通道
            train_ratio: 训练集比例（默认 0.8，剩余各半为 valid/test）
        """
        super().__init__()
        self.transform = transform
        self.shift_transform = shift_transform
        self.decouple_input = decouple_input

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

        x_np = self._to_input(wmap)

        if self.transform is not None:
            x = self.transform(x_np)
        else:
            x = torch.from_numpy(x_np.transpose(2, 0, 1)).float() / 255.0

        # 位置感知负样本：对同一张图做大幅平移
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
    def _to_input(wmap: np.ndarray) -> np.ndarray:
        """将晶圆图 resize 到 96×96，返回 (H, W, 1) uint8 numpy 数组。"""
        arr = wmap.astype(np.uint8)
        arr = cv2.resize(arr, (96, 96), interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(arr, axis=2)


# ---------------------------------------------------------------------------
# 从已处理图像目录读取（与 WaPIRL 兼容，用于阶段一/二）
# ---------------------------------------------------------------------------

class WM811KFromDir(Dataset):
    """从 WaPIRL 处理后的图像目录读取 WM811K（单标签）。"""
    LABEL2IDX = WM811KRaw.LABEL2IDX
    NUM_CLASSES = 9

    def __init__(self, root: str, transform=None, decouple_input: bool = True,
                 proportion: float = 1.0, seed: int = 0):
        super().__init__()
        self.transform = transform
        self.decouple_input = decouple_input

        images = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))
        samples = []
        for img_path in images:
            label = pathlib.PurePath(img_path).parent.name
            if label in self.LABEL2IDX:
                samples.append((img_path, self.LABEL2IDX[label]))

        if proportion < 1.0:
            labels = [s[1] for s in samples]
            samples, _ = train_test_split(
                samples, train_size=proportion, stratify=labels,
                shuffle=True, random_state=1993 + seed,
            )
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x_np = np.expand_dims(img, axis=2)
        if self.transform is not None:
            x = self.transform(x_np)
        else:
            x = torch.from_numpy(x_np.transpose(2, 0, 1)).float() / 255.0
        if self.decouple_input:
            x = decouple_mask(x)
        return dict(x=x, y=y, idx=idx)


class WM38KFromDir(Dataset):
    """从 WaPIRL 处理后的图像目录读取 WM38K（多标签）。"""
    CLASS_NAMES = DEFECT_CLASSES
    NUM_CLASSES = NUM_CLASSES

    def __init__(self, root: str, transform=None, shift_transform=None,
                 decouple_input: bool = True):
        super().__init__()
        self.transform = transform
        self.shift_transform = shift_transform
        self.decouple_input = decouple_input

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
        x_np = np.expand_dims(img, axis=2)
        y = torch.tensor(label_vec, dtype=torch.float32)

        if self.transform is not None:
            x = self.transform(x_np)
        else:
            x = torch.from_numpy(x_np.transpose(2, 0, 1)).float() / 255.0

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
        vec = [0.0] * NUM_CLASSES
        for part in label_str.split('_'):
            if part in CLASS2IDX:
                vec[CLASS2IDX[part]] = 1.0
        return vec
