# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class WM38K(Dataset):
    idx2label = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
    num_classes = len(idx2label)

    def __init__(
        self,
        npz_file: str,
        input_size: int = 96,
        split: str = 'test',
        train_ratio: float = 0.7,
        valid_ratio: float = 0.1,
        seed: int = 1993,
        max_samples: int = None,
        decouple_input: bool = True,
    ):
        super(WM38K, self).__init__()
        data = np.load(npz_file, allow_pickle=True)
        self.maps = self._pick_array(data, ('maps', 'x', 'X', 'images', 'arr_0')).astype(np.float32)
        self.labels = self._pick_array(data, ('labels', 'y', 'Y', 'targets', 'arr_1')).astype(np.float32)
        self.input_size = input_size
        self.decouple_input = decouple_input

        indices = self._make_split_indices(len(self.maps), split, train_ratio, valid_ratio, seed)
        if max_samples is not None:
            indices = indices[:max_samples]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = int(self.indices[idx])
        raw = torch.from_numpy(self.maps[original_idx]).float()
        raw = self._normalize_bins(raw)
        raw = self._resize_nearest(raw, self.input_size)
        x = self.decouple_mask(raw) if self.decouple_input else raw.unsqueeze(0)
        defect_mask = raw.eq(2).float()
        valid_mask = raw.gt(0).float()
        return {
            'x': x,
            'raw': raw,
            'defect_mask': defect_mask,
            'valid_mask': valid_mask,
            'y': torch.from_numpy(self.labels[original_idx]).float(),
            'idx': original_idx,
        }

    @staticmethod
    def _pick_array(npz, names):
        for name in names:
            if name in npz.files:
                return npz[name]
        raise KeyError(f'None of {names} found in npz file. Available keys: {npz.files}')

    @staticmethod
    def _make_split_indices(n, split, train_ratio, valid_ratio, seed):
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        if split == 'all':
            return indices
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        if split == 'train':
            return indices[:n_train]
        if split == 'valid':
            return indices[n_train:n_train + n_valid]
        if split == 'test':
            return indices[n_train + n_valid:]
        raise ValueError(f'Unknown split: {split}')

    @staticmethod
    def _normalize_bins(raw):
        # MixedWM38K often stores defect bins as 3. WaPIRL-style tensors use 2.
        return torch.where(raw.ge(3), torch.full_like(raw, 2.0), raw)

    @staticmethod
    def _resize_nearest(raw, size):
        if raw.shape[-2:] == (size, size):
            return raw
        x = raw.unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(size, size), mode='nearest')
        return x.squeeze(0).squeeze(0)

    @staticmethod
    def decouple_mask(x):
        valid = x.gt(0).float()
        defect = torch.clamp(x - 1, min=0., max=1.)
        return torch.stack([defect, valid], dim=0)

