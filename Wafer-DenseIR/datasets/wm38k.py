# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from shared.wm38k.manifest import manifest_valid_indices
except ImportError:
    manifest_valid_indices = None


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
        split_manifest: str = None,
        query_manifest: str = None,
    ):
        super(WM38K, self).__init__()
        data = np.load(npz_file, allow_pickle=True)
        maps = self._pick_array(data, ('maps', 'x', 'X', 'images', 'arr_0')).astype(np.float32)
        labels = self._pick_array(data, ('labels', 'y', 'Y', 'targets', 'arr_1')).astype(np.float32)
        valid_mask = labels.sum(axis=1) > 0
        self.original_indices = np.where(valid_mask)[0].astype(np.int64)
        self.maps = maps[valid_mask]
        self.labels = labels[valid_mask]
        self.input_size = input_size
        self.decouple_input = decouple_input

        indices = self._make_indices(
            len(self.maps),
            split=split,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
            split_manifest=split_manifest,
            query_manifest=query_manifest,
        )
        if max_samples is not None:
            indices = indices[:max_samples]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        valid_idx = int(self.indices[idx])
        original_idx = int(self.original_indices[valid_idx])
        raw = torch.from_numpy(self.maps[valid_idx]).float()
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
            'y': torch.from_numpy(self.labels[valid_idx]).float(),
            'idx': original_idx,
            'valid_idx': valid_idx,
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
    def _make_indices(n, split, train_ratio, valid_ratio, seed, split_manifest=None, query_manifest=None):
        if query_manifest is not None:
            if split_manifest is None:
                raise ValueError('query_manifest requires split_manifest so query ids can be mapped to valid indices.')
            query_ids = _load_query_sample_ids(query_manifest)
            rows = _load_manifest_rows(split_manifest, split='test')
            by_sample_id = {int(row['sample_id']): int(row['valid_index']) for row in rows}
            missing = [sample_id for sample_id in query_ids if sample_id not in by_sample_id]
            if missing:
                raise ValueError(f'{len(missing)} query ids are not in test split manifest. First missing: {missing[:5]}')
            return np.asarray([by_sample_id[sample_id] for sample_id in query_ids], dtype=np.int64)

        if split_manifest is not None:
            if manifest_valid_indices is not None:
                return manifest_valid_indices(split_manifest, split=split)
            rows = _load_manifest_rows(split_manifest, split=split)
            return np.asarray([int(row['valid_index']) for row in rows], dtype=np.int64)

        return WM38K._make_split_indices(n, split, train_ratio, valid_ratio, seed)

    @staticmethod
    def _normalize_bins(raw):
        # MixedWM38K often stores defect bins as 3. WaPIRL-style tensors use 2.
        return torch.where(raw.ge(3), torch.full_like(raw, 2.0), raw)

    @staticmethod
    def _resize_nearest(raw, size):
        if raw.shape[-2:] == (size, size):
            return raw
        h, w = raw.shape[-2:]
        if h <= size and w <= size:
            out = torch.zeros((size, size), dtype=raw.dtype, device=raw.device)
            r0 = (size - h) // 2
            c0 = (size - w) // 2
            out[r0:r0 + h, c0:c0 + w] = raw
            return out
        x = raw.unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(size, size), mode='nearest')
        return x.squeeze(0).squeeze(0)

    @staticmethod
    def decouple_mask(x):
        valid = x.gt(0).float()
        defect = torch.clamp(x - 1, min=0., max=1.)
        # DenseIR has no selected proposal; the full defect mask occupies that channel.
        return torch.stack([defect, defect, valid], dim=0)


def _load_manifest_rows(path, split=None):
    import csv
    with open(path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    if split is not None and split != 'all':
        rows = [row for row in rows if row['split'] == split]
    return rows


def _load_query_sample_ids(path):
    import csv
    with open(path, 'r', newline='') as f:
        return [int(row['sample_id']) for row in csv.DictReader(f)]
