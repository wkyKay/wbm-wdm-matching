# -*- coding: utf-8 -*-

import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

from datasets.wm38k import WM38K


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proposed.core.cluster_patches import augment_patch


class WM38KForWaPIRL(WM38K):
    def __init__(
        self,
        npz_file: str,
        input_size: int = 96,
        split: str = 'train',
        train_ratio: float = 0.7,
        valid_ratio: float = 0.1,
        seed: int = 1993,
        max_samples: int = None,
        decouple_input: bool = True,
        quality_filter: bool = True,
        min_defect_pixels: int = 3,
        min_defect_ratio: float = 1e-4,
        max_defect_ratio: float = 0.8,
        min_valid_ratio: float = 0.05,
        max_valid_ratio: float = 0.98,
        deduplicate: bool = True,
        split_manifest: str = None,
        query_manifest: str = None,
    ):
        super(WM38KForWaPIRL, self).__init__(
            npz_file=npz_file,
            input_size=input_size,
            split=split,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
            max_samples=max_samples,
            decouple_input=decouple_input,
            split_manifest=split_manifest,
            query_manifest=query_manifest,
        )
        self.seed = int(seed)
        self.indices = self._filter_indices(
            self.indices,
            quality_filter=quality_filter,
            min_defect_pixels=min_defect_pixels,
            min_defect_ratio=min_defect_ratio,
            max_defect_ratio=max_defect_ratio,
            min_valid_ratio=min_valid_ratio,
            max_valid_ratio=max_valid_ratio,
            deduplicate=deduplicate,
        )

    def __getitem__(self, idx):
        sample = super(WM38KForWaPIRL, self).__getitem__(idx)
        patch = sample['x'].numpy()
        rng = np.random.default_rng(self.seed + int(idx) * 1009)
        sample['x'] = torch.from_numpy(augment_patch(patch, rng))
        sample['x_t'] = torch.from_numpy(augment_patch(patch, rng))
        sample['idx'] = idx
        sample['original_idx'] = int(self.indices[idx])
        return sample

    def _filter_indices(self, indices, quality_filter=True, min_defect_pixels=3, min_defect_ratio=1e-4,
                        max_defect_ratio=0.8, min_valid_ratio=0.05, max_valid_ratio=0.98,
                        deduplicate=True):
        if not quality_filter and not deduplicate:
            return indices

        kept = []
        seen = set()
        for original_idx in indices:
            raw = self._normalize_bins(torch.from_numpy(self.maps[int(original_idx)]).float())
            raw = self._resize_nearest(raw, self.input_size)
            valid = raw.gt(0)
            defect = raw.eq(2)
            valid_ratio = float(valid.float().mean().item())
            defect_pixels = int(defect.sum().item())
            defect_ratio = float(defect_pixels / max(int(valid.sum().item()), 1))

            if quality_filter:
                if valid_ratio < min_valid_ratio or valid_ratio > max_valid_ratio:
                    continue
                if defect_pixels < min_defect_pixels:
                    continue
                if defect_ratio < min_defect_ratio or defect_ratio > max_defect_ratio:
                    continue

            if deduplicate:
                key = hashlib.sha1(raw.numpy().astype(np.uint8).tobytes()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)

            kept.append(int(original_idx))
        return np.asarray(kept, dtype=np.int64)
