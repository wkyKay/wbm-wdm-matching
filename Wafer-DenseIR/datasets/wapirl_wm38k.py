# -*- coding: utf-8 -*-

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from datasets.wm38k import WM38K


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
        augmentation: str = 'crop_noise_rotate',
        crop_min_scale: float = 0.85,
        noise_prob: float = 0.002,
        rotate_prob: float = 0.5,
        quality_filter: bool = True,
        min_defect_pixels: int = 3,
        min_defect_ratio: float = 1e-4,
        max_defect_ratio: float = 0.8,
        min_valid_ratio: float = 0.05,
        max_valid_ratio: float = 0.98,
        deduplicate: bool = True,
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
        )
        self.augmentation = augmentation
        self.crop_min_scale = crop_min_scale
        self.noise_prob = noise_prob
        self.rotate_prob = rotate_prob
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
        raw_t = self._augment_raw(sample['raw'].float())
        x_t = self.decouple_mask(raw_t) if self.decouple_input else raw_t.unsqueeze(0)
        sample['x_t'] = x_t
        sample['idx'] = idx
        sample['original_idx'] = int(self.indices[idx])
        return sample

    def _augment_raw(self, raw):
        out = raw
        if self.augmentation in ('crop', 'crop_noise', 'crop_rotate', 'crop_noise_rotate'):
            out = self._random_resized_crop(out, self.crop_min_scale)
        if self.augmentation in ('crop_rotate', 'crop_noise_rotate') and torch.rand(()) < self.rotate_prob:
            out = torch.rot90(out, int(torch.randint(0, 4, ()).item()), dims=(0, 1))
        if self.augmentation in ('crop_noise', 'crop_noise_rotate') and self.noise_prob > 0:
            out = self._inject_sparse_noise(out, self.noise_prob)
        return out

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

    @staticmethod
    def _random_resized_crop(raw, min_scale):
        h, w = raw.shape
        min_scale = min(max(min_scale, 0.1), 1.0)
        scale = float(torch.empty(()).uniform_(min_scale, 1.0).item())
        crop_h = max(1, int(round(h * scale)))
        crop_w = max(1, int(round(w * scale)))
        top = int(torch.randint(0, h - crop_h + 1, ()).item()) if crop_h < h else 0
        left = int(torch.randint(0, w - crop_w + 1, ()).item()) if crop_w < w else 0
        crop = raw[top:top + crop_h, left:left + crop_w]
        out = F.interpolate(crop[None, None].float(), size=(h, w), mode='nearest')
        return out.squeeze(0).squeeze(0)

    @staticmethod
    def _inject_sparse_noise(raw, prob):
        valid = raw.gt(0)
        if not valid.any():
            return raw
        out = raw.clone()
        flip = torch.rand_like(out.float()).lt(prob) & valid
        out[flip] = torch.where(out[flip].eq(2), torch.ones_like(out[flip]), torch.full_like(out[flip], 2.0))
        return out
