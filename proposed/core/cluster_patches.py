# -*- coding: utf-8 -*-
"""Fixed-size masked patch construction from immutable proposal tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from proposed.core.proposal import ClusterToken


@dataclass(frozen=True)
class PatchConfig:
    patch_size: int = 64
    channels: int = 3


@dataclass
class PatchSample:
    x: np.ndarray
    map_id: int
    token_id: int
    center_row: float
    center_col: float


class PatchBuilder:
    def __init__(self, config: PatchConfig):
        self.config = config

    def build(self, raw_map: np.ndarray, token: ClusterToken) -> PatchSample:
        local_defect = (raw_map == 2).astype(np.float32)
        valid = ((raw_map == 1) | (raw_map == 2)).astype(np.float32)
        proposal = np.zeros_like(local_defect, dtype=np.float32)
        for r, c in token.pixels:
            if 0 <= r < proposal.shape[0] and 0 <= c < proposal.shape[1]:
                proposal[r, c] = 1.0
        channels = np.stack([local_defect, proposal, valid], axis=0)
        patch = _crop_centered(channels, token.centroid_row, token.centroid_col, self.config.patch_size)
        return PatchSample(
            x=patch.astype(np.float32),
            map_id=token.map_id,
            token_id=token.token_id,
            center_row=token.centroid_row,
            center_col=token.centroid_col,
        )


def augment_patch(x: np.ndarray, rng: np.random.Generator, rotate: bool = True, max_shift: int = 3,
                  noise_prob: float = 0.01, dropout_prob: float = 0.02) -> np.ndarray:
    out = np.array(x, copy=True)
    if rotate:
        out = np.rot90(out, int(rng.integers(0, 4)), axes=(1, 2)).copy()
    if max_shift > 0:
        dr = int(rng.integers(-max_shift, max_shift + 1))
        dc = int(rng.integers(-max_shift, max_shift + 1))
        out = _shift_zero(out, dr, dc)
    if dropout_prob > 0:
        keep = rng.random(out.shape[1:]) >= dropout_prob
        out[0] *= keep
        out[1] *= keep
    if noise_prob > 0:
        noise = rng.random(out.shape[1:]) < noise_prob
        out[0] = np.maximum(out[0], noise.astype(np.float32))
    return out.astype(np.float32)


def _crop_centered(channels: np.ndarray, center_row: float, center_col: float, patch_size: int) -> np.ndarray:
    _, h, w = channels.shape
    size = int(patch_size)
    half = size // 2
    cr = int(round(center_row))
    cc = int(round(center_col))
    r0 = cr - half
    c0 = cc - half
    out = np.zeros((channels.shape[0], size, size), dtype=np.float32)
    src_r0 = max(r0, 0)
    src_c0 = max(c0, 0)
    src_r1 = min(r0 + size, h)
    src_c1 = min(c0 + size, w)
    dst_r0 = src_r0 - r0
    dst_c0 = src_c0 - c0
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[:, dst_r0:dst_r0 + (src_r1 - src_r0), dst_c0:dst_c0 + (src_c1 - src_c0)] = channels[:, src_r0:src_r1, src_c0:src_c1]
    return out
def _shift_zero(x: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(x)
    _, h, w = x.shape
    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)
    dst_r0 = max(0, dr)
    dst_c0 = max(0, dc)
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[:, dst_r0:dst_r0 + (src_r1 - src_r0), dst_c0:dst_c0 + (src_c1 - src_c0)] = x[:, src_r0:src_r1, src_c0:src_c1]
    return out
