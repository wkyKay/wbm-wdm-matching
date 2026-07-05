# -*- coding: utf-8 -*-
"""Deterministic wafer-map transformations for Experiment B."""

from __future__ import annotations

import numpy as np


def normalize_bins(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw).copy()
    raw[raw >= 3] = 2
    return raw.astype(np.uint8)


def valid_mask(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw) > 0


def defect_mask(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw) == 2


def compose_map(valid: np.ndarray, defect: np.ndarray) -> np.ndarray:
    out = np.zeros(valid.shape, dtype=np.uint8)
    out[valid] = 1
    out[valid & defect] = 2
    return out


def transform_map(raw: np.ndarray, transform_type: str, rng: np.random.Generator) -> np.ndarray:
    raw = normalize_bins(raw)
    if transform_type == 'identity':
        return raw.copy()
    if transform_type == 'rot_90':
        return _rotate_k(raw, 1)
    if transform_type == 'rot_180':
        return _rotate_k(raw, 2)
    if transform_type == 'shift_mild':
        return _shift(raw, max(1, int(round(raw.shape[0] * 0.04))), max(1, int(round(raw.shape[1] * -0.03))))
    if transform_type == 'shift_strong':
        return _shift(raw, max(1, int(round(raw.shape[0] * 0.14))), max(1, int(round(raw.shape[1] * -0.12))))
    if transform_type == 'scale_mild':
        return _scale_defects(raw, 1.10)
    if transform_type == 'noise_mild':
        return _add_noise(raw, rng, ratio=0.03)
    if transform_type == 'noise_strong':
        return _add_noise(raw, rng, ratio=0.15)
    if transform_type == 'dropout_mild':
        return _dropout(raw, rng, ratio=0.10)
    if transform_type == 'dropout_strong':
        return _dropout(raw, rng, ratio=0.45)
    if transform_type == 'cluster_extra':
        return _add_extra_cluster(raw, rng)
    if transform_type == 'cluster_dropout':
        return _drop_largest_component(raw)
    raise ValueError(f'Unknown transform_type: {transform_type}')


def _rotate_k(raw: np.ndarray, k: int) -> np.ndarray:
    valid = np.rot90(valid_mask(raw), k)
    defect = np.rot90(defect_mask(raw), k)
    return compose_map(valid, defect)


def _shift(raw: np.ndarray, dr: int, dc: int) -> np.ndarray:
    valid = valid_mask(raw)
    defect = defect_mask(raw)
    shifted = _shift_bool(defect, dr, dc) & valid
    return compose_map(valid, shifted)


def _shift_bool(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    r_src0 = max(0, -dr)
    r_src1 = min(h, h - dr)
    c_src0 = max(0, -dc)
    c_src1 = min(w, w - dc)
    r_dst0 = max(0, dr)
    r_dst1 = min(h, h + dr)
    c_dst0 = max(0, dc)
    c_dst1 = min(w, w + dc)
    if r_src0 < r_src1 and c_src0 < c_src1:
        out[r_dst0:r_dst1, c_dst0:c_dst1] = mask[r_src0:r_src1, c_src0:c_src1]
    return out


def _scale_defects(raw: np.ndarray, factor: float) -> np.ndarray:
    valid = valid_mask(raw)
    defect = defect_mask(raw)
    coords = np.argwhere(defect)
    if len(coords) == 0:
        return raw.copy()
    center = np.asarray(defect.shape, dtype=np.float32) / 2.0
    scaled = np.rint((coords.astype(np.float32) - center) * factor + center).astype(np.int64)
    keep = (
        (scaled[:, 0] >= 0) & (scaled[:, 0] < raw.shape[0]) &
        (scaled[:, 1] >= 0) & (scaled[:, 1] < raw.shape[1])
    )
    scaled = scaled[keep]
    out_defect = np.zeros_like(defect)
    out_defect[scaled[:, 0], scaled[:, 1]] = True
    return compose_map(valid, out_defect & valid)


def _add_noise(raw: np.ndarray, rng: np.random.Generator, ratio: float) -> np.ndarray:
    valid = valid_mask(raw)
    defect = defect_mask(raw)
    valid_coords = np.argwhere(valid & ~defect)
    n = max(1, int(round(max(int(defect.sum()), 1) * ratio)))
    if len(valid_coords) == 0:
        return raw.copy()
    chosen = valid_coords[rng.choice(len(valid_coords), size=min(n, len(valid_coords)), replace=False)]
    out_defect = defect.copy()
    out_defect[chosen[:, 0], chosen[:, 1]] = True
    return compose_map(valid, out_defect)


def _dropout(raw: np.ndarray, rng: np.random.Generator, ratio: float) -> np.ndarray:
    valid = valid_mask(raw)
    defect = defect_mask(raw)
    coords = np.argwhere(defect)
    if len(coords) == 0:
        return raw.copy()
    n = min(len(coords), max(1, int(round(len(coords) * ratio))))
    drop = coords[rng.choice(len(coords), size=n, replace=False)]
    out_defect = defect.copy()
    out_defect[drop[:, 0], drop[:, 1]] = False
    return compose_map(valid, out_defect)


def _add_extra_cluster(raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    valid = valid_mask(raw)
    defect = defect_mask(raw)
    h, w = raw.shape
    radius = max(2, int(round(min(h, w) * 0.045)))
    candidates = np.argwhere(valid & ~defect)
    if len(candidates) == 0:
        return raw.copy()
    center = candidates[int(rng.integers(0, len(candidates)))]
    rr, cc = np.ogrid[:h, :w]
    blob = (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2
    out_defect = defect | (blob & valid)
    return compose_map(valid, out_defect)


def _drop_largest_component(raw: np.ndarray) -> np.ndarray:
    valid = valid_mask(raw)
    defect = defect_mask(raw)
    labels, count = _connected_components(defect)
    if count == 0:
        return raw.copy()
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    out_defect = defect & (labels != largest)
    if not out_defect.any():
        return _dropout(raw, np.random.default_rng(0), ratio=0.45)
    return compose_map(valid, out_defect)


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        if labels[r, c] != 0:
            continue
        current += 1
        stack = [(int(r), int(c))]
        labels[r, c] = current
        while stack:
            rr, cc = stack.pop()
            for nr in (rr - 1, rr, rr + 1):
                for nc in (cc - 1, cc, cc + 1):
                    if nr == rr and nc == cc:
                        continue
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current
                        stack.append((nr, nc))
    return labels, current
