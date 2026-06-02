# -*- coding: utf-8 -*-
"""
生成 mixed-pattern synthetic WDM 数据。

用法（从 MixedWa/ 目录运行）：
  python process_mixed_synthetic_wdm.py \
    --wm811k_pkl ../../data/wm811k/LSWMD.pkl \
    --output_dir ../../data/synthetic_wdm_mixed \
    --num_samples 5000 \
    --save_preview

输出：
  output_dir/synthetic_wdm.npz   # arr_0: (N,H,W), 值域 {0,1,2}
                                  # 0=背景黑，1=晶圆有效区灰，2=缺陷白
  output_dir/metadata.json       # 每张 synthetic WDM 的 pattern 组合与生成参数
  output_dir/preview/*.png       # 可选预览图
"""

import os
import sys
import json
import argparse
import importlib

import cv2
import tqdm
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    # 允许从 MixedWa/ 目录直接运行 processors 内脚本，并导入项目模块。
    sys.path.insert(0, ROOT)

from datasets.datasets import DEFECT_CLASSES


DEFAULT_RECIPES = [
    ['edge-ring', 'scratch'],
    ['center', 'edge-ring'],
    ['scratch', 'loc'],
    ['donut', 'scratch'],
    ['edge-loc', 'loc'],
    ['center', 'loc'],
    ['donut', 'edge-ring'],
    ['edge-ring', 'scratch', 'loc'],
    ['center', 'edge-ring', 'scratch'],
    ['center', 'loc', 'scratch'],
    ['edge-ring', 'random'],
    ['scratch', 'random'],
    ['center', 'random'],
]


def parse_args():
    p = argparse.ArgumentParser('Process mixed-pattern synthetic WDM data')
    p.add_argument('--wm811k_pkl', type=str,
                   default='/Users/kayw/Documents/trae_projects/match-test/data/wm811k/LSWMD.pkl',
                   help='WM811K LSWMD.pkl 文件路径')
    p.add_argument('--output_dir', type=str, required=True,
                   help='输出目录')
    p.add_argument('--num_samples', type=int, default=5000)
    p.add_argument('--out_size', type=int, default=96)
    p.add_argument('--generation_mode', type=str, default='probability_map',
                   choices=['sparse_mask', 'probability_map'],
                   help='sparse_mask 保留旧逻辑；probability_map 将 WBM mask 转为概率场后采样 WDM 点云')
    p.add_argument('--min_components', type=int, default=2)
    p.add_argument('--max_components', type=int, default=3)
    p.add_argument('--primary_keep_min', type=float, default=0.5)
    p.add_argument('--primary_keep_max', type=float, default=0.9)
    p.add_argument('--secondary_keep_min', type=float, default=0.2)
    p.add_argument('--secondary_keep_max', type=float, default=0.6)
    p.add_argument('--jitter', type=float, default=1.5)
    p.add_argument('--dropout', type=float, default=0.05)
    p.add_argument('--dilation_prob', type=float, default=0.3)
    p.add_argument('--noise_density', type=float, default=0.002)
    p.add_argument('--max_density', type=float, default=0.20)
    p.add_argument('--min_density', type=float, default=0.002,
                   help='probability_map 模式下单个 component 的最低采样密度')
    p.add_argument('--core_weight', type=float, default=0.60,
                   help='probability_map: resized defect core 权重 α')
    p.add_argument('--blur_weight', type=float, default=0.35,
                   help='probability_map: blurred neighborhood 权重 β')
    p.add_argument('--background_weight', type=float, default=0.05,
                   help='probability_map: wafer background prior 权重 γ')
    p.add_argument('--blur_sigma_ratio', type=float, default=0.015,
                   help='probability_map: Gaussian sigma = out_size * ratio')
    p.add_argument('--cluster_prob', type=float, default=0.35,
                   help='probability_map: 对采样点做局部扩散的概率')
    p.add_argument('--background_noise_for_random', type=float, default=0.01,
                   help='recipe 含 random 时额外提高背景噪声密度')
    p.add_argument('--recipe_mode', type=str, default='default',
                   choices=['default', 'random'],
                   help='default 使用预设 pattern 组合；random 从样本中随机抽 component')
    p.add_argument('--save_preview', action='store_true')
    p.add_argument('--preview_limit', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def get_wm811k_label_string(value) -> str:
    """解析 WM811K failureType 字段，逻辑与 WaPIRL/process_wm811k.py 保持一致。"""
    if len(value) != 1:
        return '-'
    return value[0][0].strip().lower()


def install_legacy_pandas_pickle_aliases():
    """
    WM811K 的 LSWMD.pkl 常由旧 pandas 版本生成，pickle 中可能引用
    pandas.indexes.*。新版 pandas 已将这些模块移动到 pandas.core.indexes.*，
    直接 pd.read_pickle 会触发 ModuleNotFoundError: pandas.indexes。
    """
    aliases = {
        'pandas.indexes': 'pandas.core.indexes',
        'pandas.indexes.base': 'pandas.core.indexes.base',
        'pandas.indexes.numeric': 'pandas.core.indexes.numeric',
        'pandas.indexes.range': 'pandas.core.indexes.range',
        'pandas.indexes.multi': 'pandas.core.indexes.multi',
        'pandas.indexes.datetimes': 'pandas.core.indexes.datetimes',
        'pandas.indexes.timedeltas': 'pandas.core.indexes.timedeltas',
        'pandas.indexes.period': 'pandas.core.indexes.period',
    }
    for old_name, new_name in aliases.items():
        if old_name in sys.modules:
            continue
        try:
            sys.modules[old_name] = importlib.import_module(new_name)
        except ModuleNotFoundError:
            pass


def load_wm811k_pkl(pkl_file: str):
    install_legacy_pandas_pickle_aliases()
    data = pd.read_pickle(pkl_file)
    samples = []

    for idx, row in data.iterrows():
        label = get_wm811k_label_string(row['failureType'])
        if label not in DEFECT_CLASSES:
            continue

        wafer_map = np.asarray(row['waferMap'])
        if wafer_map.ndim != 2:
            continue
        defect = wafer_map == 2
        if defect.sum() == 0:
            continue

        samples.append({
            'mask': defect.astype(np.uint8),
            'labels': [label],
            'source_id': int(idx),
        })
    return samples


def load_source_samples(args):
    samples = load_wm811k_pkl(args.wm811k_pkl)
    if len(samples) == 0:
        raise ValueError('未加载到有效 pattern source samples')
    return samples, 'wm811k_pkl'


def resize_mask(mask: np.ndarray, out_size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def random_affine(mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    h, w = mask.shape
    angle = rng.uniform(-180, 180)
    scale = rng.uniform(0.85, 1.15)
    tx = rng.uniform(-0.08, 0.08) * w
    ty = rng.uniform(-0.08, 0.08) * h
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += [tx, ty]
    return cv2.warpAffine(mask.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sparse_sample(mask: np.ndarray, keep_ratio: float, rng: np.random.RandomState) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    keep = rng.rand(len(xs)) < keep_ratio
    out = np.zeros_like(mask, dtype=np.uint8)
    out[ys[keep], xs[keep]] = 1
    return out


def coordinate_jitter(mask: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    if sigma <= 0:
        return mask.astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    ys = ys + np.rint(rng.normal(0, sigma, size=len(ys))).astype(int)
    xs = xs + np.rint(rng.normal(0, sigma, size=len(xs))).astype(int)
    ys = np.clip(ys, 0, mask.shape[0] - 1)
    xs = np.clip(xs, 0, mask.shape[1] - 1)

    out = np.zeros_like(mask, dtype=np.uint8)
    out[ys, xs] = 1
    return out


def local_diffusion(mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    kernel_size = int(rng.choice([2, 3]))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def gaussian_blur_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.astype(np.float32)
    k = int(max(3, round(sigma * 6) | 1))
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)


def normalize_probability(prob: np.ndarray, wafer_mask: np.ndarray) -> np.ndarray:
    prob = np.maximum(prob.astype(np.float64), 0.0)
    prob = prob * (wafer_mask > 0)
    total = prob.sum()
    if total <= 0:
        prob = wafer_mask.astype(np.float64)
        total = prob.sum()
    return prob / max(total, 1e-12)


def sample_from_probability(prob: np.ndarray, num_points: int,
                            rng: np.random.RandomState) -> np.ndarray:
    if num_points <= 0:
        return np.zeros(prob.shape, dtype=np.uint8)
    flat = prob.reshape(-1)
    replace = num_points > np.count_nonzero(flat)
    indices = rng.choice(flat.size, size=num_points, replace=replace, p=flat)
    ys, xs = np.unravel_index(indices, prob.shape)
    out = np.zeros(prob.shape, dtype=np.uint8)
    out[ys, xs] = 1
    return out


def build_probability_map(mask: np.ndarray, wafer_mask: np.ndarray, args) -> np.ndarray:
    """
    将放大后的 WBM defect mask 转换为 WDM 缺陷点采样概率场。

    core 保留原 pattern 拓扑；blurred neighborhood 允许真实 WDM 式空间扩散；
    background prior 只提供极低概率背景散点，避免全 wafer 均匀采样破坏 pattern。
    """
    core = mask.astype(np.float32)
    sigma = max(1.0, args.out_size * args.blur_sigma_ratio)
    blurred = gaussian_blur_mask(core, sigma=sigma)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()

    background = wafer_mask.astype(np.float32)
    prob = (args.core_weight * core +
            args.blur_weight * blurred +
            args.background_weight * background)
    return normalize_probability(prob, wafer_mask)


def random_dropout(mask: np.ndarray, dropout: float, rng: np.random.RandomState) -> np.ndarray:
    if dropout <= 0:
        return mask.astype(np.uint8)
    keep = rng.rand(*mask.shape) >= dropout
    return np.logical_and(mask > 0, keep).astype(np.uint8)


def add_background_noise(mask: np.ndarray, density: float, rng: np.random.RandomState) -> np.ndarray:
    if density <= 0:
        return mask.astype(np.uint8)
    noise = rng.rand(*mask.shape) < density
    return np.logical_or(mask > 0, noise).astype(np.uint8)


def limit_density(mask: np.ndarray, max_density: float, rng: np.random.RandomState) -> np.ndarray:
    if max_density <= 0 or mask.mean() <= max_density:
        return mask.astype(np.uint8)

    ys, xs = np.where(mask > 0)
    target_n = max(1, int(max_density * mask.size))
    if len(xs) <= target_n:
        return mask.astype(np.uint8)

    keep_idx = rng.choice(len(xs), size=target_n, replace=False)
    out = np.zeros_like(mask, dtype=np.uint8)
    out[ys[keep_idx], xs[keep_idx]] = 1
    return out


def circular_wafer_mask(size: int) -> np.ndarray:
    """生成圆形晶圆有效区域：圆外背景=0，圆内正常区域=1。"""
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    radius = size * 0.48
    dist2 = (yy - center) ** 2 + (xx - center) ** 2
    return (dist2 <= radius ** 2).astype(np.uint8)


def stylize_component(sample: dict, out_size: int, keep_ratio: float, args,
                      rng: np.random.RandomState) -> np.ndarray:
    mask = resize_mask(sample['mask'], out_size)
    mask = random_affine(mask, rng)
    if args.generation_mode == 'sparse_mask':
        mask = sparse_sample(mask, keep_ratio, rng)
        mask = coordinate_jitter(mask, args.jitter, rng)
        if rng.rand() < args.dilation_prob:
            mask = local_diffusion(mask, rng)
        mask = random_dropout(mask, args.dropout, rng)
        return mask.astype(np.uint8)

    wafer_mask = circular_wafer_mask(out_size)
    prob = build_probability_map(mask, wafer_mask, args)
    target_density = rng.uniform(args.min_density, args.max_density) * keep_ratio
    num_points = max(1, int(target_density * wafer_mask.sum()))
    sampled = sample_from_probability(prob, num_points, rng)
    sampled = coordinate_jitter(sampled, args.jitter, rng)
    if rng.rand() < args.cluster_prob:
        sampled = local_diffusion(sampled, rng)
    sampled = random_dropout(sampled, args.dropout, rng)
    sampled = np.logical_and(sampled > 0, wafer_mask > 0).astype(np.uint8)
    return sampled.astype(np.uint8)


def build_label_index(samples):
    index = {}
    for sample in samples:
        for label in sample['labels']:
            index.setdefault(label, []).append(sample)
    return index


def sample_by_label(label_index, label: str, rng: np.random.RandomState):
    candidates = label_index.get(label, [])
    if len(candidates) == 0:
        return None
    return candidates[rng.randint(len(candidates))]


def sample_components(samples, label_index, args, rng: np.random.RandomState):
    if args.recipe_mode == 'default':
        recipe = DEFAULT_RECIPES[rng.randint(len(DEFAULT_RECIPES))]
        selected = []
        for label in recipe:
            sample = sample_by_label(label_index, label, rng)
            if sample is not None:
                selected.append((label, sample))
        if len(selected) >= args.min_components:
            return selected[:args.max_components]

    n_components = rng.randint(args.min_components, args.max_components + 1)
    indices = rng.choice(len(samples), size=n_components, replace=len(samples) < n_components)
    selected = []
    for idx in indices:
        sample = samples[idx]
        label = sample['labels'][rng.randint(len(sample['labels']))]
        selected.append((label, sample))
    return selected


def synthesize_mixed_wdm(samples, label_index, args, rng: np.random.RandomState):
    components = sample_components(samples, label_index, args, rng)
    final_mask = np.zeros((args.out_size, args.out_size), dtype=np.uint8)
    metadata_components = []

    for i, (label, sample) in enumerate(components):
        role = 'primary' if i == 0 else 'secondary'
        if role == 'primary':
            keep_ratio = rng.uniform(args.primary_keep_min, args.primary_keep_max)
        else:
            keep_ratio = rng.uniform(args.secondary_keep_min, args.secondary_keep_max)

        comp = stylize_component(sample, args.out_size, keep_ratio, args, rng)
        final_mask = np.logical_or(final_mask > 0, comp > 0).astype(np.uint8)
        metadata_components.append({
            'role': role,
            'label': label,
            'source_labels': list(sample['labels']),
            'source_id': sample['source_id'],
            'keep_ratio': float(keep_ratio),
            'generation_mode': args.generation_mode,
        })

    labels = []
    for comp in metadata_components:
        if comp['label'] not in labels:
            labels.append(comp['label'])

    noise_density = args.noise_density
    if 'random' in labels:
        noise_density = max(noise_density, args.background_noise_for_random)
    final_mask = add_background_noise(final_mask, noise_density, rng)
    final_mask = limit_density(final_mask, args.max_density, rng)
    wafer_mask = circular_wafer_mask(args.out_size)
    final_mask = np.logical_and(final_mask > 0, wafer_mask > 0).astype(np.uint8)
    wdm = wafer_mask.astype(np.uint8)
    wdm[final_mask > 0] = 2

    metadata = {
        'patterns': labels,
        'primary_pattern': metadata_components[0]['label'],
        'component_count': len(metadata_components),
        'density': float(final_mask.mean()),
        'noise_density': float(noise_density),
        'generation_mode': args.generation_mode,
        'components': metadata_components,
    }
    return wdm, metadata


def save_preview_image(arr: np.ndarray, path: str):
    img = ((arr.astype(np.float32) / 2.0) * 255).astype(np.uint8)
    cv2.imwrite(path, img)


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    samples, source_format = load_source_samples(args)
    label_index = build_label_index(samples)
    print(f'Loaded {len(samples)} source samples from {source_format}.')
    print(f'Available labels: {sorted(label_index.keys())}')

    if args.save_preview:
        os.makedirs(os.path.join(args.output_dir, 'preview'), exist_ok=True)

    arrays = []
    metadata = []
    for idx in tqdm.tqdm(range(args.num_samples), dynamic_ncols=True):
        wdm, meta = synthesize_mixed_wdm(samples, label_index, args, rng)
        meta['index'] = idx
        arrays.append(wdm)
        metadata.append(meta)

        if args.save_preview and idx < args.preview_limit:
            preview_path = os.path.join(args.output_dir, 'preview', f'{idx:06}.png')
            save_preview_image(wdm, preview_path)

    arrays = np.stack(arrays).astype(np.uint8)
    np.savez_compressed(os.path.join(args.output_dir, 'synthetic_wdm.npz'), arr_0=arrays)

    config = vars(args).copy()
    config['source_format'] = source_format
    config['num_source_samples'] = len(samples)
    result = {
        'config': config,
        'metadata': metadata,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'Saved synthetic WDM npz: {os.path.join(args.output_dir, "synthetic_wdm.npz")}')
    print(f'Saved metadata: {os.path.join(args.output_dir, "metadata.json")}')
    print(f'Output shape: {arrays.shape}, value range: {arrays.min()}..{arrays.max()}')


if __name__ == '__main__':
    main()
