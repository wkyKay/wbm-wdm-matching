# -*- coding: utf-8 -*-
"""
清洗 512x512 生产 WDM 数据，用于 MixedWa 第二阶段 domain adaptation。

用法（从 MixedWa/ 目录运行）：
  python processors/process_wdm_512_cleaning.py \
    --wdm_npz ../../data/production/wdm_512.npz \
    --output_dir ../../data/production/wdm_512_cleaned \
    --supervised_ckpt ./checkpoints/train/best_model.pt \
    --save_preview

输出：
  cleaned_wdm.npz             # high + sampled medium
  high_confidence_wdm.npz     # high-confidence
  medium_confidence_wdm.npz   # medium-confidence
  rejected_wdm.npz            # rejected / low-confidence
  cleaning_metadata.json      # 每张图的指标、分组、过滤原因
  preview/                    # 可选预览
"""

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    # 允许从 MixedWa/ 目录直接运行 processors 内脚本，并导入项目模块。
    sys.path.insert(0, ROOT)

from datasets.datasets import DEFECT_CLASSES, decouple_mask
from datasets.transforms import WaferTransform
from tasks.domain_adapt import generate_pseudo_wbm


def parse_args():
    """解析清洗脚本参数。阈值均设置为保守默认值，建议先跑一版后根据 metadata 分布微调。"""
    p = argparse.ArgumentParser('Clean 512x512 production WDM data')
    p.add_argument('--wdm_npz', type=str, default=None,
                   help='输入 WDM npz，默认读取 arr_0 或第一个 key')
    p.add_argument('--wdm_dir', type=str, default=None,
                   help='输入 WDM 图像目录，支持 png/jpg/jpeg/bmp/tif/tiff')
    p.add_argument('--wdm_format', type=str, default='auto',
                   choices=['auto', 'binary', 'wm811k_png', 'wbm_values'])
    p.add_argument('--wafer_mask_mode', type=str, default='auto',
                   choices=['auto', 'circular', 'nonzero', 'full'])
    p.add_argument('--output_dir', type=str, required=True)

    p.add_argument('--expected_size', type=int, default=512,
                   help='期望输入尺寸；非该尺寸时会 resize 到该尺寸后清洗')
    p.add_argument('--model_img_size', type=int, default=96,
                   help='stage1 classifier 输入尺寸')
    p.add_argument('--pseudo_out_size', type=int, default=96)

    p.add_argument('--min_density', type=float, default=0.00005)
    p.add_argument('--max_density', type=float, default=0.50)
    p.add_argument('--min_wafer_area_ratio', type=float, default=0.50)
    p.add_argument('--small_component_max_area', type=int, default=4)
    p.add_argument('--max_small_component_ratio', type=float, default=0.85)
    p.add_argument('--min_max_component_area', type=int, default=3)
    p.add_argument('--min_max_component_ratio_for_structure', type=float, default=0.08)
    p.add_argument('--max_fragmented_components', type=int, default=1500)

    p.add_argument('--pseudo_min_fail_die', type=int, default=2)
    p.add_argument('--pseudo_max_fail_die', type=int, default=100)
    p.add_argument('--pseudo_min_density', type=float, default=0.02)
    p.add_argument('--pseudo_max_density', type=float, default=0.85)

    p.add_argument('--supervised_ckpt', type=str, default=None,
                   help='可选 stage1 supervised checkpoint')
    p.add_argument('--backbone', type=str, default='resnet18')
    p.add_argument('--in_channels', type=int, default=2)
    p.add_argument('--stage1_batch_size', type=int, default=128)
    p.add_argument('--stage1_high_prob', type=float, default=0.70)
    p.add_argument('--stage1_medium_prob', type=float, default=0.45)
    p.add_argument('--stage1_max_entropy', type=float, default=0.80)
    p.add_argument('--device', type=str, default=None)

    p.add_argument('--medium_keep_ratio', type=float, default=0.35,
                   help='cleaned_wdm.npz 中保留 medium-confidence 的比例')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_preview', action='store_true')
    p.add_argument('--preview_limit', type=int, default=300)
    return p.parse_args()


def make_circular_mask(shape: tuple) -> np.ndarray:
    """生成圆形 wafer 有效区域，适用于只有缺陷点、没有显式 wafer mask 的 binary WDM。"""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = min(h, w) * 0.48
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def normalize_wdm_array(arr: np.ndarray, data_format: str, wafer_mask_mode: str) -> np.ndarray:
    """
    将不同来源的 WDM 统一为 MixedWa 使用的 WBM-like 值域：
      0 = 背景，1 = wafer 有效区域正常，2 = 缺陷。

    binary WDM 通常只有缺陷点，没有 wafer 有效区域，因此默认补圆形 wafer mask；
    wbm_values / wm811k_png 通常自带非零有效区域，优先使用 nonzero mask。
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0]
        elif arr.shape[0] in (1, 3, 4):
            arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D WDM array, got shape {arr.shape}')

    if data_format == 'auto':
        # 根据值域自动推断来源格式。若无法明确判断，按 binary 缺陷点图处理。
        unique = np.unique(arr)
        unique_set = set(unique.tolist())
        if unique_set.issubset({0, 1}):
            data_format = 'binary'
        elif unique_set.issubset({0, 1, 2}):
            data_format = 'wbm_values'
        elif unique.max() > 2 and np.any(unique >= 200):
            data_format = 'wm811k_png'
        else:
            data_format = 'binary'

    if data_format == 'binary':
        defect = arr > 0
        default_mask_mode = 'circular'
    elif data_format == 'wm811k_png':
        defect = arr >= 200
        default_mask_mode = 'nonzero'
    elif data_format == 'wbm_values':
        defect = arr == 2
        default_mask_mode = 'nonzero'
    else:
        raise ValueError(f'Unknown wdm_format: {data_format}')

    mask_mode = default_mask_mode if wafer_mask_mode == 'auto' else wafer_mask_mode
    if mask_mode == 'circular':
        valid = make_circular_mask(arr.shape)
    elif mask_mode == 'nonzero':
        valid = arr > 0
        if valid.sum() <= defect.sum():
            # 只有缺陷点的图 nonzero mask 会退化为 defect mask，此时回退到圆形 wafer mask。
            valid = make_circular_mask(arr.shape)
    elif mask_mode == 'full':
        valid = np.ones(arr.shape, dtype=bool)
    else:
        raise ValueError(f'Unknown wafer_mask_mode: {wafer_mask_mode}')

    defect = np.logical_and(defect, valid)
    out = valid.astype(np.uint8)
    out[defect] = 2
    return out


def resize_to_expected(arr: np.ndarray, expected_size: int) -> np.ndarray:
    """把输入统一到清洗分辨率，使用最近邻避免把离散值 {0,1,2} 插成连续灰度。"""
    if arr.shape == (expected_size, expected_size):
        return arr.astype(np.uint8)
    return cv2.resize(arr.astype(np.uint8), (expected_size, expected_size), interpolation=cv2.INTER_NEAREST)


def load_wdm_arrays(args):
    """从 npz 或图像目录读取 WDM，并完成值域归一化与尺寸统一。"""
    source_ids = []
    if args.wdm_npz:
        data = np.load(args.wdm_npz)
        key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
        arrays = np.asarray(data[key])
        source_ids = [str(i) for i in range(len(arrays))]
    elif args.wdm_dir:
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(args.wdm_dir, '**', pattern), recursive=True))
        paths = sorted(paths)
        if len(paths) == 0:
            raise ValueError(f'No image files found under {args.wdm_dir}')
        arrays = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                arrays.append(img)
                source_ids.append(path)
        if len(arrays) == 0:
            raise ValueError(f'Failed to read image files under {args.wdm_dir}')
    else:
        raise ValueError('必须指定 --wdm_npz 或 --wdm_dir')

    normalized = []
    for arr in arrays:
        wdm = normalize_wdm_array(arr, args.wdm_format, args.wafer_mask_mode)
        normalized.append(resize_to_expected(wdm, args.expected_size))
    return np.stack(normalized).astype(np.uint8), source_ids


def safe_div(a: float, b: float) -> float:
    """安全除法，避免空 wafer / 空 defect 样本导致除零。"""
    return float(a / b) if b > 0 else 0.0


def entropy_from_counts(counts: np.ndarray) -> float:
    """连通域面积熵，用于衡量缺陷是否由大量小碎片均匀组成。"""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log(p)).sum())


def compute_connected_components(defect: np.ndarray, args):
    """
    计算连通域与碎片化指标。

    先做轻量 closing 是为了把相邻缺陷点合并成局部区域，减少单像素噪声对
    connected components 的干扰；这里的 closing 仅用于统计，不会修改输出 WDM。
    """
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(defect.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    defect_area = int(defect.sum())
    if len(areas) == 0:
        return {
            'num_components': 0,
            'max_component_area': 0,
            'max_component_ratio': 0.0,
            'small_component_ratio': 0.0,
            'component_area_entropy': 0.0,
        }
    small_area = int(areas[areas <= args.small_component_max_area].sum())
    return {
        'num_components': int(len(areas)),
        'max_component_area': int(areas.max()),
        'max_component_ratio': safe_div(float(areas.max()), float(defect_area)),
        'small_component_ratio': safe_div(float(small_area), float(defect_area)),
        'component_area_entropy': entropy_from_counts(areas),
    }


def compute_spatial_features(defect: np.ndarray, valid: np.ndarray):
    """
    提取 wafer-level 空间结构特征。

    这些指标不做精确分类，只用于判断样本是否存在明显 pattern 证据，例如中心聚集、
    边缘聚集、环状分布、线状 scratch 或局部 cluster。
    """
    h, w = defect.shape
    ys, xs = np.where(defect)
    if len(xs) == 0:
        return {
            'centroid': [None, None],
            'centroid_distance_norm': None,
            'radial_mean': 0.0,
            'radial_std': 0.0,
            'center_defect_ratio': 0.0,
            'edge_defect_ratio': 0.0,
            'ring_score': 0.0,
            'linearity_score': 0.0,
            'bbox_aspect_ratio': 0.0,
        }

    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = min(h, w) * 0.48
    dy = ys.astype(np.float64) - cy
    dx = xs.astype(np.float64) - cx
    radial = np.sqrt(dx * dx + dy * dy) / max(radius, 1.0)
    # 径向直方图最大 bin 占比越高，越可能存在 edge-ring / donut / center 等径向结构。
    hist, _ = np.histogram(radial, bins=10, range=(0.0, 1.2))

    centered = np.stack([dy, dx], axis=1)
    if len(xs) >= 3:
        # PCA 主方向特征。最大/最小特征值差异越大，缺陷越接近线状 scratch。
        cov = np.cov(centered, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        linearity = 1.0 - safe_div(float(eigvals[0]), float(eigvals[-1] + 1e-8))
    else:
        linearity = 0.0

    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_aspect = float(max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w)))
    centroid_y = float(ys.mean() / max(1, h - 1))
    centroid_x = float(xs.mean() / max(1, w - 1))
    centroid_dist = float(np.sqrt((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2) / max(radius, 1.0))

    return {
        'centroid': [centroid_y, centroid_x],
        'centroid_distance_norm': centroid_dist,
        'radial_mean': float(radial.mean()),
        'radial_std': float(radial.std()),
        'center_defect_ratio': float((radial < 0.30).mean()),
        'edge_defect_ratio': float((radial > 0.80).mean()),
        'ring_score': safe_div(float(hist.max()), float(hist.sum())),
        'linearity_score': float(linearity),
        'bbox_aspect_ratio': bbox_aspect,
    }


def pseudo_metrics(wdm: np.ndarray, args):
    """
    生成 pseudo-WBM 并检查退化质量。

    domain adaptation 的正样本对依赖 WDM -> pseudo-WBM。如果 pseudo-WBM 变成全 0、
    全 1 或 fail die 数异常，则该正样本对不可靠，应过滤或降权。
    """
    pseudo = generate_pseudo_wbm(wdm, out_size=args.pseudo_out_size)
    fail = pseudo == 2
    valid = pseudo > 0
    fail_die_count_11 = int(cv2.resize(fail.astype(np.uint8), (11, 11), interpolation=cv2.INTER_NEAREST).sum())
    pseudo_density = safe_div(float(fail.sum()), float(valid.sum()))
    return pseudo, {
        'pseudo_fail_die_count': fail_die_count_11,
        'pseudo_density': pseudo_density,
        'pseudo_all_zero': bool(fail.sum() == 0),
        'pseudo_all_one': bool(valid.sum() > 0 and fail.sum() >= valid.sum()),
    }


def structure_score(metrics: dict) -> float:
    """
    将多种弱结构证据合成一个粗略分数。

    该分数不是类别概率，只用于 high / medium / rejected 分组。分数来源包括：
    大连通域、径向结构、线状结构、中心/边缘聚集，以及碎片化程度。
    """
    score = 0.0
    if metrics['max_component_ratio'] >= 0.08:
        score += 1.0
    if metrics['ring_score'] >= 0.35:
        score += 1.0
    if metrics['linearity_score'] >= 0.80 and metrics['bbox_aspect_ratio'] >= 3.0:
        score += 1.0
    if metrics['center_defect_ratio'] >= 0.35:
        score += 0.75
    if metrics['edge_defect_ratio'] >= 0.35:
        score += 0.75
    if metrics['small_component_ratio'] <= 0.50 and metrics['num_components'] <= 1000:
        score += 0.5
    return float(score)


def base_reject_reasons(metrics: dict, args):
    """根据基础质量指标和 pseudo-WBM 指标生成可追溯的过滤原因。"""
    reasons = []
    if metrics['wafer_area_ratio'] < args.min_wafer_area_ratio:
        reasons.append('wafer_area_too_small')
    if metrics['defect_density'] < args.min_density:
        reasons.append('density_too_low')
    if metrics['defect_density'] > args.max_density:
        reasons.append('density_too_high')
    if metrics['max_component_area'] < args.min_max_component_area:
        reasons.append('max_component_too_small')
    if (metrics['num_components'] > args.max_fragmented_components and
            metrics['max_component_ratio'] < args.min_max_component_ratio_for_structure):
        reasons.append('too_fragmented')
    if metrics['small_component_ratio'] > args.max_small_component_ratio:
        reasons.append('small_components_dominant')
    if metrics['pseudo_fail_die_count'] < args.pseudo_min_fail_die:
        reasons.append('pseudo_fail_die_too_few')
    if metrics['pseudo_fail_die_count'] > args.pseudo_max_fail_die:
        reasons.append('pseudo_fail_die_too_many')
    if metrics['pseudo_density'] < args.pseudo_min_density:
        reasons.append('pseudo_density_too_low')
    if metrics['pseudo_density'] > args.pseudo_max_density:
        reasons.append('pseudo_density_too_high')
    if metrics['pseudo_all_zero']:
        reasons.append('pseudo_all_zero')
    if metrics['pseudo_all_one']:
        reasons.append('pseudo_all_one')
    return reasons


def analyze_sample(wdm: np.ndarray, args):
    """汇总单张 WDM 的所有无监督清洗指标。"""
    valid = wdm > 0
    defect = wdm == 2
    wafer_area = int(valid.sum())
    defect_area = int(defect.sum())
    metrics = {
        'wafer_area': wafer_area,
        'wafer_area_ratio': safe_div(float(wafer_area), float(wdm.size)),
        'defect_pixels': defect_area,
        'defect_density': safe_div(float(defect_area), float(wafer_area)),
    }
    metrics.update(compute_connected_components(defect, args))
    metrics.update(compute_spatial_features(defect, valid))
    _, pmetrics = pseudo_metrics(wdm, args)
    metrics.update(pmetrics)
    metrics['structure_score'] = structure_score(metrics)
    return metrics


def run_stage1_classifier(arrays: np.ndarray, args):
    """
    可选：用 stage1 supervised encoder/classifier 给真实 WDM 打分。

    stage1 模型是在 WM38K/WBM 上训练的，面对真实 WDM 有 domain gap。因此这里的分数只作为
    辅助筛选信号，不作为唯一过滤标准；低置信度样本仍可能因几何结构清楚而进入 medium 组。
    """
    if not args.supervised_ckpt:
        return [None] * len(arrays)

    import torch
    from models.factory import build_backbone
    from models.head import LinearClassifier

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = build_backbone(args.backbone, in_channels=args.in_channels, img_size=args.model_img_size)
    classifier = LinearClassifier(in_dim=backbone.out_dim, num_classes=len(DEFECT_CLASSES))
    backbone.load_weights_from_checkpoint(args.supervised_ckpt, strict=False)
    classifier.load_weights_from_checkpoint(args.supervised_ckpt, key='classifier', strict=False)
    backbone.to(device).eval()
    classifier.to(device).eval()

    transform = WaferTransform(size=(args.model_img_size, args.model_img_size), mode='test')
    scores = []
    with torch.no_grad():
        for start in tqdm.tqdm(range(0, len(arrays), args.stage1_batch_size), desc='Stage1 scoring', dynamic_ncols=True):
            batch = []
            for arr in arrays[start:start + args.stage1_batch_size]:
                x = transform(arr[:, :, None])
                if args.in_channels == 2:
                    x = decouple_mask(x)
                batch.append(x)
            x = torch.stack(batch).to(device)
            probs = torch.sigmoid(classifier(backbone(x))).cpu().numpy()
            for p in probs:
                eps = 1e-8
                # 多标签二分类熵的平均值，越低表示预测越确定。
                entropy = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps)).mean() / math.log(2)
                top_idx = np.argsort(-p)[:3]
                scores.append({
                    'stage1_probs': [float(v) for v in p.tolist()],
                    'stage1_top_labels': [DEFECT_CLASSES[int(i)] for i in top_idx],
                    'stage1_top_probs': [float(p[int(i)]) for i in top_idx],
                    'stage1_max_prob': float(p.max()),
                    'stage1_entropy': float(entropy),
                })
    return scores


def assign_group(metrics: dict, stage1: dict, reasons: list, args):
    """
    根据几何结构、pseudo-WBM 可靠性和 stage1 置信度分组。

    设计原则：
      - pseudo-WBM 退化异常是 fatal，直接 rejected；
      - 有 stage1 时，high 组要求 stage1 高置信 + 几何强结构；
      - medium 组允许 stage1 一般，但必须有几何结构且 pseudo-WBM 正常；
      - 无 stage1 checkpoint 时，仅使用无监督几何与 pseudo-WBM 指标分组。
    """
    has_stage1 = stage1 is not None
    stage1_high = has_stage1 and stage1['stage1_max_prob'] >= args.stage1_high_prob and stage1['stage1_entropy'] <= args.stage1_max_entropy
    stage1_medium = has_stage1 and stage1['stage1_max_prob'] >= args.stage1_medium_prob
    geometry_good = metrics['structure_score'] >= 1.0
    geometry_strong = metrics['structure_score'] >= 1.75
    pseudo_ok = not any(r.startswith('pseudo_') for r in reasons)
    fatal = [r for r in reasons if r in {
        'wafer_area_too_small', 'density_too_low', 'density_too_high',
        'pseudo_all_zero', 'pseudo_all_one', 'pseudo_fail_die_too_few',
        'pseudo_fail_die_too_many', 'pseudo_density_too_low', 'pseudo_density_too_high'
    }]

    if fatal:
        return 'rejected'
    if pseudo_ok and geometry_strong and (stage1_high or not has_stage1):
        return 'high'
    if pseudo_ok and geometry_good and (stage1_high or stage1_medium or not has_stage1):
        return 'medium'
    if pseudo_ok and geometry_good:
        return 'medium'
    return 'rejected'


def save_npz(path: str, arrays: list):
    """保存 npz。空分组也写出空数组，便于下游脚本固定读取文件名。"""
    if len(arrays) == 0:
        np.savez_compressed(path, arr_0=np.empty((0,), dtype=np.uint8))
    else:
        np.savez_compressed(path, arr_0=np.stack(arrays).astype(np.uint8))


def save_preview(arr: np.ndarray, path: str):
    """保存预览图，将 {0,1,2} 映射到 0/127/255 灰度。"""
    img = ((arr.astype(np.float32) / 2.0) * 255).astype(np.uint8)
    cv2.imwrite(path, img)


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    arrays, source_ids = load_wdm_arrays(args)
    print(f'Loaded {len(arrays)} WDM samples, shape={arrays.shape[1:]}')

    stage1_scores = run_stage1_classifier(arrays, args)

    buckets = {'high': [], 'medium': [], 'rejected': []}
    metadata = []
    for idx, wdm in enumerate(tqdm.tqdm(arrays, desc='Cleaning WDM', dynamic_ncols=True)):
        # 先做无监督几何/pseudo-WBM 分析，再合并可选的 stage1 分类器分数。
        metrics = analyze_sample(wdm, args)
        reasons = base_reject_reasons(metrics, args)
        stage1 = stage1_scores[idx]
        if stage1 is not None:
            metrics.update(stage1)
        else:
            metrics.update({
                'stage1_probs': None,
                'stage1_top_labels': None,
                'stage1_top_probs': None,
                'stage1_max_prob': None,
                'stage1_entropy': None,
            })
        group = assign_group(metrics, stage1, reasons, args)
        buckets[group].append((idx, wdm))
        metadata.append({
            'index': int(idx),
            'source_id': source_ids[idx],
            'confidence_group': group,
            'reject_reason': reasons,
            **metrics,
        })

    medium = buckets['medium']
    if len(medium) > 0 and args.medium_keep_ratio < 1.0:
        # cleaned_wdm 默认只抽样一部分 medium，避免弱/不确定样本在 domain adaptation 中占比过高。
        keep_n = int(round(len(medium) * max(0.0, args.medium_keep_ratio)))
        keep_n = max(0, min(len(medium), keep_n))
        keep_indices = set(rng.choice(len(medium), size=keep_n, replace=False).tolist()) if keep_n > 0 else set()
        sampled_medium = [item for i, item in enumerate(medium) if i in keep_indices]
    else:
        sampled_medium = medium

    high_arrays = [arr for _, arr in buckets['high']]
    medium_arrays = [arr for _, arr in buckets['medium']]
    rejected_arrays = [arr for _, arr in buckets['rejected']]
    cleaned_arrays = high_arrays + [arr for _, arr in sampled_medium]

    save_npz(os.path.join(args.output_dir, 'high_confidence_wdm.npz'), high_arrays)
    save_npz(os.path.join(args.output_dir, 'medium_confidence_wdm.npz'), medium_arrays)
    save_npz(os.path.join(args.output_dir, 'rejected_wdm.npz'), rejected_arrays)
    save_npz(os.path.join(args.output_dir, 'cleaned_wdm.npz'), cleaned_arrays)

    result = {
        'config': vars(args),
        'summary': {
            'total': len(arrays),
            'high': len(high_arrays),
            'medium': len(medium_arrays),
            'medium_sampled_into_cleaned': len(sampled_medium),
            'rejected': len(rejected_arrays),
            'cleaned': len(cleaned_arrays),
        },
        'metadata': metadata,
    }
    with open(os.path.join(args.output_dir, 'cleaning_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if args.save_preview:
        for group, items in buckets.items():
            group_dir = os.path.join(args.output_dir, 'preview', group)
            os.makedirs(group_dir, exist_ok=True)
            for idx, arr in items[:args.preview_limit]:
                save_preview(arr, os.path.join(group_dir, f'{idx:06}.png'))

    print('Cleaning finished.')
    print(json.dumps(result['summary'], indent=2, ensure_ascii=False))
    print(f'Output directory: {args.output_dir}')


if __name__ == '__main__':
    main()
