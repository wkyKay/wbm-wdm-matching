# -*- coding: utf-8 -*-
"""
Metadata 生成模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .data_io import get_label_signature, CLASS_NAMES
from .preprocessing import compute_map_statistics


def generate_metadata(
    maps: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    split_indices: Dict[str, List[int]] = None
) -> pd.DataFrame:
    """
    生成 metadata DataFrame。

    Args:
        maps: (N, H, W) 过滤后的晶圆图
        labels: (N, 8) 过滤后的标签
        original_indices: 对应原始数据集的下标
        split_indices: split 结果，可选

    Returns:
        metadata DataFrame
    """
    N = len(maps)

    # 初始化数据
    data = {
        'sample_id': np.arange(N),
        'orig_index': original_indices,
        'label_vec': [tuple(lbl) for lbl in labels],
        'label_set': [get_label_signature(lbl) for lbl in labels],
        'label_cardinality': labels.sum(axis=1),
    }

    # 添加 split 信息
    if split_indices is not None:
        split_assignments = [''] * N
        for split_name, indices in split_indices.items():
            for idx in indices:
                split_assignments[idx] = split_name
        data['split'] = split_assignments

    # 添加统计信息
    stats_keys = [
        'valid_area', 'defect_area', 'defect_ratio',
        'defect_bbox_row_min', 'defect_bbox_col_min',
        'defect_bbox_row_max', 'defect_bbox_col_max',
        'defect_centroid_row', 'defect_centroid_col',
        'defect_centroid_row_norm', 'defect_centroid_col_norm',
    ]
    for key in stats_keys:
        data[key] = []

    # 计算每张图的统计
    for i in range(N):
        stats = compute_map_statistics(
            defect_mask=(maps[i] == 2),
            valid_mask=((maps[i] == 1) | (maps[i] == 2))
        )
        for key in stats_keys:
            data[key].append(stats[key])

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 添加类别名称列
    for i, class_name in enumerate(CLASS_NAMES):
        df[f'label_{class_name}'] = labels[:, i].astype(bool)

    return df


def analyze_metadata(df: pd.DataFrame) -> Dict:
    """
    分析 metadata，生成统计报告。

    Args:
        df: metadata DataFrame

    Returns:
        统计报告字典
    """
    report = {}

    # 总体统计
    report['total_samples'] = len(df)

    # 标签基数统计
    report['cardinality_distribution'] = df['label_cardinality'].value_counts().sort_index().to_dict()

    # 类别分布
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_counts[class_name] = int(df[f'label_{class_name}'].sum())
    report['class_distribution'] = class_counts

    # Split 统计（如果有）
    if 'split' in df.columns:
        split_stats = {}
        for split_name in ['train', 'validation', 'test']:
            split_df = df[df['split'] == split_name]
            split_stats[split_name] = {
                'n_samples': len(split_df),
                'cardinality_distribution': split_df['label_cardinality'].value_counts().sort_index().to_dict(),
            }
        report['split_stats'] = split_stats

    # 缺陷面积统计
    report['defect_area_stats'] = {
        'mean': float(df['defect_area'].mean()),
        'std': float(df['defect_area'].std()),
        'min': int(df['defect_area'].min()),
        'max': int(df['defect_area'].max()),
        'median': float(df['defect_area'].median()),
    }

    # 类别-wise 统计
    class_wise_stats = {}
    for class_name in CLASS_NAMES:
        class_df = df[df[f'label_{class_name}']]
        class_wise_stats[class_name] = {
            'n_samples': len(class_df),
            'defect_area_mean': float(class_df['defect_area'].mean()),
            'defect_area_std': float(class_df['defect_area'].std()),
        }
    report['class_wise_stats'] = class_wise_stats

    return report
