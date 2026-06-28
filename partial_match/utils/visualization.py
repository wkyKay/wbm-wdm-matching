# -*- coding: utf-8 -*-
"""
可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

from partial_match.data.data_io import CLASS_NAMES


def plot_sample_maps(
    raw_map: np.ndarray,
    preprocessed: Dict[str, np.ndarray],
    save_path: str = None
):
    """
    绘制单张样本的各种预处理结果。

    Args:
        raw_map: 原始晶圆图
        preprocessed: 预处理结果字典
        save_path: 保存路径，可选
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # 原始图和状态图
    ax = axes[0]
    im = ax.imshow(raw_map, cmap='viridis')
    ax.set_title('Raw Map')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(preprocessed['status_map'], cmap='viridis')
    ax.set_title('Status Map')
    plt.colorbar(im, ax=ax)

    # Masks
    ax = axes[2]
    ax.imshow(preprocessed['valid_mask'], cmap='gray')
    ax.set_title('Valid Mask')

    ax = axes[3]
    ax.imshow(preprocessed['defect_mask'], cmap='gray')
    ax.set_title('Defect Mask')

    # Maps
    ax = axes[4]
    ax.imshow(preprocessed['binary_map'], cmap='gray')
    ax.set_title('Binary Map')

    ax = axes[5]
    im = ax.imshow(preprocessed['density_map'], cmap='viridis')
    ax.set_title('Density Map')
    plt.colorbar(im, ax=ax)

    ax = axes[6]
    im = ax.imshow(preprocessed['soft_map'], cmap='viridis')
    ax.set_title('Soft Map')
    plt.colorbar(im, ax=ax)

    ax = axes[7]
    im = ax.imshow(preprocessed['three_value_map'], cmap='viridis')
    ax.set_title('Three Value Map')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_cluster_proposals(
    raw_map: np.ndarray,
    proposal_type: str = 'raw',
    min_area: int = 3,
    save_path: str = None
):
    """
    可视化 cluster proposal 结果。
    
    Args:
        raw_map: 原始晶圆图
        proposal_type: 'raw', 'closing', or 'filtered'
        min_area: 用于 filtered proposal 的最小面积
        save_path: 保存路径，可选
    """
    from partial_match.core.cluster_proposal import cluster_proposal
    
    # 生成 defect mask 和 valid mask
    defect_mask = (raw_map == 2)
    valid_mask = (raw_map == 1) | (raw_map == 2)
    
    # 获取 clusters
    clusters = cluster_proposal(
        defect_mask, 
        valid_mask, 
        proposal_type, 
        min_area
    )
    
    # 绘制
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图
    axes[0].imshow(raw_map, cmap='viridis')
    axes[0].set_title('Raw Map')
    axes[0].axis('off')
    
    # Defect mask
    axes[1].imshow(defect_mask, cmap='gray')
    axes[1].set_title('Defect Mask')
    axes[1].axis('off')
    
    # Cluster overlay
    cluster_overlay = np.zeros_like(raw_map, dtype=float)
    cluster_overlay[valid_mask] = 0.2
    
    # 用不同颜色标记不同 clusters
    color_idx = 1
    for cluster in clusters:
        color = 0.3 + (color_idx % 10) * 0.07
        
        # 处理两种不同格式的像素坐标
        if 'pixel_coords' in cluster:
            coords = cluster['pixel_coords']
            if coords and isinstance(coords[0], dict):
                # 格式 1: [{'row': r, 'col': c}, ...]
                for p in coords:
                    cluster_overlay[p['row'], p['col']] = color
            else:
                # 格式 2: [(r, c), ...]
                for (row, col) in coords:
                    cluster_overlay[row, col] = color
        elif 'pixels' in cluster:
            # 格式 3: 直接的像素列表
            for (row, col) in cluster['pixels']:
                cluster_overlay[row, col] = color
                
        color_idx += 1
    
    im = axes[2].imshow(cluster_overlay, cmap='viridis')
    axes[2].set_title(f'Cluster Proposal: {proposal_type}\n{len(clusters)} clusters')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_cluster_analysis(
    valid_maps: np.ndarray,
    original_indices: np.ndarray,
    out_dir: str,
    num_samples: int = 10
):
    """
    批量可视化多个样本的 cluster proposal 结果。
    
    Args:
        valid_maps: 验证集晶圆图
        original_indices: 原始索引
        out_dir: 输出目录
        num_samples: 要可视化的样本数量
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    # 随机选择样本
    import random
    sample_indices = random.sample(range(len(valid_maps)), min(num_samples, len(valid_maps)))
    
    for idx in sample_indices:
        raw_map = valid_maps[idx]
        sample_id = original_indices[idx]
        
        # 可视化所有三种 proposal 类型
        for proposal_type in ['raw', 'closing', 'filtered']:
            save_path = os.path.join(
                out_dir, 
                f'cluster_{proposal_type}_sample_{sample_id}.png'
            )
            visualize_cluster_proposals(
                raw_map, 
                proposal_type, 
                min_area=3, 
                save_path=save_path
            )


def plot_class_samples(
    maps: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray = None,
    n_samples_per_class: int = 3,
    save_dir: str = None
):
    """
    绘制每个类别的样本。

    Args:
        maps: 晶圆图数组
        labels: 标签数组
        sample_ids: 样本 ID 数组，可选
        n_samples_per_class: 每个类别抽取的样本数
        save_dir: 保存目录，可选
    """
    if sample_ids is None:
        sample_ids = np.arange(len(maps))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        # 找属于该类别的样本
        class_mask = labels[:, class_idx] == 1
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        # 随机抽取样本
        selected = np.random.choice(
            class_indices,
            size=min(n_samples_per_class, len(class_indices)),
            replace=False
        )

        # 绘制
        n_cols = min(len(selected), 5)
        n_rows = (len(selected) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, idx in enumerate(selected):
            ax = axes[i]
            ax.imshow(maps[idx], cmap='viridis')
            sid = sample_ids[idx]
            ax.set_title(f'{class_name}\nSample {sid}')
            ax.axis('off')

        # 隐藏多余的子图
        for i in range(len(selected), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'class_{class_name}_samples.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.suptitle(f'Class: {class_name}', y=1.02)
            plt.show()


def plot_label_cardinality_distribution(
    df,
    save_path: str = None
):
    """
    绘制标签基数分布。

    Args:
        df: metadata DataFrame
        save_path: 保存路径，可选
    """
    plt.figure(figsize=(10, 6))
    df['label_cardinality'].value_counts().sort_index().plot(kind='bar')
    plt.title('Label Cardinality Distribution')
    plt.xlabel('Number of Labels')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_defect_area_distribution(
    df,
    save_path: str = None
):
    """
    绘制缺陷面积分布。

    Args:
        df: metadata DataFrame
        save_path: 保存路径，可选
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(df['defect_area'], bins=50)
    axes[0].set_title('Defect Area Distribution')
    axes[0].set_xlabel('Defect Area')
    axes[0].set_ylabel('Number of Samples')
    axes[0].grid(axis='y', alpha=0.3)

    # 箱线图（按类别）
    class_areas = []
    class_names_plot = []
    for class_name in CLASS_NAMES:
        class_mask = df[f'label_{class_name}']
        if class_mask.sum() > 0:
            class_areas.append(df[class_mask]['defect_area'].values)
            class_names_plot.append(class_name)

    if class_areas:
        axes[1].boxplot(class_areas)
        axes[1].set_xticklabels(class_names_plot, rotation=45)
        axes[1].set_title('Defect Area by Class')
        axes[1].set_ylabel('Defect Area')
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_split_distribution(
    df,
    save_path: str = None
):
    """
    绘制 split 分布。

    Args:
        df: metadata DataFrame
        save_path: 保存路径，可选
    """
    if 'split' not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Split 样本数
    split_counts = df['split'].value_counts()
    split_counts.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Number of Samples by Split')
    axes[0].set_ylabel('Number of Samples')
    axes[0].grid(axis='y', alpha=0.3)

    # Split 中类别分布
    split_names = ['train', 'validation', 'test']
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    for i, split_name in enumerate(split_names):
        split_df = df[df['split'] == split_name]
        class_counts = [split_df[f'label_{cn}'].sum() for cn in CLASS_NAMES]
        axes[1].bar(x + i*width, class_counts, width, label=split_name)

    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Class Distribution by Split')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
