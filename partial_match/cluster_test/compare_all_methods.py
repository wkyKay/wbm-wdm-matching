# -*- coding: utf-8 -*-
"""
对比多种 cluster proposal 方法 — 支持 AC 清洗前后对比。

生成三张图：
1. all_proposal_methods_comparison.png — 多方法 × 6 样本（无清洗）
2. all_proposal_methods_cleaned_comparison.png — 多方法 × 6 样本（AC 清洗后）
3. before_vs_after_cleaning.png — 每种方法 × 2 样本的清洗前/后对比
"""

import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
_partial_match = os.path.dirname(_here)
_wbm_wdm = os.path.dirname(_partial_match)
if _wbm_wdm not in sys.path:
    sys.path.insert(0, _wbm_wdm)

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from partial_match.core.clustering import cluster
from partial_match.data.preprocessing import ac_clean_mask


DATA_PATH = ("/Users/kayw/Documents/trae_projects/match-test/"
             "data/wm38k/Wafer_Map_Datasets.npz")
OUT_DIR = ("/Users/kayw/Documents/trae_projects/match-test/"
           "artifacts/week1/figures")


def load_data():
    return np.load(DATA_PATH)['arr_0']


def build_overlay(clusters, H, W, valid_mask):
    """构建 cluster 可视化 overlay"""
    overlay = np.zeros((H, W), dtype=float)
    overlay[valid_mask] = 0.08
    for ci, cl in enumerate(clusters):
        coords = cl.get('pixel_coords', cl.get('pixels', []))
        color = 0.25 + (ci % 16) * 0.045
        for p in coords:
            if isinstance(p, dict):
                overlay[p['row'], p['col']] = color
            else:
                overlay[p[0], p[1]] = color
    return overlay


def main():
    maps = load_data()
    samples = [100, 500, 800, 1200, 2000, 3500]
    methods = [
        ('raw',             '8-Connected'),
        ('filtered',        'Filtered\n(area>=3)'),
        ('adhesion',        'Adhesion\nSplit'),
        ('dilated_group',   'Dilated\nGroup'),
        ('dilated_adhesion','Dilated\n+ Adhesion'),
        ('topk',            'TopK\n(k=5)'),
        ('topk_dilated',    'TopK Dilated\n(k=5)'),
        ('closing',         'Closing\n(Morph)'),
        ('simi_paper',      'SIMI Paper'),
        ('dbscan',          'DBSCAN'),
        ('adjacency_iwmm',  'Adj-Cluster\n+ iWMM'),
        ('spectral',        'Spectral\nClustering'),
        ('tensor_voting',   'Tensor\nVoting'),
    ]

    n_methods = len(methods)
    H_maps = {sid: maps[sid].shape[0] for sid in samples}
    W_maps = {sid: maps[sid].shape[1] for sid in samples}

    # ============================================================
    # Figure 1: 无清洗 — 多方法 × 6 样本
    # ============================================================
    print("Generating Figure 1: Without cleaning...")
    fig1, axes1 = plt.subplots(len(samples), n_methods + 1, figsize=(54, 22))
    axes1[0, 0].set_title('Raw Map', fontsize=10, fontweight='bold')
    for m, (_, name) in enumerate(methods):
        axes1[0, m + 1].set_title(name, fontsize=10, fontweight='bold')

    results_no_clean = {}
    for row, sid in enumerate(samples):
        raw_map = maps[sid]
        dm = raw_map == 2
        vm = (raw_map == 1) | (raw_map == 2)
        H, W = raw_map.shape

        axes1[row, 0].imshow(raw_map, cmap='viridis')
        axes1[row, 0].set_ylabel(f'Sample {sid}\n({dm.sum()} pts)', fontsize=9)

        for col, (method, _) in enumerate(methods):
            clusters = cluster(dm, vm, method=method)
            results_no_clean[(sid, method)] = clusters
            overlay = build_overlay(clusters, H, W, vm)
            ax = axes1[row, col + 1]
            ax.imshow(overlay, cmap='tab20' if len(clusters) > 2 else 'gray')
            ax.set_xlabel(f'{len(clusters)} clusters', fontsize=8)

    for ax_row in axes1:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f'Cluster Proposal Methods — Without AC Cleaning\n{n_methods} Methods x 6 Samples',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    sp1 = os.path.join(OUT_DIR, "all_proposal_methods_comparison.png")
    plt.savefig(sp1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sp1}")

    # ============================================================
    # Figure 2: AC 清洗后 — 多方法 × 6 样本
    # ============================================================
    print("Generating Figure 2: With AC cleaning...")
    fig2, axes2 = plt.subplots(len(samples), n_methods + 1, figsize=(54, 22))

    # 第一列：清洗后的 mask
    axes2[0, 0].set_title('AC Cleaned\nMask', fontsize=10, fontweight='bold')
    for m, (_, name) in enumerate(methods):
        axes2[0, m + 1].set_title(f'AC → {name}', fontsize=10, fontweight='bold')

    results_clean = {}
    for row, sid in enumerate(samples):
        raw_map = maps[sid]
        dm = raw_map == 2
        vm = (raw_map == 1) | (raw_map == 2)
        H, W = raw_map.shape

        # 展示清洗后的 mask
        cleaned_mask, _ = ac_clean_mask(dm & vm)
        cleaned_viz = np.zeros((H, W), dtype=float)
        cleaned_viz[vm & ~cleaned_mask] = 0.15  # 被移除的噪声（浅灰）
        cleaned_viz[cleaned_mask] = 0.70         # 保留的系统缺陷（深色）
        axes2[row, 0].imshow(cleaned_viz, cmap='gray')
        axes2[row, 0].set_ylabel(
            f'Sample {sid}\n{int(dm.sum())}→{int(cleaned_mask.sum())} pts',
            fontsize=9)

        for col, (method, _) in enumerate(methods):
            clusters = cluster(dm, vm, method=method, use_clean=True)
            results_clean[(sid, method)] = clusters
            overlay = build_overlay(clusters, H, W, vm)
            ax = axes2[row, col + 1]
            ax.imshow(overlay, cmap='tab20' if len(clusters) > 2 else 'gray')
            ax.set_xlabel(f'{len(clusters)} clusters', fontsize=8)

    for ax_row in axes2:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f'Cluster Proposal Methods — With AC Cleaning\n{n_methods} Methods x 6 Samples',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    sp2 = os.path.join(OUT_DIR, "all_proposal_methods_cleaned_comparison.png")
    plt.savefig(sp2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sp2}")

    # ============================================================
    # Figure 3: Before vs After — 每种方法 × 代表性样本
    # ============================================================
    print("Generating Figure 3: Before vs After cleaning...")
    # 选 2 个最有代表性的样本
    showcase_samples = [500, 1200]
    n_rows = len(showcase_samples) * n_methods
    fig3, axes3 = plt.subplots(n_rows, 4, figsize=(20, n_rows * 2.5))

    for row_idx, sid in enumerate(showcase_samples):
        raw_map = maps[sid]
        dm = raw_map == 2
        vm = (raw_map == 1) | (raw_map == 2)
        H, W = raw_map.shape

        # AC 清洗详情
        cleaned_mask, removed_mask = ac_clean_mask(dm & vm)
        n_before = int(dm.sum())
        n_after = int(cleaned_mask.sum())

        for col, (method, name) in enumerate(methods):
            r = row_idx * n_methods + col

            # Before
            clusters_before = results_no_clean[(sid, method)]
            ov_before = build_overlay(clusters_before, H, W, vm)
            axes3[r, 0].imshow(ov_before, cmap='tab20' if len(clusters_before) > 2 else 'gray')
            axes3[r, 0].set_title(
                f'{name}\n({len(clusters_before)} clusters)',
                fontsize=9)
            if col == 0:
                axes3[r, 0].set_ylabel(f'Sample {sid}\nBefore', fontsize=8)

            # After (cleaned)
            clusters_after = results_clean[(sid, method)]
            ov_after = build_overlay(clusters_after, H, W, vm)
            axes3[r, 1].imshow(ov_after, cmap='tab20' if len(clusters_after) > 2 else 'gray')
            axes3[r, 1].set_title(
                f'{name}\n({len(clusters_after)} clusters)',
                fontsize=9)

            # AC Cleaned mask detail
            detail = np.zeros((H, W), dtype=float)
            detail[vm] = 0.10
            detail[cleaned_mask] = 0.70
            detail[removed_mask] = 0.40
            axes3[r, 2].imshow(detail, cmap='gray')
            axes3[r, 2].set_title(
                f'Cleaned Mask\n{n_before}→{n_after} pts',
                fontsize=9)
            if col == 0:
                axes3[r, 2].set_ylabel(f'Sample {sid}\nCleaned', fontsize=8)

        # 最后一列放原始 map
        for col in range(n_methods):
            r = row_idx * n_methods + col
            axes3[r, 3].imshow(raw_map, cmap='viridis')
            axes3[r, 3].set_title(f'Raw Map\n{n_before} pts', fontsize=9)

    for ax in axes3.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # 列标题
    col_titles = ['Before Cleaning', 'After AC Cleaning', 'AC Mask Detail', 'Raw Map']
    for c, title in enumerate(col_titles):
        axes3[0, c].set_title(f'{title}\n{axes3[0, c].get_title()}',
                              fontsize=11, fontweight='bold')

    plt.suptitle('AC Cleaning Effect — Before vs After\n'
                 f'Samples {showcase_samples}, {n_methods} Methods',
                 fontsize=14, y=1.005)
    plt.tight_layout()
    sp3 = os.path.join(OUT_DIR, "before_vs_after_cleaning.png")
    plt.savefig(sp3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sp3}")

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "=" * 120)
    header = f"{'Sample':>8}"
    for _, name in methods:
        header += f" {'Before':>8} {'After':>8}"
    print(header)
    print("-" * 120)

    for sid in samples:
        n_total = int((maps[sid] == 2).sum())
        row_str = f"{sid:>8}"
        for method, _ in methods:
            n_before = len(results_no_clean[(sid, method)])
            n_after = len(results_clean[(sid, method)])
            row_str += f" {n_before:>8} {n_after:>8}"
        print(row_str)

    # 清洗统计汇总
    print("\n" + "=" * 70)
    print("AC Cleaning Statistics")
    print("-" * 70)
    print(f"{'Sample':>8} {'Before':>8} {'After':>8} {'Removed':>8} {'Ratio':>8}")
    print("-" * 70)
    for sid in samples:
        dm = maps[sid] == 2
        vm = (maps[sid] == 1) | (maps[sid] == 2)
        mask = dm & vm
        n_before = int(mask.sum())
        cleaned, _ = ac_clean_mask(mask)
        n_after = int(cleaned.sum())
        n_removed = n_before - n_after
        ratio = n_removed / max(n_before, 1)
        print(f"{sid:>8} {n_before:>8} {n_after:>8} {n_removed:>8} {ratio:>7.1%}")

    print("\nDone! All figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
