# -*- coding: utf-8 -*-
"""
粘连分离方法对比 — 骨架切割 vs 分水岭 vs 现有方法

生成三类对比图：
  1. Synthetic: 纯人工合成图案（线-线、线-圆、T 形粘连）—— 最直观
  2. Real WM38K: 真实数据中的粘连样本
  3. (可选) Detail: 单个样本放大
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
_partial_match = os.path.dirname(_here)
_wbm_wdm = os.path.dirname(_partial_match)
_WORKDIR = os.path.dirname(_wbm_wdm)
if _wbm_wdm not in sys.path:
    sys.path.insert(0, _wbm_wdm)

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from partial_match.core.clustering import cluster
from partial_match.core.adhesion_split import (
    skeleton_split, watershed_split,
    tv_junction_split, tv_direction_cluster, tv_hybrid_cluster,
)

DATA_PATH = os.path.join(_WORKDIR, "data/wm38k/Wafer_Map_Datasets.npz")
OUT_DIR   = os.path.join(_WORKDIR, "artifacts/week1/figures")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ── 合成测试图案 ──────────────────────────────────────────

def make_cross_lines(size=52):
    """X 形交叉线：两条线在中心粘连"""
    mask = np.zeros((size, size), dtype=bool)
    cx, cy = size // 2, size // 2
    # 对角线 1: 左上→右下
    for d in range(-12, 13):
        mask[cx + d, cy + d] = True
    # 对角线 2: 右上→左下
    for d in range(-12, 13):
        mask[cx + d, cy - d] = True
    return mask

def make_line_and_blob(size=52):
    """一条水平线 + 中心一个圆形 blob，线穿过圆"""
    mask = np.zeros((size, size), dtype=bool)
    cx, cy = size // 2, size // 2
    # 圆
    for i in range(size):
        for j in range(size):
            if (i - cx) ** 2 + (j - cy) ** 2 <= 6 ** 2:
                mask[i, j] = True
    # 水平线
    mask[cx - 1:cx + 2, 5:size - 5] = True
    return mask

def make_two_lines_head_tail(size=52):
    """两条线头尾靠近但中间断开（半粘连）"""
    mask = np.zeros((size, size), dtype=bool)
    cx = size // 2
    # 左边斜线
    for d in range(0, 10):
        mask[cx - 5 + d, cx - 10 + d] = True
    # 右边斜线
    for d in range(0, 10):
        mask[cx - 4 - d, cx + 8 + d] = True
    # 粘连的桥接点
    mask[cx - 1, cx + 1] = True
    mask[cx - 1, cx + 2] = True
    return mask

def make_t_junction(size=52):
    """T 形：一条水平线 + 一条垂直线顶部相连"""
    mask = np.zeros((size, size), dtype=bool)
    cx, cy = size // 2, size // 2
    # 水平线
    mask[cx - 1:cx + 2, cy - 15:cy + 15] = True
    # 垂直线（从 T 接点向下）
    mask[cx:size - 8, cy - 1:cy + 2] = True
    return mask


SYNTHETIC_CASES = [
    ("Cross\nLines",          make_cross_lines()),
    ("Line\n+ Blob",          make_line_and_blob()),
    ("Head-Tail\nLines",      make_two_lines_head_tail()),
    ("T-Junction",            make_t_junction()),
]

# ── 真实 WM38K 样本（找有粘连特征的）─────────────────────

def load_wm38k():
    return np.load(DATA_PATH)['arr_0']

# ── 可视化工具 ────────────────────────────────────────────

def build_overlay(clusters, H, W):
    """每个 cluster 不同灰度"""
    overlay = np.zeros((H, W), dtype=float)
    for ci, cl in enumerate(clusters):
        coords = cl.get('pixel_coords', cl.get('pixels', []))
        color = 0.25 + (ci % 16) * 0.045
        for p in coords:
            if isinstance(p, dict):
                overlay[p['row'], p['col']] = color
            else:
                overlay[p[0], p[1]] = color
    return overlay

def build_rgb_overlay(clusters, H, W):
    """每个 cluster 不同 RGB 颜色"""
    overlay = np.ones((H, W, 3))  # white bg
    colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#ffff33', '#a65628', '#f781bf',
        '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
        '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',
    ]
    from matplotlib.colors import to_rgb
    rgb_colors = [to_rgb(c) for c in colors]
    for ci, cl in enumerate(clusters):
        c = rgb_colors[ci % len(rgb_colors)]
        coords = cl.get('pixel_coords', cl.get('pixels', []))
        for p in coords:
            if isinstance(p, dict):
                overlay[p['row'], p['col']] = c
            else:
                overlay[p[0], p[1]] = c
    return overlay


# ── 主函数 ────────────────────────────────────────────────

def main():
    maps = load_wm38k()

    # ─── 图 1: 合成图案 ───────────────────────────────
    fig1, axes1 = plt.subplots(len(SYNTHETIC_CASES), 9, figsize=(30, 14))

    col_titles = ['Input Mask', 'Raw\n(8-Conn)', 'DBSCAN', 'Spectral',
                  'Skeleton\nSplit', 'Watershed\nSplit',
                  'TV\nJunction', 'TV\nDirection', 'TV\nHybrid']
    for ci, t in enumerate(col_titles):
        axes1[0, ci].set_title(t, fontsize=10, fontweight='bold')

    for row, (name, syn_mask) in enumerate(SYNTHETIC_CASES):
        H, W = syn_mask.shape
        vm = np.ones((H, W), dtype=bool)  # 全有效

        # Input mask
        axes1[row, 0].imshow(syn_mask, cmap='gray_r')
        axes1[row, 0].set_ylabel(name, fontsize=10, fontweight='bold')

        # Raw (8-connected)
        clusters = _safe_run(lambda: cluster(syn_mask, method='raw'))
        axes1[row, 1].imshow(build_rgb_overlay(clusters, H, W))
        axes1[row, 1].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # DBSCAN
        clusters = _safe_run(lambda: cluster(syn_mask, vm, method='dbscan', auto_eps=True))
        axes1[row, 2].imshow(build_rgb_overlay(clusters, H, W))
        axes1[row, 2].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # Spectral
        clusters = _safe_run(lambda: cluster(syn_mask, vm, method='spectral', sigma=3.0))
        axes1[row, 3].imshow(build_rgb_overlay(clusters, H, W))
        axes1[row, 3].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # Skeleton Split
        clusters = _safe_run(lambda: skeleton_split(syn_mask, vm))
        axes1[row, 4].imshow(build_rgb_overlay(clusters, H, W))
        axes1[row, 4].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # Watershed Split
        clusters = _safe_run(lambda: watershed_split(syn_mask, vm))
        axes1[row, 5].imshow(build_rgb_overlay(clusters, H, W))
        axes1[row, 5].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # TV Junction Split
        clusters = _safe_run(lambda: tv_junction_split(syn_mask, vm, sigma=5.0))
        axes1[row, 6].imshow(build_rgb_overlay(clusters, H, W))
        axes1[row, 6].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # TV Direction Cluster
        clusters = _safe_run(lambda: tv_direction_cluster(syn_mask, vm, sigma=5.0))
        axes1[row, 7].imshow(build_rgb_overlay(clusters, H, W))
        axes1[row, 7].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # TV Hybrid Cluster
        clusters = _safe_run(lambda: tv_hybrid_cluster(syn_mask, vm, sigma=5.0))
        axes1[row, 8].imshow(build_rgb_overlay(clusters, H, W))
        axes1[row, 8].set_xlabel(f'{len(clusters)} cl', fontsize=8)

    for ax_row in axes1:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Adhesion Split — Synthetic Test Cases', fontsize=14, y=1.01)
    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "adhesion_split_synthetic.png")
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {path1}")

    # ─── 图 2: 真实 WM38K 样本 ─────────────────────────
    real_samples = [500, 800, 1200, 2000, 3500]

    fig2, axes2 = plt.subplots(len(real_samples), 10, figsize=(34, 17))

    col_titles2 = ['Raw Map', 'Raw\n(8-Conn)', 'DBSCAN', 'Spectral',
                    'Adj-iWMM', 'Skeleton\nSplit', 'Watershed\nSplit',
                    'TV\nJunction', 'TV\nDirection', 'TV\nHybrid']
    for ci, t in enumerate(col_titles2):
        axes2[0, ci].set_title(t, fontsize=10, fontweight='bold')

    for row, sid in enumerate(real_samples):
        raw_map = maps[sid]
        dm = raw_map == 2
        vm = (raw_map == 1) | (raw_map == 2)
        H, W = raw_map.shape

        axes2[row, 0].imshow(raw_map, cmap='viridis')
        axes2[row, 0].set_ylabel(f'Sample {sid}\n({dm.sum()} pts)', fontsize=9)

        # Raw
        clusters = _safe_run(lambda: cluster(dm, vm, method='raw'))
        axes2[row, 1].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 1].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # DBSCAN
        clusters = _safe_run(lambda: cluster(dm, vm, method='dbscan', auto_eps=True))
        axes2[row, 2].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 2].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # Spectral
        clusters = _safe_run(lambda: cluster(dm, vm, method='spectral', sigma=3.0))
        axes2[row, 3].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 3].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # Adjacency-iWMM
        clusters = _safe_run(lambda: cluster(dm, vm, method='adjacency_iwmm'))
        axes2[row, 4].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 4].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # Skeleton Split
        clusters = _safe_run(lambda: skeleton_split(dm, vm))
        axes2[row, 5].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 5].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # Watershed Split
        clusters = _safe_run(lambda: watershed_split(dm, vm))
        axes2[row, 6].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 6].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # TV Junction Split
        clusters = _safe_run(lambda: tv_junction_split(dm, vm, sigma=5.0))
        axes2[row, 7].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 7].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # TV Direction Cluster
        clusters = _safe_run(lambda: tv_direction_cluster(dm, vm, sigma=5.0))
        axes2[row, 8].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 8].set_xlabel(f'{len(clusters)} cl', fontsize=8)

        # TV Hybrid Cluster
        clusters = _safe_run(lambda: tv_hybrid_cluster(dm, vm, sigma=5.0))
        axes2[row, 9].imshow(build_rgb_overlay(clusters, H, W))
        axes2[row, 9].set_xlabel(f'{len(clusters)} cl', fontsize=8)

    for ax_row in axes2:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Adhesion Split — Real WM38K Samples', fontsize=14, y=1.01)
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "adhesion_split_real_samples.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {path2}")

    # ─── 图 3: 粘连样本特写（sample 800 和 1200）─────────
    detail_samples = [800, 1200]
    fig3, axes3 = plt.subplots(len(detail_samples), 10, figsize=(34, 9))

    detail_col_titles = col_titles2  # reuse the same 10-column titles

    for ci, t in enumerate(col_titles2):
        axes3[0, ci].set_title(t, fontsize=11, fontweight='bold')

    for row, sid in enumerate(detail_samples):
        raw_map = maps[sid]
        dm = raw_map == 2
        vm = (raw_map == 1) | (raw_map == 2)
        H, W = raw_map.shape

        axes3[row, 0].imshow(raw_map, cmap='viridis')
        axes3[row, 0].set_ylabel(f'Sample {sid}\n({dm.sum()} pts)', fontsize=10)

        for col, (method_label, func) in enumerate([
            ('raw',          lambda m=dm, v=vm: cluster(m, v, method='raw')),
            ('dbscan',       lambda m=dm, v=vm: cluster(m, v, method='dbscan')),
            ('spectral',     lambda m=dm, v=vm: cluster(m, v, method='spectral')),
            ('adj_iwmm',     lambda m=dm, v=vm: cluster(m, v, method='adjacency_iwmm')),
            ('skeleton',     lambda m=dm, v=vm: skeleton_split(m, v)),
            ('watershed',    lambda m=dm, v=vm: watershed_split(m, v)),
            ('tv_junction',  lambda m=dm, v=vm: tv_junction_split(m, v, sigma=5.0)),
            ('tv_direction', lambda m=dm, v=vm: tv_direction_cluster(m, v, sigma=5.0)),
            ('tv_hybrid',    lambda m=dm, v=vm: tv_hybrid_cluster(m, v, sigma=5.0)),
        ], start=1):
            clusters = _safe_run(func)
            axes3[row, col].imshow(build_rgb_overlay(clusters, H, W))
            axes3[row, col].set_xlabel(f'{len(clusters)} cl', fontsize=9)

    for ax_row in axes3:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Adhesion Split — Detail View (Samples 800 & 1200)', fontsize=13, y=1.02)
    plt.tight_layout()
    path3 = os.path.join(OUT_DIR, "adhesion_split_detail.png")
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {path3}")

    # ─── 文本汇总 ──────────────────────────────────────
    print("\n" + "=" * 120)
    print("SYNTHETIC CASES")
    print("-" * 120)
    header = (f"{'Pattern':<16} {'Raw':>5} {'DBSCAN':>7} {'Spectral':>9} "
              f"{'Skeleton':>9} {'Watershed':>10} "
              f"{'TV-Junc':>8} {'TV-Dir':>7} {'TV-Hyb':>8}")
    print(header)
    for name, syn_mask in SYNTHETIC_CASES:
        vm = np.ones_like(syn_mask, dtype=bool)
        raw_n = len(cluster(syn_mask, method='raw'))
        db_n  = len(cluster(syn_mask, vm, method='dbscan', auto_eps=True))
        sp_n  = len(cluster(syn_mask, vm, method='spectral', sigma=3.0))
        sk_n  = len(skeleton_split(syn_mask, vm))
        ws_n  = len(watershed_split(syn_mask, vm))
        tvj_n = len(tv_junction_split(syn_mask, vm, sigma=5.0))
        tvd_n = len(tv_direction_cluster(syn_mask, vm, sigma=5.0))
        tvh_n = len(tv_hybrid_cluster(syn_mask, vm, sigma=5.0))
        print(f"{name:<16} {raw_n:>5} {db_n:>7} {sp_n:>9} "
              f"{sk_n:>9} {ws_n:>10} {tvj_n:>8} {tvd_n:>7} {tvh_n:>8}")

    print("\nREAL WM38K SAMPLES")
    print("-" * 120)
    header = (f"{'Sample':>7} {'Raw':>5} {'DBSCAN':>7} {'Spectral':>9} "
              f"{'Adj-iWMM':>9} {'Skeleton':>9} {'Watershed':>10} "
              f"{'TV-Junc':>8} {'TV-Dir':>7} {'TV-Hyb':>8}")
    print(header)
    for sid in real_samples:
        dm = maps[sid] == 2
        vm = (maps[sid] == 1) | (maps[sid] == 2)
        raw_n = len(cluster(dm, vm, method='raw'))
        db_n  = len(cluster(dm, vm, method='dbscan', auto_eps=True))
        sp_n  = len(cluster(dm, vm, method='spectral', sigma=3.0))
        ai_n  = len(cluster(dm, vm, method='adjacency_iwmm'))
        sk_n  = len(skeleton_split(dm, vm))
        ws_n  = len(watershed_split(dm, vm))
        tvj_n = len(tv_junction_split(dm, vm, sigma=5.0))
        tvd_n = len(tv_direction_cluster(dm, vm, sigma=5.0))
        tvh_n = len(tv_hybrid_cluster(dm, vm, sigma=5.0))
        print(f"{sid:>7} {raw_n:>5} {db_n:>7} {sp_n:>9} "
              f"{ai_n:>9} {sk_n:>9} {ws_n:>10} {tvj_n:>8} {tvd_n:>7} {tvh_n:>8}")

    print("\nAll done!")


def _safe_run(fn):
    try:
        return fn()
    except Exception as e:
        print(f"  [WARN] {e}")
        return []


if __name__ == "__main__":
    main()
