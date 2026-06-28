# -*- coding: utf-8 -*-
"""Visualize retrieval_compact proposal steps for manual debugging."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from partial_match.core.retrieval_compact import retrieval_compact_proposal
from partial_match.data.data_io import CLASS_NAMES, filter_valid_samples, load_wm38k


def main():
    args = parse_args()
    visualize_retrieval_compact_steps(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/wm38k/Wafer_Map_Datasets.npz')
    parser.add_argument('--out-dir', type=str, default='artifacts/proposal_based/retrieval_compact_steps')
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--sample-ids', type=int, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=1993)
    parser.add_argument('--min-area', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--edge-r-min', type=float, default=0.65)
    parser.add_argument('--ring-band-width', type=float, default=0.10)
    parser.add_argument('--min-ring-area', type=int, default=12)
    parser.add_argument('--min-ring-angular-coverage', type=float, default=0.16)
    parser.add_argument('--min-ring-area-ratio', type=float, default=0.12)
    parser.add_argument('--max-ring-radial-std', type=float, default=0.12)
    parser.add_argument('--max-defect-ratio-for-ring', type=float, default=0.45)
    parser.add_argument('--min-edge-defect-fraction-for-ring', type=float, default=0.45)
    parser.add_argument('--disable-ring-aware', action='store_true')
    return parser.parse_args()


def visualize_retrieval_compact_steps(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps, labels = load_wm38k(args.data_file)
    maps, labels, original_indices = filter_valid_samples(maps, labels)
    selected = _select_samples(labels, original_indices, args.samples, args.sample_ids, args.seed)

    saved_paths = []
    for valid_pos in selected:
        raw = maps[valid_pos]
        label = labels[valid_pos]
        orig_id = int(original_indices[valid_pos])
        defect_mask = raw == 2
        valid_mask = (raw == 1) | (raw == 2)
        clusters, steps = retrieval_compact_proposal(
            defect_mask,
            valid_mask,
            min_area=args.min_area,
            top_k=args.top_k,
            edge_r_min=args.edge_r_min,
            ring_band_width=args.ring_band_width,
            min_ring_area=args.min_ring_area,
            min_ring_angular_coverage=args.min_ring_angular_coverage,
            min_ring_area_ratio=args.min_ring_area_ratio,
            max_ring_radial_std=args.max_ring_radial_std,
            max_defect_ratio_for_ring=args.max_defect_ratio_for_ring,
            min_edge_defect_fraction_for_ring=args.min_edge_defect_fraction_for_ring,
            enable_ring_aware=not args.disable_ring_aware,
            return_steps=True,
        )
        label_names = _label_names(label)
        save_path = out_dir / f'sample_{orig_id}_{label_names.replace("|", "_")}.png'
        _plot_steps(raw, clusters, steps, orig_id, label_names, save_path)
        saved_paths.append(save_path)
        print(f'Saved {save_path}')
    return saved_paths


def _select_samples(labels, original_indices, samples, sample_ids, seed):
    if sample_ids:
        lookup = {int(orig): pos for pos, orig in enumerate(original_indices)}
        return [lookup[x] for x in sample_ids if x in lookup]
    rng = np.random.default_rng(seed)
    selected = []
    per_class = max(1, int(np.ceil(samples / len(CLASS_NAMES))))
    for class_idx in range(len(CLASS_NAMES)):
        positions = np.where(labels[:, class_idx] == 1)[0]
        if len(positions) == 0:
            continue
        take = min(per_class, len(positions))
        selected.extend(int(x) for x in rng.choice(positions, size=take, replace=False))
    selected = list(dict.fromkeys(selected))
    if len(selected) > samples:
        selected = selected[:samples]
    return selected


def _label_names(label_vec):
    names = [name for name, flag in zip(CLASS_NAMES, label_vec) if int(flag) == 1]
    return '|'.join(names) if names else 'unlabeled'


def _plot_steps(raw, final_clusters, steps, orig_id, label_names, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    panels = [
        ('Raw', raw),
        ('Defect mask', steps['original_mask']),
        ('Denoised mask', steps['denoised_mask']),
        (_ring_title(steps), steps['ring_mask']),
        ('Residual after ring removal', steps['residual_mask']),
        ('Residual components', _overlay_tokens(raw, steps['component_tokens'])),
        ('Final compact tokens', _overlay_tokens(raw, final_clusters)),
        ('Tiny removed', steps['tiny_removed_mask']),
    ]

    for ax, (title, image) in zip(axes, panels):
        if image.dtype == bool:
            ax.imshow(image, cmap='gray', interpolation='nearest')
        else:
            ax.imshow(image, cmap='viridis', interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if title == 'Residual components':
            _annotate_tokens(ax, steps['component_tokens'])
        if title == 'Final compact tokens':
            _annotate_tokens(ax, final_clusters)

    fig.suptitle(f'orig_index={orig_id} labels={label_names}', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _ring_title(steps):
    info = steps.get('ring_debug', {})
    return (
        f"Ring mask: {info.get('reason', 'unknown')}\n"
        f"area={info.get('candidate_area', 0)} cov={info.get('angular_coverage', 0.0):.2f} "
        f"def={info.get('defect_ratio', 0.0):.2f} edge={info.get('edge_fraction', 0.0):.2f}"
    )


def _overlay_tokens(raw, clusters):
    overlay = np.zeros(raw.shape, dtype=np.float32)
    valid_mask = (raw == 1) | (raw == 2)
    overlay[valid_mask] = 0.05
    for idx, cluster in enumerate(clusters):
        value = 0.18 + (idx % 18) * 0.045
        for coord in cluster.get('pixels', []):
            r, c = int(coord[0]), int(coord[1])
            overlay[r, c] = value
    return overlay


def _annotate_tokens(ax, clusters):
    abbreviations = {
        'edge_ring': 'R',
        'line': 'L',
        'blob': 'B',
        'central': 'C',
        'irregular': 'I',
    }
    for idx, cluster in enumerate(clusters):
        label = abbreviations.get(cluster.get('geometry_type', 'irregular'), '?')
        r = float(cluster.get('centroid_row', 0.0))
        c = float(cluster.get('centroid_col', 0.0))
        ax.text(
            c,
            r,
            f'{idx}:{label}',
            color='white',
            fontsize=7,
            ha='center',
            va='center',
            bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 1, 'edgecolor': 'none'},
        )


if __name__ == '__main__':
    main()
