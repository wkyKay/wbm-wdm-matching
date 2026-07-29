# -*- coding: utf-8 -*-
"""Visualize each query with top-k retrieved candidates in one panel."""

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
import pandas as pd

from partial_match.core.arc_ring_retrieval import ArcRingConfig, prepare_tokens
from partial_match.data.data_io import CLASS_NAMES, filter_valid_samples, load_wm38k


def main():
    args = parse_args()
    visualize_topk_retrieval(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/wm38k/Wafer_Map_Datasets.npz')
    parser.add_argument('--rankings', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='artifacts/proposal_based/top3_review')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--max-queries', type=int, default=50)
    parser.add_argument('--query-ids', type=int, nargs='*', default=None)
    parser.add_argument('--min-area', type=int, default=5)
    parser.add_argument('--top-k-proposals', type=int, default=5)
    return parser.parse_args()


def visualize_topk_retrieval(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps, labels = load_wm38k(args.data_file)
    maps, labels, original_indices = filter_valid_samples(maps, labels)
    id_to_pos = {int(orig_id): pos for pos, orig_id in enumerate(original_indices)}
    rankings = pd.read_csv(args.rankings)

    query_ids = _select_query_ids(rankings, args.query_ids, args.max_queries)
    token_cache = {}
    saved_paths = []
    for qid in query_ids:
        if qid not in id_to_pos:
            continue
        group = rankings[rankings['query_id'] == qid].sort_values('rank').head(args.top_k)
        candidate_ids = [int(x) for x in group['candidate_id'].tolist() if int(x) in id_to_pos]
        scores = [float(x) for x in group['similarity_score'].tolist()[:len(candidate_ids)]]
        if not candidate_ids:
            continue
        save_path = out_dir / f'query_{qid}_top{args.top_k}.png'
        _plot_query_topk(
            qid,
            candidate_ids,
            scores,
            maps,
            labels,
            id_to_pos,
            token_cache,
            args,
            save_path,
        )
        saved_paths.append(save_path)
        print(f'Saved {save_path}')
    return saved_paths


def _select_query_ids(rankings, query_ids, max_queries):
    if query_ids:
        return [int(x) for x in query_ids]
    ordered = []
    for qid in rankings['query_id'].tolist():
        qid = int(qid)
        if qid not in ordered:
            ordered.append(qid)
        if len(ordered) >= max_queries:
            break
    return ordered


def _plot_query_topk(qid, candidate_ids, scores, maps, labels, id_to_pos, token_cache, args, save_path):
    panel_ids = [qid] + candidate_ids
    n_cols = len(panel_ids)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    for col, map_id in enumerate(panel_ids):
        pos = id_to_pos[map_id]
        raw = maps[pos]
        clusters = _clusters_for_map(map_id, raw, token_cache, args)
        overlay = _overlay_tokens(raw, clusters)
        axes[col].imshow(overlay, cmap='viridis', interpolation='nearest')
        _annotate_tokens(axes[col], clusters)
        title = _title_for_panel(col, map_id, labels[pos], scores)
        axes[col].set_title(title, fontsize=10)
        axes[col].set_xticks([])
        axes[col].set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _clusters_for_map(map_id, raw, token_cache, args):
    if map_id in token_cache:
        return token_cache[map_id]
    defect_mask = raw == 2
    valid_mask = (raw == 1) | (raw == 2)
    clusters = prepare_tokens(
        defect_mask,
        valid_mask,
        ArcRingConfig(min_area=args.min_area, top_k=args.top_k_proposals),
    )
    token_cache[map_id] = clusters
    return clusters


def _title_for_panel(col, map_id, label_vec, scores):
    label_names = '|'.join(name for name, flag in zip(CLASS_NAMES, label_vec) if int(flag) == 1)
    if col == 0:
        return f'Query {map_id}\n{label_names}'
    score = scores[col - 1] if col - 1 < len(scores) else 0.0
    return f'Top {col}: {map_id}\nscore={score:.4f}\n{label_names}'


def _overlay_tokens(raw, clusters):
    overlay = np.zeros(raw.shape, dtype=np.float32)
    valid_mask = (raw == 1) | (raw == 2)
    overlay[valid_mask] = 0.05
    overlay[raw == 2] = 0.18
    for idx, cluster in enumerate(clusters):
        value = 0.30 + (idx % 12) * 0.055
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
