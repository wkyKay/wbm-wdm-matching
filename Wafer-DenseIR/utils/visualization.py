# -*- coding: utf-8 -*-

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_retrieval_explanation(path, query_raw, candidate_raw, query_heatmap, candidate_heatmap,
                               query_id, candidate_id, score, matches=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    _show_map(axes[0, 0], query_raw, f'Query {query_id}')
    _show_map(axes[0, 1], candidate_raw, f'Candidate {candidate_id}\nscore={score:.4f}')
    _show_overlay(axes[1, 0], query_raw, query_heatmap, 'Query correspondence heatmap')
    _show_overlay(axes[1, 1], candidate_raw, candidate_heatmap, 'Candidate response heatmap')
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _show_map(ax, raw, title):
    cmap = plt.get_cmap('viridis', 3)
    ax.imshow(raw, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    ax.set_title(title, fontsize=10)


def _show_overlay(ax, raw, heatmap, title):
    heatmap = _resize_nearest(heatmap, raw.shape)
    ax.imshow(raw, cmap=plt.get_cmap('gray', 3), vmin=0, vmax=2, interpolation='nearest')
    ax.imshow(heatmap, cmap='magma', alpha=0.58, interpolation='nearest')
    ax.set_title(title, fontsize=10)


def _resize_nearest(x, shape):
    h, w = shape
    if x.shape == shape:
        return x
    yy = np.linspace(0, x.shape[0] - 1, h).round().astype(int)
    xx = np.linspace(0, x.shape[1] - 1, w).round().astype(int)
    return x[yy][:, xx]
