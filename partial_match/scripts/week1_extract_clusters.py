# -*- coding: utf-8 -*-
"""
Week 1: Extract Cluster Tokens
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from partial_match.core.cluster_proposal import (
    generate_cluster_tokens, 
    save_cluster_tokens,
    compute_cluster_statistics
)
from partial_match.utils.visualization import plot_sample_maps


def visualize_clusters(valid_maps: np.ndarray, 
                     original_indices: np.ndarray, 
                     out_dir: Path, 
                     num_samples: int = 10):
    """Visualize sample cluster overlays."""
    import random
    
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    sample_indices = random.sample(range(len(valid_maps)), min(num_samples, len(valid_maps)))
    
    for idx in sample_indices:
        raw_map = valid_maps[idx]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Raw map
        axes[0, 0].imshow(raw_map, cmap='viridis')
        axes[0, 0].set_title('Raw Map')
        axes[0, 0].axis('off')
        
        # Defect mask
        defect_mask = raw_map == 2
        axes[0, 1].imshow(defect_mask, cmap='gray')
        axes[0, 1].set_title('Defect Mask')
        axes[0, 1].axis('off')
        
        # Cluster visualization for raw proposal
        from partial_match.core.cluster_proposal import cluster_proposal
        valid_mask = (raw_map == 1) | (raw_map == 2)
        clusters_raw = cluster_proposal(defect_mask, valid_mask, 'raw')
        
        raw_vis = np.zeros_like(raw_map, dtype=np.float32)
        raw_vis[raw_map == 1] = 0.3
        raw_vis[defect_mask] = 0.7
        for i, cluster in enumerate(clusters_raw):
            color = 0.5 + (i % 10) * 0.05
            for (x, y) in cluster['pixels']:
                raw_vis[x, y] = color
        
        im = axes[1, 0].imshow(raw_vis, cmap='viridis')
        axes[1, 0].set_title(f'Clusters (raw): {len(clusters_raw)}')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        # Cluster visualization for closing proposal
        clusters_closed = cluster_proposal(defect_mask, valid_mask, 'closing')
        
        closed_vis = np.zeros_like(raw_map, dtype=np.float32)
        closed_vis[raw_map == 1] = 0.3
        closed_vis[defect_mask] = 0.7
        for i, cluster in enumerate(clusters_closed):
            color = 0.5 + (i % 10) * 0.05
            for (x, y) in cluster['pixels']:
                closed_vis[x, y] = color
        
        im = axes[1, 1].imshow(closed_vis, cmap='viridis')
        axes[1, 1].set_title(f'Clusters (closing): {len(clusters_closed)}')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'clusters_sample_{original_indices[idx]}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extract cluster tokens for Week 1")
    parser.add_argument("--maps", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/wm38k_maps.npz",
                        help="Path to wm38k_maps.npz")
    parser.add_argument("--metadata", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/wm38k_metadata.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--out-dir", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1",
                        help="Output directory")
    parser.add_argument("--proposal-types", type=str, nargs='+',
                        default=['topk'],
                        help="Cluster proposal types")
    parser.add_argument("--min-area", type=int, default=5, help="Minimum area for filtered clusters")
    parser.add_argument("--proposal-top-k", type=int, default=5,
                        help="Number of regions to keep for topk proposal")
    parser.add_argument("--topk-base-method", type=str, default='geometry_merge',
                        help="Candidate generator used by topk proposal")
    parser.add_argument("--dilation-radius", type=int, default=1,
                        help="Dilation radius for dilated grouping proposals")
    parser.add_argument("--use-closing-for-grouping", action="store_true",
                        help="Use closing instead of dilation for dilated grouping")
    parser.add_argument("--suspicious-area", type=int, default=40,
                        help="Minimum area before adhesion suspicious checks")
    parser.add_argument("--min-suspicious-cues", type=int, default=1,
                        help="Number of shape cues required before adhesion split")
    parser.add_argument("--max-split-count", type=int, default=12,
                        help="Reject adhesion split results with too many fragments")
    parser.add_argument("--min-split-coverage", type=float, default=0.5,
                        help="Reject adhesion split results that keep too little original area")
    parser.add_argument("--disable-ring-guard", action="store_true",
                        help="Allow adhesion split on ring-like dilated groups")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load preprocessed maps
    print("Loading preprocessed maps...")
    maps_data = np.load(args.maps, allow_pickle=True)
    valid_maps = maps_data['valid_maps']
    original_indices = maps_data['original_indices']
    
    # Generate cluster tokens
    print("Generating cluster tokens...")
    tokens = generate_cluster_tokens(
        valid_maps,
        original_indices,
        args.proposal_types,
        args.min_area,
        top_k=args.proposal_top_k,
        topk_base_method=args.topk_base_method,
        dilation_radius=args.dilation_radius,
        use_closing=args.use_closing_for_grouping,
        suspicious_area=args.suspicious_area,
        min_suspicious_cues=args.min_suspicious_cues,
        max_split_count=args.max_split_count,
        min_split_coverage=args.min_split_coverage,
        skip_ring_like=not args.disable_ring_guard,
    )
    
    # Save tokens
    print("Saving cluster tokens...")
    save_cluster_tokens(
        tokens,
        json_path=out_dir / 'wm38k_cluster_tokens.jsonl',
        npz_path=out_dir / 'wm38k_cluster_tokens.npz'
    )
    
    # Compute and save statistics
    print("Computing cluster statistics...")
    cluster_stats = compute_cluster_statistics(tokens)
    
    with open(out_dir / 'cluster_statistics.json', 'w') as f:
        json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
    
    print("Cluster statistics:")
    for pt, stats in cluster_stats.items():
        print(f"\n{pt}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Visualize clusters
    print("Visualizing cluster samples...")
    visualize_clusters(valid_maps, original_indices, out_dir)
    
    print("Done!")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
