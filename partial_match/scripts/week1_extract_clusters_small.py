# -*- coding: utf-8 -*-
"""
Week 1: Extract Cluster Tokens - Small Version
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd
import json

from partial_match.core.cluster_proposal import (
    generate_cluster_tokens, 
    save_cluster_tokens,
    compute_cluster_statistics
)


def main():
    parser = argparse.ArgumentParser(description="Extract cluster tokens for Week 1 (small version)")
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
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to process")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load preprocessed maps
    print("Loading preprocessed maps...")
    maps_data = np.load(args.maps, allow_pickle=True)
    valid_maps = maps_data['valid_maps']
    original_indices = maps_data['original_indices']
    
    # Use a subset of samples for speed
    num_samples = min(args.num_samples, len(valid_maps))
    valid_maps_small = valid_maps[:num_samples]
    original_indices_small = original_indices[:num_samples]
    
    print(f"Processing {num_samples} samples...")
    
    # Generate cluster tokens
    print("Generating cluster tokens...")
    tokens = generate_cluster_tokens(
        valid_maps_small,
        original_indices_small,
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
        json_path=out_dir / 'wm38k_cluster_tokens_small.jsonl',
        npz_path=out_dir / 'wm38k_cluster_tokens_small.npz'
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
    
    print("Done!")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
