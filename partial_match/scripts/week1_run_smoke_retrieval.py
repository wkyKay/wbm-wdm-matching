# -*- coding: utf-8 -*-
"""
Week 1: Run Smoke Baseline Retrieval
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd

from partial_match.evaluation.smoke_baseline import generate_smoke_rankings


def main():
    parser = argparse.ArgumentParser(description="Run smoke baseline retrieval for Week 1")
    parser.add_argument("--maps", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/wm38k_maps.npz",
                        help="Path to wm38k_maps.npz")
    parser.add_argument("--splits", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/wm38k_splits.json",
                        help="Path to wm38k_splits.json")
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], 
                        default='validation',
                        help="Which split to use")
    parser.add_argument("--method", type=str, choices=['iou', 'coverage_leakage'], 
                        default='coverage_leakage',
                        help="Smoke baseline method")
    parser.add_argument("--top-k", type=int, default=100, 
                        help="Top-K candidates to return")
    parser.add_argument("--beta", type=float, default=0.5, 
                        help="Beta parameter for coverage-leakage")
    parser.add_argument("--out", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/eval_smoke_rankings.csv",
                        help="Output CSV path")
    
    args = parser.parse_args()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading maps and splits...")
    maps_data = np.load(args.maps, allow_pickle=True)
    binary_maps = maps_data['binary_maps']
    
    import json
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    
    sample_ids = splits[args.split]
    split_binary_maps = binary_maps[sample_ids]
    
    print(f"Running {args.method} baseline on {args.split} split...")
    print(f"Number of samples: {len(sample_ids)}")
    
    # Generate rankings
    rankings_df = generate_smoke_rankings(
        split_binary_maps,
        sample_ids,
        method=args.method,
        top_k=args.top_k,
        beta=args.beta
    )
    
    # Save
    print(f"Saving rankings to {args.out}")
    rankings_df.to_csv(args.out, index=False)
    
    print("Done!")


if __name__ == "__main__":
    main()
