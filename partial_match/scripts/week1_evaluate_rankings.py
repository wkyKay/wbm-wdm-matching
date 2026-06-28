# -*- coding: utf-8 -*-
"""
Week 1: Evaluate Retrieval Rankings
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import pandas as pd
import json

from partial_match.evaluation.metrics import evaluate_rankings


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval rankings for Week 1")
    parser.add_argument("--rankings", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/eval_smoke_rankings.csv",
                        help="Path to rankings CSV")
    parser.add_argument("--metadata", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/wm38k_metadata.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--k", type=int, nargs='+', default=[1, 5, 10], 
                        help="K values to evaluate")
    parser.add_argument("--out", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/eval_smoke_metrics.json",
                        help="Output JSON path")
    
    args = parser.parse_args()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading rankings and metadata...")
    rankings_df = pd.read_csv(args.rankings)
    metadata_df = pd.read_csv(args.metadata)
    
    print(f"Number of queries: {rankings_df['query_id'].nunique()}")
    
    # Evaluate
    print("Evaluating...")
    metrics = evaluate_rankings(
        rankings_df,
        metadata_df,
        k_values=args.k
    )
    
    # Save
    print("Saving metrics...")
    with open(args.out, 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    
    print("\nMicro Average:")
    for key, value in sorted(metrics['micro_average'].items()):
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTotal queries: {metrics['total_queries']}")
    print(f"Skipped exact queries: {metrics['skipped_exact_queries']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
