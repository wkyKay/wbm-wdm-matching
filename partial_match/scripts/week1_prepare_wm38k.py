# -*- coding: utf-8 -*-
"""
Week 1: Prepare MixedWM38K Data
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd
import json

from partial_match.data.data_io import load_wm38k, filter_valid_samples, get_label_info
from partial_match.data.preprocessing import preprocess_batch
from partial_match.data.split import split_by_signature, get_split_info
from partial_match.data.metadata import generate_metadata, analyze_metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare MixedWM38K data for Week 1")
    parser.add_argument("--npz", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/data/wm38k/Wafer_Map_Datasets.npz",
                        help="Path to Wafer_Map_Datasets.npz")
    parser.add_argument("--out-dir", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and filter data
    print("Loading data...")
    maps, labels = load_wm38k(args.npz)
    valid_maps, valid_labels, original_indices = filter_valid_samples(maps, labels)
    
    print(f"Total samples: {len(maps)}")
    print(f"Valid samples: {len(valid_maps)}")
    
    # Step 2: Split data
    print("Splitting data...")
    split_indices = split_by_signature(
        valid_labels,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        args.seed
    )
    
    split_info = get_split_info(split_indices, valid_labels)
    print("Split info:", split_info)
    
    # Save splits
    splits_data = {
        'seed': args.seed,
        'source': args.npz,
        'class_names': ['center', 'donut', 'edge-loc', 'edge-ring', 
                        'loc', 'random', 'scratch', 'near-full'],
        'train': [int(i) for i in split_indices['train']],
        'validation': [int(i) for i in split_indices['validation']],
        'test': [int(i) for i in split_indices['test']],
    }
    
    with open(out_dir / 'wm38k_splits.json', 'w') as f:
        json.dump(splits_data, f, ensure_ascii=False, indent=2)
    
    # Step 3: Preprocess maps
    print("Preprocessing maps...")
    preprocessed = preprocess_batch(valid_maps)
    
    # Save preprocessed maps
    np.savez_compressed(
        out_dir / 'wm38k_maps.npz',
        status_maps=preprocessed['status_maps'],
        binary_maps=preprocessed['binary_maps'],
        count_maps=preprocessed['count_maps'],
        density_maps=preprocessed['density_maps'],
        soft_maps=preprocessed['soft_maps'],
        three_value_maps=preprocessed['three_value_maps'],
        valid_maps=valid_maps,
        valid_labels=valid_labels,
        original_indices=original_indices
    )
    
    # Step 4: Generate metadata
    print("Generating metadata...")
    metadata_df = generate_metadata(
        valid_maps,
        valid_labels,
        original_indices,
        split_indices
    )
    
    metadata_df.to_csv(out_dir / 'wm38k_metadata.csv', index=True, index_label='sample_id')
    
    # Step 5: Analyze and save report
    print("Analyzing data...")
    analysis = analyze_metadata(metadata_df)
    
    # Convert numpy types to native types
    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    analysis_converted = convert_numpy(analysis)
    
    with open(out_dir / 'week1_data_report.json', 'w') as f:
        json.dump(analysis_converted, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
