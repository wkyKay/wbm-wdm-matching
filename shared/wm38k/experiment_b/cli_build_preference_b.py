# -*- coding: utf-8 -*-
"""CLI for building Experiment B benchmark files."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.wm38k.experiment_b.preference_benchmark import build_experiment_b


def main():
    parser = argparse.ArgumentParser(description='Build Experiment B transformation preference benchmark.')
    parser.add_argument('--data-file', type=str, default='data/wm38k/Wafer_Map_Datasets.npz')
    parser.add_argument('--split-manifest', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='artifacts/preference_b/wm38k_seed2026')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num-queries', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    config = build_experiment_b(
        data_file=args.data_file,
        split_manifest=args.split_manifest,
        out_dir=args.out_dir,
        split=args.split,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
