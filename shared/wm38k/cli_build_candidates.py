# -*- coding: utf-8 -*-
"""Build a fixed per-query candidate manifest for WM38K retrieval evaluation."""

import argparse

from .candidates import build_stratified_candidate_rows, write_candidate_manifest


def main():
    args = parse_args()
    rows = build_stratified_candidate_rows(
        split_manifest=args.split_manifest,
        query_manifest=args.query_manifest,
        split=args.split,
        seed=args.seed,
        exact=args.exact,
        high=args.high,
        weak=args.weak,
        none=args.none,
        total=args.total,
    )
    write_candidate_manifest(args.out, rows)
    print(f'Saved candidate manifest to {args.out}')
    print(f'Rows: {len(rows)}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-manifest', type=str, required=True)
    parser.add_argument('--query-manifest', type=str, required=True)
    parser.add_argument('--out', type=str, default='artifacts/splits/wm38k_seed2026_test_candidates_1000.csv')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--exact', type=int, default=50)
    parser.add_argument('--high', type=int, default=100)
    parser.add_argument('--weak', type=int, default=100)
    parser.add_argument('--none', type=int, default=750)
    parser.add_argument('--total', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    main()
