# -*- coding: utf-8 -*-
"""Build frozen MixedWM38K split and query manifests."""

import argparse
import json
from pathlib import Path

from .io import CLASS_NAMES, label_signature, load_valid_wm38k, signature_counts
from .manifest import write_query_manifest, write_split_manifest
from .query import stratified_query_sample
from .split import stratified_split_by_signature


def main():
    args = parse_args()
    _, labels, original_indices = load_valid_wm38k(args.data_file)
    split_indices = stratified_split_by_signature(
        labels,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_split_manifest(args.split_out, labels, original_indices, split_indices)

    query_indices = stratified_query_sample(
        labels,
        split_indices['test'],
        num_queries=args.num_queries,
        seed=args.seed,
    )
    write_query_manifest(args.query_out, labels, original_indices, query_indices)

    meta = _build_meta(labels, split_indices, query_indices, args)
    meta_path = Path(args.meta_out) if args.meta_out else Path(args.split_out).with_suffix('.meta.json')
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Saved split manifest: {args.split_out}')
    print(f'Saved query manifest: {args.query_out}')
    print(f'Saved metadata: {meta_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--split-out', type=str, default='artifacts/splits/wm38k_seed2026_sig_70_10_20.csv')
    parser.add_argument('--query-out', type=str, default='artifacts/splits/wm38k_seed2026_test_queries_2000.csv')
    parser.add_argument('--meta-out', type=str, default=None)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--num-queries', type=int, default=2000)
    return parser.parse_args()


def _build_meta(labels, split_indices, query_indices, args):
    counts = signature_counts(labels)
    split_counts = {
        split: _signature_count_for_indices(labels, indices)
        for split, indices in split_indices.items()
    }
    return {
        'data_file': args.data_file,
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'valid_ratio': args.valid_ratio,
        'test_ratio': args.test_ratio,
        'num_valid_samples': int(len(labels)),
        'num_queries': int(len(query_indices)),
        'class_names': CLASS_NAMES,
        'num_signatures': int(len(counts)),
        'signature_counts': {_sig_to_str(k): int(v) for k, v in sorted(counts.items())},
        'split_sizes': {k: int(len(v)) for k, v in split_indices.items()},
        'query_size': int(len(query_indices)),
        'split_signature_counts': split_counts,
    }


def _signature_count_for_indices(labels, indices):
    out = {}
    for valid_index in indices:
        sig = _sig_to_str(label_signature(labels[int(valid_index)]))
        out[sig] = out.get(sig, 0) + 1
    return dict(sorted(out.items()))


def _sig_to_str(sig):
    return '|'.join(str(x) for x in sig)


if __name__ == '__main__':
    main()
