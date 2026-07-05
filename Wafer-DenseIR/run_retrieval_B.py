# -*- coding: utf-8 -*-
"""Run Experiment B for Wafer-DenseIR."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.experiment_b.evaluate_preferences import evaluate_preferences_from_files, write_details


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    hash_name = args.hash
    cmd = [
        sys.executable,
        str(Path(__file__).with_name('run_dense_retrieval.py')),
        '--data_file', args.b_data,
        '--split_manifest', args.b_split_manifest,
        '--query_manifest', args.b_queries,
        '--candidate_manifest', args.b_candidates,
        '--split', 'test',
        '--output_root', str(out_root),
        '--hash', hash_name,
        '--device', args.device,
        '--backbone_type', args.backbone_type,
        '--backbone_config', args.backbone_config,
        '--input_size', str(args.input_size),
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--token_mode', args.token_mode,
        '--topk_tokens', str(args.topk_tokens),
        '--sigma_pos', str(args.sigma_pos),
        '--max_tokens', str(args.max_tokens),
        '--topk_retrieval', '1', '5', '10',
        '--explain_top_queries', str(args.explain_top_queries),
    ]
    if args.no_decouple_input:
        cmd.append('--no_decouple_input')
    else:
        cmd.append('--decouple_input')
    if args.pretrained_model_file:
        cmd.extend(['--pretrained_model_file', args.pretrained_model_file])
    if args.pretrained_model_key:
        cmd.extend(['--pretrained_model_key', args.pretrained_model_key])
    if args.save_features:
        cmd.append('--save_features')
    subprocess.run(cmd, check=True)

    out_dir = out_root / 'wm38k' / 'denseir' / f'{args.backbone_type}.{_normalized_backbone(args)}' / hash_name
    rankings = out_dir / 'rankings.csv'
    metrics = evaluate_preferences_from_files(str(rankings), args.b_preferences)
    details = metrics.pop('details')
    (out_dir / 'preference_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    if args.save_details:
        write_details(details, out_dir / 'preference_details.csv')
    print(json.dumps(metrics, indent=2))
    print(f'Rankings: {rankings}')
    print(f'Preference metrics: {out_dir / "preference_metrics.json"}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiment B for Wafer-DenseIR.')
    parser.add_argument('--b-data', type=str, required=True)
    parser.add_argument('--b-split-manifest', type=str, required=True)
    parser.add_argument('--b-queries', type=str, required=True)
    parser.add_argument('--b-candidates', type=str, required=True)
    parser.add_argument('--b-preferences', type=str, required=True)
    parser.add_argument('--out-root', type=str, default='artifacts/preference_b/denseir')
    parser.add_argument('--hash', type=str, default='experiment_b')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'vit'])
    parser.add_argument('--backbone-config', type=str, default='18')
    parser.add_argument('--input-size', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--token-mode', type=str, default='defect_band')
    parser.add_argument('--topk-tokens', type=int, default=5)
    parser.add_argument('--sigma-pos', type=float, default=0.35)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--explain-top-queries', type=int, default=0)
    parser.add_argument('--pretrained-model-file', type=str, default=None)
    parser.add_argument('--pretrained-model-key', type=str, default=None)
    parser.add_argument('--no-decouple-input', action='store_true')
    parser.add_argument('--save-features', action='store_true')
    parser.add_argument('--save-details', action='store_true')
    return parser.parse_args()


def _normalized_backbone(args):
    if args.backbone_type == 'vit' and args.backbone_config == '18':
        return 'tiny'
    return args.backbone_config


if __name__ == '__main__':
    main()
