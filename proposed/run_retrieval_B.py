# -*- coding: utf-8 -*-
"""Run Experiment B for the proposed learned local retrieval method."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.experiment_b.evaluate_preferences import evaluate_preferences_from_files, write_details
from proposed.core.cluster_patches import PatchBuilder, PatchConfig
from proposed.core.matching import MatchingConfig
from proposed.core.proposal import PartialMatchProposalProvider, ProposalConfig, load_tokens_csv, save_proposal_config
from proposed.datasets.wm38k_maps import load_split_records
from proposed.models.encoder import build_encoder
from proposed.tasks.learned_retrieval import run_learned_retrieval


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_config = ProposalConfig(
        min_area=args.min_area,
        top_k=args.top_k_proposals,
    )
    patch_config = PatchConfig(patch_size=args.patch_size)
    save_proposal_config(out_dir / 'proposal_config.json', proposal_config)
    (out_dir / 'patch_config.json').write_text(json.dumps(asdict(patch_config), indent=2), encoding='utf-8')
    records = load_split_records(args.b_data, args.b_split_manifest, split='test')
    if args.proposal_tokens and Path(args.proposal_tokens).exists():
        tokens = load_tokens_csv(args.proposal_tokens)
    else:
        provider = PartialMatchProposalProvider(proposal_config)
        tokens = []
        for i, record in enumerate(records, start=1):
            tokens.extend(provider.extract(record['map_id'], record['raw_map']))
            if i % 100 == 0:
                print(f'Extracted proposal tokens for {i}/{len(records)} maps')
    device = _get_device(args.device)
    encoder = build_encoder(args.encoder, in_channels=3, embedding_dim=args.embedding_dim, width=args.encoder_width)
    _load_encoder(encoder, args.checkpoint, args.checkpoint_key, device)
    result = run_learned_retrieval(
        records=records,
        tokens=tokens,
        patch_builder=PatchBuilder(patch_config),
        encoder=encoder,
        device=device,
        out_dir=out_dir,
        split_manifest=args.b_split_manifest,
        query_manifest=args.b_queries,
        candidate_manifest=args.b_candidates,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        matching_config=_matching_config(args),
        metric_k=(1, 5, 10),
    )
    metrics = evaluate_preferences_from_files(str(result['rankings_path']), args.b_preferences)
    details = metrics.pop('details')
    (out_dir / 'preference_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    if args.save_details:
        write_details(details, out_dir / 'preference_details.csv')
    print(json.dumps(metrics, indent=2))
    print(f'Rankings: {result["rankings_path"]}')
    print(f'Preference metrics: {out_dir / "preference_metrics.json"}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiment B for proposed learned retrieval.')
    parser.add_argument('--b-data', type=str, required=True)
    parser.add_argument('--b-split-manifest', type=str, required=True)
    parser.add_argument('--b-queries', type=str, required=True)
    parser.add_argument('--b-candidates', type=str, required=True)
    parser.add_argument('--b-preferences', type=str, required=True)
    parser.add_argument('--proposal-tokens', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--checkpoint-key', type=str, default='encoder')
    parser.add_argument('--min-area', type=int, default=5)
    parser.add_argument('--top-k-proposals', type=int, default=5)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--encoder', choices=['simple', 'resnet18'], default='simple')
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--encoder-width', type=int, default=32, help='Base width for --encoder simple only.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--sigma-pos', type=float, default=0.35)
    parser.add_argument('--sigma-scale', type=float, default=1.5)
    parser.add_argument('--min-token-score', type=float, default=0.30)
    parser.add_argument('--min-relative-token-area', type=float, default=0.10)
    parser.add_argument('--scale-ratio-min', type=float, default=0.20)
    parser.add_argument('--score-shape-weight', type=float, default=0.60)
    parser.add_argument('--score-position-weight', type=float, default=0.25)
    parser.add_argument('--score-scale-weight', type=float, default=0.15)
    parser.add_argument('--scale-area-weight', type=float, default=0.30)
    parser.add_argument('--scale-pca-weight', type=float, default=0.70)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out-dir', type=str, default='artifacts/preference_b/proposed')
    parser.add_argument('--save-details', action='store_true')
    return parser.parse_args()


def _load_encoder(encoder, checkpoint, key, device):
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt
    encoder.load_state_dict(state)


def _get_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; use --device cpu for local debugging.')
    return device


def _matching_config(args):
    return MatchingConfig(
        sigma_pos=args.sigma_pos,
        sigma_scale=args.sigma_scale,
        min_token_score=args.min_token_score,
        min_relative_token_area=args.min_relative_token_area,
        scale_ratio_min=args.scale_ratio_min,
        shape_weight=args.score_shape_weight,
        position_weight=args.score_position_weight,
        scale_weight=args.score_scale_weight,
        scale_area_weight=args.scale_area_weight,
        scale_pca_weight=args.scale_pca_weight,
    )


if __name__ == '__main__':
    main()
