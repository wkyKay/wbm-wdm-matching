# -*- coding: utf-8 -*-
"""Run proposed learned local retrieval and official label evaluation."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from proposed.core.cluster_patches import PatchBuilder, PatchConfig
from proposed.core.matching import MatchingConfig
from proposed.core.proposal import PartialMatchProposalProvider, ProposalConfig, load_tokens_csv, save_proposal_config
from proposed.datasets.wm38k_maps import load_split_records
from proposed.models.encoder import build_encoder
from proposed.tasks.learned_retrieval import run_learned_retrieval
from shared.wm38k.candidates import candidate_manifest_ids
from shared.wm38k.manifest import load_query_ids


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_config(out_dir / 'configs.json', vars(args))
    proposal_config = ProposalConfig(
        min_area=args.min_area,
        top_k=args.top_k_proposals,
    )
    patch_config = PatchConfig(patch_size=args.patch_size)
    save_proposal_config(out_dir / 'proposal_config.json', proposal_config)
    _save_config(out_dir / 'patch_config.json', asdict(patch_config))

    records = load_split_records(args.data_file, args.split_manifest, split=args.split)
    records = _restrict_records(records, args.query_manifest, args.candidate_manifest)
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
        split_manifest=args.split_manifest,
        query_manifest=args.query_manifest,
        candidate_manifest=args.candidate_manifest,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        matching_config=_matching_config(args),
        metric_k=tuple(args.metric_k),
    )
    print(f'Rankings: {result["rankings_path"]}')
    print(f'Label metrics: {result["metrics_path"]}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--split-manifest', type=str, required=True)
    parser.add_argument('--query-manifest', type=str, required=True)
    parser.add_argument('--candidate-manifest', type=str, default=None)
    parser.add_argument('--proposal-tokens', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--checkpoint-key', type=str, default='encoder')
    parser.add_argument('--split', type=str, default='test')
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
    parser.add_argument('--metric-k', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out-dir', type=str, default='artifacts/proposed/retrieval')
    return parser.parse_args()


def _restrict_records(records, query_manifest, candidate_manifest):
    if not candidate_manifest:
        return records
    needed = set(load_query_ids(query_manifest))
    needed.update(candidate_manifest_ids(candidate_manifest))
    return [record for record in records if int(record['map_id']) in needed]


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


def _save_config(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
