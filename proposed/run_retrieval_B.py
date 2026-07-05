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
from proposed.core.proposal import PartialMatchProposalProvider, ProposalConfig, load_tokens_csv, save_proposal_config
from proposed.datasets.wm38k_maps import load_split_records
from proposed.models.encoder import ClusterEncoder
from proposed.tasks.learned_retrieval import run_learned_retrieval


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_config = ProposalConfig(
        method=args.proposal_method,
        min_area=args.min_area,
        top_k=args.top_k_proposals,
        enable_ring_aware=not args.disable_ring_aware,
        max_defect_ratio_for_ring=args.max_defect_ratio_for_ring,
        min_edge_defect_fraction_for_ring=args.min_edge_defect_fraction_for_ring,
    )
    patch_config = PatchConfig(patch_size=args.patch_size, window_size=args.patch_window_size)
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
    encoder = ClusterEncoder(in_channels=3, embedding_dim=args.embedding_dim, width=args.encoder_width)
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
        topk_match=args.topk_match,
        sigma_pos=args.sigma_pos,
        sigma_area=args.sigma_area,
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
    parser.add_argument('--proposal-method', type=str, default='retrieval_compact')
    parser.add_argument('--min-area', type=int, default=5)
    parser.add_argument('--top-k-proposals', type=int, default=6)
    parser.add_argument('--disable-ring-aware', action='store_true')
    parser.add_argument('--max-defect-ratio-for-ring', type=float, default=0.45)
    parser.add_argument('--min-edge-defect-fraction-for-ring', type=float, default=0.45)
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--patch-window-size', type=int, default=52)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--encoder-width', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--topk-match', type=int, default=1)
    parser.add_argument('--sigma-pos', type=float, default=0.35)
    parser.add_argument('--sigma-area', type=float, default=1.0)
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


if __name__ == '__main__':
    main()
