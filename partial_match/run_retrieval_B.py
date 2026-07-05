# -*- coding: utf-8 -*-
"""Run Experiment B for the handcrafted local proposal baseline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.experiment_b.evaluate_preferences import evaluate_preferences_from_files, write_details
from partial_match.scripts.run_proposal_local_retrieval import run_proposal_local_retrieval
from shared.wm38k.candidates import load_candidate_manifest


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = out_dir / 'rankings.csv'
    run_proposal_local_retrieval(argparse.Namespace(
        data_file=args.b_data,
        out=str(rankings_path),
        max_samples=None,
        sample_strategy='head',
        seed=args.seed,
        split_manifest=args.b_split_manifest,
        query_manifest=args.b_queries,
        candidate_manifest=args.b_candidates,
        split='test',
        method=args.method,
        min_area=args.min_area,
        top_k_proposals=args.top_k_proposals,
        topk_match=args.topk_match,
        sigma_pos=args.sigma_pos,
        sigma_area=args.sigma_area,
        disable_ring_aware=args.disable_ring_aware,
        max_defect_ratio_for_ring=args.max_defect_ratio_for_ring,
        min_edge_defect_fraction_for_ring=args.min_edge_defect_fraction_for_ring,
        save_token_details=args.save_token_details,
        save_match_details=args.save_match_details,
        match_detail_top_queries=args.match_detail_top_queries,
        match_detail_top_candidates=args.match_detail_top_candidates,
    ))
    _filter_rankings_to_b_candidates(rankings_path, args.b_candidates)
    metrics = evaluate_preferences_from_files(str(rankings_path), args.b_preferences)
    details = metrics.pop('details')
    (out_dir / 'preference_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    if args.save_details:
        write_details(details, out_dir / 'preference_details.csv')
    print(json.dumps(metrics, indent=2))
    print(f'Rankings: {rankings_path}')
    print(f'Preference metrics: {out_dir / "preference_metrics.json"}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiment B for partial_match.')
    parser.add_argument('--b-data', type=str, required=True)
    parser.add_argument('--b-split-manifest', type=str, required=True)
    parser.add_argument('--b-queries', type=str, required=True)
    parser.add_argument('--b-candidates', type=str, required=True)
    parser.add_argument('--b-preferences', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='artifacts/preference_b/partial_match')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--method', type=str, default='retrieval_compact')
    parser.add_argument('--min-area', type=int, default=5)
    parser.add_argument('--top-k-proposals', type=int, default=6)
    parser.add_argument('--topk-match', type=int, default=1)
    parser.add_argument('--sigma-pos', type=float, default=0.35)
    parser.add_argument('--sigma-area', type=float, default=1.0)
    parser.add_argument('--disable-ring-aware', action='store_true')
    parser.add_argument('--max-defect-ratio-for-ring', type=float, default=0.45)
    parser.add_argument('--min-edge-defect-fraction-for-ring', type=float, default=0.45)
    parser.add_argument('--save-token-details', action='store_true')
    parser.add_argument('--save-match-details', action='store_true')
    parser.add_argument('--match-detail-top-queries', type=int, default=20)
    parser.add_argument('--match-detail-top-candidates', type=int, default=10)
    parser.add_argument('--save-details', action='store_true')
    return parser.parse_args()


def _filter_rankings_to_b_candidates(rankings_path, candidate_manifest):
    import csv
    allowed = {
        (int(query_id), int(candidate_id))
        for query_id, candidate_ids in load_candidate_manifest(candidate_manifest).items()
        for candidate_id in candidate_ids
    }
    rows = []
    with Path(rankings_path).open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['query_id']), int(row['candidate_id']))
            if key in allowed:
                rows.append(row)
    rows.sort(key=lambda row: (int(row['query_id']), int(row['rank'])))
    current_query = None
    current_rank = 0
    with Path(rankings_path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'rank', 'candidate_id', 'similarity_score'])
        writer.writeheader()
        for row in rows:
            query_id = int(row['query_id'])
            if query_id != current_query:
                current_query = query_id
                current_rank = 1
            else:
                current_rank += 1
            writer.writerow({
                'query_id': query_id,
                'rank': current_rank,
                'candidate_id': int(row['candidate_id']),
                'similarity_score': row['similarity_score'],
            })


if __name__ == '__main__':
    main()
