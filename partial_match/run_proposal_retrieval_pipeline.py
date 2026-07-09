# -*- coding: utf-8 -*-
"""Run the proposal-based local retrieval pipeline (evaluation is done separately).

Pipeline:
  1) proposal extraction + handcrafted descriptors + local retrieval rankings
  2) query + top-k retrieved result visualization
  3) optional retrieval_compact proposal step visualization
"""

import argparse
# import csv
# import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# from partial_match.scripts.evaluate_proposal_retrieval import evaluate_proposal_retrieval
from partial_match.scripts.run_proposal_local_retrieval import run_proposal_local_retrieval
from partial_match.scripts.visualize_retrieval_compact_steps import visualize_retrieval_compact_steps
from partial_match.scripts.visualize_topk_retrieval import visualize_topk_retrieval
# from evaluation.experiment_a.evaluate_rankings import evaluate_rankings_from_files


def main():
    parser = argparse.ArgumentParser(
        description='Run proposal extraction, handcrafted local retrieval, evaluation, and review visualization.'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='../data/wm38k/Wafer_Map_Datasets.npz',
        help='Path to the WM38K npz file. Expected arrays: wafer maps and multi-hot labels.',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='../artifacts/proposal_based/system_test',
        help='Output directory for rankings, token details, metrics, and generated review figures.',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=512,
        help='Number of valid wafer maps used as the retrieval pool. Each selected map is also used as a query.',
    )
    parser.add_argument(
        '--sample-strategy',
        type=str,
        default='stratified',
        choices=['head', 'random', 'stratified'],
        help='How to select max-samples maps: head keeps dataset order, random samples uniformly, stratified balances label signatures.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for random/stratified sampling and proposal step figure sample selection.',
    )
    parser.add_argument(
        '--split-manifest',
        type=str,
        default=None,
        help='Frozen WM38K split manifest. When set, retrieval pool is selected from this manifest.',
    )
    parser.add_argument(
        '--query-manifest',
        type=str,
        default=None,
        help='Frozen query manifest. When set, only these test samples are used as queries.',
    )
    parser.add_argument(
        '--candidate-manifest',
        type=str,
        default=None,
        help='Optional per-query candidate manifest. When set, only listed query-candidate pairs are scored.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'valid', 'test', 'all'],
        help='Split selected from split-manifest for the retrieval candidate pool.',
    )

    parser.add_argument(
        '--method',
        type=str,
        default='retrieval_compact',
        help='Proposal method passed to partial_match.core.clustering.cluster. Default is retrieval_compact.',
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=5,
        help='Minimum connected-component area kept as a proposal token.',
    )
    parser.add_argument(
        '--top-k-proposals',
        type=int,
        default=6,
        help='Maximum number of proposal tokens retained per wafer map.',
    )
    parser.add_argument(
        '--topk-match',
        type=int,
        default=1,
        help='For each query token, average the top-k matched candidate tokens when computing map similarity.',
    )
    parser.add_argument(
        '--sigma-pos',
        type=float,
        default=0.35,
        help='Position affinity bandwidth for token matching. Larger values make matching less sensitive to location shifts.',
    )
    parser.add_argument(
        '--sigma-area',
        type=float,
        default=1.0,
        help='Area affinity bandwidth for token matching. Larger values make matching less sensitive to area differences.',
    )
    parser.add_argument(
        '--disable-ring-aware',
        action='store_true',
        help='Disable the ring-aware proposal token extraction stage.',
    )
    parser.add_argument(
        '--max-defect-ratio-for-ring',
        type=float,
        default=0.45,
        help='Skip ring-aware extraction when defect pixels occupy more than this fraction of the valid wafer area.',
    )
    parser.add_argument(
        '--min-edge-defect-fraction-for-ring',
        type=float,
        default=0.45,
        help='Require at least this fraction of defect pixels to lie near the wafer edge before ring-aware extraction is attempted.',
    )

    parser.add_argument(
        '--metric-k',
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help='K values used for retrieval metrics such as Precision@K, HitRate@K, NDCG@K, and ExactRate@K.',
    )
    parser.add_argument(
        '--review-top-k',
        type=int,
        default=3,
        help='Number of retrieved candidates shown next to each query in review figures.',
    )
    parser.add_argument(
        '--review-max-queries',
        type=int,
        default=64,
        help='Maximum number of query review figures to generate. This controls the number of TopK comparison images.',
    )
    parser.add_argument(
        '--review-query-ids',
        type=int,
        nargs='*',
        default=None,
        help='Optional explicit original dataset ids to visualize. If set, overrides review-max-queries selection.',
    )

    parser.add_argument(
        '--save-match-details',
        action='store_true',
        help='Save token-level match explanations for the highest-ranked candidates.',
    )
    parser.add_argument(
        '--match-detail-top-queries',
        type=int,
        default=20,
        help='Number of query maps for which token-level match details are saved when save-match-details is enabled.',
    )
    parser.add_argument(
        '--match-detail-top-candidates',
        type=int,
        default=10,
        help='Number of top candidates per query included in token-level match details.',
    )
    parser.add_argument(
        '--skip-review',
        action='store_true',
        help='Skip generation of query plus TopK retrieval review figures.',
    )
    parser.add_argument(
        '--save-step-figures',
        action='store_true',
        help='Generate proposal step debugging figures showing raw mask, denoising, ring mask, residual, and final tokens.',
    )
    parser.add_argument(
        '--step-samples',
        type=int,
        default=24,
        help='Maximum number of proposal step figures to generate when save-step-figures is enabled.',
    )
    parser.add_argument(
        '--step-sample-ids',
        type=int,
        nargs='*',
        default=None,
        help='Optional explicit original dataset ids for proposal step figures. If set, overrides step-samples selection.',
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rankings_path = out_dir / 'rankings.csv'
    tokens_path = out_dir / 'tokens.csv'
    # metrics_path = out_dir / 'metrics_summary.json'
    # label_metrics_path = out_dir / 'label_metrics.json'
    review_dir = out_dir / f'top{args.review_top_k}_review'
    steps_dir = out_dir / 'proposal_steps'

    print('\n[1/3] Running proposal extraction, descriptors, and local retrieval...', flush=True)
    run_proposal_local_retrieval(argparse.Namespace(
        data_file=args.data_file,
        out=str(rankings_path),
        max_samples=args.max_samples,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
        split_manifest=args.split_manifest,
        query_manifest=args.query_manifest,
        candidate_manifest=args.candidate_manifest,
        split=args.split,
        method=args.method,
        min_area=args.min_area,
        top_k_proposals=args.top_k_proposals,
        topk_match=args.topk_match,
        sigma_pos=args.sigma_pos,
        sigma_area=args.sigma_area,
        disable_ring_aware=args.disable_ring_aware,
        max_defect_ratio_for_ring=args.max_defect_ratio_for_ring,
        min_edge_defect_fraction_for_ring=args.min_edge_defect_fraction_for_ring,
        save_token_details=True,
        save_match_details=args.save_match_details,
        match_detail_top_queries=args.match_detail_top_queries,
        match_detail_top_candidates=args.match_detail_top_candidates,
    ))

    # --- Evaluation is done separately via evaluation/experiment_a/evaluate_rankings.py ---
    # print('\n[2/4] Evaluating retrieval metrics...', flush=True)
    # proposal_metrics = evaluate_proposal_retrieval(argparse.Namespace(
    #     data_file=args.data_file,
    #     rankings=str(rankings_path),
    #     tokens=str(tokens_path),
    #     out=str(metrics_path),
    #     k=args.metric_k,
    # ))
    # _write_flat_metrics(metrics_path, out_dir / 'metrics_summary_flat.csv')
    # if args.split_manifest and args.query_manifest:
    #     label_metrics = evaluate_rankings_from_files(
    #         rankings_path=str(rankings_path),
    #         split_manifest=args.split_manifest,
    #         query_manifest=args.query_manifest,
    #         candidate_manifest=args.candidate_manifest,
    #         split=args.split,
    #         ks=args.metric_k,
    #         relevance_mode='jaccard',
    #         gain_mode='identity',
    #         strict=False,
    #     )
    #     label_metrics['proposal_stats'] = proposal_metrics.get('proposal_stats', {})
    #     label_metrics_path.write_text(json.dumps(label_metrics, indent=2), encoding='utf-8')
    #     _write_flat_label_metrics(label_metrics, out_dir / 'label_metrics_flat.csv')
    #     print(f'Saved official label metrics to {label_metrics_path}')

    if not args.skip_review:
        print('\n[2/3] Rendering top-k retrieval review figures...', flush=True)
        visualize_topk_retrieval(argparse.Namespace(
            data_file=args.data_file,
            rankings=str(rankings_path),
            out_dir=str(review_dir),
            top_k=args.review_top_k,
            max_queries=args.review_max_queries,
            query_ids=args.review_query_ids,
            min_area=args.min_area,
            top_k_proposals=args.top_k_proposals,
            disable_ring_aware=args.disable_ring_aware,
            max_defect_ratio_for_ring=args.max_defect_ratio_for_ring,
            min_edge_defect_fraction_for_ring=args.min_edge_defect_fraction_for_ring,
        ))

    if args.save_step_figures:
        print('\n[3/3] Rendering proposal step figures...', flush=True)
        visualize_retrieval_compact_steps(argparse.Namespace(
            data_file=args.data_file,
            out_dir=str(steps_dir),
            samples=args.step_samples,
            sample_ids=args.step_sample_ids,
            seed=args.seed,
            min_area=args.min_area,
            top_k=args.top_k_proposals,
            edge_r_min=0.65,
            ring_band_width=0.10,
            min_ring_area=12,
            min_ring_angular_coverage=0.16,
            min_ring_area_ratio=0.12,
            max_ring_radial_std=0.12,
            max_defect_ratio_for_ring=args.max_defect_ratio_for_ring,
            min_edge_defect_fraction_for_ring=args.min_edge_defect_fraction_for_ring,
            disable_ring_aware=args.disable_ring_aware,
        ))

    print('Pipeline finished.')
    print(f'Rankings: {rankings_path}')
    print(f'Tokens: {tokens_path}')
    # print(f'Metrics: {metrics_path}')
    # if args.split_manifest and args.query_manifest:
    #     print(f'Official label metrics: {label_metrics_path}')
    if not args.skip_review:
        print(f'Review figures: {review_dir}')
    if args.save_step_figures:
        print(f'Proposal step figures: {steps_dir}')

# def _write_flat_metrics(metrics_path, out_path):
#     metrics = json.loads(metrics_path.read_text())
#     rows = []
#     for section in ['retrieval', 'exact_set']:
#         for metric, value in metrics.get(section, {}).items():
#             rows.append({'section': section, 'metric': metric, 'value': value})
#     for class_name, values in metrics.get('per_class', {}).items():
#         for metric, value in values.items():
#             rows.append({'section': f'per_class:{class_name}', 'metric': metric, 'value': value})
#     for metric, value in metrics.get('proposal_stats', {}).items():
#         if isinstance(value, (int, float, str)):
#             rows.append({'section': 'proposal_stats', 'metric': metric, 'value': value})
#
#     with out_path.open('w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=['section', 'metric', 'value'])
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f'Saved flat metrics to {out_path}')
#
#
# def _write_flat_label_metrics(metrics, out_path):
#     rows = []
#     for section in ['retrieval', 'counts', 'skipped']:
#         for metric, value in metrics.get(section, {}).items():
#             rows.append({'section': section, 'metric': metric, 'value': value})
#     for class_name, values in metrics.get('per_class', {}).items():
#         for metric, value in values.items():
#             rows.append({'section': f'per_class:{class_name}', 'metric': metric, 'value': value})
#     for metric, value in metrics.get('proposal_stats', {}).items():
#         if isinstance(value, (int, float, str)):
#             rows.append({'section': 'proposal_stats', 'metric': metric, 'value': value})
#     with out_path.open('w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=['section', 'metric', 'value'])
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f'Saved flat official label metrics to {out_path}')


if __name__ == '__main__':
    main()
