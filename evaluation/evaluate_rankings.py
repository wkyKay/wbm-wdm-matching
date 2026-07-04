# -*- coding: utf-8 -*-
"""Evaluate method ranking files with shared WM38K split/query manifests."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.metrics import average_precision, mean_or_none, ndcg_at_k
from evaluation.relevance import exact_match, gain, jaccard, overlap_hit, relevance_value
from evaluation.schemas import validate_ranking_columns
from shared.wm38k.io import CLASS_NAMES
from shared.wm38k.candidates import load_candidate_manifest
from shared.wm38k.manifest import load_query_ids, load_split_manifest


def main():
    args = parse_args()
    metrics = evaluate_rankings_from_files(
        rankings_path=args.rankings,
        split_manifest=args.split_manifest,
        query_manifest=args.query_manifest,
        candidate_manifest=args.candidate_manifest,
        split=args.split,
        ks=args.k,
        relevance_mode=args.relevance_mode,
        gain_mode=args.gain_mode,
        strict=args.strict,
    )
    out_path = Path(args.out) if args.out else Path(args.rankings).with_name('label_metrics.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    if args.flat_out:
        write_flat_metrics(metrics, args.flat_out)
    print(json.dumps(metrics, indent=2))
    print(f'Saved metrics to {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate label-derived retrieval metrics from a ranking file.')
    parser.add_argument('--rankings', type=str, required=True)
    parser.add_argument('--split-manifest', type=str, required=True)
    parser.add_argument('--query-manifest', type=str, required=True)
    parser.add_argument('--candidate-manifest', type=str, default=None)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--flat-out', type=str, default=None)
    parser.add_argument('--k', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--relevance-mode', type=str, default='jaccard', choices=['jaccard', 'tier'])
    parser.add_argument('--gain-mode', type=str, default='identity', choices=['identity', 'exp2'])
    parser.add_argument('--strict', action='store_true', help='Fail if queries or candidates from the manifest are missing in the ranking file.')
    return parser.parse_args()


def evaluate_rankings_from_files(rankings_path, split_manifest, query_manifest, candidate_manifest=None, split='test', ks=(1, 5, 10),
                                 relevance_mode='jaccard', gain_mode='identity', strict=False):
    label_by_id, candidate_ids = load_labels_from_split_manifest(split_manifest, split=split)
    query_ids = load_query_ids(query_manifest)
    candidate_ids_by_query = load_candidate_manifest(candidate_manifest) if candidate_manifest else None
    rankings = load_rankings(rankings_path)
    return evaluate_rankings(
        rankings=rankings,
        label_by_id=label_by_id,
        candidate_ids=candidate_ids,
        query_ids=query_ids,
        candidate_ids_by_query=candidate_ids_by_query,
        ks=ks,
        relevance_mode=relevance_mode,
        gain_mode=gain_mode,
        strict=strict,
    )


def load_labels_from_split_manifest(split_manifest, split='test'):
    rows = load_split_manifest(split_manifest, split=split)
    label_by_id = {}
    candidate_ids = []
    for row in rows:
        sample_id = int(row['sample_id'])
        label = np.asarray([int(row[f'label_{i}']) for i in range(len(CLASS_NAMES))], dtype=np.int32)
        label_by_id[sample_id] = label
        candidate_ids.append(sample_id)
    return label_by_id, candidate_ids


def load_rankings(path):
    with Path(path).open('r', newline='') as f:
        reader = csv.DictReader(f)
        validate_ranking_columns(reader.fieldnames)
        grouped = defaultdict(list)
        has_rank = 'rank' in (reader.fieldnames or [])
        for row in reader:
            query_id = int(row['query_id'])
            candidate_id = int(row['candidate_id'])
            score = float(row['similarity_score'])
            rank = int(row['rank']) if has_rank and row.get('rank') not in (None, '') else None
            grouped[query_id].append({'candidate_id': candidate_id, 'similarity_score': score, 'rank': rank})
    out = {}
    for query_id, rows in grouped.items():
        if any(row['rank'] is not None for row in rows):
            rows = sorted(rows, key=lambda row: (row['rank'] if row['rank'] is not None else 10**18, -row['similarity_score']))
        else:
            rows = sorted(rows, key=lambda row: -row['similarity_score'])
        seen = set()
        deduped = []
        for row in rows:
            candidate_id = row['candidate_id']
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            deduped.append(row)
        out[query_id] = deduped
    return out


def evaluate_rankings(rankings, label_by_id, candidate_ids, query_ids, candidate_ids_by_query=None, ks=(1, 5, 10),
                      relevance_mode='jaccard', gain_mode='identity', strict=False):
    candidate_ids = [int(x) for x in candidate_ids]
    query_ids = [int(x) for x in query_ids]
    ks = sorted(int(k) for k in ks)
    max_k = max(ks)
    missing_queries = [qid for qid in query_ids if qid not in rankings]
    if strict and missing_queries:
        raise ValueError(f'Ranking file is missing {len(missing_queries)} query ids. First missing: {missing_queries[:5]}')

    per_k = {
        k: {
            'LabelNDCG': [],
            'Precision': [],
            'Recall': [],
            'HitRate': [],
            'MeanJaccard': [],
            'ExactRate': [],
        }
        for k in ks
    }
    per_class = {
        class_name: {k: {'LabelNDCG': [], 'HitRate': [], 'MeanJaccard': [], 'ExactRate': []} for k in ks}
        for class_name in CLASS_NAMES
    }
    ap_hit = []
    ap_exact = []
    evaluated_queries = 0
    skipped_idcg_zero = {k: 0 for k in ks}
    ranking_candidate_counts = []
    invalid_candidate_count = 0

    for query_id in query_ids:
        if query_id not in label_by_id or query_id not in rankings:
            continue
        query_label = label_by_id[query_id]
        if candidate_ids_by_query is None:
            full_pool = [cid for cid in candidate_ids if cid != query_id]
        else:
            full_pool = [cid for cid in candidate_ids_by_query.get(query_id, []) if cid != query_id and cid in label_by_id]
            if strict and not full_pool:
                raise ValueError(f'Candidate manifest has no valid candidates for query {query_id}')
        full_hit = np.asarray([overlap_hit(query_label, label_by_id[cid]) for cid in full_pool], dtype=np.float32)
        full_exact = np.asarray([exact_match(query_label, label_by_id[cid]) for cid in full_pool], dtype=np.float32)
        full_gain = np.asarray([
            gain(relevance_value(query_label, label_by_id[cid], mode=relevance_mode), mode=gain_mode)
            for cid in full_pool
        ], dtype=np.float32)

        ranked_ids = []
        allowed = set(full_pool)
        for row in rankings[query_id]:
            candidate_id = int(row['candidate_id'])
            if candidate_id == query_id:
                continue
            if candidate_id not in label_by_id:
                invalid_candidate_count += 1
                continue
            if candidate_ids_by_query is not None and candidate_id not in allowed:
                continue
            ranked_ids.append(candidate_id)
        ranking_candidate_counts.append(len(ranked_ids))

        ranked_hit = np.asarray([overlap_hit(query_label, label_by_id[cid]) for cid in ranked_ids], dtype=np.float32)
        ranked_exact = np.asarray([exact_match(query_label, label_by_id[cid]) for cid in ranked_ids], dtype=np.float32)
        ranked_jaccard = np.asarray([jaccard(query_label, label_by_id[cid]) for cid in ranked_ids], dtype=np.float32)
        ranked_gain = np.asarray([
            gain(relevance_value(query_label, label_by_id[cid], mode=relevance_mode), mode=gain_mode)
            for cid in ranked_ids
        ], dtype=np.float32)

        ap_hit.append(average_precision(ranked_hit, total_relevant=float(full_hit.sum())))
        ap_exact.append(average_precision(ranked_exact, total_relevant=float(full_exact.sum())))
        evaluated_queries += 1

        for k in ks:
            kk = min(k, len(ranked_ids))
            top_hit = ranked_hit[:kk]
            top_exact = ranked_exact[:kk]
            top_jaccard = ranked_jaccard[:kk]
            ndcg = ndcg_at_k(ranked_gain, full_gain, k)
            if ndcg is None:
                skipped_idcg_zero[k] += 1
            per_k[k]['LabelNDCG'].append(ndcg)
            per_k[k]['Precision'].append(float(top_hit.mean()) if kk else 0.0)
            per_k[k]['Recall'].append(float(top_hit.sum() / max(float(full_hit.sum()), 1.0)))
            per_k[k]['HitRate'].append(float(top_hit.max()) if kk else 0.0)
            per_k[k]['MeanJaccard'].append(float(top_jaccard.mean()) if kk else 0.0)
            per_k[k]['ExactRate'].append(float(top_exact.max()) if kk else 0.0)

            for class_idx, class_name in enumerate(CLASS_NAMES):
                if int(query_label[class_idx]) != 1:
                    continue
                per_class[class_name][k]['LabelNDCG'].append(ndcg)
                per_class[class_name][k]['HitRate'].append(float(top_hit.max()) if kk else 0.0)
                per_class[class_name][k]['MeanJaccard'].append(float(top_jaccard.mean()) if kk else 0.0)
                per_class[class_name][k]['ExactRate'].append(float(top_exact.max()) if kk else 0.0)

    out = {
        'protocol': {
            'metric_group': 'A.label_derived',
            'relevance_mode': relevance_mode,
            'gain_mode': gain_mode,
            'candidate_pool': 'candidate_manifest' if candidate_ids_by_query is not None else 'split_manifest',
            'candidate_manifest': bool(candidate_ids_by_query is not None),
            'query_set': 'query_manifest',
            'ranking_format': 'query_id,candidate_id,similarity_score',
        },
        'counts': {
            'num_manifest_candidates': len(candidate_ids),
            'num_manifest_queries': len(query_ids),
            'num_evaluated_queries': evaluated_queries,
            'num_missing_queries_in_rankings': len(missing_queries),
            'num_invalid_ranked_candidates': invalid_candidate_count,
            'ranked_candidates_per_query_min': int(min(ranking_candidate_counts)) if ranking_candidate_counts else 0,
            'ranked_candidates_per_query_max': int(max(ranking_candidate_counts)) if ranking_candidate_counts else 0,
            'ranked_candidates_per_query_mean': float(np.mean(ranking_candidate_counts)) if ranking_candidate_counts else 0.0,
        },
        'retrieval': {
            'mAP_hit': float(np.mean(ap_hit)) if ap_hit else 0.0,
            'mAP_exact': float(np.mean(ap_exact)) if ap_exact else 0.0,
        },
        'per_class': {},
        'skipped': {
            f'idcg_zero@{k}': int(skipped_idcg_zero[k])
            for k in ks
        },
    }

    for k in ks:
        out['retrieval'][f'LabelNDCG@{k}'] = mean_or_none(per_k[k]['LabelNDCG'])
        out['retrieval'][f'Precision@{k}'] = mean_or_none(per_k[k]['Precision'])
        out['retrieval'][f'Recall@{k}'] = mean_or_none(per_k[k]['Recall'])
        out['retrieval'][f'HitRate@{k}'] = mean_or_none(per_k[k]['HitRate'])
        out['retrieval'][f'MeanJaccard@{k}'] = mean_or_none(per_k[k]['MeanJaccard'])
        out['retrieval'][f'ExactRate@{k}'] = mean_or_none(per_k[k]['ExactRate'])

    for class_name, class_metrics in per_class.items():
        values = {}
        for k in ks:
            values[f'LabelNDCG@{k}'] = mean_or_none(class_metrics[k]['LabelNDCG'])
            values[f'HitRate@{k}'] = mean_or_none(class_metrics[k]['HitRate'])
            values[f'MeanJaccard@{k}'] = mean_or_none(class_metrics[k]['MeanJaccard'])
            values[f'ExactRate@{k}'] = mean_or_none(class_metrics[k]['ExactRate'])
        if any(value is not None for value in values.values()):
            out['per_class'][class_name] = values
    return out


def write_flat_metrics(metrics, out_path):
    rows = []
    for section in ('retrieval', 'counts', 'skipped'):
        for metric, value in metrics.get(section, {}).items():
            rows.append({'section': section, 'metric': metric, 'value': value})
    for class_name, values in metrics.get('per_class', {}).items():
        for metric, value in values.items():
            rows.append({'section': f'per_class:{class_name}', 'metric': metric, 'value': value})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['section', 'metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
