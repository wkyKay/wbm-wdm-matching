# -*- coding: utf-8 -*-
"""Evaluate proposal-based retrieval rankings and proposal token statistics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from partial_match.data.data_io import CLASS_NAMES, filter_valid_samples, load_wm38k


def main():
    args = parse_args()
    evaluate_proposal_retrieval(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/wm38k/Wafer_Map_Datasets.npz')
    parser.add_argument('--rankings', type=str, required=True)
    parser.add_argument('--tokens', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--k', type=int, nargs='+', default=[1, 3, 5, 10])
    return parser.parse_args()


def evaluate_proposal_retrieval(args):
    maps, labels = load_wm38k(args.data_file)
    _, labels, original_indices = filter_valid_samples(maps, labels)

    rankings = pd.read_csv(args.rankings)
    eval_ids = _ranking_ids(rankings)
    full_label_by_id = {int(orig_id): labels[pos].astype(np.int32) for pos, orig_id in enumerate(original_indices)}
    label_by_id = {sample_id: full_label_by_id[sample_id] for sample_id in eval_ids if sample_id in full_label_by_id}
    metrics = evaluate_rankings(rankings, label_by_id, ks=args.k)
    if args.tokens:
        metrics['proposal_stats'] = proposal_stats(pd.read_csv(args.tokens))

    out_path = Path(args.out) if args.out else Path(args.rankings).with_name('metrics_summary.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f'Saved metrics to {out_path}')
    return metrics


def evaluate_rankings(rankings: pd.DataFrame, label_by_id: dict, ks=(1, 3, 5, 10)):
    query_ids = [int(x) for x in rankings['query_id'].drop_duplicates().tolist()]
    out = {
        'num_queries': len(query_ids),
        'num_eval_samples': len(label_by_id),
        'candidate_pool': 'rankings_file',
        'retrieval': {},
        'exact_set': {},
        'per_class': {},
    }

    ap_hit = []
    ap_exact = []
    per_k = {
        k: {
            'precision_hit': [],
            'recall_hit': [],
            'ndcg_jaccard': [],
            'hit_rate': [],
            'exact_rate': [],
            'mean_jaccard': [],
        }
        for k in ks
    }
    per_class_acc = {
        name: {k: {'hit_rate': [], 'exact_rate': [], 'mean_jaccard': []} for k in ks}
        for name in CLASS_NAMES
    }

    for qid in query_ids:
        if qid not in label_by_id:
            continue
        q_label = label_by_id[qid]
        q_set = _label_set(q_label)
        group = rankings[rankings['query_id'] == qid].sort_values('rank')
        candidate_ids = [int(x) for x in group['candidate_id'].tolist() if int(x) in label_by_id and int(x) != qid]
        hit_rel = np.asarray([_overlap_hit(q_label, label_by_id[cid]) for cid in candidate_ids], dtype=np.float32)
        exact_rel = np.asarray([_label_set(label_by_id[cid]) == q_set for cid in candidate_ids], dtype=np.float32)
        jac_rel = np.asarray([_jaccard(q_label, label_by_id[cid]) for cid in candidate_ids], dtype=np.float32)

        total_hit = max(float(hit_rel.sum()), 1.0)
        total_exact = max(float(exact_rel.sum()), 1.0)
        ap_hit.append(_average_precision(hit_rel, total_hit))
        ap_exact.append(_average_precision(exact_rel, total_exact))

        for k in ks:
            kk = min(k, len(candidate_ids))
            top_hit = hit_rel[:kk]
            top_exact = exact_rel[:kk]
            top_jac = jac_rel[:kk]
            per_k[k]['precision_hit'].append(float(top_hit.mean()) if kk else 0.0)
            per_k[k]['recall_hit'].append(float(top_hit.sum() / total_hit))
            per_k[k]['hit_rate'].append(float(top_hit.max()) if kk else 0.0)
            per_k[k]['exact_rate'].append(float(top_exact.max()) if kk else 0.0)
            per_k[k]['mean_jaccard'].append(float(top_jac.mean()) if kk else 0.0)
            per_k[k]['ndcg_jaccard'].append(_ndcg(top_jac, _ideal_gains(jac_rel, kk)))

            for class_idx, class_name in enumerate(CLASS_NAMES):
                if int(q_label[class_idx]) != 1:
                    continue
                per_class_acc[class_name][k]['hit_rate'].append(float(top_hit.max()) if kk else 0.0)
                per_class_acc[class_name][k]['exact_rate'].append(float(top_exact.max()) if kk else 0.0)
                per_class_acc[class_name][k]['mean_jaccard'].append(float(top_jac.mean()) if kk else 0.0)

    out['retrieval']['mAP_hit'] = float(np.mean(ap_hit)) if ap_hit else 0.0
    out['exact_set']['mAP_exact'] = float(np.mean(ap_exact)) if ap_exact else 0.0
    for k in ks:
        out['retrieval'][f'Precision@{k}'] = _mean(per_k[k]['precision_hit'])
        out['retrieval'][f'Recall@{k}'] = _mean(per_k[k]['recall_hit'])
        out['retrieval'][f'NDCG@{k}'] = _mean(per_k[k]['ndcg_jaccard'])
        out['retrieval'][f'HitRate@{k}'] = _mean(per_k[k]['hit_rate'])
        out['retrieval'][f'MeanJaccard@{k}'] = _mean(per_k[k]['mean_jaccard'])
        out['exact_set'][f'ExactRate@{k}'] = _mean(per_k[k]['exact_rate'])

    for class_name, class_metrics in per_class_acc.items():
        if not any(class_metrics[k]['hit_rate'] for k in ks):
            continue
        out['per_class'][class_name] = {}
        for k in ks:
            out['per_class'][class_name][f'HitRate@{k}'] = _mean(class_metrics[k]['hit_rate'])
            out['per_class'][class_name][f'ExactRate@{k}'] = _mean(class_metrics[k]['exact_rate'])
            out['per_class'][class_name][f'MeanJaccard@{k}'] = _mean(class_metrics[k]['mean_jaccard'])
    return out


def _ranking_ids(rankings: pd.DataFrame):
    ids = set()
    for column in ('query_id', 'candidate_id'):
        if column in rankings:
            ids.update(int(x) for x in rankings[column].dropna().tolist())
    return ids


def proposal_stats(tokens: pd.DataFrame):
    out = {
        'num_maps_with_tokens': int(tokens['map_id'].nunique()) if not tokens.empty else 0,
        'num_tokens': int(len(tokens)),
        'tokens_per_map_mean': float(tokens.groupby('map_id').size().mean()) if not tokens.empty else 0.0,
        'tokens_per_map_median': float(tokens.groupby('map_id').size().median()) if not tokens.empty else 0.0,
        'area_mean': float(tokens['area'].mean()) if 'area' in tokens else 0.0,
        'area_median': float(tokens['area'].median()) if 'area' in tokens else 0.0,
        'geometry_type_counts': tokens['geometry_type'].value_counts().to_dict() if 'geometry_type' in tokens else {},
        'proposal_type_counts': tokens['proposal_type'].value_counts().to_dict() if 'proposal_type' in tokens else {},
    }
    if 'geometry_type' in tokens:
        per_type = {}
        for geometry_type, group in tokens.groupby('geometry_type'):
            per_type[geometry_type] = {
                'count': int(len(group)),
                'area_mean': float(group['area'].mean()),
                'area_median': float(group['area'].median()),
            }
        out['per_geometry_type'] = per_type
    return out


def _label_set(label):
    return tuple(np.where(np.asarray(label).astype(np.int32) == 1)[0].tolist())


def _overlap_hit(a, b):
    return int(np.dot(np.asarray(a).astype(np.int32), np.asarray(b).astype(np.int32)) > 0)


def _jaccard(a, b):
    a = np.asarray(a).astype(np.int32)
    b = np.asarray(b).astype(np.int32)
    inter = int(np.dot(a, b))
    union = int(a.sum() + b.sum() - inter)
    return float(inter / max(union, 1))


def _average_precision(rel, total):
    if len(rel) == 0:
        return 0.0
    cum = np.cumsum(rel)
    pos = np.where(rel > 0)[0]
    return float((cum[pos] / (pos + 1)).sum() / max(total, 1)) if len(pos) else 0.0


def _ndcg(actual_gains, ideal_gains):
    if len(actual_gains) == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, len(actual_gains) + 2))
    dcg = float((actual_gains * discount).sum())
    idcg = float((ideal_gains * discount).sum())
    return dcg / idcg if idcg > 0 else 0.0


def _ideal_gains(gains, k):
    return np.asarray(sorted(np.asarray(gains, dtype=np.float32), reverse=True)[:k], dtype=np.float32)


def _mean(values):
    return float(np.mean(values)) if values else 0.0


if __name__ == '__main__':
    main()
