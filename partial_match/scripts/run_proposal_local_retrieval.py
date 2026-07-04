# -*- coding: utf-8 -*-
"""Run proposal-based handcrafted local retrieval baseline."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from partial_match.core.clustering import cluster
from partial_match.core.descriptors import clusters_to_records, explain_map_similarity, map_similarity
from partial_match.data.data_io import filter_valid_samples, load_wm38k
from shared.wm38k.candidates import load_candidate_manifest
from shared.wm38k.manifest import load_query_ids, load_split_manifest


def main():
    args = parse_args()
    run_proposal_local_retrieval(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/wm38k/Wafer_Map_Datasets.npz')
    parser.add_argument('--out', type=str, default='artifacts/proposal_based/rankings.csv')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--sample-strategy', type=str, default='head', choices=['head', 'random', 'stratified'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-manifest', type=str, default=None)
    parser.add_argument('--query-manifest', type=str, default=None)
    parser.add_argument('--candidate-manifest', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test', 'all'])
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
    return parser.parse_args()


def run_proposal_local_retrieval(args):
    maps, labels = load_wm38k(args.data_file)
    maps, labels, original_indices = filter_valid_samples(maps, labels)
    query_original_ids = None
    candidate_ids_by_query = None
    if getattr(args, 'split_manifest', None):
        rows = load_split_manifest(args.split_manifest, split=getattr(args, 'split', 'test'))
        sample_indices = np.asarray([int(row['valid_index']) for row in rows], dtype=np.int64)
        maps = maps[sample_indices]
        labels = labels[sample_indices]
        original_indices = original_indices[sample_indices]
        if getattr(args, 'query_manifest', None):
            query_original_ids = set(load_query_ids(args.query_manifest))
        if getattr(args, 'candidate_manifest', None):
            candidate_ids_by_query = load_candidate_manifest(args.candidate_manifest)
    elif args.max_samples is not None:
        sample_indices = _sample_indices(labels, args.max_samples, args.sample_strategy, args.seed)
        maps = maps[sample_indices]
        labels = labels[sample_indices]
        original_indices = original_indices[sample_indices]

    records = []
    descriptor_arrays = {}
    token_rows = []
    for i, raw in enumerate(maps):
        defect_mask = raw == 2
        valid_mask = (raw == 1) | (raw == 2)
        clusters = cluster(
            defect_mask,
            valid_mask,
            method=args.method,
            min_area=args.min_area,
            top_k=args.top_k_proposals,
            enable_ring_aware=not args.disable_ring_aware,
            max_defect_ratio_for_ring=args.max_defect_ratio_for_ring,
            min_edge_defect_fraction_for_ring=args.min_edge_defect_fraction_for_ring,
        )
        tokens = clusters_to_records(clusters, raw.shape)
        for token in tokens:
            key = f'{int(original_indices[i])}_{token["token_id"]}'
            descriptor_arrays[key] = token['descriptor'].astype(np.float32)
            token_rows.append(_token_row(int(original_indices[i]), token))
        records.append({
            'idx': int(original_indices[i]),
            'tokens': tokens,
            'label': labels[i],
        })
        if (i + 1) % 100 == 0:
            print(f'Prepared {i + 1}/{len(maps)} maps')

    id_to_position = {int(record['idx']): i for i, record in enumerate(records)}
    query_positions = _query_positions(records, query_original_ids)
    scores = np.full((len(query_positions), len(records)), -np.inf, dtype=np.float32)
    for qi, i in enumerate(query_positions):
        query = records[i]
        candidate_positions = _candidate_positions_for_query(query['idx'], records, id_to_position, candidate_ids_by_query)
        for j in candidate_positions:
            candidate = records[j]
            if i == j:
                continue
            scores[qi, j] = map_similarity(
                query['tokens'],
                candidate['tokens'],
                sigma_pos=args.sigma_pos,
                sigma_area=args.sigma_area,
                topk=args.topk_match,
            )
        if (qi + 1) % 50 == 0:
            print(f'Scored {qi + 1}/{len(query_positions)} queries')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_token_details:
        _save_token_details(out_path.parent, token_rows, descriptor_arrays)

    rankings = np.argsort(-scores, axis=1)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'rank', 'candidate_id', 'similarity_score'])
        for qi, ranked in enumerate(rankings):
            i = query_positions[qi]
            rank_out = 1
            for j in ranked:
                if i == j:
                    continue
                writer.writerow([records[i]['idx'], rank_out, records[j]['idx'], float(scores[qi, j])])
                rank_out += 1
    if args.save_match_details:
        _save_match_details(
            out_path.parent,
            records,
            query_positions,
            rankings,
            scores,
            sigma_pos=args.sigma_pos,
            sigma_area=args.sigma_area,
            topk_match=args.topk_match,
            max_queries=args.match_detail_top_queries,
            max_candidates=args.match_detail_top_candidates,
        )
    print(f'Saved rankings to {out_path}')
    return {
        'rankings_path': out_path,
        'tokens_path': out_path.parent / 'tokens.csv',
        'descriptors_path': out_path.parent / 'descriptors.npz',
    }


def _query_positions(records, query_original_ids):
    if query_original_ids is None:
        return list(range(len(records)))
    positions = [i for i, record in enumerate(records) if int(record['idx']) in query_original_ids]
    missing = sorted(query_original_ids - {int(records[i]['idx']) for i in positions})
    if missing:
        raise ValueError(f'{len(missing)} query ids are not in the selected candidate split. First missing: {missing[:5]}')
    return positions


def _candidate_positions_for_query(query_id, records, id_to_position, candidate_ids_by_query):
    if candidate_ids_by_query is None:
        return list(range(len(records)))
    candidate_ids = candidate_ids_by_query.get(int(query_id), [])
    missing = [candidate_id for candidate_id in candidate_ids if int(candidate_id) not in id_to_position]
    if missing:
        raise ValueError(f'{len(missing)} candidate ids for query {query_id} are not in selected split. First missing: {missing[:5]}')
    return [id_to_position[int(candidate_id)] for candidate_id in candidate_ids]


def _sample_indices(labels, max_samples, strategy, seed):
    n = len(labels)
    max_samples = min(max_samples, n)
    if strategy == 'head':
        return np.arange(max_samples)

    rng = np.random.default_rng(seed)
    if strategy == 'random':
        return np.sort(rng.choice(n, size=max_samples, replace=False))

    groups = {}
    for idx, label in enumerate(labels):
        signature = tuple(np.where(label.astype(np.int32) == 1)[0].tolist())
        groups.setdefault(signature, []).append(idx)

    selected = []
    shuffled_groups = []
    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        shuffled_groups.append(indices.tolist())

    while len(selected) < max_samples:
        made_progress = False
        for indices in shuffled_groups:
            if indices and len(selected) < max_samples:
                selected.append(indices.pop())
                made_progress = True
        if not made_progress:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _token_row(map_id, token):
    cluster = token['cluster']
    return {
        'map_id': map_id,
        'token_id': int(token['token_id']),
        'geometry_type': token['geometry_type'],
        'area': float(token['area']),
        'area_ratio': float(token['area_ratio']),
        'centroid_row': float(cluster.get('centroid_row', 0.0)),
        'centroid_col': float(cluster.get('centroid_col', 0.0)),
        'bbox_row_min': int(cluster.get('bbox_row_min', 0)),
        'bbox_col_min': int(cluster.get('bbox_col_min', 0)),
        'bbox_row_max': int(cluster.get('bbox_row_max', 0)),
        'bbox_col_max': int(cluster.get('bbox_col_max', 0)),
        'bbox_height': int(cluster.get('bbox_height', 0)),
        'bbox_width': int(cluster.get('bbox_width', 0)),
        'compactness': float(cluster.get('compactness', 0.0)),
        'orientation': float(cluster.get('orientation', 0.0)),
        'radial_distance_norm': float(cluster.get('radial_distance_norm', 0.0)),
        'proposal_type': cluster.get('proposal_type', ''),
        'proposal_source': cluster.get('proposal_source', ''),
        'angular_coverage': float(cluster.get('angular_coverage', 0.0)),
        'radial_std': float(cluster.get('radial_std', 0.0)),
    }


def _save_token_details(out_dir, token_rows, descriptor_arrays):
    token_path = out_dir / 'tokens.csv'
    fieldnames = list(token_rows[0].keys()) if token_rows else ['map_id', 'token_id']
    with open(token_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in token_rows:
            writer.writerow(row)
    np.savez_compressed(out_dir / 'descriptors.npz', **descriptor_arrays)
    print(f'Saved token details to {token_path}')


def _save_match_details(out_dir, records, query_positions, rankings, scores, sigma_pos, sigma_area, topk_match,
                        max_queries, max_candidates):
    path = out_dir / 'match_details.csv'
    fieldnames = [
        'query_id',
        'candidate_id',
        'candidate_rank',
        'map_similarity_score',
        'query_token_id',
        'candidate_token_id',
        'match_rank',
        'query_type',
        'candidate_type',
        'query_area',
        'candidate_area',
        'score',
        'shape_sim',
        'position_affinity',
        'scale_affinity',
        'type_affinity',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qi in range(min(max_queries, len(query_positions))):
            query_pos = query_positions[qi]
            query = records[query_pos]
            rank_out = 1
            for cj in rankings[qi]:
                if query_pos == cj:
                    continue
                candidate = records[int(cj)]
                explanation = explain_map_similarity(
                    query['tokens'],
                    candidate['tokens'],
                    sigma_pos=sigma_pos,
                    sigma_area=sigma_area,
                    topk=topk_match,
                )
                for match in explanation['matches']:
                    row = {
                        'query_id': query['idx'],
                        'candidate_id': candidate['idx'],
                        'candidate_rank': rank_out,
                        'map_similarity_score': float(scores[qi, cj]),
                    }
                    row.update(match)
                    writer.writerow(row)
                rank_out += 1
                if rank_out > max_candidates:
                    break
    print(f'Saved match details to {path}')


if __name__ == '__main__':
    main()
