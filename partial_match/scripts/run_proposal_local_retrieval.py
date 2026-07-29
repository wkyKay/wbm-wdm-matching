# -*- coding: utf-8 -*-
"""Run proposal-based handcrafted local retrieval baseline."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from partial_match.core.arc_ring_retrieval import ArcRingConfig, prepare_tokens, score_tokens, token_row as arc_ring_token_row
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
    parser.add_argument('--min-area', type=int, default=5)
    parser.add_argument('--top-k-proposals', type=int, default=5)
    parser.add_argument('--sigma-pos', type=float, default=0.35)
    parser.add_argument('--min-token-score', type=float, default=0.30)
    parser.add_argument('--min-relative-token-area', type=float, default=0.10)
    parser.add_argument('--scale-ratio-min', type=float, default=0.50)
    parser.add_argument('--sigma-scale', type=float, default=1.5)
    parser.add_argument('--score-shape-weight', type=float, default=0.60)
    parser.add_argument('--score-position-weight', type=float, default=0.25)
    parser.add_argument('--score-scale-weight', type=float, default=0.15)
    parser.add_argument('--scale-area-weight', type=float, default=0.30)
    parser.add_argument('--scale-pca-weight', type=float, default=0.70)
    parser.add_argument('--moment-weight', type=float, default=0.75)
    parser.add_argument('--geometry-weight', type=float, default=0.25)
    parser.add_argument('--rotation-tolerance', action='store_true')
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

    arc_config = _arc_ring_config(args)
    records = []
    descriptor_arrays = {}
    token_rows = []
    for i, raw in enumerate(maps):
        defect_mask = raw == 2
        valid_mask = (raw == 1) | (raw == 2)
        tokens = prepare_tokens(defect_mask, valid_mask, arc_config)
        for token_id, token in enumerate(tokens):
            key = f'{int(original_indices[i])}_{token_id}'
            descriptor_arrays[key] = token['descriptor'].astype(np.float32)
            token_rows.append(arc_ring_token_row(int(original_indices[i]), token_id, token))
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
            scores[qi, j] = score_tokens(query['tokens'], candidate['tokens'], arc_config)['score']
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
                score = scores[qi, j]
                if score == float('-inf'):
                    continue
                writer.writerow([records[i]['idx'], rank_out, records[j]['idx'], float(score)])
                rank_out += 1
    if args.save_match_details:
        _save_match_details(
            out_path.parent,
            records,
            query_positions,
            rankings,
            scores,
            arc_config=arc_config,
            max_queries=args.match_detail_top_queries,
            max_candidates=args.match_detail_top_candidates,
        )
    print(f'Saved rankings to {out_path}')
    return {
        'rankings_path': out_path,
        'tokens_path': out_path.parent / 'tokens.csv',
        'descriptors_path': out_path.parent / 'descriptors.npz',
    }


def _arc_ring_config(args):
    return ArcRingConfig(
        min_area=args.min_area,
        top_k=args.top_k_proposals,
        sigma_pos=args.sigma_pos,
        sigma_scale=getattr(args, 'sigma_scale', 1.5),
        min_token_score=getattr(args, 'min_token_score', 0.30),
        min_relative_token_area=getattr(args, 'min_relative_token_area', 0.10),
        scale_ratio_min=getattr(args, 'scale_ratio_min', 0.50),
        shape_weight=getattr(args, 'score_shape_weight', 0.60),
        position_weight=getattr(args, 'score_position_weight', 0.25),
        scale_weight=getattr(args, 'score_scale_weight', 0.15),
        scale_area_weight=getattr(args, 'scale_area_weight', 0.30),
        scale_pca_weight=getattr(args, 'scale_pca_weight', 0.70),
        moment_weight=getattr(args, 'moment_weight', 0.75),
        geometry_weight=getattr(args, 'geometry_weight', 0.25),
        rotation_tolerance=getattr(args, 'rotation_tolerance', False),
    )


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


def _save_match_details(out_dir, records, query_positions, rankings, scores, arc_config, max_queries, max_candidates):
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
                explanation = score_tokens(query['tokens'], candidate['tokens'], arc_config)
                matches = _legacy_match_rows(explanation['matches'])
                for match in matches:
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


def _legacy_match_rows(matches):
    rows = []
    for match in matches:
        query = match['query_token']
        candidate = match['candidate_token']
        rows.append({
            'query_token_id': int(match['query_token_id']),
            'candidate_token_id': int(match['candidate_token_id']),
            'match_rank': int(match['rank']),
            'query_type': query.get('geometry_type', 'irregular'),
            'candidate_type': candidate.get('geometry_type', 'irregular'),
            'query_area': float(query.get('area', 0.0)),
            'candidate_area': float(candidate.get('area', 0.0)),
            'score': float(match['score']),
            'shape_sim': float(match['shape_sim']),
            'position_affinity': float(match['position_affinity']),
            'scale_affinity': float(match['scale_affinity']),
            'type_affinity': float(match['type_affinity']),
        })
    return rows


if __name__ == '__main__':
    main()
