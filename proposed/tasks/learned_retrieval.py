# -*- coding: utf-8 -*-
"""Learned local retrieval over fixed query/candidate manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.evaluate_rankings import evaluate_rankings_from_files, write_flat_metrics
from proposed.core.cluster_patches import PatchBuilder
from proposed.core.learned_descriptors import extract_learned_records, group_records_by_map, save_embeddings_npz, token_rows
from proposed.core.matching import map_similarity
from proposed.core.proposal import save_tokens_csv
from shared.wm38k.candidates import load_candidate_manifest
from shared.wm38k.manifest import load_query_ids


def run_learned_retrieval(records, tokens, patch_builder: PatchBuilder, encoder, device, out_dir,
                          split_manifest, query_manifest, candidate_manifest=None, split='test',
                          batch_size=128, num_workers=0, topk_match=1, sigma_pos=0.35, sigma_area=1.0,
                          metric_k=(1, 5, 10)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_by_map = {int(record['map_id']): record for record in records}
    tokens = [token for token in tokens if int(token.map_id) in records_by_map]
    save_tokens_csv(out_dir / 'proposal_tokens.csv', tokens)
    learned_records = extract_learned_records(records_by_map, tokens, patch_builder, encoder, device,
                                              batch_size=batch_size, num_workers=num_workers)
    grouped = group_records_by_map(learned_records)
    save_embeddings_npz(out_dir / 'embeddings.npz', learned_records)
    _write_token_rows(out_dir / 'tokens.csv', token_rows(learned_records))

    query_ids = load_query_ids(query_manifest)
    candidate_ids_by_query = load_candidate_manifest(candidate_manifest) if candidate_manifest else None
    all_ids = sorted(grouped)
    rankings_path = out_dir / 'rankings.csv'
    with rankings_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'rank', 'candidate_id', 'similarity_score'])
        for n, query_id in enumerate(query_ids, start=1):
            if int(query_id) not in grouped:
                continue
            if candidate_ids_by_query:
                candidate_ids = [cid for cid in candidate_ids_by_query.get(int(query_id), []) if cid in grouped and cid != int(query_id)]
            else:
                candidate_ids = [cid for cid in all_ids if cid != int(query_id)]
            scored = []
            for candidate_id in candidate_ids:
                score = map_similarity(
                    grouped[int(query_id)],
                    grouped[int(candidate_id)],
                    sigma_pos=sigma_pos,
                    sigma_area=sigma_area,
                    topk=topk_match,
                )
                scored.append((int(candidate_id), float(score)))
            scored.sort(key=lambda item: (-item[1], item[0]))
            for rank, (candidate_id, score) in enumerate(scored, start=1):
                writer.writerow([int(query_id), rank, candidate_id, score])
            if n % 50 == 0:
                print(f'Scored {n}/{len(query_ids)} queries')

    metrics = evaluate_rankings_from_files(
        rankings_path=str(rankings_path),
        split_manifest=split_manifest,
        query_manifest=query_manifest,
        candidate_manifest=candidate_manifest,
        split=split,
        ks=tuple(metric_k),
    )
    metrics_path = out_dir / 'label_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    write_flat_metrics(metrics, out_dir / 'label_metrics_flat.csv')
    return {
        'rankings_path': rankings_path,
        'metrics_path': metrics_path,
        'metrics': metrics,
    }


def _write_token_rows(path, rows):
    fieldnames = list(rows[0].keys()) if rows else ['map_id', 'token_id']
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

