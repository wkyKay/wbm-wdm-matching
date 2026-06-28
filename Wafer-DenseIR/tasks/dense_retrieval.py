# -*- coding: utf-8 -*-

import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from tasks.base import Task
from utils.matching import dense_match, dense_tokenize, make_heatmap
from utils.metrics import retrieval_metrics
from utils.visualization import save_retrieval_explanation


class DenseRetrieval(Task):
    def __init__(self, backbone, device, output_dir):
        super(DenseRetrieval, self).__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.output_dir = output_dir

    @torch.no_grad()
    def extract_features(self, dataset, batch_size=128, num_workers=0, token_mode='defect_band',
                         token_dilation=1, max_tokens=256):
        self.backbone.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        records = []
        for batch in loader:
            x = batch['x'].to(self.device)
            fmap = self.backbone(x).detach().cpu()
            for i in range(fmap.shape[0]):
                tokens = dense_tokenize(
                    fmap[i],
                    batch['defect_mask'][i],
                    batch['valid_mask'][i],
                    token_mode=token_mode,
                    token_dilation=token_dilation,
                    max_tokens=max_tokens,
                )
                records.append({
                    'idx': int(batch['idx'][i]),
                    'label': batch['y'][i].numpy().astype(np.float32),
                    'raw': batch['raw'][i].numpy().astype(np.float32),
                    'tokens': tokens,
                })
        return records

    def run_retrieval(self, records, topk_tokens=5, sigma_pos=0.35, ks=(1, 5, 10)):
        n = len(records)
        scores = np.full((n, n), -np.inf, dtype=np.float32)
        match_cache = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                score, q_scores, c_scores, matches = dense_match(
                    records[i]['tokens'],
                    records[j]['tokens'],
                    topk_tokens=topk_tokens,
                    sigma_pos=sigma_pos,
                )
                scores[i, j] = score
                match_cache[(i, j)] = (q_scores, c_scores, matches)

        rankings = np.argsort(-scores, axis=1)
        labels = np.stack([r['label'] for r in records], axis=0)
        metrics = retrieval_metrics(rankings, labels, ks=ks)
        self._save_rankings(records, rankings, scores)
        self._save_metrics(metrics)
        return rankings, scores, metrics, match_cache

    def save_explanations(self, records, rankings, scores, match_cache, num_queries=8):
        explain_dir = os.path.join(self.output_dir, 'explanations')
        os.makedirs(explain_dir, exist_ok=True)
        for q in range(min(num_queries, len(records))):
            c = int(rankings[q, 0])
            q_scores, c_scores, matches = match_cache[(q, c)]
            q_heat = make_heatmap(q_scores, records[q]['tokens']['pos'], records[q]['tokens']['grid_size'])
            c_heat = make_heatmap(c_scores, records[c]['tokens']['pos'], records[c]['tokens']['grid_size'])
            out = os.path.join(explain_dir, f'query_{records[q]["idx"]}_top1_{records[c]["idx"]}.png')
            save_retrieval_explanation(
                out,
                records[q]['raw'],
                records[c]['raw'],
                q_heat,
                c_heat,
                records[q]['idx'],
                records[c]['idx'],
                float(scores[q, c]),
                matches=matches,
            )

    def save_features(self, records, path=None):
        path = path or os.path.join(self.output_dir, 'dense_features.npz')
        arrays = {}
        for i, rec in enumerate(records):
            arrays[f'idx_{i}'] = np.array(rec['idx'], dtype=np.int64)
            arrays[f'label_{i}'] = rec['label']
            arrays[f'tokens_{i}'] = rec['tokens']['tokens']
            arrays[f'pos_{i}'] = rec['tokens']['pos']
            arrays[f'weights_{i}'] = rec['tokens']['weights']
        np.savez_compressed(path, **arrays)

    def _save_rankings(self, records, rankings, scores):
        path = os.path.join(self.output_dir, 'rankings.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['query_id', 'rank', 'candidate_id', 'similarity_score'])
            for i, ranked in enumerate(rankings):
                for rank, j in enumerate(ranked, start=1):
                    if i == j:
                        continue
                    writer.writerow([records[i]['idx'], rank, records[j]['idx'], float(scores[i, j])])

    def _save_metrics(self, metrics):
        path = os.path.join(self.output_dir, 'metrics.json')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

