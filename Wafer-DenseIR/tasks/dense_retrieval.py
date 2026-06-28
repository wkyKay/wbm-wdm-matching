# -*- coding: utf-8 -*-

import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tasks.base import Task
from utils.matching import dense_match, dense_tokenize, make_heatmap, _pair_score_gpu
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
        use_pin = (self.device.type == 'cuda')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)
        records = []
        print(f'[Extract Features] Device: {self.device}, Images: {len(dataset)}, '
              f'Batch size: {batch_size}, Token mode: {token_mode}',
              f'| pin_memory={use_pin}')
        start_time = time.time()
        for batch in tqdm(loader, desc='Extracting features', unit='batch', ncols=100):
            x = batch['x'].to(self.device, non_blocking=use_pin)
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
        elapsed = time.time() - start_time
        avg_tokens = np.mean([r['tokens']['tokens'].shape[0] for r in records])
        print(f'[Extract Features] Done in {elapsed:.1f}s, '
              f'{len(records)} images, avg {avg_tokens:.1f} tokens/image')
        return records

    def run_retrieval(self, records, topk_tokens=5, sigma_pos=0.35, ks=(1, 5, 10)):
        n = len(records)
        use_gpu = (self.device.type == 'cuda')
        
        if use_gpu:
            return self._run_retrieval_gpu(records, topk_tokens, sigma_pos, ks)
        return self._run_retrieval_cpu(records, topk_tokens, sigma_pos, ks)

    def _run_retrieval_cpu(self, records, topk_tokens, sigma_pos, ks):
        n = len(records)
        print(f'[Retrieval] Computing {n}x{n} dense matches (CPU, topk_tokens={topk_tokens})...')
        scores = np.full((n, n), -np.inf, dtype=np.float32)
        match_cache = {}
        total_pairs = n * (n - 1)
        start_time = time.time()
        with tqdm(total=total_pairs, desc='Dense matching', unit='pair', ncols=100) as pbar:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    score, q_scores, c_scores, matches = dense_match(
                        records[i]['tokens'], records[j]['tokens'],
                        topk_tokens=topk_tokens, sigma_pos=sigma_pos,
                    )
                    scores[i, j] = score
                    match_cache[(i, j)] = (q_scores, c_scores, matches)
                    pbar.update(1)
        elapsed = time.time() - start_time
        print(f'[Retrieval] Done in {elapsed:.1f}s ({total_pairs/elapsed:.0f} pairs/s)')

        rankings = np.argsort(-scores, axis=1)
        labels = np.stack([r['label'] for r in records], axis=0)
        metrics = retrieval_metrics(rankings, labels, ks=ks)
        self._save_rankings(records, rankings, scores)
        self._save_metrics(metrics)
        return rankings, scores, metrics, match_cache

    def _run_retrieval_gpu(self, records, topk_tokens, sigma_pos, ks):
        n = len(records)
        print(f'[Retrieval GPU] Computing {n}x{n} dense matches '
              f'(topk_tokens={topk_tokens}, sigma_pos={sigma_pos})...')

        # Pre-load all token data to GPU
        gpu_records = []
        for r in records:
            gpu_records.append({
                'tokens': torch.from_numpy(r['tokens']['tokens']).to(self.device).float(),
                'pos': torch.from_numpy(r['tokens']['pos']).to(self.device).float(),
                'weights': torch.from_numpy(r['tokens']['weights']).to(self.device).float(),
            })

        scores = np.full((n, n), -np.inf, dtype=np.float32)
        total_pairs = n * (n - 1)
        start_time = time.time()
        
        with tqdm(total=total_pairs, desc='Dense matching (GPU)', unit='pair', ncols=100) as pbar:
            for i in range(n):
                qi = gpu_records[i]
                for j in range(n):
                    if i == j:
                        continue
                    score = _pair_score_gpu(
                        qi['tokens'], gpu_records[j]['tokens'],
                        qi['pos'], gpu_records[j]['pos'],
                        qi['weights'], topk_tokens, sigma_pos,
                    )
                    scores[i, j] = score
                    pbar.update(1)
                    
                    # Free GPU memory of last used tensors periodically
                    if j % 100 == 0:
                        torch.cuda.synchronize()

        elapsed = time.time() - start_time
        print(f'[Retrieval GPU] Done in {elapsed:.1f}s ({total_pairs/elapsed:.0f} pairs/s)')

        rankings = np.argsort(-scores, axis=1)
        labels = np.stack([r['label'] for r in records], axis=0)
        metrics = retrieval_metrics(rankings, labels, ks=ks)
        self._save_rankings(records, rankings, scores)
        self._save_metrics(metrics)
        
        # Free GPU records to reclaim memory
        gpu_records.clear()
        torch.cuda.empty_cache()
        
        return rankings, scores, metrics, {}

    def save_explanations(self, records, rankings, scores, match_cache, num_queries=8):
        explain_dir = os.path.join(self.output_dir, 'explanations')
        os.makedirs(explain_dir, exist_ok=True)
        for q in range(min(num_queries, len(records))):
            c = int(rankings[q, 0])
            key = (q, c)
            if key in match_cache:
                q_scores, c_scores, matches = match_cache[key]
            else:
                # Recompute match data for this pair (GPU path)
                _, q_scores, c_scores, matches = dense_match(
                    records[q]['tokens'], records[c]['tokens'],
                )
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

