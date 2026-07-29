# -*- coding: utf-8 -*-
"""Cluster-level contrastive dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from proposed.core.cluster_patches import PatchBuilder, augment_patch
from proposed.core.proposal import ClusterToken, ProposalProvider, load_tokens_csv, save_tokens_csv
from proposed.datasets.wm38k_maps import records_by_id


class ClusterContrastiveDataset(Dataset):
    def __init__(self, records, proposal_provider: ProposalProvider, patch_builder: PatchBuilder,
                 max_clusters: int = None, seed: int = 2026, tokens_csv: str = None, progress_desc: str = None):
        self.records = list(records)
        self.record_by_id = records_by_id(self.records)
        self.patch_builder = patch_builder
        self.seed = int(seed)
        if tokens_csv and Path(tokens_csv).exists():
            if progress_desc:
                print(f'{progress_desc}: loading cached tokens from {tokens_csv}', flush=True)
            tokens = load_tokens_csv(tokens_csv)
        else:
            tokens = []
            iterator = self.records
            if progress_desc:
                iterator = tqdm(self.records, desc=progress_desc)
            for record in iterator:
                tokens.extend(proposal_provider.extract(record['map_id'], record['raw_map']))
            if tokens_csv:
                if progress_desc:
                    print(f'{progress_desc}: saving {len(tokens)} tokens to {tokens_csv}', flush=True)
                save_tokens_csv(tokens_csv, tokens)
        tokens = [token for token in tokens if int(token.map_id) in self.record_by_id]
        tokens.sort(key=lambda item: (item.map_id, item.token_id))
        if max_clusters is not None and len(tokens) > max_clusters:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(tokens), size=int(max_clusters), replace=False))
            tokens = [tokens[int(i)] for i in idx]
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[int(idx)]
        raw_map = self.record_by_id[int(token.map_id)]['raw_map']
        patch = self.patch_builder.build(raw_map, token).x
        rng = np.random.default_rng(self.seed + int(idx) * 1009)
        x = augment_patch(patch, rng)
        x_t = augment_patch(patch, rng)
        return {
            'idx': torch.tensor(int(idx), dtype=torch.long),
            'x': torch.from_numpy(x),
            'x_t': torch.from_numpy(x_t),
            'map_id': torch.tensor(int(token.map_id), dtype=torch.long),
            'token_id': torch.tensor(int(token.token_id), dtype=torch.long),
        }


def write_patch_manifest(path, tokens, patch_config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['map_id', 'token_id', 'proposal_signature', 'patch_size'])
        writer.writeheader()
        for token in tokens:
            writer.writerow({
                'map_id': token.map_id,
                'token_id': token.token_id,
                'proposal_signature': token.proposal_signature,
                'patch_size': patch_config.patch_size,
            })
