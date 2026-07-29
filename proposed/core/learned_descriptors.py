# -*- coding: utf-8 -*-
"""Learned token descriptor extraction."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from proposed.core.cluster_patches import PatchBuilder
from proposed.core.proposal import ClusterToken, group_tokens_by_map


class TokenPatchDataset(Dataset):
    def __init__(self, records_by_map: Dict[int, dict], tokens: Iterable[ClusterToken], patch_builder: PatchBuilder):
        self.records_by_map = records_by_map
        self.tokens = list(tokens)
        self.patch_builder = patch_builder

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[int(idx)]
        raw_map = self.records_by_map[int(token.map_id)]['raw_map']
        patch = self.patch_builder.build(raw_map, token).x
        return {'idx': int(idx), 'x': torch.from_numpy(patch)}


@torch.no_grad()
def extract_learned_records(records_by_map, tokens: List[ClusterToken], patch_builder: PatchBuilder, encoder,
                            device, batch_size: int = 128, num_workers: int = 0):
    encoder = encoder.to(device)
    encoder.eval()
    dataset = TokenPatchDataset(records_by_map, tokens, patch_builder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings = [None] * len(tokens)
    non_blocking = getattr(device, 'type', str(device)) == 'cuda'
    for batch in loader:
        x = batch['x'].to(device, non_blocking=non_blocking)
        z = encoder(x).detach().cpu().numpy().astype(np.float32)
        for local_idx, emb in zip(batch['idx'].numpy().tolist(), z):
            norm = np.linalg.norm(emb)
            embeddings[int(local_idx)] = emb / max(norm, 1e-8)
    records = []
    for token, embedding in zip(tokens, embeddings):
        records.append(token_to_record(token, embedding))
    return records


def token_to_record(token: ClusterToken, embedding: np.ndarray):
    return {
        'map_id': int(token.map_id),
        'token_id': int(token.token_id),
        'embedding': embedding.astype(np.float32),
        'descriptor': embedding.astype(np.float32),
        'shape_descriptor': embedding.astype(np.float32),
        'area': float(token.area),
        'area_ratio': float(token.area_ratio),
        'support_area_ratio': float(token.area_ratio),
        'pca_lambda1': float(token.pca_lambda1),
        'pca_lambda2': float(token.pca_lambda2),
        'pos': np.asarray([
            token.centroid_row / max(token.map_height, 1),
            token.centroid_col / max(token.map_width, 1),
        ], dtype=np.float32),
        'geometry_type': token.geometry_type,
        'radial_distance_norm': float(token.radial_distance_norm),
        'angular_coverage': float(token.angular_coverage),
        'max_angular_run_coverage': float(token.max_angular_run_coverage),
        'radial_std': float(token.radial_std),
        'ring_arc_angular_coverage': float(token.ring_arc_angular_coverage),
        'ring_contour_angular_coverage': float(token.ring_contour_angular_coverage),
        'cluster': token,
    }


def group_records_by_map(records):
    out = {}
    for record in records:
        out.setdefault(int(record['map_id']), []).append(record)
    for values in out.values():
        values.sort(key=lambda item: item['token_id'])
    return out


def save_embeddings_npz(path, records):
    arrays = {}
    for record in records:
        key = f'{record["map_id"]}_{record["token_id"]}'
        arrays[key] = record['embedding'].astype(np.float32)
    np.savez_compressed(path, **arrays)


def token_rows(records):
    rows = []
    for record in records:
        token = record['cluster']
        rows.append({
            'map_id': token.map_id,
            'token_id': token.token_id,
            'area': token.area,
            'area_ratio': token.area_ratio,
            'pca_lambda1': token.pca_lambda1,
            'pca_lambda2': token.pca_lambda2,
            'centroid_row': token.centroid_row,
            'centroid_col': token.centroid_col,
            'bbox_row_min': token.bbox_row_min,
            'bbox_col_min': token.bbox_col_min,
            'bbox_row_max': token.bbox_row_max,
            'bbox_col_max': token.bbox_col_max,
            'bbox_height': token.bbox_height,
            'bbox_width': token.bbox_width,
            'geometry_type': token.geometry_type,
            'proposal_method': token.proposal_method,
            'proposal_type': token.proposal_type,
            'proposal_source': token.proposal_source,
            'proposal_signature': token.proposal_signature,
        })
    return rows
