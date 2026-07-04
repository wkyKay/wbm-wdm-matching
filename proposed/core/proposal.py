# -*- coding: utf-8 -*-
"""Proposal adapter and stable token schema for the proposed method."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from partial_match.core.clustering import cluster


@dataclass(frozen=True)
class ProposalConfig:
    method: str = 'retrieval_compact'
    min_area: int = 5
    top_k: int = 6
    enable_ring_aware: bool = True
    max_defect_ratio_for_ring: float = 0.45
    min_edge_defect_fraction_for_ring: float = 0.45


@dataclass(frozen=True)
class ClusterToken:
    map_id: int
    token_id: int
    pixels: Tuple[Tuple[int, int], ...]
    area: float
    area_ratio: float
    centroid_row: float
    centroid_col: float
    bbox_row_min: int
    bbox_row_max: int
    bbox_col_min: int
    bbox_col_max: int
    bbox_height: int
    bbox_width: int
    geometry_type: str
    proposal_method: str
    proposal_type: str
    proposal_source: str
    proposal_signature: str

    @property
    def pos(self) -> np.ndarray:
        return np.asarray([self.centroid_row, self.centroid_col], dtype=np.float32)


class ProposalProvider:
    def extract(self, map_id: int, raw_map: np.ndarray) -> List[ClusterToken]:
        raise NotImplementedError


class PartialMatchProposalProvider(ProposalProvider):
    """Adapter around partial_match proposal code.

    This is the only proposed-method module that should call
    partial_match.core.clustering.cluster directly.
    """

    def __init__(self, config: ProposalConfig):
        self.config = config

    def extract(self, map_id: int, raw_map: np.ndarray) -> List[ClusterToken]:
        defect_mask = raw_map == 2
        valid_mask = (raw_map == 1) | (raw_map == 2)
        clusters = cluster(
            defect_mask,
            valid_mask,
            method=self.config.method,
            min_area=self.config.min_area,
            top_k=self.config.top_k,
            enable_ring_aware=self.config.enable_ring_aware,
            max_defect_ratio_for_ring=self.config.max_defect_ratio_for_ring,
            min_edge_defect_fraction_for_ring=self.config.min_edge_defect_fraction_for_ring,
        )
        h, w = raw_map.shape
        return [cluster_to_token(map_id, idx, item, (h, w), self.config.method) for idx, item in enumerate(clusters)]


def cluster_to_token(map_id: int, token_id: int, item: dict, map_shape, proposal_method: str) -> ClusterToken:
    pixels = _normalize_pixels(item)
    h, w = map_shape
    area = float(item.get('area', len(pixels)))
    signature = proposal_signature(
        map_id=map_id,
        token_id=token_id,
        pixels=pixels,
        proposal_method=proposal_method,
        proposal_type=item.get('proposal_type', ''),
        proposal_source=item.get('proposal_source', ''),
    )
    return ClusterToken(
        map_id=int(map_id),
        token_id=int(token_id),
        pixels=tuple(pixels),
        area=area,
        area_ratio=area / max(float(h * w), 1.0),
        centroid_row=float(item.get('centroid_row', 0.0)),
        centroid_col=float(item.get('centroid_col', 0.0)),
        bbox_row_min=int(item.get('bbox_row_min', 0)),
        bbox_row_max=int(item.get('bbox_row_max', 0)),
        bbox_col_min=int(item.get('bbox_col_min', 0)),
        bbox_col_max=int(item.get('bbox_col_max', 0)),
        bbox_height=int(item.get('bbox_height', 0)),
        bbox_width=int(item.get('bbox_width', 0)),
        geometry_type=str(item.get('geometry_type', 'irregular')),
        proposal_method=str(proposal_method),
        proposal_type=str(item.get('proposal_type', '')),
        proposal_source=str(item.get('proposal_source', '')),
        proposal_signature=signature,
    )


def proposal_signature(map_id: int, token_id: int, pixels: Sequence[Tuple[int, int]], proposal_method: str,
                       proposal_type: str = '', proposal_source: str = '') -> str:
    payload = {
        'map_id': int(map_id),
        'token_id': int(token_id),
        'proposal_method': proposal_method,
        'proposal_type': proposal_type,
        'proposal_source': proposal_source,
        'pixels': sorted((int(r), int(c)) for r, c in pixels),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha1(encoded).hexdigest()


def save_proposal_config(path, config: ProposalConfig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')


def save_tokens_csv(path, tokens: Iterable[ClusterToken]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [token_to_row(token) for token in tokens]
    fieldnames = list(rows[0].keys()) if rows else _token_columns()
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_tokens_csv(path) -> List[ClusterToken]:
    with Path(path).open('r', newline='') as f:
        return [token_from_row(row) for row in csv.DictReader(f)]


def token_to_row(token: ClusterToken) -> dict:
    row = asdict(token)
    row['pixels'] = json.dumps([[int(r), int(c)] for r, c in token.pixels], separators=(',', ':'))
    return row


def token_from_row(row: dict) -> ClusterToken:
    pixels = tuple((int(r), int(c)) for r, c in json.loads(row['pixels']))
    return ClusterToken(
        map_id=int(row['map_id']),
        token_id=int(row['token_id']),
        pixels=pixels,
        area=float(row['area']),
        area_ratio=float(row['area_ratio']),
        centroid_row=float(row['centroid_row']),
        centroid_col=float(row['centroid_col']),
        bbox_row_min=int(row['bbox_row_min']),
        bbox_row_max=int(row['bbox_row_max']),
        bbox_col_min=int(row['bbox_col_min']),
        bbox_col_max=int(row['bbox_col_max']),
        bbox_height=int(row['bbox_height']),
        bbox_width=int(row['bbox_width']),
        geometry_type=row['geometry_type'],
        proposal_method=row['proposal_method'],
        proposal_type=row.get('proposal_type', ''),
        proposal_source=row.get('proposal_source', ''),
        proposal_signature=row['proposal_signature'],
    )


def group_tokens_by_map(tokens: Iterable[ClusterToken]) -> dict:
    out = {}
    for token in tokens:
        out.setdefault(int(token.map_id), []).append(token)
    for values in out.values():
        values.sort(key=lambda item: item.token_id)
    return out


def _normalize_pixels(item: dict) -> Tuple[Tuple[int, int], ...]:
    coords = item.get('pixels', item.get('pixel_coords', []))
    pixels = []
    for coord in coords:
        if isinstance(coord, dict):
            pixels.append((int(coord['row']), int(coord['col'])))
        else:
            pixels.append((int(coord[0]), int(coord[1])))
    return tuple(sorted(pixels))


def _token_columns():
    return [
        'map_id', 'token_id', 'pixels', 'area', 'area_ratio', 'centroid_row', 'centroid_col',
        'bbox_row_min', 'bbox_row_max', 'bbox_col_min', 'bbox_col_max', 'bbox_height', 'bbox_width',
        'geometry_type', 'proposal_method', 'proposal_type', 'proposal_source', 'proposal_signature',
    ]

