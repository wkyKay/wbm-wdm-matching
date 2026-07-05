# -*- coding: utf-8 -*-
"""Shared per-query candidate manifest helpers for WM38K retrieval."""

import csv
from pathlib import Path
from typing import Dict, Iterable, List


CANDIDATE_COLUMNS = [
    'query_id',
    'candidate_id',
    'bucket',
    'jaccard',
    'query_signature',
    'candidate_signature',
]


def write_candidate_manifest(path, rows: Iterable[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_candidate_manifest(path) -> Dict[int, List[int]]:
    by_query = {}
    with Path(path).open('r', newline='') as f:
        for row in csv.DictReader(f):
            by_query.setdefault(int(row['query_id']), []).append(int(row['candidate_id']))
    return by_query


def candidate_manifest_ids(path) -> List[int]:
    ids = set()
    with Path(path).open('r', newline='') as f:
        for row in csv.DictReader(f):
            ids.add(int(row['query_id']))
            ids.add(int(row['candidate_id']))
    return sorted(ids)
