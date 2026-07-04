# -*- coding: utf-8 -*-
"""Check that proposed and partial_match token metadata use the same proposal output."""

import argparse
import csv
from pathlib import Path


def main():
    args = parse_args()
    left = _load(args.partial_tokens, key_fields=('map_id', 'token_id'))
    right = _load(args.proposed_tokens, key_fields=('map_id', 'token_id'))
    missing = sorted(set(left) - set(right))[:10]
    extra = sorted(set(right) - set(left))[:10]
    mismatches = []
    fields = ['area', 'centroid_row', 'centroid_col', 'bbox_row_min', 'bbox_row_max', 'bbox_col_min', 'bbox_col_max', 'geometry_type']
    for key in sorted(set(left) & set(right)):
        for field in fields:
            if not _same(left[key].get(field, ''), right[key].get(field, '')):
                mismatches.append((key, field, left[key].get(field), right[key].get(field)))
                break
        if len(mismatches) >= 10:
            break
    if missing or extra or mismatches:
        print(f'Missing in proposed: {missing}')
        print(f'Extra in proposed: {extra}')
        print(f'Mismatches: {mismatches}')
        raise SystemExit(1)
    print(f'Proposal token metadata match for {len(left)} tokens.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial-tokens', required=True)
    parser.add_argument('--proposed-tokens', required=True)
    return parser.parse_args()


def _load(path, key_fields):
    out = {}
    with Path(path).open('r', newline='') as f:
        for row in csv.DictReader(f):
            key = tuple(int(row[field]) for field in key_fields)
            out[key] = row
    return out


def _same(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-6
    except (TypeError, ValueError):
        return str(a) == str(b)


if __name__ == '__main__':
    main()

