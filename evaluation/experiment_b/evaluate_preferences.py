# -*- coding: utf-8 -*-
"""Evaluate Experiment B preference accuracy from method rankings."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.experiment_b.preference_metrics import evaluate_preferences, load_preferences, load_scores


def main():
    args = parse_args()
    metrics = evaluate_preferences_from_files(args.scores, args.preferences)
    out_path = Path(args.out) if args.out else Path(args.scores).with_name('preference_metrics.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    details = metrics.pop('details')
    out_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    if args.details_out:
        write_details(details, args.details_out)
    elif args.save_details:
        write_details(details, out_path.with_name('preference_details.csv'))
    print(json.dumps(metrics, indent=2))
    print(f'Saved preference metrics to {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Experiment B preference accuracy.')
    parser.add_argument('--scores', type=str, required=True, help='CSV with query_id,candidate_id,similarity_score.')
    parser.add_argument('--preferences', type=str, required=True, help='Experiment B b_preferences.csv.')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--details-out', type=str, default=None)
    parser.add_argument('--save-details', action='store_true')
    return parser.parse_args()


def evaluate_preferences_from_files(scores_path, preferences_path):
    return evaluate_preferences(load_scores(scores_path), load_preferences(preferences_path))


def write_details(details, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'query_id',
        'preferred_candidate_id',
        'less_preferred_candidate_id',
        'rule_type',
        'rule_group',
        'preferred_score',
        'less_preferred_score',
        'credit',
        'outcome',
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)


if __name__ == '__main__':
    main()
