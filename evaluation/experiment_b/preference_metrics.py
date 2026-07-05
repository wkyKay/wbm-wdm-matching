# -*- coding: utf-8 -*-
"""Preference accuracy metrics for Experiment B."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_scores(path):
    scores = {}
    with Path(path).open('r', newline='') as f:
        for row in csv.DictReader(f):
            scores[(int(row['query_id']), int(row['candidate_id']))] = float(row['similarity_score'])
    return scores


def load_preferences(path):
    with Path(path).open('r', newline='') as f:
        return [
            {
                'query_id': int(row['query_id']),
                'preferred_candidate_id': int(row['preferred_candidate_id']),
                'less_preferred_candidate_id': int(row['less_preferred_candidate_id']),
                'rule_type': row['rule_type'],
                'rule_group': row['rule_group'],
            }
            for row in csv.DictReader(f)
        ]


def evaluate_preferences(scores, preferences):
    overall = _Accumulator()
    by_group = defaultdict(_Accumulator)
    by_rule = defaultdict(_Accumulator)
    missing = []
    details = []
    for pref in preferences:
        query_id = pref['query_id']
        preferred_id = pref['preferred_candidate_id']
        less_id = pref['less_preferred_candidate_id']
        preferred_score = scores.get((query_id, preferred_id))
        less_score = scores.get((query_id, less_id))
        if preferred_score is None or less_score is None:
            missing.append(pref)
            details.append(_detail_row(pref, preferred_score, less_score, None, 'missing'))
            continue
        if preferred_score > less_score:
            credit = 1.0
            outcome = 'correct'
        elif preferred_score == less_score:
            credit = 0.5
            outcome = 'tie'
        else:
            credit = 0.0
            outcome = 'incorrect'
        overall.add(credit, outcome)
        by_group[pref['rule_group']].add(credit, outcome)
        by_rule[pref['rule_type']].add(credit, outcome)
        details.append(_detail_row(pref, preferred_score, less_score, credit, outcome))
    return {
        'protocol': {
            'metric_group': 'B.transformation_derived_preference',
            'tie_credit': 0.5,
        },
        'counts': {
            'num_preferences': len(preferences),
            'num_evaluated_preferences': overall.count,
            'num_missing_preferences': len(missing),
        },
        'overall': overall.summary(),
        'by_group': {key: acc.summary() for key, acc in sorted(by_group.items())},
        'by_rule': {key: acc.summary() for key, acc in sorted(by_rule.items())},
        'details': details,
    }


class _Accumulator:
    def __init__(self):
        self.values = []
        self.correct = 0
        self.tie = 0
        self.incorrect = 0

    @property
    def count(self):
        return len(self.values)

    def add(self, credit, outcome):
        self.values.append(float(credit))
        if outcome == 'correct':
            self.correct += 1
        elif outcome == 'tie':
            self.tie += 1
        elif outcome == 'incorrect':
            self.incorrect += 1

    def summary(self):
        count = self.count
        return {
            'preference_accuracy': float(np.mean(self.values)) if self.values else None,
            'count': count,
            'correct': self.correct,
            'tie': self.tie,
            'incorrect': self.incorrect,
            'tie_rate': float(self.tie / count) if count else None,
        }


def _detail_row(pref, preferred_score, less_score, credit, outcome):
    row = dict(pref)
    row.update({
        'preferred_score': preferred_score,
        'less_preferred_score': less_score,
        'credit': credit,
        'outcome': outcome,
    })
    return row
