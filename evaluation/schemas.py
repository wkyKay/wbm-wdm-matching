# -*- coding: utf-8 -*-
"""Input schema checks for method-independent retrieval evaluation."""

REQUIRED_RANKING_COLUMNS = ('query_id', 'candidate_id', 'similarity_score')
OPTIONAL_RANKING_COLUMNS = ('rank',)


def validate_ranking_columns(fieldnames):
    fieldnames = set(fieldnames or [])
    missing = [name for name in REQUIRED_RANKING_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError(f'Ranking file is missing required columns: {missing}')

