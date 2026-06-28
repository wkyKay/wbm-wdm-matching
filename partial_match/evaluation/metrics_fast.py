# -*- coding: utf-8 -*-
"""
Fast Evaluation Metrics Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import json


def compute_label_overlap(query_labels: List[int], candidate_labels: List[int]) -> float:
    """
    Compute Jaccard overlap between query and candidate label sets.
    """
    q_set = set(query_labels)
    c_set = set(candidate_labels)
    intersection = q_set & c_set
    union = q_set | c_set
    return len(intersection) / max(len(union), 1)


def compute_relevance(query_labels: List[int], candidate_labels: List[int]) -> int:
    """
    Compute relevance grade:
    - 2 for exact match
    - 1 for partial overlap
    - 0 for no overlap
    """
    q_set = set(query_labels)
    c_set = set(candidate_labels)
    
    if q_set == c_set:
        return 2
    elif q_set & c_set:
        return 1
    else:
        return 0


def dcg(relevance_scores: List[float], k: int = None) -> float:
    """
    Compute Discounted Cumulative Gain (DCG) at k.
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    dcg_val = 0.0
    for i, rel in enumerate(relevance_scores, 1):
        dcg_val += rel / np.log2(i + 1)
    return dcg_val


def ndcg(relevance_scores: List[float], k: int = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k.
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    ideal_relevance = sorted(relevance_scores, reverse=True)
    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(ideal_relevance, k)
    
    return actual_dcg / max(ideal_dcg, 1e-10)


def mean_reciprocal_rank(relevance_list: List[List[int]]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    """
    mrr = 0.0
    count = 0
    
    for rels in relevance_list:
        for i, rel in enumerate(rels, 1):
            if rel == 2:  # Exact match
                mrr += 1.0 / i
                count += 1
                break
    
    return mrr / max(count, 1)


def average_precision(relevance_scores: List[int]) -> float:
    """
    Compute Average Precision for a single query.
    """
    ap = 0.0
    num_relevant = 0
    
    for i, rel in enumerate(relevance_scores, 1):
        if rel == 2:  # Exact match
            num_relevant += 1
            ap += num_relevant / i
    
    return ap / max(num_relevant, 1)


def evaluate_rankings_fast(rankings_df: pd.DataFrame, 
                          metadata_df: pd.DataFrame,
                          k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Fast evaluation of retrieval rankings.
    Preprocesses metadata first for speed.
    """
    # Preprocess metadata into a list of label sets for faster lookup
    label_sets = []
    for _, row in metadata_df.iterrows():
        label_set = eval(row['label_set']) if isinstance(row['label_set'], str) else row['label_set']
        label_sets.append(label_set)
    
    # Precompute total relevant for each query
    total_relevant_dict = {}
    for query_id in rankings_df['query_id'].unique():
        q_labels = label_sets[query_id]
        total = 0
        for cand_id in range(len(label_sets)):
            if cand_id == query_id:
                continue
            if compute_relevance(q_labels, label_sets[cand_id]) == 2:
                total += 1
        total_relevant_dict[query_id] = total
    
    # Group rankings by query
    query_groups = rankings_df.groupby('query_id')
    
    query_metrics = defaultdict(list)
    skipped_exact_queries = 0
    
    for query_id, group in query_groups:
        q_labels = label_sets[query_id]
        
        # Sort group by similarity score (descending)
        sorted_group = group.sort_values('similarity_score', ascending=False)
        
        # Compute relevance scores for candidates
        relevance_scores = []
        label_overlaps = []
        
        for _, row in sorted_group.iterrows():
            cand_id = int(row['candidate_id'])
            c_labels = label_sets[cand_id]
            
            rel = compute_relevance(q_labels, c_labels)
            relevance_scores.append(rel)
            
            overlap = compute_label_overlap(q_labels, c_labels)
            label_overlaps.append(overlap)
        
        # Compute metrics for this query
        total_relevant = total_relevant_dict[query_id]
        
        for k in k_values:
            # Exact consistency @k
            exact_consistency = int(2 in relevance_scores[:k])
            query_metrics[f'exact_consistency@{k}'].append(exact_consistency)
            
            # Partial consistency @k
            partial_consistency = int(any(r >= 1 for r in relevance_scores[:k]))
            query_metrics[f'partial_consistency@{k}'].append(partial_consistency)
            
            # Mean label overlap @k
            mean_overlap = np.mean(label_overlaps[:k]) if label_overlaps[:k] else 0
            query_metrics[f'mean_label_overlap@{k}'].append(mean_overlap)
            
            # NDCG @k
            ndcg_val = ndcg(relevance_scores, k)
            query_metrics[f'ndcg@{k}'].append(ndcg_val)
            
            # Recall @k
            if total_relevant > 0:
                relevant_at_k = sum(1 for rel in relevance_scores[:k] if rel == 2)
                recall_val = relevant_at_k / total_relevant
            else:
                recall_val = 0.0
            query_metrics[f'recall@{k}'].append(recall_val)
        
        # MRR and mAP (uses full ranking)
        query_metrics['relevance_list'].append(relevance_scores)
        
        if total_relevant > 0:
            ap = average_precision(relevance_scores)
            query_metrics['average_precision'].append(ap)
        else:
            skipped_exact_queries += 1
    
    # Aggregate micro average metrics
    metrics = {
        'total_queries': len(query_groups),
        'skipped_exact_queries': skipped_exact_queries,
        'micro_average': {},
        'macro_average_per_class': {},
    }
    
    for k in k_values:
        metrics['micro_average'][f'exact_consistency@{k}'] = np.mean(query_metrics[f'exact_consistency@{k}'])
        metrics['micro_average'][f'partial_consistency@{k}'] = np.mean(query_metrics[f'partial_consistency@{k}'])
        metrics['micro_average'][f'mean_label_overlap@{k}'] = np.mean(query_metrics[f'mean_label_overlap@{k}'])
        metrics['micro_average'][f'ndcg@{k}'] = np.mean(query_metrics[f'ndcg@{k}'])
        metrics['micro_average'][f'recall@{k}'] = np.mean(query_metrics[f'recall@{k}'])
    
    metrics['micro_average']['mrr'] = mean_reciprocal_rank(query_metrics['relevance_list'])
    metrics['micro_average']['map'] = np.mean(query_metrics['average_precision']) if query_metrics['average_precision'] else 0.0
    
    return metrics
