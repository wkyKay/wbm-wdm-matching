# -*- coding: utf-8 -*-
"""
Evaluation Metrics Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


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


def dcg(relevance_scores: List[float], k: Optional[int] = None) -> float:
    """
    Compute Discounted Cumulative Gain (DCG) at k.
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    dcg_val = 0.0
    for i, rel in enumerate(relevance_scores, 1):
        dcg_val += rel / np.log2(i + 1)
    return dcg_val


def ndcg(relevance_scores: List[float], k: Optional[int] = None) -> float:
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
    Relevance_list contains for each query the list of relevance (1 for relevant, 0 for not).
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


def recall_at_k(relevance_scores: List[int], k: int, total_relevant: int) -> float:
    """
    Compute Recall@k.
    """
    if total_relevant == 0:
        return 0.0
    relevant_at_k = sum(1 for rel in relevance_scores[:k] if rel == 2)
    return relevant_at_k / total_relevant


def evaluate_rankings(rankings_df: pd.DataFrame, 
                      metadata_df: pd.DataFrame,
                      k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Evaluate retrieval rankings.
    
    Args:
        rankings_df: DataFrame with columns ['query_id', 'candidate_id', 'similarity_score']
        metadata_df: Metadata DataFrame with label information
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Group rankings by query
    query_groups = rankings_df.groupby('query_id')
    
    query_metrics = defaultdict(list)
    per_class_metrics = defaultdict(lambda: defaultdict(list))
    skipped_exact_queries = 0
    
    class_names = ['center', 'donut', 'edge-loc', 'edge-ring', 
                   'loc', 'random', 'scratch', 'near-full']
    
    for query_id, group in query_groups:
        # Get query label information
        query_meta = metadata_df.iloc[query_id]
        query_label_set = eval(query_meta['label_set']) if isinstance(query_meta['label_set'], str) else query_meta['label_set']
        
        # Get primary class for per-class metrics
        primary_class = None
        for cn in class_names:
            if query_meta[f'label_{cn}']:
                primary_class = cn
                break
        
        # Compute relevance scores for candidates
        relevance_scores = []
        label_overlaps = []
        exact_match_found = False
        partial_match_found = False
        
        # Sort group by similarity score (descending)
        sorted_group = group.sort_values('similarity_score', ascending=False)
        
        for _, row in sorted_group.iterrows():
            candidate_id = int(row['candidate_id'])
            candidate_meta = metadata_df.iloc[candidate_id]
            candidate_label_set = eval(candidate_meta['label_set']) if isinstance(candidate_meta['label_set'], str) else candidate_meta['label_set']
            
            rel = compute_relevance(query_label_set, candidate_label_set)
            relevance_scores.append(rel)
            
            overlap = compute_label_overlap(query_label_set, candidate_label_set)
            label_overlaps.append(overlap)
            
            if rel == 2:
                exact_match_found = True
            if rel >= 1:
                partial_match_found = True
        
        # Compute total relevant candidates in the full candidate pool
        total_relevant = 0
        for cand_id in range(len(metadata_df)):
            if cand_id == query_id:
                continue
            cand_meta = metadata_df.iloc[cand_id]
            cand_label_set = eval(cand_meta['label_set']) if isinstance(cand_meta['label_set'], str) else cand_meta['label_set']
            if compute_relevance(query_label_set, cand_label_set) == 2:
                total_relevant += 1
        
        # Compute metrics for this query
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
            recall_val = recall_at_k(relevance_scores, k, total_relevant)
            query_metrics[f'recall@{k}'].append(recall_val)
            
            # Per-class metrics
            if primary_class:
                per_class_metrics[primary_class][f'exact_consistency@{k}'].append(exact_consistency)
                per_class_metrics[primary_class][f'partial_consistency@{k}'].append(partial_consistency)
                per_class_metrics[primary_class][f'ndcg@{k}'].append(ndcg_val)
        
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
    
    # Aggregate macro average per class
    for class_name in class_names:
        if class_name not in per_class_metrics:
            continue
        
        class_metrics = per_class_metrics[class_name]
        metrics['macro_average_per_class'][class_name] = {}
        
        for k in k_values:
            if f'exact_consistency@{k}' in class_metrics and class_metrics[f'exact_consistency@{k}']:
                metrics['macro_average_per_class'][class_name][f'exact_consistency@{k}'] = np.mean(class_metrics[f'exact_consistency@{k}'])
                metrics['macro_average_per_class'][class_name][f'partial_consistency@{k}'] = np.mean(class_metrics[f'partial_consistency@{k}'])
                metrics['macro_average_per_class'][class_name][f'ndcg@{k}'] = np.mean(class_metrics[f'ndcg@{k}'])
    
    return metrics
