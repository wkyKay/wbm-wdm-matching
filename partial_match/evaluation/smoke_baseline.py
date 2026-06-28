# -*- coding: utf-8 -*-
"""
Smoke Baseline Module
Implements simple baselines for retrieval evaluation:
- Global IoU baseline
- Coverage-leakage baseline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_iou(query_bin: np.ndarray, candidate_bin: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(query_bin, candidate_bin).sum()
    union = np.logical_or(query_bin, candidate_bin).sum()
    return intersection / max(union, 1)


def compute_coverage_leakage(query_bin: np.ndarray, candidate_bin: np.ndarray, beta: float = 0.5) -> float:
    """
    Compute coverage-leakage score:
    coverage = sum(min(query, candidate)) / sum(query)
    leakage = sum(candidate - min(query, candidate)) / sum(candidate)
    score = coverage - beta * leakage
    """
    query_sum = query_bin.sum()
    candidate_sum = candidate_bin.sum()
    
    if query_sum == 0 or candidate_sum == 0:
        return 0.0
    
    overlap = np.logical_and(query_bin, candidate_bin).sum()
    coverage = overlap / query_sum
    
    leakage = (candidate_sum - overlap) / candidate_sum
    score = coverage - beta * leakage
    
    return score


def generate_smoke_rankings(binary_maps: np.ndarray, 
                          sample_ids: List[int],
                          method: str = 'iou',
                          top_k: int = 100,
                          beta: float = 0.5) -> pd.DataFrame:
    """
    Generate retrieval rankings using smoke baselines.
    
    Args:
        binary_maps: Array of binary defect maps
        sample_ids: List of sample IDs
        method: 'iou' or 'coverage_leakage'
        top_k: Number of top candidates to return per query
        beta: Beta parameter for coverage-leakage
        
    Returns:
        DataFrame with rankings
    """
    N = len(binary_maps)
    rankings = []
    
    for query_idx in range(N):
        query_map = binary_maps[query_idx]
        scores = []
        
        for candidate_idx in range(N):
            if query_idx == candidate_idx:
                continue  # Skip self
            
            candidate_map = binary_maps[candidate_idx]
            
            if method == 'iou':
                score = compute_iou(query_map, candidate_map)
            elif method == 'coverage_leakage':
                score = compute_coverage_leakage(query_map, candidate_map, beta)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            scores.append({
                'query_id': sample_ids[query_idx],
                'candidate_id': sample_ids[candidate_idx],
                'similarity_score': score,
                'method': method,
            })
        
        # Sort and take top_k
        scores_sorted = sorted(scores, key=lambda x: x['similarity_score'], reverse=True)
        rankings.extend(scores_sorted[:top_k])
    
    return pd.DataFrame(rankings)
