# -*- coding: utf-8 -*-
"""Handcrafted descriptors for proposal-based local retrieval."""

from typing import Dict, List

import numpy as np


GEOMETRY_TYPES = ['blob', 'line', 'edge_ring', 'central', 'irregular']


def cluster_descriptor(cluster: Dict, map_shape) -> np.ndarray:
    H, W = map_shape
    area = float(cluster.get('area', 0))
    bbox_h = float(cluster.get('bbox_height', 1))
    bbox_w = float(cluster.get('bbox_width', 1))
    bbox_area = max(bbox_h * bbox_w, 1.0)
    fill_ratio = area / bbox_area
    aspect = max(bbox_h / max(bbox_w, 1.0), bbox_w / max(bbox_h, 1.0))
    elongation = cluster.get('pca_lambda1', 0.0) / max(cluster.get('pca_lambda2', 0.0), 1e-6)
    compactness = float(cluster.get('compactness', 0.0))
    orientation = float(cluster.get('orientation', 0.0))
    radial = float(cluster.get('radial_distance_norm', 0.0))
    centroid_r = float(cluster.get('centroid_row', 0.0)) / max(H, 1)
    centroid_c = float(cluster.get('centroid_col', 0.0)) / max(W, 1)

    type_onehot = np.zeros(len(GEOMETRY_TYPES), dtype=np.float32)
    geometry_type = cluster.get('geometry_type', 'irregular')
    if geometry_type in GEOMETRY_TYPES:
        type_onehot[GEOMETRY_TYPES.index(geometry_type)] = 1.0

    profile = _shape_profiles(cluster, map_shape, bins=8)
    features = np.array([
        np.log1p(area) / np.log1p(max(H * W, 1)),
        bbox_h / max(H, 1),
        bbox_w / max(W, 1),
        fill_ratio,
        np.log1p(aspect) / np.log(16.0),
        np.log1p(elongation) / np.log(64.0),
        min(compactness / 4.0, 1.0),
        np.cos(np.deg2rad(orientation)),
        np.sin(np.deg2rad(orientation)),
        radial,
        centroid_r,
        centroid_c,
        float(cluster.get('angular_coverage', 0.0)),
        float(cluster.get('radial_std', 0.0)),
    ], dtype=np.float32)
    desc = np.concatenate([features, type_onehot, profile]).astype(np.float32)
    norm = np.linalg.norm(desc)
    if norm > 1e-8:
        desc = desc / norm
    return desc


def clusters_to_records(clusters: List[Dict], map_shape) -> List[Dict]:
    records = []
    for idx, cluster in enumerate(clusters):
        desc = cluster_descriptor(cluster, map_shape)
        records.append({
            'token_id': idx,
            'descriptor': desc,
            'area': float(cluster.get('area', 0)),
            'area_ratio': float(cluster.get('area', 0)) / max(map_shape[0] * map_shape[1], 1),
            'pos': np.array([
                float(cluster.get('centroid_row', 0.0)) / max(map_shape[0], 1),
                float(cluster.get('centroid_col', 0.0)) / max(map_shape[1], 1),
            ], dtype=np.float32),
            'geometry_type': cluster.get('geometry_type', 'irregular'),
            'cluster': cluster,
        })
    return records


def token_match_components(query: Dict, candidate: Dict, sigma_pos: float = 0.35, sigma_area: float = 1.0) -> Dict:
    desc_sim = float(np.dot(query['descriptor'], candidate['descriptor']))
    desc_sim = max(desc_sim, 0.0)
    pos_dist2 = float(((query['pos'] - candidate['pos']) ** 2).sum())
    pos_affinity = float(np.exp(-pos_dist2 / max(sigma_pos ** 2, 1e-6)))
    area_q = max(query['area'], 1.0)
    area_c = max(candidate['area'], 1.0)
    area_affinity = float(np.exp(-abs(np.log(area_q / area_c)) / max(sigma_area, 1e-6)))
    type_affinity = _type_affinity(query['geometry_type'], candidate['geometry_type'])
    score = desc_sim * pos_affinity * area_affinity * type_affinity
    return {
        'score': float(score),
        'shape_sim': float(desc_sim),
        'position_affinity': float(pos_affinity),
        'area_affinity': float(area_affinity),
        'type_affinity': float(type_affinity),
    }


def token_match_score(query: Dict, candidate: Dict, sigma_pos: float = 0.35, sigma_area: float = 1.0) -> float:
    return token_match_components(query, candidate, sigma_pos=sigma_pos, sigma_area=sigma_area)['score']


def map_similarity(query_tokens: List[Dict], candidate_tokens: List[Dict],
                   sigma_pos: float = 0.35, sigma_area: float = 1.0, topk: int = 1) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    scores = []
    weights = []
    for qt in query_tokens:
        pair_scores = [
            token_match_score(qt, ct, sigma_pos=sigma_pos, sigma_area=sigma_area)
            for ct in candidate_tokens
        ]
        k = min(topk, len(pair_scores))
        top = np.sort(pair_scores)[-k:]
        scores.append(float(top.mean()))
        weights.append(float(np.sqrt(max(qt['area'], 1.0))))
    scores = np.asarray(scores, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    return float((scores * weights).sum() / max(weights.sum(), 1e-6))


def explain_map_similarity(query_tokens: List[Dict], candidate_tokens: List[Dict],
                           sigma_pos: float = 0.35, sigma_area: float = 1.0, topk: int = 1) -> Dict:
    if not query_tokens or not candidate_tokens:
        return {'score': 0.0, 'matches': []}

    matches = []
    weighted_scores = []
    weights = []
    for qt in query_tokens:
        comps = [
            (ct, token_match_components(qt, ct, sigma_pos=sigma_pos, sigma_area=sigma_area))
            for ct in candidate_tokens
        ]
        comps.sort(key=lambda x: x[1]['score'], reverse=True)
        selected = comps[:min(topk, len(comps))]
        token_score = float(np.mean([item[1]['score'] for item in selected])) if selected else 0.0
        weight = float(np.sqrt(max(qt['area'], 1.0)))
        weighted_scores.append(token_score * weight)
        weights.append(weight)
        for rank, (ct, comp) in enumerate(selected, start=1):
            matches.append({
                'query_token_id': int(qt['token_id']),
                'candidate_token_id': int(ct['token_id']),
                'match_rank': rank,
                'query_type': qt['geometry_type'],
                'candidate_type': ct['geometry_type'],
                'query_area': float(qt['area']),
                'candidate_area': float(ct['area']),
                **comp,
            })

    score = float(np.sum(weighted_scores) / max(np.sum(weights), 1e-6))
    return {'score': score, 'matches': matches}


def _shape_profiles(cluster: Dict, map_shape, bins: int = 8) -> np.ndarray:
    pixels = _pixels_array(cluster)
    if len(pixels) == 0:
        return np.zeros(bins * 2, dtype=np.float32)
    H, W = map_shape
    center = np.array([cluster.get('centroid_row', H / 2.0), cluster.get('centroid_col', W / 2.0)], dtype=np.float32)
    rel = pixels - center
    radius = np.linalg.norm(rel, axis=1)
    radius = radius / max(radius.max(), 1.0)
    radial_hist, _ = np.histogram(radius, bins=bins, range=(0.0, 1.0))
    theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
    theta_hist, _ = np.histogram(theta, bins=bins, range=(0.0, 360.0))
    profile = np.concatenate([radial_hist, theta_hist]).astype(np.float32)
    profile = profile / max(profile.sum(), 1.0)
    return profile


def _pixels_array(cluster: Dict) -> np.ndarray:
    coords = cluster.get('pixels', cluster.get('pixel_coords', []))
    pixels = []
    for coord in coords:
        if isinstance(coord, dict):
            pixels.append((int(coord['row']), int(coord['col'])))
        else:
            pixels.append((int(coord[0]), int(coord[1])))
    return np.asarray(pixels, dtype=np.float32)


def _type_affinity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    compatible = {
        ('line', 'irregular'),
        ('blob', 'irregular'),
        ('central', 'blob'),
    }
    if (a, b) in compatible or (b, a) in compatible:
        return 0.6
    return 0.25
