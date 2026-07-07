from __future__ import annotations

from typing import Dict

import numpy as np


GEOMETRY_TYPES = ["blob", "line", "edge_ring", "central", "irregular"]


def _classify_token(token: Dict) -> str:
    if (
        token.get("radial_distance_norm", 0.0) >= 0.65
        and token.get("angular_coverage", 0.0) >= 0.16
        and token.get("radial_std", 1.0) <= 0.14
    ):
        return "edge_ring"

    area = max(token.get("area", 1), 1)
    bbox_area = max(token.get("bbox_height", 1) * token.get("bbox_width", 1), 1)
    fill_ratio = area / bbox_area
    elongation = token.get("pca_lambda1", 0.0) / max(token.get("pca_lambda2", 0.0), 1e-6)
    aspect = max(
        token.get("bbox_height", 1) / max(token.get("bbox_width", 1), 1),
        token.get("bbox_width", 1) / max(token.get("bbox_height", 1), 1),
    )
    if elongation >= 6.0 or aspect >= 4.0:
        return "line"
    if fill_ratio >= 0.45 and token.get("compactness", 0.0) <= 1.6:
        return "blob"
    if token.get("radial_distance_norm", 1.0) <= 0.35:
        return "central"
    return "irregular"


def _type_affinity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    compatible = {
        ("line", "irregular"),
        ("blob", "irregular"),
        ("central", "blob"),
    }
    if (a, b) in compatible or (b, a) in compatible:
        return 0.6
    return 0.25


def _shape_descriptor(token: Dict, map_shape, mode: str = "normal", rotation_tolerance: bool = False) -> np.ndarray:
    bbox_h = float(token.get("bbox_height", 1))
    bbox_w = float(token.get("bbox_width", 1))
    area = float(token.get("area", 0))
    bbox_area = max(bbox_h * bbox_w, 1.0)
    fill_ratio = area / bbox_area
    aspect = max(bbox_h / max(bbox_w, 1.0), bbox_w / max(bbox_h, 1.0))
    elongation = float(token.get("pca_lambda1", 0.0)) / max(float(token.get("pca_lambda2", 0.0)), 1e-6)
    compactness = float(token.get("compactness", 0.0))
    orientation = float(token.get("orientation", 0.0))

    if rotation_tolerance:
        bins = 4 if mode == "coarse" else 8
        features = np.array([
            fill_ratio,
            np.log1p(aspect) / np.log(16.0),
            np.log1p(elongation) / np.log(64.0),
            min(compactness / 4.0, 1.0),
            float(token.get("radial_distance_norm", 0.0)),
            float(token.get("angular_coverage", 0.0)),
            float(token.get("radial_std", 0.0)),
        ], dtype=np.float32)
        desc = np.concatenate([features, _radial_profile(token, map_shape, bins=bins)]).astype(np.float32)
    elif mode == "coarse":
        features = np.array([
            fill_ratio,
            np.log1p(aspect) / np.log(16.0),
            np.log1p(elongation) / np.log(64.0),
            min(compactness / 4.0, 1.0),
            float(token.get("radial_distance_norm", 0.0)),
            float(token.get("angular_coverage", 0.0)),
            float(token.get("radial_std", 0.0)),
        ], dtype=np.float32)
        desc = np.concatenate([features, _shape_profiles(token, map_shape, bins=4)]).astype(np.float32)
    else:
        features = np.array([
            fill_ratio,
            np.log1p(aspect) / np.log(16.0),
            np.log1p(elongation) / np.log(64.0),
            min(compactness / 4.0, 1.0),
            np.cos(np.deg2rad(orientation)),
            np.sin(np.deg2rad(orientation)),
            float(token.get("radial_distance_norm", 0.0)),
            float(token.get("angular_coverage", 0.0)),
            float(token.get("radial_std", 0.0)),
        ], dtype=np.float32)
        desc = np.concatenate([features, _shape_profiles(token, map_shape, bins=8)]).astype(np.float32)
    norm = float(np.linalg.norm(desc))
    if norm > 1e-8:
        desc /= norm
    return desc


def _shape_profiles(token: Dict, map_shape, bins: int) -> np.ndarray:
    pixels = np.asarray(token.get("pixels", []), dtype=np.float32)
    if len(pixels) == 0:
        return np.zeros(bins * 2, dtype=np.float32)
    h, w = map_shape
    center = np.array([token.get("centroid_row", h / 2.0), token.get("centroid_col", w / 2.0)], dtype=np.float32)
    rel = pixels - center
    radius = np.linalg.norm(rel, axis=1)
    radius = radius / max(float(radius.max()), 1.0)
    radial_hist, _ = np.histogram(radius, bins=bins, range=(0.0, 1.0))
    theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
    theta_hist, _ = np.histogram(theta, bins=bins, range=(0.0, 360.0))
    profile = np.concatenate([radial_hist, theta_hist]).astype(np.float32)
    profile /= max(float(profile.sum()), 1.0)
    return profile


def _radial_profile(token: Dict, map_shape, bins: int) -> np.ndarray:
    pixels = np.asarray(token.get("pixels", []), dtype=np.float32)
    if len(pixels) == 0:
        return np.zeros(bins, dtype=np.float32)
    h, w = map_shape
    center = np.array([token.get("centroid_row", h / 2.0), token.get("centroid_col", w / 2.0)], dtype=np.float32)
    rel = pixels - center
    radius = np.linalg.norm(rel, axis=1)
    radius = radius / max(float(radius.max()), 1.0)
    radial_hist, _ = np.histogram(radius, bins=bins, range=(0.0, 1.0))
    profile = radial_hist.astype(np.float32)
    profile /= max(float(profile.sum()), 1.0)
    return profile
