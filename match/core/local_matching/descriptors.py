from __future__ import annotations

from math import factorial
from typing import Dict

import numpy as np


GEOMETRY_TYPES = ["blob", "line", "edge_ring", "central", "irregular"]
ZERNIKE_SMALL_CANVAS_SIZE = 16
ZERNIKE_LARGE_CANVAS_SIZE = 48
ZERNIKE_SMALL_MAP_SHORT_SIDE = 12
ZERNIKE_DEGREE = 8


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


def _shape_descriptor(token: Dict, map_shape, mode: str = "normal", rotation_tolerance: bool = False,
                      moment_weight: float = 0.75, geometry_weight: float = 0.25) -> np.ndarray:
    geometry = _geometry_descriptor(token, include_orientation=not rotation_tolerance and mode != "coarse")
    canvas_size = _zernike_canvas_size(map_shape)
    moment = _zernike_descriptor(token, canvas_size=canvas_size, degree=ZERNIKE_DEGREE)
    token["descriptor_parts"] = {
        "moment": moment,
        "geometry": geometry,
        "moment_weight": float(max(moment_weight, 0.0)),
        "geometry_weight": float(max(geometry_weight, 0.0)),
        "kind": "zernike_geometry",
        "zernike_canvas_size": canvas_size,
        "zernike_degree": ZERNIKE_DEGREE,
    }
    desc = np.concatenate([moment, geometry]).astype(np.float32)
    norm = float(np.linalg.norm(desc))
    if norm > 1e-8:
        desc /= norm
    return desc


def _legacy_shape_descriptor(token: Dict, map_shape, mode: str = "normal", rotation_tolerance: bool = False) -> np.ndarray:
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
            float(token.get("angular_coverage", 0.0)),
            float(token.get("radial_std", 0.0)),
        ], dtype=np.float32)
        desc = np.concatenate([features, _shape_profiles(token, map_shape, bins=8)]).astype(np.float32)
    norm = float(np.linalg.norm(desc))
    if norm > 1e-8:
        desc /= norm
    return desc


def _geometry_descriptor(token: Dict, include_orientation: bool) -> np.ndarray:
    bbox_h = float(token.get("bbox_height", 1))
    bbox_w = float(token.get("bbox_width", 1))
    area = float(token.get("area", 0))
    bbox_area = max(bbox_h * bbox_w, 1.0)
    fill_ratio = area / bbox_area
    aspect = max(bbox_h / max(bbox_w, 1.0), bbox_w / max(bbox_h, 1.0))
    elongation = float(token.get("pca_lambda1", 0.0)) / max(float(token.get("pca_lambda2", 0.0)), 1e-6)
    compactness = float(token.get("compactness", 0.0))
    features = [
        fill_ratio,
        np.log1p(aspect) / np.log(16.0),
        np.log1p(elongation) / np.log(64.0),
        min(compactness / 4.0, 1.0),
        float(token.get("angular_coverage", 0.0)),
        float(token.get("radial_std", 0.0)),
    ]
    features.extend(_local_geometry_features(token))
    if include_orientation:
        orientation = float(token.get("orientation", 0.0))
        features.extend([
            0.5 + 0.5 * np.cos(np.deg2rad(orientation)),
            0.5 + 0.5 * np.sin(np.deg2rad(orientation)),
        ])
    return np.clip(np.asarray(features, dtype=np.float32), 0.0, 1.0)


def _local_geometry_features(token: Dict) -> list[float]:
    """Describe radial fill and angular continuity in token-local coordinates."""
    pixels = np.asarray(token.get("pixels", []), dtype=np.float32)
    if len(pixels) == 0:
        return [0.0] * 8
    center = np.array([
        float(token.get("centroid_row", pixels[:, 0].mean())),
        float(token.get("centroid_col", pixels[:, 1].mean())),
    ], dtype=np.float32)
    rel = pixels - center
    radius = np.linalg.norm(rel, axis=1)
    radius /= max(float(radius.max()), 1.0)
    radial_hist, _ = np.histogram(radius, bins=4, range=(0.0, 1.0))
    radial_hist = radial_hist.astype(np.float32)
    radial_hist /= max(float(radial_hist.sum()), 1.0)
    inner = float(radial_hist[:2].sum())
    outer = float(radial_hist[2:].sum())
    center_occupancy = float(radial_hist[0])
    inner_outer_ratio = inner / max(outer, 1e-6)
    inner_outer_ratio = float(np.clip(inner_outer_ratio, 0.0, 1.0))
    radial_concentration = float(np.clip(radial_hist.max(), 0.0, 1.0))
    theta = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360.0) % 360.0
    occupied = np.unique(np.floor(theta / 30.0).astype(np.int64))
    local_angular_coverage = float(len(occupied) / 12.0)
    return [*radial_hist.tolist(), center_occupancy, inner_outer_ratio, radial_concentration, local_angular_coverage]


def _zernike_canvas_size(map_shape) -> int:
    return ZERNIKE_SMALL_CANVAS_SIZE if min(map_shape) <= ZERNIKE_SMALL_MAP_SHORT_SIDE else ZERNIKE_LARGE_CANVAS_SIZE


def _zernike_descriptor(token: Dict, canvas_size: int, degree: int = 8) -> np.ndarray:
    mask = _token_fixed_canvas_mask(token, canvas_size)
    if mask.size == 0 or not mask.any():
        return _empty_zernike_descriptor(degree)

    mask = mask.astype(np.float32)
    yy, xx = np.indices((canvas_size, canvas_size), dtype=np.float32)
    center = (canvas_size - 1) / 2.0
    radius = max(center, 1.0)
    x = (xx - center) / radius
    y = (yy - center) / radius
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    unit_disk = rho <= 1.0
    values = mask * unit_disk.astype(np.float32)
    mass = float(values.sum())
    if mass <= 1e-8:
        return _empty_zernike_descriptor(degree)

    moments = []
    for n in range(degree + 1):
        for m in range(n + 1):
            if (n - m) % 2 != 0 or (n == 0 and m == 0):
                continue
            radial = _zernike_radial(n, m, rho)
            real = values * radial * np.cos(m * theta)
            imag = values * radial * np.sin(m * theta)
            moment = (n + 1) * complex(float(real[unit_disk].sum()), -float(imag[unit_disk].sum())) / mass
            moments.append(abs(moment))

    desc = np.asarray(moments, dtype=np.float32)
    norm = float(np.linalg.norm(desc))
    if norm > 1e-8:
        desc /= norm
    return desc


def _token_fixed_canvas_mask(token: Dict, canvas_size: int) -> np.ndarray:
    pixels = np.asarray(token.get("pixels", []), dtype=np.int64)
    if pixels.size == 0:
        return np.zeros((0, 0), dtype=bool)
    center_row = float(token.get("centroid_row", pixels[:, 0].mean()))
    center_col = float(token.get("centroid_col", pixels[:, 1].mean()))
    origin_row = int(np.floor(center_row - (canvas_size - 1) / 2.0))
    origin_col = int(np.floor(center_col - (canvas_size - 1) / 2.0))
    rows = pixels[:, 0] - origin_row
    cols = pixels[:, 1] - origin_col
    keep = (rows >= 0) & (rows < canvas_size) & (cols >= 0) & (cols < canvas_size)
    mask = np.zeros((canvas_size, canvas_size), dtype=bool)
    mask[rows[keep], cols[keep]] = True
    return mask


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    radial = np.zeros_like(rho, dtype=np.float32)
    for s in range((n - m) // 2 + 1):
        coeff = (
            ((-1) ** s)
            * factorial(n - s)
            / (
                factorial(s)
                * factorial((n + m) // 2 - s)
                * factorial((n - m) // 2 - s)
            )
        )
        radial += float(coeff) * np.power(rho, n - 2 * s)
    return radial


def _empty_zernike_descriptor(degree: int) -> np.ndarray:
    size = sum(1 for n in range(degree + 1) for m in range(n + 1) if (n - m) % 2 == 0 and not (n == 0 and m == 0))
    return np.zeros(size, dtype=np.float32)


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
