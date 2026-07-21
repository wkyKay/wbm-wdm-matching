"""Discrete rigid registration and raw-mask overlap scoring for small wafer maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)


@dataclass(frozen=True)
class RigidOverlayResult:
    score: float
    dice: float
    iou: float
    angle_deg: int
    shift_row: int
    shift_col: int
    retained_fraction: float
    transformed_mask: np.ndarray


def score_rigid_overlay(
    query_mask: np.ndarray,
    candidate_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    score_mode: str = "dice",
    angles: Iterable[int] = DEFAULT_ANGLES,
    max_shift: int = 1,
    min_retained_fraction: float = 0.95,
) -> RigidOverlayResult:
    """Return the best overlap under discrete rotation and bounded translation.

    Rotation uses nearest-cell placement around the wafer center. No interpolation,
    dilation, or synthetic defect cells are introduced.
    """
    if score_mode not in {"dice", "iou"}:
        raise ValueError(f"Unsupported rigid-overlay score mode: {score_mode}")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    query = np.asarray(query_mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    if query.shape != candidate.shape or query.shape != valid_mask.shape:
        raise ValueError("query_mask, candidate_mask, and valid_mask must share the same shape")

    candidate_count = int(candidate.sum())
    if not query.any() or candidate_count == 0:
        return _empty_result(query.shape)

    candidate_points = np.argwhere(candidate)
    center = np.asarray([query.shape[0] / 2.0, query.shape[1] / 2.0], dtype=np.float32)
    best: RigidOverlayResult | None = None
    normalized_angles = tuple(dict.fromkeys(int(angle) % 360 for angle in angles))
    if not normalized_angles:
        raise ValueError("angles must contain at least one angle")

    for angle_deg in normalized_angles:
        rotated = _rotate_points(candidate_points, center, angle_deg)
        for shift_row in range(-max_shift, max_shift + 1):
            for shift_col in range(-max_shift, max_shift + 1):
                transformed = _place_points(rotated, query.shape, valid_mask, shift_row, shift_col)
                retained_fraction = float(transformed.sum() / candidate_count)
                if retained_fraction < min_retained_fraction:
                    continue
                dice, iou = _overlap_scores(query, transformed)
                score = dice if score_mode == "dice" else iou
                result = RigidOverlayResult(
                    score=score,
                    dice=dice,
                    iou=iou,
                    angle_deg=angle_deg,
                    shift_row=shift_row,
                    shift_col=shift_col,
                    retained_fraction=retained_fraction,
                    transformed_mask=transformed,
                )
                if best is None or _is_better(result, best):
                    best = result

    return best if best is not None else _empty_result(query.shape)


def score_proposal_rigid_overlay(
    query_pixels: Iterable[tuple[int, int]],
    candidate_pixels: Iterable[tuple[int, int]],
    *,
    score_mode: str = "dice",
    angles: Iterable[int] = DEFAULT_ANGLES,
    max_shift: int = 1,
    min_retained_fraction: float = 0.95,
) -> RigidOverlayResult:
    """Score two proposal masks after aligning their integer centroid anchors.

    The local canvas has enough padding for every requested translation, so the
    retention check detects only losses from the discrete rotation itself, not
    artificial clipping caused by a proposal bounding box.
    """
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    query_points = _as_points(query_pixels)
    candidate_points = _as_points(candidate_pixels)
    if not len(query_points) or not len(candidate_points):
        return _empty_result((1, 1))

    query_anchor = np.rint(query_points.mean(axis=0)).astype(np.int64)
    candidate_anchor = np.rint(candidate_points.mean(axis=0)).astype(np.int64)
    query_offsets = query_points - query_anchor
    candidate_offsets = candidate_points - candidate_anchor
    extent = int(max(np.abs(query_offsets).max(), np.abs(candidate_offsets).max())) + max_shift + 1
    size = 2 * extent + 2
    center = np.asarray([size // 2, size // 2], dtype=np.int64)

    query_mask = _mask_from_points(query_offsets + center, (size, size))
    candidate_mask = _mask_from_points(candidate_offsets + center, (size, size))
    return score_rigid_overlay(
        query_mask,
        candidate_mask,
        np.ones((size, size), dtype=bool),
        score_mode=score_mode,
        angles=angles,
        max_shift=max_shift,
        min_retained_fraction=min_retained_fraction,
    )


def _rotate_points(points: np.ndarray, center: np.ndarray, angle_deg: int) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    return np.rint((points.astype(np.float32) - center) @ rotation.T + center).astype(np.int64)


def _as_points(pixels: Iterable[tuple[int, int]]) -> np.ndarray:
    points = np.asarray(list(pixels), dtype=np.int64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("proposal pixels must be an iterable of (row, col) pairs")
    return points


def _mask_from_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[points[:, 0], points[:, 1]] = True
    return mask


def _place_points(
    points: np.ndarray,
    shape: tuple[int, int],
    valid_mask: np.ndarray,
    shift_row: int,
    shift_col: int,
) -> np.ndarray:
    shifted = points + np.asarray([shift_row, shift_col], dtype=np.int64)
    in_bounds = (
        (shifted[:, 0] >= 0)
        & (shifted[:, 0] < shape[0])
        & (shifted[:, 1] >= 0)
        & (shifted[:, 1] < shape[1])
    )
    shifted = shifted[in_bounds]
    transformed = np.zeros(shape, dtype=bool)
    if len(shifted):
        transformed[shifted[:, 0], shifted[:, 1]] = True
    return transformed & valid_mask


def _overlap_scores(query: np.ndarray, transformed: np.ndarray) -> tuple[float, float]:
    intersection = int((query & transformed).sum())
    query_count = int(query.sum())
    candidate_count = int(transformed.sum())
    union = query_count + candidate_count - intersection
    dice_denominator = query_count + candidate_count
    dice = float(2 * intersection / dice_denominator) if dice_denominator else 0.0
    iou = float(intersection / union) if union else 0.0
    return dice, iou


def _is_better(candidate: RigidOverlayResult, incumbent: RigidOverlayResult) -> bool:
    if candidate.score != incumbent.score:
        return candidate.score > incumbent.score
    candidate_shift = candidate.shift_row**2 + candidate.shift_col**2
    incumbent_shift = incumbent.shift_row**2 + incumbent.shift_col**2
    if candidate_shift != incumbent_shift:
        return candidate_shift < incumbent_shift
    candidate_angle = min(candidate.angle_deg, 360 - candidate.angle_deg)
    incumbent_angle = min(incumbent.angle_deg, 360 - incumbent.angle_deg)
    return candidate_angle < incumbent_angle


def _empty_result(shape: tuple[int, int]) -> RigidOverlayResult:
    return RigidOverlayResult(
        score=0.0,
        dice=0.0,
        iou=0.0,
        angle_deg=0,
        shift_row=0,
        shift_col=0,
        retained_fraction=0.0,
        transformed_mask=np.zeros(shape, dtype=bool),
    )
