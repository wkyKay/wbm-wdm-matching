"""Proposal generation: extract candidate defect regions from wafer maps.

Module structure:
- proposal_utils.py   -- shared utilities, constants, _token_stats, circular helpers
- proposal_density.py -- sparse-density mode (KDE-based)
- proposal_retrieval.py -- compact / arc / tangential-ring modes (ring & arc detection)
- proposal.py         -- public entry points, config, token selection & finalization
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import ProposalConfig
from .morphology import _connected_components
from .proposal_utils import _finalize_token, _is_sparse_density_mode, _token_stats
# Re-export utilities that external consumers depend on
from .proposal_utils import _bridge_short_circular_gaps, _circular_arc_runs_with_gap_limits  # noqa: F401
from .proposal_density import _tokens_from_sparse_density
from .proposal_retrieval import (
    _retrieval_compact_tokens,
    _tangential_ring_tokens,
)

DEFAULT_REQUESTED_MIN_AREA = 5
DEFAULT_REQUESTED_TOP_K = 6

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _tokens_from_mask(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
    proposal_debug: Optional[Dict] = None,
) -> List[Dict]:
    weight_map = mask.astype(np.float32)
    if _is_sparse_density_mode(proposal_config.proposal_mode):
        return _tokens_from_sparse_density(weight_map, valid_mask, proposal_config, source="wbm")
    return _tokens_from_components(
        mask,
        valid_mask,
        weight_map,
        proposal_config=proposal_config,
        source="wbm",
        proposal_debug=proposal_debug,
    )


def _tokens_from_count(
    count_map: np.ndarray,
    valid_mask: np.ndarray,
    proposal_config: ProposalConfig,
) -> List[Dict]:
    mask = (count_map > 0) & valid_mask
    return _tokens_from_weighted_mask(mask, valid_mask, count_map.astype(np.float32), proposal_config=proposal_config)


def _tokens_from_weighted_mask(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    weight_map: np.ndarray,
    proposal_config: ProposalConfig,
    raw_weight_map: Optional[np.ndarray] = None,
    proposal_debug: Optional[Dict] = None,
) -> List[Dict]:
    if _is_sparse_density_mode(proposal_config.proposal_mode):
        return _tokens_from_sparse_density(
            weight_map,
            valid_mask,
            proposal_config,
            source="wdm",
            raw_weight_map=raw_weight_map,
        )
    return _tokens_from_components(
        mask,
        valid_mask,
        weight_map,
        proposal_config=proposal_config,
        source="wdm",
        proposal_debug=proposal_debug,
    )


# ---------------------------------------------------------------------------
# Component-mode dispatch
# ---------------------------------------------------------------------------


def _tokens_from_components(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    weight_map: np.ndarray,
    proposal_config: ProposalConfig,
    source: str,
    proposal_debug: Optional[Dict] = None,
) -> List[Dict]:
    h, w = mask.shape
    if proposal_config.proposal_mode in {"compact", "arc", "arc-band-residual", "arc-ring-residual"}:
        tokens, ring_debug = _retrieval_compact_tokens(
            mask & valid_mask, weight_map, valid_mask, proposal_config, source=source,
        )
        if proposal_debug is not None:
            proposal_debug[source] = ring_debug
        for token in tokens:
            _finalize_token(token, (h, w), proposal_config)
        return tokens

    if proposal_config.proposal_mode == "tangential-ring":
        tokens, ring_debug = _tangential_ring_tokens(
            mask & valid_mask, weight_map, valid_mask, proposal_config, source=source,
        )
        if proposal_debug is not None:
            proposal_debug[source] = ring_debug
        for token in tokens:
            _finalize_token(token, (h, w), proposal_config)
        return tokens

    tokens: List[Dict] = []
    total_mass = float(weight_map[mask].sum())
    for comp in _connected_components(mask, connectivity=proposal_config.connectivity):
        if len(comp) < proposal_config.min_area:
            continue
        token = _token_stats(comp, weight_map, valid_mask, total_mass=total_mass, source=source)
        tokens.append(token)

    tokens.sort(key=_token_importance, reverse=True)
    tokens = tokens[:proposal_config.top_k]
    for token in tokens:
        _finalize_token(token, (h, w), proposal_config)
    return tokens


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _proposal_config(
    shape: tuple[int, int],
    valid_area: int,
    min_area: int,
    top_k: int,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    density_sigmas: tuple[float, ...] = (0.8, 1.6, 3.2),
    density_threshold: float = 0.20,
    density_min_raw_points: int = 3,
    density_min_raw_mass: float = 3.0,
    density_merge_iou: float = 0.60,
    density_weight_transform: str = "sqrt",
    ring_min_area: Optional[int] = None,
    ring_edge_r_min: Optional[float] = None,
    ring_band_width: Optional[float] = None,
    ring_min_angular_coverage: Optional[float] = None,
    ring_angular_bins: Optional[int] = None,
    ring_max_radial_std: Optional[float] = None,
    ring_max_defect_ratio: Optional[float] = None,
    ring_min_edge_defect_fraction: Optional[float] = None,
) -> ProposalConfig:
    proposal_mode = proposal_mode.lower().strip()
    if proposal_mode not in {"cc", "compact", "arc", "arc-band-residual", "arc-ring-residual", "tangential-ring", "sparse-density"}:
        raise ValueError(f"Unsupported count-partial proposal mode: {proposal_mode}")
    density_sigmas = tuple(float(sigma) for sigma in density_sigmas)
    if not density_sigmas or any(sigma <= 0.0 for sigma in density_sigmas):
        raise ValueError("density_sigmas must contain positive values")
    if density_weight_transform not in {"count", "sqrt", "log1p"}:
        raise ValueError(f"Unsupported density weight transform: {density_weight_transform}")

    short_side = min(int(shape[0]), int(shape[1]))
    valid_area = max(int(valid_area), 1)

    if short_side <= 12:
        adaptive_min_area = 2
        adaptive_top_k = 4
        descriptor_mode = "coarse"
        ring_defaults = (6, 0.65, 0.10, 0.10, 24, 0.18, 0.60, 0.45)
    elif short_side <= 25:
        adaptive_min_area = max(3, int(round(valid_area * 0.01)))
        adaptive_top_k = 6
        descriptor_mode = "normal"
        ring_defaults = (12, 0.65, 0.10, 0.16, 72, 0.12, 0.45, 0.45)
    else:
        adaptive_min_area = max(5, int(round(valid_area * 0.005)))
        adaptive_top_k = 8
        descriptor_mode = "normal"
        ring_defaults = (12, 0.65, 0.10, 0.16, 72, 0.12, 0.45, 0.45)

    (
        default_ring_min_area,
        default_ring_edge_r_min,
        default_ring_band_width,
        default_ring_min_angular_coverage,
        default_ring_angular_bins,
        default_ring_max_radial_std,
        default_ring_max_defect_ratio,
        default_ring_min_edge_defect_fraction,
    ) = ring_defaults

    return ProposalConfig(
        min_area=_effective_proposal_min_area(int(min_area), adaptive_min_area),
        top_k=_effective_proposal_top_k(int(top_k), adaptive_top_k),
        connectivity=8,
        descriptor_mode=descriptor_mode,
        proposal_mode=proposal_mode,
        rotation_tolerance=bool(rotation_tolerance),
        density_sigmas=tuple(sorted(density_sigmas)),
        density_threshold=max(float(density_threshold), 0.0),
        density_min_raw_points=max(int(density_min_raw_points), 1),
        density_min_raw_mass=max(float(density_min_raw_mass), 0.0),
        density_merge_iou=min(max(float(density_merge_iou), 0.0), 1.0),
        density_weight_transform=density_weight_transform,
        ring_min_area=max(int(ring_min_area if ring_min_area is not None else default_ring_min_area), 1),
        ring_edge_r_min=min(max(float(ring_edge_r_min if ring_edge_r_min is not None else default_ring_edge_r_min), 0.0), 1.0),
        ring_band_width=max(float(ring_band_width if ring_band_width is not None else default_ring_band_width), 0.0),
        ring_min_angular_coverage=min(max(float(ring_min_angular_coverage if ring_min_angular_coverage is not None else default_ring_min_angular_coverage), 0.0), 1.0),
        ring_angular_bins=max(int(ring_angular_bins if ring_angular_bins is not None else default_ring_angular_bins), 1),
        ring_max_radial_std=max(float(ring_max_radial_std if ring_max_radial_std is not None else default_ring_max_radial_std), 0.0),
        ring_max_defect_ratio=min(max(float(ring_max_defect_ratio if ring_max_defect_ratio is not None else default_ring_max_defect_ratio), 0.0), 1.0),
        ring_min_edge_defect_fraction=min(max(float(ring_min_edge_defect_fraction if ring_min_edge_defect_fraction is not None else default_ring_min_edge_defect_fraction), 0.0), 1.0),
    )


def _effective_proposal_min_area(requested: int, adaptive: int) -> int:
    if requested == DEFAULT_REQUESTED_MIN_AREA:
        return max(1, min(requested, adaptive))
    return max(1, requested)


def _effective_proposal_top_k(requested: int, adaptive: int) -> int:
    if requested == DEFAULT_REQUESTED_TOP_K:
        return max(1, min(requested, adaptive))
    return max(1, requested)


# ---------------------------------------------------------------------------
# Token importance (used by cc mode in _tokens_from_components)
# ---------------------------------------------------------------------------


def _token_importance(token: Dict) -> float:
    mass = float(token.get("mass", token.get("area", 0)))
    area = float(token.get("area", 0))
    type_bonus = {
        "edge_ring": 2.0,
        "central": 1.5,
        "blob": 1.2,
        "line": 1.0,
        "irregular": 0.8,
    }.get(token.get("geometry_type"), 0.8)
    return float(np.sqrt(max(mass, 0.0)) + 0.25 * np.sqrt(max(area, 0.0)) + type_bonus)
