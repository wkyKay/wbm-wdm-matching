from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalMatchResult:
    score: float
    mean_shape: float
    mean_position: float
    mean_scale: float
    mean_type: float
    matched_tokens: int
    wbm_tokens: int
    wdm_tokens: int


@dataclass(frozen=True)
class ProposalConfig:
    min_area: int
    top_k: int
    connectivity: int
    descriptor_mode: str
    moment_weight: float
    geometry_weight: float
    proposal_mode: str
    rotation_tolerance: bool
    density_sigmas: tuple[float, ...]
    density_threshold: float
    density_min_raw_points: int
    density_min_raw_mass: float
    density_merge_iou: float
    density_weight_transform: str
    ring_min_area: int
    ring_edge_r_min: float
    ring_band_width: float
    ring_min_angular_coverage: float
    ring_angular_bins: int
    ring_max_radial_std: float
    ring_max_defect_ratio: float
    ring_min_edge_defect_fraction: float
