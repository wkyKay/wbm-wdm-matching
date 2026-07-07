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
    proposal_mode: str
    rotation_tolerance: bool