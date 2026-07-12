"""Focused regression tests for sparse-density WBM/WDM proposal generation."""
from __future__ import annotations

import unittest

import numpy as np

from match.core.local_matching import explain_count_partial_match
from match.core.models import BACKGROUND, GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


def _grid(points: list[tuple[int, int]], count: float = 1.0) -> GridMaps:
    shape = (25, 25)
    status = np.full(shape, VALID_NO_DEFECT, dtype=np.uint8)
    status[[0, -1], :] = BACKGROUND
    status[:, [0, -1]] = BACKGROUND
    for row, col in points:
        status[row, col] = VALID_HAS_DEFECT
    count_map = np.zeros(shape, dtype=np.float32)
    for row, col in points:
        count_map[row, col] = count
    density_map = count_map / max(float(count_map.sum()), 1.0)
    return GridMaps(
        count_map=count_map,
        binary_map=(count_map > 0).astype(np.uint8),
        density_map=density_map,
        status_map=status,
        representation_map=density_map,
        representation_maps={"count": count_map, "binary": (count_map > 0).astype(np.uint8)},
        metadata={},
    )


class SparseDensityProposalTest(unittest.TestCase):
    def test_sparse_wbm_and_wdm_share_a_supported_token(self) -> None:
        points = [(5, 5), (7, 7), (9, 9), (11, 11), (13, 13)]
        result = explain_count_partial_match(
            _grid(points),
            _grid(points, count=4.0),
            proposal_mode="sparse-density",
            density_sigmas=(0.8, 1.6),
            density_threshold=0.15,
            density_min_raw_points=3,
            density_min_raw_mass=3.0,
        )
        self.assertTrue(result["wbm_tokens"])
        self.assertTrue(result["wdm_tokens"])
        self.assertGreater(result["result"].score, 0.75)
        self.assertEqual(result["wbm_tokens"][0]["raw_point_count"], len(points))
        self.assertEqual(result["wdm_tokens"][0]["raw_mass"], float(len(points) * 4))

    def test_insufficient_sparse_evidence_does_not_create_tokens(self) -> None:
        points = [(5, 5), (9, 9)]
        result = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="sparse-density",
            density_min_raw_points=3,
            density_min_raw_mass=3.0,
        )
        self.assertFalse(result["wbm_tokens"])
        self.assertFalse(result["wdm_tokens"])

    def test_auto_uses_a_shared_sparse_representation_for_fragmented_pair(self) -> None:
        points = [(5, 5), (7, 7), (9, 9), (11, 11), (13, 13)]
        result = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="auto",
            density_sigmas=(0.8, 1.6),
            density_min_raw_points=3,
            density_min_raw_mass=3.0,
        )
        self.assertEqual(result["wbm_tokens"][0]["proposal_config"]["proposal_mode"], "sparse-density")
        self.assertEqual(result["wdm_tokens"][0]["proposal_config"]["proposal_mode"], "sparse-density")


if __name__ == "__main__":
    unittest.main()
