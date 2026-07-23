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

    def test_kde_support_does_not_expand_the_matched_token_region(self) -> None:
        points = [(5, 5), (7, 7), (9, 9), (11, 11), (13, 13)]
        result = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="sparse-density",
            density_sigmas=(1.6,),
            density_threshold=0.15,
            density_min_raw_points=3,
            density_min_raw_mass=3.0,
        )

        token = result["wbm_tokens"][0]
        self.assertEqual(set(token["pixels"]), set(points))
        self.assertEqual(token["area"], len(points))
        self.assertEqual(token["support_area"], len(points))
        self.assertGreater(token["kde_support_area"], token["area"])
        self.assertGreater(len(token["kde_support_pixels"]), len(token["pixels"]))

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

    def test_fragmented_ring_segments_merge_into_one_sparse_ring_token(self) -> None:
        shape = (25, 25)
        center = np.array([12.0, 12.0], dtype=np.float32)
        radius = 8.0
        angles_deg = list(range(0, 75, 4)) + list(range(90, 165, 4)) + list(range(180, 255, 4)) + list(range(270, 345, 4))
        points = set()
        for angle_deg in angles_deg:
            theta = np.deg2rad(angle_deg)
            row = int(round(center[0] + radius * np.sin(theta)))
            col = int(round(center[1] + radius * np.cos(theta)))
            points.add((row, col))
        points = sorted(points)

        rows, cols = np.ogrid[:shape[0], :shape[1]]
        valid = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= 11.5 ** 2
        status = np.where(valid, VALID_NO_DEFECT, BACKGROUND).astype(np.uint8)
        for row, col in points:
            status[row, col] = VALID_HAS_DEFECT
        count = np.zeros(shape, dtype=np.float32)
        for row, col in points:
            count[row, col] = 1.0
        grid = GridMaps(
            count_map=count,
            binary_map=(count > 0).astype(np.uint8),
            density_map=count / max(float(count.sum()), 1.0),
            status_map=status,
            representation_map=count / max(float(count.sum()), 1.0),
            representation_maps={"count": count, "binary": (count > 0).astype(np.uint8)},
            metadata={},
        )

        result = explain_count_partial_match(
            grid,
            grid,
            proposal_mode="sparse-density",
            density_sigmas=(0.8,),
            density_threshold=0.50,
            density_min_raw_points=3,
            density_min_raw_mass=3.0,
        )

        ring_tokens = [token for token in result["wbm_tokens"] if token.get("proposal_source") == "sparse_density_ring_merge"]
        self.assertTrue(ring_tokens)
        self.assertEqual(ring_tokens[0]["geometry_type"], "edge_ring")
        self.assertGreaterEqual(ring_tokens[0]["ring_angular_coverage"], 0.65)
        self.assertEqual(set(ring_tokens[0]["pixels"]), set(points))


if __name__ == "__main__":
    unittest.main()
