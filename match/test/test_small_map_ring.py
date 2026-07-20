"""Regression tests for compact ring proposals on tiny wafer maps."""
from __future__ import annotations

import unittest

import numpy as np

from match.core.local_matching import explain_count_partial_match
from match.core.local_matching.proposal import _bridge_short_circular_gaps
from match.core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


def _grid(points: list[tuple[int, int]]) -> GridMaps:
    shape = (12, 12)
    status_map = np.full(shape, VALID_NO_DEFECT, dtype=np.uint8)
    count_map = np.zeros(shape, dtype=np.float32)
    for row, col in points:
        status_map[row, col] = VALID_HAS_DEFECT
        count_map[row, col] = 1.0
    density_map = count_map / max(float(count_map.sum()), 1.0)
    return GridMaps(
        count_map=count_map,
        binary_map=(count_map > 0).astype(np.uint8),
        density_map=density_map,
        status_map=status_map,
        representation_map=density_map,
        representation_maps={"count": count_map, "binary": (count_map > 0).astype(np.uint8)},
        metadata={},
    )


class SmallMapRingProposalTest(unittest.TestCase):
    def test_tangential_gap_bridge_fills_two_bins_but_not_three(self) -> None:
        two_cell_gap = np.array([True, False, False, True, False, False, False, True], dtype=bool)
        bridged = _bridge_short_circular_gaps(two_cell_gap, max_gap_bins=2)
        self.assertTrue(bridged[1])
        self.assertTrue(bridged[2])
        self.assertFalse(bridged[4])
        self.assertFalse(bridged[5])
        self.assertFalse(bridged[6])

    def test_tangential_ring_mode_keeps_only_raw_pixels(self) -> None:
        points = [(0, 6), (2, 2), (2, 10), (6, 0), (6, 11), (10, 2), (10, 10), (11, 6)]
        explanation = explain_count_partial_match(_grid(points), _grid(points), proposal_mode="tangential-ring", min_area=5)
        token = next(token for token in explanation["wbm_tokens"] if token["geometry_type"] == "edge_ring")

        self.assertEqual(token["proposal_source"], "tangential_ring")
        self.assertTrue(set(token["pixels"]).issubset(set(points)))
        self.assertGreaterEqual(token["ring_contour_angular_coverage"], token["ring_raw_angular_coverage"])
        self.assertTrue(explanation["proposal_debug"]["wbm"]["accepted"])

    def test_single_cell_ring_fragments_survive_before_closing(self) -> None:
        points = [(0, 6), (2, 2), (2, 10), (6, 0), (6, 11), (10, 2), (10, 10), (11, 6)]
        explanation = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="compact",
            min_area=5,
        )
        ring_tokens = [token for token in explanation["wbm_tokens"] if token["geometry_type"] == "edge_ring"]

        self.assertEqual(len(ring_tokens), 1)
        self.assertEqual(ring_tokens[0]["proposal_config"]["ring_min_area"], 6)
        self.assertEqual(ring_tokens[0]["proposal_config"]["ring_angular_bins"], 24)
        self.assertTrue(explanation["proposal_debug"]["wbm"]["accepted"])
        self.assertEqual(explanation["proposal_debug"]["wbm"]["reason"], "accepted")

    def test_closing_is_contour_evidence_not_ring_token_pixels(self) -> None:
        points = [(0, 6), (2, 2), (2, 10), (6, 0), (6, 11), (10, 2), (10, 10), (11, 6)]
        explanation = explain_count_partial_match(_grid(points), _grid(points), proposal_mode="compact", min_area=5)
        token = next(token for token in explanation["wbm_tokens"] if token["geometry_type"] == "edge_ring")

        self.assertTrue(set(token["pixels"]).issubset(set(points)))
        self.assertEqual(token["area"], len(token["pixels"]))
        self.assertGreaterEqual(token["ring_contour_area"], token["area"])
        self.assertGreaterEqual(
            explanation["proposal_debug"]["wbm"]["contour_area"],
            explanation["proposal_debug"]["wbm"]["raw_ring_area"],
        )

    def test_small_map_uses_coarser_angular_coverage_than_large_map(self) -> None:
        points = [(0, 6), (2, 2), (2, 10), (6, 0), (6, 11), (10, 2), (10, 10), (11, 6)]
        explanation = explain_count_partial_match(_grid(points), _grid(points), proposal_mode="compact")

        token = explanation["wbm_tokens"][0]
        self.assertGreaterEqual(token["ring_contour_angular_coverage"], 4 / 24)
        self.assertEqual(token["proposal_config"]["ring_angular_bins"], 24)

    def test_explicit_ring_settings_override_small_map_defaults(self) -> None:
        points = [(0, 6), (2, 2), (2, 10), (6, 0), (6, 11), (10, 2), (10, 10), (11, 6)]
        explanation = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="compact",
            ring_min_area=6,
            ring_angular_bins=72,
            ring_min_angular_coverage=0.10,
        )

        token = explanation["wbm_tokens"][0]
        self.assertEqual(token["proposal_config"]["ring_angular_bins"], 72)


if __name__ == "__main__":
    unittest.main()
