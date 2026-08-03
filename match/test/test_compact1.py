"""Regression tests for the raw-pixel small-map compact1 proposal mode."""
from __future__ import annotations

import unittest

import numpy as np

from match.core.local_matching import explain_count_partial_match
from match.core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


def _grid(points: list[tuple[int, int]]) -> GridMaps:
    shape = (7, 12)
    status = np.full(shape, VALID_NO_DEFECT, dtype=np.uint8)
    count = np.zeros(shape, dtype=np.float32)
    for row, col in points:
        status[row, col] = VALID_HAS_DEFECT
        count[row, col] = 1.0
    density = count / max(float(count.sum()), 1.0)
    return GridMaps(
        count_map=count,
        binary_map=(count > 0).astype(np.uint8),
        density_map=density,
        status_map=status,
        representation_map=density,
        representation_maps={"count": count, "binary": (count > 0).astype(np.uint8), "density": density},
        metadata={},
    )


class Compact1ProposalTest(unittest.TestCase):
    def test_one_pixel_gap_groups_raw_arc_without_adding_bridge_pixels(self) -> None:
        # The two vertical fragments have a one-cell break at (3, 0).
        points = [(1, 0), (2, 0), (4, 0), (5, 0)]
        explanation = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="compact1",
            min_area=5,
            top_k=4,
        )

        tokens = explanation["wbm_tokens"]
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["geometry_type"], "ring_arc")
        self.assertEqual(set(tokens[0]["pixels"]), set(points))
        self.assertNotIn((3, 0), tokens[0]["pixels"])
        self.assertEqual(explanation["proposal_debug"]["wbm"]["mode"], "compact1")


if __name__ == "__main__":
    unittest.main()
