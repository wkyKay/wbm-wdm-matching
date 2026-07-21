"""Regression tests for small-map discrete rigid-overlay matching."""
from __future__ import annotations

import unittest

import numpy as np

from match.core.local_matching import explain_count_partial_match
from match.core.local_matching.rigid_overlay import score_rigid_overlay
from match.core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


def _grid(points: list[tuple[int, int]], shape: tuple[int, int] = (12, 12)) -> GridMaps:
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
        representation_maps={"count": count, "binary": (count > 0).astype(np.uint8)},
        metadata={},
    )


class RigidOverlayTest(unittest.TestCase):
    def test_rotation_and_shift_recover_exact_overlap(self) -> None:
        candidate_points = [(4, 5), (4, 6), (5, 5)]
        query_points = [(8, 3), (7, 3), (8, 4)]
        valid = np.ones((12, 12), dtype=bool)
        candidate = _grid(candidate_points).binary_map > 0
        query = _grid(query_points).status_map == VALID_HAS_DEFECT

        result = score_rigid_overlay(query, candidate, valid, score_mode="dice", angles=(0, 90), max_shift=1)

        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.dice, 1.0)
        self.assertEqual(result.iou, 1.0)
        self.assertEqual((result.angle_deg, result.shift_row, result.shift_col), (90, 1, -1))
        self.assertEqual(int(result.transformed_mask.sum()), len(candidate_points))

    def test_dice_and_iou_are_selectable(self) -> None:
        valid = np.ones((12, 12), dtype=bool)
        query = np.zeros((12, 12), dtype=bool)
        candidate = np.zeros((12, 12), dtype=bool)
        query[4, 4] = query[4, 5] = True
        candidate[4, 4] = True

        dice = score_rigid_overlay(query, candidate, valid, score_mode="dice", angles=(0,), max_shift=0)
        iou = score_rigid_overlay(query, candidate, valid, score_mode="iou", angles=(0,), max_shift=0)

        self.assertAlmostEqual(dice.score, 2.0 / 3.0)
        self.assertAlmostEqual(iou.score, 0.5)

    def test_explain_path_uses_overlay_without_tokens(self) -> None:
        candidate_points = [(4, 5), (4, 6), (5, 5)]
        query_points = [(8, 3), (7, 3), (8, 4)]
        explanation = explain_count_partial_match(
            _grid(query_points),
            _grid(candidate_points),
            small_map_match_mode="rigid-overlay",
            rigid_overlay_score="iou",
            rigid_overlay_max_shift=1,
        )

        self.assertEqual(explanation["result"].score, 1.0)
        self.assertEqual(explanation["rigid_overlay"]["score_mode"], "iou")
        self.assertEqual(explanation["rigid_overlay"]["angle_deg"], 90)
        self.assertEqual(explanation["wbm_tokens"], [])
        self.assertEqual(explanation["wdm_tokens"], [])

    def test_large_map_is_not_silently_scored_by_overlay(self) -> None:
        with self.assertRaisesRegex(ValueError, "short side <= 12"):
            explain_count_partial_match(
                _grid([(8, 8)], shape=(16, 16)),
                _grid([(8, 8)], shape=(16, 16)),
                small_map_match_mode="rigid-overlay",
            )

    def test_large_map_proposals_can_use_rigid_overlay(self) -> None:
        candidate_points = [(4, 5), (4, 6), (5, 5)]
        query_points = [(8, 3), (7, 3), (8, 4)]
        explanation = explain_count_partial_match(
            _grid(query_points, shape=(16, 16)),
            _grid(candidate_points, shape=(16, 16)),
            min_area=1,
            small_map_match_mode="proposal-rigid-overlay",
            rigid_overlay_score="dice",
            rigid_overlay_max_shift=0,
        )

        self.assertEqual(explanation["result"].score, 1.0)
        self.assertEqual(explanation["result"].wbm_tokens, 1)
        self.assertEqual(explanation["result"].wdm_tokens, 1)
        self.assertEqual(explanation["matches"][0]["overlay_angle_deg"], 90)
        self.assertEqual(explanation["rigid_overlay"]["scope"], "proposal")

    def test_small_map_rejects_proposal_rigid_overlay(self) -> None:
        with self.assertRaisesRegex(ValueError, "short side > 12"):
            explain_count_partial_match(
                _grid([(4, 4)]),
                _grid([(4, 4)]),
                small_map_match_mode="proposal-rigid-overlay",
            )


if __name__ == "__main__":
    unittest.main()
