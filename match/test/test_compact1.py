"""Regression tests for the raw-pixel small-map compact1 proposal mode."""
from __future__ import annotations

import unittest

import numpy as np

from match.core.local_matching import explain_count_partial_match
from match.core.local_matching.proposal_retrieval import _extract_retrieval_arc_tokens
from match.core.local_matching.scoring import _ring_topology_similarity
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
    def test_outer_domain_keeps_radially_incompatible_nearby_components_separate(self) -> None:
        # The points are separated by one cell, so their one-cell dilations touch.
        # Their normalized radii differ by more than one grid cell, however, and
        # must therefore remain separate compact1 candidate groups.
        mask = np.zeros((7, 12), dtype=bool)
        mask[0, 0] = True
        mask[1, 2] = True
        valid_mask = np.ones_like(mask, dtype=bool)
        weights = mask.astype(np.float32)

        _, _, debug = _extract_retrieval_arc_tokens(
            mask,
            weights,
            valid_mask,
            source="wbm",
            min_area=1,
            edge_r_min=0.65,
            band_width=0.12,
            min_angular_coverage=0.0,
            max_angular_coverage=1.0,
            angular_bins=24,
            max_radial_std=1.0,
            allowed_gap_cells=1.0,
            max_gap_count=1,
            min_parent_fraction=0.0,
            raw_mask=mask,
            max_gap_ratio=1.0,
            max_merge_gap_count=1,
            min_band_width_cells=1.0,
            merge_dilation_radius=1,
            outer_component_grouping=True,
        )

        self.assertEqual(debug["band_area_before_cc"], 2)
        self.assertEqual(debug["band_cc_count"], 2)
        self.assertEqual(debug["band_cc_merged_count"], 2)
        self.assertEqual(len(debug["band_group_radial_centers"]), 2)
        self.assertGreater(
            abs(debug["band_group_radial_centers"][0] - debug["band_group_radial_centers"][1]),
            debug["effective_radial_band_width"],
        )

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

    def test_scale_gate_rejects_match_without_removing_proposals(self) -> None:
        points = [(1, 0), (2, 0), (4, 0), (5, 0)]
        explanation = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="compact1",
            min_area=5,
            top_k=4,
            min_scale_score=1.01,
        )

        self.assertEqual(len(explanation["wbm_tokens"]), 1)
        self.assertEqual(len(explanation["wdm_tokens"]), 1)
        self.assertEqual(explanation["matches"], [])
        self.assertEqual(explanation["result"].matched_tokens, 0)

    def test_ring_topology_separates_edge_arc_from_central_token(self) -> None:
        edge_arc = {
            "geometry_type": "ring_arc",
            "radial_distance_norm": 0.85,
            "radial_std": 0.03,
            "max_angular_run_coverage": 0.20,
            "max_gap_coverage": 0.50,
            "radial_band_width": 0.06,
        }
        central = {
            "geometry_type": "central",
            "radial_distance_norm": 0.10,
            "radial_std": 0.18,
            "max_angular_run_coverage": 0.04,
            "max_gap_coverage": 0.90,
            "radial_band_width": 0.30,
        }
        score, active = _ring_topology_similarity(edge_arc, central)
        self.assertTrue(active)
        self.assertLess(score, 0.45)


if __name__ == "__main__":
    unittest.main()
