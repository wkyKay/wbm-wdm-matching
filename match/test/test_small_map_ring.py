"""Regression tests for compact ring proposals on tiny wafer maps."""
from __future__ import annotations

import unittest

import numpy as np

from match.core.local_matching import explain_count_partial_match
from match.core.local_matching.proposal import _bridge_short_circular_gaps, _circular_arc_runs_with_gap_limits, _proposal_config
from match.core.models import BACKGROUND, GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


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


def _large_circular_grid(points: list[tuple[int, int]]) -> GridMaps:
    shape = (25, 25)
    center = np.array([12.0, 12.0], dtype=np.float32)
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    valid = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= 11.5 ** 2
    status_map = np.where(valid, VALID_NO_DEFECT, BACKGROUND).astype(np.uint8)
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

    def test_arc_runs_are_limited_by_angle_and_gap_count(self) -> None:
        occupied = np.zeros(72, dtype=bool)
        occupied[np.arange(0, 18)] = True
        accepted = _circular_arc_runs_with_gap_limits(
            occupied,
            max_gap_bins=1,
            max_gap_count=4,
            min_angular_coverage=30.0 / 360.0,
            max_angular_coverage=90.0 / 360.0,
        )
        self.assertEqual(len(accepted), 1)

        long_arc = np.zeros(72, dtype=bool)
        long_arc[np.arange(0, 36)] = True
        self.assertEqual(
            len(
                _circular_arc_runs_with_gap_limits(
                    long_arc,
                    max_gap_bins=1,
                    max_gap_count=4,
                    min_angular_coverage=30.0 / 360.0,
                    max_angular_coverage=1.0,
                )
            ),
            1,
        )

        too_many_gaps = np.zeros(72, dtype=bool)
        too_many_gaps[[0, 2, 4, 6, 8, 10, 12, 14]] = True
        self.assertEqual(
            _circular_arc_runs_with_gap_limits(
                too_many_gaps,
                max_gap_bins=1,
                max_gap_count=4,
                min_angular_coverage=30.0 / 360.0,
                max_angular_coverage=90.0 / 360.0,
            ),
            [],
        )

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

    def test_arc_mode_rejects_sparse_outer_ring(self) -> None:
        center = np.array([12.0, 12.0], dtype=np.float32)
        radius = 8.0
        points = set()
        for angle_deg in range(0, 360, 18):
            theta = np.deg2rad(angle_deg)
            row = int(round(center[0] + radius * np.sin(theta)))
            col = int(round(center[1] + radius * np.cos(theta)))
            points.add((row, col))
        grid = _large_circular_grid(sorted(points))

        compact = explain_count_partial_match(grid, grid, proposal_mode="compact", min_area=5)
        arc = explain_count_partial_match(grid, grid, proposal_mode="arc", min_area=5)

        self.assertTrue(any(token["geometry_type"] == "edge_ring" for token in compact["wbm_tokens"]))
        self.assertFalse(any(token["geometry_type"] == "edge_ring" for token in arc["wbm_tokens"]))
        self.assertFalse(any(token["geometry_type"] == "ring_arc" for token in arc["wbm_tokens"]))
        self.assertEqual(arc["proposal_debug"]["wbm"]["arc_detection"]["reason"], "no_valid_arc_runs")
        self.assertEqual(arc["proposal_debug"]["wbm"]["arc_detection"]["allowed_gap_cells"], 2.0)
        self.assertEqual(arc["proposal_debug"]["wbm"]["arc_detection"]["max_gap_count"], 1)

    def test_arc_mode_falls_back_to_full_component_detection(self) -> None:
        center = np.array([12.0, 12.0], dtype=np.float32)
        radius = 8.0
        points = {(row, col) for row in range(10, 13) for col in range(10, 13)}
        for angle_deg in range(0, 360, 18):
            theta = np.deg2rad(angle_deg)
            row = int(round(center[0] + radius * np.sin(theta)))
            col = int(round(center[1] + radius * np.cos(theta)))
            points.add((row, col))
        grid = _large_circular_grid(sorted(points))

        arc = explain_count_partial_match(grid, grid, proposal_mode="arc", min_area=5)

        self.assertFalse(any(token["geometry_type"] == "edge_ring" for token in arc["wbm_tokens"]))
        self.assertTrue(any(token["geometry_type"] in {"central", "blob"} for token in arc["wbm_tokens"]))
        self.assertFalse(arc["proposal_debug"]["wbm"]["accepted"])

    def test_arc_mode_detects_short_arc(self) -> None:
        center = np.array([12.0, 12.0], dtype=np.float32)
        radius = 10.0
        points = set()
        for row in range(25):
            for col in range(25):
                rel = np.array([row, col], dtype=np.float32) - center
                dist = float(np.linalg.norm(rel))
                angle = (np.degrees(np.arctan2(rel[0], rel[1])) + 360.0) % 360.0
                if abs(dist - radius) <= 0.45 and 0.0 <= angle <= 48.0:
                    points.add((row, col))
        grid = _large_circular_grid(sorted(points))

        arc = explain_count_partial_match(grid, grid, proposal_mode="arc", min_area=5)
        arc_tokens = [token for token in arc["wbm_tokens"] if token["geometry_type"] == "ring_arc"]

        self.assertFalse(arc["proposal_debug"]["wbm"]["accepted"])
        self.assertEqual(arc["proposal_debug"]["wbm"]["arc_detection"]["accepted_count"], 1)
        self.assertEqual(len(arc_tokens), 1)
        self.assertGreaterEqual(arc_tokens[0]["ring_contour_angular_coverage"], 30.0 / 360.0)

    def test_arc_mode_accepts_one_pixel_ring_break_without_counting_bridge_pixel(self) -> None:
        points = {
            (12, 20),
            (13, 20),
            (14, 20),
            (15, 19),
            (17, 17),
            (18, 16),
            (19, 15),
            (20, 14),
        }
        bridge_point = (16, 18)
        grid = _large_circular_grid(sorted(points))

        arc = explain_count_partial_match(
            grid,
            grid,
            proposal_mode="arc",
            min_area=5,
            ring_angular_bins=24,
            ring_band_width=0.20,
            ring_edge_r_min=0.50,
        )
        ring_tokens = [token for token in arc["wbm_tokens"] if token["geometry_type"] == "ring_arc"]

        self.assertEqual(len(ring_tokens), 1)
        self.assertEqual(ring_tokens[0]["ring_arc_allowed_gap_cells"], 2.0)
        self.assertTrue(ring_tokens[0]["arc_connected"])
        self.assertEqual(ring_tokens[0]["arc_connectivity_mode"], "pixel_gap")
        self.assertEqual(ring_tokens[0]["arc_raw_component_count"], 2)
        self.assertEqual(ring_tokens[0]["arc_bridge_pixel_count"], 1)
        self.assertEqual(ring_tokens[0]["arc_max_allowed_bridge_pixels"], 2)
        self.assertEqual(ring_tokens[0]["raw_point_count"], ring_tokens[0]["area"])
        self.assertTrue(set(ring_tokens[0]["pixels"]).issubset(points))
        self.assertNotIn(bridge_point, set(ring_tokens[0]["pixels"]))
        self.assertGreaterEqual(ring_tokens[0]["ring_contour_area"], ring_tokens[0]["raw_point_count"])

    def test_arc_mode_rejects_ring_slice_inside_large_connected_region(self) -> None:
        points = {
            (12, 20),
            (13, 20),
            (14, 20),
            (15, 19),
            (17, 17),
            (18, 16),
            (19, 15),
            (20, 14),
        }
        for row in range(10, 21):
            for col in range(10, 21):
                points.add((row, col))
        grid = _large_circular_grid(sorted(points))

        arc = explain_count_partial_match(
            grid,
            grid,
            proposal_mode="arc",
            min_area=5,
            ring_angular_bins=24,
            ring_band_width=0.20,
            ring_edge_r_min=0.50,
        )

        self.assertFalse(any(token["geometry_type"] == "ring_arc" for token in arc["wbm_tokens"]))

    def test_explicit_min_area_and_top_k_are_not_adaptively_capped(self) -> None:
        default_cfg = _proposal_config((12, 12), valid_area=144, min_area=5, top_k=6)
        explicit_cfg = _proposal_config((12, 12), valid_area=144, min_area=20, top_k=10)

        self.assertEqual(default_cfg.min_area, 2)
        self.assertEqual(default_cfg.top_k, 4)
        self.assertEqual(explicit_cfg.min_area, 20)
        self.assertEqual(explicit_cfg.top_k, 10)

    def test_arc_mode_uses_compact_ring_defaults(self) -> None:
        compact_small = _proposal_config((12, 12), valid_area=144, min_area=5, top_k=6, proposal_mode="compact")
        compact_large = _proposal_config((25, 25), valid_area=441, min_area=5, top_k=6, proposal_mode="compact")
        arc_small = _proposal_config((12, 12), valid_area=144, min_area=5, top_k=6, proposal_mode="arc")
        arc_large = _proposal_config((25, 25), valid_area=441, min_area=5, top_k=6, proposal_mode="arc")

        self.assertAlmostEqual(compact_small.ring_min_angular_coverage, 0.10)
        self.assertAlmostEqual(compact_large.ring_min_angular_coverage, 0.16)
        self.assertAlmostEqual(arc_small.ring_min_angular_coverage, compact_small.ring_min_angular_coverage)
        self.assertAlmostEqual(arc_large.ring_min_angular_coverage, compact_large.ring_min_angular_coverage)


if __name__ == "__main__":
    unittest.main()
