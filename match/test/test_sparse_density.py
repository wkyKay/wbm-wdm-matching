"""Focused regression tests for sparse-density WBM/WDM proposal generation."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
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
        self.assertTrue(set(points).issubset(set(token["pixels"])))
        self.assertGreaterEqual(token["area"], len(points))
        self.assertEqual(token["support_area"], token["area"])
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

    def test_density_arc_ring_residual_uses_support_geometry_and_disjoint_raw_evidence(self) -> None:
        center = np.array([12.5, 12.5])
        arc_points = sorted({
            (int(round(center[0] + 11 * np.sin(angle))), int(round(center[1] + 11 * np.cos(angle))))
            for angle in np.linspace(0.0, np.pi / 2.0, 18)
        })
        residual_points = [(7, 7), (7, 8), (8, 7), (8, 8)]
        points = arc_points + residual_points
        result = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="sparse-density-arc-ring-residual",
            density_sigmas=(1.2,),
            density_threshold=0.12,
            density_min_raw_points=3,
            density_min_raw_mass=3.0,
            min_area=3,
            ring_min_angular_coverage=0.08,
            sparse_support_contact_ratio_max=1.0,
        )

        tokens = result["wbm_tokens"]
        arc = next(token for token in tokens if token["proposal_type"] == "ring_arc_band")
        residual = next(token for token in tokens if token["proposal_type"] == "density_residual")
        self.assertGreater(arc["area"], arc["raw_area"])
        self.assertEqual(arc["area"], arc["kde_support_area"])
        self.assertTrue(set(arc["raw_pixels"]).isdisjoint(residual["raw_pixels"]))
        self.assertEqual(set(arc["raw_pixels"]) | set(residual["raw_pixels"]), set(points))

        top_one = explain_count_partial_match(
            _grid(points),
            _grid(points),
            proposal_mode="sparse-density-arc-ring-residual",
            density_sigmas=(1.2,),
            density_threshold=0.12,
            density_min_raw_points=3,
            density_min_raw_mass=3.0,
            min_area=3,
            top_k=1,
            ring_min_angular_coverage=0.08,
            sparse_support_contact_ratio_max=1.0,
        )
        self.assertEqual(len(top_one["wbm_tokens"]), 1)
        self.assertEqual(top_one["wbm_tokens"][0]["proposal_type"], "ring_arc_band")
        self.assertTrue(top_one["wbm_tokens"][0]["raw_pixels"])

    def test_existing_proposal_modes_still_run(self) -> None:
        points = [(5, 5), (5, 6), (6, 5), (6, 6), (12, 20), (13, 20), (14, 20), (15, 20)]
        for mode in (
            "cc",
            "compact",
            "arc-ring-residual",
            "sparse-density",
        ):
            with self.subTest(proposal_mode=mode):
                result = explain_count_partial_match(
                    _grid(points),
                    _grid(points, count=2.0),
                    proposal_mode=mode,
                    min_area=3,
                    density_sigmas=(1.2,),
                    density_threshold=0.12,
                    density_min_raw_points=3,
                    density_min_raw_mass=3.0,
                )
                self.assertTrue(result["wbm_tokens"])
                self.assertTrue(result["wdm_tokens"])

    def test_sparse_modes_write_review_figures(self) -> None:
        from matplotlib import pyplot as plt
        from match.viz.count_partial_visualization import (
            _cluster_color_image,
            plot_count_partial_steps,
            plot_count_partial_topk,
        )

        center = np.array([12.5, 12.5])
        arc_points = sorted({
            (int(round(center[0] + 11 * np.sin(angle))), int(round(center[1] + 11 * np.cos(angle))))
            for angle in np.linspace(0.0, np.pi / 2.0, 18)
        })
        points = arc_points + [(7, 7), (7, 8), (8, 7), (8, 8)]
        explanation = explain_count_partial_match(
            _grid(points),
            _grid(points, count=2.0),
            proposal_mode="sparse-density-arc-ring-residual",
            density_sigmas=(1.2,),
            density_threshold=0.12,
            density_min_raw_points=3,
            density_min_raw_mass=3.0,
            min_area=3,
            ring_min_angular_coverage=0.08,
        )
        plain = _cluster_color_image(_grid(points).status_map, explanation["wbm_tokens"], source="wbm")
        with_support = _cluster_color_image(
            _grid(points).status_map,
            explanation["wbm_tokens"],
            source="wbm",
            show_kde_support=True,
        )
        self.assertFalse(np.array_equal(plain, with_support))

        with TemporaryDirectory() as temp_dir:
            reference = _grid(points)
            candidate = _grid(points, count=2.0)
            for mode in ("sparse-density", "sparse-density-arc-ring-residual"):
                with self.subTest(proposal_mode=mode):
                    steps_path = Path(temp_dir) / f"{mode}_steps.png"
                    figure, axes = plot_count_partial_steps(
                        reference,
                        candidate,
                        proposal_mode=mode,
                        density_sigmas=(1.2,),
                        density_threshold=0.12,
                        density_min_raw_points=3,
                        density_min_raw_mass=3.0,
                        min_area=3,
                        ring_min_angular_coverage=0.08,
                        save_path=steps_path,
                    )
                    self.assertEqual(len(axes), 5)
                    self.assertTrue(steps_path.is_file())
                    self.assertGreater(steps_path.stat().st_size, 10_000)
                    plt.close(figure)

                    topk_path = Path(temp_dir) / f"{mode}_topk.png"
                    figure, _axes = plot_count_partial_topk(
                        reference,
                        [("candidate", candidate)],
                        proposal_mode=mode,
                        density_sigmas=(1.2,),
                        density_threshold=0.12,
                        density_min_raw_points=3,
                        density_min_raw_mass=3.0,
                        min_area=3,
                        ring_min_angular_coverage=0.08,
                        save_path=topk_path,
                    )
                    self.assertTrue(topk_path.is_file())
                    self.assertGreater(topk_path.stat().st_size, 10_000)
                    plt.close(figure)


if __name__ == "__main__":
    unittest.main()
