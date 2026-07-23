"""Regression tests for hard scale gating in token matching."""
from __future__ import annotations

import unittest

import numpy as np

from match.core.local_matching.scoring import _token_match_components


def _token(area: int, pca_lambda1: float, pca_lambda2: float) -> dict:
    return {
        "area": area,
        "support_area_ratio": area / 1000.0,
        "pca_lambda1": pca_lambda1,
        "pca_lambda2": pca_lambda2,
        "pos": np.array([0.5, 0.5], dtype=np.float32),
        "descriptor_parts": {
            "kind": "zernike_geometry",
            "moment": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "geometry": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            "moment_weight": 0.75,
            "geometry_weight": 0.25,
        },
    }


class ScaleThresholdTest(unittest.TestCase):
    def test_scale_ratio_gate_rejects_excessive_size_mismatch(self) -> None:
        query = _token(area=9, pca_lambda1=1.0, pca_lambda2=1.0)
        candidate = _token(area=49, pca_lambda1=16.0, pca_lambda2=16.0)

        without_gate = _token_match_components(
            query,
            candidate,
            sigma_pos=0.35,
            sigma_scale=1.5,
            score_weights=(0.60, 0.25, 0.15),
            scale_component_weights=(0.30, 0.70),
            scale_ratio_min=0.0,
        )
        with_gate = _token_match_components(
            query,
            candidate,
            sigma_pos=0.35,
            sigma_scale=1.5,
            score_weights=(0.60, 0.25, 0.15),
            scale_component_weights=(0.30, 0.70),
            scale_ratio_min=0.35,
        )

        self.assertGreater(without_gate["score"], 0.0)
        self.assertEqual(with_gate["score"], 0.0)
        self.assertLess(with_gate["scale_area_ratio"], 0.35)


if __name__ == "__main__":
    unittest.main()
