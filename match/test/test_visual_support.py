"""Regression tests for proposal support visualization."""
from __future__ import annotations

import unittest

import numpy as np

from match.core.models import VALID_NO_DEFECT
from match.viz.count_partial_visualization import _cluster_color_image


class VisualSupportTest(unittest.TestCase):
    def test_kde_support_pixels_are_colored_even_when_not_raw_pixels(self) -> None:
        status = np.full((5, 5), VALID_NO_DEFECT, dtype=np.uint8)
        token = {
            "pixels": [(2, 2)],
            "kde_support_pixels": [(2, 1), (2, 2), (2, 3)],
        }

        image = _cluster_color_image(status, [token], source="wdm", count_map=np.zeros((5, 5), dtype=np.float32))

        self.assertFalse(np.allclose(image[2, 1], image[0, 0]))
        self.assertFalse(np.allclose(image[2, 3], image[0, 0]))
        self.assertFalse(np.allclose(image[2, 2], image[2, 1]))


if __name__ == "__main__":
    unittest.main()
