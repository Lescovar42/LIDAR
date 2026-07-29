from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from diagnostics.qc_patch_viewer import slido_mask_visuals


class SlidoMaskVisualTests(unittest.TestCase):
    def test_three_state_mask_separates_positive_and_ignore_pixels(self) -> None:
        mask = np.array([[0, 1], [255, 1]], dtype=np.uint8)

        overlay, positive_contour = slido_mask_visuals(mask)

        np.testing.assert_array_equal(
            positive_contour,
            np.array([[0, 1], [0, 1]], dtype=np.uint8),
        )
        np.testing.assert_allclose(overlay[0, 0], (0.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(overlay[0, 1], (1.0, 0.0, 0.0, 0.5))
        np.testing.assert_allclose(overlay[1, 0], (0.1, 0.35, 1.0, 0.5))
        self.assertFalse(np.array_equal(overlay[0, 1], overlay[1, 0]))

    def test_binary_mask_keeps_existing_positive_semantics(self) -> None:
        mask = np.array([[False, True], [True, False]])

        overlay, positive_contour = slido_mask_visuals(mask)

        np.testing.assert_array_equal(positive_contour, mask.astype(np.uint8))
        self.assertTrue(np.all(overlay[mask, 0] == 1.0))
        self.assertTrue(np.all(overlay[~mask, 3] == 0.0))


if __name__ == "__main__":
    unittest.main()
