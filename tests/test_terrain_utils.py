import sys
import time
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import generic_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon"))
from terrain_utils import compute_tri


def reference_tri(dem):
    def kernel(window):
        center = window[len(window) // 2]
        return float(np.sqrt(np.mean((window - center) ** 2)))
    return generic_filter(dem, kernel, size=3, mode="nearest").astype(np.float32)


class TerrainTests(unittest.TestCase):
    def test_vectorized_tri_matches_reference_and_reports_speedup(self):
        rng = np.random.default_rng(42)
        dem = rng.normal(500, 30, size=(256, 256)).astype(np.float32)
        start = time.perf_counter()
        expected = reference_tri(dem)
        reference_seconds = time.perf_counter() - start
        start = time.perf_counter()
        actual = compute_tri(dem)
        vectorized_seconds = time.perf_counter() - start
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
        speedup = reference_seconds / max(vectorized_seconds, 1e-12)
        print(f"TRI speedup: {speedup:.1f}x")
        self.assertGreater(speedup, 1.0)


if __name__ == "__main__":
    unittest.main()
