from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1] / "diagnostics"
if str(DIAGNOSTICS_DIR) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_DIR))

MODULE_PATH = DIAGNOSTICS_DIR / "export_prediction_maps.py"
SPEC = importlib.util.spec_from_file_location(
    "export_prediction_maps",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import inspect_visual_errors as ive


def make_result(
    *,
    patch_id: str,
    row_offset: int,
    col_offset: int,
    target: np.ndarray,
    probability: np.ndarray,
) -> ive.PatchResult:
    return ive.PatchResult(
        index=0,
        row={
            "patch_id": patch_id,
            "tile_name": "tile_a",
            "row_offset": str(row_offset),
            "col_offset": str(col_offset),
            "x_min": str(col_offset),
            "x_max": str(col_offset + target.shape[1]),
            "y_min": str(-(row_offset + target.shape[0])),
            "y_max": str(-row_offset),
            "crs": "EPSG:3857",
        },
        target=target.astype(np.uint8),
        probabilities=probability.astype(np.float32),
        raw_features=np.zeros(
            (7, target.shape[0], target.shape[1]),
            dtype=np.float32,
        ),
        selected_counts={},
        selected_metrics={},
    )


class PredictionMapExportTests(unittest.TestCase):
    def test_tile_mosaic_averages_overlap(self) -> None:
        first = make_result(
            patch_id="a",
            row_offset=0,
            col_offset=0,
            target=np.array([[1, 0], [0, 0]]),
            probability=np.array([[0.9, 0.2], [0.1, 0.2]]),
        )
        second = make_result(
            patch_id="b",
            row_offset=0,
            col_offset=1,
            target=np.array([[0, 1], [0, 0]]),
            probability=np.array([[0.4, 0.8], [0.6, 0.3]]),
        )

        mosaic = MODULE.build_tile_mosaic(
            "tile_a",
            [first, second],
            max_mosaic_pixels=100,
        )

        self.assertEqual(mosaic.target.shape, (2, 3))
        self.assertAlmostEqual(
            float(mosaic.probability[0, 1]),
            0.3,
            places=6,
        )
        self.assertEqual(int(mosaic.source_count[0, 1]), 2)
        self.assertFalse(bool(mosaic.conflict_mask[0, 1]))
        self.assertEqual(int(mosaic.target[0, 0]), 1)
        self.assertEqual(int(mosaic.target[0, 2]), 1)

    def test_tile_mosaic_detects_label_conflict(self) -> None:
        first = make_result(
            patch_id="a",
            row_offset=0,
            col_offset=0,
            target=np.array([[1, 0], [0, 0]]),
            probability=np.zeros((2, 2)),
        )
        second = make_result(
            patch_id="b",
            row_offset=0,
            col_offset=1,
            target=np.array([[1, 0], [0, 0]]),
            probability=np.zeros((2, 2)),
        )

        mosaic = MODULE.build_tile_mosaic(
            "tile_a",
            [first, second],
            max_mosaic_pixels=100,
        )
        self.assertTrue(bool(mosaic.conflict_mask[0, 1]))

    def test_error_rgb_semantics(self) -> None:
        target = np.array([[1, 0], [1, 255]], dtype=np.uint8)
        probability = np.array(
            [[0.9, 0.9], [0.1, 0.9]],
            dtype=np.float32,
        )
        rgb = MODULE.error_rgb(probability, target, 0.5)
        np.testing.assert_allclose(rgb[0, 0], (1.0, 1.0, 0.0))
        np.testing.assert_allclose(rgb[0, 1], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(rgb[1, 0], (0.0, 1.0, 0.0))
        np.testing.assert_allclose(rgb[1, 1], (0.45, 0.45, 0.45))


if __name__ == "__main__":
    unittest.main()
