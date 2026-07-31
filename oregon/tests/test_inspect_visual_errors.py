from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "diagnostics"
    / "inspect_visual_errors.py"
)
SPEC = importlib.util.spec_from_file_location("inspect_visual_errors", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VisualErrorInspectionTests(unittest.TestCase):
    def test_confusion_counts_ignore_aware(self) -> None:
        probability = np.array(
            [[0.9, 0.8, 0.1], [0.2, 0.7, 0.4]],
            dtype=np.float32,
        )
        target = np.array(
            [[1, 0, 0], [1, 255, 0]],
            dtype=np.uint8,
        )
        counts = MODULE.confusion_counts(probability, target, 0.5)
        self.assertEqual(
            counts,
            {"tp": 1, "fp": 1, "fn": 1, "tn": 2, "ignored": 1, "total": 6},
        )
        metrics = MODULE.metrics_from_counts(counts)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["ignore_fraction"], 1 / 6)

    def test_representative_selection_rejects_ignore_heavy_best(self) -> None:
        records = [
            {
                "patch_id": "ignore_heavy",
                "tile_name": "A",
                "dice": 0.99,
                "ignore_fraction": 0.95,
                "valid_pixels": 10,
                "gt_positive_fraction_valid": 0.5,
                "false_positive_fraction_valid": 0.0,
                "false_negative_fraction_valid": 0.0,
                "contains_positive_boundary": "true",
            },
            {
                "patch_id": "good",
                "tile_name": "B",
                "dice": 0.8,
                "ignore_fraction": 0.1,
                "valid_pixels": 100,
                "gt_positive_fraction_valid": 0.4,
                "false_positive_fraction_valid": 0.05,
                "false_negative_fraction_valid": 0.05,
                "contains_positive_boundary": "true",
            },
            {
                "patch_id": "median",
                "tile_name": "C",
                "dice": 0.4,
                "ignore_fraction": 0.2,
                "valid_pixels": 100,
                "gt_positive_fraction_valid": 0.3,
                "false_positive_fraction_valid": 0.1,
                "false_negative_fraction_valid": 0.1,
                "contains_positive_boundary": "false",
            },
            {
                "patch_id": "poor",
                "tile_name": "D",
                "dice": 0.1,
                "ignore_fraction": 0.1,
                "valid_pixels": 100,
                "gt_positive_fraction_valid": 0.3,
                "false_positive_fraction_valid": 0.1,
                "false_negative_fraction_valid": 0.25,
                "contains_positive_boundary": "true",
            },
            {
                "patch_id": "negative_fp",
                "tile_name": "E",
                "dice": 0.0,
                "ignore_fraction": 0.1,
                "valid_pixels": 100,
                "gt_positive_fraction_valid": 0.0,
                "false_positive_fraction_valid": 0.8,
                "false_negative_fraction_valid": 0.0,
                "contains_positive_boundary": "false",
            },
        ]
        selected, used_limit = MODULE.select_representative_records(
            records,
            max_ignore_fraction=0.30,
        )
        ids = {record["patch_id"] for record in selected}
        self.assertNotIn("ignore_heavy", ids)
        self.assertIn("negative_fp", ids)
        self.assertLessEqual(used_limit, 0.30)

    def test_dominant_error(self) -> None:
        self.assertEqual(
            MODULE.dominant_error({"tp": 2, "fp": 20, "fn": 2}),
            "false_positive_dominant",
        )
        self.assertEqual(
            MODULE.dominant_error({"tp": 2, "fp": 1, "fn": 20}),
            "false_negative_dominant",
        )
        self.assertEqual(
            MODULE.dominant_error({"tp": 0, "fp": 10, "fn": 0}),
            "false_positive_on_negative",
        )


if __name__ == "__main__":
    unittest.main()
