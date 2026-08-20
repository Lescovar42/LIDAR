from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))

from phase_common import (  # noqa: E402
    choose_best_threshold,
    confusion_from_arrays,
    metrics_from_counts,
    potential_redundancy,
    summarize_manifest,
    threshold_values,
)


class ThresholdTests(unittest.TestCase):
    def test_threshold_values_are_inclusive(self) -> None:
        self.assertEqual(threshold_values(0.30, 0.40, 0.05), [0.3, 0.35, 0.4])

    def test_ignore_pixels_do_not_enter_confusion(self) -> None:
        probability = np.asarray([[0.9, 0.7], [0.2, 0.8]], dtype=np.float32)
        target = np.asarray([[1, 0], [0, 255]], dtype=np.uint8)
        counts = confusion_from_arrays(probability, target, 0.5)
        self.assertEqual(counts, {"tp": 1, "fp": 1, "fn": 0, "tn": 1, "ignored": 1, "total": 4})
        metrics = metrics_from_counts(counts)
        self.assertAlmostEqual(float(metrics["precision"]), 0.5)
        self.assertAlmostEqual(float(metrics["recall"]), 1.0)
        self.assertAlmostEqual(float(metrics["predicted_positive_fraction"]), 2 / 3)

    def test_best_threshold_uses_dice_first(self) -> None:
        rows = [
            {"threshold": 0.4, "dice": 0.5, "iou": 0.3, "precision": 0.4, "recall": 0.8},
            {"threshold": 0.6, "dice": 0.6, "iou": 0.4, "precision": 0.7, "recall": 0.5},
        ]
        self.assertEqual(choose_best_threshold(rows)["threshold"], 0.6)


class DiversityTests(unittest.TestCase):
    def test_manifest_summary_and_redundancy(self) -> None:
        rows = [
            {
                "patch_id": "a",
                "split": "train",
                "tile_name": "tile1",
                "positive_fraction": "0.2",
                "patch_polygon_keys": "p1",
                "contains_positive_boundary": "true",
                "is_hard_negative": "false",
                "category": "positive_boundary",
                "coverage_class": "mixed_positive",
                "x_min": "0", "y_min": "0", "x_max": "10", "y_max": "10",
            },
            {
                "patch_id": "b",
                "split": "train",
                "tile_name": "tile1",
                "positive_fraction": "0.3",
                "patch_polygon_keys": "p1",
                "contains_positive_boundary": "true",
                "is_hard_negative": "false",
                "category": "positive_boundary",
                "coverage_class": "mixed_positive",
                "x_min": "2", "y_min": "0", "x_max": "12", "y_max": "10",
            },
            {
                "patch_id": "c",
                "split": "train",
                "tile_name": "tile2",
                "positive_fraction": "0",
                "patch_polygon_keys": "",
                "contains_positive_boundary": "false",
                "is_hard_negative": "true",
                "category": "negative",
                "coverage_class": "negative",
                "x_min": "100", "y_min": "100", "x_max": "110", "y_max": "110",
            },
        ]
        summary = summarize_manifest(rows)
        self.assertEqual(summary["train"]["patches"], 3)
        self.assertEqual(summary["train"]["unique_positive_polygon_keys"], 1)
        self.assertEqual(summary["train"]["hard_negative_patches"], 1)
        redundancy = potential_redundancy(rows, overlap_threshold=0.5)
        self.assertEqual(redundancy["same_polygon_pairs_overlap_fraction_ge_threshold"], 1)


if __name__ == "__main__":
    unittest.main()
