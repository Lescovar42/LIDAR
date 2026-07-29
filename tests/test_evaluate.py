import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from oregon.evaluate import (
    aggregate_region_metrics,
    load_normalization,
    select_overlay_records,
    validate_checkpoint_normalization,
    valid_pixel_error_rate,
)
from oregon.train_baseline import EXPECTED_CHANNELS, PatchDataset


class PersistedNormalizationTests(unittest.TestCase):
    def test_evaluation_loads_persisted_normalization_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                name: {"mean": float(index + 1), "std": float(index + 2)}
                for index, name in enumerate(EXPECTED_CHANNELS)
            }
            normalization_path = root / "normalization.json"
            normalization_path.write_text(json.dumps(payload), encoding="utf-8")
            mean, std = load_normalization(normalization_path, list(EXPECTED_CHANNELS))

            features = np.stack([
                np.full((2, 2), mean[index] + 2.0 * std[index], dtype=np.float32)
                for index in range(len(EXPECTED_CHANNELS))
            ])
            np.savez(root / "patch.npz", features=features, mask=np.zeros((2, 2), dtype=np.uint8))
            dataset = PatchDataset(root, [{"patch_path": "patch.npz"}], mean, std)
            normalized, _ = dataset[0]

        np.testing.assert_allclose(mean, np.arange(1, 8, dtype=np.float32))
        np.testing.assert_allclose(std, np.arange(2, 9, dtype=np.float32))
        np.testing.assert_allclose(normalized.numpy(), 2.0)

    def test_normalization_must_match_checkpoint_fingerprint(self) -> None:
        mean = np.arange(1, 8, dtype=np.float32)
        std = np.arange(2, 9, dtype=np.float32)
        validate_checkpoint_normalization({"mean": mean, "std": std}, mean, std)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_checkpoint_normalization({"mean": mean + 1, "std": std}, mean, std)


class PerRegionAggregationTests(unittest.TestCase):
    def test_matches_hand_computed_pixel_aggregation(self) -> None:
        records = [
            {"region_id": "R1", "tp": 1, "fp": 1, "fn": 0, "tn": 2, "ignored": 1, "total": 5, "loss": 0.2},
            {"region_id": "R1", "tp": 1, "fp": 0, "fn": 1, "tn": 0, "ignored": 0, "total": 2, "loss": 0.4},
            {"region_id": "R2", "tp": 0, "fp": 1, "fn": 1, "tn": 1, "ignored": 1, "total": 4, "loss": 0.5},
        ]

        result = aggregate_region_metrics(records)
        r1 = result["R1"]
        self.assertEqual(r1["patches"], 2)
        self.assertEqual(r1["valid_pixels"], 6)
        self.assertEqual((r1["tp"], r1["fp"], r1["fn"], r1["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(r1["dice"], 2.0 / 3.0)
        self.assertAlmostEqual(r1["iou"], 0.5)
        self.assertAlmostEqual(r1["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(r1["recall"], 2.0 / 3.0)
        self.assertAlmostEqual(r1["specificity"], 2.0 / 3.0)
        self.assertAlmostEqual(r1["ignore_fraction"], 1.0 / 7.0)
        self.assertAlmostEqual(r1["loss"], (0.2 * 4 + 0.4 * 2) / 6)

        r2 = result["R2"]
        self.assertEqual(r2["patches"], 1)
        self.assertEqual(r2["valid_pixels"], 3)
        self.assertEqual(r2["dice"], 0.0)
        self.assertEqual(r2["iou"], 0.0)
        self.assertEqual(r2["precision"], 0.0)
        self.assertEqual(r2["recall"], 0.0)
        self.assertEqual(r2["specificity"], 0.5)
        self.assertEqual(r2["ignore_fraction"], 0.25)


class OverlaySelectionTests(unittest.TestCase):
    @staticmethod
    def record(
        patch_id: str,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
        tn: int = 0,
        loss: float = 0.0,
        dice: float = 0.0,
    ) -> dict[str, float | int | str]:
        return {
            "patch_id": patch_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "loss": loss,
            "dice": dice,
        }

    def test_all_negative_best_is_perfect_and_worst_is_false_positive(self) -> None:
        perfect_negative = self.record("perfect-negative", tn=100, loss=0.01)
        false_positive = self.record("false-positive", fp=20, tn=80, loss=0.4)

        best, worst = select_overlay_records([false_positive, perfect_negative])

        self.assertEqual(best["patch_id"], "perfect-negative")
        self.assertEqual(worst["patch_id"], "false-positive")
        self.assertEqual(valid_pixel_error_rate(perfect_negative), 0.0)
        self.assertEqual(valid_pixel_error_rate(false_positive), 0.2)

    def test_positive_support_is_preferred_for_best_and_fn_is_worst(self) -> None:
        perfect_negative = self.record("perfect-negative", tn=100, loss=0.01)
        false_negative = self.record("false-negative", fn=25, tn=75, loss=0.5)

        best, worst = select_overlay_records([perfect_negative, false_negative])

        self.assertEqual(best["patch_id"], "false-negative")
        self.assertEqual(worst["patch_id"], "false-negative")

    def test_error_then_loss_then_patch_id_determine_worst(self) -> None:
        records = [
            self.record("z-patch", fp=10, tn=90, loss=0.3),
            self.record("b-patch", fn=10, tn=90, loss=0.4),
            self.record("a-patch", fn=10, tn=90, loss=0.4),
        ]

        _, worst = select_overlay_records(records)

        self.assertEqual(worst["patch_id"], "a-patch")


if __name__ == "__main__":
    unittest.main()
