import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from oregon.train_baseline import (
    compute_pos_weight,
    confusion_counts,
    masked_bce_with_logits,
    metrics_from_counts,
    segmentation_loss,
    select_training_rows,
    select_validation_rows,
    soft_dice_loss,
    validate_training_region_roles,
)


class IgnoreAwareTrainingTests(unittest.TestCase):
    def test_loss_exactly_excludes_ignored_pixels(self) -> None:
        logits = torch.tensor([[[[0.0, 0.0, 100.0]]]])
        target = torch.tensor([[[[1.0, 0.0, 255.0]]]])

        self.assertAlmostEqual(masked_bce_with_logits(logits, target).item(), math.log(2.0), places=6)
        self.assertAlmostEqual(soft_dice_loss(logits, target).item(), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(
            segmentation_loss(logits, target).item(), math.log(2.0) + 1.0 / 3.0, places=6
        )

        changed_ignored_logit = logits.clone()
        changed_ignored_logit[..., 2] = -100.0
        self.assertAlmostEqual(
            segmentation_loss(logits, target).item(),
            segmentation_loss(changed_ignored_logit, target).item(),
            places=6,
        )

    def test_all_metrics_and_counts_exclude_ignored_pixels(self) -> None:
        logits = torch.tensor([[[[10.0, -10.0, -10.0, 10.0]]]])
        target = torch.tensor([[[[1.0, 0.0, 255.0, 255.0]]]])

        counts = confusion_counts(logits, target)
        self.assertEqual(counts, {"tp": 1, "fp": 0, "fn": 0, "tn": 1, "ignored": 2, "total": 4})
        metrics = metrics_from_counts(counts)
        for name in ("dice", "iou", "precision", "recall", "specificity"):
            self.assertEqual(metrics[name], 1.0)
        self.assertEqual(metrics["ignore_fraction"], 0.5)

    def test_class_weight_excludes_ignored_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "patch.npz", mask=np.array([[1, 0, 0, 255]], dtype=np.uint8))
            weight = compute_pos_weight(root, [{"patch_path": "patch.npz"}])
        self.assertEqual(weight, 2.0)


class RegionRoleGuardTests(unittest.TestCase):
    def test_accepts_one_train_val_region(self) -> None:
        rows = [
            {"region_id": "R1", "region_role": "train_val"},
            {"region_id": "R1", "region_role": "train_val"},
        ]
        self.assertEqual(validate_training_region_roles(rows), {"R1": "train_val"})

    def test_rejects_test_role_in_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "train_val"):
            validate_training_region_roles([{"region_id": "R2", "region_role": "test_rural"}])

    def test_rejects_training_spanning_multiple_roles(self) -> None:
        rows = [
            {"region_id": "R1", "region_role": "train_val"},
            {"region_id": "R2", "region_role": "test_rural"},
        ]
        with self.assertRaisesRegex(ValueError, "more than one region role"):
            validate_training_region_roles(rows)

    def test_rejects_multiple_train_val_regions_without_selection(self) -> None:
        rows = [
            {"region_id": "R1", "region_role": "train_val"},
            {"region_id": "R4", "region_role": "train_val"},
        ]
        with self.assertRaisesRegex(ValueError, "--training-region"):
            validate_training_region_roles(rows)

    def test_explicit_region_prevents_r4_joining_r1(self) -> None:
        rows = [
            {
                "split": "train", "patch_path": "r1.npz", "region_id": "R1",
                "region_role": "train_val", "qc_status": "accept",
            },
            {
                "split": "train", "patch_path": "r4.npz", "region_id": "R4",
                "region_role": "train_val", "qc_status": "accept",
            },
        ]
        selected, training_regions = select_training_rows(
            rows, require_qc=True, training_region="R1"
        )
        self.assertEqual([row["patch_path"] for row in selected], ["r1.npz"])
        self.assertEqual(training_regions, {"R1": "train_val"})

    def test_rejects_unknown_explicit_training_region(self) -> None:
        rows = [
            {"split": "train", "region_id": "R1", "region_role": "train_val"},
            {"split": "train", "region_id": "R4", "region_role": "train_val"},
        ]
        with self.assertRaisesRegex(ValueError, "has no training rows"):
            select_training_rows(rows, require_qc=False, training_region="R9")

    def test_role_guard_runs_before_qc_filtering(self) -> None:
        rows = [
            {
                "split": "train", "region_id": "R1", "region_role": "train_val",
                "qc_status": "accept",
            },
            {
                "split": "train", "region_id": "R2", "region_role": "test_rural",
                "qc_status": "reject_misaligned",
            },
        ]
        with self.assertRaisesRegex(ValueError, "more than one region role"):
            select_training_rows(rows, require_qc=True, training_region="R1")

    def test_validation_uses_only_selected_train_val_region(self) -> None:
        rows = [
            {
                "split": "validation", "patch_path": "r1.npz", "region_id": "R1",
                "region_role": "train_val", "qc_status": "accept",
            },
            {
                "split": "validation", "patch_path": "r1-rejected.npz", "region_id": "R1",
                "region_role": "train_val", "qc_status": "reject_misaligned",
            },
            {
                "split": "validation", "patch_path": "r4.npz", "region_id": "R4",
                "region_role": "train_val", "qc_status": "accept",
            },
        ]
        selected, excluded = select_validation_rows(rows, "R1", require_qc=True)
        self.assertEqual([row["patch_path"] for row in selected], ["r1.npz"])
        self.assertEqual(excluded, ["R4"])

    def test_validation_role_guard_runs_before_qc_filtering(self) -> None:
        rows = [
            {
                "split": "validation", "region_id": "R1", "region_role": "test_rural",
                "qc_status": "reject_misaligned",
            }
        ]
        with self.assertRaisesRegex(ValueError, "region_role='train_val'"):
            select_validation_rows(rows, "R1", require_qc=True)


if __name__ == "__main__":
    unittest.main()
