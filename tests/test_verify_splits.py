import csv
import sys
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon"))
from verify_splits import MetricPatch, find_split_violations, verify_manifest


class VerifySplitsTests(unittest.TestCase):
    def test_cross_split_near_pair_is_reported_but_same_split_is_not(self):
        patches = [
            MetricPatch("train_a", "train", "R1", box(0, 0, 100, 100)),
            MetricPatch("train_b", "train", "R1", box(110, 0, 210, 100)),
            MetricPatch("val_near", "validation", "R1", box(300, 0, 400, 100)),
            MetricPatch("test_far", "test_rural", "R2", box(2000, 0, 2100, 100)),
        ]
        violations = find_split_violations(patches, buffer_m=500)
        pairs = {(item["patch_a"], item["patch_b"]) for item in violations}
        self.assertTrue(any({left, right} == {"train_a", "val_near"} for left, right in pairs))
        self.assertFalse(any({left, right} == {"train_a", "train_b"} for left, right in pairs))
        self.assertFalse(any("test_far" in pair for pair in pairs))

    def test_manifest_bounds_are_reprojected_and_violation_counted(self):
        fields = ["patch_id", "split", "region_id", "x_min", "y_min", "x_max", "y_max", "crs"]
        rows = [
            {"patch_id": "a", "split": "train", "region_id": "R1", "x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100, "crs": "EPSG:3857"},
            {"patch_id": "b", "split": "validation", "region_id": "R1", "x_min": 200, "y_min": 0, "x_max": 300, "y_max": 100, "crs": "EPSG:3857"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patches.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            report = verify_manifest(path, buffer_m=500)
        self.assertEqual(2, report["patch_count"])
        self.assertEqual(1, report["violation_count"])


if __name__ == "__main__":
    unittest.main()
