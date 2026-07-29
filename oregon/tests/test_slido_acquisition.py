from __future__ import annotations

import sys
import unittest
from pathlib import Path

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from acquire_slido import annotate_and_count, enforce_feature_cap
from slido_utils import normalize_confidence


class ConfidenceNormalizationTests(unittest.TestCase):
    def test_dirty_live_values(self) -> None:
        expected = {
            "High (=>30)": "high",
            "High (>30)": "high",
            "High (=<30)": "high",
            "Moderate (11-29) ": "moderate",
            " Low ": "low",
            None: "unknown",
            "": "unknown",
            "not assessed": "unknown",
        }
        for raw, confidence_class in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_confidence(raw), confidence_class)

    def test_annotation_and_count_summaries(self) -> None:
        features = [
            {"properties": {"CONFIDENCE": "High (>30)", "REF_ID_COD": "source-a"}},
            {"properties": {"confidence": "Moderate (11-29) ", "ref_id_cod": "source-a"}},
            {"properties": {"CONFIDENCE": None, "REF_ID_COD": None}},
        ]
        confidence, sources = annotate_and_count(features)
        self.assertEqual(confidence, {"high": 1, "moderate": 1, "low": 0, "unknown": 1})
        self.assertEqual(sources, {"source-a": 2, "unknown": 1})
        self.assertEqual(features[0]["properties"]["confidence_class"], "high")

    def test_equal_to_feature_cap_is_hard_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "may be truncated"):
            enforce_feature_cap(5000, 5000)
        enforce_feature_cap(4999, 5000)
        enforce_feature_cap(999999, 0)


if __name__ == "__main__":
    unittest.main()
