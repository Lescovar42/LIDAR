from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from fetch_naip_qc import compute_year_gap, stratified_sample_rows


class YearGapTests(unittest.TestCase):
    def test_computes_signed_gap_and_flags_only_over_two_years(self) -> None:
        self.assertEqual(compute_year_gap(2019, 2022), (3, True))
        self.assertEqual(compute_year_gap(2022, 2019), (-3, True))
        self.assertEqual(compute_year_gap(2019, 2021), (2, False))
        self.assertEqual(compute_year_gap(2019, 2017), (-2, False))

    def test_missing_or_invalid_year_has_no_gap_or_flag(self) -> None:
        self.assertEqual(compute_year_gap("", 2022), (None, None))
        self.assertEqual(compute_year_gap(2019, "unknown"), (None, None))


class StratifiedSamplingTests(unittest.TestCase):
    @staticmethod
    def fixture_rows() -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for region in ("R1", "R2"):
            for category in (
                "negative",
                "positive_boundary",
                "positive_interior",
            ):
                for number in range(6):
                    rows.append(
                        {
                            "patch_id": f"{region}-{category}-{number}",
                            "region_id": region,
                            "category": category,
                            "tile_name": f"tile-{number % 2}",
                            "row_offset": str(number),
                            "col_offset": str(number * 2),
                        }
                    )
        return rows

    def test_sampling_is_seeded_deterministic_and_order_independent(self) -> None:
        rows = self.fixture_rows()
        first = stratified_sample_rows(rows, sample_per_region=8, seed=123)
        repeated = stratified_sample_rows(rows, sample_per_region=8, seed=123)
        reversed_input = stratified_sample_rows(
            list(reversed(rows)), sample_per_region=8, seed=123
        )
        other_seed = stratified_sample_rows(rows, sample_per_region=8, seed=456)

        first_ids = {row["patch_id"] for row in first}
        self.assertEqual(first_ids, {row["patch_id"] for row in repeated})
        self.assertEqual(first_ids, {row["patch_id"] for row in reversed_input})
        self.assertNotEqual(first_ids, {row["patch_id"] for row in other_seed})

        per_region = Counter(row["region_id"] for row in first)
        self.assertEqual(per_region, Counter({"R1": 8, "R2": 8}))
        for region in per_region:
            per_category = Counter(
                row["category"] for row in first if row["region_id"] == region
            )
            self.assertLessEqual(max(per_category.values()) - min(per_category.values()), 1)

    def test_sampling_disabled_preserves_all_rows_in_manifest_order(self) -> None:
        rows = self.fixture_rows()
        sampled = stratified_sample_rows(rows, sample_per_region=0, seed=123)
        self.assertEqual(sampled, rows)


if __name__ == "__main__":
    unittest.main()
