from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from diagnostics.qc_patch_viewer import (
    LEGACY_ACQUISITION_SOURCE,
    slido_mask_visuals,
    vintage_context,
)
from lidar_vintage import REGISTRY_ORIGIN


def patch_row(**overrides: str) -> dict[str, str]:
    row = {
        "patch_id": "R1_tile_r000000_c000000",
        "lidar_year": "2020",
        "lidar_year_source": REGISTRY_ORIGIN,
        "lidar_acquisition_year": "2020",
        "lidar_acquisition_start_year": "2020",
        "lidar_acquisition_end_year": "2020",
        "lidar_acquisition_source": "USGS project/acquisition metadata",
        "lidar_acquisition_evidence": "regions/tillamook_lidar_acquisition.md",
        "lidar_acquisition_verified": "true",
        "lidar_file_creation_year": "2024",
        "lidar_file_creation_date": "2024-03-14",
        "lidar_inferred_year_hint": "2022",
        "lidar_inferred_year_hint_source": "project_name",
    }
    row.update(overrides)
    return row


class VintageDisplayTests(unittest.TestCase):
    """Acceptance tests 15 and 16: correct and unknown vintage display."""

    def test_acquisition_2020_file_2024_naip_2022_shows_gap_plus_two(self) -> None:
        vintage = vintage_context(patch_row(), {"naip_year": "2022"})

        self.assertEqual(2020, vintage.acquisition_year)
        self.assertEqual(2024, vintage.file_creation_year)
        self.assertEqual(2022, vintage.naip_year)
        self.assertEqual(2, vintage.year_gap)
        self.assertFalse(vintage.gap_flag)
        self.assertTrue(vintage.gap_available)
        self.assertEqual(
            "LiDAR acquisition 2020 | LAS file created 2024 | NAIP 2022 | gap +2 years",
            vintage.title_text,
        )
        self.assertIn("within 2 years", vintage.banner_text)
        self.assertIn("LiDAR acquisition", vintage.banner_text)

    def test_file_creation_year_never_enters_the_gap(self) -> None:
        vintage = vintage_context(patch_row(), {"naip_year": "2022"})

        # A file-creation-driven gap would have been -2 (2022 - 2024).
        self.assertNotEqual(-2, vintage.year_gap)
        self.assertEqual(2, vintage.year_gap)

    def test_unknown_acquisition_reports_gap_unavailable(self) -> None:
        row = patch_row(
            lidar_year="",
            lidar_year_source="unknown",
            lidar_acquisition_year="",
            lidar_acquisition_start_year="",
            lidar_acquisition_end_year="",
            lidar_acquisition_source="",
            lidar_acquisition_evidence="",
            lidar_acquisition_verified="",
        )
        vintage = vintage_context(row, {"naip_year": "2022"})

        self.assertIsNone(vintage.acquisition_year)
        self.assertEqual("unknown", vintage.acquisition_text)
        self.assertEqual(2024, vintage.file_creation_year)
        self.assertIsNone(vintage.year_gap)
        self.assertIsNone(vintage.gap_flag)
        self.assertFalse(vintage.gap_available)
        self.assertEqual(
            "LiDAR acquisition unknown | LAS file created 2024 | NAIP 2022 | "
            "gap unavailable",
            vintage.title_text,
        )
        self.assertIn("Vintage gap unavailable", vintage.banner_text)
        self.assertNotIn("within 2 years", vintage.banner_text)

    def test_missing_naip_year_also_reports_gap_unavailable(self) -> None:
        vintage = vintage_context(patch_row(), {})

        self.assertEqual(2020, vintage.acquisition_year)
        self.assertIsNone(vintage.naip_year)
        self.assertFalse(vintage.gap_available)
        self.assertIn("Vintage gap unavailable", vintage.banner_text)
        self.assertNotIn("within 2 years", vintage.banner_text)

    def test_wide_gap_produces_an_explicit_warning(self) -> None:
        vintage = vintage_context(patch_row(), {"naip_year": "2016"})

        self.assertEqual(-4, vintage.year_gap)
        self.assertTrue(vintage.gap_flag)
        self.assertIn("exceeds 2 years", vintage.banner_text)
        self.assertIn("VINTAGE WARNING", vintage.title_text)

    def test_stale_naip_manifest_gap_columns_are_ignored(self) -> None:
        stale = {
            "naip_year": "2022",
            "lidar_acquisition_year": "2020",
            "lidar_acquisition_source": "USGS project/acquisition metadata",
            "year_gap": "-2",
            "gap_flag": "True",
        }
        vintage = vintage_context(patch_row(), stale)

        self.assertEqual(2, vintage.year_gap)
        self.assertFalse(vintage.gap_flag)

    def test_legacy_manifest_year_is_not_displayed_as_acquisition_by_default(self) -> None:
        legacy_row = {
            "patch_id": "legacy",
            "lidar_year": "2024",
            "lidar_year_source": "las_header.creation_date",
        }
        untrusted = vintage_context(legacy_row, {"naip_year": "2022"})
        trusted = vintage_context(
            legacy_row, {"naip_year": "2022"}, trust_legacy_lidar_year=True
        )

        self.assertIsNone(untrusted.acquisition_year)
        self.assertFalse(untrusted.gap_available)
        self.assertIn("Vintage gap unavailable", untrusted.banner_text)
        self.assertEqual(2024, trusted.acquisition_year)
        self.assertEqual(LEGACY_ACQUISITION_SOURCE, trusted.acquisition_source)
        self.assertEqual(-2, trusted.year_gap)

    def test_naip_manifest_provenance_takes_precedence_over_patch_row(self) -> None:
        vintage = vintage_context(
            patch_row(),
            {
                "naip_year": "2022",
                "lidar_acquisition_year": "2019",
                "lidar_acquisition_source": "refreshed manifest",
                "lidar_acquisition_verified": "false",
            },
        )

        self.assertEqual(2019, vintage.acquisition_year)
        self.assertEqual("refreshed manifest", vintage.acquisition_source)
        self.assertIs(False, vintage.acquisition_verified)
        self.assertEqual(3, vintage.year_gap)
        self.assertTrue(vintage.gap_flag)

    def test_acquisition_source_and_hint_are_available_for_display(self) -> None:
        vintage = vintage_context(patch_row(), {"naip_year": "2022"})

        self.assertEqual(
            "USGS project/acquisition metadata", vintage.acquisition_source
        )
        self.assertIs(True, vintage.acquisition_verified)
        self.assertEqual(2022, vintage.inferred_hint_year)
        self.assertEqual("project_name", vintage.inferred_hint_source)


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
