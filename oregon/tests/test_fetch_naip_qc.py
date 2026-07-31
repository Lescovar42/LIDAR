from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from fetch_naip_qc import (
    NAIP_MANIFEST_FIELDS,
    acquisition_year_for_row,
    acquisition_year_for_tile,
    choose_year,
    compute_year_gap,
    manifest_provenance,
    refresh_manifest_from_cache,
    stratified_sample_rows,
)
from lidar_vintage import REGISTRY_ORIGIN, AcquisitionMetadataError


def patch_row(**overrides: str) -> dict[str, str]:
    """A post-fix patches.csv row: acquisition 2020, LAS file created 2024."""
    row = {
        "patch_id": "R1_tile_r000000_c000000",
        "split": "train",
        "tile_name": "USGS_LPC_OR_WesternWildfires_A22_w2051n2776.laz",
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


def write_cached_npz(
    outdir: Path, row: dict[str, str], *, selected_year: int, stale_extras: dict | None = None
) -> Path:
    """Write a cached NAIP NPZ, optionally carrying stale LiDAR metadata."""
    path = outdir / "patches" / row["split"] / f"{row['patch_id']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": 1,
        "selected_year": selected_year,
        "has_nir": True,
        "actual_pixel_size_x": 0.6,
    }
    metadata.update(stale_extras or {})
    np.savez_compressed(
        path,
        bands=np.zeros((4, 4, 4), dtype=np.uint8),
        valid_mask=np.ones((4, 4), dtype=np.uint8),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return path


class YearGapTests(unittest.TestCase):
    def test_computes_signed_gap_and_flags_only_over_two_years(self) -> None:
        self.assertEqual(compute_year_gap(2019, 2022), (3, True))
        self.assertEqual(compute_year_gap(2022, 2019), (-3, True))
        self.assertEqual(compute_year_gap(2019, 2021), (2, False))
        self.assertEqual(compute_year_gap(2019, 2017), (-2, False))

    def test_missing_or_invalid_year_has_no_gap_or_flag(self) -> None:
        self.assertEqual(compute_year_gap("", 2022), (None, None))
        self.assertEqual(compute_year_gap(2019, "unknown"), (None, None))

    def test_gap_uses_acquisition_not_file_creation_year(self) -> None:
        row = patch_row()
        acquisition = acquisition_year_for_row(row)

        self.assertEqual(2020, acquisition)
        self.assertEqual((2, False), compute_year_gap(acquisition, 2022))
        # Had the LAS file-creation year been used, the gap would be -2 from 2024.
        self.assertEqual((-2, False), compute_year_gap(2024, 2022))


class AcquisitionYearReadTests(unittest.TestCase):
    def test_reads_authoritative_acquisition_year(self) -> None:
        self.assertEqual(2020, acquisition_year_for_row(patch_row()))
        self.assertEqual(2020, acquisition_year_for_tile([patch_row(), patch_row()]))

    def test_legacy_manifest_year_is_not_authoritative_by_default(self) -> None:
        legacy = {
            "patch_id": "p1",
            "tile_name": "USGS_LPC_OR_WesternWildfires_A22_w2051n2776.laz",
            "lidar_year": "2024",
        }
        self.assertIsNone(acquisition_year_for_row(legacy))
        self.assertIsNone(acquisition_year_for_tile([legacy]))
        self.assertEqual(
            2024, acquisition_year_for_row(legacy, trust_legacy_lidar_year=True)
        )

    def test_filename_year_is_never_used_as_the_target(self) -> None:
        legacy = {"patch_id": "p1", "tile_name": "USGS_LPC_OR_OLCMetro_2019_x.laz"}
        self.assertIsNone(acquisition_year_for_row(legacy))
        self.assertIsNone(
            acquisition_year_for_row(legacy, trust_legacy_lidar_year=True)
        )

    def test_invalid_acquisition_year_raises(self) -> None:
        with self.assertRaises(ValueError):
            acquisition_year_for_row(patch_row(lidar_acquisition_year="20x0"))

    def test_conflicting_tile_years_raise_naming_tile_and_years(self) -> None:
        rows = [patch_row(), patch_row(lidar_acquisition_year="2021", patch_id="p2")]
        with self.assertRaises(ValueError) as caught:
            acquisition_year_for_tile(rows)
        message = str(caught.exception)
        self.assertIn("USGS_LPC_OR_WesternWildfires_A22_w2051n2776.laz", message)
        self.assertIn("2020", message)
        self.assertIn("2021", message)


class NearestYearSelectionTests(unittest.TestCase):
    @staticmethod
    def records(*years: int) -> list[dict[str, object]]:
        return [{"Year": year} for year in years]

    def test_selects_the_matching_year_for_acquisition_2020(self) -> None:
        selected = choose_year(
            self.records(2018, 2020, 2022),
            target_year=acquisition_year_for_row(patch_row()),
            requested_year=None,
        )
        self.assertEqual(2020, selected)

    def test_prefers_imagery_after_acquisition_when_equally_close(self) -> None:
        selected = choose_year(
            self.records(2018, 2022), target_year=2020, requested_year=None
        )
        self.assertEqual(2022, selected)

    def test_file_creation_year_would_have_selected_a_different_image(self) -> None:
        available = self.records(2018, 2020, 2022)
        self.assertEqual(
            2020, choose_year(available, target_year=2020, requested_year=None)
        )
        self.assertEqual(
            2022, choose_year(available, target_year=2024, requested_year=None)
        )

    def test_unknown_acquisition_falls_back_to_latest_available_year(self) -> None:
        selected = choose_year(
            self.records(2018, 2020, 2022), target_year=None, requested_year=None
        )
        self.assertEqual(2022, selected)


class ManifestProvenanceTests(unittest.TestCase):
    def test_manifest_row_carries_acquisition_and_file_creation_metadata(self) -> None:
        provenance = manifest_provenance(patch_row())

        self.assertEqual(2020, provenance["lidar_acquisition_year"])
        self.assertEqual(2020, provenance["lidar_acquisition_start_year"])
        self.assertEqual(2020, provenance["lidar_acquisition_end_year"])
        self.assertEqual(
            "USGS project/acquisition metadata", provenance["lidar_acquisition_source"]
        )
        self.assertEqual(
            "regions/tillamook_lidar_acquisition.md",
            provenance["lidar_acquisition_evidence"],
        )
        self.assertEqual("true", provenance["lidar_acquisition_verified"])
        self.assertEqual(2024, provenance["lidar_file_creation_year"])
        self.assertEqual(2022, provenance["lidar_inferred_year_hint"])
        self.assertEqual(2020, provenance["lidar_year"])

    def test_every_provenance_key_is_a_declared_manifest_column(self) -> None:
        self.assertTrue(set(manifest_provenance(patch_row())) <= set(NAIP_MANIFEST_FIELDS))


class CachedManifestRefreshTests(unittest.TestCase):
    """Acceptance test 12: refresh provenance without redownloading imagery."""

    def test_stale_cached_npz_cannot_force_the_old_year_into_a_new_manifest(self) -> None:
        row = patch_row()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "naip"
            cached = write_cached_npz(
                outdir,
                row,
                selected_year=2022,
                stale_extras={
                    "lidar_year": 2024,
                    "lidar_year_source": "las_header.creation_date",
                    "year_gap": -2,
                },
            )
            before = cached.read_bytes()

            rows, status_counts = refresh_manifest_from_cache(
                [row], outdir=outdir, default_resolution_m=0.6
            )

            self.assertEqual(Counter({"cached": 1}), status_counts)
            self.assertEqual(1, len(rows))
            manifest = rows[0]
            self.assertEqual(2020, manifest["lidar_acquisition_year"])
            self.assertEqual(2020, manifest["lidar_year"])
            self.assertEqual(2024, manifest["lidar_file_creation_year"])
            self.assertEqual(2022, manifest["naip_year"])
            self.assertEqual(2, manifest["year_gap"])
            self.assertIs(False, manifest["gap_flag"])
            self.assertEqual("cached", manifest["status"])
            # Imagery is reused byte-for-byte; nothing was redownloaded.
            self.assertEqual(before, cached.read_bytes())

    def test_unknown_acquisition_leaves_gap_blank(self) -> None:
        row = patch_row(
            lidar_acquisition_year="",
            lidar_acquisition_start_year="",
            lidar_acquisition_end_year="",
            lidar_acquisition_source="",
            lidar_acquisition_evidence="",
            lidar_acquisition_verified="",
            lidar_year="",
            lidar_year_source="unknown",
        )
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "naip"
            write_cached_npz(outdir, row, selected_year=2022)

            rows, _ = refresh_manifest_from_cache(
                [row], outdir=outdir, default_resolution_m=0.6
            )

            manifest = rows[0]
            self.assertEqual("", manifest["lidar_acquisition_year"])
            self.assertEqual("", manifest["year_gap"])
            self.assertEqual("", manifest["gap_flag"])
            self.assertEqual(2022, manifest["naip_year"])
            self.assertEqual(2024, manifest["lidar_file_creation_year"])

    def test_missing_cache_is_reported_without_claiming_a_gap(self) -> None:
        row = patch_row()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "naip"
            outdir.mkdir(parents=True)

            rows, status_counts = refresh_manifest_from_cache(
                [row], outdir=outdir, default_resolution_m=0.6
            )

            self.assertEqual(Counter({"missing_cache": 1}), status_counts)
            self.assertEqual("missing_cache", rows[0]["status"])
            self.assertEqual("", rows[0]["year_gap"])
            self.assertEqual("", rows[0]["gap_flag"])
            self.assertEqual(2020, rows[0]["lidar_acquisition_year"])

    def test_legacy_patch_row_refresh_stays_unknown_unless_trusted(self) -> None:
        legacy = {
            "patch_id": "legacy_patch",
            "split": "train",
            "tile_name": "USGS_LPC_OR_WesternWildfires_A22_w2051n2776.laz",
            "lidar_year": "2024",
            "lidar_year_source": "las_header.creation_date",
        }
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "naip"
            write_cached_npz(outdir, legacy, selected_year=2022)

            untrusted, _ = refresh_manifest_from_cache(
                [legacy], outdir=outdir, default_resolution_m=0.6
            )
            trusted, _ = refresh_manifest_from_cache(
                [legacy],
                outdir=outdir,
                default_resolution_m=0.6,
                trust_legacy_lidar_year=True,
            )

            self.assertEqual("", untrusted[0]["lidar_acquisition_year"])
            self.assertEqual("", untrusted[0]["year_gap"])
            self.assertEqual(2024, trusted[0]["lidar_acquisition_year"])
            self.assertEqual(-2, trusted[0]["year_gap"])


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
