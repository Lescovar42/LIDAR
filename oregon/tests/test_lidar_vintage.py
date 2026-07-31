"""Core LiDAR acquisition-provenance tests.

No network access, no LAZ downloads, no generated dataset files are required.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date, datetime
from pathlib import Path

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from lidar_vintage import (
    CLI_ORIGIN,
    MIN_ACQUISITION_YEAR,
    REGISTRY_ORIGIN,
    UNKNOWN_YEAR_SOURCE,
    AcquisitionConflictError,
    AcquisitionMetadataError,
    LidarAcquisition,
    LidarVintage,
    acquisition_from_cli,
    acquisition_provenance_from_row,
    acquisition_year_from_row,
    file_metadata_from_header,
    infer_year_hint,
    max_acquisition_year,
    parse_acquisition,
    parse_year_hint,
    resolve_acquisition,
    row_has_acquisition_columns,
    summarize_vintages,
)

TILLAMOOK_PROJECT = "OR_WesternWildfires_A22"
A22_TILE = "USGS_LPC_OR_WesternWildfires_A22_w2051n2776.laz"


def tillamook_acquisition(**overrides: object) -> LidarAcquisition:
    payload = {
        "start_year": 2020,
        "end_year": 2020,
        "nominal_year": 2020,
        "source": "USGS project/acquisition metadata",
        "evidence": "regions/tillamook_lidar_acquisition.md",
        "verified": True,
        "lidar_project": TILLAMOOK_PROJECT,
    }
    payload.update(overrides)
    acquisition = parse_acquisition(payload, origin=REGISTRY_ORIGIN)
    assert acquisition is not None
    return acquisition


class HeaderCannotOverrideAcquisitionTests(unittest.TestCase):
    """Acceptance test 1: header 2024 + A22 must not beat acquisition 2020."""

    def build_vintage(self) -> LidarVintage:
        return LidarVintage(
            acquisition=tillamook_acquisition(),
            file_metadata=file_metadata_from_header(date(2024, 3, 14)),
            hint=infer_year_hint(TILLAMOOK_PROJECT, A22_TILE),
        )

    def test_acquisition_year_wins_and_other_years_stay_separate(self) -> None:
        vintage = self.build_vintage()

        self.assertEqual(2020, vintage.acquisition_nominal_year)
        self.assertEqual(2024, vintage.file_metadata.creation_year)
        self.assertEqual("2024-03-14", vintage.file_metadata.creation_date)
        self.assertEqual(2022, vintage.hint.year)
        self.assertEqual("project_name", vintage.hint.source)

    def test_temporal_filtering_and_naip_selection_receive_the_acquisition_year(self) -> None:
        vintage = self.build_vintage()

        self.assertEqual(2020, vintage.temporal_filter_year())
        self.assertNotEqual(vintage.file_metadata.creation_year, vintage.temporal_filter_year())
        self.assertNotEqual(vintage.hint.year, vintage.temporal_filter_year())

    def test_row_fields_carry_all_three_concepts_separately(self) -> None:
        fields = self.build_vintage().as_row_fields()

        self.assertEqual(2020, fields["lidar_acquisition_year"])
        self.assertEqual(2020, fields["lidar_acquisition_start_year"])
        self.assertEqual(2020, fields["lidar_acquisition_end_year"])
        self.assertEqual("USGS project/acquisition metadata", fields["lidar_acquisition_source"])
        self.assertEqual(
            "regions/tillamook_lidar_acquisition.md", fields["lidar_acquisition_evidence"]
        )
        self.assertEqual("true", fields["lidar_acquisition_verified"])
        self.assertEqual(2024, fields["lidar_file_creation_year"])
        self.assertEqual(2022, fields["lidar_inferred_year_hint"])
        self.assertEqual("project_name", fields["lidar_inferred_year_hint_source"])
        # Legacy alias mirrors acquisition, never the header year.
        self.assertEqual(2020, fields["lidar_year"])
        self.assertEqual(REGISTRY_ORIGIN, fields["lidar_year_source"])


class UnknownAcquisitionTests(unittest.TestCase):
    """Acceptance test 2: no explicit metadata means unknown, not 2024 or 2022."""

    def test_header_and_project_token_alone_leave_acquisition_unknown(self) -> None:
        vintage = LidarVintage(
            file_metadata=file_metadata_from_header(date(2024, 3, 14)),
            hint=infer_year_hint(TILLAMOOK_PROJECT, A22_TILE),
        )

        self.assertFalse(vintage.acquisition_is_known)
        self.assertIsNone(vintage.acquisition_nominal_year)
        self.assertIsNone(vintage.temporal_filter_year())
        self.assertEqual(2024, vintage.file_metadata.creation_year)
        self.assertEqual(2022, vintage.hint.year)

        fields = vintage.as_row_fields()
        self.assertEqual("", fields["lidar_acquisition_year"])
        self.assertEqual("", fields["lidar_year"])
        self.assertEqual(UNKNOWN_YEAR_SOURCE, fields["lidar_year_source"])
        self.assertEqual("", fields["lidar_acquisition_verified"])
        self.assertEqual(2024, fields["lidar_file_creation_year"])

    def test_filename_year_is_a_hint_not_acquisition(self) -> None:
        vintage = LidarVintage(hint=infer_year_hint("", "USGS_LPC_OR_OLCMetro_2019_x.laz"))

        self.assertIsNone(vintage.acquisition_nominal_year)
        self.assertEqual(2019, vintage.hint.year)
        self.assertEqual("tile_name", vintage.hint.source)

    def test_year_hint_parser_matches_historical_token_behavior(self) -> None:
        self.assertEqual(2019, parse_year_hint("USGS_LPC_OR_OLCMetro_2019"))
        self.assertEqual(2022, parse_year_hint("OR_WesternWildfires_A22_w123n456"))
        self.assertIsNone(parse_year_hint("project_without_year"))


class CliOverrideTests(unittest.TestCase):
    """Acceptance test 3: explicit CLI override."""

    def test_scalar_override_sets_range_and_preserves_source(self) -> None:
        acquisition = acquisition_from_cli(
            year=2020, source="verified project metadata"
        )

        assert acquisition is not None
        self.assertEqual((2020, 2020, 2020), (
            acquisition.start_year, acquisition.end_year, acquisition.nominal_year
        ))
        self.assertEqual("verified project metadata", acquisition.source)
        self.assertEqual(CLI_ORIGIN, acquisition.origin)
        self.assertFalse(acquisition.verified)

    def test_override_keeps_file_creation_separate_in_outputs(self) -> None:
        vintage = LidarVintage(
            acquisition=acquisition_from_cli(
                year=2020,
                source="verified project metadata",
                evidence="regions/tillamook_lidar_acquisition.md",
                verified=True,
            ),
            file_metadata=file_metadata_from_header(date(2024, 1, 2)),
            hint=infer_year_hint(TILLAMOOK_PROJECT, A22_TILE),
        )
        fields = vintage.as_row_fields()

        self.assertEqual(2020, fields["lidar_acquisition_year"])
        self.assertEqual(2024, fields["lidar_file_creation_year"])
        self.assertEqual("true", fields["lidar_acquisition_verified"])
        self.assertEqual(CLI_ORIGIN, fields["lidar_year_source"])

    def test_no_acquisition_arguments_means_no_override(self) -> None:
        self.assertIsNone(acquisition_from_cli())

    def test_year_without_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "source is required"):
            acquisition_from_cli(year=2020)

    def test_partial_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "supplied together"):
            acquisition_from_cli(start_year=2008, source="project report")

    def test_source_without_any_year_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "are required"):
            acquisition_from_cli(source="project report")

    def test_out_of_range_year_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "outside the allowed range"):
            acquisition_from_cli(year=1971, source="project report")
        with self.assertRaisesRegex(AcquisitionMetadataError, "outside the allowed range"):
            acquisition_from_cli(year=max_acquisition_year() + 1, source="project report")

    def test_minimum_year_boundary_is_accepted(self) -> None:
        acquisition = acquisition_from_cli(
            year=MIN_ACQUISITION_YEAR, source="project report"
        )
        assert acquisition is not None
        self.assertEqual(MIN_ACQUISITION_YEAR, acquisition.nominal_year)


class RegistryMetadataTests(unittest.TestCase):
    """Acceptance test 4: registry acquisition metadata and its invalid cases."""

    def test_valid_registry_metadata_resolves(self) -> None:
        acquisition = tillamook_acquisition()

        self.assertEqual(2020, acquisition.nominal_year)
        self.assertEqual(REGISTRY_ORIGIN, acquisition.origin)
        self.assertTrue(acquisition.verified)
        self.assertEqual(TILLAMOOK_PROJECT, acquisition.lidar_project)

    def test_absent_block_is_not_an_error(self) -> None:
        self.assertIsNone(parse_acquisition(None, origin=REGISTRY_ORIGIN))

    def test_empty_block_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "present but empty"):
            parse_acquisition({}, origin=REGISTRY_ORIGIN)

    def test_year_out_of_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "outside the allowed range"):
            tillamook_acquisition(start_year=1899, end_year=1899, nominal_year=1899)

    def test_start_after_end_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "must not exceed"):
            tillamook_acquisition(start_year=2011, end_year=2009, nominal_year=2010)

    def test_nominal_outside_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "falls outside the acquisition range"):
            tillamook_acquisition(start_year=2008, end_year=2009, nominal_year=2012)

    def test_missing_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, r"\.source must be a non-empty"):
            parse_acquisition(
                {"nominal_year": 2020, "start_year": 2020, "end_year": 2020},
                origin=REGISTRY_ORIGIN,
            )

    def test_blank_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, r"\.source must be a non-empty"):
            tillamook_acquisition(source="   ")

    def test_missing_evidence_is_rejected_when_required(self) -> None:
        payload = {
            "nominal_year": 2020,
            "start_year": 2020,
            "end_year": 2020,
            "source": "USGS project/acquisition metadata",
        }
        with self.assertRaisesRegex(AcquisitionMetadataError, r"\.evidence must be a non-empty"):
            parse_acquisition(payload, origin=REGISTRY_ORIGIN, require_evidence=True)
        self.assertIsNotNone(
            parse_acquisition(payload, origin=CLI_ORIGIN, require_evidence=False)
        )

    def test_project_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "does not match the selected"):
            parse_acquisition(
                {
                    "nominal_year": 2020,
                    "start_year": 2020,
                    "end_year": 2020,
                    "source": "USGS project/acquisition metadata",
                    "evidence": "regions/tillamook_lidar_acquisition.md",
                    "lidar_project": TILLAMOOK_PROJECT,
                },
                origin=REGISTRY_ORIGIN,
                expected_project="OR_TILLAMOOK_ODF_2007",
            )

    def test_project_outside_candidate_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "one of candidate_projects"):
            parse_acquisition(
                {
                    "nominal_year": 2020,
                    "start_year": 2020,
                    "end_year": 2020,
                    "source": "USGS project/acquisition metadata",
                    "evidence": "regions/tillamook_lidar_acquisition.md",
                    "lidar_project": "SOME_OTHER_PROJECT",
                },
                origin=REGISTRY_ORIGIN,
                allowed_projects=[TILLAMOOK_PROJECT],
            )

    def test_partially_specified_object_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "start_year and end_year together"):
            parse_acquisition(
                {
                    "start_year": 2020,
                    "source": "USGS project/acquisition metadata",
                    "evidence": "e",
                },
                origin=REGISTRY_ORIGIN,
            )
        with self.assertRaisesRegex(AcquisitionMetadataError, "must define nominal_year"):
            parse_acquisition(
                {"source": "USGS project/acquisition metadata", "evidence": "e"},
                origin=REGISTRY_ORIGIN,
            )

    def test_unsupported_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "unsupported field"):
            tillamook_acquisition(creation_date="2024-01-01")

    def test_non_boolean_verified_is_rejected(self) -> None:
        with self.assertRaisesRegex(AcquisitionMetadataError, "must be true or false"):
            tillamook_acquisition(verified="yes")


class ConflictTests(unittest.TestCase):
    """Acceptance test 5: authoritative sources that disagree must fail."""

    def test_conflicting_sources_fail_and_name_both(self) -> None:
        registry = tillamook_acquisition()
        tile_manifest = tillamook_acquisition(
            start_year=2021, end_year=2021, nominal_year=2021
        )

        with self.assertRaises(AcquisitionConflictError) as caught:
            resolve_acquisition(
                [("registry R1", registry), ("tile manifest probe.json", tile_manifest)],
                context="region R1",
            )
        message = str(caught.exception)
        self.assertIn("registry R1", message)
        self.assertIn("tile manifest probe.json", message)
        self.assertIn("nominal=2020", message)
        self.assertIn("nominal=2021", message)
        self.assertIn("region R1", message)

    def test_agreeing_sources_collapse_to_highest_precedence(self) -> None:
        cli = acquisition_from_cli(year=2020, source="CLI restatement")
        registry = tillamook_acquisition()

        resolved = resolve_acquisition([("CLI", cli), ("registry", registry)])

        assert resolved is not None
        self.assertEqual(CLI_ORIGIN, resolved.origin)
        self.assertEqual(2020, resolved.nominal_year)

    def test_single_source_passes_through_and_no_source_is_none(self) -> None:
        registry = tillamook_acquisition()
        self.assertIs(registry, resolve_acquisition([("registry", registry)]))
        self.assertIsNone(resolve_acquisition([("registry", None), ("cli", None)]))


class MultiYearSurveyTests(unittest.TestCase):
    """Acceptance test 6: multi-year ranges need an explicit nominal year."""

    def test_range_without_nominal_fails_only_where_a_scalar_is_required(self) -> None:
        acquisition = parse_acquisition(
            {
                "start_year": 2008,
                "end_year": 2009,
                "source": "USGS project/acquisition metadata",
                "evidence": "regions/example.md",
            },
            origin=REGISTRY_ORIGIN,
        )
        assert acquisition is not None
        self.assertEqual((2008, 2009), (acquisition.start_year, acquisition.end_year))
        self.assertIsNone(acquisition.nominal_year)
        self.assertTrue(acquisition.is_multi_year)

        vintage = LidarVintage(acquisition=acquisition)
        with self.assertRaisesRegex(AcquisitionMetadataError, "requires one authoritative"):
            vintage.temporal_filter_year("SLIDO temporal filtering")

        fields = vintage.as_row_fields()
        self.assertEqual(2008, fields["lidar_acquisition_start_year"])
        self.assertEqual(2009, fields["lidar_acquisition_end_year"])
        self.assertEqual("", fields["lidar_acquisition_year"])

    def test_range_with_nominal_preserves_all_three_fields(self) -> None:
        acquisition = parse_acquisition(
            {
                "start_year": 2008,
                "end_year": 2009,
                "nominal_year": 2009,
                "source": "USGS project/acquisition metadata",
                "evidence": "regions/example.md",
            },
            origin=REGISTRY_ORIGIN,
        )
        assert acquisition is not None
        vintage = LidarVintage(acquisition=acquisition)

        self.assertEqual(2009, vintage.temporal_filter_year())
        fields = vintage.as_row_fields()
        self.assertEqual(2008, fields["lidar_acquisition_start_year"])
        self.assertEqual(2009, fields["lidar_acquisition_end_year"])
        self.assertEqual(2009, fields["lidar_acquisition_year"])
        self.assertEqual(2009, fields["lidar_year"])


class FileMetadataTests(unittest.TestCase):
    def test_date_datetime_and_text_headers_are_preserved(self) -> None:
        self.assertEqual(
            (2024, "2024-03-14"),
            (
                file_metadata_from_header(date(2024, 3, 14)).creation_year,
                file_metadata_from_header(date(2024, 3, 14)).creation_date,
            ),
        )
        self.assertEqual(
            2024, file_metadata_from_header(datetime(2024, 3, 14, 6, 30)).creation_year
        )
        self.assertEqual(2024, file_metadata_from_header("2024-03-14").creation_year)

    def test_missing_or_implausible_header_year_is_recorded_as_unknown(self) -> None:
        self.assertIsNone(file_metadata_from_header(None).creation_year)
        self.assertFalse(file_metadata_from_header(None).is_known)
        broken = file_metadata_from_header("not-a-date")
        self.assertIsNone(broken.creation_year)
        self.assertEqual("not-a-date", broken.creation_date)


class RowReaderTests(unittest.TestCase):
    def test_new_schema_row_is_read_as_authoritative(self) -> None:
        row = {
            "patch_id": "p1",
            "lidar_year": "2020",
            "lidar_acquisition_year": "2020",
            "lidar_file_creation_year": "2024",
        }
        self.assertTrue(row_has_acquisition_columns(row))
        self.assertEqual(2020, acquisition_year_from_row(row))

    def test_legacy_row_is_unknown_unless_explicitly_trusted(self) -> None:
        row = {"patch_id": "p1", "lidar_year": "2024", "tile_name": A22_TILE}

        self.assertFalse(row_has_acquisition_columns(row))
        self.assertIsNone(acquisition_year_from_row(row))
        self.assertEqual(
            2024, acquisition_year_from_row(row, trust_legacy_lidar_year=True)
        )

    def test_invalid_acquisition_year_raises(self) -> None:
        row = {"patch_id": "p1", "lidar_acquisition_year": "20x0"}
        with self.assertRaises(AcquisitionMetadataError):
            acquisition_year_from_row(row)

    def test_provenance_copy_never_promotes_file_creation_year(self) -> None:
        row = {
            "patch_id": "p1",
            "lidar_year": "2020",
            "lidar_year_source": REGISTRY_ORIGIN,
            "lidar_acquisition_year": "",
            "lidar_acquisition_start_year": "",
            "lidar_acquisition_end_year": "",
            "lidar_acquisition_source": "",
            "lidar_acquisition_evidence": "",
            "lidar_acquisition_verified": "",
            "lidar_file_creation_year": "2024",
        }
        provenance = acquisition_provenance_from_row(row)

        self.assertEqual("", provenance["lidar_acquisition_year"])
        self.assertEqual("", provenance["lidar_year"])
        self.assertEqual(2024, provenance["lidar_file_creation_year"])


class SummaryTests(unittest.TestCase):
    def test_summary_separates_acquisition_from_file_creation_and_counts_unknowns(self) -> None:
        known = LidarVintage(
            acquisition=tillamook_acquisition(),
            file_metadata=file_metadata_from_header(date(2024, 3, 14)),
            hint=infer_year_hint(TILLAMOOK_PROJECT, A22_TILE),
        )
        unknown = LidarVintage(
            file_metadata=file_metadata_from_header(date(2023, 1, 1)),
            hint=infer_year_hint("", "no_year_here.laz"),
        )

        summary = summarize_vintages([known, known, unknown])

        self.assertEqual(1, summary["distinct_acquisition_vintage_count"])
        self.assertEqual(2, summary["distinct_acquisition_vintages"][0]["tile_count"])
        self.assertEqual(2020, summary["distinct_acquisition_vintages"][0]["nominal_year"])
        self.assertEqual(1, summary["unknown_acquisition_tiles"])
        self.assertEqual({"2023": 1, "2024": 2}, summary["distinct_file_creation_years"])
        self.assertEqual({"2022": 2, "unknown": 1}, summary["inferred_year_hints"])


if __name__ == "__main__":
    unittest.main()
