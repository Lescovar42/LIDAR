from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from report_regions import (
    build_region_report,
    group_tnm_projects,
    naip_years,
    project_key,
    refresh_region_metadata,
)


REAL_TNM_RECORD = {
    "title": "USGS Lidar Point Cloud OR_OLCMetro_2019_A19 w2051n2776",
    "sourceId": "639d2f03d34e0de3a1f25974",
    "vendorMetaUrl": (
        "https://prd-tnm.s3.amazonaws.com/index.html?prefix="
        "StagedProducts/Elevation/metadata/OR_OLCMetro_2019_A19/OR_OLCMetro_2019"
    ),
    "downloadURL": (
        "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
        "OR_OLCMetro_2019_A19/OR_OLCMetro_2019/LAZ/"
        "USGS_LPC_OR_OLCMetro_2019_A19_w2051n2776.laz"
    ),
    "sizeInBytes": 167_885_409,
    "boundingBox": {
        "minX": -122.54091628731729,
        "maxX": -122.52511901558759,
        "minY": 45.37796153850478,
        "maxY": 45.38908672861234,
    },
}


def region(**overrides):
    value = {
        "id": "R3",
        "slug": "oregon_city",
        "name": "Oregon City",
        "status": "active",
        "role": "test_urban_ood",
        "bbox": [-122.90, 45.35, -122.55, 45.65],
        "tile_budget": 10,
        "storage_budget_gb": 0,
        "candidate_projects": ["USGS_LPC_OR_OLCMetro_2019"],
        "slido_output": "slido.geojson",
        "tnm_records": "tnm.json",
        "naip_records": "naip.json",
    }
    value.update(overrides)
    return value


class RegionReportTests(unittest.TestCase):
    def test_project_grouping_counts_bytes_budget_and_union_coverage(self) -> None:
        items = [
            {
                "project": "project-a",
                "sourceId": "per-tile-a",
                "title": "Project A 001",
                "sizeInBytes": 100_000_000,
                "bbox": [0, 0, 1.25, 2],
            },
            {
                "project": "project-a",
                "sourceId": "per-tile-b",
                "title": "Project A 002",
                "sizeInBytes": 300_000_000,
                "boundingBox": {"minX": 0.75, "minY": 0, "maxX": 2, "maxY": 2},
            },
            {
                "title": "Fallback Project 123",
                "sizeInBytes": 50_000_000,
                "bbox": [0, 0, 1, 1],
            },
        ]
        projects = group_tnm_projects(items, [0, 0, 2, 2], tile_budget=10)
        self.assertEqual(projects["project-a"]["tile_count"], 2)
        self.assertEqual(projects["project-a"]["summed_bytes"], 400_000_000)
        self.assertEqual(projects["project-a"]["aoi_coverage_share"], 1.0)
        self.assertEqual(projects["project-a"]["projected_gb_at_tile_budget"], 2.0)
        self.assertEqual(projects["Fallback Project"]["aoi_coverage_share"], 0.25)

    def test_project_key_ignores_per_tile_source_id_and_uses_real_tnm_paths(self) -> None:
        second = dict(REAL_TNM_RECORD, sourceId="different-per-tile-id")
        self.assertEqual(project_key(REAL_TNM_RECORD), "USGS_LPC_OR_OLCMetro_2019_A19")
        self.assertEqual(project_key(second), project_key(REAL_TNM_RECORD))
        self.assertEqual(project_key({"title": "Project_Name-0042"}), "Project_Name")

    def test_registry_candidate_matches_real_tnm_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slido.geojson").write_text('{"features": []}', encoding="utf-8")
            (root / "tnm.json").write_text(json.dumps({"items": [REAL_TNM_RECORD]}), encoding="utf-8")
            (root / "naip.json").write_text('{"features": []}', encoding="utf-8")
            report = build_region_report(region(), registry_path=root / "regions.json")
        project = report["tnm_lpc"]["projects"]["USGS_LPC_OR_OLCMetro_2019_A19"]
        self.assertEqual(project["registry_candidates"], ["USGS_LPC_OR_OLCMetro_2019"])

    def test_naip_year_collection_including_arcgis_attributes(self) -> None:
        records = [
            {"attributes": {"Year": 2020}},
            {"properties": {"Year": 2022}},
            {"year": "2018 imagery"},
            {"properties": {"Year": None}},
            {"naip_year": "2022"},
        ]
        self.assertEqual(naip_years(records), [2018, 2020, 2022])

    def test_active_report_fails_on_missing_metadata_unless_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "regions.json"
            with self.assertRaisesRegex(FileNotFoundError, "--allow-missing"):
                build_region_report(region(), registry_path=registry_path)
            skeleton = build_region_report(region(), registry_path=registry_path, allow_missing=True)
        self.assertFalse(skeleton["tnm_lpc"]["available"])

    def test_refresh_persists_mocked_tnm_and_arcgis_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_session = Mock()
            tnm_fetcher = Mock(return_value=[REAL_TNM_RECORD])
            naip_fetcher = Mock(return_value=[{"OBJECTID": 7, "Year": 2022}])
            with patch("report_regions.requests.Session", return_value=fake_session):
                paths = refresh_region_metadata(
                    region(),
                    registry_path=root / "regions.json",
                    tnm_fetcher=tnm_fetcher,
                    naip_fetcher=naip_fetcher,
                )
            saved_tnm = json.loads(paths["tnm"].read_text(encoding="utf-8"))
            saved_naip = json.loads(paths["naip"].read_text(encoding="utf-8"))
        tnm_fetcher.assert_called_once_with(tuple(region()["bbox"]))
        naip_fetcher.assert_called_once()
        fake_session.close.assert_called_once()
        self.assertEqual(saved_tnm["items"][0]["sourceId"], REAL_TNM_RECORD["sourceId"])
        self.assertEqual(naip_years(saved_naip["features"]), [2022])


if __name__ == "__main__":
    unittest.main()
