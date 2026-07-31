from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from lidar_vintage import REGISTRY_ORIGIN
from region_registry import (
    SCHEMA_VERSION,
    load_registry,
    pin_region_decision,
    region_acquisition,
    resolve_path,
    resolve_region,
    set_region_acquisition,
    validate_registry,
)


class RegionRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry()

    def test_checked_in_registry_has_four_active_regions_and_budget(self) -> None:
        self.assertEqual([entry["id"] for entry in self.registry["regions"]], ["R1", "R2", "R3", "R4"])
        self.assertEqual(sum(entry["tile_budget"] for entry in self.registry["regions"]), 125)
        self.assertEqual(len([entry for entry in [*self.registry["regions"], *self.registry["comparison_candidates"]] if entry.get("include_in_candidate_report")]), 5)

    def test_oregon_city_resolves_to_existing_data(self) -> None:
        region = resolve_region("oregon_city", self.registry)
        self.assertEqual(region["id"], "R3")
        self.assertTrue(region["existing_data"])
        self.assertTrue(resolve_path(region, "slido_output").is_file())
        self.assertTrue((OREGON_DIR / region["lidar_dir"]).is_dir())

    def test_rejects_unknown_role(self) -> None:
        data = copy.deepcopy(self.registry)
        data["regions"][0]["role"] = "training"
        with self.assertRaisesRegex(ValueError, "role is unknown"):
            validate_registry(data)

    def test_rejects_overlapping_active_bboxes(self) -> None:
        data = copy.deepcopy(self.registry)
        data["regions"][1]["bbox"] = list(data["regions"][0]["bbox"])
        with self.assertRaisesRegex(ValueError, "bboxes overlap"):
            validate_registry(data)

    def test_rejects_missing_path(self) -> None:
        data = copy.deepcopy(self.registry)
        del data["regions"][0]["slido_output"]
        with self.assertRaisesRegex(ValueError, "path is required"):
            validate_registry(data)

    def test_rejects_empty_candidate_projects(self) -> None:
        data = copy.deepcopy(self.registry)
        data["regions"][0]["candidate_projects"] = []
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_registry(data)

    def test_unpinned_entry_is_valid_but_partial_or_invalid_pins_are_rejected(self) -> None:
        data = copy.deepcopy(self.registry)
        validate_registry(data)
        data["regions"][0]["lidar_project"] = "not-a-candidate"
        with self.assertRaisesRegex(ValueError, "define lidar_project, cell_size, and selection_decision together"):
            validate_registry(data)

        entry = data["regions"][0]
        entry["cell_size"] = 0
        entry["selection_decision"] = {"reason": "measured probe comparison"}
        with self.assertRaisesRegex(ValueError, "one of candidate_projects"):
            validate_registry(data)
        entry["lidar_project"] = entry["candidate_projects"][0]
        with self.assertRaisesRegex(ValueError, "cell_size must be positive"):
            validate_registry(data)
        entry["cell_size"] = 1.5
        entry["selection_decision"] = {"reason": ""}
        with self.assertRaisesRegex(ValueError, "reason must be a non-empty string"):
            validate_registry(data)

    def test_checked_in_registry_declares_schema_two_and_tillamook_acquisition(self) -> None:
        self.assertGreaterEqual(self.registry["schema_version"], SCHEMA_VERSION)
        region = resolve_region("tillamook", self.registry)
        acquisition = region_acquisition(region)

        assert acquisition is not None
        self.assertEqual(2020, acquisition.nominal_year)
        self.assertEqual(2020, acquisition.start_year)
        self.assertEqual(2020, acquisition.end_year)
        self.assertEqual("OR_WesternWildfires_A22", acquisition.lidar_project)
        self.assertEqual(REGISTRY_ORIGIN, acquisition.origin)
        self.assertTrue(acquisition.verified)
        self.assertTrue(acquisition.source.strip())
        self.assertTrue((OREGON_DIR / acquisition.evidence).is_file())

    def test_regions_without_acquisition_metadata_stay_unknown(self) -> None:
        for slug in ("buxton_vernonia", "oregon_city", "marion"):
            with self.subTest(slug=slug):
                self.assertIsNone(
                    region_acquisition(resolve_region(slug, self.registry))
                )

    def test_registry_rejects_invalid_acquisition_metadata(self) -> None:
        valid = {
            "start_year": 2020,
            "end_year": 2020,
            "nominal_year": 2020,
            "source": "USGS project/acquisition metadata",
            "evidence": "regions/tillamook_lidar_acquisition.md",
            "verified": True,
            "lidar_project": "OR_WesternWildfires_A22",
        }
        cases = {
            "outside the allowed range": {**valid, "start_year": 1899, "end_year": 1899, "nominal_year": 1899},
            "must not exceed": {**valid, "start_year": 2021, "end_year": 2019, "nominal_year": 2020},
            "falls outside the acquisition range": {**valid, "start_year": 2008, "end_year": 2009, "nominal_year": 2012},
            r"\.source must be a non-empty": {key: value for key, value in valid.items() if key != "source"},
            r"\.evidence must be a non-empty": {key: value for key, value in valid.items() if key != "evidence"},
            "one of candidate_projects": {**valid, "lidar_project": "NOT_A_CANDIDATE"},
            "start_year and end_year together": {
                "start_year": 2020,
                "source": valid["source"],
                "evidence": valid["evidence"],
                "lidar_project": valid["lidar_project"],
            },
            "present but empty": {},
        }
        for pattern, payload in cases.items():
            with self.subTest(pattern=pattern):
                data = copy.deepcopy(self.registry)
                data["regions"][0]["lidar_acquisition"] = payload
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_registry(data)

    def test_registry_rejects_acquisition_attached_to_a_different_pinned_project(self) -> None:
        data = copy.deepcopy(self.registry)
        entry = data["regions"][0]
        entry["lidar_project"] = "OR_TILLAMOOK_ODF_2007"
        entry["cell_size"] = 1.5
        entry["selection_decision"] = {"reason": "measured diagnostic result"}
        with self.assertRaisesRegex(ValueError, "does not match the selected LiDAR project"):
            validate_registry(data)

    def test_set_region_acquisition_persists_validated_metadata_atomically(self) -> None:
        data = copy.deepcopy(self.registry)
        data.pop("_registry_path", None)
        del data["regions"][0]["lidar_acquisition"]
        data["schema_version"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regions.json"
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

            updated = set_region_acquisition(
                path,
                "R1",
                lidar_project="OR_WesternWildfires_A22",
                nominal_year=2020,
                source="USGS project/acquisition metadata",
                evidence="regions/tillamook_lidar_acquisition.md",
                verified=True,
            )
            self.assertEqual(2020, updated["lidar_acquisition"]["nominal_year"])
            self.assertEqual(2020, updated["lidar_acquisition"]["start_year"])
            self.assertEqual(2020, updated["lidar_acquisition"]["end_year"])
            persisted = json.loads(path.read_text(encoding="utf-8"))
            validate_registry(persisted)
            self.assertGreaterEqual(persisted["schema_version"], SCHEMA_VERSION)

            stable = path.read_text(encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one of candidate_projects"):
                set_region_acquisition(
                    path,
                    "R1",
                    lidar_project="NOT_A_CANDIDATE",
                    nominal_year=2020,
                    source="s",
                    evidence="e",
                )
            with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
                set_region_acquisition(
                    path,
                    "R1",
                    lidar_project="OR_WesternWildfires_A22",
                    nominal_year=2020,
                    source="   ",
                    evidence="e",
                )
            self.assertEqual(stable, path.read_text(encoding="utf-8"))
            self.assertEqual([], list(path.parent.glob(".regions.json.*.tmp")))

    def test_pin_region_decision_validates_and_replaces_registry_atomically(self) -> None:
        data = copy.deepcopy(self.registry)
        data.pop("_registry_path", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regions.json"
            original = json.dumps(data, indent=2) + "\n"
            path.write_text(original, encoding="utf-8")
            # R1 stores acquisition metadata for OR_WesternWildfires_A22, so the
            # pinned project must be that same project.
            project = data["regions"][0]["lidar_acquisition"]["lidar_project"]
            pinned = pin_region_decision(
                path,
                "R1",
                lidar_project=project,
                cell_size=1.5,
                reason="best measured coverage and fragmentation tradeoff",
                decision_metadata={"probe_output": "probe.json"},
            )
            self.assertEqual(project, pinned["lidar_project"])
            self.assertEqual(1.5, pinned["cell_size"])
            self.assertEqual("probe.json", pinned["selection_decision"]["probe_output"])
            validate_registry(json.loads(path.read_text(encoding="utf-8")))

            stable = path.read_text(encoding="utf-8")
            with mock.patch("region_registry.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    pin_region_decision(
                        path,
                        "R1",
                        lidar_project=project,
                        cell_size=2.0,
                        reason="another measured decision",
                    )
            self.assertEqual(stable, path.read_text(encoding="utf-8"))
            self.assertEqual([], list(path.parent.glob(".regions.json.*.tmp")))

            with self.assertRaisesRegex(ValueError, "one of candidate_projects"):
                pin_region_decision(
                    path,
                    "R1",
                    lidar_project="not-a-candidate",
                    cell_size=1.0,
                    reason="invalid",
                )

            with self.assertRaisesRegex(
                ValueError, "attached to the wrong project"
            ):
                pin_region_decision(
                    path,
                    "R1",
                    lidar_project="OR_TILLAMOOK_ODF_2007",
                    cell_size=1.5,
                    reason="switching projects without updating acquisition metadata",
                )


if __name__ == "__main__":
    unittest.main()
